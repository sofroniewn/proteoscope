import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm
import esm
from peft import LoraConfig, get_peft_model
import gc
from contextlib import nullcontext

from .esm_bottleneck import ESMBottleneck
from .scheduler import get_cosine_schedule_with_warmup
from .cytoself import CytoselfLM


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)

        
class ProteocytoclassLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteocytoclassLM, self).__init__()

        if module_config.model.esm is None:
            self.esm = None
            self.esm_converter = None
            self.esm_embedding_layer = None
            self.esm_trainable = False
        else:
            self.esm, alphabet = esm.pretrained.load_model_and_alphabet_hub(module_config.model.esm.model_name)
            self.esm = self.esm.half()
            if module_config.model.esm.lora is not None:
                peft_config = LoraConfig(**module_config.model.esm.lora)
                self.esm = get_peft_model(self.esm, peft_config)
                self.esm_trainable = True
            else:
                self.esm.eval()
                self.esm_trainable = False
                for param in self.esm.parameters():
                    param.requires_grad = False
            self.esm_converter = alphabet.get_batch_converter(module_config.model.esm.truncation_seq_length)
            self.esm_embedding_layer = module_config.model.esm.embedding_layer

        self.esm_bottleneck = ESMBottleneck(module_config.model.esm_bottleneck)

        self.optim_config = module_config.optimizer

        if module_config.model.cytoself is not None:
            cytoself_checkpoint = module_config.model.cytoself.checkpoint
            cytoself_config = module_config.copy()
            cytoself_config.model = module_config.model.cytoself.model
            clm = CytoselfLM.load_from_checkpoint(
                cytoself_checkpoint,
                module_config=cytoself_config,
                num_class=module_config.model.cytoself.num_class,
            )
            self.cytoself = clm.model
            self.cytoself.eval()
        else:
            self.cytoself = None

        if module_config.model.projection_dims > 0:
            if module_config.model.esm_bottleneck.d_model > 0:
                emb = module_config.model.esm_bottleneck.d_model
            else:
                emb = module_config.model.protein_embedding_dims

            self.protein_projection = ProjectionHead(
                embedding_dim=emb,
                projection_dim=module_config.model.projection_dims,
                dropout=module_config.model.dropout,
            )
        else:
            self.protein_projection = None

        self.criteion_latents = nn.MSELoss()
        self.criteion_classifier = nn.CrossEntropyLoss()
        self.classifier_weight =  module_config.model.classifier_weight

    def embed_sequence(self, batch, sequence_condition_probability=1.0):
        with torch.no_grad() if not self.esm_trainable else nullcontext():
            if self.esm is None:
                seq_embeds = batch["sequence_embed"]
                seq_mask = batch["sequence_mask"]
            else:
                labels = batch['index']
                sequence = batch['peptide']
                result = list(zip(labels, sequence))
                labels, strs, toks = self.esm_converter(result)
                toks = toks.to(self.device)
                embedding_layer = self.esm_embedding_layer
                out = self.esm(toks, repr_layers=[embedding_layer], return_contacts=False)
                seq_embeds = out["representations"][embedding_layer]
                seq_embeds = seq_embeds[:, 1:-1]
                seq_mask = torch.zeros(seq_embeds.shape[:-1]).bool().to(self.device)
                for i, ind in enumerate(batch['truncation']):
                    seq_mask[i, :ind] = True

        seq_embeds, seq_mask = self.esm_bottleneck(
            seq_embeds, seq_mask, sequence_condition_probability
        )

        return seq_embeds, seq_mask

    def embed(self, batch, sequence_condition_probability=1.0):
        seq_embeds, seq_mask = self.embed_sequence(batch, sequence_condition_probability)
        # Mean pool
        seq_mask = seq_mask.unsqueeze(-1)
        seq_embeds = (seq_embeds * seq_mask).sum(dim=1) / seq_mask.sum(dim=1)
        if self.protein_projection is not None:
            seq_embeds = self.protein_projection(seq_embeds)
        return seq_embeds   

    def forward(self, batch, sequence_condition_probability=1.0):        
        seq_embeds = self.embed(batch, sequence_condition_probability)

        # Create latents
        with torch.no_grad():
            image_embeds = self.cytoself(batch["image"], "vqvec2")
            image_logits = self.cytoself.fc_layers[1](image_embeds.reshape(image_embeds.size(0), -1))

        seq_embeds = seq_embeds.reshape(image_embeds.shape)
        seq_logits = self.cytoself.fc_layers[1](seq_embeds.reshape(seq_embeds.size(0), -1))

        return image_embeds, image_logits, seq_embeds, seq_logits


    def loss(self, image_embeds, image_logits, seq_embeds, seq_logits, labels):
        loss = {}
        loss['mse'] = self.criteion_latents(image_embeds, seq_embeds)
        loss['seq_ce'] = self.criteion_classifier(seq_logits, labels)
        loss['image_ce'] = self.criteion_classifier(image_logits, labels)
        loss['loss'] = loss['mse'] + self.classifier_weight * loss['seq_ce']
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):       
        image_embeds, image_logits, seq_embeds, seq_logits = self(batch)
        loss = self.loss(image_embeds, image_logits, seq_embeds, seq_logits, batch['label'])

        self.log(
            "train_loss", loss['loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "train_mse", loss['mse'], on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "train_ce_seq", loss['seq_ce'], on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "train_ce_image", loss['image_ce'], on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image_embeds, image_logits, seq_embeds, seq_logits = self(batch)
        loss = self.loss(image_embeds, image_logits, seq_embeds, seq_logits, batch['label'])

        self.log(
            "val_loss", loss['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
        )
        self.log(
            "val_mse", loss['mse'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
        )
        self.log(
            "val_ce_seq", loss['seq_ce'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
        )
        self.log(
            "val_ce_image", loss['image_ce'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
        )

    def configure_optimizers(self):
        if self.cytoself is not None:
            for param in self.cytoself.parameters():
                    param.requires_grad = False

        params = list(self.esm_bottleneck.parameters())
        
        if self.protein_projection is not None:
            params = params + list(self.protein_projection.parameters())

        if self.esm is not None:
            params = params + list(self.esm.parameters())

        params = [p for p in params if p.requires_grad]

        optimizer = optim.AdamW(
            params,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.optim_config.warmup,
            num_training_steps=self.optim_config.max_iters,
            min_value = self.optim_config.learning_rate_min_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",  # 'step' since you're updating per batch/iteration
            },
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration