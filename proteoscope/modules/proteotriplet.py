import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm
import esm
from peft import LoraConfig, get_peft_model
import gc

from .autoencoder import AutoencoderLM
from .esm_bottleneck import ESMBottleneck
from .scheduler import get_cosine_schedule_with_warmup


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

        
class ProteoclipLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteoclipLM, self).__init__()

        if module_config.model.esm is None:
            self.esm = None
            self.esm_converter = None
            self.esm_embedding_layer = None
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

        autoencoder_checkpoint = module_config.model.autoencoder.checkpoint
        autoencoder_config = module_config.copy()
        autoencoder_config.model = module_config.model.autoencoder.model

        alm = AutoencoderLM.load_from_checkpoint(
            autoencoder_checkpoint,
            module_config=autoencoder_config,
        )

        self.autoencoder = alm.vae
        self.autoencoder.eval()

        self.image_projection = ProjectionHead(
            embedding_dim= module_config.model.image_embedding_dims,
            projection_dim= module_config.model.projection_dims,
            dropout= module_config.model.dropout,
        )
        self.protein_projection = ProjectionHead(
            embedding_dim=module_config.model.protein_embedding_dims,
            projection_dim= module_config.model.projection_dims,
            dropout=module_config.model.dropout,
        )
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bias =  module_config.model.bias

    def embed_sequence(self, batch, sequence_condition_probability=1.0):
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
        seq_embeds = self.protein_projection(seq_embeds)
        return seq_embeds   

    def forward(self, batch, sequence_condition_probability=1.0):        
        seq_embeds = self.embed(batch, sequence_condition_probability)

        # Create latents
        with torch.no_grad():
            image_embeds = self.autoencoder.encode(batch["image"]).latent_dist.sample()
            image_embeds = image_embeds.view((image_embeds.shape[0], -1))
            
            image_embeds_negative = self.autoencoder.encode(batch["image_negative"]).latent_dist.sample()
            image_embeds_negative = image_embeds_negative.view((image_embeds_negative.shape[0], -1))
        image_embeds = self.image_projection(image_embeds)
        image_embeds_negative = self.image_projection(image_embeds_negative)

        return image_embeds, image_embeds_negative, seq_embeds

    def triplet_loss(self, image_embeds, image_embeds_negative, seq_embeds):
        positive_cos = self.cosine_similarity(image_embeds, seq_embeds)
        negative_cos = self.cosine_similarity(image_embeds_negative, seq_embeds)
        return torch.clamp_min(negative_cos - positive_cos + self.bias, 0)

    def training_step(self, batch, batch_idx, dataloader_idx=0):       
        image_embeds, image_embeds_negative, seq_embeds = self(batch)

        # Compute the loss using all embeddings
        loss = self.triplet_loss(image_embeds, image_embeds_negative, seq_embeds).mean()

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image_embeds, image_embeds_negative, seq_embeds = self(batch)

        # Compute the loss using all embeddings
        loss = self.triplet_loss(image_embeds, image_embeds_negative, seq_embeds).mean()

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,

        )

    def configure_optimizers(self):
        if self.autoencoder is not None:
            for param in self.autoencoder.parameters():
                    param.requires_grad = False

        params = list(self.esm_bottleneck.parameters()) + list(self.protein_projection.parameters()) + list(self.image_projection.parameters())

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