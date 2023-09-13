import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
from .esm_bottleneck import ESMBottleneck
import esm
from peft import LoraConfig, get_peft_model


class ProteolocLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteolocLM, self).__init__()
        if module_config.model.esm_model is None:
            self.esm_bottleneck = ESMBottleneck(module_config.model)
            self.esm = None
            self.converter = None
        else:
            self.esm, alphabet = esm.pretrained.load_model_and_alphabet_hub(module_config.model.esm_model)
            if module_config.model.lora is not None:
                peft_config = LoraConfig(**module_config.model.lora)
                self.esm = get_peft_model(self.esm, peft_config)
            self.converter = alphabet.get_batch_converter(module_config.model.truncation_seq_length)
            self.embedding_layer = module_config.model.embedding_layer
            self.esm_bottleneck = None

        self.prediction_head = nn.Linear(module_config.model.d_model, module_config.model.num_class)
        self.optim_config = module_config.optimizer
        self.criterion = nn.CrossEntropyLoss()

    def embed(self, batch, shift_embedding_layer=0):
        if self.esm is not None:
            labels = batch['index']
            sequence = batch['peptide']
            result = list(zip(labels, sequence))
            labels, strs, toks = self.converter(result)
            toks = toks.to(self.device)
            embedding_layer = self.embedding_layer - shift_embedding_layer
            out = self.esm(toks, repr_layers=[embedding_layer], return_contacts=False)
            seq_embeds = out["representations"][embedding_layer]
            seq_mask = torch.zeros_like(seq_embeds).bool()
            for i, ind in enumerate(batch['truncation']):
                seq_mask[i, 1:ind+1] = True
        else:
            seq_embeds, seq_mask = self.esm_bottleneck(
                batch["sequence_embed"], batch["sequence_mask"]
            )
        seq_embeds[~seq_mask] = 0
        return seq_embeds

    def forward(self, batch):        
        seq_embeds = self.embed(batch)
        embeds = seq_embeds.sum(dim=-2) / batch['truncation'][:, None]
        logits = self.prediction_head(embeds)
        return logits

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        loss = self.criterion(logits, batch['localization'])

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        loss = self.criterion(logits, batch['localization'])

        self.log(
            f"val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        if self.esm is None:
            params = list(self.prediction_head.parameters()) + list(self.esm_bottleneck.parameters())
        else:
            params = list(self.prediction_head.parameters()) + list(self.esm.parameters())

        optimizer = optim.AdamW(
            params,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )

        self.lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.optim_config.max_iters,
            eta_min=self.optim_config.learning_rate_min_factor
            * self.optim_config.learning_rate,
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