import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
from .esm_bottleneck import ESMBottleneck


class ProteolocLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteolocLM, self).__init__()
        self.esm_bottleneck = ESMBottleneck(module_config.model)
        self.prediction_head = nn.Linear(module_config.model.d_model, module_config.model.num_class)
        self.optim_config = module_config.optimizer
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):        
        seq_embeds, seq_mask = self.esm_bottleneck(
            batch["sequence_embed"], batch["sequence_mask"]
        )
        seq_embeds[~seq_mask] = 0
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
        params = list(self.prediction_head.parameters()) + list(self.esm_bottleneck.parameters())

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