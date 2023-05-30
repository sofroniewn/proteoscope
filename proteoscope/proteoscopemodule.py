import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule

from imagen_pytorch import Unet, Imagen
from .utils import CosineWarmupScheduler
from omegaconf import OmegaConf


class ProteoscopeLightningModule(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteoscopeLightningModule, self).__init__()

        self.unet_number = module_config.unet_number
        unet1_args = OmegaConf.to_container(module_config.model.unet1)
        unet2_args = OmegaConf.to_container(module_config.model.unet2)

        unet1 = Unet(**unet1_args)
        unet2 = Unet(**unet2_args)

        self.model = Imagen(
            unets = (unet1, unet2),
            image_sizes = tuple(module_config.model.image_sizes),
            timesteps = module_config.model.timesteps,
            cond_drop_prob = module_config.model.cond_drop_prob,
            channels=module_config.model.channels,
            text_embed_dim=module_config.model.text_embed_dim,
        )

        self.optim_config = module_config.optimizer

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        seq_embeds = batch['sequence_embed']
        seq_mask = batch['sequence_mask']
        images = batch['image'][:, 0, :, :].unsqueeze(dim=1)
        cond_images = batch['image'][:, 1, :, :].unsqueeze(dim=1)

        loss = self.model(images, text_embeds = seq_embeds, text_masks=seq_mask, cond_images = cond_images, unet_number = self.unet_number)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        seq_embeds = batch['sequence_embed']
        seq_mask = batch['sequence_mask']
        images = batch['image'][:, 0, :, :].unsqueeze(dim=1)
        cond_images = batch['image'][:, 1, :, :].unsqueeze(dim=1)

        loss = self.model(images, text_embeds = seq_embeds, text_masks=seq_mask, cond_images = cond_images, unet_number = self.unet_number)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )

    def configure_optimizers(self):
        params = self.model.parameters()

        optimizer = optim.AdamW(
            params,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.optim_config.warmup,
            max_iters=self.optim_config.max_iters,
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
