import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule

from imagen_pytorch import Unet, Imagen
from .utils import CosineWarmupScheduler


class ProteoscopeLightningModule(LightningModule):
    def __init__(
        self,
        unet_number,
        module_config,
    ):
        super(ProteoscopeLightningModule, self).__init__()

        # model_args = module_config.model
        self.unet_number = unet_number

        unet1 = Unet(
            dim = 128,
            cond_dim = 128,
            dim_mults = (1, 2, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True),
            layer_cross_attns = (False, True, True),
            cond_images_channels = 1,
            channels=1,
        )

        unet2 = Unet(
            dim = 128,
            cond_dim = 128,
            dim_mults = (1, 2, 4),
            num_resnet_blocks = (2, 4, 8),
            layer_attns = (False, False, True),
            layer_cross_attns = (False, False, True),
            cond_images_channels = 1,
            channels=1,
        )

        self.model = Imagen(
            unets = (unet1, unet2),
            image_sizes = (32, 100),
            timesteps = 1000,
            cond_drop_prob = 0.1,
            channels=1,
            text_embed_dim=1280,
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
