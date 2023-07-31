import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule

from imagen_pytorch import Unet, Imagen
# from .utils import CosineWarmupScheduler
from omegaconf import OmegaConf
from .cytoselfmodule import CytoselfLightningModule


class ProteoscopeLightningModule(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteoscopeLightningModule, self).__init__()

        self.unet_number = module_config.unet_number
        self.cond_images = module_config.model.unet1.cond_images_channels > 0

        unet1_args = OmegaConf.to_container(module_config.model.unet1)

        unet1 = Unet(**unet1_args)

        self.imagen = Imagen(
            # condition_on_text = False, ###
            unets = (unet1,),
            image_sizes = (module_config.model.latent_size,),
            timesteps = module_config.model.timesteps,
            cond_drop_prob = module_config.model.cond_drop_prob,
            channels = module_config.model.channels,
            text_embed_dim=module_config.model.text_embed_dim,
            auto_normalize_img = False,
            dynamic_thresholding = False,
        )

        self.optim_config = module_config.optimizer

        self.cytoself_layer = module_config.model.cytoself_layer
        cytoself_checkpoint = module_config.model.cytoself_checkpoint
        module_config.model = module_config.model.cytoself
        clm = CytoselfLightningModule.load_from_checkpoint(
            cytoself_checkpoint,
            module_config=module_config,
            num_class=None,
        )
        self.cytoself_model = clm.model
        self.cytoself_model.eval()
        self.cytoself_model.to(self.imagen.device)

    def forward(self, batch):
        seq_embeds = batch['sequence_embed']
        seq_mask = batch['sequence_mask']
        if self.cond_images:
            cond_images = batch['image'][:, 1, :, :].unsqueeze(dim=1)
        else:
            cond_images = None

        with torch.no_grad():
            latents = self.cytoself_model(batch['image'], self.cytoself_layer).float()
       
        return self.imagen.forward(latents, text_embeds = seq_embeds, text_masks=seq_mask, cond_images = cond_images, unet_number = self.unet_number)
        # return self.imagen.forward(latents, text_embeds = None, text_masks=None, cond_images = None, unet_number = self.unet_number)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(batch)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )

    def sample(self, batch, cond_scale=1.0, cond_images=None):
        seq_embeds = batch['sequence_embed'].to(self.imagen.device)
        seq_mask = batch['sequence_mask'].to(self.imagen.device)
        
        if cond_images is None and self.cond_images:
            cond_images = batch['image'][:, 1, :, :].unsqueeze(dim=1)

        if cond_images is not None:
            cond_images.to(self.imagen.device)

        return self.imagen.sample(text_embeds=seq_embeds, text_masks=seq_mask, cond_images=cond_images, cond_scale=cond_scale)
        # return self.imagen.sample(text_embeds=None, text_masks=None, cond_images=None, cond_scale=cond_scale)

    def configure_optimizers(self):
        params = self.imagen.parameters()

        optimizer = optim.AdamW(
            params,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )
        # self.lr_scheduler = CosineWarmupScheduler(
        #     optimizer,
        #     warmup=self.optim_config.warmup,
        #     max_iters=self.optim_config.max_iters,
        # )
        return optimizer

    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     self.lr_scheduler.step()  # Step per iteration
