import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from .cytoselfmodule import CytoselfLightningModule


class ProteoscopeLightningModule(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteoscopeLightningModule, self).__init__()

        self.unet = UNet2DConditionModel(
            sample_size=4,  # the target image resolution
            in_channels=64,  # the number of input channels, 3 for RGB images
            out_channels=64,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "CrossAttnDownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=512,
        )

        self.cond_images = False

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=100)

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
        self.cytoself_model.to(self.unet.device)

        self.latents_shape = (64, 4, 4)
        self.latents_init_scale = 5.0
        self.unconditioned_probability = 0.2

    def forward(self, batch):
        seq_embeds = batch['sequence_embed']
        seq_mask = batch['sequence_mask']

        if torch.rand(1) < self.unconditioned_probability:
            seq_embeds = torch.zeros_like(seq_embeds)

        if self.cond_images:
            cond_images = batch['image'][:, 1, :, :].unsqueeze(dim=1)
        else:
            cond_images = None

        with torch.no_grad():
            latents = self.cytoself_model(batch['image'], self.cytoself_layer).float() / self.latents_init_scale
       
        # Sample noise to add to the latents
        noise = torch.randn(latents.shape).to(latents)

        # Sample a random timestep for each latent in batch
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],))
        timesteps = timesteps.to(latents).long()

        # Add noise to the clean latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=seq_embeds, encoder_attention_mask=seq_mask, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)

        return loss

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

    def sample(self, batch, guidance_scale=1.0, cond_images=None, num_inference_steps=None):
        seq_embeds = batch['sequence_embed'].to(self.unet.device)
        seq_mask = batch['sequence_mask'].to(self.unet.device)
        
        if cond_images is None and self.cond_images:
            cond_images = batch['image'][:, 1, :, :].unsqueeze(dim=1)

        if cond_images is not None:
            cond_images.to(self.unet.device)

        bs = batch['image'].shape[0]
        latents_shape = (bs,) + self.latents_shape

        # Initialize latents
        latents = torch.randn(latents_shape).to(self.unet.device)

        # set step values
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.noise_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents, torch.zeros_like(latents)])
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=seq_embeds, encoder_attention_mask=seq_mask).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        return latents * self.latents_init_scale

    def configure_optimizers(self):
        params = self.unet.parameters()

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
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
