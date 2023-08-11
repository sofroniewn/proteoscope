import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf

from diffusers import AutoencoderKL
from torchvision.ops import MLP
from diffusers.optimization import get_cosine_schedule_with_warmup
from piqa import SSIM


def combine_images(img_set1, img_set2):
    n = img_set1.shape[0]
    row1 = torch.cat(img_set1.chunk(n, dim=0), dim=2).squeeze(0)
    row2 = torch.cat(img_set2.chunk(n, dim=0), dim=2).squeeze(0)
    return torch.cat([row1, row2], dim=0)


# class GradualKLStepScheduler:
#     def __init__(self, start_step, total_steps, start_value=0.0, end_value=1.0):
#         self.start_step = start_step
#         self.total_steps = total_steps
#         self.start_value = start_value
#         self.end_value = end_value
#         self.kl_weight = start_value

#     def step(self, current_step):
#         if current_step >= self.start_step:
#             progress = min(1, (current_step - self.start_step) / self.total_steps)
#             self.kl_weight = self.start_value + progress * (self.end_value - self.start_value)
#         return self.kl_weight


class AutoencoderLightningModule(LightningModule):
    def __init__(
        self,
        num_class,
        module_config,
        image_variance = 0.0167,
    ):
        super(AutoencoderLightningModule, self).__init__()
        self.image_variance = image_variance

        # model_args = module_config.model
        # if num_class is not None:
        #     model_args['num_class'] = num_class
        # # Conversion needed due to https://github.com/royerlab/cytoself/blob/9f482391a8e7101fde007184f321471cb983d94e/cytoself/trainer/autoencoder/cytoselffull.py#L382
        # model_args = OmegaConf.to_container(model_args)
        # del model_args['vq_coeff']
        # del model_args['fc_coeff']

        # model_args['encoder_args'] = [{}] #default_block_args[:len(model_args['emb_shapes'])]
        # self.model = CytoselfFull(**model_args)

        self.vae = AutoencoderKL(
            in_channels=2,  # the number of input channels, 3 for RGB images
            out_channels=2,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 128, 128),  # the number of output channels for each UNet block
            latent_channels=16,
            down_block_types=(
                "DownEncoderBlock2D",  # a regular ResNet downsampling block
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",  # a regular ResNet upsampling block
                "UpDecoderBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),    
        )

        in_channels = 16 * 12 * 12
        self.classifier = MLP(in_channels, [in_channels*2, num_class], dropout=0.2, inplace=False)

        self.optim_config = module_config.optimizer

        self.image_criterion = nn.MSELoss()
        self.labels_criterion = nn.CrossEntropyLoss()
        self.classifier_coeff = 0.1 # module_config.model.fc_coeff
        self.kl_coeff = 1e-8
        self.ssim = SSIM(n_channels=1)
        # self.kl_scheduler = GradualKLStepScheduler(1000, 100_000, 1e-5, 1e-3)

    def encode(self, images):
        latent_dist = self.vae.encode(images).latent_dist
        return latent_dist.mean

    def sample(self, images):
        latent_dist = self.vae.encode(images).latent_dist
        mu = latent_dist.mean
        reconstructed = self.vae.decode(mu).sample
        return reconstructed

    def forward(self, images):
        latent_dist = self.vae.encode(images).latent_dist
        mu, logvar = latent_dist.mean, latent_dist.logvar
        
        # Reparameterization trick
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std

        reconstructed = self.vae.decode(z).sample
        logits = self.classifier(mu.reshape(mu.shape[0], -1))

        return reconstructed, mu, logvar, logits

    def _calc_losses(self, images, labels, output_images, output_mu, output_logvar, output_logits):

        loss = {}
        loss['kl_divergence'] = -0.5 * torch.sum(1 + output_logvar - output_mu.pow(2) - output_logvar.exp())
        loss['reconstruction'] = self.image_criterion(output_images, images) / self.image_variance
        loss['classification'] = self.labels_criterion(output_logits, labels)

        loss['loss'] = loss['reconstruction'] + self.kl_coeff * loss['kl_divergence'] + self.classifier_coeff * loss['classification']
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['image']
        labels = batch['label']
        output_images, output_mu, output_logvar, output_logits = self(images)

        # self.kl_coeff = self.kl_scheduler.step(self.global_step)
        loss = self._calc_losses(images, labels, output_images, output_mu, output_logvar, output_logits)        

        for key, value in loss.items():
            self.log(
                "train_" + key, value, on_step=True, on_epoch=False, prog_bar=True, logger=True
            )
        
        # self.log(
        #     "kl_coeff", self.kl_coeff, on_step=True, on_epoch=False, prog_bar=True, logger=True
        # )

        return loss['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['image']
        labels = batch['label']
        output_images, output_mu, output_logvar, output_logits = self(images)
        loss = self._calc_losses(images, labels, output_images, output_mu, output_logvar, output_logits)            

        for key, value in loss.items():
            self.log(
                "val_" + key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )

        ssim_score = {}
        ssim_score['ssim_pro'] = self.ssim(images[:, 0].unsqueeze_(1), torch.clip(output_images[:, 0].unsqueeze_(1), 0, 1))
        ssim_score['ssim_nuc'] = self.ssim(images[:, 1].unsqueeze_(1), torch.clip(output_images[:, 1].unsqueeze_(1), 0, 1))

        for key, value in ssim_score.items():
            self.log(
                "val_" + key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )

        return images[0].unsqueeze_(0), output_images[0].unsqueeze_(0)

    def validation_epoch_end(self, results):
        if len(results) == 0:
            return

        images = torch.cat([r[0] for r in results])
        output_images = torch.cat([r[1] for r in results])

        pro = combine_images(images[:16:2, 0], output_images[:16:2, 0])
        nuc = combine_images(images[:16:2, 1], output_images[:16:2, 1])

        if self.global_step > 0:
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_image(
                "pro",
                pro,
                self.global_step,
                dataformats="HW",
            )

            tensorboard_logger.add_image(
                "nuc",
                nuc,
                self.global_step,
                dataformats="HW",
            )


    def configure_optimizers(self):
        params = list(self.vae.parameters()) + list(self.classifier.parameters())

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
