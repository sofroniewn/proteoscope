import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from piqa import SSIM
from pytorch_lightning import LightningModule
from torchvision.ops import MLP


def combine_images(img_set1, img_set2):
    n = img_set1.shape[0]
    row1 = torch.cat(img_set1.chunk(n, dim=0), dim=2).squeeze(0)
    row2 = torch.cat(img_set2.chunk(n, dim=0), dim=2).squeeze(0)
    return torch.cat([row1, row2], dim=0)


class AutoencoderLightningModule(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(AutoencoderLightningModule, self).__init__()

        self.vae = AutoencoderKL(
            in_channels=module_config.model.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=module_config.model.out_channels,  # the number of output channels
            layers_per_block=module_config.model.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=module_config.model.block_out_channels,  # the number of output channels for each UNet block
            latent_channels=module_config.model.latent_channels,
            down_block_types=module_config.model.down_block_types,
            up_block_types=module_config.model.up_block_types,
        )

        self.image_variance = module_config.image_variance
        # self.classifier_coeff = module_config.model.classifier_coeff
        self.kl_coeff = module_config.model.kl_coeff

        # height = module_config.image_height / (2 ** (len(module_config.model.block_out_channels) - 1))
        # in_channels = int(module_config.model.latent_channels * height * height)
        # self.classifier = MLP(in_channels, [in_channels*2, module_config.num_class], dropout=module_config.dropout, inplace=False)

        self.optim_config = module_config.optimizer

        self.image_criterion = nn.MSELoss()
        self.labels_criterion = nn.CrossEntropyLoss()

        self.ssim = SSIM(n_channels=1)
        self.results = []

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
        # logits = self.classifier(mu.reshape(mu.shape[0], -1))

        return reconstructed, mu, logvar, None

    def _calc_losses(
        self, images, labels, output_images, output_mu, output_logvar, output_logits
    ):
        loss = {}
        loss["kl_divergence"] = -0.5 * torch.sum(
            1 + output_logvar - output_mu.pow(2) - output_logvar.exp()
        )
        loss["reconstruction"] = (
            self.image_criterion(output_images, images) / self.image_variance
        )
        # loss['classification'] = self.labels_criterion(output_logits, labels)

        loss["loss"] = (
            loss["reconstruction"] + self.kl_coeff * loss["kl_divergence"]
        )  # + self.classifier_coeff * loss['classification']
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        labels = batch["label"]
        output_images, output_mu, output_logvar, output_logits = self(images)

        loss = self._calc_losses(
            images, labels, output_images, output_mu, output_logvar, output_logits
        )

        for key, value in loss.items():
            self.log(
                "train_" + key,
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )

        return loss["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        labels = batch["label"]
        output_images, output_mu, output_logvar, output_logits = self(images)
        loss = self._calc_losses(
            images, labels, output_images, output_mu, output_logvar, output_logits
        )

        for key, value in loss.items():
            self.log(
                "val_" + key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        ssim_score = {}
        ssim_score["ssim_pro"] = self.ssim(
            images[:, 0].unsqueeze_(1),
            torch.clip(output_images[:, 0].unsqueeze_(1), 0, 1),
        )
        ssim_score["ssim_nuc"] = self.ssim(
            images[:, 1].unsqueeze_(1),
            torch.clip(output_images[:, 1].unsqueeze_(1), 0, 1),
        )

        for key, value in ssim_score.items():
            self.log(
                "val_" + key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if self.global_rank == 0 and len(self.results) < 16:
            self.results.append((images[0].unsqueeze_(0), output_images[0].unsqueeze_(0)))

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        if len(self.results) == 0:
            return

        images = torch.cat([r[0] for r in self.results])
        output_images = torch.cat([r[1] for r in self.results])
        self.results = []
    
        pro = combine_images(images[:16, 0], output_images[:16, 0])
        nuc = combine_images(images[:16, 1], output_images[:16, 1])

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
        params = list(self.vae.parameters())  # + list(self.classifier.parameters())

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
