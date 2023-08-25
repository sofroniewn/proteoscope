import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from piqa import SSIM
from pytorch_lightning import LightningModule
# from torchvision.ops import MLP
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from ..losses import LPIPSWithDiscriminator


def combine_images(img_set1, img_set2):
    n = img_set1.shape[0]
    row1 = torch.cat(img_set1.chunk(n, dim=0), dim=2).squeeze(0)
    row2 = torch.cat(img_set2.chunk(n, dim=0), dim=2).squeeze(0)
    return torch.cat([row1, row2], dim=0)


class AutoencoderLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(AutoencoderLM, self).__init__()

        self.vae = AutoencoderKL(
            in_channels=module_config.model.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=module_config.model.out_channels,  # the number of output channels
            layers_per_block=module_config.model.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=module_config.model.block_out_channels,  # the number of output channels for each UNet block
            latent_channels=module_config.model.latent_channels,
            down_block_types=module_config.model.down_block_types,
            up_block_types=module_config.model.up_block_types,
        )

        self.loss = LPIPSWithDiscriminator(
            module_config.model.loss.disc_start,
            kl_weight=module_config.model.loss.kl_weight,
            pixelloss_weight=module_config.model.loss.pixel_weight,
            perceptual_weight=module_config.model.loss.perceptual_weight,
            disc_weight=module_config.model.loss.disc_weight,
            disc_in_channels=module_config.model.out_channels
        )
        self.automatic_optimization = False
        
        # height = module_config.image_height / (
        #     2 ** (len(module_config.model.block_out_channels) - 1)
        # )
        # in_channels = int(module_config.model.latent_channels * height * height)
        # self.classifier = MLP(
        #     in_channels,
        #     [in_channels * 2, module_config.num_class],
        #     dropout=module_config.dropout,
        #     inplace=False,
        # )

        self.optim_config = module_config.optimizer

        # self.image_criterion = nn.L1Loss()
        # self.perceptual_criterion = LearnedPerceptualImagePatchSimilarity(
        #     net_type="vgg"
        # )
        # self.labels_criterion = nn.CrossEntropyLoss()

        self.ssim = SSIM(n_channels=1)
        self.results = []

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def forward(self, images, sample_posterior=True, return_embed=False):
        posterior = self.vae.encode(images).latent_dist

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        reconstructed = self.vae.decode(z).sample.clip(0, 1)
        # logits = self.classifier(z.reshape(z.shape[0], -1))

        if return_embed:
            return reconstructed, posterior, z
        else:
            return reconstructed, posterior

    def training_step(self, batch, batch_idx):        
        optimizer_ae, optimizer_disc = self.optimizers()
        scheduler_ae, scheduler_disc = self.lr_schedulers()

        inputs = batch["image"]
        # labels = batch["label"]
        reconstructions, posterior = self(inputs)

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )

        optimizer_ae.zero_grad()
        self.manual_backward(aeloss)
        optimizer_ae.step()
        scheduler_ae.step()

        # train the discriminator
        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        self.log(
            "discloss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        
        optimizer_disc.zero_grad()
        self.manual_backward(discloss)
        optimizer_disc.step()
        scheduler_disc.step()

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        # labels = batch["label"]
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log(
            "val_loss",
            log_dict_ae["val/rec_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            log_dict_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            log_dict_disc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        ssim_score = {}
        ssim_score["ssim_pro"] = self.ssim(
            inputs[:, 0].unsqueeze_(1),
            torch.clip(reconstructions[:, 0].unsqueeze_(1), 0, 1),
        )
        ssim_score["ssim_nuc"] = self.ssim(
            inputs[:, 1].unsqueeze_(1),
            torch.clip(reconstructions[:, 1].unsqueeze_(1), 0, 1),
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
            self.results.append((inputs[0].unsqueeze_(0), reconstructions[0].unsqueeze_(0)))
        return self.log_dict

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
        params_ae = list(self.vae.parameters())
        params_disc = list(self.loss.discriminator.parameters())

        optimizer_ae = optim.AdamW(
            params_ae,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )
        optimizer_disc = optim.AdamW(
            params_disc,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )

        lr_scheduler_ae = CosineAnnealingLR(
            optimizer_ae,
            T_max=self.optim_config.max_iters,
            eta_min=self.optim_config.learning_rate_min_factor
            * self.optim_config.learning_rate,
        )

        lr_scheduler_disc = CosineAnnealingLR(
            optimizer_disc,
            T_max=self.optim_config.max_iters,
            eta_min=self.optim_config.learning_rate_min_factor
            * self.optim_config.learning_rate,
        )
        return (
            {
                "optimizer": optimizer_ae,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_ae,
                    "interval": "step",
                },
            },
            {
                "optimizer": optimizer_disc,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_disc,
                    "interval": "step",
                },
            },
        )

    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     self.lr_scheduler_ae.step()  # Step per iteration
    #     self.lr_scheduler_disc.step()  # Step per iteration
