import torch
import torch.nn as nn
import torch.optim as optim
from cytoself.trainer.autoencoder.cytoselffull import (CytoselfFull,
                                                       default_block_args)
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule


class CytoselfLightningModule(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(CytoselfLightningModule, self).__init__()
        self.image_variance = module_config.image_variance

        model_args = module_config.model
        model_args["num_class"] = module_config.num_class
        # Conversion needed due to https://github.com/royerlab/cytoself/blob/9f482391a8e7101fde007184f321471cb983d94e/cytoself/trainer/autoencoder/cytoselffull.py#L382
        model_args = OmegaConf.to_container(model_args)
        del model_args["vq_coeff"]
        del model_args["fc_coeff"]

        self.model = CytoselfFull(**model_args)

        self.optim_config = module_config.optimizer

        self.image_criterion = nn.MSELoss()
        self.labels_criterion = nn.CrossEntropyLoss()
        self.vq_coeff = module_config.model.vq_coeff
        self.fc_coeff = module_config.model.fc_coeff

    def forward(self, batch):
        return self.model.forward(batch)

    def _calc_losses(self, images, labels, output_images, output_logits):
        self.model.mse_loss["reconstruction1_loss"] = (
            self.image_criterion(output_images, images) / self.image_variance
        )
        self.model.fc_loss = {
            f"fc{self.model.fc_output_idx[0]}_loss": self.labels_criterion(
                output_logits, labels
            )
        }
        vq_loss = torch.stack([d["loss"] for d in self.model.vq_loss.values()]).sum()
        mse_loss = torch.stack([d for d in self.model.mse_loss.values()]).sum()
        fc_loss = torch.stack([d for d in self.model.fc_loss.values()]).sum()

        loss = mse_loss + self.fc_coeff * fc_loss + self.vq_coeff * vq_loss

        output = {"loss": loss.item()}
        output.update({k: v.item() for k, v in self.model.fc_loss.items()})
        output.update({k: v.item() for k, v in self.model.perplexity.items()})
        output.update(
            {k: self.model.mse_loss[k].item() for k in sorted(self.model.mse_loss)}
        )
        vq_loss_dict = {}
        for key0, val0 in self.model.vq_loss.items():
            for key1, val1 in val0.items():
                vq_loss_dict[key0 + "_" + key1] = val1.item()
        output.update(vq_loss_dict)

        return loss, output

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        labels = batch["label"]
        output_images, output_logits = self.model(images)
        loss, all_outputs = self._calc_losses(
            images, labels, output_images, output_logits
        )

        for key, value in all_outputs.items():
            self.log(
                "train_" + key,
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        labels = batch["label"]
        output_images, output_logits = self.model(images)
        loss, all_outputs = self._calc_losses(
            images, labels, output_images, output_logits
        )

        for key, value in all_outputs.items():
            self.log(
                "val_" + key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
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
