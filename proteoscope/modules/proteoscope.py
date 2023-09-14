import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from piqa import SSIM, PSNR
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm
from torchvision.transforms.functional import resize
from ema_pytorch import EMA
import esm
from peft import LoraConfig, get_peft_model

from .autoencoder import AutoencoderLM
from .cytoself import CytoselfLM
from .esm_bottleneck import ESMBottleneck
from .scheduler import get_cosine_schedule_with_warmup


def combine_images(img_set1, img_set2):
    n = img_set1.shape[0]
    row1 = torch.cat(img_set1.chunk(n, dim=0), dim=2).squeeze(0)
    row2 = torch.cat(img_set2.chunk(n, dim=0), dim=2).squeeze(0)
    return torch.cat([row1, row2], dim=0)


def zero_weights_init(m):
    if isinstance(m, (nn.Linear, nn.LayerNorm, nn.GroupNorm)):
        m.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)


def freeze_cross_attention(model):
    for module in model.modules():
        if isinstance(module, BasicTransformerBlock):
            crossattention = module.attn2
            if crossattention is not None:
                crossattention.apply(zero_weights_init)
                for param in crossattention.parameters():
                    param.requires_grad = False


def all_requires_grad(model):
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = True


def default_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def only_cross_attention(model):
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = True
    for module in model.modules():
        if isinstance(module, BasicTransformerBlock):
            crossattention = module.attn2
            if crossattention is not None:
                crossattention.apply(default_weights_init)
                # for param in crossattention.parameters():
                #     param.requires_grad = True
                # for param in module.norm2.parameters():
                #     param.requires_grad = True
                # for param in module.ff.parameters():
                #     param.requires_grad = True


class LinearRampScheduler:
    def __init__(self, initial_steps, ramp_steps):
        self.initial_steps = initial_steps
        self.ramp_steps = ramp_steps

    def get_val(self, current_step):
        if current_step <= self.initial_steps:
            return 0.0
        elif current_step <= self.initial_steps + self.ramp_steps:
            progress = (current_step - self.initial_steps) / self.ramp_steps
            return progress
        else:
            return 1.0


class ProteoscopeLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteoscopeLM, self).__init__()
        self.unet = UNet2DConditionModel(
            sample_size=module_config.model.sample_size,  # the target image resolution
            in_channels=module_config.model.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=module_config.model.out_channels,  # the number of output channels
            layers_per_block=module_config.model.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=module_config.model.block_out_channels,  # the number of output channels for each UNet block
            down_block_types=module_config.model.down_block_types,
            up_block_types=module_config.model.up_block_types,
            cross_attention_dim=module_config.model.cross_attention_dim,
        )
        self.ema = EMA(
            self.unet,
            beta=module_config.model.ema_decay,
            update_after_step=module_config.model.ema_update_after_step,
        )
        self.latents_shape = (
            module_config.model.out_channels,
            module_config.model.sample_size,
            module_config.model.sample_size,
        )
        if module_config.model.sequence_condition_probability is not None:
            self.sequence_condition_probability = module_config.model.sequence_condition_probability
        else:
            self.sequence_condition_probability = 0.0

        if module_config.model.unconditioned_initial_steps is not None and module_config.model.sequence_condition_ramp_steps is not None:
            self.sequence_condition_scheduler = LinearRampScheduler(module_config.model.unconditioned_initial_steps, module_config.model.sequence_condition_ramp_steps)
        else:
            self.sequence_condition_scheduler = None

        self.latents_init_scale = module_config.model.latents_init_scale
        self.cond_latents_init_scale = module_config.model.cond_latents_init_scale
        self.guidance_scale = module_config.model.guidance_scale
        self.num_val_timesteps = module_config.model.num_val_timesteps
        self.cond_images = module_config.model.cond_images

        if module_config.model.esm is None:
            self.esm = None
            self.esm_converter = None
            self.esm_embedding_layer = None
        else:
            self.esm, alphabet = esm.pretrained.load_model_and_alphabet_hub(module_config.model.esm.model_name)
            if module_config.model.esm.lora is not None:
                peft_config = LoraConfig(**module_config.model.esm.lora)
                self.esm = get_peft_model(self.esm, peft_config)
            self.esm_converter = alphabet.get_batch_converter(module_config.model.esm.truncation_seq_length)
            self.esm_embedding_layer = module_config.model.esm.embedding_layer

        self.esm_bottleneck = ESMBottleneck(module_config.model.esm_bottleneck)

        self.freeze_cross_attention = module_config.model.freeze_cross_attention
        self.only_cross_attention = module_config.model.only_cross_attention

        if module_config.model.scheduler == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=module_config.model.num_train_timesteps,
                clip_sample=False,
            )
        elif module_config.model.scheduler == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=module_config.model.num_train_timesteps,
                clip_sample=False,
            )
        else:
            raise ValueError(f"Unrecognized schedule {module_config.model.scheduler}")

        self.optim_config = module_config.optimizer

        autoencoder_checkpoint = module_config.model.autoencoder.checkpoint
        autoencoder_config = module_config.copy()
        autoencoder_config.model = module_config.model.autoencoder.model

        alm = AutoencoderLM.load_from_checkpoint(
            autoencoder_checkpoint,
            module_config=autoencoder_config,
        )

        self.autoencoder = alm.vae
        self.autoencoder.eval()
        self.autoencoder.to(self.unet.device)

        if module_config.model.cytoself is not None:
            cytoself_checkpoint = module_config.model.cytoself.checkpoint
            cytoself_config = module_config.copy()
            cytoself_config.model = module_config.model.cytoself.model
            clm = CytoselfLM.load_from_checkpoint(
                cytoself_checkpoint,
                module_config=cytoself_config,
                num_class=module_config.model.cytoself.num_class,
            )
            self.cytoself = clm.model
            self.cytoself.eval()
            self.cytoself.to(self.unet.device)
        else:
            self.cytoself = None

        self.ssim = SSIM(n_channels=1)
        self.psnr = PSNR()
        self.results = []

    def embed_sequence(self, batch, sequence_condition_probability=1.0):
        if self.esm is None:
            seq_embeds = batch["sequence_embed"]
            seq_mask = batch["sequence_mask"]
        else:
            labels = batch['index']
            sequence = batch['peptide']
            result = list(zip(labels, sequence))
            labels, strs, toks = self.esm_converter(result)
            toks = toks.to(self.device)
            embedding_layer = self.esm_embedding_layer
            out = self.esm(toks, repr_layers=[embedding_layer], return_contacts=False)
            seq_embeds = out["representations"][embedding_layer]
            seq_embeds = seq_embeds[:, 1:-1]
            seq_mask = torch.zeros(seq_embeds.shape[:-1]).bool().to(self.device)
            for i, ind in enumerate(batch['truncation']):
                seq_mask[i, :ind] = True

        seq_embeds, seq_mask = self.esm_bottleneck(
            seq_embeds, seq_mask, sequence_condition_probability
        )

        return seq_embeds, seq_mask

    def forward(self, batch, sequence_condition_probability=1.0):        
        seq_embeds, seq_mask = self.embed_sequence(batch, sequence_condition_probability)

        if self.cond_images:
            cond_images = batch["nuclei_distance"]
        else:
            cond_images = None

        # Create latents
        with torch.no_grad():
            if self.latents_init_scale is None:
                first_batch_latents = self.autoencoder.encode(
                    batch["image"]
                ).latent_dist.mode()
                latent_init_mean = first_batch_latents.mean()
                self.latents_init_scale = (
                    (first_batch_latents - latent_init_mean).pow(2).mean().pow(0.5)
                )

            latents = (
                self.autoencoder.encode(batch["image"]).latent_dist.sample()
                / self.latents_init_scale
            )

            if cond_images is not None:
                latents_cond = (
                    resize(cond_images, self.latents_shape[-2:])
                    / self.cond_latents_init_scale
                )
                latents_cond = latents_cond.unsqueeze(dim=1)
            else:
                latents_cond = None

        # Sample noise to add to the latents
        noise = torch.randn(latents.shape).to(latents)

        # Sample a random timestep for each latent in batch
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],)
        )
        timesteps = timesteps.to(latents).long()

        # Add noise to the clean latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        if latents_cond is not None:
            model_input = torch.cat([noisy_latents, latents_cond], dim=1)
        else:
            model_input = noisy_latents

        noise_pred = self.unet(
            model_input,
            timesteps,
            encoder_hidden_states=seq_embeds,
            encoder_attention_mask=seq_mask,
            return_dict=False,
        )[0]
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        if self.freeze_cross_attention:
            scp = 0.0
        else:
            scp = self.sequence_condition_probability
            if self.sequence_condition_scheduler is not None:
                scp = scp * self.sequence_condition_scheduler.get_val(self.global_step)
        loss = self(batch, scp)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        self.log(
            "sequence_condition_probability", scp, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.freeze_cross_attention:
            scp = 0.0
        else:
            scp = self.sequence_condition_probability
            if self.sequence_condition_scheduler is not None:
                scp = scp * self.sequence_condition_scheduler.get_val(self.global_step)
        loss = self(batch, scp)

        if dataloader_idx == 1:
            extra = "_train"
        else:
            extra = ""

        self.log(
            f"val{extra}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        output_latents = self.sample(
            batch,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_val_timesteps,
            seed=42,
        )
        output_images = self.autoencoder.decode(output_latents).sample.clip(0, 1)

        scores = self.score(batch["image"], output_images, batch["label"])

        for key, value in scores.items():
            self.log(
                key + "_val" + extra,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        if self.global_rank == 0 and len(self.results) < 16 and dataloader_idx == 0:
            self.results.append(
                (
                    batch["image"],
                    output_images,
                )
            )

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        if len(self.results) == 0:
            return

        input_images = torch.cat([r[0] for r in self.results[:16]])
        output_images = torch.cat([r[1] for r in self.results[:16]])
        self.results = []

        pro = combine_images(input_images[:, 0], output_images[:, 0])
        nuc = combine_images(input_images[:, 1], output_images[:, 1])

        if self.global_step > 0:
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_image(
                "img_pro",
                pro,
                self.global_step,
                dataformats="HW",
            )

            tensorboard_logger.add_image(
                "img_nuc",
                nuc,
                self.global_step,
                dataformats="HW",
            )

    @torch.no_grad()
    def sample(
        self,
        batch,
        guidance_scale=1.0,
        cond_images=None,
        num_inference_steps=None,
        seed=None,
    ):
        if seed is not None:
            generator = torch.Generator(self.unet.device).manual_seed(seed)
        else:
            generator = None

        seq_embeds, seq_mask = self.embed_sequence(batch)

        if cond_images is None and self.cond_images:
            cond_images = batch["nuclei_distance"]

        if cond_images is not None:
            cond_images.to(self.unet.device)

        bs = batch["image"].shape[0]
        latents_shape = (bs,) + self.latents_shape

        # Initialize latents
        latents = torch.randn(
            latents_shape, generator=generator, device=self.unet.device
        )

        if cond_images is not None:
            latents_cond = (
                resize(cond_images, self.latents_shape[-2:])
                / self.cond_latents_init_scale
            )
            latents_cond = latents_cond.unsqueeze(dim=1)
            latents_cond = torch.cat([latents_cond] * 2).to(self.unet.device)
        else:
            latents_cond = None

        # set step values
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        seq_embeds = torch.cat([torch.zeros_like(seq_embeds), seq_embeds])
        seq_mask = torch.cat([torch.ones_like(seq_mask), seq_mask])

        for t in tqdm(self.noise_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            if latents_cond is not None:
                latent_model_input = torch.cat(
                    [latent_model_input, latents_cond], dim=1
                )

            # predict the noise residual
            noise_pred = self.ema(
                latent_model_input,
                t,
                encoder_hidden_states=seq_embeds,
                encoder_attention_mask=seq_mask,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        return latents * self.latents_init_scale

    @torch.no_grad()
    def score(self, input_image, output_image, labels):
        scores = {}

        if self.cytoself is not None:
            input_cytoself_embed = self.cytoself(input_image, "vqvec2")
            _, input_cytoself_logits = self.cytoself(input_image)

            output_cytoself_embed = self.cytoself(
                output_image, "vqvec2"
            )  # make 'vqvec2' param ....
            _, output_cytoself_logits = self.cytoself(output_image)

        scores["cytoself_distance"] = F.mse_loss(
            input_cytoself_embed, output_cytoself_embed
        )
        scores["cytoself_input_classifcation"] = F.cross_entropy(
            input_cytoself_logits, labels
        )
        scores["cytoself_output_classifcation"] = F.cross_entropy(
            output_cytoself_logits, labels
        )

        scores["ssim_pro"] = self.ssim(
            input_image[:, 0].unsqueeze_(1), output_image[:, 0].unsqueeze_(1)
        )
        scores["ssim_nuc"] = self.ssim(
            input_image[:, 1].unsqueeze_(1), output_image[:, 1].unsqueeze_(1)
        )

        scores["psnr_pro"] = self.psnr(
            input_image[:, 0].unsqueeze_(1), output_image[:, 0].unsqueeze_(1)
        )
        scores["psnr_nuc"] = self.psnr(
            input_image[:, 1].unsqueeze_(1), output_image[:, 1].unsqueeze_(1)
        )
        return scores

    def configure_optimizers(self):
        if self.freeze_cross_attention:
            freeze_cross_attention(self.unet)
            freeze_cross_attention(self.esm_bottleneck)

        if self.only_cross_attention:
            only_cross_attention(self.unet)
            all_requires_grad(self.esm_bottleneck)

        params = list(self.unet.parameters()) + list(self.esm_bottleneck.parameters())

        if self.esm is not None:
            params = params + list(self.esm.parameters())

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
            min_value = self.optim_config.learning_rate_min_factor,
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
        self.ema.update()
