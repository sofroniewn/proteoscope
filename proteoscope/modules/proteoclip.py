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
import torch.distributed as dist

from .autoencoder import AutoencoderLM
from .esm_bottleneck import ESMBottleneck
from .scheduler import get_cosine_schedule_with_warmup


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)

        
class ProteoclipLM(LightningModule):
    def __init__(
        self,
        module_config,
    ):
        super(ProteoclipLM, self).__init__()

        if module_config.model.esm is None:
            self.esm = None
            self.esm_converter = None
            self.esm_embedding_layer = None
        else:
            self.esm, alphabet = esm.pretrained.load_model_and_alphabet_hub(module_config.model.esm.model_name)
            self.esm = self.esm.half()
            if module_config.model.esm.lora is not None:
                peft_config = LoraConfig(**module_config.model.esm.lora)
                self.esm = get_peft_model(self.esm, peft_config)
            self.esm_converter = alphabet.get_batch_converter(module_config.model.esm.truncation_seq_length)
            self.esm_embedding_layer = module_config.model.esm.embedding_layer

        self.esm_bottleneck = ESMBottleneck(module_config.model.esm_bottleneck)

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
        self.autoencoder.to(self.device)

        self.image_projection = ProjectionHead(
            embedding_dim= module_config.model.image_embedding_dims,
            projection_dim= module_config.model.projection_dims,
            dropout= module_config.model.dropout,
        )
        self.protein_projection = ProjectionHead(
            embedding_dim=module_config.model.protein_embedding_dims,
            projection_dim= module_config.model.projection_dims,
            dropout=module_config.model.dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.temperature =  module_config.model.temperature

        self.image_embeddings = []
        self.protein_embeddings = []
        self.image_embeddings_val = []
        self.protein_embeddings_val = []

    def gather_all(self, tensor_list):
        # # Gather from all processes
        # world_size = dist.get_world_size()
        # gathered_list = [torch.zeros_like(tensor_list[0]) for _ in range(world_size)]
        # dist.all_gather(gathered_list, tensor_list[0])
        # tensor_list = gathered_list
        return torch.cat(tensor_list, dim=0)

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
        # Mean pool
        seq_mask = seq_mask.unsqueeze(-1)
        seq_embeds = (seq_embeds * seq_mask).sum(dim=1) / seq_mask.sum(dim=1)
        seq_embeds = self.protein_projection(seq_embeds)

        # Create latents
        with torch.no_grad():
            image_embeds = self.autoencoder.encode(batch["image"]).latent_dist.sample()
            image_embeds = image_embeds.view((image_embeds.shape[0], -1))
        image_embeds = self.image_projection(image_embeds)

        return image_embeds, seq_embeds

    def clip_loss(self, image_embeddings, protein_embeddings):
        # Create the logits
        logits_for_images = (protein_embeddings @ image_embeddings.T) / self.temperature
        logits_for_proteins = (image_embeddings @ protein_embeddings.T) / self.temperature

        # Create a target tensor where each image/protein is its own class
        targets = torch.arange(logits_for_images.size(0)).to(self.device)

        # Compute the losses using CrossEntropyLoss
        images_loss = self.criterion(logits_for_images, targets)
        proteins_loss = self.criterion(logits_for_proteins, targets)

        # Combine and return the losses
        combined_loss = (images_loss + proteins_loss) / 2.0
        return combined_loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        image_embeds, seq_embeds = self(batch)

        # Store embeddings
        self.image_embeddings.append(image_embeds)
        self.protein_embeddings.append(seq_embeds)

        # If we've reached our effective batch size, compute the loss
        if len(self.image_embeddings) >= self.trainer.accumulate_grad_batches:
            # Gather embeddings across GPUs
            all_image_embeddings = self.gather_all(self.image_embeddings)
            all_protein_embeddings = self.gather_all(self.protein_embeddings)

            # Compute the loss using all embeddings
            loss = self.clip_loss(all_image_embeddings, all_protein_embeddings).sum()

            self.log(
                "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
            )

            # Clear the stored embeddings for the next set of samples
            del self.image_embeddings
            del self.protein_embeddings
            torch.cuda.empty_cache()
            self.image_embeddings = []
            self.protein_embeddings = []
            
            return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if len(self.image_embeddings) > 0:
            # Clear the stored embeddings for the next set of samples
            del self.image_embeddings
            del self.protein_embeddings
            torch.cuda.empty_cache()
            self.image_embeddings = []
            self.protein_embeddings = []

        image_embeds, seq_embeds = self(batch)

        # Store embeddings
        self.image_embeddings_val.append(image_embeds)
        self.protein_embeddings_val.append(seq_embeds)

        # If we've reached our effective batch size, compute the loss
        if len(self.image_embeddings_val) >= self.trainer.accumulate_grad_batches:
            # Gather embeddings across GPUs
            all_image_embeddings = self.gather_all(self.image_embeddings_val)
            all_protein_embeddings = self.gather_all(self.protein_embeddings_val)

            # Compute the loss using all embeddings
            loss = self.clip_loss(all_image_embeddings, all_protein_embeddings).sum()

            self.log(
                "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,

            )

            # Clear the stored embeddings for the next set of samples
            del self.image_embeddings_val
            del self.protein_embeddings_val
            torch.cuda.empty_cache()
            self.image_embeddings_val = []
            self.protein_embeddings_val = []

    def configure_optimizers(self):
        if self.autoencoder is not None:
            for param in self.autoencoder.parameters():
                    param.requires_grad = False

        params = list(self.esm_bottleneck.parameters()) + list(self.protein_projection.parameters()) + list(self.image_projection.parameters())

        if self.esm is not None:
            params = params + list(self.esm.parameters())

        params = [p for p in params if p.requires_grad]

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