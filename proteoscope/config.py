from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Union


@dataclass
class DataConfig:
    images_path: str
    labels_path: str
    sequences_path: Optional[str]
    trim: Optional[int]
    sequence_embedding: Optional[str]


@dataclass
class VqArgs:
    num_embeddings: int
    embedding_dim: int


@dataclass
class FcArgs:
    num_layers: int


@dataclass
class CytoselfModelConfig:
    input_shape: Tuple[int]
    emb_shapes: Tuple[Tuple[int]]
    output_shape: Tuple[int]
    vq_args: VqArgs
    fc_args: FcArgs
    fc_output_idx: Tuple[int]
    fc_input_type: str
    num_class: Optional[int]
    vq_coeff: int
    fc_coeff: int


@dataclass
class AutoencoderModelConfig:
    in_channels: int
    out_channels: int
    layers_per_block: int
    block_out_channels: Tuple[int]
    latent_channels: int
    down_block_types: Tuple[str]
    up_block_types: Tuple[str]
    classifier_weight: float
    kl_weight: float
    perceptual_weight: float
    pixel_weight: float


@dataclass
class UNetConfig:
    dim: int
    cond_dim: int
    dim_mults: Tuple[int]
    num_resnet_blocks: Tuple[int]
    layer_attns: Tuple[bool]
    layer_cross_attns: Tuple[bool]
    cond_images_channels: int
    channels: int


@dataclass
class ProteoscopeModelConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    layers_per_block: int
    block_out_channels: Tuple[int]
    down_block_types: Tuple[str]
    up_block_types: Tuple[str]
    cross_attention_dim: int
    num_train_timesteps: int
    cond_images: bool
    unconditioned_probability: float
    latents_init_scale: float
    guidance_scale: float
    autoencoder: AutoencoderModelConfig
    autoencoder_checkpoint: str


@dataclass
class OptimizerConfig:
    learning_rate: float
    beta_1: float
    beta_2: float
    eps: float
    weight_decay: float
    warmup: int
    max_iters: int
    learning_rate_min_factore: float


@dataclass
class ModuleConfig:
    model: Union[CytoselfModelConfig, AutoencoderModelConfig, ProteoscopeModelConfig]
    optimizer: OptimizerConfig
    image_variance: float
    dropout: float
    image_height: int


@dataclass
class TrainerConfig:
    device: str
    precision: str
    num_devices: int
    val_check_interval: int
    limit_val_batches: int
    log_every_n_steps: int
    batch_size: int
    num_workers: int
    gradient_clip_val: Optional[float]


@dataclass
class ProteoscopeConfig:
    data: DataConfig
    module: ModuleConfig
    trainer: TrainerConfig
    model_type: str
