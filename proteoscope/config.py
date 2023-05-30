from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Union


@dataclass
class DataConfig:
    images_path: str
    labels_path: str
    sequences_path: Optional[str]


@dataclass
class VqArgs:
    num_embeddings: int
    embedding_dim: int


@dataclass
class CytoselfModelConfig:
    input_shape: Tuple[int]
    emb_shapes: Tuple[Tuple[int]]
    output_shape: Tuple[int]
    fc_output_idx: Tuple[int]
    fc_input_type: str
    num_class: Optional[int]
    vq_coeff: int
    fc_coeff: int


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
    unet1: UNetConfig
    unet2: UNetConfig
    image_sizes: Tuple[int]
    timesteps: int
    cond_drop_prob: float
    channels: int
    text_embed_dim: int


@dataclass
class OptimizerConfig:
    learning_rate: float
    beta_1: float
    beta_2: float
    eps: float
    weight_decay: float
    warmup: int
    max_iters: int


@dataclass
class ModuleConfig:
    model: Union[CytoselfModelConfig, ProteoscopeModelConfig]
    optimizer: OptimizerConfig
    unet_number: Optional[int]


@dataclass
class TrainerConfig:
    device: str
    precision: str
    num_devices: int
    max_epochs: int
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
