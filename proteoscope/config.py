from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


@dataclass
class DataConfig:
    images_path: str
    labels_path: str


@dataclass
class VqArgs:
    num_embeddings: int
    embedding_dim: int


@dataclass
class ModelConfig:
    input_shape: Tuple[int]
    emb_shapes: Tuple[Tuple[int]]
    output_shape: Tuple[int]
    fc_output_idx: Tuple[int]
    fc_input_type: str
    num_class: Optional[int]


@dataclass
class OptimizerConfig:
    learning_rate: float
    beta_1: float
    beta_2: float
    eps: float
    weight_decay: float


@dataclass
class ModuleConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    vq_coeff: int
    fc_coeff: int


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
