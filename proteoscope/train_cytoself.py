import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .config import ProteoscopeConfig
from .modules import AutoencoderLM, CytoselfLM
from .data import ProteoscopeDM


def train_cytoself(config: ProteoscopeConfig) -> None:
    pdm = ProteoscopeDM(
        images_path=config.data.images_path,
        labels_path=config.data.labels_path,
        trim=config.data.trim,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        splits=config.splits,
    )
    pdm.setup()

    if config.model_type == "cytoself":
        clm = CytoselfLM(module_config=config.module, num_class=pdm.num_class)
    elif config.model_type == "autoencoder":
        clm = AutoencoderLM(module_config=config.module)
    else:
        raise ValueError(f"Unrecognized model type {config.model_type}")

    print(clm)
    print(f"Train samples {len(pdm.train_dataset)}, Val samples {len(pdm.val_dataset)}")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val_loss", mode="min", save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    if config.trainer.num_devices > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    trainer = Trainer(
        max_steps=config.trainer.max_steps,
        check_val_every_n_epoch=None,  # config.trainer.check_val_every_n_epoch,
        val_check_interval=config.trainer.val_check_interval,  # 1000,
        limit_val_batches=config.trainer.limit_val_batches,  # 20,
        log_every_n_steps=config.trainer.log_every_n_steps,  # 50,
        logger=TensorBoardLogger(".", "", ""),
        accelerator=config.trainer.device,
        devices=config.trainer.num_devices,
        strategy=strategy,
        precision=config.trainer.precision,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        accumulate_grad_batches=config.trainer.accumulate,
        gradient_clip_val=config.trainer.gradient_clip_val,
        deterministic=False,
    )
    trainer.fit(
        clm,
        ckpt_path=config.chkpt,
        train_dataloaders=pdm.train_dataloader(),
        val_dataloaders=pdm.val_dataloader(),
    )
