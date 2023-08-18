import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .config import ProteoscopeConfig
from .data import ProteoscopeDM
from .modules import ProteoscopeLM


def train_proteoscope(config: ProteoscopeConfig) -> None:
    pdm = ProteoscopeDM(
        images_path=config.data.images_path,
        labels_path=config.data.labels_path,
        trim=config.data.trim,
        sequences_path=config.data.sequences_path,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        sequence_embedding=config.data.sequence_embedding,
    )
    pdm.setup()

    clm = ProteoscopeLM(
        module_config=config.module,
    )
    print(clm)
    print(
        f"Train samples {len(pdm.train_dataset)}, Val images {len(pdm.val_images_dataset)},  Val proteins {len(pdm.val_proteins_dataset)}"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="val_loss", mode="min", save_last=True
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
        deterministic=True,
    )
    trainer.fit(
        clm,
        ckpt_path=config.chkpt,
        train_dataloaders=pdm.train_dataloader(),
        val_dataloaders=pdm.val_dataloader(novel_proteins=True),
    )
