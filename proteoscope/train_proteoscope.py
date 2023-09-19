import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .config import ProteoscopeConfig
from .data import ProteoscopeDM
from .modules import ProteoscopeLM, ProteoclipLM


def train_proteoscope(config: ProteoscopeConfig) -> None:
    pdm = ProteoscopeDM(
        images_path=config.data.images_path,
        labels_path=config.data.labels_path,
        trim=config.data.trim,
        sequences_path=config.data.sequences_path,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        sequence_embedding=config.data.sequence_embedding,
        splits=config.splits,
        sequence_dropout=config.data.sequence_dropout
    )
    pdm.setup()

    if config.model_type == "proteoscope":
        clm = ProteoscopeLM(module_config=config.module)
        dls = [pdm.val_dataloader(), pdm.train_dataloader()]
    elif config.model_type == "proteoclip":
        clm = ProteoclipLM(module_config=config.module)
        dls = pdm.val_dataloader()
    else:
        raise ValueError(f"Unrecognized model type {config.model_type}")

    print(clm)
    print(f"Train samples {len(pdm.train_dataset)}, Val samples {len(pdm.val_dataset)}")

    if config.reset:
        checkpoint = torch.load(config.chkpt)
        clm.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint
        ckpt_path = None
    else:
        ckpt_path = config.chkpt

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
        ckpt_path=ckpt_path,
        train_dataloaders=pdm.train_dataloader(),
        val_dataloaders=dls,
    )
