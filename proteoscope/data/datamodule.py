from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import zarr
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import ProteoscopeDataset, ProteolocDataset


class ProteoscopeDM(LightningDataModule):
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        batch_size: int,
        num_workers: int,
        splits,
        sequences_path: Optional[str] = None,
        trim: Optional[int] = None,
        sequence_embedding: Optional[str] = None,
        sequence_dropout: Optional[float] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels_path = labels_path
        self.sequences_path = sequences_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trim = trim
        self.sequence_embedding = sequence_embedding
        self.splits = splits
        self.sequence_dropout = sequence_dropout

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.images = zarr.open(self.images_path, mode="r")
        self.labels = pd.read_csv(self.labels_path, index_col=0)
        self.labels = self.labels.fillna("other")

        if self.sequences_path is not None:
            self.sequences = zarr.open(self.sequences_path, "r")
        else:
            self.sequences = None

        self.train_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein=self.splits.train_protein,
            split_images=self.splits.train_images,
            trim=self.trim,
            sequence_embedding=self.sequence_embedding,
            sequence_dropout=self.sequence_dropout,
        )

        self.val_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein=self.splits.val_protein,
            split_images=self.splits.val_images,
            transform=None,
            trim=self.trim,
            shuffle=42,  # Seed value for shuffle
            sequence_embedding=self.sequence_embedding,
        )

        self.val_train_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein=self.splits.train_protein,
            split_images=self.splits.train_images,
            transform=None,
            trim=self.trim,
            shuffle=42,  # Seed value for shuffle
            sequence_embedding=self.sequence_embedding,
        )

        self.predict_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein=None,
            split_images=None,
            transform=None,
            trim=self.trim,
            sequence_embedding=self.sequence_embedding,
        )

        self.num_class = self.train_dataset.num_label_class

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_train_dataloader(self):
        return DataLoader(
            self.val_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def custom_dataloader(
        self,
        split_protein=None,
        split_images=None,
        unique_protein=False,
        shuffle=None,
        batch_size=None,
        downsample=None
    ):
        dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein=split_protein,
            split_images=split_images,
            unique_protein=unique_protein,
            transform=None,
            shuffle=shuffle,
            trim=self.trim,
            sequence_embedding=self.sequence_embedding,
            downsample=downsample,
        )
        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage=None):
        pass


class ProteolocDM(LightningDataModule):
    def __init__(
        self,
        labels_path: str,
        batch_size: int,
        num_workers: int,
        sequences_path: Optional[str] = None,
        sequence_embedding: Optional[str] = None,
        sequence_dropout: Optional[float] = None,
    ):
        super().__init__()

        self.labels_path = labels_path
        self.sequences_path = sequences_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_embedding = sequence_embedding
        self.sequence_dropout = sequence_dropout

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.labels = pd.read_csv(self.labels_path, index_col=0)

        if self.sequences_path is not None:
            self.sequences = zarr.open(self.sequences_path, "r")
        else:
            self.sequences = None

        self.train_dataset = ProteolocDataset(
            labels=self.labels,
            sequences=self.sequences,
            split_protein='train',
            sequence_embedding=self.sequence_embedding,
            sequence_dropout=self.sequence_dropout,
        )

        self.val_dataset = ProteolocDataset(
            labels=self.labels,
            sequences=self.sequences,
            split_protein='val',
            shuffle=42,  # Seed value for shuffle
            sequence_embedding=self.sequence_embedding,
        )

        self.test_dataset = ProteolocDataset(
            labels=self.labels,
            sequences=self.sequences,
            split_protein='test',
            shuffle=42,  # Seed value for shuffle
            sequence_embedding=self.sequence_embedding,
        )

        self.predict_dataset = ProteolocDataset(
            labels=self.labels,
            sequences=self.sequences,
            split_protein=None,
            sequence_embedding=self.sequence_embedding,
        )

        self.num_class = self.train_dataset.num_label_class

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def custom_dataloader(
        self,
        split_protein=None,
        unique_protein=False,
        shuffle=None,
        batch_size=None,
        downsample=None
    ):
        dataset = ProteolocDataset(
            labels=self.labels,
            sequences=self.sequences,
            split_protein=split_protein,
            unique_protein=unique_protein,
            shuffle=shuffle,
            sequence_embedding=self.sequence_embedding,
            downsample=downsample,
        )
        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage=None):
        pass
