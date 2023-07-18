from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import zarr
from anndata import read_h5ad  # type: ignore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import ProteoscopeDataset


class ProteoscopeDataModule(LightningDataModule):
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        batch_size: int,
        num_workers: int,
        sequences_path: Optional[str] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels_path = labels_path
        self.sequences_path = sequences_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.images = zarr.open(self.images_path, mode="r")
        self.labels = pd.read_csv(self.labels_path, index_col=0)
        self.labels = self.labels.fillna('')
        # self.labels['full_index'] = range(len(self.labels))

        if self.sequences_path is not None:
            self.sequences = zarr.open(self.sequences_path, 'r')
        else:
            self.sequences = None

        self.train_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein="train",
            split_images="train",
        )

        self.val_images_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein="train",
            split_images="val",
            transform=None,
        )

        self.val_proteins_dataset = ProteoscopeDataset(
            images=self.images,
            labels=self.labels,
            sequences=self.sequences,
            split_protein="val",
            split_images="",
            transform=None,
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

    def val_dataloader(self, novel_proteins=False, shuffle=False):
        if novel_proteins:
            dataset = self.val_proteins_dataset
        else:
            dataset = self.val_images_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage=None):
        pass
