from typing import Dict, Optional, Sequence, Union

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class ProteoscopeDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        split_protein: str,
        trim: Optional[int] = None,
        split_images: str = "",
        sequences = None,
        sequence_index = None,
        unique_protein = False,
        sequence_embedding = None,
        transform: Optional[Sequence] = (
            transforms.RandomApply(
                [
                    lambda x: transforms.functional.rotate(x, 0),
                    lambda x: transforms.functional.rotate(x, 90),
                    lambda x: transforms.functional.rotate(x, 180),
                    lambda x: transforms.functional.rotate(x, 270),
                ]
            ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ),
    ) -> None:
        super(Dataset, self).__init__()

        self.split_protein = split_protein
        self.split_images = split_images
        self.labels = labels[
            (labels["split_protein"] == self.split_protein)
            & (labels["split_images"] == self.split_images)
        ]
        if unique_protein:
            self.labels = self.labels.drop_duplicates(subset='ensg')

        self.num_label_class = len(self.labels['label'].unique())

        self.images = images
        self.trim = trim
        self.transform = transforms.Compose(
            [torch.from_numpy, lambda x: torch.permute(x, [2, 0, 1])]
            + ([] if transform is None else list(transform))
        )

        self.sequences = sequences
        self.sequence_index = sequence_index
        self.sequence_embedding = sequence_embedding

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int, str]]:
        row = self.labels.iloc[idx]
        index = row.name
        images = self.images[index, :, :, :2]
        if self.trim is not None:
            images = images[self.trim:-self.trim, self.trim:-self.trim]
        images = self.transform(images)

        item = dict()
        item["index"] = index
        item["ensg"] = row.ensg
        item["name"] = row["name"]
        item["loc_grade1"] = row.loc_grade1
        item["loc_grade2"] = row.loc_grade2
        item["loc_grade3"] = row.loc_grade3
        item["protein_id"] = row.protein_id
        item["peptide"] = row['Peptide']
        item["ensp"] = row['Protein stable ID']
        item["FOV_id"] = row.FOV_id
        item["seq_embedding_index"] = row['seq_embedding_index']
        item["truncation"] = row['truncation']
        item["label"] = row.label
        item["image"] = images.float()
        item["localization"] = row['localization']
        item["complex"] = row['complex']
        item["complex_fig"] = row['complex_fig']

        if self.sequences is not None and self.sequence_embedding is not None:
            if self.sequence_index is not None:
                sequence_embed = self.sequences[item["seq_embedding_index"], self.sequence_index]
                sequence_mask = None
            else:
                sequence_embed = self.sequences[item["seq_embedding_index"], 1:]
                sequence_mask = torch.zeros(len(sequence_embed), dtype=torch.bool)
                sequence_mask[:row['truncation']] = True

            if self.sequence_embedding == 'ESM-mean':
                item['sequence_embed'] = sequence_embed.mean(axis=0)[None, ...]
                item['sequence_mask'] = torch.ones(1, dtype=torch.bool)
            elif self.sequence_embedding == 'one-hot':
                item['sequence_embed'] = torch.zeros((1, 1280))
                item['sequence_embed'][0, item["label"]] = 1.0
                item['sequence_mask'] = torch.ones(1, dtype=torch.bool)
            elif self.sequence_embedding == 'random':
                item['sequence_embed'] = torch.randn((1, 1280))
                item['sequence_mask'] = torch.ones(1, dtype=torch.bool)
            elif self.sequence_embedding == 'ESM-full':
                item['sequence_embed'] = sequence_embed
                item['sequence_mask'] = sequence_mask
        return item
