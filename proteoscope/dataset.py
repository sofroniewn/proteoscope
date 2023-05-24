from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class ProteoscopeDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        split_protein: str,
        split_images: str = "",
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

        self.images = images
        self.transform = transforms.Compose(
            [torch.from_numpy, lambda x: torch.permute(x, [2, 0, 1])]
            + ([] if transform is None else list(transform))
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int, str]]:
        row = self.labels.iloc[idx]
        index = row.name
        images = self.images[index, :, :, :2]
        images = self.transform(images)

        item = dict()
        item["index"] = index
        item["ensg"] = row.ensg
        item["name"] = row["name"]
        item["loc_grade1"] = row.loc_grade1
        item["loc_grade2"] = row.loc_grade2
        item["loc_grade3"] = row.loc_grade3
        item["protein_id"] = row.protein_id
        item["FOV_id"] = row.FOV_id
        item["label"] = row.label
        item["image"] = images.float()
        return item
