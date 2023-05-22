from typing import Dict, Union, Optional, Sequence
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
import torch


class ProteoscopeDataset(Dataset):
    def __init__(self,
                 images,
                 labels,
                 split_protein: str,
                 split_images: str = '',
                 transform: Optional[Sequence] = (
                    transforms.RandomApply(
                        [
                            lambda x: transforms.functional.rotate(x, 0),
                            lambda x: transforms.functional.rotate(x, 90),
                            lambda x: transforms.functional.rotate(x, 180),
                            lambda x: transforms.functional.rotate(x, 270),
                        ]
                    ),
                    # transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                ),
                 ) -> None:
        super(Dataset, self).__init__()

        self.split_protein = split_protein
        self.split_images = split_images
        self.labels = labels[labels['split_protein'] == self.split_protein and labels['split_images'] == self.split_images]
        self.images = self.images[self.labels.index.values]
        self.transform = transforms.Compose([torch.from_numpy] + ([] if transform is None else list(transform)))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int, str]]:
        row = self.labels[idx]
        images = self.images[idx]

        if self.transform is not None:
            images = self.transform(images)

        item = dict()
        item["index"] = row.index.value
        item["ensg"] = row.ensg
        item["name"] = row.name
        item["loc_grade1"] = row.loc_grade1
        item["loc_grade2"] = row.loc_grade2
        item["loc_grade3"] = row.loc_grade3
        item["protein_id"] = row.protein_id
        item["FOV_id"] = row.FOV_id
        item["prot"] = images[:, :, 0]
        item["nuc"] = images[:, :, 1]
        return item