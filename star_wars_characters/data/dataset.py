from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from star_wars_characters.utils.images import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class LabelEncoder:
    classes: List[str]

    @classmethod
    def from_labels(cls, labels: List[str]) -> "LabelEncoder":
        return cls(classes=sorted(set(labels)))

    def encode(self, label: str) -> int:
        return self.classes.index(label)

    def decode(self, idx: int) -> str:
        return self.classes[idx]


def build_transforms(cfg: Any, train: bool) -> transforms.Compose:
    image_size = int(cfg.data.image_size)
    t = [transforms.Resize((image_size, image_size))]

    if train and bool(cfg.train.augment.enabled):
        t.append(transforms.RandomHorizontalFlip(p=float(cfg.train.augment.hflip_p)))
        cj = float(cfg.train.augment.color_jitter)
        t.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=0.02))

    t.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(t)


class StarWarsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder, tfm: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        label = str(row["label"])
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        y = self.le.encode(label)
        return x, torch.tensor(y, dtype=torch.long)
