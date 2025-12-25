from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from star_wars_characters.data.dataset import LabelEncoder, StarWarsDataset, build_transforms


class StarWarsDataModule(L.LightningDataModule):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self._le: Optional[LabelEncoder] = None
        self._train = None
        self._val = None
        self._test = None

    @property
    def label_encoder(self) -> LabelEncoder:
        assert self._le is not None
        return self._le

    def setup(self, stage: str | None = None) -> None:
        splits_dir = Path(self.cfg.data.dataset.splits_dir)
        train_df = pd.read_parquet(splits_dir / "train.parquet")
        val_df = pd.read_parquet(splits_dir / "val.parquet")
        test_df = pd.read_parquet(splits_dir / "test.parquet")

        self._le = LabelEncoder.from_labels(train_df["label"].tolist())
        tfm_train = build_transforms(self.cfg, train=True)
        tfm_eval = build_transforms(self.cfg, train=False)

        self._train = StarWarsDataset(train_df, self._le, tfm_train)
        self._val = StarWarsDataset(val_df, self._le, tfm_eval)
        self._test = StarWarsDataset(test_df, self._le, tfm_eval)

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=False,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=True,
        )
