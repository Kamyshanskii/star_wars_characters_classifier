from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from star_wars_characters.models.backbone import build_backbone


class StarWarsLitModule(L.LightningModule):
    def __init__(self, cfg: Any, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])
        self.model = build_backbone(cfg, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )
