from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models


def build_backbone(cfg: Any, num_classes: int) -> nn.Module:
    name = str(cfg.model.name).lower()
    pretrained = bool(cfg.model.pretrained)
    dropout = float(cfg.model.dropout)

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return m

    raise ValueError(f"Unknown model.name: {cfg.model.name}")
