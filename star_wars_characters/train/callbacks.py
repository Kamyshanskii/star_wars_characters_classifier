from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import lightning as L
import matplotlib.pyplot as plt


class MetricsPlotterCallback(L.Callback):
    def __init__(self, plots_dir: str = "plots"):
        self.plots_dir = Path(plots_dir)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        for k in ["train_loss", "val_loss", "val_acc"]:
            v = metrics.get(k)
            if v is not None:
                val = float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v)
                self.history[k].append(val)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        for k, values in self.history.items():
            if not values:
                continue
            plt.figure()
            plt.plot(range(1, len(values) + 1), values)
            plt.xlabel("epoch")
            plt.ylabel(k)
            plt.title(k)
            out = self.plots_dir / f"{k}.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
