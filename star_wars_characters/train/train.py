from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from star_wars_characters.data.download import download_data
from star_wars_characters.data.prepare import prepare_data
from star_wars_characters.export.export_onnx import export_onnx
from star_wars_characters.models.lit_module import StarWarsLitModule
from star_wars_characters.train.callbacks import MetricsPlotterCallback
from star_wars_characters.train.datamodule import StarWarsDataModule
from star_wars_characters.utils.dvc_utils import dvc_pull
from star_wars_characters.utils.git import get_git_commit_id


def _ensure_inputs(cfg: Any) -> None:
    if dvc_pull(["data", "artifacts"]):
        return
    raw_dir = Path(cfg.data.dataset.raw_dir)
    if not raw_dir.exists():
        download_data(cfg)
    splits_dir = Path(cfg.data.dataset.splits_dir)
    if not (splits_dir / "train.parquet").exists():
        prepare_data(cfg)


def train_model(cfg: Any) -> None:
    _ensure_inputs(cfg)
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)

    dm = StarWarsDataModule(cfg)
    dm.setup("fit")
    model = StarWarsLitModule(cfg, num_classes=len(dm.label_encoder.classes))

    callbacks = [MetricsPlotterCallback("plots")]

    if bool(cfg.train.early_stopping.enabled):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.train.early_stopping.monitor),
                mode=str(cfg.train.early_stopping.mode),
                patience=int(cfg.train.early_stopping.patience),
            )
        )

    if bool(cfg.train.checkpoint.enabled):
        callbacks.append(
            ModelCheckpoint(
                dirpath="artifacts/checkpoints",
                filename="{epoch}-{val_acc:.4f}",
                monitor=str(cfg.train.checkpoint.monitor),
                mode=str(cfg.train.checkpoint.mode),
                save_top_k=int(cfg.train.checkpoint.save_top_k),
            )
        )

    mlf = MLFlowLogger(
        experiment_name=str(cfg.mlflow.experiment_name),
        tracking_uri=str(cfg.mlflow.tracking_uri),
    )
    commit = get_git_commit_id()
    if commit:
        mlf.experiment.set_tag(mlf.run_id, "git_commit_id", commit)

    trainer = L.Trainer(
        accelerator=str(cfg.train.device),
        devices="auto",
        max_epochs=int(cfg.train.max_epochs),
        precision=str(cfg.train.precision),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        logger=mlf,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)

    for p in Path("plots").glob("*.png"):
        mlf.experiment.log_artifact(mlf.run_id, str(p), artifact_path="plots")

    if bool(cfg.export.export_onnx_on_fit_end):
        export_onnx(cfg, label_encoder=dm.label_encoder, mlflow_logger=mlf)

    print(f"[train] MLflow run_id: {mlf.run_id}")
