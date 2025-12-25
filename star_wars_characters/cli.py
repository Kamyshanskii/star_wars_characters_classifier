from __future__ import annotations

from typing import Any, Optional

import fire
from hydra import compose, initialize

from star_wars_characters.data.download import download_data
from star_wars_characters.data.prepare import prepare_data
from star_wars_characters.export.export_onnx import export_onnx
from star_wars_characters.infer.predict import infer_one
from star_wars_characters.train.train import train_model
from star_wars_characters.utils.mlflow_serving import serve_mlflow_model


def _load_cfg(overrides: Optional[list[str]] = None) -> Any:
    overrides = overrides or []
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


class Commands:
    def download_data(self) -> None:
        download_data(_load_cfg())

    def prepare_data(self) -> None:
        prepare_data(_load_cfg())

    def train(self, *overrides: str) -> None:
        train_model(_load_cfg(list(overrides)))

    def export_onnx(self, *overrides: str) -> None:
        export_onnx(_load_cfg(list(overrides)))

    def infer(self, image_path: str, *overrides: str) -> None:
        infer_one(_load_cfg(list(overrides)), image_path)

    def serve_mlflow(self, run_id: str, port: int = 5000) -> None:
        serve_mlflow_model(_load_cfg(), run_id=run_id, port=port)


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
