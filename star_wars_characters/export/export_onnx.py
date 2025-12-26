from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from star_wars_characters.models.lit_module import StarWarsLitModule


def export_onnx(
    cfg: Any,
    checkpoint_path: Optional[str] = None,
    label_encoder=None,
    mlflow_logger=None,
) -> None:
    onnx_path = Path(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(label_encoder.classes) if label_encoder is not None else 2
    lit = StarWarsLitModule(cfg, num_classes=num_classes)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        lit.load_state_dict(ckpt["state_dict"], strict=False)

    lit.eval()
    model = lit.model

    image_size = int(cfg.data.image_size)
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(cfg.export.opset),
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"[export_onnx] Saved: {onnx_path}")

    if mlflow_logger is not None:
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, str(onnx_path), artifact_path="artifacts"
        )
