from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import onnxruntime as ort

from star_wars_characters.utils.dvc_utils import dvc_pull
from star_wars_characters.utils.images import load_image_rgb, preprocess_pil, softmax


def _load_labels(raw_dir: Path) -> List[str]:
    classes = sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders in {raw_dir}")
    return classes


def predict(cfg: Any, image_path: str) -> Tuple[str, List[Tuple[str, float]]]:
    dvc_pull(["artifacts", "data"])

    onnx_path = Path(cfg.infer.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}. Run training/export first.")

    classes = _load_labels(Path(cfg.data.dataset.raw_dir))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    img = load_image_rgb(image_path)
    x = preprocess_pil(img, int(cfg.data.image_size))
    x = np.expand_dims(x, 0)

    logits = sess.run(["logits"], {"input": x})[0]
    probs = softmax(logits, axis=1)[0]
    idxs = np.argsort(-probs)[: int(cfg.infer.top_k)]
    top = [(classes[int(i)], float(probs[int(i)])) for i in idxs]
    return top[0][0], top


def infer_one(cfg: Any, image_path: str) -> None:
    pred, top = predict(cfg, image_path=image_path)
    print(f"pred: {pred}")
    for lbl, p in top:
        print(f"  {lbl}: {p:.4f}")
