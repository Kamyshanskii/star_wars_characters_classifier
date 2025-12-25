from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_pil(img: Image.Image, image_size: int) -> np.ndarray:
    img = img.resize((image_size, image_size))
    x = np.asarray(img).astype("float32") / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype="float32")) / np.array(IMAGENET_STD, dtype="float32")
    x = np.transpose(x, (2, 0, 1))
    return x.astype("float32")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
