from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from star_wars_characters.utils.seed import seed_everything

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _scan_dataset(raw_dir: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for class_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((str(p), label))
    if not items:
        raise RuntimeError(f"No images found in {raw_dir}")
    return items


def _split(items: List[Tuple[str, str]], ratios: Dict[str, float], seed: int):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def prepare_data(cfg: Any) -> None:
    seed = int(cfg.data.seed)
    seed_everything(seed)

    raw_dir = Path(cfg.data.dataset.raw_dir)
    splits_dir = Path(cfg.data.dataset.splits_dir)
    examples_dir = Path(cfg.data.dataset.examples_dir)

    splits_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    items = _scan_dataset(raw_dir)
    ratios = dict(cfg.data.splits)
    train, val, test = _split(items, ratios, seed)

    for name, split_items in [("train", train), ("val", val), ("test", test)]:
        df = pd.DataFrame(split_items, columns=["path", "label"])
        out = splits_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"[prepare_data] Wrote {out} ({len(df)} rows)")

    for i, (p, label) in enumerate(train[:3]):
        src = Path(p)
        dst = examples_dir / f"example_{i+1}_{label}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"[prepare_data] Exported examples into {examples_dir}")
