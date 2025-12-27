from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any


def _has_kaggle_creds() -> bool:
    import os

    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_data(cfg: Any) -> None:
    raw_dir = Path(cfg.data.dataset.raw_dir)
    if raw_dir.exists() and any(raw_dir.rglob("*.jpg")):
        print(f"[download_data] Raw dir already exists: {raw_dir}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    if not _has_kaggle_creds():
        raise RuntimeError(
            "Kaggle credentials not found. Put kaggle.json into ~/.kaggle/ "
            "or export KAGGLE_USERNAME and KAGGLE_KEY."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset = str(cfg.data.dataset.kaggle_dataset)
    tmp = Path("data/tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"[download_data] Downloading: {dataset}")
    api.dataset_download_files(dataset, path=str(tmp), unzip=False, quiet=False)

    zips = list(tmp.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"Expected a zip in {tmp}, but found none.")
    zip_path = zips[0]

    print(f"[download_data] Extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    try:
        zip_path.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    print("[download_data] Done.")
