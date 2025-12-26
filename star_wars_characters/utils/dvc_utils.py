from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def _has_dvc_config() -> bool:
    """Return True if the workspace is configured to use DVC."""
    return Path("dvc.yaml").exists() or Path("dvc.lock").exists() or any(Path(".").glob("*.dvc"))


def dvc_pull(paths: Optional[list[str]] = None) -> bool:
    """Try to `dvc pull`.

    This project is meant to run without DVC as well. When there is no DVC
    pipeline/tracking configuration in the repository, we skip silently to
    avoid noisy warnings (e.g. about missing dvc.yaml).
    """
    if not _has_dvc_config():
        return False

    cmd = ["dvc", "pull"]
    if paths:
        cmd.extend(paths)

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
