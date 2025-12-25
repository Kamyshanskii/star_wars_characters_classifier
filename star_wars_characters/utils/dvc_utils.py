from __future__ import annotations

import subprocess
from typing import Optional


def dvc_pull(paths: Optional[list[str]] = None) -> bool:
    cmd = ["dvc", "pull"]
    if paths:
        cmd.extend(paths)
    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False
