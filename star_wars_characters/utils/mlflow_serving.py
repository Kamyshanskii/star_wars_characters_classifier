from __future__ import annotations

import os
import subprocess
from typing import Any


def serve_mlflow_model(cfg: Any, run_id: str, port: int = 5000) -> None:
    tracking_uri = str(cfg.mlflow.tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "--env-manager",
        "local",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    env = dict(os.environ)
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    subprocess.check_call(cmd, env=env)
