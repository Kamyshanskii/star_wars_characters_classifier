# Star Wars Character Classifier

Image classification project: **predict a Star Wars character** from an input image.

This repository is structured to satisfy the MLOps course homework requirements:
- dependencies via **Poetry**
- code quality via **pre-commit** (pre-commit-hooks, black, isort, flake8, prettier)
- configs via **Hydra**
- experiments logging via **MLflow**
- data and artifacts via **DVC**
- training via **PyTorch Lightning**
- production packaging via **ONNX**
- inference server via **MLflow Serving** (pyfunc wrapper around ONNXRuntime)

Dataset (Kaggle): `mathurinache/star-wars-images`.

---

## Setup

### 1) Requirements
- Python 3.10+ (recommended 3.11)
- `git`, `dvc`
- `docker` (optional, for local MLflow server)

### 2) Install dependencies
```bash
poetry lock
poetry install --with train,dev
```

### 3) Enable git hooks
```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

### 4) MLflow (optional)
Homework assumes MLflow at `http://127.0.0.1:8080`.

Run locally:
```bash
docker compose up -d
```

---

## Data management (DVC)

This project uses DVC to store:
- `data/` (raw images, fixed splits, examples)
- `artifacts/` (checkpoints, ONNX)

### Quick start (local DVC remote)
```bash
dvc init
dvc remote add -d local_remote dvc_remote
```

### Download dataset (Kaggle)
Put Kaggle creds (`~/.kaggle/kaggle.json`) or set env vars `KAGGLE_USERNAME`, `KAGGLE_KEY`.

```bash
poetry run swc download_data
```

### Prepare fixed splits + export examples
```bash
poetry run swc prepare_data
```

Track and push data/artifacts to your DVC remote:
```bash
dvc add data
dvc push
```

---

## Train

One command (tries `dvc pull` first; if missing, downloads from Kaggle and prepares splits):
```bash
poetry run swc train
```

Override configs via Hydra:
```bash
poetry run swc train train.batch_size=8 train.max_epochs=5 data.image_size=224
```

Outputs:
- MLflow run (metrics + params + artifacts)
- plots in `plots/` (**at least 3 curves**)
- checkpoints in `artifacts/checkpoints/`
- ONNX in `artifacts/model.onnx` (if enabled in config)

---

## Production preparation

Export to ONNX (separate command):
```bash
poetry run swc export_onnx
```

Optional TensorRT conversion stub:
```bash
bash scripts/export_tensorrt.sh
```

---

## Infer

Inference uses **ONNXRuntime** (minimal deps).

Predict one image:
```bash
poetry run swc infer path/to/image.jpg
```

---

## Inference server (MLflow Serving, max 5 points)

Find `RUN_ID` in MLflow UI and run:
```bash
poetry run swc serve_mlflow <RUN_ID> 5000
```

Example request:
```bash
curl -X POST -H "Content-Type: application/json" \
  --data '{"instances":[{"image_path":"data/examples/example_1_*.jpg"}]}' \
  http://127.0.0.1:5000/invocations
```
