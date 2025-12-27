# Star Wars character classifier

**Камышанский Андрей**

## Постановка задачи

Задача — по изображению определить персонажа из вселенной Звездных Войн.

В качестве источника данных используется датасет с [Kaggle](https://www.kaggle.com/datasets/mathurinache/star-wars-images/data).

### Формат входных и выходных данных

**Вход:** изображение (`.jpg/.jpeg/.png`).
**Выход:** имя класса (персонажа) и top-k вероятностей.

### Метрики

- **Accuracy** на train/val/test
- **Cross-Entropy Loss** на train/val

Ориентир по качеству: `val_acc` ≈ **0.75+** на baseline (ResNet18).

### Валидация

Данные разбиваются на `train/val/test` в соотношении 80/10/10 с seed = 42.

### Данные

Датасет: [star-wars-images](https://www.kaggle.com/datasets/mathurinache/star-wars-images/data).
После скачивания и подготовки структура такая:

- `data/raw/star_wars_images/<class_name>/*.jpg`
- `data/splits/train.parquet`, `val.parquet`, `test.parquet`
- `data/examples/` — несколько изображений для быстрого инференса

---

## Моделирование

### Основная модель

- Архитектура: **ResNet18**
- Обучение: fine-tune
- Фреймворк: **PyTorch Lightning**

---

## Setup

Разработка и тестирование проводились на Linux с GPU **GTX 1650**.
Проект использует **Poetry**. Требуемая версия Python: **>=3.10,<3.13**.

### 1) Клонировать репозиторий

```bash
git clone https://github.com/Kamyshanskii/star_wars_characters_classifier.git
cd star_wars_characters
```

### 2) Python

```bash
conda create -n swc_py312 python=3.12 -y
conda activate swc_py312
```

### 3) Установить зависимости

```bash
poetry install --with train,dev
```

Проверка:

```bash
poetry run swc --help
```

### 4) Kaggle credentials (для скачивания датасета)

Нужно положить `kaggle.json` в `~/.kaggle/`:

```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

`kaggle.json` создаётся на странице Kaggle → Settings → API → **Create Legacy API Key**.

---

## Train

### 1) Скачать данные и подготовить сплиты

```bash
poetry run swc download_data
poetry run swc prepare_data
```

### 2) Запуск обучения

```bash
poetry run swc train mlflow.tracking_uri="file:./mlruns" train.device=gpu train.batch_size=8 train.max_epochs=10
```

### 3) Результаты обучения

- чекпоинты: `artifacts/checkpoints/*.ckpt`
- графики: `plots/train_loss.png`, `plots/val_loss.png`, `plots/val_acc.png`
- ONNX: `artifacts/model.onnx` (экспортируется автоматически из **лучшего checkpoint**)

---

## Метрики и MLflow

### Консоль

PyTorch Lightning печатает метрики по эпохам прямо в терминал (`train_loss`, `train_acc`, `val_loss`, `val_acc`).

### Графики

Генерируются в `plots/` автоматически callback’ом.

### MLflow UI (локально)

```bash
poetry run mlflow ui --backend-store-uri "$(pwd)/mlruns" --port 8081
```

Открыть: `http://127.0.0.1:8081`

---

## Production preparation

### ONNX

ONNX экспортируется после обучения в:

- `artifacts/model.onnx`

Проверка:

```bash
poetry run python -c "import onnx; m=onnx.load('artifacts/model.onnx'); print('OK', m.graph.output[0].name)"
```

### TensorRT (скрипт-заготовка)

Есть скрипт:

- `scripts/export_tensorrt.sh`

(Требует установленного TensorRT/драйверов. В учебном проекте это “заготовка” под прод-пайплайн.)

---

## Infer

### Инференс на одном изображении

```bash
poetry run swc infer data/examples/example_1_R2-D2.jpg
```

Пример вывода:

```
pred: R2-D2
  R2-D2: 0.9998
  BB-8: 0.0001
  ...
```

# Ссылки

- Датасет:
  - https://www.kaggle.com/datasets/mathurinache/star-wars-images/data
- ResNet:
  - https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
