# Hotdog / Not Hotdog Classifier

A tiny "Hotdog / Not Hotdog" classifier inspired by *that* famous Silicon Valley scene.

- 📸 iPhone-friendly web UI (uses camera/file input)
- ⚡ FastAPI backend
- 🧠 PyTorch model trained **from scratch** (no transfer learning)
- ✅ Returns: HOTDOG ✅ or NOT HOTDOG ❌

This was a learning project — the goal was to see how far a from-scratch CNN could be pushed on a small dataset through iterative tweaking, without reaching for pretrained weights. It was a fun exercise in understanding what actually moves the needle (and what doesn't) when you're data-constrained.

## Install

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies (CUDA build)

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
pip install fastapi uvicorn[standard] python-multipart pillow
```

GPU sanity check:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Dataset

Trained on the [Hotdog - Not Hotdog](https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog/data) Kaggle dataset.

- Training: 2×2121 images (hotdog / not hotdog)
- Validation: 2×200 images

## Architecture

`ResNN` — a 4-block residual CNN with stride-2 downsampling, batch normalization, and a global average pooling head. Defined in `cnnmodels.py` and shared between training and inference.

## Training

```bash
python train.py
```

Key settings for the best run: image size 192, batch size 128, 80 epochs, AdamW with lr=1e-4 and weight_decay=2e-4, cosine annealing schedule, and RandomErasing augmentation.

## Inference

```bash
uvicorn app:app --reload
```

FastAPI app serving predictions at `POST /predict`. Upload a JPEG/PNG/WebP image and get back a classification with confidence score.

## Experiments

Each row is a change from the previous run. Not every change helped — some were rolled back.

| Val Acc | Change |
|---------|--------|
| 60% | SmallCNN, 25 epochs, lr 4e-3, ReduceLROnPlateau |
| 70% | Changed scheduler to CosineAnnealingLR, added wandb |
| 68% | Removed pos_weight from BCEWithLogitsLoss (hurt — was masking a real issue) |
| 75% | 80 epochs, lr 1e-3, image size 192 |
| 77% | BetterCNN, AdamW, weight_decay=2e-4 |
| 77% | Removed cuda sync, removed rotation, added random jitter, corrected scheduler |
| 78% | Added residual skip connections (ResNN) |
| **82%** | **Added RandomErasing augmentation** |

Things that were tried and didn't help: Mixup (underfitting), heavier augmentation (underfitting), 5th residual block (overfitting), SE channel attention (overfitting), larger image size 224/256 (overfitting), test-time augmentation (no gain).

The main lesson: with only ~4k training images and no pretrained features, the model sits on a knife edge between underfitting and overfitting. Most changes push it one way or the other without improving the balance.

## Future Ideas

- **Ensemble** — train 3 models with different random seeds and average predictions; most reliable way to gain 1-3%
- **Offline augmentation** — generate 5-10× copies of the training set with heavy transforms saved to disk, giving the model a genuinely larger dataset per epoch
- **More training data** — even an extra 2k images per class from other sources would likely push past 85%
- **Knowledge distillation** — train a pretrained model (not cheating if the small model is the final product), then use its soft labels to supervise the from-scratch CNN
- **Progressive resizing** — train at 128px first, then fine-tune at 192px; can help learn coarse features faster
- **Stochastic depth** — randomly drop entire residual blocks during training as regularization

## Files

- `cnnmodels.py` — model definitions (SmallCNN, BetterCNN, ResNN)
- `train.py` — training script with wandb logging
- `app.py` — FastAPI inference server
- `model/model.pt` — saved weights
- `model/class_names.json` — class label mapping