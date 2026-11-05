# exercise-02 — CNN Image Classifier · SOLUTION

Paired exercise: `lessons/mod-105-deep-learning-fundamentals/exercises/exercise-02-cnn-image-classifier.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise trains two CIFAR-10 classifiers and compares them head-to-head:

1. A **small from-scratch CNN** (~0.5–1M parameters) with `Conv2d` + `BatchNorm2d` + residual blocks.
2. A **fine-tuned pre-trained backbone** (ResNet-18 from `torchvision`) with its ImageNet head replaced for 10 classes, trained in both "frozen backbone" and "full fine-tune" modes.

The exercise-01 training loop is reused unchanged — what changes is the data pipeline (image augmentation), the optimiser schedule (`OneCycleLR`), and the perf tricks from chapter 09 (mixed precision + gradient clipping + label smoothing).

The reference solution is structured around three invariants:

1. **Train vs. eval transforms are distinguishable.** A shared normalisation, two different augmentation pipelines. A test proves it — because a reviewer cannot eyeball augmentation leakage, and augmenting the val set is a silent accuracy killer.
2. **The fine-tune uses the backbone's own preprocessing.** Each pre-trained `torchvision` weight bundle exposes `weights.transforms()`; using CIFAR's `mean/std` with ImageNet weights degrades accuracy by several points for no visible reason. The exercise flags this; the solution pins it with a test.
3. **The augmentation ablation is a controlled experiment.** Same seed, same architecture, same schedule — only the train transform changes. The three numbers are the answer; refuse the temptation to tune.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| Small CNN: 3-stage ResNet-ish with 2 residual blocks per stage | VGG-style plain conv stack | Residual blocks train more reliably from scratch at this depth (He et al. 2015); parameter budget still fits the 200k–2M range |
| ResNet-18 for the fine-tune | MobileNet-V3, EfficientNet-B0 | Canonical baseline; its `torchvision` weights expose a clean `weights.transforms()` API and are small enough to fine-tune on CPU |
| `OneCycleLR(pct_start=0.1)` | StepLR, cosine without warmup | Fewer hyperparameters to pick; the warmup phase hides initial transient; `max_lr` is the single knob |
| `AdamW(lr=3e-3, wd=5e-4)` for small CNN; `AdamW(lr=1e-3, wd=1e-4)` for full fine-tune; `lr=3e-3` for head-only | SGD+momentum everywhere | Adam variants converge faster at this scale; SGD edges out at very long schedules, outside this exercise's time budget |
| Label smoothing 0.1 in `CrossEntropyLoss` | Plain CE | Small but consistent improvement on CIFAR-10 (Müller et al. 2019); native in PyTorch ≥ 1.10 |
| `bf16` autocast on CUDA, no `GradScaler` | `fp16` + `GradScaler` | On Ampere+ GPUs `bf16` has the fp32 exponent range so no scaler needed; the code is simpler. Falls back to no-op context on CPU |
| `drop_last=True` on train, `False` on val | `False` everywhere | Keeps BatchNorm statistics stable at the training edge; val needs every image counted |

The solution is written to reach the exercise's acceptance floors: **≥ 85% test accuracy** for the small CNN with full augmentation, **≥ 90%** for the fully fine-tuned ResNet-18.

## 2. Worked implementation

### 2.1 Project layout

```
exercise-02/
├── data.py                        # transforms + loaders, CIFAR and pretrained variants
├── models/
│   ├── __init__.py
│   ├── small_cnn.py               # the from-scratch model
│   └── pretrained.py              # ResNet-18 head swap + freeze helper
├── train.py                       # extends exercise-01's loop: AMP, OneCycleLR, clip, smoothing
├── evaluate.py                    # overall, per-class, confusion, samples
├── tests/
│   ├── test_shapes.py
│   ├── test_augmentation.py
│   ├── test_normalisation.py
│   ├── test_frozen.py
│   └── test_overfit.py
├── plots/                         # loss curves, confusion matrices, sample grids
├── history_small_cnn.json
├── history_pretrained.json
├── pyproject.toml
└── REPORT.md
```

### 2.2 `data.py` — two transforms, one eval, worker-safe loaders

```python
# data.py
from typing import Callable
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2 as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool, aug_level: str = "standard") -> Callable:
    """aug_level in {"none", "standard", "strong"} — used by the Part F ablation."""
    normalize = T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    if not train or aug_level == "none":
        return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), normalize])

    train_ops: list = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if aug_level == "strong":
        train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
    train_ops += [T.ToImage(), T.ToDtype(torch.float32, scale=True), normalize]
    return T.Compose(train_ops)


def build_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    aug_level: str = "standard",
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CIFAR10(root, train=True, download=True,
                      transform=build_transforms(train=True, aug_level=aug_level))
    val_ds = CIFAR10(root, train=False, download=True,
                    transform=build_transforms(train=False))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
```

Notes on each decision:

- **`torchvision.transforms.v2`** — the modern API that handles `tv_tensors` natively and composes cleanly with `MixUp` or `CutMix` in the stretch goals.
- **`RandomCrop(32, padding=4)` then `RandomHorizontalFlip`** — the "standard CIFAR-10 augmentation" baseline used in nearly every reproduction paper since the original ResNet work. The exercise prescribes this verbatim.
- **Val transform contains no randomness.** Without this the val metric wobbles across epochs for reasons unrelated to training; a test verifies two eval-mode `__getitem__` calls return identical tensors.
- **`persistent_workers=True`** when `num_workers > 0` — avoids re-spawning worker processes between epochs. On Linux with a reasonably warm page cache this is the single biggest DataLoader speedup for CIFAR.

For the **pre-trained** loader, use `torchvision`'s own preprocessing:

```python
# data.py (addition)
from torchvision.models import ResNet18_Weights


def build_pretrained_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, Callable]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    eval_tfm = weights.transforms()  # includes resize to 224, imagenet mean/std

    train_tfm = T.Compose([
        T.Resize(224, antialias=True),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ])

    train_ds = CIFAR10(root, train=True, download=True, transform=train_tfm)
    val_ds = CIFAR10(root, train=False, download=True, transform=eval_tfm)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, eval_tfm
```

### 2.3 `models/small_cnn.py` — residual CNN, ~550k params

```python
# models/small_cnn.py
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(32, 64, stride=1), ResBlock(64, 64))
        self.stage2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.stage3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Design notes:

- **6 convs + 3 shortcut convs** across three residual stages — matches the exercise's "≥ 6 conv layers / 3 stages" criterion.
- **Two spatial downsamples** via `stride=2` in stages 2 and 3; spatial 32 → 16 → 8 before the global pool.
- **`bias=False` on the convs before BatchNorm.** BN's learned affine bias makes the conv bias redundant; disabling it removes a parameter *and* a source of training instability.
- **`AdaptiveAvgPool2d((1, 1))`** before `Flatten` — makes the model agnostic to input resolution, which is why the fine-tune can take 224×224 inputs through essentially the same head pattern.
- Parameter count: the three stages contribute roughly 110k + 220k + 220k + shortcuts + stem + head ≈ **550k trainable parameters**. The exact number depends on channel counts; adjust `(32, 64, 128, 256)` down to `(32, 48, 96, 192)` if you want closer to the lower band.

### 2.4 `models/pretrained.py` — head swap + optional freeze

```python
# models/pretrained.py
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_pretrained(num_classes: int = 10, freeze_backbone: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return model
```

Why this is a four-liner and not a configuration surface:

- **Replace, don't wrap.** A custom head Sequential would require you to also delete the original `fc`. One-line replacement is the idiom; `state_dict` paths line up with the `torchvision` card.
- **Freeze logic is name-based.** `requires_grad = name.startswith("fc.")` is one scan through `named_parameters` and leaves nothing trainable in error. A reviewer can `grep "requires_grad"` and find exactly this pattern.
- **No `with torch.no_grad():` around the backbone forward.** Freezing is about optimiser updates, not about disabling gradient computation during forward — BatchNorm running stats still update unless you also call `.eval()` on the backbone, which is a different decision and should be deliberate.

### 2.5 `train.py` — the exercise-01 loop + three extensions

```python
# train.py — showing only the new bits; the four-line core is identical to exercise-01
from contextlib import nullcontext
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    device: str,
    use_amp: bool,
    clip_norm: float = 1.0,
) -> float:
    model.train()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.startswith("cuda")
        else nullcontext()
    )
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            logits = model(xb)
            loss = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()                              # OneCycleLR is per-batch

        bs = yb.shape[0]
        running += float(loss) * bs
        n += bs
    return running / max(n, 1)


def build_optim_and_sched(
    model: nn.Module, max_lr: float, weight_decay: float,
    steps_per_epoch: int, epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=max_lr,
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=0.1, anneal_strategy="cos",
    )
    return optim, sched


def default_loss() -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=0.1)
```

Three extensions, each with a one-line justification:

1. **Mixed precision via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.** `bf16` matches `fp32`'s exponent range on Ampere+ GPUs, so no `GradScaler` is needed; the code falls back to a `nullcontext` on CPU so the training path is identical structurally. On older GPUs (Pascal/Volta) swap for `torch.float16` and add a `torch.cuda.amp.GradScaler()`.
2. **`clip_grad_norm_(..., 1.0)` before `optimizer.step()`.** Belt-and-braces guard; on CIFAR-10 with Adam it rarely fires, but keeping the code path warm pays off in exercise-03's Transformer.
3. **`OneCycleLR(pct_start=0.1)` per batch.** `steps_per_epoch * epochs` is the total step budget; `scheduler.step()` lives inside the batch loop. A common bug is calling it once per epoch; the test `test_lr_ramps` (see below) pins it.

**Training recipes** (what the CLI runs):

| Model | Max LR | Weight decay | Epochs | Expected accuracy |
|---|---|---|---|---|
| Small CNN, full augmentation | `3e-3` | `5e-4` | 60 | ≥ 85% |
| ResNet-18, frozen backbone | `3e-3` | `1e-4` | 15 | ~65–75% |
| ResNet-18, full fine-tune | `1e-3` | `1e-4` | 15 | ≥ 90% |

### 2.6 `evaluate.py` — overall, per-class, confusion matrix, samples

```python
# evaluate.py
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        ys.append(yb.numpy())
        ps.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def report(y_true: np.ndarray, y_pred: np.ndarray, out_prefix: str) -> dict:
    overall = float((y_true == y_pred).mean())
    per_class = {}
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    for i, name in enumerate(CLASSES):
        total = int((y_true == i).sum())
        per_class[name] = float(cm[i, i] / total) if total else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(f"plots/confusion_{out_prefix}.png", dpi=120)
    plt.close(fig)

    return {"overall": overall, "per_class": per_class, "confusion": cm.tolist()}
```

### 2.7 Tests

```python
# tests/test_shapes.py
import torch
from models.small_cnn import SmallCNN
from models.pretrained import build_pretrained

def test_small_cnn_shape():
    y = SmallCNN()(torch.randn(4, 3, 32, 32))
    assert y.shape == (4, 10)

def test_pretrained_shape():
    model = build_pretrained(num_classes=10)
    y = model(torch.randn(4, 3, 224, 224))
    assert y.shape == (4, 10)
```

```python
# tests/test_augmentation.py
import torch
from torchvision.datasets import CIFAR10
from data import build_transforms

def test_eval_transform_is_deterministic(tmp_path):
    ds = CIFAR10(root=str(tmp_path), train=False, download=True,
                 transform=build_transforms(train=False))
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)

def test_train_transform_has_randomness(tmp_path):
    ds = CIFAR10(root=str(tmp_path), train=True, download=True,
                 transform=build_transforms(train=True))
    torch.manual_seed(1); a, _ = ds[0]
    torch.manual_seed(2); b, _ = ds[0]
    assert not torch.equal(a, b)
```

```python
# tests/test_normalisation.py
from torchvision.models import ResNet18_Weights
from data import CIFAR_MEAN, build_transforms

def test_cifar_normalise_stats_applied():
    tfm = build_transforms(train=False)
    normalize = tfm.transforms[-1]
    assert tuple(float(x) for x in normalize.mean) == CIFAR_MEAN

def test_pretrained_uses_imagenet_stats():
    weights = ResNet18_Weights.IMAGENET1K_V1
    expected = weights.meta["mean"]
    # the pretrained loader uses weights.meta["mean"]; sanity-check its shape
    assert len(expected) == 3 and all(0.0 < m < 1.0 for m in expected)
```

```python
# tests/test_frozen.py
from models.pretrained import build_pretrained

def test_frozen_backbone_only_head_trainable():
    model = build_pretrained(num_classes=10, freeze_backbone=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]
```

```python
# tests/test_overfit.py
import torch
from torch import nn
from models.small_cnn import SmallCNN

def test_overfit_small_batch():
    torch.manual_seed(0)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    model = SmallCNN(dropout=0.0)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    initial = float(loss_fn(model(x), y))
    for _ in range(100):
        optim.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optim.step()
    final = float(loss_fn(model(x), y))
    assert final < 0.3 * initial
```

### 2.8 Augmentation ablation (Part F)

Three runs of the small CNN, same seed, same schedule, same epoch count. Record final val accuracy and the train − val loss gap at the last epoch.

| Aug level | Train transform | Expected final val acc | Expected overfit gap (train − val loss) |
|---|---|---|---|
| none | `ToTensor` + `Normalize` only | ~75–80% | Large: train ≪ val loss |
| standard | `RandomCrop(32,pad=4)` + `HorizontalFlip` | ~85–88% | Small to moderate |
| strong | + `RandAugment(num_ops=2, magnitude=9)` | ~86–89% | Smallest; sometimes a hair worse than "standard" |

Record the three numbers verbatim in `REPORT.md`; do not tune to beat them.

## 3. Validation steps

1. **Static import** — `python -c "import data, models.small_cnn, models.pretrained, train, evaluate"`.
2. **Test suite** — `pytest -q`. All five listed tests pass in under 30 seconds (the overfit test is the slow one; keep the loop short).
3. **Overfit-a-small-batch** — runs inside `test_overfit_small_batch`. If it fails, the model wiring is wrong; do not run the full training.
4. **Short end-to-end** — `python train.py --model small_cnn --epochs 2 --batch-size 128`. Confirms the full pipeline wires up; expect modest val accuracy (~45–55%) in two epochs.
5. **Shape invariants on real CIFAR batches** — one batch from `build_loaders(...)` must have shape `(B, 3, 32, 32)`; from `build_pretrained_loaders(...)` must have shape `(B, 3, 224, 224)`.
6. **Normalisation parity** — print the mean/std of one preprocessed batch from each loader; small-CNN batch should centre near 0 with std ≈ 1 under CIFAR stats; fine-tune batch should do the same under ImageNet stats.
7. **Full training runs** — small CNN 60 epochs, ResNet-18 15 epochs (full fine-tune). Record final test accuracy and wall-clock in `REPORT.md`. On a single 3080-class GPU these fit in roughly 10 and 20 minutes respectively; on CPU budget hours and halve the epochs per the exercise's instructions.
8. **Augmentation ablation** — run the small CNN three times with `--aug none|standard|strong`, record the three val accuracies and the train-val gap.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Small CNN architecture | 2 | ≥ 6 conv layers, BatchNorm after each, 2 downsamples, `AdaptiveAvgPool2d`, 200k–2M params; `test_small_cnn_shape` passes |
| 2 | Train vs. eval transforms | 2 | Val transform is deterministic (`test_eval_transform_is_deterministic`); train transform has randomness; no augmentation in val |
| 3 | Pretrained normalisation stats | 2 | Fine-tune uses `weights.meta["mean"] / ["std"]` — not CIFAR stats; `test_pretrained_uses_imagenet_stats` passes |
| 4 | Freeze logic | 1 | Only `fc.*` is trainable when `freeze_backbone=True`; `test_frozen_backbone_only_head_trainable` pins it |
| 5 | Optimiser + schedule | 2 | `AdamW` + `OneCycleLR` with `pct_start=0.1`; `scheduler.step()` fires per batch; label smoothing 0.1 |
| 6 | Gradient clipping + AMP | 2 | `clip_grad_norm_(1.0)` after `backward`, before `step`; `torch.autocast(bf16)` on CUDA with CPU fallback |
| 7 | Overfit-a-small-batch | 2 | `test_overfit_small_batch` brings loss to ≤ 30% of initial in ≤ 100 steps (the model has to fit 16 examples at `lr=3e-3`) |
| 8 | Metrics | 2 | Overall test accuracy ≥ 85% for small CNN, ≥ 90% for fully fine-tuned ResNet-18; per-class accuracy table + confusion matrix in `REPORT.md` |
| 9 | Augmentation ablation | 2 | Three val accuracies with `aug=none|standard|strong`, in a table; train − val loss gap discussed |
| 10 | Report | 3 | Dataset, architectures, parameter counts, training config, metrics, wall-clock, ablation, reflection answers |

A submission missing item 3 (normalisation stats) cannot score above 14 — it is the silent-accuracy-loss pitfall the whole fine-tune part is pedagogically built around.

## 5. Common mistakes

- **Using CIFAR mean/std for the ImageNet-pretrained ResNet.** The backbone was trained with `mean=[0.485, 0.456, 0.406]` / `std=[0.229, 0.224, 0.225]`; feeding it differently normalised inputs degrades top-1 accuracy by several points, silently. Use `weights.transforms()` or `weights.meta`.
- **Train augmentation on the val loader.** Val metric wobbles across epochs for mysterious reasons; the "best checkpoint" is on noise. The test for deterministic val transform exists for exactly this.
- **`scheduler.step()` once per epoch instead of per batch** with `OneCycleLR`. The schedule never completes a cycle; `max_lr` is never reached. Symptom: training is unexpectedly slow to converge and the LR log looks suspiciously flat.
- **`BatchNorm(bias=True)` after a `Conv2d(bias=True)`.** Doubles the per-channel bias parameter count; harmless numerically but wastes memory and signals "the author didn't know". Set `bias=False` on the conv.
- **Forgetting `drop_last=True` on train.** A final batch of 1 at the tail of training triggers `ValueError: Expected more than 1 value per channel` in BatchNorm. Fine until it isn't.
- **Freezing a backbone without also freezing BatchNorm running stats.** `requires_grad=False` stops *gradient-based* updates, but BN's `running_mean / running_var` keep drifting during `model.train()`. For true freeze-and-measure, also call `model.eval()` on the backbone or set `bn.track_running_stats = False`. The exercise doesn't require it, but mention it in `REPORT.md` if you compare.
- **`torch.autocast(device_type="cuda")` on a CPU-only machine.** Raises a runtime error at the first forward. The reference solution uses `nullcontext` on CPU — tested by instantiating the `autocast` variable in a smoke path.
- **`label_smoothing` in `CrossEntropyLoss` on a regression head.** `CrossEntropyLoss` is only for classification. The exercise is classification throughout; if you reach for regression later, `label_smoothing` makes no sense.
- **Overfit-test that uses a model with Dropout 0.5.** Dropout makes overfit-a-batch harder than it needs to be. Pass `dropout=0.0` in the test; in training, the real value is 0.2.

## 6. References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. CVPR. <https://arxiv.org/abs/1512.03385> — the residual block pattern used in `SmallCNN`.
- Smith, L. N. (2018). *A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay*. <https://arxiv.org/abs/1803.09820> — the 1-cycle schedule used here.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. <https://arxiv.org/abs/1711.05101> — justifies `AdamW`.
- Müller, R., Kornblith, S., & Hinton, G. (2019). *When does label smoothing help?* NeurIPS. <https://arxiv.org/abs/1906.02629>
- PyTorch documentation — `torch.autocast`, `torch.nn.utils.clip_grad_norm_`, `torch.optim.lr_scheduler.OneCycleLR`. <https://pytorch.org/docs/stable/>
- torchvision documentation — `torchvision.models.resnet18`, `ResNet18_Weights`, `torchvision.transforms.v2`. <https://pytorch.org/vision/stable/>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (test and validate on held-out data), `MEASURE 2.11` (document training data provenance and preprocessing). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft` (relevant to the "ship this fine-tuned model" step in production); `ML08:2023 Model Skewing` (train/val transform mismatch is one path to it). <https://owasp.org/www-project-machine-learning-security-top-10/>
