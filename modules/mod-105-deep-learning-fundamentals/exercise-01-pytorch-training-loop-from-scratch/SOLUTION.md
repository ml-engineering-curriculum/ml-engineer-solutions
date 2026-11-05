# exercise-01 — PyTorch Training Loop From Scratch · SOLUTION

Paired exercise: `lessons/mod-105-deep-learning-fundamentals/exercises/exercise-01-pytorch-training-loop-from-scratch.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to assemble a complete, vanilla-PyTorch training pipeline — dataset, DataLoader, MLP, training loop, tests, and a report — without any high-level wrapper (no Lightning, no `accelerate`, no `transformers.Trainer`). The point of the from-scratch loop is diagnostic: when training fails in a real project, the learner knows which of the four canonical lines (`zero_grad → forward+loss → backward → step`) is wrong.

The reference solution is organised around three invariants:

1. **Determinism given a seed** — same seed ⇒ same split, same shuffle order, same init. Non-determinism makes bugs non-reproducible, which makes them un-fixable.
2. **Fail-loud smoke tests before full training** — the single-batch overfit test is the "did I write a training loop" oracle. Every pass of the loop runs it first.
3. **Four lines in one order** — `optimizer.zero_grad() → outputs = model(xb) → loss = loss_fn(outputs, yb) → loss.backward() → optimizer.step()`. Any deviation is a bug; the tests pin that order.

The worked implementation uses **California Housing** (regression) because it loads with one `scikit-learn` call, has no categoricals, and gives an unambiguous metric (MAE). A classification variant on **UCI Adult** is noted inline where it differs.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| `AdamW(lr=1e-3, weight_decay=1e-4)` default | SGD + momentum, plain Adam | Works on almost every small tabular problem with no tuning; weight decay decoupled from grad as in Loshchilov & Hutter (2019) |
| Half-open val split via `train_test_split(stratify=...)` | Hand-rolled index split | `sklearn` gives deterministic, stratified splits for free; reproducing a split from a seed is one line |
| Standardise features on the train split only | Standardise on the full data | No leakage; the exercise explicitly calls this out, and it is the single most common first-mistake in tabular pipelines |
| `TensorDataset` + `DataLoader` with `shuffle=True, drop_last=True` on train | Hand-rolled minibatcher | Reaches for the right abstraction (chapter 04) without obscuring the loop |
| Early stopping as a tiny helper class, not a callback framework | `EarlyStopping(patience, min_delta)` in a wrapper | Keeps the loop readable; the test can drive the helper directly |
| Save `{"epoch", "model", "optimizer", "val_loss", "meta"}` per checkpoint | Save only `state_dict` | Reloading optimiser state is what lets a resume actually continue training; `meta` carries the scaler stats so inference can standardise identically |

The solution is written to pass all seven required tests and reach the dataset-appropriate validation-metric floors listed in the exercise's acceptance criteria.

## 2. Worked implementation

### 2.1 Project layout

```
exercise-01/
├── data.py                  # load_split, make_loaders
├── model.py                 # MLP, count_parameters
├── train.py                 # EarlyStopper, train(), evaluate(), CLI entry
├── tests/
│   ├── test_loop.py         # overfit-batch, grad flow, zero_grad
│   ├── test_model.py        # dropout toggle, shapes, param count
│   ├── test_data.py         # deterministic split, dtype guarantees
│   └── test_checkpoint.py   # roundtrip, early stop
├── plots/
│   └── loss_curve.png
├── history.json
├── pyproject.toml
└── REPORT.md
```

### 2.2 `data.py` — deterministic split, no leakage

```python
# data.py
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SplitMeta:
    n_features: int
    task: str                  # "regression" | "classification"
    n_classes: int | None
    feature_names: list[str]
    scaler_mean: list[float]
    scaler_std: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "task": self.task,
            "n_classes": self.n_classes,
            "feature_names": list(self.feature_names),
            "scaler_mean": list(self.scaler_mean),
            "scaler_std": list(self.scaler_std),
        }


def load_split(
    seed: int = 0,
    test_frac: float = 0.2,
) -> tuple[TensorDataset, TensorDataset, SplitMeta]:
    """
    California Housing regression. Deterministic given `seed`.
    Fits the scaler on train only, applies to both splits.
    """
    bundle = fetch_california_housing(as_frame=True)
    X = bundle.data.to_numpy(dtype=np.float64)
    y = bundle.target.to_numpy(dtype=np.float64)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_frac, random_state=seed, shuffle=True,
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32, copy=False)
    X_val = scaler.transform(X_val).astype(np.float32, copy=False)
    y_tr = y_tr.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    meta = SplitMeta(
        n_features=X.shape[1],
        task="regression",
        n_classes=None,
        feature_names=list(bundle.feature_names),
        scaler_mean=scaler.mean_.tolist(),
        scaler_std=scaler.scale_.tolist(),
    )
    return train_ds, val_ds, meta


def make_loaders(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, val_loader
```

Key choices, each with a predictable failure mode it blocks:

- **`shuffle` seeded by a `torch.Generator`.** Default `shuffle=True` pulls from the global RNG and the order drifts across runs even with the same `seed`. The generator is the deterministic fix.
- **`drop_last=True` on train, `False` on val.** Training prefers batches of identical size for predictable BatchNorm statistics (not used here but a habit worth forming); validation needs every example to count.
- **`float32` features, `float32` targets for regression** (`int64` for classification labels). `float64` silently doubles memory and halves throughput for no accuracy gain.
- **Scaler fit on train only.** This is the single most common tabular-leak bug; the explicit comment in the function docstring is for the reviewer, not you.

For a classification variant (UCI Adult), swap `fetch_california_housing` for `fetch_openml("adult", version=2, as_frame=True)`, one-hot encode categoricals, and change `task="classification", n_classes=2`, `y_*` dtype to `int64`.

### 2.3 `model.py` — small MLP

```python
# model.py
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- No activation on the final layer — the model outputs logits (classification) or raw targets (regression). Combining it with `BCEWithLogitsLoss` or `CrossEntropyLoss` (classification) or `L1Loss` / `MSELoss` (regression) keeps numerical stability where it belongs.
- Dropout is between linears, not after the final linear.
- `count_parameters` only counts trainable params — important when the fine-tune exercises freeze a backbone later.

### 2.4 `train.py` — the four-line loop, plus plumbing

```python
# train.py
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import load_split, make_loaders, SplitMeta
from model import MLP, count_parameters

log = logging.getLogger("train")


@dataclass
class EarlyStopper:
    patience: int
    min_delta: float = 0.0
    best: float = math.inf
    bad_epochs: int = 0

    def step(self, val_loss: float) -> bool:
        """Return True to stop, False to keep going."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    task: str,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    abs_err = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = model(xb)
        if task == "regression":
            out = out.squeeze(-1)
        loss = loss_fn(out, yb)
        bs = yb.shape[0]
        total_loss += float(loss) * bs
        total_n += bs
        if task == "classification":
            correct += int((out.argmax(dim=-1) == yb).sum())
        else:
            abs_err += float((out - yb).abs().sum())
    val_loss = total_loss / max(total_n, 1)
    metric = correct / total_n if task == "classification" else abs_err / total_n
    return val_loss, metric


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    ckpt_path: Path,
    task: str = "regression",
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_step: str = "epoch",    # "epoch" | "batch"
    early_patience: int | None = None,
    log_every: int = 50,
    meta: dict | None = None,
) -> dict:
    model.to(device)
    stopper = EarlyStopper(patience=early_patience) if early_patience else None
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_metric": [], "epoch_seconds": [],
    }
    best_val = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        running_n = 0

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if task == "regression":
                out = out.squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            if scheduler is not None and scheduler_step == "batch":
                scheduler.step()

            bs = yb.shape[0]
            running_loss += float(loss) * bs
            running_n += bs
            if (step + 1) % log_every == 0:
                log.info(
                    "epoch=%d step=%d running_train_loss=%.4f",
                    epoch, step + 1, running_loss / running_n,
                )

        train_loss = running_loss / max(running_n, 1)
        val_loss, val_metric = evaluate(model, val_loader, loss_fn, task, device)
        epoch_seconds = time.perf_counter() - t0

        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        history["epoch_seconds"].append(epoch_seconds)

        log.info(
            "epoch=%d train=%.4f val=%.4f metric=%.4f secs=%.2f",
            epoch, train_loss, val_loss, val_metric, epoch_seconds,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "meta": meta or {},
                },
                ckpt_path,
            )

        if stopper is not None and stopper.step(val_loss):
            log.info("early stop at epoch=%d best_val=%.4f", epoch, stopper.best)
            break

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ckpt", type=Path, default=Path("./best.pt"))
    parser.add_argument("--history", type=Path, default=Path("./history.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(args.seed)

    train_ds, val_ds, meta = load_split(seed=args.seed)
    train_loader, val_loader = make_loaders(
        train_ds, val_ds, batch_size=args.batch_size, seed=args.seed,
    )

    out_features = 1 if meta.task == "regression" else (meta.n_classes or 2)
    model = MLP(
        in_features=meta.n_features,
        hidden_dims=list(args.hidden),
        out_features=out_features,
        dropout=args.dropout,
    )
    log.info("model params=%d", count_parameters(model))

    loss_fn: nn.Module = nn.L1Loss() if meta.task == "regression" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    history = train(
        model, train_loader, val_loader,
        loss_fn=loss_fn, optimizer=optimizer,
        epochs=args.epochs, device=args.device,
        ckpt_path=args.ckpt, task=meta.task,
        early_patience=args.patience,
        meta=meta.to_dict(),
    )
    args.history.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
```

Notes on each ordered step inside the batch loop:

1. **`optimizer.zero_grad(set_to_none=True)`** — faster than `zero_grad()` and the only form that lets `test_zero_grad_resets_grads` pass (checks `.grad is None`, not `.grad.abs().sum() == 0`).
2. **`out = model(xb)` then `loss = loss_fn(out, yb)`** — two statements, not one, so a shape mismatch error points at the real line.
3. **`loss.backward()`** — exactly once per batch. Calling it twice without `zero_grad` accumulates grads into the next step, which is a silent bug.
4. **`optimizer.step()`** — after `backward`, before the next `zero_grad`. Swap these two and the optimiser steps on the *previous* batch's grads.

### 2.5 Tests — the seven required checks

```python
# tests/test_loop.py
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLP


def _tiny_batch(n: int = 8, d: int = 8, classes: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return x, y


def test_overfit_single_batch() -> None:
    torch.manual_seed(0)
    xb, yb = _tiny_batch()
    model = MLP(in_features=8, hidden_dims=[32, 32], out_features=2, dropout=0.0)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    initial = float(loss_fn(model(xb), yb))
    for _ in range(300):
        optim.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optim.step()
    final = float(loss_fn(model(xb), yb))

    assert final < 0.01 * initial, (initial, final)


def test_gradients_flow() -> None:
    torch.manual_seed(0)
    xb, yb = _tiny_batch()
    model = MLP(8, [16], 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()(model(xb), yb)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert float(p.grad.norm()) > 0.0, name


def test_zero_grad_resets_grads() -> None:
    model = MLP(8, [16], 2)
    xb, yb = _tiny_batch()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nn.CrossEntropyLoss()(model(xb), yb).backward()
    optim.zero_grad(set_to_none=True)
    for p in model.parameters():
        assert p.grad is None
```

```python
# tests/test_model.py
import torch
from model import MLP, count_parameters


def test_model_train_eval_toggle_dropout() -> None:
    torch.manual_seed(0)
    model = MLP(8, [16, 16], 4, dropout=0.5)
    x = torch.randn(4, 8)
    model.train()
    a, b = model(x), model(x)
    assert not torch.allclose(a, b)
    model.eval()
    c, d = model(x), model(x)
    assert torch.allclose(c, d)


def test_count_parameters_matches_manual() -> None:
    model = MLP(10, [8], 2, dropout=0.0)
    manual = (10 * 8 + 8) + (8 * 2 + 2)  # Linear + bias, twice
    assert count_parameters(model) == manual
```

```python
# tests/test_data.py
import numpy as np
import torch
from data import load_split, make_loaders


def test_deterministic_split() -> None:
    a, b, _ = load_split(seed=0)
    c, d, _ = load_split(seed=0)
    assert torch.equal(a.tensors[0], c.tensors[0])
    assert torch.equal(a.tensors[1], c.tensors[1])
    assert torch.equal(b.tensors[0], d.tensors[0])


def test_dtype_contract() -> None:
    train_ds, val_ds, meta = load_split(seed=0)
    assert train_ds.tensors[0].dtype == torch.float32
    assert val_ds.tensors[0].dtype == torch.float32
    if meta.task == "classification":
        assert train_ds.tensors[1].dtype == torch.int64
    else:
        assert train_ds.tensors[1].dtype == torch.float32
```

```python
# tests/test_checkpoint.py
from pathlib import Path
import torch
from torch import nn
from train import EarlyStopper, train
from data import load_split, make_loaders
from model import MLP


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    train_ds, val_ds, meta = load_split(seed=0)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=64, seed=0)
    model = MLP(meta.n_features, [16], 1, dropout=0.0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "best.pt"
    train(
        model, train_loader, val_loader,
        loss_fn=nn.L1Loss(), optimizer=optim,
        epochs=1, device="cpu", ckpt_path=ckpt, task="regression",
    )
    assert ckpt.exists()

    reloaded = MLP(meta.n_features, [16], 1, dropout=0.0)
    state = torch.load(ckpt, map_location="cpu")
    reloaded.load_state_dict(state["model"])
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_early_stopping_triggers() -> None:
    stopper = EarlyStopper(patience=2)
    series = [1.0, 0.9, 0.95, 1.1, 1.2, 1.3]  # improves once then degrades
    fired_at = None
    for i, v in enumerate(series):
        if stopper.step(v):
            fired_at = i
            break
    assert fired_at == 4  # best at index 1; +patience(2) + 1 bad step
```

Mapping to the acceptance criteria:

| Required test | Where it lives |
|---|---|
| `test_overfit_single_batch` | `tests/test_loop.py` |
| `test_gradients_flow` | `tests/test_loop.py` |
| `test_zero_grad_resets_grads` | `tests/test_loop.py` |
| `test_model_train_eval_toggle_dropout` | `tests/test_model.py` |
| `test_checkpoint_roundtrip` | `tests/test_checkpoint.py` |
| `test_early_stopping_triggers` | `tests/test_checkpoint.py` |
| `test_deterministic_split` | `tests/test_data.py` |

### 2.6 `REPORT.md` skeleton

```markdown
# California Housing — MLP from scratch

## Dataset
- Source: sklearn `fetch_california_housing(as_frame=True)` — 20,640 rows × 8 features.
- Split: 80/20 train/val, seed=0, stratified by binned target quartile (optional).

## Model
- Architecture: MLP(8 → 128 → 64 → 1), ReLU, Dropout 0.2.
- Parameters: 9,537 trainable.

## Training
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-4).
- Loss: L1Loss (MAE on target scaled to $100k units).
- Batch size: 128. Epochs: 30 with early stopping patience=5.
- Hardware: <your CPU/GPU, wall-clock per epoch>.

## Results
- Best val MAE: <fill>.
- mod-104 baseline (HistGradientBoostingRegressor, default params): MAE <fill>, wall-clock <fill>.

![loss curve](plots/loss_curve.png)

## Reflection
- (answers to the four reflection questions)
```

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **Static import check** — `python -c "import data, model, train"`. Catches missing dependencies and import-time typos.
2. **Overfit-a-batch smoke test** — `pytest -q tests/test_loop.py::test_overfit_single_batch`. If this fails, the loop is wrong; stop and fix before running on real data.
3. **Full test suite** — `pytest -q`. All seven listed tests should pass in well under 30 seconds on CPU.
4. **End-to-end run** — `python train.py --epochs 2`. Confirms the CLI, DataLoaders, scheduler, and checkpoint save all wire up.
5. **No-high-level-wrapper grep** — `grep -RIn "lightning\|accelerate\|transformers.Trainer\|fabric" .` must return nothing.
6. **Reproducibility** — run `python train.py` twice with the same `--seed`; compare `history.json` first-epoch `train_loss` values; they should be bit-identical on CPU, within 1e-4 on CUDA (non-deterministic kernels).
7. **Baseline comparison** — fit `HistGradientBoostingRegressor` on the same split; log MAE and wall-clock in `REPORT.md`. The MLP need not beat it — it needs to be in the same ball-park and the report must say so honestly.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Loop ordering | 3 | `zero_grad → forward → loss → backward → step` in that order, once per batch. Pin with `test_overfit_single_batch` and `test_gradients_flow` |
| 2 | Determinism given seed | 2 | `load_split(seed=S)` is byte-identical across calls; DataLoader shuffle uses a seeded `torch.Generator` |
| 3 | No leakage | 2 | Scaler fit on train only; applied to val. Reviewer scans `data.py` and finds the fit call inside the train branch |
| 4 | Dtype contract | 1 | `float32` features; `int64` labels for classification; `float32` targets for regression. Enforced by `test_dtype_contract` |
| 5 | Dropout train/eval toggle | 1 | Two forward passes in `train()` differ; two in `eval()` match. Enforced by `test_model_train_eval_toggle_dropout` |
| 6 | Validation hygiene | 2 | `model.eval()` + `@torch.inference_mode()` in the eval path; no gradients allocated |
| 7 | Checkpoint shape | 2 | Saves `{epoch, model, optimizer, val_loss, meta}`; roundtrip test passes; `meta` carries the scaler stats for inference parity |
| 8 | Early stopping | 2 | Monotonically-increasing val-loss series stops after `patience + 1` bad epochs; `test_early_stopping_triggers` pins it |
| 9 | Vanilla PyTorch only | 2 | No `lightning`, `accelerate`, `transformers.Trainer`, `fabric`. grep returns nothing |
| 10 | Report | 3 | Dataset + source URL + shape; model + parameter count; training config; loss curve PNG; `mod-104` baseline comparison; reflection answers |

Partial credit: a submission missing items 1 or 3 cannot score above 14 — those are the two items that cause real downstream incidents (silently-wrong training; leaked validation metric).

## 5. Common mistakes

- **`optimizer.step()` before `loss.backward()`.** Silently steps on the previous batch's grads (or on zero). The `test_gradients_flow` test catches it only if `backward` ran at all; the overfit-batch test is the real oracle — loss never drops.
- **Forgetting `optimizer.zero_grad()`.** Grads accumulate across batches; training "works" but is effectively seeing a running sum of batches. Loss drops slowly and plateaus high. Pattern: val loss flattens at a weird value that isn't near the floor.
- **`float64` in the DataLoader.** Doubles memory bandwidth for no gain on an MLP; a common first-run surprise when `sklearn` hands back `np.float64`. Explicit `.astype(np.float32)` fixes it.
- **Scaler fit on the full data.** Classic tabular leak. Val metric looks a few percent better than it should; the production metric regresses. The fix is one line; the habit is the whole point.
- **Shuffled val loader.** Harmless for a metric; catastrophic if you also want to cache per-example predictions for error analysis. Default `shuffle=False` on val.
- **`torch.save(model)` instead of `torch.save({...})`.** Pickles the module path; renaming the class breaks reload. Always save `state_dict()` inside a dict with metadata.
- **BatchNorm in the MLP with `drop_last=False`.** A batch of 1 at the tail of training triggers "Expected more than 1 value per channel". The exercise sidesteps this with Dropout, but it is worth naming because `drop_last=True` is why we use it.
- **Early stop that compares to the **previous** val loss, not the **best** so far.** A noisy training signal can "improve" one epoch and reset the patience counter indefinitely. The reference `EarlyStopper` keeps `self.best`.

## 6. References

- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. <https://arxiv.org/abs/1711.05101> — justifies `AdamW` as the "default" optimiser.
- PyTorch documentation — `torch.optim.AdamW`, `torch.utils.data.DataLoader`, `torch.nn.Module.train/eval`. <https://pytorch.org/docs/stable/>
- scikit-learn documentation — `fetch_california_housing`, `train_test_split`, `StandardScaler`. <https://scikit-learn.org/stable/>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (test and validate on held-out data) and `MAP 1.6` (document experimental setup). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` as the broader frame for "validate inputs before they reach training." <https://owasp.org/www-project-machine-learning-security-top-10/>
