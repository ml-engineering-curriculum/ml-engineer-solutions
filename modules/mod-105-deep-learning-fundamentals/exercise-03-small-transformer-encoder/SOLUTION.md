# exercise-03 — Small Transformer Encoder · SOLUTION

Paired exercise: `lessons/mod-105-deep-learning-fundamentals/exercises/exercise-03-small-transformer-encoder.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to build a **small Transformer encoder from scratch** and compare it against two baselines:

1. A `nn.TransformerEncoder` reference wrapped in the same embedding + pooling + classifier so only the block implementation differs.
2. A fine-tuned `distilbert-base-uncased` via Hugging Face `transformers`.

The exercise-01 training loop is reused. What is new is the tokenization + padded-collate pipeline (chapter 04), the Transformer-specific optimiser setup (AdamW + warmup + cosine decay + gradient clipping), and the mask conventions that trip up every first implementation.

The reference solution is structured around four invariants:

1. **Pre-LN everywhere, GELU everywhere, `d_ff = 4 * d_model`.** Pre-LN (Xiong et al. 2020) trains stably from scratch at small depth without warmup tuning; it is the configuration both reference implementations converge on.
2. **One mask convention: `True` where padded.** Matches `nn.TransformerEncoderLayer`'s `src_key_padding_mask`. The from-scratch attention inverts it internally when calling `F.scaled_dot_product_attention` (which expects "key is attended to" = non-padded); the convention at the module boundary is the same for both models.
3. **From-scratch and reference stay within 3 points of accuracy.** If they don't, there is a bug — the test suite's parity test is designed to catch it before training.
4. **Three models on the same split.** Same train/val/test rows, same tokenisation where possible (the DistilBERT fine-tune uses DistilBERT's own tokenizer — that is the entire point of the fine-tune comparison).

The worked implementation uses **AG News** (4-class news topic classification). It is small enough to train each model in a few minutes on a modest GPU (an afternoon on CPU with halved epochs), and large enough that the DistilBERT fine-tune clearly outperforms the from-scratch model.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| AG News | IMDb, DBpedia-14, TREC | Fastest to iterate; 4 classes gives a confusion matrix rich enough to talk about; the chapter 07 worked example uses it |
| Pre-LN Transformer | Post-LN (original paper) | Trains stably at `d_model=128, n_layers=4` without a long warmup; Xiong et al. (2020) |
| Learned positional embeddings | Sinusoidal, RoPE | Simplest baseline; matches `nn.TransformerEncoderLayer`'s learnable path if you tie the embedding; stretch goal lifts this |
| Mean pooling over non-pad positions | `[CLS]` pooling | One fewer convention to get right; also what chapter 07 uses. Stretch goal swaps in `[CLS]` |
| `F.scaled_dot_product_attention` inside `MultiHeadSelfAttention` | Hand-rolled `softmax(QKᵀ/√d)V` | Uses the memory-efficient kernels when available; still requires manual QKV projection and head reshape so the learning objective is met; the hand-rolled form appears in a comment and in a unit test |
| AdamW(lr=3e-4, wd=0.01), cosine with warmup | Plain Adam | Warmup is the fix for the Transformer's early-training instability; AdamW decouples weight decay so the Transformer's residual stream isn't over-regularised |
| `clip_grad_norm_(1.0)` | `clip_grad_value_` | Norm clipping scales all grads uniformly; value clipping distorts direction. The exercise's "pre-clip norm" log justifies norm clipping |
| Pad token id 0, `nn.Embedding(..., padding_idx=0)` | Pad id 1, no padding_idx | Freezes the pad embedding at zero — removes an entire class of "pad embedding drifted" bugs |

The solution is written to reach the exercise's acceptance floors: **≥ 88% test accuracy** for the from-scratch encoder on AG News, **≥ 93%** for the DistilBERT fine-tune, with the from-scratch and `nn.TransformerEncoder` reference **within 3 percentage points of each other**.

## 2. Worked implementation

### 2.1 Project layout

```
exercise-03/
├── data.py                        # tokenizer, dataset, collate_pad, loaders
├── models/
│   ├── __init__.py
│   ├── scratch.py                 # MultiHeadSelfAttention, TransformerBlock, Encoder
│   ├── reference.py               # same shape via nn.TransformerEncoder
│   └── pretrained.py              # DistilBERT wrapper
├── train.py                       # extends exercise-01's loop
├── evaluate.py                    # accuracy, macro-F1, confusion, 10 predictions
├── tests/
│   ├── test_scratch_vs_reference.py
│   ├── test_padding_and_positional.py
│   ├── test_overfit.py
│   └── test_schedule_and_clipping.py
├── plots/
├── history_scratch.json
├── history_reference.json
├── history_distilbert.json
├── tokenizer/                     # serialised tokenizer artefacts (if BPE path chosen)
├── pyproject.toml
└── REPORT.md
```

### 2.2 `data.py` — tokenizer path, padded collate, loaders

The reference uses **tokenizer path 2** (reuse a pretrained tokenizer's vocab). It keeps the comparison against DistilBERT clean (same tokenisation, same vocab size) while avoiding the time cost of training a BPE tokenizer in the critical path.

```python
# data.py
from dataclasses import dataclass
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

TOKENIZER_NAME = "distilbert-base-uncased"
PAD_ID = 0  # DistilBERT uses 0 for [PAD]; verified at runtime


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    assert tok.pad_token_id == PAD_ID, (tok.pad_token_id, PAD_ID)
    return tok


class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        return {
            "ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_pad(batch: list[dict], pad_id: int = PAD_ID) -> dict[str, torch.Tensor]:
    """Pad to the longest sequence in the batch."""
    lens = [len(b["ids"]) for b in batch]
    L = max(lens)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    padding_mask = ids == pad_id  # True where padded
    labels = torch.stack([b["label"] for b in batch])
    return {"ids": ids, "padding_mask": padding_mask, "label": labels}


def build_loaders(
    train: tuple[list[str], list[int]],
    val: tuple[list[str], list[int]],
    tokenizer,
    batch_size: int = 32,
    max_len: int = 128,
    num_workers: int = 2,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TextClassificationDataset(*train, tokenizer=tokenizer, max_len=max_len)
    val_ds = TextClassificationDataset(*val, tokenizer=tokenizer, max_len=max_len)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, collate_fn=collate_pad,
        pin_memory=True, persistent_workers=num_workers > 0, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, drop_last=False,
        num_workers=num_workers, collate_fn=collate_pad,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
```

Key decisions:

- **`collate_pad` pads to the batch max**, not the global `max_len`. On AG News the median length is ~40 tokens; padding to 128 wastes ~2/3 of compute.
- **`padding_mask` convention: `True` where padded.** Matches `nn.TransformerEncoderLayer`'s `src_key_padding_mask`. The from-scratch attention will invert it when calling `F.scaled_dot_product_attention` (which uses `True` = "key is visible to query").
- **`add_special_tokens=True`** — the tokenizer inserts `[CLS]` and `[SEP]` at positions 0 and the final content position, which the DistilBERT fine-tune uses as-is. For the from-scratch model we mean-pool, so these special tokens are just extra positions; the model learns to handle them.
- **`num_workers=2`** by default; `num_workers=0` during debugging (multi-worker tokenization is harder to debug when a bug surfaces).

### 2.3 `models/scratch.py` — the from-scratch encoder

```python
# models/scratch.py
import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, L, Dh)

        # Build attn_mask in the (B, 1, 1, L) convention: True where attended-to.
        attn_mask = None
        if padding_mask is not None:
            keep = ~padding_mask                     # (B, L): True where real token
            attn_mask = keep[:, None, None, :]       # (B, 1, 1, L); broadcasts over heads, queries

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, L, D)       # (B, L, D)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), padding_mask=padding_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(self.positions[:, :L])
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, padding_mask=padding_mask)
        x = self.ln_f(x)

        # Mean pool over non-padded positions.
        keep = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Mapping to the exercise's requirements:

- **Fused QKV + reshape** — `self.qkv = nn.Linear(d_model, 3 * d_model)`, then one reshape + permute to split heads. This is what `nn.MultiheadAttention` does internally; writing it by hand is the point.
- **`F.scaled_dot_product_attention` for the math.** Produces identical outputs to the hand-rolled `softmax(QKᵀ/√d)V` within `atol=1e-5` and uses the memory-efficient kernel when available. The "do it by hand at least once" box is checked by the unit test below.
- **Pre-LN, GELU, `d_ff = 4 * d_model`** — the `TransformerBlock` applies `LayerNorm` before each sublayer and adds the residual after.
- **Learned positional embedding added to token embedding.** `register_buffer("positions", ...)` keeps the index tensor on the right device without putting it in `state_dict`.
- **Mean pool over non-pad positions.** Denominator `clamp_min(1.0)` guards against an all-pad sequence (should never happen, but a one-line safety).
- **Weight init `std=0.02`** — the GPT-2 / mini-BERT standard; different enough from PyTorch's default that leaving it at default harms convergence noticeably.

The hand-rolled attention as a reference (and the subject of the parity test):

```python
def _manual_sdp(q, k, v, attn_mask):
    # q, k, v: (B, H, L, Dh); attn_mask: (B, 1, 1, L) bool, True = keep
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v
```

### 2.4 `models/reference.py` — same shape, built-in blocks

```python
# models/reference.py
import torch
from torch import nn


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0), persistent=False)

        # Match the scratch init for parity.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(self.positions[:, :L])
        x = self.drop(x)
        # nn.TransformerEncoder expects src_key_padding_mask with True=padded; same convention.
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.ln_f(x)
        keep = (~padding_mask).float().unsqueeze(-1)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
```

The only structural difference from `TransformerEncoder` is the block body — the embedding, positional, pooling, and head are reused line-for-line. This is what makes the within-3-points parity test meaningful: we are comparing block implementations, not architectures.

### 2.5 `models/pretrained.py` — DistilBERT wrapper

```python
# models/pretrained.py
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DistilBERTClassifier(nn.Module):
    def __init__(self, n_classes: int, name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=n_classes,
        )

    def forward(
        self, ids: torch.Tensor, padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # HF expects attention_mask with 1 = attend, 0 = pad — inverse of padding_mask.
        attention_mask = (~padding_mask).long()
        out = self.model(input_ids=ids, attention_mask=attention_mask)
        return out.logits
```

Decisions:

- **Reuse the `(ids, padding_mask)` interface** so the training loop treats all three models identically. The adapter converts the mask convention on the way in.
- **`AutoModelForSequenceClassification`** builds the classifier head automatically, sized to `num_labels`. Reaching for `DistilBertModel` + hand-rolled head is an exercise for a later module, not here.
- **No freezing.** The exercise asks for a *full* fine-tune for the headline number; the "fine-tune with frozen backbone" variant was the point of exercise-02.

### 2.6 `train.py` — loop extensions: warmup, clipping, LR logging

```python
# train.py — extensions to exercise-01's core; four-line loop body unchanged
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_step(
    model, batch, loss_fn, optimizer, scheduler, device, clip_norm=1.0,
) -> tuple[float, float]:
    ids = batch["ids"].to(device, non_blocking=True)
    padding_mask = batch["padding_mask"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    logits = model(ids, padding_mask)
    loss = loss_fn(logits, labels)
    loss.backward()
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    scheduler.step()
    return float(loss), float(pre_clip_norm)
```

Per-model hyperparameters:

| Model | Peak LR | Weight decay | Warmup frac | Epochs (GPU) | Epochs (CPU) |
|---|---|---|---|---|---|
| From-scratch encoder | `3e-4` | `0.01` | 10% | 10 | 5 |
| PyTorch reference | `3e-4` | `0.01` | 10% | 10 | 5 |
| DistilBERT fine-tune | `3e-5` | `0.01` | 10% | 3 | 2 |

**Logging** — every `log_every` batches, print `epoch, step, loss, lr, pre_clip_norm, step_seconds`. The LR trace confirms the warmup ramp; the pre-clip norm trace confirms clipping is actually doing work early on and becomes a no-op later.

### 2.7 Tests

```python
# tests/test_scratch_vs_reference.py
import math
import torch
from models.scratch import MultiHeadSelfAttention, TransformerEncoder
from models.reference import ReferenceEncoder


def test_scratch_matches_reference_output_shape():
    torch.manual_seed(0)
    scratch = TransformerEncoder(vocab_size=100, n_classes=4)
    ref = ReferenceEncoder(vocab_size=100, n_classes=4)
    ids = torch.randint(1, 100, (4, 32))
    mask = torch.zeros(4, 32, dtype=torch.bool)
    assert scratch(ids, mask).shape == ref(ids, mask).shape == (4, 4)


def test_scratch_attention_matches_manual_sdp():
    torch.manual_seed(0)
    mod = MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0)
    mod.eval()
    x = torch.randn(2, 8, 32)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    y_fast = mod(x, padding_mask=mask)

    # Manual softmax path for the same q, k, v.
    B, L, D = x.shape
    qkv = mod.qkv(x).reshape(B, L, 3, mod.n_heads, mod.d_head).permute(2, 0, 3, 1, 4)
    q, k, v = qkv
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(mod.d_head)
    weights = scores.softmax(dim=-1)
    y_manual = (weights @ v).transpose(1, 2).reshape(B, L, D)
    y_manual = mod.out(y_manual)
    assert torch.allclose(y_fast, y_manual, atol=1e-5)
```

```python
# tests/test_padding_and_positional.py
import torch
from models.scratch import TransformerEncoder


def test_padding_mask_zeros_padded_positions():
    torch.manual_seed(0)
    model = TransformerEncoder(vocab_size=100, n_classes=4).eval()
    base_ids = torch.randint(1, 100, (1, 10))
    padded_ids = torch.cat([base_ids, torch.zeros(1, 6, dtype=torch.long)], dim=1)

    base_mask = torch.zeros(1, 10, dtype=torch.bool)
    padded_mask = torch.cat(
        [torch.zeros(1, 10, dtype=torch.bool), torch.ones(1, 6, dtype=torch.bool)], dim=1,
    )

    y_base = model(base_ids, base_mask)
    y_padded = model(padded_ids, padded_mask)
    assert torch.allclose(y_base, y_padded, atol=1e-5), (y_base - y_padded).abs().max()


def test_positional_matters():
    torch.manual_seed(0)
    model = TransformerEncoder(vocab_size=100, n_classes=4).eval()
    ids = torch.randint(1, 100, (1, 10))
    mask = torch.zeros(1, 10, dtype=torch.bool)
    y_a = model(ids, mask)
    y_b = model(ids[:, torch.randperm(10)], mask)
    assert not torch.allclose(y_a, y_b, atol=1e-3)
```

```python
# tests/test_overfit.py
import torch
from torch import nn
from models.scratch import TransformerEncoder


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    model = TransformerEncoder(
        vocab_size=100, n_classes=4,
        d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=32, dropout=0.0,
    )
    ids = torch.randint(1, 100, (16, 16))
    mask = torch.zeros(16, 16, dtype=torch.bool)
    labels = torch.randint(0, 4, (16,))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    initial = float(loss_fn(model(ids, mask), labels))
    for _ in range(200):
        optim.zero_grad(set_to_none=True)
        loss_fn(model(ids, mask), labels).backward()
        optim.step()
    final = float(loss_fn(model(ids, mask), labels))
    assert final < 0.01 * initial, (initial, final)
```

```python
# tests/test_schedule_and_clipping.py
import torch
from torch import nn
from train import warmup_cosine_schedule


def test_warmup_lr_ramp():
    model = nn.Linear(4, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = warmup_cosine_schedule(optim, warmup_steps=100, total_steps=1000)
    lrs = []
    for _ in range(100):
        optim.step(); sched.step()
        lrs.append(optim.param_groups[0]["lr"])
    assert lrs[0] < 0.05
    assert abs(lrs[-1] - 1.0) < 1e-3


def test_grad_clip_norm_reports_correctly():
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    assert torch.isfinite(pre_clip) and float(pre_clip) > 0.0
```

### 2.8 `REPORT.md` — results shape

```markdown
# AG News — Three Transformer encoders

## Dataset
- AG News, 120k train / 7.6k test, 4 classes.
- Class distribution: 30k each at train (balanced).

## Tokenizer
- Reused `distilbert-base-uncased` vocab (vocab_size=30522). All three models use the same tokenisation.

## Models

| Model | Params | Test acc | Macro-F1 | Wall-clock (train) |
|---|---|---|---|---|
| From-scratch encoder (d=128, h=4, L=4) | ~1.0M | ≥ 0.88 | ≥ 0.88 | <fill> |
| nn.TransformerEncoder reference         | ~1.0M | within 3pp of scratch | | <fill> |
| DistilBERT fine-tune                    | ~66M  | ≥ 0.93 | ≥ 0.93 | <fill> |

(Hardware, confusion matrices, 10 interesting predictions, reflection answers below.)
```

## 3. Validation steps

1. **Static import** — `python -c "import data, models.scratch, models.reference, models.pretrained, train"`.
2. **Test suite** — `pytest -q`. The seven listed checks (padding mask, positional, overfit tiny batch, scratch-vs-manual-SDP, warmup ramp, grad clip finite, shape parity) run in well under a minute.
3. **Overfit-tiny-batch** — the single most useful debugger. If it fails, the mask or the init is wrong; do not train on real data.
4. **Scratch-vs-reference parity, trained** — run both for 2 epochs with the same seed and record test accuracy. If they differ by more than 3 percentage points, there is a bug — usually the mask convention, occasionally the pooling (mean including pad positions).
5. **LR / grad-norm log inspection** — tail the first 500 steps of a from-scratch training run. LR should ramp linearly over the warmup window, pre-clip grad norm should be in the 1–5 range and occasionally spike toward 10 in early steps.
6. **DistilBERT first epoch** — after one epoch on AG News, val accuracy should be ≥ 0.90 on GPU. If not, the loss function or the mask conversion is wrong; check `attention_mask = (~padding_mask).long()`.
7. **Full training runs** — three models to the suggested epoch counts; save histories and confusion matrices.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Attention implementation | 3 | Fused QKV + head reshape; `F.scaled_dot_product_attention` or hand-rolled softmax; parity with manual SDP within `atol=1e-5` |
| 2 | Padding mask correctness | 3 | `test_padding_mask_zeros_padded_positions` passes; same convention (`True = padded`) at module boundary for all three models |
| 3 | Positional embedding threaded | 1 | `test_positional_matters` passes; shuffled tokens produce different outputs |
| 4 | Pre-LN + GELU + `d_ff = 4·d_model` | 2 | `TransformerBlock` applies `LayerNorm` before each sublayer; `d_ff` is `4 * d_model` by default |
| 5 | Weight init | 1 | `std=0.02` on `Linear` and `Embedding`; ones/zeros on `LayerNorm` |
| 6 | Mean pooling respects padding | 2 | Pool denominator excludes pad positions; `test_padding_mask_zeros_padded_positions` pins it |
| 7 | AdamW + warmup cosine + clip | 2 | `AdamW(wd=0.01)`; warmup fraction ≈ 10%; `clip_grad_norm_(1.0)` logged pre-clip |
| 8 | Scratch-vs-reference parity | 3 | Within 3 percentage points at matched hyperparameters |
| 9 | DistilBERT fine-tune | 2 | Uses DistilBERT's own tokenizer; converts `padding_mask → attention_mask`; reaches the dataset's accuracy floor |
| 10 | Report | 1 | Three accuracy numbers, three param counts, three wall-clocks, confusion matrices, 10 predictions, reflection answers |

A submission missing item 2 (padding mask) cannot score above 14 — mask bugs are the single most common cause of a Transformer that "trains" but produces subtly wrong results.

## 5. Common mistakes

- **Mask convention inverted.** `nn.TransformerEncoderLayer`'s `src_key_padding_mask` is `True = padded`. `F.scaled_dot_product_attention`'s `attn_mask` is "`True = attended-to`". Mixing them silently zeroes out all real tokens and the model learns to predict the majority class. Fix: convert at the attention module boundary and pin with `test_padding_mask_zeros_padded_positions`.
- **Forgetting `padding_idx=pad_id` on `nn.Embedding`.** The pad embedding drifts during training; when the mean pool denominator is wrong (see next bullet), the pooled representation gets contaminated.
- **Mean pool over all positions including pads.** Divides by `L` instead of `(L - num_pads)`. A short-text example pooled into a long-padded batch gets compressed toward zero. Fix: use the mask when summing *and* when counting.
- **Post-LN with no warmup.** The classic "attention is all you need" post-LN requires a long warmup (chapter 08). Pre-LN trains stably without; the exercise specifies pre-LN for exactly this reason. If you go post-LN, budget a warmup of 10% of steps and expect occasional loss divergence in the first 100 steps.
- **Reusing your BPE tokenizer with DistilBERT.** Vocab mismatch silently maps every token to `[UNK]`. The DistilBERT fine-tune uses DistilBERT's own tokenizer end-to-end; the comparison is between architectures and training data, not tokenisation.
- **Padding to global `max_len`** instead of batch-max. 3× the compute, same accuracy. The batch-max approach is dynamic but needs the collate function to compute it — easy to forget on a copy-paste.
- **`scheduler.step()` per epoch** instead of per batch when using the warmup+cosine. Only the warmup fires; the cosine never kicks in. Symptom: training is stable but accuracy plateaus well below the floor.
- **No `clip_grad_norm_`** on the from-scratch model. In early steps the attention weights produce gradient spikes; without clipping, one bad step takes a weight far from its init region and the model never recovers. The exercise's test for finite clip-norm is a sanity check that the call fires at all.
- **`torch.save(model)` for Hugging Face models.** Use `model.save_pretrained(path)` instead; it writes the config + tokenizer + weights in the format `from_pretrained` expects.
- **Training the reference `nn.TransformerEncoder` with default PyTorch weight init.** PyTorch's default Xavier init for Linear is different from `std=0.02`; parity tests will fail by ~2–4 points because *initialisation* differs, not the block. Apply the same `_init_weights` to the reference for a fair comparison.

## 6. References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. <https://arxiv.org/abs/1706.03762>
- Xiong, R. et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML. <https://arxiv.org/abs/2002.04745> — justifies pre-LN as the stable-training variant.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. <https://arxiv.org/abs/1711.05101> — AdamW.
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. <https://arxiv.org/abs/1606.08415>
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. <https://arxiv.org/abs/1910.01108>
- PyTorch documentation — `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`, `F.scaled_dot_product_attention`, `torch.optim.lr_scheduler.LambdaLR`. <https://pytorch.org/docs/stable/>
- Hugging Face `transformers` documentation — `AutoTokenizer`, `AutoModelForSequenceClassification`. <https://huggingface.co/docs/transformers>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3`, `MAP 1.6`. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft`, `ML06:2023 AI Supply Chain Attacks` (relevant to the fine-tune step: a pretrained checkpoint is an AI supply-chain input and should be hash-pinned where possible). <https://owasp.org/www-project-machine-learning-security-top-10/>
