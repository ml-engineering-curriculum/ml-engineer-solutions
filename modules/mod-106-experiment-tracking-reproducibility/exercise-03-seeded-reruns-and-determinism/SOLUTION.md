# exercise-03 — Seeded Reruns and Determinism · SOLUTION

Paired exercise: `lessons/mod-106-experiment-tracking-reproducibility/exercises/exercise-03-seeded-reruns-and-determinism.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to make a training script **bit-for-bit reproducible across reruns on the same machine, and metric-for-metric reproducible across machines**. These are two different problems with two different costs:

- **Same-machine reproducibility** is cheap: seed every RNG, pin dependencies, lock the data hash. Pay this cost always.
- **Cross-machine reproducibility** (two different GPUs, different driver versions, different OS) can require disabling optimizations that are worth 20–40% throughput. Pay this cost only when a downstream auditor needs to recompute a specific run; otherwise record enough environment metadata to *explain* a drift of ±1e-5.

Design priorities, in order:

1. **Seed every source of randomness, not just the obvious one.** `random`, `numpy`, framework RNG (torch/TF), CUDA, and `PYTHONHASHSEED` must all be set. Missing one is enough to break reproducibility — and it will be the one you forget.
2. **Set seeds in one place, early.** Before any import that lazily consumes an RNG (notably `transformers`, `tokenizers`, and anything that calls `os.urandom` at import time) runs.
3. **Make nondeterminism opt-in, not opt-out.** The training loop runs in deterministic mode by default; a flag enables the fast cuDNN algorithms when the user explicitly accepts the tradeoff.
4. **Record the environment, don't trust the code.** The seed is in the code; the CUDA version, driver, and hardware are not. Both halves have to be logged, or "it reproduces" is unverifiable.
5. **Verify reproducibility as a test.** Running the pipeline twice must produce identical final metrics (same-machine) within a documented tolerance. The test is the only thing that catches a regression when someone adds a non-seeded operation six months from now.

This matters beyond engineering taste: NIST AI RMF `MEASURE 2.3` expects the training record to be reproducible, and OWASP ML Top 10 `ML06:2023 AI Supply Chain Attacks` has "confirm the shipped model was trained by the recorded pipeline" as a defender's checkpoint — you cannot confirm what you cannot rerun.

| Choice | Alternative | Why this one |
|---|---|---|
| `set_global_seed(seed)` called once at script entry | Per-callsite `random.seed(seed)` | A single entry point is auditable; scattered seeds will miss one |
| `PYTHONHASHSEED` set via env, before interpreter starts | Set in `set_global_seed` | By the time Python calls user code, `hash(str)` has already been initialised |
| `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` | `torch.backends.cudnn.deterministic = True` alone | The cuDNN flag is a subset; the broader call is what makes torch raise on a non-deterministic op |
| Fixed `num_workers` + `worker_init_fn` for `DataLoader` | Default workers | Multiprocess workers each get their own RNG; without `worker_init_fn`, two runs shuffle augmentations differently |
| `generator=torch.Generator().manual_seed(seed)` on `DataLoader` | Rely on global torch seed | The DataLoader's shuffle RNG is separate from the global one |
| Log hardware/CUDA/driver as run tags | Assume CI uniformity | CI images drift; drivers upgrade; the tag is the forensic breadcrumb |
| Determinism **test** in CI | "Trust me, I seeded it" | Future PRs will add a non-seeded op; the test catches it the day it lands |

## 2. Worked implementation

### 2.1 Global seeding — one function, called once

```python
# repro/seed.py
import os
import random
import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every RNG this project touches. Call ONCE, as early as possible."""
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(
            f"PYTHONHASHSEED must be set in the environment to {seed} before "
            f"starting Python; currently {os.environ.get('PYTHONHASHSEED')!r}. "
            "Export it in your shell or entry script."
        )
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass
```

Three details earn their complexity:

- **`PYTHONHASHSEED` is checked, not set.** Python reads this env var during interpreter startup to seed `hash(str)`; by the time `set_global_seed` runs, it is too late. The check-and-raise forces the operator to set it in the shell or entrypoint script, where it still has effect.
- **`torch.use_deterministic_algorithms(True, warn_only=False)`.** This is strictly stronger than `cudnn.deterministic = True` — it also covers operations outside cuDNN (certain scatter, index_add, etc.). `warn_only=False` is deliberate: a non-deterministic op should **raise**, not log a warning the training loop happily ignores.
- **`CUBLAS_WORKSPACE_CONFIG`** is a requirement for deterministic cuBLAS matmul on CUDA 10.2+. Without it, `use_deterministic_algorithms(True)` raises at the first matmul. `:4096:8` is the documented value that preserves most performance.

### 2.2 Entry script — seed is set *before* heavy imports

```python
# repro/entry.py
import os
import sys

SEED = int(os.environ.get("SEED", "42"))
os.environ["PYTHONHASHSEED"] = str(SEED)

# If PYTHONHASHSEED was not already set, re-exec ourselves so the interpreter
# picks it up. This is the only way to make it stick on Python 3.
if sys.flags.hash_randomization and os.environ.get("_RESEEDED") != "1":
    os.environ["_RESEEDED"] = "1"
    os.execvpe(sys.executable, [sys.executable, "-m", "repro.entry"] + sys.argv[1:], os.environ)

from repro.seed import set_global_seed
set_global_seed(SEED)

from repro.train import train
train()
```

The `os.execvpe` dance is unusual, but it is the only reliable way in Python 3 to guarantee `PYTHONHASHSEED` was set *before* interpreter startup. A learner can simplify this by requiring operators to set the env var in their shell / Dockerfile; the re-exec is the belt-and-braces option.

### 2.3 DataLoader determinism (PyTorch example)

```python
# repro/data.py
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, seed: int, train: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        worker_init_fn=_worker_init_fn,
        generator=generator,
        drop_last=train,                # last partial batch perturbs the loss
        persistent_workers=True,        # new workers each epoch would reseed
    )
```

The subtle part: each DataLoader worker is a *separate process*. The main process seeded itself in `set_global_seed`, but the worker was `fork`/`spawn`ed afterward. Without `worker_init_fn`, two workers process the same batch with the same transforms applied — but *different* workers across epochs reshuffle augmentations non-deterministically. The `generator=` on the DataLoader itself handles the *order* of batches; `worker_init_fn` handles the randomness *inside* each sample.

### 2.4 Environment capture

```python
# repro/env.py
import os
import platform
import subprocess
import sys


def capture() -> dict[str, str]:
    env: dict[str, str] = {
        "python.version": sys.version.split()[0],
        "python.implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu": platform.processor() or "unknown",
        "env.PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset"),
        "env.CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
    }
    try:
        import torch
        env["torch.version"] = torch.__version__
        env["torch.cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["torch.cuda_version"] = torch.version.cuda
            env["torch.cudnn_version"] = str(torch.backends.cudnn.version())
            env["torch.gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        env["nvidia.driver"] = out.stdout.strip()
    except Exception:
        env["nvidia.driver"] = "unavailable"
    return env
```

These keys answer the question "why do my metrics differ by 1e-5 across machines?" — the answer is almost always "different driver" or "different cuDNN build." Log them on every run as MLflow tags.

### 2.5 Determinism self-test

```python
# repro/selftest.py
import json
import os
from pathlib import Path

from repro.seed import set_global_seed
from repro.train import train_once


def determinism_report(seed: int, out: Path) -> dict:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    results = []
    for trial in range(2):
        set_global_seed(seed)
        metrics = train_once(seed=seed, epochs=1, batch_size=32)
        results.append(metrics)

    diff = {
        k: abs(results[0][k] - results[1][k]) for k in results[0]
    }
    report = {"runs": results, "abs_diff": diff}
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    report = determinism_report(seed=42, out=Path("artifacts/determinism.json"))
    max_diff = max(report["abs_diff"].values())
    tol = 1e-6
    assert max_diff <= tol, f"Non-deterministic: max diff {max_diff} > {tol}\n{report}"
    print(f"deterministic within {tol}; max diff {max_diff:.2e}")
```

A learner can wire this script into CI as `pytest tests/test_determinism.py -q` and it becomes impossible to silently add a non-seeded op. When the tolerance needs to be relaxed (e.g., `1e-4` for a specific layer), it should be done in the test with a comment linking the cause — relaxing it silently is how determinism dies.

### 2.6 Logging seeds alongside the run

```python
import mlflow
from repro.env import capture
from repro.seed import set_global_seed

SEED = 42
set_global_seed(SEED)
with mlflow.start_run():
    mlflow.log_param("seed", SEED)
    mlflow.set_tags({"determinism": "strict", **capture()})
    ...  # train
```

The seed is a *param* (it would be swept in a robustness study); the environment is a *tag* (it is context, not a knob).

## 3. Validation steps

Each step exists to catch a specific failure mode.

1. **Env var check** — `python -c "import os; print(os.environ.get('PYTHONHASHSEED'))"` must print the seed, not `None`. If it prints `None`, the interpreter started without the seed and reruns will diverge on any code path that uses `dict`/`set` iteration order of strings (tokenization, feature ordering).
2. **Self-test, CPU** — `python -m repro.selftest` with CUDA disabled. The two runs must produce identical metrics bit-for-bit. If they don't, CPU-side nondeterminism is present (usually a missing `worker_init_fn`).
3. **Self-test, GPU** — same script with CUDA available. Metrics must match to 1e-6. If they don't, suspect: missing `CUBLAS_WORKSPACE_CONFIG`, missing `torch.use_deterministic_algorithms(True)`, or a non-deterministic op (the latter now raises instead of silently differing — good).
4. **Across-process match** — run the entry script twice in separate processes (not two trials in one process). Metrics must match. This catches state that leaked between trials inside one process (e.g., a global `torch.hub` cache being populated).
5. **Tolerance characterization** — rerun on a *different* machine (different GPU model or driver). Record the magnitude of divergence in the model card. This is informational, not a pass/fail, but it must exist.
6. **Non-determinism opt-in works** — rerun with `set_global_seed(seed, deterministic_torch=False)`. Expected: faster, and metrics may diverge. If metrics still match bit-for-bit, the "fast" path is also deterministic — unusual but possible depending on the model.
7. **Regression test** — intentionally break determinism by removing `worker_init_fn`. The self-test must fail. This proves the test is doing its job, and is a check a reviewer can perform at PR time.
8. **Environment delta check** — run twice, check that logged env tags are identical. Any difference (e.g., `torch.cudnn_version` changed) must be explainable — usually a dependency upgrade that should be captured in the lockfile.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Single seed entry point | 2 | `set_global_seed(seed)` called once; no stray `random.seed()` elsewhere |
| 2 | `PYTHONHASHSEED` handled | 2 | Set before interpreter startup; a run without it fails loudly, not silently |
| 3 | Framework-specific seeds | 3 | `torch.manual_seed`, `cuda.manual_seed_all`, `use_deterministic_algorithms(True)`; equivalents for other frameworks in use |
| 4 | DataLoader determinism | 3 | `generator=`, `worker_init_fn=`, fixed `num_workers`, `drop_last` on train |
| 5 | Environment capture | 2 | Python / torch / CUDA / cuDNN / GPU name / driver all logged as tags |
| 6 | Deterministic self-test | 3 | Two in-process trials produce identical metrics to a documented tolerance; test is part of CI |
| 7 | Opt-in nondeterminism | 1 | A flag selects fast/non-deterministic mode; the default is strict |
| 8 | Seeds logged on the run | 2 | `log_param("seed", ...)`; a reviewer can filter runs by seed |
| 9 | Cross-machine tolerance documented | 2 | Model card / README states the observed tolerance across machines and the mechanism (cuDNN, driver) |

Item 6 is mandatory: a submission without a deterministic self-test cannot score above 14. "It was deterministic when I ran it" is not evidence; the test is.

## 5. Common mistakes

- **Seeding only `numpy`.** Catches 30% of nondeterminism; the rest comes from framework-level RNG and DataLoader workers.
- **Setting `PYTHONHASHSEED` inside Python.** By the time `os.environ["PYTHONHASHSEED"] = ...` runs, the interpreter has already consumed the random hash seed. The env var must be set in the shell / Dockerfile / entry script re-exec.
- **`torch.backends.cudnn.deterministic = True` alone.** Only covers cuDNN ops. Use `torch.use_deterministic_algorithms(True, warn_only=False)` for the full guarantee.
- **Non-deterministic ops with `warn_only=True`.** The warning is ignored; metrics quietly diverge. Setting `warn_only=False` turns the warning into a loud `RuntimeError` at the exact operator.
- **Default DataLoader workers.** Without `worker_init_fn` and a `generator`, two training runs reshuffle augmentations differently even with everything else seeded.
- **`shuffle=True` with no `generator`.** The DataLoader's internal RNG is independent of the global torch RNG; it must be seeded via `generator=torch.Generator().manual_seed(seed)`.
- **Trusting the lockfile alone.** `requirements.txt` pins Python packages but not CUDA, cuDNN, or driver. Log these on the run; they move independently of pip.
- **No regression test.** Determinism rots. The day someone adds a `torch.scatter_add` on CUDA and does not set `CUBLAS_WORKSPACE_CONFIG`, metrics drift by 1e-4 and nobody notices until an auditor asks. A CI self-test is the one defense.
- **Equating "same seed" with "same metrics across hardware."** It isn't. Even with everything seeded, floating-point associativity breaks across GPU architectures. Document the observed tolerance; don't promise bit-for-bit across hardware.
- **Removing the determinism guard "temporarily" for throughput.** The flag exists for exactly this case. Use it, log the choice as a tag, and leave the strict mode as the default.

## 6. References

- PyTorch — Reproducibility notes (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, DataLoader workers). <https://pytorch.org/docs/stable/notes/randomness.html>
- NumPy — RNG seeding and the `default_rng` vs. legacy seeding tradeoffs. <https://numpy.org/doc/stable/reference/random/index.html>
- Python — `PYTHONHASHSEED` and hash randomization at interpreter startup. <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` on training records being reproducible, and `MANAGE 2.1` on retained lineage. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML06:2023 AI Supply Chain Attacks` as the motivation for confirming "the shipped model matches a recorded training run." <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0018 Backdoor ML Model` for the threat model that determinism helps detect: if the training recipe does not reproduce the shipped weights, something off-path happened. <https://atlas.mitre.org/>
