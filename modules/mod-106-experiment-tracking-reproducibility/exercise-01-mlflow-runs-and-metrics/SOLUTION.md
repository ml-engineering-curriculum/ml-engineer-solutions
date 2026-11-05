# exercise-01 — MLflow Runs and Metrics · SOLUTION

Paired exercise: `lessons/mod-106-experiment-tracking-reproducibility/exercises/exercise-01-mlflow-runs-and-metrics.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to instrument a training script with [MLflow](https://mlflow.org/docs/latest/index.html) so every run captures: parameters, metrics over training steps, the final fitted model, the input data signature, and the code version. The goal is not "MLflow installed"; the goal is **every number in the README of a future model card is traceable to one MLflow run**.

Design priorities, in order:

1. **One run = one trainable artifact.** A run is created *once* at the start of training and closed at the end. Nested runs are used only for sub-steps that themselves train a model (e.g., K-fold CV where each fold is a child run).
2. **Params logged *before* training, metrics logged *during*, artifacts logged *after*.** This ordering is what makes a crashed run still useful — you can see what was attempted even if it never produced a model.
3. **Explicit `experiment` and `run_name`.** Falling back to the `Default` experiment loses the dataset/task scoping that makes the UI usable after a few dozen runs.
4. **Autolog + explicit logs, not autolog alone.** Autolog is a floor, not a ceiling. Business metrics (precision at a product-specified recall threshold, calibration) are not in autolog's menu and must be logged explicitly.
5. **Provenance is logged as tags**, not params. Git SHA, data hash, and Python version are *context*, not knobs — tags are the right field for "search for all runs on commit abc123," and params are reserved for things a sweep would vary.

This design sits inside the NIST AI RMF `MAP 4.1` / `MEASURE 2.3` expectation that every model has a reproducible training record, and the OWASP ML Top 10 `ML06:2023 AI Supply Chain Attacks` control motivation that the training process itself is an auditable step in the supply chain.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Local `file:./mlruns` for dev, remote tracking URI via env for CI | Always-remote | Keeps the inner dev loop offline-fast; prod runs still land in the shared store |
| `mlflow.start_run(...)` as a context manager | Explicit `start_run` / `end_run` | A crash inside the `with` block still ends the run and records failure status |
| Explicit `log_metric("val_loss", v, step=epoch)` | `mlflow.autolog()` only | Autolog logs the framework's metrics; the metric the product owner reads must be logged by name |
| Signature + input_example on `log_model` | Bare `log_model` | The signature is what the server uses to reject malformed requests at inference time |
| Git SHA + `uname -a` logged as tags | As params | Tags are provenance; params are hyperparameters. Mixing them makes the sweep UI unreadable |
| One experiment per task/dataset pair | One experiment per project | 500 runs per experiment stay searchable; 5,000 do not |

## 2. Worked implementation

### 2.1 Project layout

```
tracking/
├── __init__.py
├── config.py          # tracking URI, experiment name, run name
├── provenance.py      # git SHA, data hash, python version
├── train.py           # the trainable; sklearn here, but shape identical for torch
└── run.py             # entrypoint; owns the run lifecycle
```

### 2.2 `config.py` — where the run gets its identity

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: str
    experiment_name: str
    run_name: str | None  # None lets MLflow autogenerate

    @classmethod
    def from_env(cls) -> "TrackingConfig":
        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
            experiment_name=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", "churn-baseline"
            ),
            run_name=os.environ.get("MLFLOW_RUN_NAME"),
        )
```

Reading the tracking URI from the env is what lets a single script land runs on a laptop today and on a Databricks/MLflow server tomorrow with no code change. The `experiment_name` default is not `Default` — the moment a second model enters the picture, a shared `Default` experiment becomes a graveyard.

### 2.3 `provenance.py` — what the model was trained *on*, not *with*

```python
import hashlib
import platform
import subprocess
from pathlib import Path


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        sha = out.stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_tags(data_path: Path) -> dict[str, str]:
    return {
        "git.sha": git_sha(),
        "data.path": str(data_path),
        "data.sha256": file_sha256(data_path),
        "python.version": platform.python_version(),
        "platform": platform.platform(),
    }
```

`-dirty` on the git SHA is deliberate: a run trained from uncommitted code must *announce that fact* in the tracking server, otherwise "it reproduces on commit abc123" is a lie waiting to happen. The data SHA-256 is what links a run to a dataset partition without having to trust a filename.

### 2.4 `train.py` — plain training, no MLflow coupling

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainParams:
    n_estimators: int
    max_depth: int
    learning_rate: float
    random_state: int


@dataclass
class TrainResult:
    model: GradientBoostingClassifier
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    per_iteration_val_loss: list[float]
    feature_names: list[str]
    x_example: pd.DataFrame
    y_example: pd.Series


def train(df: pd.DataFrame, target: str, params: TrainParams) -> TrainResult:
    y = df[target].astype("int8")
    x = df.drop(columns=[target])

    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=params.random_state,
    )
    model = GradientBoostingClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        random_state=params.random_state,
    )
    model.fit(x_tr, y_tr)

    per_iter = []
    for p in model.staged_predict_proba(x_val):
        per_iter.append(float(log_loss(y_val, p[:, 1], labels=[0, 1])))

    train_proba = model.predict_proba(x_tr)[:, 1]
    val_proba = model.predict_proba(x_val)[:, 1]

    return TrainResult(
        model=model,
        train_metrics={
            "train_logloss": float(log_loss(y_tr, train_proba, labels=[0, 1])),
            "train_auc": float(roc_auc_score(y_tr, train_proba)),
        },
        val_metrics={
            "val_logloss": per_iter[-1],
            "val_auc": float(roc_auc_score(y_val, val_proba)),
        },
        per_iteration_val_loss=per_iter,
        feature_names=list(x_tr.columns),
        x_example=x_tr.head(5).copy(),
        y_example=y_tr.head(5).copy(),
    )
```

Training is a pure function of `(df, params)`. No MLflow import in this file — that way `train(...)` is unit-testable with no tracking server, and the tracking layer can be swapped (W&B, Neptune, in-house) without rewriting the model code.

### 2.5 `run.py` — the single owner of the run lifecycle

```python
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature

from .config import TrackingConfig
from .provenance import runtime_tags
from .train import TrainParams, train


def run(data_path: Path, params: TrainParams, cfg: TrackingConfig) -> str:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    df = pd.read_parquet(data_path)

    with mlflow.start_run(run_name=cfg.run_name) as active:
        mlflow.set_tags(runtime_tags(data_path))
        mlflow.log_params({
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "learning_rate": params.learning_rate,
            "random_state": params.random_state,
            "rows": len(df),
        })

        result = train(df, target="churned_90d", params=params)

        for step, val in enumerate(result.per_iteration_val_loss):
            mlflow.log_metric("val_logloss_per_iter", val, step=step)
        mlflow.log_metrics(result.train_metrics)
        mlflow.log_metrics(result.val_metrics)

        signature = infer_signature(result.x_example, result.y_example)
        mlflow.sklearn.log_model(
            sk_model=result.model,
            artifact_path="model",
            signature=signature,
            input_example=result.x_example,
        )

        return active.info.run_id


if __name__ == "__main__":
    run(
        data_path=Path("data/features/orders.parquet"),
        params=TrainParams(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        cfg=TrackingConfig.from_env(),
    )
```

Three details earn their complexity:

- The context manager (`with mlflow.start_run(...) as active`) guarantees run termination with `FAILED` status if training raises. A learner who writes `start_run()` / `end_run()` inline will end up with "running forever" zombie runs.
- `log_metric(..., step=epoch)` (plural `log_metrics` is for a batch of final values) is what populates the step-vs-value chart in the UI. Logging only the final loss wastes the richest signal MLflow can capture.
- `infer_signature` is called on the actual training batch, not a synthesised example. A signature that disagrees with the data the model was trained on is worse than no signature — it will greenlight malformed requests at serving time.

### 2.6 Cross-validation with nested runs

```python
from sklearn.model_selection import StratifiedKFold


def run_cv(data_path, params, cfg, n_splits=5):
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    df = pd.read_parquet(data_path)
    y = df["churned_90d"].astype("int8")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=params.random_state)

    with mlflow.start_run(run_name=f"cv-{n_splits}fold") as parent:
        mlflow.log_params({"n_splits": n_splits, **params.__dict__})
        fold_aucs = []
        for fold, (tr, va) in enumerate(skf.split(df, y)):
            with mlflow.start_run(nested=True, run_name=f"fold-{fold}"):
                result = train(df.iloc[tr].reset_index(drop=True), "churned_90d", params)
                mlflow.log_metrics(result.val_metrics)
                fold_aucs.append(result.val_metrics["val_auc"])
        mlflow.log_metric("cv_auc_mean", float(np.mean(fold_aucs)))
        mlflow.log_metric("cv_auc_std", float(np.std(fold_aucs)))
        return parent.info.run_id
```

The parent run holds the aggregate; each child holds the fold. The UI groups them automatically, and a reviewer can click "best fold" without losing the summary.

## 3. Validation steps

Run these in order; each exists to catch one real failure mode.

1. **Static check** — `python -c "from tracking.run import run"`. Catches missing `mlflow` / `sklearn` / typos before the first long job.
2. **Smoke run, local URI** — `MLFLOW_TRACKING_URI=file:./mlruns python -m tracking.run`. Open `mlflow ui` and confirm: one run, parameters present, `val_logloss_per_iter` has N points (not 1), `model/` artifact logged with signature.
3. **Re-run identical** — rerun with the same `random_state` and same data. The new run must appear with identical params, identical `data.sha256` tag, and metrics within numerical tolerance (sklearn is deterministic for the same seed + data).
4. **Dirty-tree run** — `touch dirty.txt && git add dirty.txt && python -m tracking.run`. The run's `git.sha` tag must end in `-dirty`.
5. **CV smoke** — run `run_cv` with `n_splits=3`. UI must show one parent run and three nested child runs; parent's `cv_auc_mean` must equal the mean of child `val_auc` values.
6. **Model round-trip** — `mlflow.sklearn.load_model(f"runs:/{run_id}/model").predict(x_example)` must return an array of length 5 with values in `[0, 1]` after passing through `predict_proba`. Catches a broken signature or model format.
7. **Remote URI parity** — run against a real tracking server (`MLFLOW_TRACKING_URI=http://...`). The *only* difference between the two runs should be where the artifacts live.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Run lifecycle | 2 | `start_run` used as a context manager; crashes still end the run as `FAILED` |
| 2 | Experiment scoping | 1 | Not `Default`; experiment name is dataset/task scoped and set from config |
| 3 | Params | 2 | Every knob the next sweep would vary is `log_param`'d; no secrets, no booleans masquerading as ints |
| 4 | Metrics over steps | 3 | Per-iteration metric uses `step=`; the training-curve chart in the UI is populated, not a single point |
| 5 | Tags for provenance | 3 | Git SHA (with `-dirty` detection), data SHA-256, python & platform versions logged as tags |
| 6 | Model artifact + signature | 3 | `log_model` with `signature` inferred from the real training batch and an `input_example` |
| 7 | Separation of concerns | 2 | Training function has no `import mlflow`; tracking lives in the entrypoint |
| 8 | Nested runs for CV | 2 | K-fold produces one parent + K children; parent holds aggregate metrics |
| 9 | Env-driven config | 2 | Tracking URI, experiment name, run name all from env; no hard-coded tracking server |

Items 1 and 5 are mandatory: a submission missing either cannot score above 12. "Run crashed and the UI shows nothing" and "the run has no git SHA" are both blockers to the downstream reproducibility exercises.

## 5. Common mistakes

- **Autolog alone.** `mlflow.autolog()` logs the framework's native metrics, which often does not include the metric the model will actually be judged on (e.g., precision at a product-specified recall, business-defined cost). Autolog is additive; keep it, but still log business metrics explicitly by name.
- **Logging metrics without `step=`.** The UI collapses to the latest value, and the training curve — the first thing anyone looks at — is a single dot. The step argument is the signal.
- **Hard-coded `mlflow.set_tracking_uri("http://prod-mlflow:5000")`.** Now the test suite writes to prod. Env-drive the URI.
- **Secrets as params.** API tokens, database URLs, and S3 keys end up in the params panel, which is world-readable in most team setups. Params are for hyperparameters; secrets go through the env or a secret manager.
- **Not calling `end_run` on crash.** Using `start_run()` without `with` leaves runs in `RUNNING` state forever when training raises. Use the context manager.
- **Signature inferred from synthetic data.** `infer_signature(np.random.rand(1, n_features), ...)` greenlights a schema the model was never actually trained on. Infer from the real training batch.
- **One experiment for everything.** After 500 runs, the UI's search gets slow and humans stop using it. Scope experiments by task/dataset, not by project.
- **Logging a `_dirty.txt` into `mlruns/`.** The tracking directory is for MLflow's internal state. Artifacts belong on the run via `log_artifact(...)`.
- **Forgetting to `log_model`.** A run with metrics but no model is a benchmark, not a deliverable. The deployment pipeline reads from `runs:/<id>/model` or the registry — both require the model to have been logged.

## 6. References

- MLflow — Tracking concepts, API reference, and the file/URI layout for the default tracking store. <https://mlflow.org/docs/latest/tracking.html>
- MLflow — Model signatures and input examples (what `infer_signature` produces and how serving consumes it). <https://mlflow.org/docs/latest/model/signatures.html>
- NIST AI Risk Management Framework 1.0 — `MAP 4.1` (training record completeness) and `MEASURE 2.3` (metric documentation). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML06:2023 AI Supply Chain Attacks` as the motivation for logging provenance (code SHA, data hash) on every run. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` for the broader threat model; tracked runs are the forensic record when you need to answer "which model shipped, trained from what code and data?" <https://atlas.mitre.org/>
