# exercise-04 — Hyperparameter Sweep with Optuna · SOLUTION

Paired exercise: `lessons/mod-106-experiment-tracking-reproducibility/exercises/exercise-04-hyperparameter-sweep-with-optuna.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to run a hyperparameter sweep with [Optuna](https://optuna.readthedocs.io/en/stable/), log each trial as its own MLflow run, prune bad trials early, and surface the final best configuration in a reproducible way. The deliverable is not "best score I found"; it is **a sweep that can be rerun, compared, and extended** — which means the trials are tracked, the search space is version-controlled, and the trial that won has a provenance record identical to any other training run.

Design priorities, in order:

1. **One Optuna trial = one MLflow run.** The trial carries the sampled params; the run carries metrics, artifacts, and tags. Linking them via `trial.number` as a run tag is what makes the sweep forensically useful.
2. **Pruning is a first-class budget decision.** Running 1000 trials to completion when a median pruner would have killed 70% of them at epoch 2 is wasted compute. Tune the pruner before you tune the model.
3. **Parallel workers share a storage, not a Python process.** Optuna's RDB storage (SQLite for dev, Postgres for prod) lets N workers on N machines cooperate on the same study. The study name is the coordination key.
4. **Search space is in code, under version control.** A YAML search space that nobody reviews is where "we accidentally searched over 20 architectures nobody wanted" happens.
5. **The sweep has a budget *and* an early-stop condition.** Budget: `n_trials` or walltime. Early stop: "best value has not improved in K trials." Either alone wastes time; together they bound the worst case.
6. **Re-promote the winning trial through the same registration path.** Do not give the sweep its own ad-hoc save-to-disk route. The winner is logged exactly like any other training run and promoted via the registry (see exercise-02).

This is where the NIST AI RMF `MEASURE 2.3` "document the sensitivity of metrics to hyperparameters" expectation lands in practice — the sweep is the sensitivity analysis.

| Choice | Alternative | Why this one |
|---|---|---|
| `TPESampler` (default) | Random search | TPE converges faster on continuous spaces; random is a baseline, not a production strategy |
| `MedianPruner` with `n_warmup_steps` | No pruner | Up to 70% compute savings; `n_warmup_steps` prevents killing trials before they've had a fair shot |
| `HyperbandPruner` for very large sweeps | `MedianPruner` | Statistically stronger for `>` 100 trials; more complex to tune |
| RDB storage (`sqlite:///optuna.db` or Postgres) | In-memory | Survives worker crash; parallel workers can cooperate |
| One MLflow run per trial, via `trial.set_user_attr("mlflow_run_id", ...)` | Only log the winner | Preserves the full exploration record — the loser trials are the sensitivity analysis |
| `study.optimize(..., callbacks=[...])` for early stop | Fixed `n_trials` | Stops when no progress; saves the tail of the budget |
| Seed the sampler explicitly | Default (nondeterministic) | Reproducible sweep = reviewable sweep |
| Register only the winning trial to the model registry | Register every trial | Registry would fill with 500 near-identical models |

## 2. Worked implementation

### 2.1 Project layout

```
sweep/
├── __init__.py
├── space.py          # search space, in code
├── objective.py      # objective function (closure over data)
├── storage.py        # study storage + name, from env
└── run.py            # entrypoint; launches workers / collects the winner
```

### 2.2 Search space — Python, reviewable in a PR

```python
# sweep/space.py
import optuna


def suggest(trial: optuna.Trial) -> dict:
    """Define the search space. Every change here is a reviewable PR."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
    }
```

Three deliberate choices:

- `log=True` on `learning_rate` and `min_samples_leaf`. Both have effect over orders of magnitude; a linear search wastes half its budget in the uninteresting tail.
- `step=50` on `n_estimators`. The model is not sensitive to `n_estimators=137 vs. 150`; stepping prevents TPE from spending budget on indistinguishable points.
- Nothing is read from a YAML file. The search space *is* code; reviewers see it in PRs; `git log sweep/space.py` is the history of "what we have ever searched over."

### 2.3 Objective — pure function of trial + (fixed) data

```python
# sweep/objective.py
from dataclasses import dataclass
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .space import suggest


@dataclass(frozen=True)
class ObjectiveContext:
    df: pd.DataFrame
    target: str
    experiment_name: str
    seed: int


def make_objective(ctx: ObjectiveContext):
    y = ctx.df[ctx.target].astype("int8")
    x = ctx.df.drop(columns=[ctx.target])
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=ctx.seed,
    )
    mlflow.set_experiment(ctx.experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=False) as active:
            trial.set_user_attr("mlflow_run_id", active.info.run_id)
            mlflow.set_tag("optuna.study", trial.study.study_name)
            mlflow.set_tag("optuna.trial", str(trial.number))
            mlflow.log_params({**params, "seed": ctx.seed})

            model = GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                min_samples_leaf=params["min_samples_leaf"],
                subsample=params["subsample"],
                random_state=ctx.seed,
            )
            model.fit(x_tr, y_tr)

            per_iter = []
            for step, proba in enumerate(model.staged_predict_proba(x_val)):
                val_ll = float(log_loss(y_val, proba[:, 1], labels=[0, 1]))
                per_iter.append(val_ll)
                mlflow.log_metric("val_logloss", val_ll, step=step)

                # Report to Optuna for pruning
                trial.report(val_ll, step=step)
                if trial.should_prune():
                    mlflow.set_tag("optuna.pruned", "true")
                    raise optuna.TrialPruned()

            val_proba = model.predict_proba(x_val)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_proba))
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_logloss_final", per_iter[-1])
            return per_iter[-1]  # minimize

    return objective
```

Three details earn their complexity:

- **`trial.report(..., step=step)` + `trial.should_prune()`** inside the per-iteration loop. Without these, the pruner has no signal. With them, the median pruner can cut a hopeless trial after 10% of its budget.
- **Minimize `val_logloss`, not maximize `val_auc`.** Log-loss is strictly proper (every improvement is real); AUC can improve on a weakly-calibrated model that log-loss would reject. AUC is still logged as a secondary metric.
- **`seed` is a param, not a tag.** A follow-on seed-robustness sweep would vary it; params are what a sweep varies.

### 2.4 Storage — parallel-safe by default

```python
# sweep/storage.py
import os
import optuna


def get_study(direction: str = "minimize") -> optuna.Study:
    name = os.environ.get("OPTUNA_STUDY_NAME", "churn-baseline-sweep")
    url = os.environ.get("OPTUNA_STORAGE", "sqlite:///artifacts/optuna.db")
    sampler = optuna.samplers.TPESampler(seed=int(os.environ.get("OPTUNA_SEED", "42")))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    return optuna.create_study(
        study_name=name,
        storage=url,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )
```

- `load_if_exists=True` is what makes N parallel workers all `optuna.create_study(...)` with the same name and converge on *one* shared study rather than N parallel studies.
- SQLite is fine on one machine; for multiple hosts use Postgres. The code is identical — only the URL changes.
- `TPESampler(seed=...)` makes the *sampler* deterministic; the sweep itself is still slightly nondeterministic across parallel runs because the order of trial completion is nondeterministic. That's OK — it is documented and the winner is re-trained from scratch with a fixed seed anyway.

### 2.5 Entry point — budget + early stop + winner promotion

```python
# sweep/run.py
import os
from pathlib import Path
import mlflow
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from .objective import ObjectiveContext, make_objective
from .storage import get_study


EXPERIMENT = "churn-baseline-sweep"
MODEL_NAME = "churn-baseline"
SEED = 42
PATIENCE = 20
MAX_TRIALS = 200


def early_stop(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < PATIENCE:
        return
    best_at = study.best_trial.number
    last_trial = completed[-1].number
    if last_trial - best_at >= PATIENCE:
        study.stop()


def refit_winner(ctx: ObjectiveContext, best_params: dict) -> str:
    """Retrain the winning configuration from scratch and register it."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    y = ctx.df[ctx.target].astype("int8")
    x = ctx.df.drop(columns=[ctx.target])
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=ctx.seed,
    )
    model = GradientBoostingClassifier(random_state=ctx.seed, **best_params)
    model.fit(x_tr, y_tr)

    mlflow.set_experiment(ctx.experiment_name)
    with mlflow.start_run(run_name="winner-refit") as active:
        mlflow.log_params({**best_params, "seed": ctx.seed})
        mlflow.set_tag("sweep.winner", "true")
        signature = infer_signature(x_tr.head(5), y_tr.head(5))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_tr.head(5),
        )

        client = MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME)
        except mlflow.exceptions.RestException:
            pass
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=f"runs:/{active.info.run_id}/model",
            run_id=active.info.run_id,
        )
        client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
        return mv.version


def main() -> None:
    data_path = Path(os.environ.get("DATA_PATH", "data/features/orders.parquet"))
    df = pd.read_parquet(data_path)
    ctx = ObjectiveContext(
        df=df, target="churned_90d", experiment_name=EXPERIMENT, seed=SEED,
    )

    study = get_study(direction="minimize")
    study.optimize(
        make_objective(ctx),
        n_trials=MAX_TRIALS,
        callbacks=[early_stop],
        gc_after_trial=True,
    )

    print(f"best trial: #{study.best_trial.number} val_logloss={study.best_value:.4f}")
    print(f"params: {study.best_trial.params}")

    version = refit_winner(ctx, study.best_trial.params)
    print(f"registered as {MODEL_NAME} v{version} @challenger")


if __name__ == "__main__":
    main()
```

### 2.6 Running multiple workers

```bash
# Worker 1
OPTUNA_STORAGE=postgresql://user:pwd@host/optuna \
OPTUNA_STUDY_NAME=churn-sweep-2026-11-05 \
MLFLOW_TRACKING_URI=http://mlflow:5000 \
python -m sweep.run &

# Worker 2 (identical env, different host)
OPTUNA_STORAGE=postgresql://user:pwd@host/optuna \
OPTUNA_STUDY_NAME=churn-sweep-2026-11-05 \
MLFLOW_TRACKING_URI=http://mlflow:5000 \
python -m sweep.run &
```

Both workers see the same study; Optuna's storage coordinates which trial each worker picks next. Each worker still writes its own MLflow runs, so the UI shows `MAX_TRIALS` total runs regardless of worker count.

## 3. Validation steps

Each step exists to catch a specific failure mode.

1. **Static check** — `python -c "from sweep.run import main"`. Catches missing `optuna` / `mlflow` / typos.
2. **Smoke sweep** — `MAX_TRIALS=10 python -m sweep.run`. Expect 10 MLflow runs (some may be pruned, tagged `optuna.pruned=true`); exactly one `winner-refit` run; one new registered model version at `@challenger`.
3. **Pruning is active** — inspect the study: `study.trials[*].state`. Some must be `PRUNED`. If none are, `MedianPruner` is not wired up correctly, or `trial.report` is missing in the objective.
4. **Parallel safety** — launch two workers against the same SQLite file; both should reach `optuna.TrialState.COMPLETE` without raising `OperationalError: database is locked` or duplicating trial numbers. If they do, switch to Postgres or add `OPTUNA_STORAGE=postgresql://...`.
5. **Resumability** — kill a worker mid-trial. Restart. `study.trials` must show the killed trial as `FAIL` or `RUNNING` (depending on storage), and the new worker must pick up the next trial. No manual cleanup.
6. **Winner promotion** — confirm `models:/churn-baseline@challenger` resolves, `mlflow.pyfunc.load_model(...)` works, `predict` returns sensible outputs. This is the end-of-pipeline smoke test.
7. **Reproducible sampler** — rerun the sweep with the same `OPTUNA_SEED` and a fresh study name. Trial params at `trial.number=0..5` must be identical across runs (TPE is deterministic with a fixed seed; after that it reacts to observations).
8. **Early-stop fires** — set `PATIENCE=5` and a weakly-informative objective (e.g., constant return). The sweep must terminate before `MAX_TRIALS`; `study.stop()` is the mechanism.
9. **Negative: broken objective** — make the objective raise. The run for that trial must be `FAILED` in MLflow (not `RUNNING`), and the Optuna trial must be `FAIL`. This verifies the context managers are correctly wired.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Search space in code | 2 | `suggest()` is Python; reviewable in PRs; log-scaled where appropriate |
| 2 | One trial → one run | 2 | Every trial produces an MLflow run; `trial.number` logged as a tag; run status matches trial status |
| 3 | Pruning wired up | 3 | `trial.report(...)` + `should_prune()` inside the training loop; pruner configured; pruned trials have tag `optuna.pruned=true` |
| 4 | Parallel-safe storage | 2 | RDB storage URL via env; `load_if_exists=True`; two workers against the same study coexist |
| 5 | Deterministic sampler | 2 | `TPESampler(seed=...)` seeded from env; two runs with same seed have identical first N trials |
| 6 | Budget + early stop | 2 | Both an `n_trials` cap and a patience-based `study.stop()` callback |
| 7 | Correct objective direction | 2 | Minimize a strictly proper metric (log-loss) or maximize a defensible one; secondary metrics logged but not optimized |
| 8 | Winner refit + register | 3 | The winning config is retrained *from scratch* with a fixed seed, logged with signature, and registered; only the winner is registered |
| 9 | End-to-end reproducibility | 2 | Rerunning the sweep with the same seed, data hash, and git SHA produces the same winner (or a documented ±tolerance) |

Items 3 and 8 are mandatory. A submission without pruning spends compute to prove it can; a submission that registers every trial has built a landfill. Either failure caps the score at 13.

## 5. Common mistakes

- **No pruner.** The default `optuna.create_study()` gives you `NopPruner`; 1000 trials run to completion, 700 of which could have been killed after a handful of iterations. Always set a pruner, even if it is `MedianPruner` with default settings.
- **`trial.report` outside the training loop.** Pruning depends on *intermediate* values. If you only report the final metric, the pruner has nothing to do.
- **Maximizing AUC with no secondary check.** AUC can improve on a model that is systematically overconfident. Minimize a proper scoring rule (log-loss, Brier) and *monitor* AUC; don't optimize it.
- **In-memory study.** Lose your session, lose your sweep. Use RDB storage from day one.
- **SQLite with many parallel workers.** SQLite handles ~4 concurrent writers before contention dominates. Switch to Postgres for `>= 8` workers.
- **Every trial registered.** The registry fills with near-identical versions; the `@challenger` alias flips hundreds of times; auditing becomes impossible. Register only the winner, after a refit.
- **Using the winning trial's model directly.** Trial models were trained during search with potentially aggressive early stopping and cached state. Retrain from scratch with the best params so the registered artifact is clean.
- **Only `n_trials`, no early stop.** Budget is bounded, but a converged sweep still runs to completion. A patience-based callback saves the tail of the budget for the next sweep.
- **Changing the search space mid-sweep.** The TPE sampler conditions its distribution on past trials; changing the space invalidates them. Start a new study with a new name instead.
- **No `trial.set_user_attr("mlflow_run_id", ...)`.** Links the two systems; without it, finding the MLflow run for trial 37 requires a timestamp search.
- **Forgetting `gc_after_trial=True` on long sweeps.** Memory from pruned trials accumulates; after 200 trials the worker OOMs. Cheap to turn on, hard to debug when missing.

## 6. References

- Optuna — Samplers, pruners, storages, and distributed optimization. <https://optuna.readthedocs.io/en/stable/index.html>
- Optuna — `MedianPruner` + `trial.report` contract. <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html>
- MLflow — Tracking + Registry APIs (how a trial becomes a run and a run becomes a version). <https://mlflow.org/docs/latest/tracking.html>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` on documenting metric sensitivity to hyperparameters. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML06:2023 AI Supply Chain Attacks` for the motivation to keep the sweep auditable; a sweep without per-trial provenance cannot answer "which configuration shipped?" <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0018 Backdoor ML Model` as a reminder that a sweep is also a surface — a tampered search space or a tampered objective can select for an adversarial winner. <https://atlas.mitre.org/>
