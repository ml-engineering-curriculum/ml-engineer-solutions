# exercise-03 — XGBoost or LightGBM Tuning With Optuna · SOLUTION

Paired exercise: `lessons/mod-104-classical-ml-modeling/exercises/exercise-03-xgboost-or-lightgbm-tuning.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to tune a gradient-boosted tree model with Optuna and produce a tuning run that another engineer could audit and resume. The reference solution picks **LightGBM** as the model (the walkthrough notes where XGBoost differs) and structures the tuning around three ideas that matter more than any particular hyperparameter choice:

1. **Nested validation.** An outer train/test split that is **never** used by the tuner; an inner K-fold used by Optuna to pick hyperparameters. The headline number is reported on the outer split.
2. **Pruning + early stopping.** Optuna's `MedianPruner` stops unpromising trials mid-flight; LightGBM's `early_stopping_rounds` stops the booster on the first fold that plateaus. Together they cut the search cost by 5–10× on typical datasets.
3. **Persisted study.** The Optuna study is stored in SQLite so the run can be killed, resumed, and inspected after the fact. A tuning run that cannot be resumed is an experiment that cannot be reproduced.

The tuning *results* (final hyperparameters, best metric) are the least interesting output. The two artefacts that matter are:

- The **study database** — every trial, every intermediate metric, timestamps.
- The **report** — best params, outer-split metric, search space, pruned-trial count, wall-clock, and a hash of the input data.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| Nested CV (outer split + inner K-fold) | Single K-fold with best-trial score as headline | Reporting the CV score Optuna optimised over is a tuning-leakage bug; the headline number must come from unseen data |
| Optuna TPE sampler | Grid / random search | TPE converges with far fewer trials in >6-dimensional spaces; grid search is a dead end past ~4 dimensions |
| `MedianPruner` | No pruning | Cuts wasted compute by stopping bottom-half trials early; critical when each fit takes minutes |
| LightGBM histogram booster + `early_stopping_rounds` | Fixed `num_boost_round` | Per-fold early stopping finds the right `n_estimators` for each trial; a fixed round count systematically under- or over-fits |
| SQLite study storage | In-memory study | Resumable; inspectable with `optuna-dashboard`; survives SIGKILL |
| Fixed seed on sampler, pruner, splitter | Default (random each run) | Two runs with the same code and data produce the same trial trajectory and the same best params |
| Objective returns the **mean** of inner-fold scores | Max/min/single fold | Mean is more stable; a single fold score is one unlucky draw away from picking a bad configuration |

## 2. Worked implementation

### 2.1 Project layout

```
tuning/
├── __init__.py
├── space.py           # the search space (one place to edit)
├── objective.py       # one trial = one hyperparameter configuration
├── study.py           # study creation / loading, pruner + sampler
├── report.py          # final metric on the outer split, persisted
└── run.py             # CLI entry point
```

### 2.2 Search space

```python
# tuning/space.py
import optuna


def suggest(trial: optuna.Trial) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        # Learning rate / depth trade-off
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # Regularisation
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # Subsampling (adds randomisation, reduces variance)
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "random_state": 42,
    }
```

Key points:

- **One file, one function.** Changing the search space means editing one place, not grepping through the codebase.
- **Log-scale on `learning_rate`, `reg_alpha`, `reg_lambda`.** Their effect is multiplicative; linear sampling spends most of its budget in a flat region.
- **`num_leaves` bounded by `max_depth` implicitly** — LightGBM allows `num_leaves > 2^max_depth` combinations that the tree can never reach; the pruner will drop them quickly.
- **No trial-level `random_state` suggestion.** The *seed* is not a hyperparameter; it is part of the reproducibility contract.

For XGBoost, swap the names (`eta`/`max_depth`/`min_child_weight`/`gamma`/`subsample`/`colsample_bytree`) and the objective (`binary:logistic`). The structure is identical.

### 2.3 Objective — inner CV with pruning and early stopping

```python
# tuning/objective.py
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from .space import suggest


def make_objective(X, y, n_splits: int = 5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        fold_scores: list[float] = []
        for fold_idx, (tr, va) in enumerate(cv.split(X, y)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            dtrain = lgb.Dataset(X_tr, y_tr)
            dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, metric="binary_logloss", valid_name="valid_0"
            )
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dvalid],
                valid_names=["valid_0"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    pruning_callback,
                ],
            )
            y_pred = booster.predict(X_va, num_iteration=booster.best_iteration)
            fold_scores.append(log_loss(y_va, y_pred))
        return float(np.mean(fold_scores))

    return objective
```

What each line is actually doing:

- **`LightGBMPruningCallback`** reports the valid score back to Optuna every boosting round. The `MedianPruner` compares against previously-completed trials at the same round and kills the trial if it's below median. This is where the 5–10× speedup comes from.
- **`early_stopping(50)`** stops the booster inside this trial once the valid score has not improved for 50 rounds. Without it, every trial runs for 5000 rounds and the slowest hyperparameter configurations dominate wall time.
- **`num_iteration=booster.best_iteration`** uses the checkpoint from before overfit started. Prediction with the *final* iteration includes the overfit rounds.
- **Return the mean** of the five fold log-losses. Returning a single fold's score makes the objective noisier than the signal in most hyperparameter changes.

### 2.4 Study — sampler, pruner, storage

```python
# tuning/study.py
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def make_study(name: str, storage_url: str) -> optuna.Study:
    return optuna.create_study(
        study_name=name,
        storage=storage_url,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True, group=True),
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=200, interval_steps=50),
        load_if_exists=True,
    )
```

Decisions baked into the study:

- **`n_startup_trials=20` on the sampler**. The first 20 trials are random — TPE needs a prior before it can bias the search. Starting TPE from trial 1 is a common bug that produces biased, slow exploration.
- **`n_warmup_steps=200` on the pruner**. Each trial must reach 200 boosting rounds before it is eligible for pruning. Pruning from round 1 kills slow-starting configurations that would eventually have been best.
- **`multivariate=True, group=True`** on TPE. The sampler then conditions suggestions on the full joint distribution, not independently per parameter — critical when `learning_rate` and `num_leaves` are correlated.
- **`load_if_exists=True`**. Resuming the same study from SQLite picks up where SIGKILL left off. The trial counter is preserved and TPE's prior keeps growing across runs.

### 2.5 Orchestrator and outer-split evaluation

```python
# tuning/run.py
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from .objective import make_objective
from .study import make_study
from .space import suggest


def _data_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="artifacts/exercise-03")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--study-name", default="lgbm-tuning")
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    y = df[args.target].astype("int64")
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{out_dir / 'optuna.db'}"

    study = make_study(args.study_name, storage_url)
    objective = make_objective(X_train, y_train)

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    wall_s = time.time() - t0

    # Refit best params on full train with early stopping against a held-out slice.
    best_params = {**study.best_trial.params, "objective": "binary", "metric": "binary_logloss",
                   "verbosity": -1, "random_state": 42}
    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    dtrain = lgb.Dataset(X_fit, y_fit)
    dvalid = lgb.Dataset(X_holdout, y_holdout, reference=dtrain)
    booster = lgb.train(
        best_params, dtrain, num_boost_round=10_000,
        valid_sets=[dvalid], valid_names=["valid_0"],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    y_proba = booster.predict(X_test, num_iteration=booster.best_iteration)

    report = {
        "data_sha256": _data_sha256(args.data),
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(study.trials),
        "n_trials_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        "wall_clock_seconds": round(wall_s, 2),
        "best_cv_logloss": float(study.best_value),
        "best_params": study.best_trial.params,
        "best_iteration": int(booster.best_iteration),
        "test_logloss": float(log_loss(y_test, y_proba)),
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "optuna_db": str(out_dir / "optuna.db"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

The outer `test_*` metrics are the **only** numbers the learner should quote in a report or model card. The best CV value is useful only for comparing tuning runs against each other.

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **Import check** — `python -c "import tuning.run"`.
2. **Unit: pruning actually fires** — run `study.optimize(objective, n_trials=30)` on a tiny dataset; `study.trials_dataframe()["state"]` must contain at least one `PRUNED` row. If every trial completes, pruning is misconfigured (likely `n_warmup_steps` too high relative to `num_boost_round`).
3. **Unit: early stopping inside a trial** — fit one trial manually; `booster.best_iteration < num_boost_round`. Confirms the callback wiring.
4. **Unit: resume preserves trial count** — run the orchestrator with `--n-trials=5`, then re-run with `--n-trials=10` using the same `--out`. The second run's `n_trials_completed` must equal 10, not 15 or 5. (Optuna adds up to the requested total since the resumed trials carry over.)
5. **Unit: determinism** — with the same seed and same data, two consecutive `optimize(..., n_trials=20)` calls on a fresh study produce the same `best_params`. If they differ, the sampler seed is not being propagated — in practice the common cause is launching worker processes without the seed getting pickled through.
6. **Unit: no tuning leakage** — assert `X_test` and `y_test` do not appear in `make_objective`. The reviewer can verify this by grep; the test should also cover the function's signature.
7. **End-to-end** — point `--data` at a fixture, assert `report.json` contains all expected keys and `test_auc > 0.5` (if it isn't, your binary target is wrong).
8. **SQLite inspectability** — open `optuna.db` with `optuna-dashboard` or `optuna.load_study(...)`; confirm trial table has non-zero intermediate values.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Nested validation (outer split not used by tuner) | 3 | Headline metric comes from a split that `make_objective` never saw |
| 2 | Pruning configured and firing | 2 | `MedianPruner` or equivalent with sane `n_warmup_steps`; `n_trials_pruned > 0` on real runs |
| 3 | Per-trial early stopping | 2 | LightGBM/XGBoost `early_stopping_rounds` set; prediction uses `best_iteration` |
| 4 | Log-scale on multiplicative parameters | 1 | `log=True` on `learning_rate`, `reg_alpha`, `reg_lambda` |
| 5 | Deterministic sampler and splitters | 2 | Fixed seeds on sampler, pruner, splits; two runs with the same `--n-trials` produce the same best params |
| 6 | Persisted study | 2 | SQLite (or equivalent) storage; interrupted run resumable from the same URL |
| 7 | Objective averages across inner folds | 1 | Mean of fold scores returned, not a single fold |
| 8 | Refit on full train before test | 2 | Final booster trained on full training data (minus a tiny early-stop holdout), not on a single CV fold |
| 9 | Report contains lineage | 2 | `data_sha256`, seeds, package versions, wall-clock in `report.json` — satisfies NIST AI RMF `MAP 1.6` provenance |
| 10 | Search space in one file | 1 | `space.py` (or equivalent) is the sole location where parameter ranges are declared |
| 11 | Comparison vs a baseline | 2 | Report shows the tuned model beats an untuned LightGBM at defaults — otherwise the whole exercise has no ROI |

Partial credit: missing item 1 (nested validation) caps the submission at 12. That is the item that makes the headline number believable.

## 5. Common mistakes

- **Reporting `study.best_value` as the test score.** That number is the inner-CV score the tuner *optimised against*. It is the training-side number of the tuning layer, not a generalisation estimate. The outer held-out split is the only unbiased number.
- **No `n_startup_trials` on the TPE sampler.** TPE needs a seed of random exploration before its density estimate is useful; starting it from trial 1 biases the search toward whatever the first lucky point was.
- **Pruning from round 1.** With `n_warmup_steps=0` and `interval_steps=1`, slow-start configurations are killed before they get the chance to recover. Set `n_warmup_steps` to ~5–10% of `num_boost_round`.
- **Fixed `num_boost_round` with no early stopping.** Either every trial overfits past the valid minimum, or short runs are capped below the real optimum. Early stopping finds the right round count per trial.
- **In-memory study on a long search.** A single SIGKILL loses hours of compute. SQLite storage is a one-line change for full resumability.
- **Different `random_state` on each trial.** The seed is not a hyperparameter; varying it inflates the apparent search-space noise and makes TPE's job harder.
- **`predict(X_test)` without `num_iteration=booster.best_iteration`.** The booster kept training past the early-stop point; `predict` without specifying iteration uses the *last* iteration, which is the overfit one.
- **Tuning on one fold, reporting on another.** Picking "the best fold" as the headline is the most common subtle form of data leakage in tuning write-ups. Mean-of-folds, always.
- **Not refitting on full train.** Reporting the best-CV booster against test rows wastes 20% of the training data (one fold worth). Refit on full training data with early stopping on a small holdout.
- **Dropping pruned trials from the report.** The pruned-trial count is useful signal — it tells the reviewer how aggressive the pruner was. Keep it in the report.

## 6. References

- Optuna — User Guide, "Pruning Unpromising Trials" and "TPESampler". <https://optuna.readthedocs.io/en/stable/tutorial/>
- LightGBM — "Parameters Tuning" (official). <https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html>
- XGBoost — "Notes on Parameter Tuning" (official, for the sibling implementation). <https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html>
- scikit-learn — "Nested versus non-nested cross-validation" (official). <https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>
- NIST AI Risk Management Framework 1.0 — `MAP 1.6` (document data/model provenance) and `MEASURE 2.3` (performance under representative conditions). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` as the control motivation for keeping the outer test split out of the tuning loop. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` for why a persisted, resumable study database is also a supply-chain artefact that must be integrity-checked. <https://atlas.mitre.org/>
