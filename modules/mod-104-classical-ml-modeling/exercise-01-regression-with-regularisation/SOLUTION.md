# exercise-01 — Regression With Regularisation · SOLUTION

Paired exercise: `lessons/mod-104-classical-ml-modeling/exercises/exercise-01-regression-with-regularisation.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to fit a regularised linear regression model on a tabular dataset with correlated features and a non-trivial number of weak predictors — the regime where plain OLS overfits and loses to a regularised variant.

The reference solution does three things in sequence:

1. **Build a `Pipeline` that standardises before fitting**, so the penalty is applied on comparable scales.
2. **Select the penalty strength by cross-validation**, not a fixed `alpha`.
3. **Compare Ridge, Lasso, and Elastic Net** on the *same* folds and choose with a rule that is explicit about the bias–variance trade-off.

Design priorities, in order:

1. **Honest error estimate** — the reported score comes from a held-out test split, never from the CV used to pick `alpha`. Reusing CV scores as the headline number is the single most common reporting error.
2. **Reproducibility** — one `random_state`, one `ColumnTransformer`, one entry point. Two runs on the same inputs produce identical coefficients.
3. **Interpretable artefacts** — coefficients are reported with their standardised feature names so a reviewer can tell which features the Lasso zeroed out.
4. **Auditable preprocessing** — the scaler fits on *train only*; test rows are transformed using train statistics. Fitting a scaler on the full dataset is a leakage bug that inflates the test R² and is invisible without reading the code.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| `Pipeline(StandardScaler → model)` | Scale once at the top of the script | Prevents leakage; cross-validation now actually simulates "unseen data" |
| `GridSearchCV` over `Pipeline(pre, model)` | `RidgeCV` / `LassoCV` / `ElasticNetCV` | Only `GridSearchCV` refits preprocessing *inside* each CV fold. The built-in path estimators run their internal alpha search *after* the enclosing `Pipeline` has already fit the scaler/encoder on the full training set — alpha is then chosen using validation rows whose statistics are already baked into the preprocessor. We trade the path algorithm's ~10× speed for a leak-free search |
| Nested evaluation (outer `train_test_split`, inner CV) | Single K-fold | The inner CV picks `alpha`; the outer split estimates generalisation. Collapsing them biases the reported score upward |
| Report standardised coefficients | Raw coefficients | On raw scale, a feature measured in millions looks like a "small" weight compared to one in `[0, 1]`; misleading |
| `random_state=42` on every splitter | Default (random each run) | Reruns are byte-identical; diffs in the manifest mean *your* code changed |
| Log-scale `alpha` grid `np.logspace(-3, 3, 50)` | Linear grid | Regularisation effect is multiplicative; log spacing covers the regime where the active set actually changes |

## 2. Worked implementation

### 2.1 Project layout

```
regularised_regression/
├── __init__.py
├── data.py            # dataset loading + train/test split
├── features.py        # ColumnTransformer: numeric scaling + one-hot
├── train.py           # fit RidgeCV / LassoCV / ElasticNetCV; pick winner
├── report.py          # coefficients, test metrics, selection rationale
└── run.py             # orchestration entry point
```

### 2.2 Dataset split

```python
# regularised_regression/data.py
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_and_split(path: str, target: str, test_size: float = 0.2) -> Split:
    df = pd.read_parquet(path)
    y = df[target].astype("float64")
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    return Split(X_train, X_test, y_train, y_test)
```

The split is drawn once at the top of the run. Everything downstream — scaler, CV, model, metrics — sees the same train/test boundary.

### 2.3 Feature pipeline

```python
# regularised_regression/features.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
```

`handle_unknown="ignore"` is defensive: a category that appears only in test rows becomes an all-zero vector rather than a run-time error. This matters once this pipeline is wrapped in a scoring service — see `mod-108` for the serving side.

### 2.4 Fit three regularisers on the same folds

```python
# regularised_regression/train.py
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

from .features import build_preprocessor


ALPHAS = np.logspace(-3, 3, 50)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


@dataclass
class Fitted:
    name: str
    pipeline: Pipeline
    best_alpha: float
    best_l1_ratio: float | None
    cv_mse: float


def _pipeline(X_train, estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(X_train)), ("model", estimator)])


def fit_all(X_train, y_train, n_splits: int = 5) -> dict[str, Fitted]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    searches = {
        "ridge": GridSearchCV(
            _pipeline(X_train, Ridge()),
            param_grid={"model__alpha": ALPHAS},
            cv=cv, scoring="neg_mean_squared_error", n_jobs=-1,
        ),
        "lasso": GridSearchCV(
            _pipeline(X_train, Lasso(max_iter=20_000, random_state=42)),
            param_grid={"model__alpha": ALPHAS},
            cv=cv, scoring="neg_mean_squared_error", n_jobs=-1,
        ),
        "enet": GridSearchCV(
            _pipeline(X_train, ElasticNet(max_iter=20_000, random_state=42)),
            param_grid={"model__alpha": ALPHAS, "model__l1_ratio": L1_RATIOS},
            cv=cv, scoring="neg_mean_squared_error", n_jobs=-1,
        ),
    }
    for search in searches.values():
        search.fit(X_train, y_train)

    def _wrap(name: str, search: GridSearchCV) -> Fitted:
        model = search.best_estimator_.named_steps["model"]
        l1 = float(model.l1_ratio) if name == "enet" else None
        return Fitted(
            name=name,
            pipeline=search.best_estimator_,
            best_alpha=float(model.alpha),
            best_l1_ratio=l1,
            cv_mse=float(-search.best_score_),
        )

    return {name: _wrap(name, s) for name, s in searches.items()}
```

Key points, each tied to a predictable failure mode:

- **Preprocessing sits *inside* the `Pipeline` that `GridSearchCV` clones per fold**, so the scaler and one-hot encoder are refit on each inner training fold. Reaching for `RidgeCV` / `LassoCV` / `ElasticNetCV` here looks cleaner but leaks: those estimators run their alpha search over an input matrix that was transformed by a preprocessor already fit on the entire outer training set — validation rows' statistics are baked in before alpha is chosen.
- All three `GridSearchCV` instances share the same `KFold` (same seed, same splits), so their `best_score_` values are directly comparable.
- `max_iter=20_000` is not paranoia — `Lasso` / `ElasticNet` will emit a `ConvergenceWarning` on real datasets with correlated features at default settings, and a non-converged fit returns silently-wrong coefficients.
- A fresh `build_preprocessor(X_train)` per pipeline keeps each `Pipeline` self-contained — the `ColumnTransformer` instance is not shared across searches.

### 2.5 Model selection rule

```python
# regularised_regression/train.py (continued)
def choose_winner(fitted: dict[str, Fitted]) -> str:
    mse = {name: f.cv_mse for name, f in fitted.items()}
    # Tie-break by model simplicity: Lasso > ENet > Ridge when within 1% of best.
    best_name = min(mse, key=mse.get)
    best = mse[best_name]
    for candidate in ("lasso", "enet", "ridge"):
        if mse[candidate] <= best * 1.01:
            return candidate
    return best_name
```

The CV MSE for each candidate is the leak-free score `GridSearchCV` already produced on the shared `KFold` — no second CV pass is needed, and introducing one would re-fit the pipeline outside of the search's folds for no benefit.

The tie-break rule is the opinionated part. "Within 1% of the best CV MSE → prefer the sparser model" trades a tiny bit of fit for a much smaller active-set and better auditability. Without a rule, learners default to "lowest number wins," which overfits the tie-break noise and produces a different model every retrain.

### 2.6 Reporting — coefficients that actually mean something

```python
# regularised_regression/report.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def report(fitted, winner: str, split, out_dir: Path) -> dict:
    pipe = fitted[winner].pipeline
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    coefs = pd.Series(model.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)

    y_pred = pipe.predict(split.X_test)
    out = {
        "winner": winner,
        "best_alpha": fitted[winner].best_alpha,
        "best_l1_ratio": fitted[winner].best_l1_ratio,
        "test_rmse": float(np.sqrt(mean_squared_error(split.y_test, y_pred))),
        "test_r2": float(r2_score(split.y_test, y_pred)),
        "n_features_total": int(len(feature_names)),
        "n_features_nonzero": int((coefs != 0).sum()),
        "top_10_coefficients": coefs.head(10).to_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    coefs.to_csv(out_dir / "coefficients.csv")
    return out
```

Why this report shape:

- **`test_*` metrics come from the *outer* split**, not the CV loop. This is what makes the number comparable to a production deployment.
- **Coefficients are ranked by absolute value** — the Lasso winners are the first few; the zeroed-out features appear at the end. A reviewer can read "which features matter" in two seconds.
- **`n_features_nonzero`** is the sparsity contract. If this collapses to a handful on your dataset, Lasso is doing its job; if it stays close to `n_features_total`, Ridge is probably the right pick.

### 2.7 Orchestrator

```python
# regularised_regression/run.py
import argparse
from pathlib import Path

from .data import load_and_split
from .train import fit_all, choose_winner
from .report import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", default="artifacts/exercise-01")
    args = parser.parse_args()

    split = load_and_split(args.data, args.target)
    fitted = fit_all(split.X_train, split.y_train)
    winner = choose_winner(fitted)
    report(fitted, winner, split, Path(args.out))


if __name__ == "__main__":
    main()
```

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **Import check** — `python -c "import regularised_regression.run"`. Catches dependency/version pinning issues before training.
2. **Unit: no-leakage split** — mock a dataset where `X[:, 0]` is standardised using only train rows; assert the resulting `mean` is ≈ 0 on train and **not necessarily** ≈ 0 on test. A buggy pipeline that fits the scaler on all data makes test mean exactly 0 — a telltale sign of leakage.
3. **Unit: determinism** — run `main()` twice on the same inputs and assert the `report.json` files are byte-identical.
4. **Unit: Lasso sparsity at high `alpha`** — set `ALPHAS = [1e4]` and refit on a small synthetic dataset; assert `n_features_nonzero == 0` (or very close). This proves the regulariser is actually active, not just the intercept.
5. **Unit: Ridge ≠ OLS when alpha > 0** — fit `Ridge(alpha=0.0)` and `Ridge(alpha=10.0)` on the same data; assert coefficient L2 norm strictly decreases. This catches "forgot to pass `alpha`" bugs.
6. **End-to-end** — point `--data` at a small fixture, assert `test_r2` is finite and `test_rmse > 0`, assert `coefficients.csv` has the expected `n_features` rows.
7. **Negative path: new category in test** — inject an unseen value in a categorical column of `X_test`; the pipeline must predict without raising (because of `handle_unknown="ignore"`).

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | No preprocessing leakage | 3 | Scaler and encoder fit inside the `Pipeline`; CV folds and test split never see each other's statistics |
| 2 | Honest test metric | 3 | Reported `test_*` comes from a held-out split that was **not** used in model selection |
| 3 | Reproducibility | 2 | A single `random_state` on splits and model; two runs produce identical `report.json` |
| 4 | `alpha` chosen by CV, not hand-picked | 2 | `RidgeCV` / `LassoCV` / `ElasticNetCV` or equivalent `GridSearchCV` is used |
| 5 | Three regularisers compared on same folds | 2 | Same `KFold` object (or same `cv=int + random_state`) passed to each |
| 6 | Convergence handled | 1 | `max_iter` raised for Lasso/ENet; no `ConvergenceWarning` at default |
| 7 | Coefficient report with feature names | 2 | `get_feature_names_out()` used; coefficients reported on standardised scale |
| 8 | Sparsity artefact | 1 | `n_features_nonzero` recorded alongside the headline metric |
| 9 | Selection rule documented | 2 | Written tie-break rule (e.g., "sparser wins within 1% of best CV"); not "lowest number blindly" |
| 10 | Model card lineage | 2 | `report.json` contains data path, random seed, package versions — satisfies NIST AI RMF `MAP 1.6` documentation of training data provenance |

Partial credit: a submission missing items 1, 2, or 3 cannot exceed 12 points — those are the items that silently inflate reported quality.

## 5. Common mistakes

- **Scaling before `train_test_split`.** The test fold now contains information from the train fold's standardiser (mean and std are leaky statistics). The reported R² is biased upward and the model surprises you in production.
- **Reporting CV score as the headline metric.** The CV score was *used to pick* `alpha`; reporting it is reporting a training-side number. The held-out split is the only unbiased estimate.
- **Comparing OLS to Ridge on `alpha=1.0`.** Default `alpha` on sklearn Ridge is 1.0, which is fine as a demo but not a fair comparison — always compare *tuned* models.
- **Lasso on unstandardised features.** The L1 penalty is scale-sensitive; a feature measured in cents will dominate the "which to zero out" decision purely by magnitude.
- **`ConvergenceWarning` ignored.** Lasso and ElasticNet with coordinate descent will silently return non-converged (and biased) coefficients when the solver stops early. Raise `max_iter` until the warning goes away.
- **Linear-spaced `alpha` grid.** `np.linspace(0.01, 10, 50)` spends 90% of its evaluations in a regime where the model is already overfit or already pathological. Use `np.logspace`.
- **One-hot without `handle_unknown="ignore"`.** First unseen category in production → `ValueError` at scoring time.
- **No seed on `KFold`.** Each run picks different folds → different `best_alpha` → different coefficients → "the model changed and nobody knows why."
- **Mixing up MSE and RMSE in the report.** Both are "lower is better," but they are not interchangeable. Pick one, label it, stick to it.

## 6. References

- scikit-learn — User Guide, "Generalized Linear Models" (Ridge, Lasso, Elastic Net) and "Cross-validation". <https://scikit-learn.org/stable/modules/linear_model.html>
- scikit-learn — User Guide, "Pipeline: chaining estimators" (the canonical pattern for leakage-free preprocessing). <https://scikit-learn.org/stable/modules/compose.html>
- NIST AI Risk Management Framework 1.0 — `MAP 1.6` (document data and model provenance) and `MEASURE 2.3` (evaluate model performance under representative conditions). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` motivates the leakage-free preprocessing control: any path from label/test rows into the training pipeline widens the attack surface. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` for the broader supply-chain threat model inside which a reproducible, auditable training run sits. <https://atlas.mitre.org/>
