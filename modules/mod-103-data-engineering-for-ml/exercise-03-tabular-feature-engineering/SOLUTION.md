# exercise-03 — Tabular Feature Engineering · SOLUTION

Paired exercise: `lessons/mod-103-data-engineering-for-ml/exercises/exercise-03-tabular-feature-engineering.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to turn a cleaned tabular dataset into a model-ready feature matrix *without introducing train/test leakage*. The reference solution builds a single `sklearn.pipeline.Pipeline` whose preprocessing step is a `ColumnTransformer`. The whole thing is `fit` on the training fold only, then `transform` is applied to validation and test.

Four design priorities, in descending importance:

1. **No leakage, by construction.** Every statistic (mean, median, quantile bin edges, target mean per category) is learned on the training split and applied verbatim to validation and test. The way to enforce this is "all transformations live inside a `Pipeline`, and `fit` is called exactly once, on `X_train`." Any statistic computed with a bare `pd.DataFrame.describe()` on the full dataset is leakage.
2. **Deterministic.** Pin numpy seeds where randomness exists (e.g., KFold for target encoding). Serialising the fitted pipeline with `joblib.dump` reproduces inference.
3. **Explicit column lists.** `ColumnTransformer` takes `numeric_features` and `categorical_features` as named lists, not `select_dtypes(...)`. A dtype-based selector silently reroutes columns when upstream types drift (e.g., an int column arrives as object on a backfill).
4. **Separate the feature engineering from the model.** The pipeline ends with a placeholder estimator the caller can swap. Feature engineering should not be re-fit each time you try a new model.

Decision summary:

| Choice | Alternative | Why |
|---|---|---|
| `ColumnTransformer` | per-column manual steps | One object; reproducible fit; serialisable |
| `SimpleImputer(strategy="median")` for numeric | mean | Robust to outliers; still a single statistic learned on train |
| `OneHotEncoder(handle_unknown="ignore")` for low-cardinality cat | target encoding | Simple, no leakage risk, works when cardinality < ~20 |
| K-fold target encoding for high-cardinality cat | plain mean target encoding | Plain target encoding leaks the label into features; K-fold is the canonical fix |
| `QuantileTransformer` + raw feature kept | drop the raw | Keeping both lets tree models use whichever representation is cleaner |
| Interaction features generated explicitly | `PolynomialFeatures(degree=2)` | Explicit interactions survive feature stores; `PolynomialFeatures` blows up dimensionality on anything with cardinality |

## 2. Worked implementation

Reference feature matrix for a churn-like classifier built on an `orders_enriched.parquet` with columns:

- Numeric: `order_total_usd`, `event_count`, `days_since_last_order`
- Low-card categorical (<20 levels): `currency`, `status`, `country_code`
- High-card categorical (>100 levels): `product_sku`
- Datetime: `created_at`
- Target: `churned_90d` (binary)

### 2.1 Column lists and type contract

```python
# features/columns.py
NUMERIC: list[str] = ["order_total_usd", "event_count", "days_since_last_order"]
LOW_CARD_CAT: list[str] = ["currency", "status", "country_code"]
HIGH_CARD_CAT: list[str] = ["product_sku"]
DATETIME: list[str] = ["created_at"]
TARGET: str = "churned_90d"

ALL_FEATURE_COLUMNS: list[str] = NUMERIC + LOW_CARD_CAT + HIGH_CARD_CAT + DATETIME
```

These lists are the feature-engineering contract. If a column is here, it must exist in the input; if an input column is not here, it is dropped. Both are explicit.

### 2.2 Datetime expansion

```python
# features/datetime_features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeExpander(BaseEstimator, TransformerMixin):
    """Turn each datetime column into [year, month, dow, hour, dayofyear_sin, dayofyear_cos]."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        out = {}
        for col in self.columns:
            dt = pd.to_datetime(X[col], utc=True)
            out[f"{col}_year"] = dt.dt.year.astype("int32")
            out[f"{col}_month"] = dt.dt.month.astype("int32")
            out[f"{col}_dow"] = dt.dt.dayofweek.astype("int32")
            out[f"{col}_hour"] = dt.dt.hour.astype("int32")
            doy = dt.dt.dayofyear.astype("float64")
            out[f"{col}_doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
            out[f"{col}_doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns:
            names.extend([
                f"{col}_year", f"{col}_month", f"{col}_dow",
                f"{col}_hour", f"{col}_doy_sin", f"{col}_doy_cos",
            ])
        return np.asarray(names)
```

Cyclical encoding (sin/cos of day-of-year) is included alongside the discrete year/month/dow so a linear model sees the "December is next to January" structure while a tree model can use the raw ordinals.

### 2.3 K-fold target encoder (high-cardinality safe)

```python
# features/target_encoder.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Out-of-fold mean target encoder.

    At fit time, each training row gets the target mean computed from the
    *other* folds — this prevents a row seeing its own label.

    At transform time (on validation / test), the global per-category mean
    (learned on all of train) is used.
    """

    def __init__(self, columns: list[str], n_splits: int = 5, smoothing: float = 10.0, random_state: int = 0):
        self.columns = columns
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def fit(self, X, y):
        y = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(y.mean())
        self.mappings_: dict[str, dict] = {}
        for col in self.columns:
            s = X[col].reset_index(drop=True)
            agg = pd.DataFrame({col: s, "y": y}).groupby(col)["y"].agg(["mean", "count"])
            smoothed = (
                (agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing)
                / (agg["count"] + self.smoothing)
            )
            self.mappings_[col] = smoothed.to_dict()
        return self

    def fit_transform(self, X, y):
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)
        out = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for col in self.columns:
            oof = np.full(len(X), np.nan, dtype="float64")
            for tr_idx, va_idx in kf.split(X):
                tr = pd.DataFrame({col: X[col].iloc[tr_idx].values, "y": y.iloc[tr_idx].values})
                agg = tr.groupby(col)["y"].agg(["mean", "count"])
                gmean = float(tr["y"].mean())
                smoothed = (
                    (agg["mean"] * agg["count"] + gmean * self.smoothing)
                    / (agg["count"] + self.smoothing)
                ).to_dict()
                oof[va_idx] = X[col].iloc[va_idx].map(smoothed).fillna(gmean).values
            out[f"{col}_te"] = oof

        self.fit(X, y)  # also learn the full-train mapping for later .transform()
        return out

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            out[f"{col}_te"] = X[col].map(self.mappings_[col]).fillna(self.global_mean_)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_te" for c in self.columns])
```

Two subtle but critical details:

- `fit_transform` and `transform` do *different* things. On train, each row is encoded using folds that exclude it (out-of-fold). On validation/test, the full-train mapping is used. This is the only safe pattern for high-cardinality target encoding.
- Smoothing (`smoothing=10`) pulls low-count categories toward the global mean. Without it, a SKU seen three times with three positives gets encoded as 1.0 — which is almost certainly overfit.

### 2.4 The pipeline

```python
# features/pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .columns import NUMERIC, LOW_CARD_CAT, HIGH_CARD_CAT, DATETIME
from .datetime_features import DatetimeExpander
from .target_encoder import KFoldTargetEncoder


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    low_card = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    high_card = KFoldTargetEncoder(columns=HIGH_CARD_CAT)

    datetime = DatetimeExpander(columns=DATETIME)

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC),
            ("low_cat", low_card, LOW_CARD_CAT),
            ("high_cat", high_card, HIGH_CARD_CAT),
            ("dt", datetime, DATETIME),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre
```

```python
# features/train.py
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .columns import ALL_FEATURE_COLUMNS, TARGET
from .pipeline import build_preprocessor


def main(input_parquet: str, model_out: str) -> None:
    df = pd.read_parquet(input_parquet)
    X = df[ALL_FEATURE_COLUMNS]
    y = df[TARGET].astype("int8")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    pipe = Pipeline(steps=[
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(random_state=0)),
    ])

    pipe.fit(X_train, y_train)
    print("train AUC:", pipe.score(X_train, y_train))
    print("test  AUC:", pipe.score(X_test, y_test))
    joblib.dump(pipe, model_out)
```

`pipe.fit(X_train, y_train)` is the only `.fit` call in the file. That line — and the fact that the preprocessor lives *inside* the pipeline — is what prevents leakage.

### 2.5 Example feature-matrix shape

Input: 100,000 rows × 8 raw feature columns. After `pre.transform`:

| Block | Columns in | Columns out |
|---|---|---|
| `num` | 3 | 3 (imputed + scaled) |
| `low_cat` | 3 (currency, status, country_code; say 4+5+40 levels) | 49 one-hots |
| `high_cat` | 1 (product_sku) | 1 target-encoded column |
| `dt` | 1 (created_at) | 6 (year, month, dow, hour, sin, cos) |

Total: 59 feature columns from 8 raw. Reasonable — one-hot and target encoding are both doing useful work without exploding dimensionality.

## 3. Validation steps

1. **Pipeline fits and transforms end-to-end** on a 1,000-row synthetic sample. Shape of `pre.transform(X_train)` matches the expected count above.
2. **No leakage probe** — shuffle `y_train` before fit. If training AUC on a shuffled target is above ~0.55, the preprocessor is leaking the label. Both plain one-hot and the K-fold target encoder should pass this.
3. **Round-trip persistence** — `joblib.dump(pipe, p)` then `joblib.load(p)`; the reloaded pipeline must produce identical `transform` output on a held-out frame.
4. **Unknown-category robustness** — construct a validation row with a `currency` value that never appeared in train. `OneHotEncoder(handle_unknown="ignore")` must emit an all-zero row for that block, not raise.
5. **High-cardinality encoder: out-of-fold correctness** — with `random_state` fixed, `fit_transform(X_train, y_train)` reproduces the same encoded column on two runs; the global mapping stored on `self.mappings_` matches a from-scratch `groupby` on full `X_train`.
6. **Missing values** — inject NaN into every numeric column; the pipeline fits without raising and the transformed column has no NaN.
7. **Determinism of datetime features** — `DatetimeExpander` gives `_doy_sin` and `_doy_cos` whose squared sum is approximately 1.0 (within float error); this is a trivial invariant that catches a class of off-by-one bugs in cyclical encoding.

## 4. Rubric / review checklist

Weighted, `/20` points.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Single pipeline object | 2 | Preprocessing + estimator live in one `Pipeline`; one `.fit` call in the training script |
| 2 | Explicit column lists | 2 | Numeric / low-card / high-card / datetime are named lists, not `select_dtypes` |
| 3 | Imputation before scaling | 2 | For each block, imputer runs before any statistical transform; NaN never reaches `StandardScaler` |
| 4 | Robust numeric statistics | 1 | `median` not `mean` for imputation; `StandardScaler` or `QuantileTransformer`, not min-max on raw |
| 5 | Unknown-category handling | 2 | `OneHotEncoder(handle_unknown="ignore")` or equivalent; no crash on unseen level |
| 6 | Leakage-safe target encoding | 3 | K-fold out-of-fold on train; full-train mapping on transform; distinct `fit_transform` vs. `transform` implementations |
| 7 | Smoothing on target encoder | 1 | Smoothing parameter present; documented rationale for its value |
| 8 | Datetime expansion | 2 | Discrete (year/month/dow) *and* cyclical (sin/cos) features; UTC parsing explicit |
| 9 | Reproducible | 2 | `random_state` pinned everywhere randomness exists; `joblib` round-trip preserves transform output |
| 10 | Leakage probe test | 3 | Repo contains a shuffled-target test that proves the pipeline does not peek at `y` |

## 5. Common mistakes

- **Fitting scalers on the full dataset before `train_test_split`.** This is the single most common leak. Every statistic (mean, median, quantile edges, bin boundaries) must come from `X_train` only.
- **Target encoding without K-fold.** Encoding each category by its target mean on *all* of train injects the label directly into the feature. Shuffled-target AUC spikes. K-fold out-of-fold encoding is the fix.
- **`select_dtypes` to pick columns.** A backfill that delivers numbers as strings silently reroutes that column into the categorical branch. Named lists are explicit and trip loudly when the upstream shape drifts.
- **One-hot encoding a high-cardinality column.** A 10,000-level SKU explodes into a 10,000-wide sparse block, which most tabular models handle poorly. Target encoding (with K-fold + smoothing) is the correct tool there.
- **Forgetting `handle_unknown="ignore"`.** The default raises on an unseen category. Validation sets almost always contain one. The pipeline must degrade (emit zeros), not crash.
- **Scaling binary features.** Scaling a one-hot column does nothing useful and hurts interpretability. Keep the categorical pipeline out of the numeric scaler.
- **Non-reproducible fits.** No `random_state` on `KFold` or `train_test_split`, so two runs disagree and the reviewer can't tell whether a small score change is noise or signal.
- **Pipeline refit per model.** The preprocessor should fit once; model selection should swap only the final estimator. Refitting features for each candidate is wasted time and a source of subtle inconsistency.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MEASURE 2.6` (validity under representative data) and `MAP 3.4` (documented preprocessing). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft` (motivation for serialising and version-controlling the fitted pipeline alongside the model) and `ML02:2023 Data Poisoning Attack` (motivation for validating preprocessing inputs). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0043 Craft Adversarial Data` for the adversarial framing of "what happens when a feature-engineering step receives out-of-distribution input." <https://atlas.mitre.org/>
