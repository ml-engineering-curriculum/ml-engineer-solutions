# exercise-02 — Leakage-Resistant Validation · SOLUTION

Paired exercise: `lessons/mod-107-model-evaluation-validation/exercises/exercise-02-leakage-resistant-validation.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to build a validation harness that does **not leak** information from training into evaluation. Data leakage is the single most common reason a model's offline AUC is 0.92 and its production AUC is 0.63: the evaluation accidentally saw something at training time that will not be present at serving time.

This solution treats leakage as three distinct bugs that need three distinct defences:

1. **Target leakage** — a feature is partially a function of the label (e.g., `days_since_churn`, `refund_amount_90d`). Defence: feature-time audit against a per-feature "allowed-as-of" timestamp.
2. **Train/test contamination** — the same entity, duplicate row, near-duplicate, or future-in-training shows up on both sides of the split. Defence: group-aware and time-aware splits, dedup before split, and a feature-time cutoff.
3. **Pipeline leakage** — fit-time statistics (mean/std, target encoder, PCA basis, vocabulary) are computed on the full dataset before splitting. Defence: everything with `fit` sits inside an `sklearn.pipeline.Pipeline`, and all cross-validation happens over the pipeline, never over a pre-fit transformer.

The design priorities, in order:

1. **Splits reflect production.** If predictions at serving time are made at time `t` for entity `e`, the training set must contain only data visible at time `≤ t` for entities whose splits do not overlap with `e`'s. `train_test_split(..., shuffle=True)` fails this test for any temporal or grouped task.
2. **Transforms learn from training only.** Every `fit()` call on a transformer must see training data only. The `Pipeline` + `cross_val_score` idiom is the mechanical enforcement; manual `.fit_transform(X_all)` followed by splitting is the anti-pattern.
3. **Features have an "available-at" time.** Each feature carries the event time it becomes observable. The training row at label-time `t_label` uses only features with `available_at ≤ t_label − embargo_window`. This is the single defence that catches target leakage from derived features.
4. **Tests over the harness, not just the model.** Deliberately inject a leaky feature and assert the harness's score *drops*; inject a non-leaky feature and assert nothing changes. A validation harness you have not tried to break is one you do not know works.
5. **Report leak-adjusted and naive metrics side by side.** The gap between the naive (shuffled random split) metric and the leak-resistant metric is a diagnostic for how much leakage was there to begin with.

This design maps to NIST AI RMF `MEASURE 2.5` (validity and reliability measurement under realistic conditions) and `MEASURE 2.11` (performance across subpopulations stays meaningful when the split is correct). The OWASP ML Top 10 `ML02:2023 Data Poisoning Attack` surface is partly a leakage surface — a poisoned sample that lives in both train and test hides its own effect.

| Choice | Alternative | Why this one |
|---|---|---|
| `TimeSeriesSplit` + embargo window for temporal data | Random `KFold` | Random splits leak the future into the past when data has any time dependency |
| `GroupKFold` keyed on the entity id | `KFold` on rows | A user who appears 40 times with 20 train / 20 test rows produces an inflated AUC that collapses on new users |
| `Pipeline([preprocess, model])` scored via `cross_val_score` | `fit_transform(X_all)` then split | The pipeline refits preprocessing *per fold*; the manual approach fits on the test set by construction |
| Feature-time cutoff per feature | Global cutoff | Some features are observable before others (join-time matters); per-feature `available_at` is accurate |
| Deduplicate on composite keys, not raw rows | Deduplicate exact rows only | Near-duplicates (same entity, slightly different timestamp) still leak; composite keys catch them |
| Leakage sanity tests in CI | Only test the model | A silent regression to `train_test_split(..., shuffle=True)` must fail the build, not the production model |

## 2. Worked implementation

### 2.1 Project layout

```
validation/
├── __init__.py
├── schema.py            # feature registry with available_at times
├── splits.py            # group + time split generators
├── dedup.py             # composite-key deduplication
├── pipeline.py          # Pipeline factory that keeps fit inside CV
├── evaluate.py          # cross-validated evaluation + leakage probe
└── tests/
    ├── test_leak_injection.py
    └── test_split_properties.py
```

### 2.2 `schema.py` — feature registry with an "available-at" time

```python
from dataclasses import dataclass
from datetime import timedelta
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    available_at_column: str   # the column in the raw frame that gives this feature's event time
    embargo: timedelta = timedelta(0)  # extra safety margin before label_time


FEATURES = [
    FeatureSpec("order_count_30d", "feature_event_time", timedelta(hours=1)),
    FeatureSpec("avg_order_value_30d", "feature_event_time", timedelta(hours=1)),
    FeatureSpec("last_login_days", "feature_event_time"),
    # DO NOT register features whose event time is >= label_time
]


def enforce_feature_time(df: pd.DataFrame, label_time_col: str) -> pd.DataFrame:
    """Null out any feature whose event_time is after label_time - embargo."""
    out = df.copy()
    for spec in FEATURES:
        event_time = df[spec.available_at_column]
        label_time = df[label_time_col]
        future = event_time > (label_time - spec.embargo)
        if future.any():
            out.loc[future, spec.name] = pd.NA
    return out
```

The registry *is* the policy — a feature that isn't registered isn't allowed in the pipeline. Nulling (rather than dropping the row) preserves the audit trail: the training frame shows that the feature was present but unavailable at that point in time.

### 2.3 `splits.py` — group- and time-aware splits

```python
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    time_col: str
    group_col: str | None
    n_splits: int = 5
    embargo: timedelta = timedelta(days=1)
    val_fraction: float = 0.2   # within train window, used for threshold selection


def rolling_time_group_splits(df: pd.DataFrame, spec: SplitSpec):
    """
    Yield (train_idx, test_idx) with:
      - test strictly after train (time-forward)
      - an embargo gap between train and test
      - groups (entities) never present in both sides
    """
    t = pd.to_datetime(df[spec.time_col]).sort_values()
    boundaries = np.quantile(t.values, np.linspace(0, 1, spec.n_splits + 2))[1:-1]

    groups = df[spec.group_col] if spec.group_col else None

    for b in boundaries:
        train_mask = pd.to_datetime(df[spec.time_col]) <= (pd.Timestamp(b) - spec.embargo)
        test_mask = pd.to_datetime(df[spec.time_col]) > pd.Timestamp(b)
        if groups is not None:
            train_groups = set(groups[train_mask])
            # drop test rows whose group also appears in train
            test_mask = test_mask & ~groups.isin(train_groups)
        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx
```

Three details:

- **Embargo gap.** Between the last train timestamp and the first test timestamp we cut a known window (hours to days, task-dependent). In prediction-at-time-of-event tasks this closes the "label is defined over a 24 h horizon" leak.
- **Group-aware test mask.** Rows in `test` whose entity id *also* appears in `train` are dropped from `test`, not moved to `train` — moving them into train contaminates the training frame with future information.
- **Boundaries by quantile of time, not by row count.** Quantiles of time give you roughly equal-duration folds; row-count splits give you equal folds that all contain the most recent, most seasonal window.

### 2.4 `dedup.py` — composite-key dedup

```python
from collections.abc import Sequence
import pandas as pd


def dedupe(df: pd.DataFrame, keys: Sequence[str], keep: str = "last") -> pd.DataFrame:
    """Drop rows that are duplicates on a composite key before splitting."""
    before = len(df)
    out = df.drop_duplicates(subset=list(keys), keep=keep).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"dedup dropped {dropped} rows on keys={list(keys)}")
    return out


# Example composite keys for a user-session event stream:
#   dedupe(df, keys=["user_id", "session_id", "event_minute"])
```

Dedup is upstream of split. If duplicates straddle the split boundary the group-aware split still saves you (same entity id), but near-duplicates with different entity ids (device telemetry rebooting into a new session id) still leak unless the composite key is chosen to catch them.

### 2.5 `pipeline.py` — fit inside the fold

```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_COLS = ["order_count_30d", "avg_order_value_30d", "last_login_days"]


def make_pipeline(random_state: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("model", GradientBoostingClassifier(random_state=random_state)),
    ])
```

The only correct way to compute scaler / imputer / target-encoder statistics in cross-validation is inside a `Pipeline` that is `.fit()`'ed on each fold's training indices. Pre-computing on the full frame is the canonical scikit-learn leakage anti-pattern and is explicitly called out in the scikit-learn user guide.

### 2.6 `evaluate.py` — cross-validated evaluation + leakage probe

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .pipeline import make_pipeline
from .schema import enforce_feature_time
from .splits import SplitSpec, rolling_time_group_splits


@dataclass
class EvalResult:
    auc_mean: float
    auc_std: float
    pr_auc_mean: float
    naive_auc: float   # from a shuffled random split, for the leak-gap diagnostic


def evaluate(df: pd.DataFrame, target: str, split_spec: SplitSpec) -> EvalResult:
    df = enforce_feature_time(df, label_time_col="label_time")

    aucs, pr_aucs = [], []
    for train_idx, test_idx in rolling_time_group_splits(df, split_spec):
        pipe = make_pipeline()
        pipe.fit(df.iloc[train_idx], df[target].iloc[train_idx])
        proba = pipe.predict_proba(df.iloc[test_idx])[:, 1]
        y_test = df[target].iloc[test_idx]
        aucs.append(roc_auc_score(y_test, proba))
        pr_aucs.append(average_precision_score(y_test, proba))

    # Naive baseline: deliberately leaky random split — the gap is the diagnostic.
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)
    pipe = make_pipeline().fit(tr, tr[target])
    naive_auc = roc_auc_score(te[target], pipe.predict_proba(te)[:, 1])

    return EvalResult(
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        naive_auc=float(naive_auc),
    )
```

The gap between `naive_auc` and `auc_mean` is the number to put on the model card. If naive = 0.94 and leak-resistant = 0.71, the headline AUC is 0.71 **and** the team knows there is 0.23 AUC worth of leakage somewhere worth hunting. If the gap is 0.01, you have a believable model.

### 2.7 `tests/test_leak_injection.py` — the harness tests itself

```python
import numpy as np
import pandas as pd
import pytest

from validation.evaluate import evaluate
from validation.splits import SplitSpec


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 10_000
    t = pd.date_range("2024-01-01", periods=n, freq="min")
    user = rng.integers(0, 2_500, size=n)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({
        "user_id": user,
        "event_time": t,
        "feature_event_time": t,
        "label_time": t + pd.Timedelta("24h"),
        "order_count_30d": x,
        "avg_order_value_30d": rng.normal(size=n),
        "last_login_days": rng.normal(size=n),
        "target": y,
    })


def test_time_split_does_not_leak(synthetic_df):
    spec = SplitSpec(time_col="event_time", group_col="user_id")
    result = evaluate(synthetic_df, target="target", split_spec=spec)
    # Leak-resistant AUC is honest; naive AUC should be close on truly i.i.d. data.
    assert abs(result.naive_auc - result.auc_mean) < 0.05


def test_injected_leak_is_detected(synthetic_df):
    # Make a copy of the label into a feature column — classic target leak.
    leaky = synthetic_df.copy()
    leaky["order_count_30d"] = leaky["target"] + np.random.normal(0, 0.01, len(leaky))
    spec = SplitSpec(time_col="event_time", group_col="user_id")
    result = evaluate(leaky, target="target", split_spec=spec)
    # Naive AUC is near-perfect; leak-resistant AUC must flag the problem.
    assert result.naive_auc > 0.98
    assert result.auc_mean < result.naive_auc  # harness recognises the gap


def test_entity_overlap_is_prevented(synthetic_df):
    spec = SplitSpec(time_col="event_time", group_col="user_id")
    from validation.splits import rolling_time_group_splits
    for tr, te in rolling_time_group_splits(synthetic_df, spec):
        tr_users = set(synthetic_df["user_id"].iloc[tr])
        te_users = set(synthetic_df["user_id"].iloc[te])
        assert tr_users.isdisjoint(te_users)
```

The second test is the critical one: inject an *obvious* leak and confirm the harness still reports a lower leak-resistant AUC than the naive AUC. Without it, "our validation is leak-resistant" is wishful.

## 3. Validation steps

Run these in order. Each catches a distinct bug class.

1. **Static check** — `python -c "from validation.evaluate import evaluate"`. Catches import errors before any run.
2. **Unit tests** — `pytest validation/tests -q`. All three leak-injection / split-property tests must pass.
3. **Manual split inspection** — for one fold of `rolling_time_group_splits`, print `max(train_time)` and `min(test_time)`. The difference must be ≥ the configured embargo.
4. **Group disjointness** — for every fold, assert `set(train_groups) ∩ set(test_groups) == ∅`. Automate as a property-based test over three datasets of increasing group-overlap density.
5. **Fit-inside-fold check** — `scaler.mean_` extracted from the fitted pipeline on fold 0 must differ from the scaler fitted on the full frame. If they are equal, the pipeline was pre-fit — a leakage bug.
6. **Null-feature probe** — drop one feature at a time and re-run. A feature whose removal drops AUC by > 0.1 on a leak-resistant split deserves a leakage audit even if no test flagged it; its signal may be legitimate or may be a label proxy.
7. **Known-leak probe** — add a column equal to the label (or `target + ε`) and re-run. Naive AUC must be near 1.0; leak-resistant AUC must drop below naive. If both are near 1.0, the "leak-resistant" split is still leaking.
8. **Production-shape backtest** — using actual production inference timestamps, confirm that for every scored prediction, every feature's `available_at` ≤ scoring time. This is the deployment-side mirror of the training-side feature-time cutoff.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Split reflects deployment shape | 3 | Temporal task → time-ordered split; entity-level task → group split; both when both apply |
| 2 | Embargo gap | 2 | Numeric embargo between train and test, configurable, non-zero for label-horizon tasks |
| 3 | Pipeline-scoped preprocessing | 3 | All `fit` calls live inside a `Pipeline`; no `fit_transform(X_all)` before split |
| 4 | Feature-time cutoff | 3 | Per-feature `available_at`; features that would arrive after the label time are nulled, not used |
| 5 | Dedup before split | 1 | Composite-key dedup documented and applied |
| 6 | Leak-injection self-test | 3 | At least one test injects a leak and asserts the harness distinguishes leaky from honest evaluation |
| 7 | Naive-vs-leak-resistant gap reported | 2 | Both numbers appear on the model card; the gap is called out |
| 8 | Group disjointness tested | 1 | Property test over multiple datasets, not just the happy-path fixture |
| 9 | CI-level enforcement | 2 | `test_leak_injection.py` runs in CI; silent regression to `train_test_split(..., shuffle=True)` fails the build |

Items 1, 3, and 6 are mandatory. A submission missing any of them cannot score above 11/20 regardless of the rest — these three cover the failure modes that produce the highest-cost production surprises.

## 5. Common mistakes

- **`train_test_split(..., shuffle=True)` on temporal data.** The classic sin. Random shuffling of event-stream data puts rows from Monday in train and the surrounding minutes in test; the model "predicts" the label by effectively memorising neighbouring rows.
- **`fit_transform` on the full dataset, then split.** Scaler / encoder / PCA / target-encoder statistics learned on the whole dataset are informed by the test rows. The `Pipeline` + `cross_val_score` pattern is the mechanical fix.
- **Target-encoding features without per-fold refit.** A target encoder that saw the test row's label during encoding is a target leak inside a plausible-looking feature. Refit inside the fold.
- **Same user on both sides.** A user with 40 observations and a stable behavioural fingerprint is a leak if any of their rows appear in both train and test. Group-aware splits (`GroupKFold`, `StratifiedGroupKFold`) are the right default for any user- or session-level task.
- **No embargo on label-horizon tasks.** If the label is "churned within 60 days of t0," test rows with `t0` just before the train/test boundary have labels defined over a 60-day window that straddles the boundary. Embargo the boundary by at least the label horizon.
- **Silent `random_state=None`.** Non-determinism in the splitter hides leakage because every CI run gets a different split; a run that passes once can fail tomorrow. Fix the seed.
- **"We'll add leakage tests later."** The leak-injection test is the only mechanical defence against a future refactor silently reintroducing a leak. Writing it later means never.
- **Trusting `cross_val_score` with a model, not a pipeline.** Passing a bare `GradientBoostingClassifier` and pre-scaled features to `cross_val_score` is the same bug as fit_transform-then-split — the scaling saw the full dataset. Score the pipeline.
- **Features with timestamps in the future.** A "last_login_days" that is computed from a feature-store snapshot taken *after* the label is a target leak dressed up as behavioural history. The feature-time cutoff catches this; a human-readable audit does not.
- **Dropping test rows that overlap on entity without telling anyone.** The group-aware split's "drop test rows whose group is in train" needs to be logged — if it drops 80 % of test, the eval set may be too small to trust, and you need to know.

## 6. References

- scikit-learn — "Common pitfalls and recommended practices" documents the preprocessing-outside-pipeline leakage pattern and the correct `Pipeline` + CV approach. <https://scikit-learn.org/stable/common_pitfalls.html>
- scikit-learn — `TimeSeriesSplit`, `GroupKFold`, `StratifiedGroupKFold` reference and semantics. <https://scikit-learn.org/stable/modules/cross_validation.html>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.5` requires that validity and reliability are measured under conditions representative of deployment; a leaky split violates `MEASURE 2.5` even when the number looks good. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF 1.0 — `MEASURE 2.11` on subpopulation performance only becomes meaningful when the split respects group boundaries. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` describes how contamination (including train/test contamination) hides the effect of a poisoned sample. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0020 Poison Training Data` and `AML.T0031 Erode ML Model Integrity` as the threat frames that leakage-resistant evaluation defends against. <https://atlas.mitre.org/>
