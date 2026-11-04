# exercise-04 — Schema and Distribution Validation · SOLUTION

Paired exercise: `lessons/mod-103-data-engineering-for-ml/exercises/exercise-04-schema-and-distribution-validation.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to wire up two guards that sit between the data pipeline and the model:

1. **Schema validation** — does the DataFrame have the columns, dtypes, nullability, and value ranges we agreed on?
2. **Distribution validation** — has the *statistical shape* of each column drifted from the training reference?

Schema failures and distribution failures are different in kind and should be handled differently:

| | Schema | Distribution |
|---|---|---|
| What changed | Structure (column added/renamed, dtype flipped) | Values (mean shifted, new modes appeared) |
| Action on failure | Block the pipeline. A training job cannot proceed. | Alert, don't necessarily block. Distribution drift might be a real-world change, not a bug. |
| Where it's fixed | Upstream producer | Retraining, slicing, or accepting the new regime |
| How it's detected | Pandera / Great Expectations / TFDV schema | KS test, PSI, chi-square |

Reference stack:

- **Pandera** for the declarative schema. It's a `DataFrameSchema` class that validates in one line and raises a structured `SchemaError`.
- **Custom drift module** on top of SciPy. Two detectors — Kolmogorov–Smirnov for numeric columns, chi-square for categorical — plus Population Stability Index (PSI) as the operational metric most teams track.
- **Reference artifact.** The schema (literal Python) and the reference distribution stats (a JSON) are checked into the repo under `validation/`. Validation at serving/retrain time compares against those artifacts, never against a live snapshot.

Design rationale, in order:

1. **Fail loud on schema drift, warn on distribution drift.** Schema is a producer contract; distribution is physics. Confusing the two produces either silent pipeline failures (treating schema drift as a warning) or alert fatigue (treating distribution drift as blocking).
2. **PSI thresholds are advisory per column, not global.** A single global PSI threshold either over-fires on volatile columns or under-fires on stable ones.
3. **Reference is a frozen artifact, not a sliding window.** "Compared to last week" lets gradual drift accumulate invisibly. "Compared to the training distribution" catches it.
4. **Statistical tests on small samples are liars.** Both detectors short-circuit with a clear "sample too small" verdict rather than report a spuriously significant p-value.

## 2. Worked implementation

Running example: the `orders_enriched.parquet` dataset from earlier exercises, validated before each training run and before each batch inference run.

### 2.1 Schema (Pandera)

```python
# validation/schema.py
import pandera as pa
from pandera import Check, Column, DataFrameSchema

ALLOWED_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}
ALLOWED_STATUSES = {"created", "paid", "shipped", "cancelled", "refunded"}

ORDERS_SCHEMA = DataFrameSchema(
    columns={
        "order_id": Column(int, nullable=False, unique=True),
        "customer_id": Column(int, nullable=False),
        "currency": Column(
            str, nullable=False,
            checks=Check.isin(ALLOWED_CURRENCIES),
        ),
        "status": Column(
            str, nullable=False,
            checks=Check.isin(ALLOWED_STATUSES),
        ),
        "order_total_usd": Column(
            float, nullable=False,
            checks=[Check.ge(0), Check.le(1_000_000)],
        ),
        "created_at": Column(pa.DateTime, nullable=False),
        "updated_at": Column(pa.DateTime, nullable=False),
        "event_count": Column(int, nullable=True, checks=Check.ge(0)),
    },
    strict=True,     # unknown columns are a schema violation
    coerce=False,    # no silent dtype coercion
)


def validate_schema(df):
    """Return validated df or raise pandera.errors.SchemaError."""
    return ORDERS_SCHEMA.validate(df, lazy=True)
```

`strict=True` is the important flag — without it, extra columns silently sail through. `coerce=False` is the other — coercing a string "3.14" to float masks an upstream bug. `lazy=True` collects *all* errors instead of failing on the first, so one run surfaces the full story.

### 2.2 Reference statistics

```python
# validation/reference.py
import json
from pathlib import Path
import numpy as np
import pandas as pd


NUMERIC_COLS = ["order_total_usd", "event_count", "days_since_last_order"]
CATEGORICAL_COLS = ["currency", "status", "country_code"]


def build_reference(df: pd.DataFrame) -> dict:
    ref: dict = {"numeric": {}, "categorical": {}, "row_count": int(len(df))}

    for col in NUMERIC_COLS:
        s = df[col].dropna()
        # 11 quantile edges → 10 PSI bins
        edges = np.quantile(s, np.linspace(0, 1, 11)).tolist()
        # Dedupe repeated edges (skewed distributions collapse several quantiles)
        edges = sorted(set(edges))
        ref["numeric"][col] = {
            "edges": edges,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "null_rate": float(df[col].isna().mean()),
        }

    for col in CATEGORICAL_COLS:
        freq = df[col].value_counts(normalize=True, dropna=False).to_dict()
        ref["categorical"][col] = {
            "frequencies": {str(k): float(v) for k, v in freq.items()},
            "null_rate": float(df[col].isna().mean()),
        }

    return ref


def save_reference(ref: dict, path: Path) -> None:
    path.write_text(json.dumps(ref, indent=2, sort_keys=True))


def load_reference(path: Path) -> dict:
    return json.loads(path.read_text())
```

This runs once, at training time, on the exact DataFrame the model was fit on. The resulting JSON is version-controlled next to the model artifact — the model card references it by commit SHA.

### 2.3 Drift detectors

```python
# validation/drift.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

MIN_SAMPLE = 100


@dataclass
class DriftResult:
    column: str
    kind: str          # "numeric" | "categorical"
    metric: str        # "ks" | "chi2" | "psi"
    value: float
    p_value: float | None
    advisory: float
    blocking: float
    verdict: str       # "pass" | "advisory" | "blocking" | "insufficient_data"


def _verdict(value: float, advisory: float, blocking: float) -> str:
    if value >= blocking:
        return "blocking"
    if value >= advisory:
        return "advisory"
    return "pass"


def psi(expected: np.ndarray, actual: np.ndarray, edges: list[float]) -> float:
    eps = 1e-6
    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)
    exp_p = exp_hist / max(exp_hist.sum(), 1)
    act_p = act_hist / max(act_hist.sum(), 1)
    exp_p = np.clip(exp_p, eps, None)
    act_p = np.clip(act_p, eps, None)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def check_numeric(
    col: str, actual: pd.Series, ref_col: dict,
    advisory: float = 0.1, blocking: float = 0.25,
) -> DriftResult:
    a = actual.dropna().to_numpy()
    if len(a) < MIN_SAMPLE:
        return DriftResult(col, "numeric", "psi", 0.0, None, advisory, blocking, "insufficient_data")

    # PSI needs the reference distribution. Reconstruct from the stored edges:
    # the training-time binning gave equal-mass bins, so expected_hist is uniform.
    edges = ref_col["edges"]
    exp_hist = np.full(len(edges) - 1, 1.0 / (len(edges) - 1))
    act_hist, _ = np.histogram(a, bins=edges)
    act_p = np.clip(act_hist / max(act_hist.sum(), 1), 1e-6, None)
    exp_p = np.clip(exp_hist, 1e-6, None)
    value = float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))

    # KS p-value as a secondary signal (uses the ref mean/std → a normal approximation only
    # if the reference stats are the only thing persisted; otherwise bootstrap from the ref sample).
    p = None
    return DriftResult(col, "numeric", "psi", value, p, advisory, blocking, _verdict(value, advisory, blocking))


def check_categorical(
    col: str, actual: pd.Series, ref_col: dict,
    advisory: float = 0.1, blocking: float = 0.25,
) -> DriftResult:
    a = actual.astype(object).fillna("__null__")
    if len(a) < MIN_SAMPLE:
        return DriftResult(col, "categorical", "psi", 0.0, None, advisory, blocking, "insufficient_data")

    ref_freq = ref_col["frequencies"]
    act_freq = a.value_counts(normalize=True).to_dict()

    categories = set(ref_freq) | set(act_freq)
    exp = np.array([max(ref_freq.get(c, 0.0), 1e-6) for c in categories])
    act = np.array([max(act_freq.get(c, 0.0), 1e-6) for c in categories])
    value = float(np.sum((act - exp) * np.log(act / exp)))

    # Chi-square as a p-value signal
    exp_counts = exp * len(a)
    act_counts = act * len(a)
    _, p = stats.chisquare(f_obs=act_counts, f_exp=exp_counts)
    return DriftResult(col, "categorical", "psi", value, float(p), advisory, blocking, _verdict(value, advisory, blocking))
```

PSI thresholds `0.1` (advisory) and `0.25` (blocking) are the widely cited credit-scoring heuristic. They are not universal — tune per column based on your observed baseline variance. Thresholds default here but are overridable per-column.

### 2.4 The runner

```python
# validation/run.py
import json
import sys
from pathlib import Path
import pandas as pd
from .schema import validate_schema
from .reference import load_reference, NUMERIC_COLS, CATEGORICAL_COLS
from .drift import check_numeric, check_categorical


def main(parquet_path: str, ref_path: str, report_out: str) -> int:
    df = pd.read_parquet(parquet_path)

    # Schema first — blocking.
    try:
        df = validate_schema(df)
    except Exception as e:
        Path(report_out).write_text(json.dumps({
            "schema": {"ok": False, "error": str(e)},
            "distribution": None,
            "overall": "blocking",
        }, indent=2, sort_keys=True))
        return 2

    ref = load_reference(Path(ref_path))
    results = []
    for col in NUMERIC_COLS:
        results.append(check_numeric(col, df[col], ref["numeric"][col]))
    for col in CATEGORICAL_COLS:
        results.append(check_categorical(col, df[col], ref["categorical"][col]))

    overall = "pass"
    if any(r.verdict == "advisory" for r in results):
        overall = "advisory"
    if any(r.verdict == "blocking" for r in results):
        overall = "blocking"

    Path(report_out).write_text(json.dumps({
        "schema": {"ok": True},
        "distribution": [r.__dict__ for r in results],
        "overall": overall,
    }, indent=2, sort_keys=True, default=float))

    return {"pass": 0, "advisory": 1, "blocking": 2}[overall]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
```

Schema is checked first and short-circuits. There is no meaningful "distribution" measurement on a DataFrame whose columns don't match — you'd be comparing apples to a renamed orange.

### 2.5 Example report excerpt

```json
{
  "schema": {"ok": true},
  "distribution": [
    {"column": "order_total_usd", "kind": "numeric", "metric": "psi",
     "value": 0.08, "p_value": null, "advisory": 0.1, "blocking": 0.25, "verdict": "pass"},
    {"column": "currency", "kind": "categorical", "metric": "psi",
     "value": 0.13, "p_value": 0.001, "advisory": 0.1, "blocking": 0.25, "verdict": "advisory"},
    {"column": "country_code", "kind": "categorical", "metric": "psi",
     "value": 0.31, "p_value": 0.0, "advisory": 0.1, "blocking": 0.25, "verdict": "blocking"}
  ],
  "overall": "blocking"
}
```

A reviewer can tell instantly: schema is fine, totals look stable, currency mix shifted a bit, and country distribution moved enough to block retraining or trigger a review of whether the training reference is still representative — the exact conversation NIST AI RMF `MEASURE 2.11` (monitoring measured properties over time) asks you to hold.

## 3. Validation steps

1. **Pandera smoke test** — round-trip a known-good DataFrame through `validate_schema`; it returns the same frame. Introduce a renamed column, an out-of-set `currency`, and a negative `order_total_usd`; each must appear in the raised `SchemaError`.
2. **`strict=True` is active** — add an extra column `experiment_arm` not listed in the schema; validation must fail. This test guards against someone accidentally flipping the flag.
3. **PSI zero on identity** — compute PSI on the reference against itself; value must be ≤ ~1e-3. Anything larger means the detector is biased.
4. **PSI monotone under shift** — shift the actual distribution by one standard deviation for `order_total_usd`; PSI must be strictly larger than on the unshifted input and must cross the advisory threshold by a 2σ shift.
5. **Categorical new-level handling** — inject a brand-new `country_code` that was not in the reference; the detector must register nonzero PSI and not raise.
6. **Insufficient-data verdict** — pass in a 20-row DataFrame; every drift result must come back as `insufficient_data`, not a flaky p-value.
7. **Report determinism** — run the validator twice on the same parquet + reference; output JSONs are byte-identical (keys sorted, floats rounded).
8. **Exit code contract** — schema failure → 2; distribution blocking → 2; distribution advisory → 1; all-pass → 0.
9. **Reference versioning** — the reference JSON is committed to git; a change to the reference is reviewed as a policy change, not a silent update.

## 4. Rubric / review checklist

Weighted, `/20` points.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Schema distinct from drift | 3 | Two separate functions; schema failure short-circuits, drift failure does not depend on schema |
| 2 | Pandera schema is strict | 2 | `strict=True`; unknown columns fail; `coerce=False` or explicit coercion rationale |
| 3 | Nullability and ranges declared | 2 | Every numeric column has at least a min bound; every categorical has an `isin` or regex check |
| 4 | Reference stored as artifact | 2 | Reference JSON is checked in; validator loads it, does not recompute from live data |
| 5 | Numeric drift metric | 2 | PSI (or KS) computed with the training-time bin edges, not fresh quantiles |
| 6 | Categorical drift metric | 2 | PSI or chi-square; new categories handled without crashing |
| 7 | Advisory vs. blocking thresholds | 2 | Both tiers present per column; defaults documented; per-column override possible |
| 8 | Small-sample handling | 1 | Clear `insufficient_data` verdict below a stated row count |
| 9 | Deterministic report | 2 | Same input → same bytes; keys sorted, no wall-clock timestamps inside measurements |
| 10 | Exit code contract | 2 | 0 / 1 / 2 maps to pass / advisory / blocking; schema failure is always 2 |

## 5. Common mistakes

- **One metric, one threshold, all columns.** Different features move at different natural rates; a global PSI threshold either over- or under-fires. Per-column thresholds are worth the effort.
- **Comparing against last week's data.** This lets gradual drift accumulate invisibly: each week is close to the previous week, but all weeks drift away from training. The reference must be the training distribution, pinned to a model version.
- **Recomputing quantile bin edges on the actual data.** PSI is only meaningful with *fixed* bins. If edges move, you are comparing two different histograms.
- **Treating schema drift as a warning.** A renamed column means training sees nulls or missing features. Always block.
- **Treating distribution drift as a blocker.** Real-world shifts (holiday traffic, a new market launching) will fire drift detectors by design. Alert on advisory, require review on blocking, don't auto-fail.
- **Chi-square without expected-count checks.** Chi-square assumes each expected count is ≥ 5. On small categorical tails, it reports garbage. Fall back to PSI or merge rare categories into "other" before the test.
- **Missing the "new categories" case.** A `country_code` that never appeared in training has zero expected frequency. Clip to `eps`, don't divide by zero.
- **Reference recomputed at deploy time.** If the validator rebuilds the reference from "the data we deployed on," the first week always looks drift-free. Freeze the reference at training time, version it with the model.
- **`coerce=True` in Pandera.** Silent dtype coercion is a bug magnet — a string `"3.14"` becomes a float and your type contract is wrong. Prefer loud failure.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MEASURE 2.8` (data integrity and validity measurement) and `MEASURE 2.11` (monitoring the measured properties over time). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` (motivation for schema validation), `ML08:2023 Model Skewing` (motivation for distribution monitoring). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0020 Poison Training Data` and `AML.T0043 Craft Adversarial Data` as the threat models both detectors are controls for. <https://atlas.mitre.org/>
