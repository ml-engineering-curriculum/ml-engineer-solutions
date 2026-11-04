# exercise-02 — Data Quality Audit · SOLUTION

Paired exercise: `lessons/mod-103-data-engineering-for-ml/exercises/exercise-02-data-quality-audit.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to audit a tabular dataset and produce a report that a reviewer can act on. The reference solution frames the audit on five dimensions that map cleanly onto the NIST AI RMF `MEASURE` function — the "characterise and measure trustworthy properties" step — and writes a machine-readable report alongside a short human-readable summary.

Dimensions (scored 0–1 per column, with a per-dataset roll-up):

| Dimension | Meaning | How it is measured |
|---|---|---|
| **Completeness** | How much data is present? | `1 − null_rate` per column |
| **Uniqueness** | Are rows or keys duplicated? | For PK columns: `1 − duplicate_rate`; for feature columns: cardinality check |
| **Validity** | Do values match the column's stated domain? | Type conformance, range bounds, allowed-value set, regex pattern |
| **Consistency** | Do related fields agree? | Cross-column rules (e.g., `shipped_at >= created_at`, `country ∈ country_code_set`) |
| **Timeliness** | How fresh is the data? | `max(updated_at) − now()` compared to an SLO |

Design rationale:

- **Score per dimension, not one aggregate number.** A single "quality score" hides *which* dimension failed. Reviewers need to know "the data is complete but stale" vs "fresh but 12% of rows duplicated."
- **Deterministic report.** The audit output is a JSON with sorted keys and no timestamps inside measured values, so two runs on the same input produce identical files. This lets CI diff reports between versions.
- **Advisory and blocking thresholds.** Each dimension has an `advisory` floor (warn) and a `blocking` floor (fail CI). Setting only one threshold produces either alert fatigue or silent slippage.
- **Named rule IDs.** Every rule has a stable ID (`VAL.001`, `CON.003`) so a failure in CI can be traced to a specific line of policy rather than "something in validity broke."

The audit is not a drift check — that's exercise-04. This one asks "is the data I have right now fit for training?" not "has it moved since last time?"

## 2. Worked implementation

Reference audit on a fictional `orders_enriched.parquet` built by exercise-01. The schema the auditor reasons over:

| Column | Type | Notes |
|---|---|---|
| `order_id` | int64 | Primary key, unique, non-null |
| `customer_id` | int64 | FK, non-null |
| `currency` | string | ISO 4217, allowed set `{USD, EUR, GBP, JPY}` |
| `status` | string | Enum `{created, paid, shipped, cancelled, refunded}` |
| `order_total_usd` | float64 | `>= 0`, null share should be `< 0.1%` |
| `created_at` | datetime64[ns, UTC] | Non-null |
| `updated_at` | datetime64[ns, UTC] | Non-null, `>= created_at` |
| `event_count` | int64 nullable | Null allowed (orders with no events); `>= 0` when present |

### 2.1 Rule library

```python
# audit/rules.py
from dataclasses import dataclass
from typing import Callable
import pandas as pd


@dataclass(frozen=True)
class Rule:
    id: str
    dimension: str          # completeness | uniqueness | validity | consistency | timeliness
    description: str
    advisory: float         # warn if score below this
    blocking: float         # fail if score below this
    check: Callable[[pd.DataFrame], float]
```

A single `Rule` object is enough: the `check` returns a 0–1 score and the thresholds are evaluated outside the rule. This keeps rules pure functions — easy to unit test one at a time.

### 2.2 Concrete rules

```python
# audit/rules_orders.py
import pandas as pd
from .rules import Rule

ALLOWED_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}
ALLOWED_STATUSES = {"created", "paid", "shipped", "cancelled", "refunded"}


def _not_null(col: str):
    def _check(df: pd.DataFrame) -> float:
        return 1.0 - df[col].isna().mean()
    return _check


def _unique(col: str):
    def _check(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 1.0
        return df[col].nunique(dropna=False) / len(df)
    return _check


def _in_set(col: str, allowed: set[str]):
    def _check(df: pd.DataFrame) -> float:
        present = df[col].dropna()
        if present.empty:
            return 1.0
        return present.isin(allowed).mean()
    return _check


def _non_negative(col: str):
    def _check(df: pd.DataFrame) -> float:
        present = df[col].dropna()
        if present.empty:
            return 1.0
        return (present >= 0).mean()
    return _check


def _updated_after_created(df: pd.DataFrame) -> float:
    if df.empty:
        return 1.0
    return (df["updated_at"] >= df["created_at"]).mean()


RULES = [
    Rule("COM.001", "completeness", "order_id not null",
         advisory=1.00, blocking=1.00, check=_not_null("order_id")),
    Rule("COM.002", "completeness", "customer_id not null",
         advisory=1.00, blocking=0.999, check=_not_null("customer_id")),
    Rule("COM.003", "completeness", "order_total_usd present",
         advisory=0.999, blocking=0.99, check=_not_null("order_total_usd")),
    Rule("UNI.001", "uniqueness", "order_id unique",
         advisory=1.00, blocking=1.00, check=_unique("order_id")),
    Rule("VAL.001", "validity", "currency ∈ allowed set",
         advisory=1.00, blocking=0.999, check=_in_set("currency", ALLOWED_CURRENCIES)),
    Rule("VAL.002", "validity", "status ∈ allowed set",
         advisory=1.00, blocking=0.999, check=_in_set("status", ALLOWED_STATUSES)),
    Rule("VAL.003", "validity", "order_total_usd >= 0",
         advisory=1.00, blocking=0.999, check=_non_negative("order_total_usd")),
    Rule("CON.001", "consistency", "updated_at >= created_at",
         advisory=1.00, blocking=0.999, check=_updated_after_created),
]
```

Blocking thresholds are deliberately *not* 1.0 everywhere. Chasing 100% on a noisy source burns reviewer time on anomalies that will not change the model. The `order_id` rules stay at 1.0 because a missing primary key breaks every join downstream.

### 2.3 Runner

```python
# audit/run.py
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
from .rules_orders import RULES


@dataclass
class RuleResult:
    id: str
    dimension: str
    description: str
    score: float
    verdict: str   # "pass" | "advisory" | "blocking"


def _verdict(score: float, advisory: float, blocking: float) -> str:
    if score < blocking:
        return "blocking"
    if score < advisory:
        return "advisory"
    return "pass"


def run_audit(df: pd.DataFrame) -> dict:
    results: list[RuleResult] = []
    for rule in RULES:
        score = float(rule.check(df))
        results.append(
            RuleResult(
                id=rule.id,
                dimension=rule.dimension,
                description=rule.description,
                score=round(score, 6),
                verdict=_verdict(score, rule.advisory, rule.blocking),
            )
        )

    dims = {}
    for r in results:
        dims.setdefault(r.dimension, []).append(r.score)
    dimension_scores = {d: round(sum(s) / len(s), 6) for d, s in dims.items()}

    overall_verdict = "pass"
    if any(r.verdict == "advisory" for r in results):
        overall_verdict = "advisory"
    if any(r.verdict == "blocking" for r in results):
        overall_verdict = "blocking"

    return {
        "row_count": int(len(df)),
        "dimension_scores": dict(sorted(dimension_scores.items())),
        "rules": [asdict(r) for r in sorted(results, key=lambda r: r.id)],
        "overall": overall_verdict,
    }


def main(parquet_path: str, out_path: str) -> int:
    df = pd.read_parquet(parquet_path)
    report = run_audit(df)
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True))
    if report["overall"] == "blocking":
        return 2
    if report["overall"] == "advisory":
        return 1
    return 0
```

Note the exit codes: `0` pass, `1` advisory, `2` blocking. A CI step that `set -e`s on anything non-zero is too coarse; three states give the scheduler a lever to say "keep going but open a ticket" vs. "stop the pipeline."

### 2.4 Example report excerpt

```json
{
  "row_count": 128412,
  "dimension_scores": {
    "completeness": 0.9994,
    "consistency": 1.0,
    "uniqueness": 1.0,
    "validity": 0.9981
  },
  "overall": "advisory",
  "rules": [
    {"id": "COM.003", "dimension": "completeness",
     "description": "order_total_usd present",
     "score": 0.9983, "verdict": "advisory"},
    {"id": "VAL.001", "dimension": "validity",
     "description": "currency ∈ allowed set",
     "score": 0.9962, "verdict": "advisory"}
  ]
}
```

A reviewer reading this sees immediately: completeness and validity each have one rule dipping below the advisory floor. Overall verdict is advisory, not blocking — the ingest proceeds, but `COM.003` and `VAL.001` should be triaged.

## 3. Validation steps

1. **Rules are pure** — each rule function is tested with (a) an empty DataFrame → score 1.0, (b) an all-null column (completeness rules → score 0.0), (c) a known bad row (validity rules → score `(n-1)/n`).
2. **Verdict boundaries** — unit test `_verdict` at exactly `advisory` and `blocking` thresholds. The spec says "strictly below" triggers the next tier; off-by-one here silently reclassifies incidents.
3. **Overall roll-up** — one blocking rule must produce `overall=blocking` regardless of how many advisories exist.
4. **Determinism** — run the audit twice on the same parquet; the two JSON files must be byte-identical (keys sorted, scores rounded).
5. **Exit code contract** — pass, advisory, and blocking inputs produce exit codes 0, 1, 2 respectively. A CI job that only checks `!= 0` would conflate advisory with blocking — this test documents the contract.
6. **Policy diff** — when a rule's threshold changes, the change is visible in version control. A reviewer can answer "when did we start tolerating 1% invalid currencies?" with `git log`.

## 4. Rubric / review checklist

Weighted, `/20` points.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Five dimensions covered | 3 | At least one rule each for completeness, uniqueness, validity, consistency, timeliness |
| 2 | Per-dimension scores reported | 2 | The report exposes dimension-level scores, not just a single number |
| 3 | Advisory vs. blocking split | 3 | Every rule has both thresholds; blocking ≤ advisory; rationale documented for any threshold < 1.0 |
| 4 | Rule IDs stable | 1 | Every rule has an ID; IDs are not reused after deletion |
| 5 | Report is deterministic | 2 | Two runs on the same input produce byte-identical reports |
| 6 | Report is machine-readable | 2 | JSON with sorted keys; a downstream tool can parse without regex |
| 7 | Exit code contract | 2 | 0 / 1 / 2 for pass / advisory / blocking |
| 8 | Rule unit tests | 3 | Each rule has positive and negative tests; thresholds have boundary tests |
| 9 | Empty-input handling | 1 | Audit on an empty DataFrame returns score 1.0 for every rule and does not raise |
| 10 | Policy versioned | 1 | Rules + thresholds live in source control, not in a notebook cell |

## 5. Common mistakes

- **Reporting one number.** "Quality score 0.94" is unactionable. Break it out per dimension so the reviewer knows which lever to pull.
- **Thresholds hard-set to 1.0.** This produces alert fatigue and gets the whole audit disabled within a quarter. Set to the actual noise floor you've measured, then tighten as the upstream improves.
- **Rules that mutate the input.** A rule that drops duplicates and then measures something is unreproducible. Rules should read, not write.
- **Measuring missingness on columns that are legitimately nullable.** `event_count` is null for orders with no events — scoring that as 0 is a bug in the policy, not the data.
- **No empty-input guard.** The first time you audit a new, empty partition, `df["x"].isna().mean()` returns NaN and the whole report collapses. Short-circuit to 1.0 on empty input.
- **Rules without IDs.** When CI says "validity dropped," the next engineer has to re-derive which rule changed. Stable IDs pay for themselves on the first incident.
- **Conflating audit with drift.** Audit answers "is this fit to train on?" Drift (exercise-04) answers "is this the same distribution we trained on?" A dataset can be high-quality *and* drifted, or low-quality *and* on-distribution.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (data quality metrics) and `GOVERN 1.4` (documented thresholds and roles). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack`; a data quality audit at ingest is one of the listed controls. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0020 Poison Training Data` as the threat model behind validity and consistency checks. <https://atlas.mitre.org/>
