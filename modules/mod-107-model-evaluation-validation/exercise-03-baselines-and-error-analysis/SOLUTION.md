# exercise-03 — Baselines and Error Analysis · SOLUTION

Paired exercise: `lessons/mod-107-model-evaluation-validation/exercises/exercise-03-baselines-and-error-analysis.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to (a) stand up a **baseline ladder** — a sequence of progressively smarter reference models — and (b) run a structured **error analysis** on the candidate model's mistakes. Both artifacts answer the same question from opposite directions: *is the headline metric telling the truth?*

The baseline ladder tells you whether the candidate model learned anything beyond the task's trivial structure. The error analysis tells you whether the candidate model's *remaining* errors are uniform noise or concentrated on a slice the team cares about.

Design priorities, in order:

1. **Every model is compared against a trivial baseline first.** The majority-class / mean-predictor sets the absolute floor. If the candidate does not clearly beat it, the model is not a model — it is noise fit to a seed.
2. **The ladder has rungs, not a single baseline.** Trivial → rule-based → single-feature → regularised linear → the candidate. Each rung answers "how much of my lift comes from this layer of complexity?" A candidate that beats a linear model by 0.5 pp of AUC is a different story than one that beats it by 15 pp.
3. **Error analysis is confusion-matrix-then-cluster, not confusion-matrix-alone.** Aggregate confusion (TP/FP/FN/TN) describes the balance of errors but hides where they live. Clustering the errors by feature values, slices, and content signals surfaces the pattern you can actually fix.
4. **Top-k worst errors are inspected individually.** For every model, the 25 worst mispredictions (by residual, by loss contribution, by confident-wrong) are read by a human. Many errors turn out to be label noise, not model error — you cannot tell from an aggregate number.
5. **Error analysis outputs change the next iteration.** The deliverable is a table of error clusters with counts, example IDs, and a "fix hypothesis" — a feature to add, a label to re-audit, or a slice to upweight. Analysis that does not change anything was not analysis.

This solution sits inside the NIST AI RMF `MEASURE 2.5` loop (performance characterised under realistic conditions) and `MEASURE 2.7` (quality and integrity assessed against error types that matter). It is also a frontline defence against OWASP ML Top 10 `ML02:2023 Data Poisoning Attack`: a concentrated error cluster near a specific feature-space region is often the earliest observable signal of poisoned training data.

| Choice | Alternative | Why this one |
|---|---|---|
| Baseline ladder of 4–5 models | A single "benchmark" | Each rung isolates a different source of lift; a single baseline collapses them |
| Rule-based baseline reflects the current production system | Only use ML baselines | If the candidate cannot beat the rules in production today, shipping ML is a regression even if it beats a linear model |
| Error clusters via feature-space slices + per-cluster rate | Raw residual histogram | Histograms are symmetric; a cluster view maps directly to a fix hypothesis |
| Top-k confident-wrong read by humans | Trust the aggregate | ~5–15 % of flagged "errors" are label noise; aggregates cannot tell |
| Per-slice bootstrap CI | Point estimates per slice | A slice with high error rate on n=30 is likely not actually worse |
| Output: `errors.md` with fix hypotheses | Output: a CSV | The deliverable is decisions, not data |
| Baselines logged to MLflow as regular runs | Baselines kept out of tracking | The comparison must be reproducible and auditable; a baseline is a model |

## 2. Worked implementation

### 2.1 Project layout

```
analysis/
├── __init__.py
├── baselines.py        # five baseline models + a scoring harness
├── errors.py           # per-example loss, confident-wrong, cluster assignment
├── slices.py           # slice definitions + per-slice metric + CI
├── report.py           # renders errors.md with clusters, examples, hypotheses
└── tests/
    └── test_baselines_improve_monotonically.py
```

### 2.2 `baselines.py` — the ladder

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    name: str
    auc: float
    pr_auc: float


def _score(y_true, proba) -> tuple[float, float]:
    return (
        float(roc_auc_score(y_true, proba)),
        float(average_precision_score(y_true, proba)),
    )


def trivial_majority(x_tr, y_tr, x_te, y_te) -> BaselineResult:
    model = DummyClassifier(strategy="most_frequent").fit(x_tr, y_tr)
    # DummyClassifier with "most_frequent" returns constant probs; use prior for AUC-friendliness.
    model = DummyClassifier(strategy="prior").fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    return BaselineResult("trivial_prior", *_score(y_te, proba))


def rule_based_from_production(x_tr, y_tr, x_te, y_te, rule_fn) -> BaselineResult:
    """rule_fn: DataFrame -> np.ndarray[float] in [0, 1]. Represents today's production."""
    proba = rule_fn(x_te)
    return BaselineResult("rules_in_production", *_score(y_te, proba))


def single_feature(x_tr, y_tr, x_te, y_te, feature: str) -> BaselineResult:
    model = LogisticRegression().fit(x_tr[[feature]], y_tr)
    proba = model.predict_proba(x_te[[feature]])[:, 1]
    return BaselineResult(f"single_feature[{feature}]", *_score(y_te, proba))


def regularised_linear(x_tr, y_tr, x_te, y_te) -> BaselineResult:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ]).fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    return BaselineResult("regularised_linear", *_score(y_te, proba))


def candidate_gbm(x_tr, y_tr, x_te, y_te) -> BaselineResult:
    model = GradientBoostingClassifier(random_state=42).fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    return BaselineResult("candidate_gbm", *_score(y_te, proba))


def run_ladder(x_tr, y_tr, x_te, y_te, rule_fn, best_single_feature: str) -> list[BaselineResult]:
    return [
        trivial_majority(x_tr, y_tr, x_te, y_te),
        rule_based_from_production(x_tr, y_tr, x_te, y_te, rule_fn),
        single_feature(x_tr, y_tr, x_te, y_te, best_single_feature),
        regularised_linear(x_tr, y_tr, x_te, y_te),
        candidate_gbm(x_tr, y_tr, x_te, y_te),
    ]
```

The ladder is read top to bottom; the gap between consecutive rungs is the thing worth discussing.

- Trivial → rules: how much of the "lift" is just reproducing the current production system?
- Rules → single feature: can one well-chosen feature match the rules? If yes, the rules are probably just encoding that feature.
- Single feature → linear: how much does interaction and multi-feature combination add?
- Linear → GBM: how much does non-linearity and interactions add?

If the last gap is small, the candidate model is a linear model wearing a complicated hat.

### 2.3 `errors.py` — where the candidate loses

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ErrorRow:
    index: int
    y_true: int
    y_proba: float
    log_loss: float
    is_confident_wrong: bool


def per_example_errors(x_te: pd.DataFrame, y_te: pd.Series, proba: np.ndarray,
                       confidence_threshold: float = 0.8) -> pd.DataFrame:
    y = y_te.values.astype(int)
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    log_loss_per = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    predicted = (p >= 0.5).astype(int)
    wrong = predicted != y
    confident = np.maximum(p, 1 - p) >= confidence_threshold
    confident_wrong = wrong & confident

    out = x_te.copy()
    out["y_true"] = y
    out["y_proba"] = p
    out["log_loss"] = log_loss_per
    out["is_wrong"] = wrong
    out["is_confident_wrong"] = confident_wrong
    return out


def top_k_worst(errors: pd.DataFrame, k: int = 25) -> pd.DataFrame:
    return errors.sort_values("log_loss", ascending=False).head(k)


def confident_wrong_only(errors: pd.DataFrame) -> pd.DataFrame:
    return errors[errors["is_confident_wrong"]].sort_values("log_loss", ascending=False)
```

The two views do different work:

- **Top-k by log_loss** finds the single most expensive wrong predictions. A labelling bug shows up here first because the label is wrong *and* the model was confident otherwise.
- **Confident-wrong** finds the pattern that matters for user trust: cases where the model is *not hedging* and is *still wrong*. These are the production incidents waiting to happen.

### 2.4 `slices.py` — slice metrics with CIs

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class SliceMetric:
    slice_name: str
    n: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    error_rate: float


def _bootstrap_auc(y_true: np.ndarray, proba: np.ndarray, n_boot: int = 1_000,
                   rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[sample])) < 2:
            continue
        scores.append(roc_auc_score(y_true[sample], proba[sample]))
    if not scores:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def slice_metrics(df: pd.DataFrame, slices: dict[str, pd.Series], y_col: str,
                  proba_col: str, min_n: int = 100) -> list[SliceMetric]:
    out: list[SliceMetric] = []
    for name, mask in slices.items():
        sub = df[mask]
        if len(sub) < min_n or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values.astype(int)
        p = sub[proba_col].values
        lo, hi = _bootstrap_auc(y, p)
        out.append(SliceMetric(
            slice_name=name,
            n=len(sub),
            auc=float(roc_auc_score(y, p)),
            auc_ci_low=lo,
            auc_ci_high=hi,
            error_rate=float(np.mean((p >= 0.5).astype(int) != y)),
        ))
    return sorted(out, key=lambda s: s.auc)
```

Slices with `n < min_n` are *skipped*, not reported as a bad number. A per-slice AUC on n=25 is noise, and a report that includes it invites bad decisions.

### 2.5 `report.py` — write `errors.md`

```python
from pathlib import Path
import pandas as pd

from .baselines import BaselineResult
from .errors import top_k_worst, confident_wrong_only
from .slices import SliceMetric


TEMPLATE = """# Error Analysis — {task}

## Baseline ladder
| Rung | Model | AUC | PR-AUC | Δ AUC vs prev |
|---|---|---:|---:|---:|
{ladder_rows}

Gap interpretation:
{ladder_commentary}

## Slice performance (sorted worst-first)
| Slice | n | AUC | 95 % CI | Error rate |
|---|---:|---:|---|---:|
{slice_rows}

## Top-{k} worst-loss examples
{top_k_table}

## Error clusters (manually reviewed)
{clusters_section}

## Fix hypotheses — ordered by expected payoff
{fix_hypotheses}
"""


def render(path: Path, task: str, ladder: list[BaselineResult],
           slice_metrics: list[SliceMetric], errors: pd.DataFrame,
           clusters_md: str, fix_hypotheses_md: str, k: int = 25) -> None:
    rows = []
    prev = None
    for b in ladder:
        delta = "" if prev is None else f"{(b.auc - prev):+.3f}"
        rows.append(f"| {len(rows)+1} | {b.name} | {b.auc:.3f} | {b.pr_auc:.3f} | {delta} |")
        prev = b.auc
    ladder_rows = "\n".join(rows)

    slice_rows = "\n".join(
        f"| {s.slice_name} | {s.n} | {s.auc:.3f} | [{s.auc_ci_low:.3f}, {s.auc_ci_high:.3f}] | {s.error_rate:.3f} |"
        for s in slice_metrics
    )

    top_k = top_k_worst(errors, k=k)[["y_true", "y_proba", "log_loss"]].to_markdown(index=True)

    path.write_text(TEMPLATE.format(
        task=task,
        ladder_rows=ladder_rows,
        ladder_commentary="(Fill in by hand after reading the deltas above.)",
        slice_rows=slice_rows,
        k=k, top_k_table=top_k,
        clusters_section=clusters_md,
        fix_hypotheses=fix_hypotheses_md,
    ))
```

### 2.6 Example `errors.md` (filled)

The rendered artifact — the deliverable — looks like this when complete:

```markdown
# Error Analysis — Payments Fraud Filter v3-candidate

## Baseline ladder
| Rung | Model | AUC | PR-AUC | Δ AUC vs prev |
|---|---|---:|---:|---:|
| 1 | trivial_prior | 0.500 | 0.006 |  |
| 2 | rules_in_production | 0.712 | 0.093 | +0.212 |
| 3 | single_feature[amount_usd] | 0.734 | 0.118 | +0.022 |
| 4 | regularised_linear | 0.814 | 0.187 | +0.080 |
| 5 | candidate_gbm | 0.871 | 0.291 | +0.057 |

Gap interpretation:
- Trivial → rules gives +0.212 AUC. The rules engine is doing real work; any model that doesn't
  beat +0.212 is net neutral vs today's production.
- Single feature → linear → GBM gives +0.137 combined. Interactions matter; the non-linearity
  (linear → GBM) contributes +0.057, which is where the win over "simple linear" lives.

## Slice performance (sorted worst-first)
| Slice | n | AUC | 95 % CI | Error rate |
|---|---:|---:|---|---:|
| issuer_bank=BankQ | 14 212 | 0.722 | [0.698, 0.745] | 0.034 |
| channel=CNP, country=XX | 8 905 | 0.741 | [0.712, 0.769] | 0.041 |
| card_type=prepaid | 11 044 | 0.766 | [0.742, 0.788] | 0.038 |
| channel=3DS | 102 553 | 0.912 | [0.907, 0.917] | 0.004 |

## Top-25 worst-loss examples
(table — redacted for brevity)

## Error clusters (manually reviewed; 25 examples each from top-k + confident-wrong)

- **Cluster A — prepaid cards, high amount, CNP.** 11 / 25 top-k errors. Model scores < 0.3;
  actual fraud. Rules engine would have caught 6 / 11. Feature `card_type` currently a
  one-hot; the GBM is treating prepaid as noise.
- **Cluster B — chargebacks coded as "goods not received" that look like legitimate returns.**
  7 / 25 top-k errors. These are label noise: the "fraud" label is applied retroactively
  by the chargeback team and the original transaction is indistinguishable from legitimate.
- **Cluster C — new-account first-transaction.** 5 / 25 top-k errors. Model has no history
  to lean on and defaults to the base rate. Fixable only by a cold-start feature.

## Fix hypotheses — ordered by expected payoff
1. **Interact `card_type=prepaid` with `amount_usd` buckets.** Addresses Cluster A directly.
   Expected: +0.01 AUC on worst-slice, no regression elsewhere.
2. **Sample-weight CNP-prepaid high-amount rows 2×.** Cheap; may help before the feature
   work lands. Expected: +0.3 pp recall-at-precision-0.9 on the worst slice.
3. **Flag Cluster B labels for the chargeback team to re-audit.** If the labels are noise,
   the headline AUC is pessimistic by ~2 pp.
4. **Add a cold-start feature: issuer-bank prior fraud rate for first-transaction accounts.**
   Addresses Cluster C. Requires data pipeline change; payoff unclear.
```

### 2.7 `tests/test_baselines_improve_monotonically.py`

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from analysis.baselines import run_ladder


def test_ladder_is_non_decreasing_on_synthetic_data():
    x, y = make_classification(n_samples=4000, n_features=10, n_informative=5,
                               weights=[0.7, 0.3], random_state=0)
    x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    y_s = pd.Series(y)
    split = len(x_df) // 2
    results = run_ladder(
        x_df.iloc[:split], y_s.iloc[:split],
        x_df.iloc[split:], y_s.iloc[split:],
        rule_fn=lambda df: (df["f0"] > df["f0"].median()).astype(float).values,
        best_single_feature="f0",
    )
    aucs = [r.auc for r in results]
    assert aucs[0] < aucs[-1], "candidate should beat trivial baseline"
    # Trivial is the only one allowed to be at-or-below everything else.
    assert aucs[0] <= min(aucs[1:]) + 0.01
```

A monotonic-ish ladder on synthetic data is a sanity check, not a correctness proof. Real ladders sometimes have *inversions* (the single-feature model beats linear on a weird dataset) — that inversion is itself a finding and belongs in the commentary, not swept under the test.

## 3. Validation steps

Run these in order.

1. **Static check** — `python -c "from analysis.baselines import run_ladder; from analysis.errors import per_example_errors"`.
2. **Unit tests** — `pytest analysis/tests -q`.
3. **Ladder sanity** — on a real eval set, print the ladder table. Candidate must beat trivial by a comfortable margin and must beat the rules-in-production baseline. If it does not beat rules, document why shipping is still worthwhile (e.g., lower latency, easier to retrain) or stop.
4. **Delta interpretation** — for each consecutive pair of rungs, write one sentence explaining the gap. An unexplained +15 pp jump between linear and GBM usually means a feature is non-monotone or interacting; an unexplained −2 pp jump is a sign the rung is misconfigured.
5. **Top-k human read** — open `top_k_worst` for the candidate. Manually label each as {label-noise, slice-A, slice-B, …}. Record the label-noise fraction; it is the floor on achievable error.
6. **Confident-wrong read** — same, for `confident_wrong_only`. These are the model's over-confident mistakes; each one deserves a cluster assignment.
7. **Slice table regression check** — compare slice metrics to the previous model version. Any slice whose AUC dropped by ≥ 3 pp (with non-overlapping CIs) is a regression even if the headline AUC went up. Block ship on unexplained slice regressions.
8. **Hypothesis follow-through** — every fix hypothesis gets an owner and a target version in the issue tracker. Analysis without follow-through is a diary.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Trivial baseline present | 1 | Dummy classifier/regressor scored on the same test set; a candidate that cannot beat it is rejected |
| 2 | Rule-based baseline from production | 2 | A function implementing (or approximating) the current production rules is in the ladder |
| 3 | Single-feature + linear rungs present | 2 | Both exist; they isolate feature-alone lift and multi-feature-linear lift |
| 4 | Candidate beats each rung with CIs | 2 | Not just a point-estimate win — the 95 % bootstrap CIs for each rung do not overlap the next |
| 5 | Top-k worst errors inspected | 3 | ≥ 25 confident-wrong + ≥ 25 top-log-loss examples manually read and cluster-labelled |
| 6 | Error clusters with example IDs | 3 | ≥ 3 named clusters, each with counts and ≥ 3 example IDs to reproduce the finding |
| 7 | Slice performance with sample sizes | 2 | Table with n, AUC, 95 % CI; slices below `min_n` skipped, not reported |
| 8 | Fix hypotheses with expected payoff | 2 | Ordered list; each hypothesis names a feature/data/sampling change and a measurable expected effect |
| 9 | Deliverable is a reviewable `errors.md` | 2 | Rendered markdown (or equivalent) committed to the repo; not a notebook cell |
| 10 | Analysis changes the next iteration | 1 | At least one hypothesis is picked up as a tracked issue or task |

Items 1, 5, and 8 are mandatory floors. A submission missing any of them cannot score above 11/20 regardless of the rest: no baseline → no claim; no human-read errors → no understanding; no fix hypotheses → no outcome.

## 5. Common mistakes

- **Reporting a candidate AUC with no baseline.** "The model gets 0.87 AUC" is meaningless without the trivial and rules baselines; the number could be 0.05 above or 0.30 above the right floor.
- **Treating the ladder as a formality.** The gaps between rungs are the point. A gap of +0.5 pp between regularised linear and the candidate means the candidate is not worth the complexity.
- **Only looking at the confusion matrix.** TP/FP/FN/TN is an aggregate; it does not tell you *where* the errors live. Slice tables + clusters do.
- **Trusting the labels on worst-loss examples.** 5–15 % of worst-loss examples are typically label noise. Reporting them as "model errors" inflates the apparent error rate and misleads the fix prioritisation.
- **Slices without CIs.** A slice with a worse-looking AUC on n=40 is probably noise. Without bootstrap CIs, you cannot tell.
- **Slice list derived from model errors.** Choosing the slice to report *after* seeing where the model fails is researcher degrees-of-freedom again. Slice definitions should come from the metric rubric (exercise-01), not from the error table.
- **Error analysis that produces no action.** If the error-analysis document does not change which issues are open in the backlog, it was performative.
- **Error analysis on train, not on a held-out set.** Analyse errors on a split the model has not seen. Analysing training errors tells you only about memorisation.
- **Confusing "confident wrong" with "worst loss."** Different views; both needed. A low-confidence wrong answer and a high-confidence wrong answer are different failure modes with different fixes.
- **No commentary on the ladder.** The ladder is numbers; the commentary is the finding. A ladder shipped without commentary ends up re-interpreted differently by every reader.

## 6. References

- scikit-learn — `DummyClassifier`, `DummyRegressor`, and the `sklearn.metrics` module for ROC-AUC, PR-AUC, and log-loss. <https://scikit-learn.org/stable/modules/model_evaluation.html>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.5` (performance under realistic conditions, including baselines) and `MEASURE 2.7` (error types explicitly characterised). <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF 1.0 — `MANAGE 2.1` on documenting the limitations and failure modes surfaced by error analysis. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` describes how concentrated error patterns in a feature region can be the first observable symptom of a poisoned training set. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0031 Erode ML Model Integrity` and `AML.T0043 Craft Adversarial Data` as threat models in which error analysis on slices is part of the detection pipeline. <https://atlas.mitre.org/>
