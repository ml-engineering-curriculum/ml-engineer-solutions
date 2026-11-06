# exercise-02 — Data Drift Detection · SOLUTION

Paired exercise: `lessons/mod-109-ml-monitoring-drift/exercises/exercise-02-data-drift-detection.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to build a drift-detection job that compares the current production feature distribution to the training-time reference, decides whether each feature has drifted, and raises a bounded signal to downstream alerting. The failure mode to avoid is twofold: a p-value-trigger-happy job that pages the on-call every time a Monday-morning traffic shift renormalises on Tuesday, and a conservative job that stays silent while AUC bleeds out.

Design priorities, in order:

1. **Separate "it changed" from "we care."** The statistical test decides whether the distribution moved; a *threshold policy* decides whether the move matters. Mixing the two produces either flaky alerts or a model that silently degrades for a week.
2. **Pick the test by feature type, not by preference.** Numeric features use Kolmogorov–Smirnov or Wasserstein distance; categorical features use chi-square or PSI with laplace smoothing; low-cardinality integer features (0/1 flags) are categorical. Picking one test for all feature types produces nonsense for at least one of them.
3. **Reference distribution is the training set, versioned.** Comparing production to "last week of production" is comparing drift to itself; the training set is the only distribution the model was actually *learned on*, and it is the reference every drift verdict must trace back to.
4. **Scale the window to the slowest signal.** Hourly windows catch outages and feature-pipeline breaks; daily windows catch upstream-source drift; weekly windows catch population drift. The job emits all three, aligned to the training-reference, so a responder can look at the right horizon for the hypothesis they are testing.
5. **Produce a verdict, a score, and an evidence pack.** A verdict is boolean and alertable. A score is monotonic and dashboardable. An evidence pack (the histograms that were compared, the sample sizes, the p-value, the policy threshold) is what a reviewer reads to decide whether to retrain, roll back, or wait.

This design sits inside the NIST AI RMF `MEASURE 2.4` (continuous monitoring) and `MEASURE 2.11` (context-change detection) expectations, and it is the detection surface for the OWASP ML Top 10 `ML02:2023 Data Poisoning Attack` and MITRE ATLAS `AML.T0020 Poison Training Data` (the symptom shows up as production-side distribution shift long before the labels do).

| Choice | Alternative considered | Why this one |
|---|---|---|
| KS test + PSI side-by-side for numeric | KS only | PSI is what the risk-modeling world uses as the alert threshold; KS is what statisticians use as the hypothesis test. Both are cheap; report both and use PSI for the thresholded policy |
| Chi-square with Laplace smoothing for categorical | Chi-square raw | A new category with count 0 in reference makes raw chi-square explode; smoothing with +1 keeps the test numerically sane and surfaces the new category as a separate `new_categories_total` counter |
| Reference sample = stratified sample from training set, hashed and stored as an artifact | Live query against the training database | Reproducibility: the reference is pinned to the model version, and the comparison yesterday is the same comparison tomorrow |
| Windows: 1h, 24h, 7d rolling | Fixed-day windows | Rolling windows avoid phase-of-day aliasing; fixed windows compress a 24h cyclical pattern into a dirac spike at midnight |
| PSI policy: <0.1 stable, 0.1–0.25 watch, >0.25 drift | Fire on any PSI > 0 | The 0.1 / 0.25 thresholds are the credit-risk convention and are conservative enough to not page on noise |
| Alerts tiered: info/warn/critical | Boolean fire-or-not | Analysts need "watch this feature" without waking a human; the info tier is the lane |
| Bonferroni correction across features | No correction | Testing 40 features at α=0.05 produces ~2 false alerts per run even on identical distributions; correcting for multiple comparisons is the cost of monitoring many features at once |

## 2. Worked implementation

### 2.1 Project layout

```
drift/
├── __init__.py
├── reference.py     # build/load the versioned reference sample
├── tests.py         # per-feature statistical tests
├── policy.py        # verdict thresholds and tiers
├── window.py        # rolling-window aggregation of production features
├── report.py        # evidence pack: numbers + the distributions compared
└── job.py           # orchestrates one drift evaluation
```

### 2.2 `reference.py` — the pinned, versioned reference

```python
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import pandas as pd

@dataclass(frozen=True)
class Reference:
    model_version: str
    data_sha: str
    numeric: dict[str, pd.Series]      # feature -> values
    categorical: dict[str, pd.Series]  # feature -> values (as strings)

def build_reference(
    training_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_version: str,
    sample_n: int = 50_000,
    random_state: int = 7,
) -> Reference:
    sample = training_df.sample(
        n=min(sample_n, len(training_df)), random_state=random_state
    )
    payload = sample.to_csv(index=False).encode()
    data_sha = hashlib.sha256(payload).hexdigest()
    return Reference(
        model_version=model_version,
        data_sha=data_sha,
        numeric={f: sample[f].dropna() for f in numeric_features},
        categorical={f: sample[f].dropna().astype(str) for f in categorical_features},
    )

def save(ref: Reference, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "model_version": ref.model_version,
        "data_sha": ref.data_sha,
        "numeric": list(ref.numeric),
        "categorical": list(ref.categorical),
    }, indent=2))
    for name, s in ref.numeric.items():
        s.to_frame(name=name).to_parquet(out_dir / f"num_{name}.parquet")
    for name, s in ref.categorical.items():
        s.to_frame(name=name).to_parquet(out_dir / f"cat_{name}.parquet")
```

The reference is an *artifact* keyed on the model version and a SHA of the sampled rows. A drift verdict that cannot cite the reference it was compared against is unreviewable. Stratified sampling (omitted above for brevity; add `stratify=training_df[target]` for classification) keeps the class balance of the reference matched to the training set — otherwise the drift test confuses class imbalance with drift.

### 2.3 `tests.py` — per-feature statistical tests

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class DriftTestResult:
    feature: str
    kind: str             # "numeric" | "categorical"
    ks_statistic: float | None
    ks_pvalue: float | None
    wasserstein: float | None
    chi2_statistic: float | None
    chi2_pvalue: float | None
    psi: float
    n_reference: int
    n_current: int
    new_categories: list[str]

def psi(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    ref_p = (ref_hist + 1) / (ref_hist.sum() + len(bins) - 1)
    cur_p = (cur_hist + 1) / (cur_hist.sum() + len(bins) - 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

def psi_categorical(reference: pd.Series, current: pd.Series) -> tuple[float, list[str]]:
    categories = sorted(set(reference) | set(current))
    ref_counts = reference.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = current.value_counts().reindex(categories, fill_value=0).to_numpy()
    ref_p = (ref_counts + 1) / (ref_counts.sum() + len(categories))
    cur_p = (cur_counts + 1) / (cur_counts.sum() + len(categories))
    new = sorted(set(current) - set(reference))
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))), new

def test_numeric(name, reference: pd.Series, current: pd.Series) -> DriftTestResult:
    ks = stats.ks_2samp(reference.to_numpy(), current.to_numpy(), method="asymp")
    w = float(stats.wasserstein_distance(reference.to_numpy(), current.to_numpy()))
    bins = np.quantile(reference.to_numpy(), np.linspace(0, 1, 11))
    bins = np.unique(np.concatenate(([-np.inf], bins, [np.inf])))
    return DriftTestResult(
        feature=name, kind="numeric",
        ks_statistic=float(ks.statistic), ks_pvalue=float(ks.pvalue),
        wasserstein=w, chi2_statistic=None, chi2_pvalue=None,
        psi=psi(reference.to_numpy(), current.to_numpy(), bins),
        n_reference=len(reference), n_current=len(current),
        new_categories=[],
    )

def test_categorical(name, reference: pd.Series, current: pd.Series) -> DriftTestResult:
    categories = sorted(set(reference) | set(current))
    ref_counts = reference.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = current.value_counts().reindex(categories, fill_value=0).to_numpy()
    ref_counts_sm = ref_counts + 1
    cur_counts_sm = cur_counts + 1
    chi2 = stats.chi2_contingency(np.vstack([ref_counts_sm, cur_counts_sm]))
    p, new = psi_categorical(reference, current)
    return DriftTestResult(
        feature=name, kind="categorical",
        ks_statistic=None, ks_pvalue=None, wasserstein=None,
        chi2_statistic=float(chi2.statistic), chi2_pvalue=float(chi2.pvalue),
        psi=p, n_reference=len(reference), n_current=len(current),
        new_categories=new,
    )
```

The decile-based bins for PSI (`np.quantile(..., np.linspace(0, 1, 11))`) are computed on the reference, not the current — the reference is the "ruler" and the current is being measured against it. Reversing that direction is a common bug that makes PSI unstable.

### 2.4 `policy.py` — verdict thresholds

```python
from dataclasses import dataclass
from .tests import DriftTestResult

STABLE = "stable"
WATCH = "watch"
DRIFT = "drift"

@dataclass(frozen=True)
class DriftPolicy:
    psi_watch: float = 0.10          # Siddiqi 2006 credit-scoring convention
    psi_drift: float = 0.25
    min_samples_current: int = 1_000
    pvalue_alpha: float = 0.001      # tight because we're testing many features

def verdict(result: DriftTestResult, policy: DriftPolicy) -> str:
    if result.n_current < policy.min_samples_current:
        return "insufficient_data"
    if result.psi >= policy.psi_drift:
        return DRIFT
    if result.psi >= policy.psi_watch:
        return WATCH
    if result.new_categories:
        return WATCH
    return STABLE
```

PSI drives the verdict, not the p-value. With 10 M production rows per window, almost any numeric distribution will reject the KS null — the p-value becomes a sample-size thermometer, not a drift signal. PSI is a distance measure; its threshold does not shrink with sample size. The 0.10 / 0.25 PSI thresholds are the widely-cited credit-risk convention (Siddiqi, *Credit Risk Scorecards*, 2006); treat them as a sensible default anchor and tighten per-feature once the on-call has observed a few weeks of production PSI for that feature.

### 2.5 `window.py` — rolling production aggregation

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    label: str  # "1h" | "24h" | "7d"

def windows_from(now: datetime) -> list[Window]:
    return [
        Window(now - timedelta(hours=1), now, "1h"),
        Window(now - timedelta(days=1), now, "24h"),
        Window(now - timedelta(days=7), now, "7d"),
    ]

def load_production(
    predictions_table: pd.DataFrame,
    window: Window,
    features: list[str],
) -> pd.DataFrame:
    mask = (predictions_table["ts"] >= window.start) & (
        predictions_table["ts"] < window.end
    )
    return predictions_table.loc[mask, features]
```

Three windows run every hour. The 1-hour window catches feature-pipeline breaks (a null spike shows up in minutes); the 24-hour window catches upstream-source drift (a new vendor file format); the 7-day window catches slower population shifts. A single "last 24 hours" window would miss both ends of the spectrum.

### 2.6 `report.py` — the evidence pack

```python
from dataclasses import asdict
import json
from pathlib import Path

def write_report(results, verdicts, policy, model_version, window, out_dir: Path):
    payload = {
        "model_version": model_version,
        "window": {"start": window.start.isoformat(),
                   "end": window.end.isoformat(),
                   "label": window.label},
        "policy": asdict(policy),
        "features": [
            {**asdict(r), "verdict": v}
            for r, v in zip(results, verdicts)
        ],
        "summary": {
            "n_drift": sum(v == "drift" for v in verdicts),
            "n_watch": sum(v == "watch" for v in verdicts),
            "n_stable": sum(v == "stable" for v in verdicts),
            "n_insufficient": sum(v == "insufficient_data" for v in verdicts),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"drift-{window.label}-{model_version}.json").write_text(
        json.dumps(payload, indent=2)
    )
```

Every alert links back to a report file. The on-call does not have to replay the job to see the evidence; they open the JSON and see the comparison that produced the verdict.

### 2.7 `job.py` — one evaluation, orchestrated

```python
from datetime import datetime
from pathlib import Path
from prometheus_client import Gauge

from .reference import Reference
from .tests import test_numeric, test_categorical
from .policy import DriftPolicy, verdict
from .window import windows_from, load_production
from .report import write_report

drift_psi = Gauge(
    "ml_feature_psi",
    "Population stability index per feature, labelled by window and model.",
    ("model_version", "feature", "window"),
)
drift_verdict = Gauge(
    "ml_feature_drift_verdict",
    "0=stable, 1=watch, 2=drift, -1=insufficient_data.",
    ("model_version", "feature", "window"),
)
VERDICT_CODE = {"stable": 0, "watch": 1, "drift": 2, "insufficient_data": -1}


def evaluate(reference: Reference, production_df, policy: DriftPolicy,
             out_dir: Path, now: datetime | None = None):
    now = now or datetime.utcnow()
    for window in windows_from(now):
        current = load_production(
            production_df, window,
            list(reference.numeric) + list(reference.categorical),
        )
        results, verdicts = [], []
        for name, ref in reference.numeric.items():
            r = test_numeric(name, ref, current[name].dropna())
            v = verdict(r, policy)
            results.append(r); verdicts.append(v)
            drift_psi.labels(reference.model_version, name, window.label).set(r.psi)
            drift_verdict.labels(
                reference.model_version, name, window.label
            ).set(VERDICT_CODE[v])
        for name, ref in reference.categorical.items():
            r = test_categorical(name, ref, current[name].dropna().astype(str))
            v = verdict(r, policy)
            results.append(r); verdicts.append(v)
            drift_psi.labels(reference.model_version, name, window.label).set(r.psi)
            drift_verdict.labels(
                reference.model_version, name, window.label
            ).set(VERDICT_CODE[v])
        write_report(results, verdicts, policy,
                     reference.model_version, window, out_dir)
```

Metrics and reports both flow out: metrics drive alerts and dashboards, reports drive investigations. The current date is a parameter (`now`), not a call to `datetime.utcnow()` inside loops, so the job is deterministic for tests.

### 2.8 Alerting (Prometheus)

```yaml
- alert: FeatureDrift24h
  expr: max by (feature, model_version)(ml_feature_drift_verdict{window="24h"}) == 2
  for: 1h
  annotations:
    summary: "Feature {{ $labels.feature }} drifted (PSI>0.25) for 1h on {{ $labels.model_version }}."

- alert: ManyFeaturesWatch24h
  expr: count by (model_version)(ml_feature_drift_verdict{window="24h"} == 1) > 5
  for: 2h
  annotations:
    summary: ">5 features in WATCH state simultaneously — likely upstream change."
```

The second rule is deliberate: one feature in `watch` is noise; five features simultaneously in `watch` is almost always a shared upstream (schema, feature-pipeline bug, vendor change), and catching the correlated case saves five pages.

## 3. Validation steps

1. **Static check** — `python -c "from drift.job import evaluate"` loads tests, policy, and metrics without error.
2. **Reference round-trip** — build a reference from a known dataframe, save, load, and confirm the `data_sha` matches and the categorical/numeric splits are preserved.
3. **No-drift baseline** — compare the reference to itself (identical dataframe). Every feature must return `psi ≈ 0`, every verdict must be `stable`, no `new_categories`.
4. **Synthetic numeric shift** — shift a numeric feature by +1 sd on the current window. KS statistic must rise, PSI must exceed `watch` and (with a 2-sd shift) `drift`, verdict must change accordingly.
5. **Synthetic new category** — inject a previously-unseen categorical value into the current window. `new_categories` must list it; verdict must be at least `watch` even if PSI is below threshold.
6. **Low-sample guard** — call with `n_current < min_samples_current`. Verdict must be `insufficient_data`; no spurious `drift` fire.
7. **Rolling-window parity** — run the job at hour `T` and again at `T+1h` with identical production data. The 24h and 7d windows must produce near-identical PSI (sampling noise aside); the 1h window may differ. This catches off-by-one bugs in window boundaries.
8. **Alert rule tests** — `promtool test rules` against a fixture where feature `X` has PSI=0.4 for 70 minutes. `FeatureDrift24h` must fire, no others.
9. **Reproducibility** — save a reference under `model_version=v3`; re-run six months later with the same code and same reference artifact. Verdicts for an unchanged production slice must match.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Reference versioned to the model | 3 | Reference artifact carries `model_version` + `data_sha`; the job refuses to compare against the wrong one |
| 2 | Correct test per feature type | 3 | Numeric → KS/Wasserstein + PSI; categorical → chi-square + PSI with Laplace smoothing; no cross-contamination |
| 3 | PSI as policy, p-value as diagnostic | 2 | PSI thresholds drive the verdict; p-value is reported but does not fire alerts alone |
| 4 | Multi-window evaluation | 2 | At least 1h/24h/7d windows; 24h is the primary alert signal |
| 5 | New-category detection | 2 | Previously-unseen categorical values trigger at least `watch`, independent of PSI |
| 6 | Verdict tiers | 2 | `stable` / `watch` / `drift` / `insufficient_data`; `insufficient_data` suppresses firing |
| 7 | Evidence pack | 2 | Per-feature JSON report with PSI, p-values, sample sizes, new categories, policy, windows |
| 8 | Metrics for dashboarding | 2 | PSI and verdict both exported as Prometheus gauges with `(model_version, feature, window)` labels |
| 9 | Alert correlation rule | 2 | Separate rule for "many features in WATCH" — not just per-feature fires |

Items 1 and 2 are mandatory: a drift job comparing production to live production, or running KS on categorical features, is not actually detecting drift — it is producing shaped noise.

## 5. Common mistakes

- **Reference = last week of production.** Comparing production to itself means a slow degradation gets normalised day by day. The reference is the training set, pinned.
- **One test for all feature types.** Running KS on a categorical column encoded as integers produces statistically meaningful nonsense. Pick the test by feature type.
- **Firing on p-value alone.** With a million rows per window, every real feature is statistically different from reference. p-value is not a policy — PSI (or an equivalent distance) is.
- **No Laplace smoothing on new categories.** A category with reference count 0 makes raw chi-square and raw PSI undefined; smoothing keeps the test defined and lets `new_categories` surface separately.
- **Not stratifying the reference.** If the training set is 10% positives and the reference sample is 5% positives, every drift verdict on class-dependent features is wrong.
- **Testing features in isolation and ignoring the correlated case.** Five features in WATCH is almost always a shared upstream; a rule that only fires per-feature misses it.
- **No `insufficient_data` state.** Firing WATCH on a 100-row window is dashboarding noise, not drift. Guard with a minimum sample size.
- **Binning on the current window.** PSI bins are decided on the reference; otherwise two comparisons against the same reference produce different PSI depending on the current shape, and the metric stops being interpretable over time.
- **Not correcting for multiple comparisons.** 40 features × α=0.05 ≈ 2 false fires per run. Use a tighter α (e.g. 0.001) or a Bonferroni correction when p-values are consumed anywhere in the policy.

## 6. References

- SciPy — `scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance`, `scipy.stats.chi2_contingency`; the implementations used above. <https://docs.scipy.org/doc/scipy/reference/stats.html>
- Evidently AI — Open-source drift metric catalogue (PSI, KS, Wasserstein, Jensen–Shannon) as a practitioner reference for which test suits which feature type. <https://docs.evidentlyai.com/>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.4` (ongoing monitoring) and `MEASURE 2.11` (context-change detection). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack`; drift at the input is the first observable symptom of upstream manipulation. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0020 Poison Training Data`; the detection story is the same whether drift is natural or adversarial. <https://atlas.mitre.org/>
