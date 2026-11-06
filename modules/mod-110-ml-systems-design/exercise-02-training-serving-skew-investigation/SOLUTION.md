# exercise-02 — Training-Serving Skew Investigation · SOLUTION

Paired exercise: `lessons/mod-110-ml-systems-design/exercises/exercise-02-training-serving-skew-investigation.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to investigate a plausible training-serving skew incident: offline AUC is up, online precision is flat or worse, and the diff between what the model "saw" in training and what it sees at inference has to be found. The deliverable is a **reproducible investigation procedure** — the set of checks that a mid-career ML engineer walks through on day one of the incident — plus a short bundle of code that a learner can actually run against a captured-request log.

Design priorities, in order:

1. **Measure before hypothesizing.** The first artifact is a side-by-side comparison of feature distributions between training data and a sample of recent inference requests, using the same transform code. Opinions about which feature is skewed are cheap; a distributional diff is expensive to fake.
2. **Rank skew sources by prior probability, not by novelty.** The three most common root causes — one transform definition evaluated two different ways, point-in-time leakage in training, and missing-feature policy mismatch — account for the overwhelming majority of real incidents. Investigate them first, even if the team's gut says "model regression."
3. **One transform library, two call sites.** The structural fix is always the same: a single Python module (or compiled UDF) that both the training batch and the online serving layer call. Every other fix is scaffolding on top.
4. **Keep the investigation artifact.** The notebook, the queries, and the distributional plots are the incident's forensic record. Save them with the postmortem; the next skew incident will not be the same, but it will rhyme.
5. **Separate "skew" from "drift."** Skew is a train-vs-serve difference *right now*. Drift is a train-time-vs-serve-time difference that grows over time. The checks overlap but the mitigations do not: skew is fixed in code; drift is fixed by retraining.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Sample recent requests + replay through training transform | Trust the feature store "as of" API | A replay catches bugs in the "as of" API itself |
| PSI + KS test per feature | Single summary number | Per-feature signal localizes the fix; a summary number only tells you "something is broken" |
| Train on-the-fly in the notebook using the real training code | Reimplement the transform in notebook | Reimplementation defeats the purpose — you are testing the system, not the notebook |
| Compare *predictions* on identical rows, not just features | Feature-only diff | A feature diff can be invisible when two bugs cancel; a prediction diff is end-to-end |
| Point-in-time correctness check as its own step | Bundled into "data audit" | Point-in-time leakage is a frequent cause and deserves its own named check |

## 2. Implementation — worked investigation

### 2.1 Incident intake (first 30 minutes)

Produce a one-page written intake before opening a notebook:

```
When did metrics diverge?           v3.2 promoted 2026-10-12; online precision fell 18→14% over 4 days
What changed in v3.2?               New feature: user_distinct_merchants_30d
What is the online health signal?   precision@hold, computed daily from analyst labels
What is the offline eval?           AUC 0.912 on held-out slice 2026-09-01..2026-09-30
What was rolled out, to whom, how?  Full rollout day 1 (no canary — flagged in postmortem)
Any infra changes in the window?    Feature stream migrated from Kafka topic v1 to v2 on 2026-10-09
```

The infra change on 2026-10-09 is already a hypothesis before any data has been pulled — the kind of signal that makes experienced engineers look at pipelines before models.

### 2.2 The five-check procedure

Each check is cheap to run and eliminates a large class of causes. Run them in order; stop at the first one that reproduces the symptom.

#### Check 1 — Prediction replay on identical inputs

Capture recent serving requests (features already resolved) and replay them through the *training-time* model artifact and the *production* serving path. Compare predictions.

```python
# tools/skew/replay.py
import pandas as pd
import numpy as np
import joblib
import requests


def replay(serving_log_path: str, model_path: str, serving_url: str, n: int = 5000):
    """Replay serving-logged requests through both the training model artifact
    and the production serving endpoint. Return rows with |pred_diff| > 1e-6."""
    log = pd.read_parquet(serving_log_path).sample(n=n, random_state=42)

    model = joblib.load(model_path)
    local_pred = model.predict_proba(log.drop(columns=["request_id"]))[:, 1]

    online_pred = []
    for _, row in log.iterrows():
        r = requests.post(
            serving_url,
            json=row.drop("request_id").to_dict(),
            timeout=1.0,
        )
        online_pred.append(r.json()["score"])

    out = log[["request_id"]].copy()
    out["local_pred"] = local_pred
    out["online_pred"] = np.array(online_pred, dtype=float)
    out["diff"] = out["local_pred"] - out["online_pred"]
    return out[out["diff"].abs() > 1e-6]
```

A nonzero result here says: **the same features produce different predictions**. That is almost always a model-binary or model-version mismatch (wrong artifact loaded, stale pickle in prod, library version drift). It is the easiest-to-fix failure and worth ruling out first.

#### Check 2 — Feature-distribution diff, training vs. serving

Pull the latest N days of *resolved* feature vectors from the serving log and compare to the training set. Use the Population Stability Index (PSI) for numeric features and a chi-squared for categoricals. PSI > 0.2 is "action required" in standard practice; PSI > 0.1 is "investigate."

```python
# tools/skew/distribution.py
import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    eps = 1e-6
    quantiles = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected.dropna(), bins=quantiles)
    a_counts, _ = np.histogram(actual.dropna(), bins=quantiles)
    e_frac = e_counts / max(e_counts.sum(), 1)
    a_frac = a_counts / max(a_counts.sum(), 1)
    return float(np.sum((a_frac - e_frac) * np.log((a_frac + eps) / (e_frac + eps))))


def feature_skew_report(train_df: pd.DataFrame, serve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shared = sorted(set(train_df.columns) & set(serve_df.columns))
    for col in shared:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            score = psi(train_df[col], serve_df[col])
            kind = "psi"
        else:
            # chi-squared on top categories
            top = train_df[col].value_counts().head(20).index
            e = train_df[col].isin(top).value_counts(normalize=True)
            a = serve_df[col].isin(top).value_counts(normalize=True)
            score = float(((a - e) ** 2 / (e + 1e-6)).sum())
            kind = "chi2-top20"
        rows.append({"feature": col, "metric": kind, "score": score})
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out["flag"] = out["score"].where(out["metric"] != "psi", out["score"] > 0.2)
    return out
```

Features with the highest PSI are the suspects. In the worked incident, `user_distinct_merchants_30d` showed PSI = 0.47 — a glaring mismatch.

#### Check 3 — Transform parity (the usual culprit)

For each flagged feature, run the training-time transform code on a captured raw input and compare to the serving-time transform's output for the same input.

```python
# tools/skew/parity.py
from features.transforms import compute_user_aggregates  # one definition
import json
import pandas as pd


def parity_check(raw_capture_path: str) -> pd.DataFrame:
    """raw_capture is a parquet file where each row has the raw inputs
    *and* the serving-time computed features. Re-run the transform and diff."""
    raw = pd.read_parquet(raw_capture_path)
    rows = []
    for _, r in raw.iterrows():
        recomputed = compute_user_aggregates(json.loads(r["raw_user_events"]))
        served = json.loads(r["served_features"])
        for key in recomputed:
            if abs(recomputed[key] - served.get(key, 0)) > 1e-9:
                rows.append({
                    "request_id": r["request_id"],
                    "feature": key,
                    "training_eval": recomputed[key],
                    "serving_eval": served.get(key),
                })
    return pd.DataFrame(rows)
```

In the worked incident this is where the smoking gun shows up: `compute_user_aggregates` running at training time reads from the historical topic `events.v2`; the serving layer, due to an incomplete config rollout, still reads from `events.v1`, which had a narrower event schema and silently dropped a subset of merchants. The 30-day distinct-merchant count was always smaller online than offline. PSI 0.47 explained.

#### Check 4 — Point-in-time correctness

Rare but fatal when it happens. Verify that each training aggregate is computed using *only* events dated strictly before the training label's event-time. The canonical bug is joining features by user-id without an event-time predicate, which leaks future information into training and makes offline AUC look wonderful.

```python
# tools/skew/point_in_time.py
def point_in_time_audit(train_df, event_log):
    """For a sample of training rows, re-derive the aggregate at label time
    using only pre-label events, and check against the training value."""
    import pandas as pd
    violations = []
    for _, r in train_df.sample(1000, random_state=0).iterrows():
        pre = event_log[
            (event_log["user_id"] == r["user_id"])
            & (event_log["event_ts"] < r["label_ts"])
        ]
        correct = pre["merchant_id"].nunique()
        if abs(correct - r["user_distinct_merchants_30d"]) > 0:
            violations.append({
                "user_id": r["user_id"],
                "label_ts": r["label_ts"],
                "training_value": r["user_distinct_merchants_30d"],
                "pit_correct_value": correct,
            })
    return pd.DataFrame(violations)
```

If this returns nonempty, the fix is in the training pipeline's feature join, not in the serving layer. Offline eval must be rerun against the corrected training set before anything is promoted.

#### Check 5 — Missing-feature policy

The least glamorous and the most frequently missed. Training almost always sees some nulls (users without 30 days of history); serving almost always sees more. If training resolves missing with the column mean and serving resolves missing with `0` or with a sentinel, the model has learned the wrong semantic.

```python
# tools/skew/missing_policy.py
def missing_rate_report(train_df, serve_df):
    import pandas as pd
    rows = []
    for col in set(train_df.columns) & set(serve_df.columns):
        t_rate = float(train_df[col].isna().mean())
        s_rate = float(serve_df[col].isna().mean())
        rows.append({
            "feature": col,
            "train_missing": t_rate,
            "serve_missing": s_rate,
            "delta": s_rate - t_rate,
        })
    return pd.DataFrame(rows).sort_values("delta", ascending=False)
```

Missing-rate deltas > 5 absolute percentage points point to a policy mismatch. The fix is to encode the missing policy in the shared transform library and to log the resolved value's provenance (`"user_distinct_merchants_30d_imputed": true`) on every serving request, so future audits do not have to guess.

### 2.3 Structural fix

The investigation's output is a one-paragraph diagnosis plus a code-level fix. For the worked incident:

> `user_distinct_merchants_30d` was computed over a stream (`events.v1`) at serving time that was not the stream used at training time (`events.v2`). The serving-side feature pipeline's topic config was not updated during the 2026-10-09 migration. Fix: unify to `events.v2`, backfill the serving feature store, and add a CI check that fails when the training transform and the serving transform resolve the same raw payload to different feature vectors on a shared fixture.

The CI check is the lasting fix. It is a 20-row fixture of raw payloads plus their expected transformed features, run by both the training pipeline's test suite and the serving service's test suite, in the same repo. The investigation's notebook is retired; the fixture is permanent.

### 2.4 Why one-transform-library is the only durable answer

The ad-hoc fix — "rewrite the serving-side feature pipeline carefully and hope" — is the fix that produces the next incident. The structural answer is a library (plain Python package, SQL views materialized the same way, or a compiled UDF) that both callers import or invoke. The library enforces that `compute_user_aggregates(payload)` is the *only* function that defines the feature, and every change goes through its tests.

This is the control that generalizes: it is not specific to the fraud system, and the same pattern applies to recommendation features, ranking features, and content moderation features. The feature-store products on the market exist to make this control easy to adopt — but the control itself is the point, not the vendor.

## 3. Validation steps

Run these in order against the exercise's provided incident log (or a synthetic one generated from a known-skewed fixture).

1. **Replay sanity.** On a 5,000-row replay, `local_pred` and `online_pred` should match within `1e-6` for ≥ 99% of rows when no skew is present. A failure here is a model-version/library-version drift and must be fixed before proceeding.
2. **PSI baseline.** On the same training set split into two time slices, PSI should be `< 0.1` for most features. Features with PSI > 0.2 in the no-drift baseline indicate a noisy metric and should be reviewed before interpreting the live comparison.
3. **Parity fixture.** The 20-row fixture of raw payloads must produce byte-identical outputs from both the training and serving transform. The CI job runs on every PR; a learner's solution is incomplete without this fixture.
4. **Point-in-time audit zero.** On a 1,000-row sample, the audit must return zero violations for a correct training pipeline. A nonzero result is a training-data bug.
5. **Missing-rate delta.** For every feature, `|train_missing - serve_missing|` must be ≤ 5 percentage points, or the shared transform must log a `_imputed` provenance field so audits can separate real values from imputed ones.
6. **End-to-end precision reproduction.** After the fix, re-score the last 14 days of serving-logged requests with the corrected transform and compare to the actually-served scores. The gap should close to within the pre-incident variance.
7. **Regression guard.** Intentionally reintroduce the bug (point the serving transform back at `events.v1`), run the five checks, and confirm the investigation localizes to `compute_user_aggregates` within check 3. This verifies the procedure itself, not just the fix.

## 4. Rubric / review checklist

Weighted, `/22` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Incident intake | 2 | One-page written intake; names infra changes within the window as candidate hypotheses |
| 2 | Prediction replay | 3 | Side-by-side `local_pred` vs. `online_pred` on ≥ 1,000 captured rows with a clear pass/fail threshold |
| 3 | Per-feature distribution diff | 3 | PSI (numeric) + chi-squared (categorical); sorted; threshold stated |
| 4 | Transform parity check | 4 | Re-runs the *training* transform on captured raw inputs and diffs against served features |
| 5 | Point-in-time correctness | 3 | Explicit audit; sample size and pass criterion stated |
| 6 | Missing-feature policy | 2 | Train vs. serve missing-rate delta surfaced; imputation policy is part of the shared transform |
| 7 | Structural fix | 3 | One shared transform library; CI fixture; not an ad-hoc patch |
| 8 | Investigation artifact | 1 | Notebook/queries committed with the postmortem; reproducible by a different engineer |
| 9 | Skew vs. drift distinction | 1 | Stated explicitly; mitigation pathways are different |

Items 3, 4, and 7 are mandatory: a submission missing the distributional diff, the parity check, or the shared transform fix cannot score above 12 — those three steps are the *point* of the exercise, and skipping them signals the learner did not internalize the usual failure modes.

## 5. Common mistakes

- **Starting with model debugging.** The investigation opens with "let me retrain and see if it fixes it." Retraining in response to a skew incident is a non-answer: the next deploy will reproduce the bug because the serving path is still wrong.
- **One summary metric.** Reporting "overall data drift = 0.3" without per-feature breakdown. The fix requires localizing to a feature; an overall number cannot localize.
- **Reimplementing the transform in the notebook.** The whole point is to test the system's transform, not the investigator's reimplementation of it. Import the training transform; do not rewrite it.
- **Confusing skew with drift.** A serving-time distribution different from training is not automatically "drift." Drift evolves over days/weeks; a step-change at an infra deploy is a skew incident.
- **Ignoring imputation policy.** Features with low PSI can still skew the model if the missing-value policy differs. A silently-imputed `0` vs. a silently-imputed mean produces different predictions even when the observed distribution looks identical.
- **Fixing the serving layer in isolation.** The investigation fixes the serving-side bug and declares victory. Six weeks later, a new feature is added and the same bug returns because there is still no shared library. The structural fix is the deliverable, not the patch.
- **No CI fixture.** The investigation's notebook is the only artifact. Nothing fails at PR time when the next engineer reintroduces the bug. Fixtures turn one-time investigations into permanent guardrails.
- **Not checking point-in-time correctness.** The team assumes the offline eval is trustworthy. Half the hardest skew bugs are label-leakage in training, which inflates offline AUC and makes the online gap look mysterious.
- **No prediction-replay check.** The investigator jumps straight to distributional analysis and misses a trivial model-version mismatch, spending days on a two-line fix.
- **Using production credentials from the investigation notebook.** The investigator's notebook ends up with write access to the feature store. Investigations are read-only; fixes land through the normal change path.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MEASURE 2.6` (test for performance characteristics in deployed systems) motivates the per-feature distributional audit as an operational practice, not a one-off task. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML08:2023 Model Skewing` names training-serving skew as a security-relevant failure mode, since an adversary can induce skew by perturbing the serving-side data pipeline. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0020 Poison Training Data` and `AML.T0051 LLM Prompt Injection` techniques both exercise skew-adjacent failure modes; the drift/skew monitors from this exercise are the detection surface for both. <https://atlas.mitre.org/>
