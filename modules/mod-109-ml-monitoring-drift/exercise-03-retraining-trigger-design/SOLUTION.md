# exercise-03 — Retraining Trigger Design · SOLUTION

Paired exercise: `lessons/mod-109-ml-monitoring-drift/exercises/exercise-03-retraining-trigger-design.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to design a retraining policy for a deployed model — the rules that decide *when* retraining is initiated, *what* data it uses, *how* the new candidate is gated before replacing production, and *who* approves the swap. The failure modes to avoid are at both ends: a monthly cron that ships stale models through an outage window, and a "retrain on every alert" policy that churns through candidates no one has validated.

Design priorities, in order:

1. **Two-layer policy: trigger + gate.** The trigger decides retraining is *attempted*; the gate decides whether the result ships. Collapsing the two into a single "if drift then deploy" rule is how auto-retrain pipelines ship broken models at 03:00.
2. **Multiple trigger types, OR-combined with priorities.** Scheduled (slow baseline), drift-based (input signals), performance-based (label signals), and manual (humans spot something). Each catches failures the others miss. Priorities exist so a performance drop interrupts the current schedule instead of waiting for it.
3. **Label-based triggers are the strongest signal, but delayed.** Prediction quality is the thing we actually care about; drift is a leading indicator. Design for both, weighted by the label-delay horizon.
4. **Promotion is never automatic for high-impact models.** The gate produces a verdict (pass/fail/watch) and a candidate artifact; a human approval or a shadow-traffic bake-in produces the actual swap. The cost of a bad model is asymmetric with the cost of a delayed-by-24h model.
5. **Cost and safety guardrails are first-class.** Minimum training-data volume, maximum retrain frequency, budget cap, cooling-off period after a rollback. Without them, a feedback loop between a flaky drift test and a cron that retrains on drift will burn the training budget in a weekend.

This design sits inside the NIST AI RMF `MANAGE 4.1` (post-deployment monitoring and response) and `GOVERN 1.4` (change-management for AI systems) expectations, and `MEASURE 2.9` (performance monitoring in deployment). The threat-model view aligns with OWASP ML Top 10 `ML05:2023 Model Theft` and `ML06:2023 AI Supply Chain Attacks`: a retraining loop is itself part of the supply chain and a trigger that is adversarially reachable (e.g., fire on malicious inputs) becomes an attack vector.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Triggers are OR-combined, priorities break ties | Single-source trigger | Drift catches upstream changes, performance catches concept drift, schedule catches drift below thresholds; one source alone leaves gaps |
| Shadow-traffic bake-in before promotion | Immediate replace | A gate passes against a validation set; shadow traffic is the only thing that catches request-shape mismatches no validation set thought of |
| Promotion requires human approval for impact >= HIGH | Fully automated | The error budget for the business is spent by the responder, not the pipeline; auto-promote is fine for low-impact slices |
| Challenger-vs-champion gate on the same, frozen holdout | Compare on the candidate's own validation | Changing the ruler and the thing being measured at the same time gives every challenger a free win |
| Retrain cadence capped at `max_per_week=1` default | Unbounded | The marginal value of the 2nd retrain inside a week is almost always below the operational cost; a hard cap forces an investigation when a 2nd is requested |
| Cool-off: no retrain for 7 days after rollback | Retry immediately | After a rollback, the system is in an abnormal state; letting the humans investigate before the pipeline retries is cheaper than fighting the pipeline |
| Freshness window for training data is explicit per trigger | Always "last 90 days" | Performance-triggered retrain uses recent-enough labels; drift-triggered retrain uses a rolling window covering the drift period; cadence retrain uses the widest training window the model handles |

## 2. Worked implementation (design)

### 2.1 Policy document (what the pipeline reads)

```yaml
# retrain_policy.yaml
model: churn-v3
impact: HIGH  # LOW | MEDIUM | HIGH — used to decide whether auto-promote is allowed

triggers:
  - name: schedule
    kind: cron
    cron: "0 2 * * MON"             # weekly Monday 02:00 UTC
    priority: 3

  - name: drift
    kind: metric
    metric: ml_feature_drift_verdict
    expression: "count(verdict == 'drift' for 24h window) >= 2"
    min_dwell_hours: 24
    priority: 2

  - name: performance
    kind: metric
    metric: ml_rolling_auc_7d
    expression: "ml_rolling_auc_7d < 0.78"    # training AUC was 0.82
    min_dwell_hours: 48
    priority: 1

  - name: manual
    kind: event
    source: oncall-approval
    priority: 0

guardrails:
  min_training_rows: 200_000
  min_days_of_data: 14
  max_retrains_per_week: 1
  cooloff_after_rollback_days: 7
  training_budget_usd_per_month: 2_000

training_window:
  schedule: 365d
  drift: 90d
  performance: 90d
  manual: operator-specified

gate:
  holdout: "holdout/churn/2026-10/frozen.parquet"
  holdout_sha: "sha256:f0e1..."
  must_match_champion_rows: true
  thresholds:
    auc_minimum: 0.80
    auc_lift_min_pp: 0.5           # challenger must beat champion by >= 0.5 pp
    calibration_ece_max: 0.03
    worst_slice_auc_min: 0.75
    latency_p99_ms_max: 150
    size_mb_max: 500
  shadow_traffic:
    duration_hours: 48
    traffic_fraction: 0.05
    max_prediction_delta: 0.10     # mean |p_challenger - p_champion| cap
  promotion:
    HIGH: human_approval_required
    MEDIUM: human_approval_required
    LOW: auto_if_gate_passes
  rollback:
    trigger: "any_alert within 2h of promotion"
    action: revert_to_previous_registry_version
```

The policy is the single source of truth. Two downstream systems read it: the scheduler (triggers + guardrails), and the gate (thresholds + promotion). Any change to retraining behavior is a change to this file, reviewed as code.

### 2.2 Trigger evaluator

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class TriggerKind(Enum):
    CRON = "cron"
    METRIC = "metric"
    EVENT = "event"

@dataclass(frozen=True)
class TriggerEvent:
    name: str
    priority: int
    fired_at: datetime
    reason: str

def evaluate_triggers(policy, metric_backend, event_backend, now):
    fired = []
    for t in policy["triggers"]:
        if t["kind"] == "cron" and _cron_due(t["cron"], now):
            fired.append(TriggerEvent(t["name"], t["priority"], now, "cron"))
        elif t["kind"] == "metric":
            ok, reason = metric_backend.evaluate(
                t["metric"], t["expression"], t["min_dwell_hours"]
            )
            if ok:
                fired.append(TriggerEvent(t["name"], t["priority"], now, reason))
        elif t["kind"] == "event":
            evt = event_backend.poll(t["source"], since=now - timedelta(hours=1))
            if evt:
                fired.append(TriggerEvent(t["name"], t["priority"], evt.ts, evt.payload))
    return sorted(fired, key=lambda e: e.priority)  # lower priority = more urgent
```

Priorities are stable and small (0–3). Priority 0 is reserved for humans; priority 1 for labelled-performance regressions (the strongest automated signal); priority 2 for drift; priority 3 for schedule. If multiple triggers fire in the same evaluation, the lowest-number priority wins and its training window is used.

### 2.3 Guardrail check (before training starts)

```python
def guardrails_ok(policy, state, trigger: TriggerEvent, now) -> tuple[bool, str]:
    g = policy["guardrails"]
    if state.rows_available < g["min_training_rows"]:
        return False, f"only {state.rows_available} rows; need {g['min_training_rows']}"
    if state.days_of_data < g["min_days_of_data"]:
        return False, f"only {state.days_of_data} days of data"
    if state.retrains_last_7d >= g["max_retrains_per_week"]:
        return False, "weekly cap reached"
    if state.last_rollback_at and (
        now - state.last_rollback_at < timedelta(days=g["cooloff_after_rollback_days"])
    ):
        return False, "in cool-off after rollback"
    if state.month_to_date_cost_usd + state.estimated_cost_usd > g["training_budget_usd_per_month"]:
        return False, "monthly training budget would be exceeded"
    return True, "ok"
```

Guardrails fail *loud*: a blocked retrain emits a Prometheus gauge (`ml_retrain_blocked{reason="..."}`) and files a ticket, it does not silently skip. Silent skipping is how stale models survive the policy that was supposed to replace them.

### 2.4 Gate (after training, before promotion)

```python
@dataclass
class GateVerdict:
    passed: bool
    reasons_passed: list[str]
    reasons_failed: list[str]
    metrics: dict
    recommendation: str  # "auto_promote" | "request_human_approval" | "reject"

def run_gate(policy, candidate, champion, holdout, slices):
    t = policy["gate"]["thresholds"]
    passed, failed = [], []

    cand = candidate.score(holdout)
    champ = champion.score(holdout)

    def check(label, value, op, bound):
        ok = (op == ">=" and value >= bound) or (op == "<=" and value <= bound)
        (passed if ok else failed).append(
            f"{label}={value:.4f} {op} {bound}"
        )
        return ok

    check("candidate_auc", cand.auc, ">=", t["auc_minimum"])
    check("auc_lift_pp", 100 * (cand.auc - champ.auc), ">=", t["auc_lift_min_pp"])
    check("calibration_ece", cand.ece, "<=", t["calibration_ece_max"])
    check("worst_slice_auc",
          min(cand.slice_auc[s] for s in slices), ">=", t["worst_slice_auc_min"])
    check("latency_p99_ms", cand.latency_p99_ms, "<=", t["latency_p99_ms_max"])
    check("size_mb", cand.size_mb, "<=", t["size_mb_max"])

    impact = policy["impact"]
    if not failed:
        rec = policy["gate"]["promotion"][impact]
    else:
        rec = "reject"
    return GateVerdict(
        passed=not failed, reasons_passed=passed, reasons_failed=failed,
        metrics={"auc": cand.auc, "champ_auc": champ.auc, "ece": cand.ece},
        recommendation="auto_promote" if rec == "auto_if_gate_passes" and not failed
                       else ("request_human_approval" if rec == "human_approval_required" and not failed
                             else "reject"),
    )
```

The gate is pure: given two models, a frozen holdout, and a set of slices, it produces a verdict with the exact numbers that supported the decision. "Reasons passed" are recorded along with "reasons failed" so a retrospective months later can see what *did* clear the bar, not only what tripped it.

### 2.5 Shadow traffic + promotion

```python
def promote_if_shadow_passes(candidate_id, policy, shadow_backend, approver=None):
    cfg = policy["gate"]["shadow_traffic"]
    result = shadow_backend.run(
        candidate_id=candidate_id,
        duration_hours=cfg["duration_hours"],
        traffic_fraction=cfg["traffic_fraction"],
    )
    if result.mean_abs_prediction_delta > cfg["max_prediction_delta"]:
        return "reject", f"shadow Δ={result.mean_abs_prediction_delta:.3f}"
    if result.errors > 0:
        return "reject", f"{result.errors} errors during shadow"
    if policy["impact"] in ("HIGH", "MEDIUM") and not approver:
        return "awaiting_approval", "human approval required"
    return "promote", "shadow passed"
```

The shadow step is what catches request-shape mismatches no offline holdout can reproduce: a feature that is encoded one way in training and another way in the live request path, a model artifact that depends on a library version absent in the serving container, latency regressions that only appear under real load. The `max_prediction_delta` guard catches the "challenger disagrees with champion wildly on live traffic" case, which is almost always a bug in the candidate rather than genuine improvement.

### 2.6 Rollback

```python
def rollback_if_regressed(policy, promoted_version, monitor, promoted_at, now):
    window_end = promoted_at + timedelta(hours=2)
    if now > window_end:
        return None
    if monitor.any_alert_between(promoted_at, now):
        return registry.revert(promoted_version)
    return None
```

Rollback is a *local* operation: revert the registry alias (`production → v2` instead of `v3`), not a retrain. Retraining in response to a promotion failure is the fastest path to a feedback loop; a revert is the fastest path to a safe state.

## 3. Validation steps

1. **Policy schema parse** — load `retrain_policy.yaml`, validate against a pydantic schema or JSON Schema. Catches typos in metric names before the pipeline fires.
2. **Trigger unit tests** — for each kind, feed a fixture metric/event/clock and assert exactly the expected `TriggerEvent` list and priority ordering.
3. **Guardrail unit tests** — construct `state` with one guardrail deliberately violated (too few rows, budget exhausted, cool-off active) and assert `guardrails_ok` returns `False` with the right reason.
4. **Dry-run on last month** — replay the last 30 days of metrics against the trigger evaluator; confirm how often each trigger would have fired. If drift would have fired 11 times, thresholds are too loose; adjust before the policy is live.
5. **Gate on known-worse candidate** — train a candidate with a degraded hyperparameter; the gate must reject with specific failed checks listed.
6. **Gate on known-better candidate** — train a candidate that genuinely beats champion on holdout; the gate must pass, and (depending on `impact`) recommend `auto_promote` or `request_human_approval`, never `reject`.
7. **Shadow traffic dry run** — route 1% of traffic to a candidate identical to champion; `mean_abs_prediction_delta` must be ~0, no errors, `promote` or `awaiting_approval`.
8. **Rollback drill** — manually promote a known-broken candidate; verify the rollback trigger fires within 2 hours and reverts the registry alias.
9. **Cool-off behavior** — immediately after step 8, attempt to re-run the retrain trigger; guardrails must block with reason "in cool-off after rollback".
10. **Budget cap** — set `training_budget_usd_per_month` to a value below the estimated cost; attempt to retrain; guardrail must block and emit the `ml_retrain_blocked` metric.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Policy-as-code | 2 | Policy lives in a reviewed file (YAML/JSON), not hard-coded in the scheduler |
| 2 | Multiple trigger types | 2 | At least schedule + metric + event; priorities break ties deterministically |
| 3 | Performance trigger included | 2 | A labelled-performance trigger exists; it is the strongest priority among automated triggers |
| 4 | Guardrails | 3 | Min rows, min days, max per week, cool-off after rollback, budget cap — all enforced |
| 5 | Gate thresholds are explicit | 3 | AUC floor, lift requirement, calibration, worst-slice floor, latency, size — all with values and reasons |
| 6 | Frozen holdout | 1 | Holdout is pinned by SHA; same holdout for champion and challenger |
| 7 | Shadow traffic | 2 | Shadow step exists between gate and promotion; has duration, fraction, delta guard |
| 8 | Human approval tied to impact | 2 | HIGH/MEDIUM impact requires human approval; LOW may auto-promote |
| 9 | Rollback is a revert, not a retrain | 1 | Rollback flips a registry alias; retrain is not the response to a bad promotion |
| 10 | Observability | 2 | Blocked retrains, gate failures, promotions, rollbacks all emit metrics; dashboards exist |

Items 4 and 5 are mandatory: a policy without guardrails will burn the training budget and ship unvalidated models; a gate without thresholds is theater.

## 5. Common mistakes

- **Retrain-on-every-alert.** A single drift fire is not a retrain signal; it is a trigger to *consider* retraining. Compose drift with dwell time and min-row guardrails before firing.
- **Collapsing trigger and promotion.** "Retrain weekly and replace production immediately" removes the gate. A trigger produces a candidate; a gate produces a deployment.
- **Gate measures on the candidate's own validation set.** Each candidate self-reports a different "test set"; no two scores are comparable. Pin a frozen holdout and compare both models on it.
- **No shadow traffic for high-impact models.** Offline holdouts cannot represent request-shape, library-version, or latency differences. For HIGH impact, shadow is mandatory.
- **Auto-promote on HIGH impact.** The cost of a bad production model and the cost of a delayed-by-a-day better model are asymmetric. Reserve auto-promote for LOW impact.
- **Unbounded retrain frequency.** Flaky drift tests will retrain daily and exhaust the budget. Cap at `max_per_week` and alert when the cap is hit (that is the signal to investigate the trigger, not to raise the cap).
- **No cool-off after rollback.** Immediately retrying after a rollback retrains on whatever broken state caused the rollback and ships the same bug twice.
- **Rollback by retraining.** Retraining takes hours; registry-alias flips take seconds. Rollback is a revert.
- **Silent guardrail skip.** A guardrail that returns `False` without emitting a metric or filing a ticket is a bug factory. Blocked retrains must be loud.
- **Trigger on inputs the attacker controls.** If a drift signal can be caused by inputs, an attacker can force retrains at will. Composite with min-dwell, min-samples, and budget caps to make the trigger non-abusable (this is the `ML06:2023 AI Supply Chain Attacks` surface).

## 6. References

- NIST AI Risk Management Framework 1.0 — `MANAGE 4.1` (post-deployment monitoring and response), `GOVERN 1.4` (change management), `MEASURE 2.9` (performance monitoring). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft`, `ML06:2023 AI Supply Chain Attacks`; the retraining loop is in the ML supply chain and must be hardened. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise`; triggers reachable from attacker-controlled inputs are a documented tactic. <https://atlas.mitre.org/>
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NIPS 2015) — the canonical treatment of hidden feedback loops in ML systems, including the trigger-retrain-deploy loops this exercise hardens. <https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
