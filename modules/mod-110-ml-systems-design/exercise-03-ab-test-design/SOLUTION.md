# exercise-03 — A/B Test Design for an ML Rollout · SOLUTION

Paired exercise: `lessons/mod-110-ml-systems-design/exercises/exercise-03-ab-test-design.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to design the A/B test that promotes a new ML model to production. The deliverable is a **written test plan** that an experienced reviewer (ML lead, data science lead, product lead) could sign off before rollout begins: hypothesis, metric hierarchy, sizing, assignment, analysis plan, guardrails, stop conditions, and a decision matrix. The plan must be committed to the repo alongside the model — not improvised on the first day of ramp.

Design priorities, in order:

1. **Write the plan before ramping.** A/B tests go wrong when the stop criteria are negotiated during the test. Every number the plan will later use to decide "ship or kill" has to appear before traffic is split.
2. **Hierarchy of metrics, not a single OEC.** One decision metric (the business number the test exists to move), one or two secondary metrics that explain the decision metric's movement, and a guardrail set that cannot regress regardless of the decision metric. Confusing these three is the single most common cause of bad rollouts.
3. **Pre-registered sample size.** The minimum detectable effect (MDE) is chosen before the test starts, sample size is computed from it, and the test runs for a planned duration — not "until significant." Peeking without correction is the dominant cause of fake wins.
4. **Guardrails fail fast, decision metric decides slow.** Guardrails use monitoring-grade alerts on short windows (hours); the decision metric uses the full pre-registered duration. A guardrail breach pauses ramp; a decision metric that is directionally bad after the full test kills it.
5. **Assignment unit = unit of independence.** If the user is the natural unit of analysis, randomize by user and compute the test statistic by user. Randomizing by request when behavior is correlated within a session is the second-most-common reason tests over-call wins.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Pre-registered plan, committed to repo | Playbook improvisation | Prevents outcome-dependent stopping, which is a false-positive factory |
| Three-tier metric hierarchy (decision / secondary / guardrails) | Single OEC | One-metric designs cannot distinguish "we hurt a subgroup" from "we moved the aggregate" |
| Two-sided test with explicit MDE | One-sided "is it better?" | Two-sided forces thinking about both regression and improvement; many "wins" in practice are regressions in disguise |
| Per-user randomization with cluster-robust SE | Per-request randomization | Prevents SUTVA/violated-independence over-calling; the standard fix for correlated observations within a user |
| Shadow + canary before any A/B | Straight to 50/50 | Shadow catches implementation bugs; canary catches operational surprises. The A/B is the last of the three |
| Stat sig *and* practical sig | Stat sig only | A 0.05% lift significant at p=0.001 on a $10B product is interesting; on a $10M product it is noise. Both thresholds are needed |

## 2. Implementation — worked plan for "Promote fraud-scoring model v3.3 (replaces v3.2)"

The plan is organized as a document the reviewer reads top-to-bottom. Section headings below mirror what a reviewer actually looks for.

### 2.1 Hypothesis

> We believe that model v3.3, which adds a merchant-category sequence feature, will increase daily chargeback dollars avoided by at least 3.0% relative to v3.2, without increasing false-positive hold rate by more than 0.5 absolute percentage points or breaching p99 latency SLOs.

Three ingredients are explicit:

- **Direction and magnitude of expected effect** — "at least 3.0% relative." A test designed to detect an unspecified effect is a test designed to find noise.
- **Guardrail with a concrete threshold** — "FP hold rate" and "p99 latency" each bounded.
- **Business metric** — chargeback dollars, not AUC. The AUC improvement is why we are running the test; the chargeback improvement is what we are testing *for*.

### 2.2 Metric hierarchy

```
Decision metric (primary):
    daily_chargeback_dollars_avoided_per_1000_txn   (unit: $ per 1k)
    MDE = 3.0% relative; two-sided; α = 0.05

Secondary metrics (interpret the decision):
    precision_at_hold              (should not drop)
    recall_on_analyst_labeled_slice (should rise)
    reason_code_coverage           (should not drop)

Guardrails (any breach pauses ramp):
    fp_hold_rate_delta             bound: +0.5 abs pp, 24h window
    p99_scoring_latency            bound: +15 ms (pre-registered), 1h window
    error_rate                     bound: +0.1 abs pp, 1h window
    hold_volume_vs_capacity        bound: ≤ 1.2× daily analyst capacity

Diagnostic metrics (not for decisions; for debugging):
    score_distribution_ks_vs_v32
    per-merchant-category hold rate
    feature-level PSI vs. v3.2's training set
```

Guardrails have their own alert paths and their own on-call owner. They are computed on short windows because the goal is to stop damage fast; the decision metric is computed on the pre-registered horizon because the goal is to avoid being fooled by a noisy early win.

### 2.3 Sizing

Chargeback dollars avoided per 1,000 transactions is the decision metric. From the last 90 days: mean ≈ $420 per 1k, standard deviation per-user-day ≈ $190 per 1k.

For a two-sample t-test with per-user randomization, α = 0.05 (two-sided), power = 0.80:

```
MDE (absolute)  = 0.03 × 420   = $12.60 per 1k
σ               = $190 per 1k
per-arm n       ≈ (z_{1-α/2} + z_{1-β})² × 2 × σ² / MDE²
                = (1.96 + 0.84)² × 2 × 190² / 12.6²
                ≈ 3,564 user-days per arm
```

Per-user independence is a strong assumption; per-user-day is used as the unit of analysis, and the sample size is computed at that unit. Cluster-robust standard errors are used in analysis to account for within-user correlation across days.

At daily enrolment of 500k unique users (step-6 ramp, below), the design is powered within one week. The plan reserves **two full weeks** of exposure to span weekend/weekday cycles and the end-of-month merchant pattern.

### 2.4 Design and assignment

```
Unit of assignment      user_id (SHA-256 bucketed, no PII leakage)
Unit of analysis        user_day (one observation per user per day)
Arms                    control (v3.2) | treatment (v3.3)
Allocation              50 / 50 at steady state
Ramp plan               1% → 5% → 25% → 50%  (24h hold at each step, guardrails green)
Duration (steady state) 14 days
Exclusions              test accounts, employee accounts, users with < 3 txns in prior 30d
                         (pre-registered; documented in plan, applied identically in both arms)
Blocking variable       merchant_category (post-stratification in analysis)
```

The ramp exists so that the *implementation* fails on a small share of traffic before the *experiment* fails on 50% of traffic. 1% → 5% → 25% → 50% is the standard shape that balances speed against blast radius; a design that ramps to 50% on day one has conflated the experiment with the deployment risk.

### 2.5 Pre-rollout checks (shadow + canary)

Before the A/B test starts, two gates:

- **Shadow (≥ 7 days).** v3.3 receives 100% of traffic in parallel with v3.2; its predictions are logged but not acted on. Checks: p99 latency within budget; score distribution matches staging within 2σ; per-feature PSI < 0.1 vs. training; no serving-time errors.
- **Canary (≥ 3 days, 1% traffic).** v3.3 predictions act. Same checks as shadow, plus: holds-per-1k is within ±10% of the training-set expectation; no spike in reason-code "unknown"; on-call has slept through two nights.

Only after both gates clear does the A/B ramp begin at 1%.

### 2.6 Analysis plan

Written *before* the test starts. Three deliverables at test-end:

1. **Primary test.** Two-sample t-test on per-user-day chargeback dollars avoided, cluster-robust SE at the user level, α = 0.05 two-sided. Report point estimate, 95% CI, p-value, and absolute $-per-1k lift.
2. **Pre-registered subgroup analyses.** Three cuts, Bonferroni-corrected across subgroup tests (α = 0.05 / 3 = 0.0167):
   - merchant_category (top 10)
   - region (US / EU / other)
   - new-user vs. established-user (prior 30d txn count ≥ 10)
3. **Diagnostic plots.** Score distribution per arm; FP rate over time; p99 latency over time. Not for decisions; for debugging a surprising decision-metric result.

Interim analysis: **none**. The test runs to the pre-registered horizon. If a guardrail breaches, ramp is paused; if a decision-metric interim is desired at all, the plan would pre-register a sequential testing procedure (e.g., alpha-spending) — but for this test, interim analysis is explicitly out of scope to keep the error rate at nominal α.

### 2.7 Decision matrix (committed before the test)

| Decision metric | Guardrails | Decision |
|---|---|---|
| Lift ≥ 3% and p < 0.05 | All green | Ship v3.3 full rollout, retire v3.2 |
| 1% ≤ lift < 3%, p < 0.05 | All green | Hold at 50/50 for 14 more days; re-run analysis on pooled data |
| Lift < 1% or p ≥ 0.05 | All green | Kill v3.3; postmortem the offline-online gap |
| Any lift | Any guardrail breached at any ramp step | Kill v3.3; postmortem the breach |
| Lift < 0, p < 0.05 at ramp step | n/a | Immediate rollback; incident review |

The matrix is signed off by ML lead and product lead **before** ramp starts. A test whose decision is negotiated after the data arrives is not an experiment; it is a committee meeting with charts.

### 2.8 Rollback plan

- A single config flip reverts 100% of traffic to v3.2 within 60 seconds.
- The rollback is rehearsed in staging before ramp begins.
- A rollback does not require sign-off from more than the on-call engineer — reversibility is prized over consensus during an incident.
- Post-rollback, a 72-hour hold is required before any re-ramp, to avoid flapping.

### 2.9 Write-up

A short write-up is committed to the repo at test-end: hypothesis, final metric values, decision, and one paragraph on anything surprising. A test whose result is not written down does not count — the next design discussion will reinvent arguments that this test already answered.

## 3. Validation steps

A test plan is validated by dry-run, not by production data. Run these in order.

1. **Simulate the decision rule on prior tests.** Take the last two A/B results with known outcomes and apply the decision matrix from §2.7. The matrix must produce the same ship/kill decisions a human made at the time — or disagree explainably.
2. **Sample size sanity.** Recompute §2.3 with power = 0.90 instead of 0.80. If sample size roughly doubles, the arithmetic is right; if it does not, the MDE/σ inputs are wrong.
3. **AA test.** Run a 50/50 split where both arms get v3.2 for 72 hours. The decision metric's p-value must be uniformly distributed across many 1-hour windows; a skewed p-value says the metric is miscomputed, the randomization is broken, or the clustering is wrong.
4. **Guardrail-breach rehearsal.** Simulate a 20 ms p99 latency regression on the staging stack and verify that the alert fires within the pre-registered 1-hour window. A guardrail that takes a day to fire is not a guardrail.
5. **Ramp-step dry run.** Run the full ramp on a low-traffic staging environment. Confirm the 1% / 5% / 25% / 50% bucket sizes are correct and that the config flip for rollback succeeds.
6. **Pre-registration diff.** After the test, diff the committed plan against the analysis actually performed. Any difference is called out in the write-up; silent amendments defeat the whole point of pre-registration.
7. **Peer review.** A reviewer who is not the test author reads the plan and tries to predict the decision for three hypothetical outcomes using only §2.7. If the reviewer cannot, the decision matrix is ambiguous and must be rewritten.

## 4. Rubric / review checklist

Weighted, `/25` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Hypothesis | 2 | Direction, magnitude, guardrail thresholds all stated numerically |
| 2 | Metric hierarchy | 4 | Three tiers (decision / secondary / guardrail) named; diagnostic metrics kept out of the decision |
| 3 | Sizing | 3 | MDE, σ, α, power stated; sample size shown with arithmetic; unit of analysis matches unit of assignment or is cluster-corrected |
| 4 | Randomization and unit | 2 | Assignment unit named; SUTVA/independence considered; blocking/stratification stated if used |
| 5 | Shadow + canary gates | 2 | Both named with exit criteria, not just "we'll canary first" |
| 6 | Ramp plan | 2 | At least 3 steps; 24h hold or stated alternative; clear bucket sizes |
| 7 | Pre-registered analysis | 3 | Primary test method, correction for subgroups, decision to do/not do interim analysis — all stated |
| 8 | Decision matrix | 3 | Signed off before ramp; covers all four quadrants (stat sig / not × guardrails green / breached) |
| 9 | Rollback | 2 | One-flip revert; rehearsed; owner named |
| 10 | Write-up commitment | 1 | Post-test write-up committed to repo; diff against pre-registration included |
| 11 | Grounded in frameworks | 1 | Guardrail set maps to NIST AI RMF `MANAGE 2.1` (monitor for deviations in deployed systems) |

Items 2, 3, and 8 are mandatory: a plan without a metric hierarchy, without power/MDE arithmetic, or without a signed-off decision matrix cannot score above 14. Those three are the ingredients that convert "we A/B test everything" from a slogan into a practice.

## 5. Common mistakes

- **Starting ramp before the plan is written.** The team runs 50/50 traffic on day one, sees a +4% lift by day three, declares victory, and ships. Day 14 the win disappears. Peeking without correction routinely produces this exact pattern.
- **One metric, especially AUC.** AUC rising online is a diagnostic, not a decision. If the test ships on AUC alone and chargeback dollars did not move, the system has the same business performance at higher complexity.
- **Randomizing by request instead of user.** Clicks within a session are correlated; treating them as independent inflates statistical power and over-calls wins. Randomize at the unit where behavior is independent.
- **"We'll just run it until significant."** Equivalent to a false-positive rate much higher than the nominal α. If interim analysis is desired, use a sequential procedure (alpha-spending, group sequential designs) that keeps α at nominal.
- **Guardrails evaluated at the decision-metric horizon.** A guardrail that takes 14 days to fire is a decision metric, not a guardrail. Guardrails must have short-window alerting.
- **Uneven traffic between arms.** 50/50 assignment that lands at 48/52 because the assignment hash is not uniform. Sanity-check the actual split hourly during ramp.
- **Missing exclusions.** Test accounts and employee accounts count in both arms; they can bias either direction. Exclusions are pre-registered.
- **No rollback rehearsal.** The one-flip rollback is written in the plan but never tested; day-one of ramp, a guardrail fires and the config flip has a typo. Rehearse in staging before ramp.
- **Changing the primary metric after the fact.** "The primary metric did not move, but secondary metric X did, so we ship." Secondary metrics interpret the primary; they do not replace it when the primary disappoints.
- **No subgroup analysis.** The aggregate win hides a 4% regression on new users. The pre-registered subgroup analysis is the control for this. Running it only when the aggregate is ambiguous is multiple-comparison abuse.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MANAGE 2.1` (monitor for and respond to deviations in deployed systems) is the framework-level grounding for the guardrail layer in §2.2; `MEASURE 2.1` (documented performance characterization) is the grounding for the pre-registered analysis plan. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft` and `ML08:2023 Model Skewing` are the security-relevant failure modes an A/B rollout surface area can introduce (exposed inference endpoints, controllable inputs). The guardrail set includes the error-rate and latency bounds that detect a hostile ramp. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0024 Verify Attack` and `AML.T0043 Craft Adversarial Data` describe adversary workflows that look similar to a model rollout gone wrong; the diagnostic metrics in §2.2 are the signal surface for both. <https://atlas.mitre.org/>
