# exercise-01 — System Design Interview Walkthrough · SOLUTION

Paired exercise: `lessons/mod-110-ml-systems-design/exercises/exercise-01-system-design-interview-walkthrough.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to answer a canonical ML system design interview prompt end-to-end. The deliverable is not "a correct architecture." It is a **reproducible method for walking through any ML design question** — one where the candidate narrates their choices, names the alternatives they rejected, and ties each decision to a risk the system has to absorb.

Design priorities, in order:

1. **Clarify before architecting.** The first five minutes are spent constraining the problem — who is the user, what is "success," what traffic, what latency, what cost of a wrong answer — not drawing boxes. A box drawn in the wrong problem is worse than a blank page.
2. **Objective → metric → loss.** The business objective, the evaluation metric, and the training loss are three different artifacts, and the design must show how they stay aligned (and what you measure when they disagree).
3. **Data flow is the architecture.** The diagram the interviewer actually wants is features → training → model artifact → serving → feedback → features again. If any arrow in that loop is missing, the system cannot improve itself and the answer is incomplete.
4. **Fail modes are first-class citizens.** Each component is sized by what it costs when it misbehaves — a serving outage costs latency SLO breaches, a data pipeline outage costs stale features, a label pipeline outage costs blind drift. The design must say which of those are acceptable and for how long.
5. **Risk register, not a diagram.** The interview's real signal is whether the candidate knows what can go wrong (adversarial inputs, drift, skew, bias, privacy) and has a *named* mitigation for each. The NIST AI RMF `MAP`/`MEASURE`/`MANAGE`/`GOVERN` functions and the OWASP ML Top 10 are the two public frameworks that let a learner organize this without inventing categories.

| Choice | Alternative considered | Why this one |
|---|---|---|
| 7-step framework (Clarify → Metrics → Data → Model → Serving → Monitoring → Risks) | Freeform architecture-first narrative | Framework prevents forgetting monitoring/feedback — the parts juniors skip and seniors are judged on |
| Business metric *and* proxy ML metric *and* training loss, documented separately | Single north-star metric | Lets the candidate discuss when metrics diverge (e.g., AUC up, revenue flat) which is the interview's real signal |
| Explicit "what we do NOT build" list | Full-stack design | Shows scope discipline; signals the candidate can defer ambiguous features to v2 |
| Risk register tied to NIST AI RMF + OWASP ML Top 10 categories | Ad-hoc "what could go wrong" | Grounds the discussion in public frameworks instead of personal opinion; resists interviewer "is that really a risk?" pushback |
| Start with offline eval, then shadow, then canary, then A/B | Straight-to-A/B rollout | The three-step promotion path is what the interviewer wants to hear — it is the standard that has emerged for ML rollouts |
| Online feature store + offline feature store, same transform code | Two independent pipelines | One transform definition is the only defense against training-serving skew that scales (see exercise-02) |

## 2. Implementation — worked walkthrough on "Design an ML system to flag likely-fraudulent transactions for manual review"

The prompt below is used as the canonical example. The same framework applies to recommendation, search ranking, content moderation, forecasting, and churn prediction prompts — only the specifics of steps 3–5 change.

### 2.1 Step 1 — Clarify (≈ 5 minutes)

Ask the interviewer. Do not guess. Produce a short, written list of assumptions before touching the whiteboard.

```
Business:
- Who uses the output?           Manual review analysts (humans), not customers
- What action is taken?          Transaction is held for review, not auto-blocked
- Cost of a FP?                  ~30s of an analyst's time, ~small customer friction if held
- Cost of a FN?                  Full transaction value charged back later; typically $50–$5,000

Scale:
- QPS (peak)?                    2,000 transactions/sec
- Daily volume?                  ~100M txns/day
- Tail latency budget?           150ms p99 at auth time (hard; blocks checkout)
- Analyst review throughput?     ~500 reviews/analyst/day; team of 20 → 10k/day total

Data:
- Label latency?                 Chargeback window: 60–120 days
- Feedback from analysts?        Yes, every held transaction gets a label in <24h
- PII in features?               Yes; email, device, geo → treat as restricted
- Cross-border?                  Yes; GDPR applies to EU subset
```

A candidate who skips this step and dives into "I'll use XGBoost" has already lost most of the signal.

### 2.2 Step 2 — Objective → metric → loss

Three distinct artifacts, all written down:

| Layer | Definition | Instrument |
|---|---|---|
| Business objective | Reduce chargeback dollars lost per month without breaching analyst capacity (10k reviews/day) | Finance report, monthly |
| Online metric (health) | $ chargebacks avoided, review precision, p99 latency, holds/day | Realtime dashboard + weekly review |
| Offline eval metric | Precision at K = daily analyst capacity, recall at that K, calibration (ECE) | Held-out time-slice backtest |
| Training loss | Weighted log-loss, class weights set from recent FP/FN cost ratio | Training script |

The critical observation — stated out loud in an interview — is that **the training loss is not the business metric**. The model is optimized on log-loss, judged on precision-at-K, and the business is judged on chargeback dollars. Each handoff is a place where the system can appear to improve while degrading.

### 2.3 Step 3 — Data

Three subsystems. Each gets a one-paragraph description and one named failure.

**Historical / training store.** Partitioned by day, immutable once written, source for training batches. Failure mode: a backfill overwrites a partition and a model retrains on silently changed history. Mitigation: dataset versioning (hash of partition range) logged on every training run.

**Online feature store.** Low-latency key-value lookup (user_id, merchant_id, device_id) at auth time. Serves derived features computed by the same transform code that computed the training features. Failure mode: feature freshness drops because the upstream stream pauses, and the model serves stale aggregates. Mitigation: per-feature freshness SLO, monitored at p99, with fallback to a safe default (and a feature flag that logs the fallback for later skew analysis).

**Label store.** Two sources — analyst decisions (within 24h) and chargebacks (60–120 days). Both write to the same table with a `label_source` column. Failure mode: training joins analyst labels only, then production sees charged-back transactions the analysts approved, so recall silently falls. Mitigation: training pipeline explicitly requires both sources and raises on missing coverage for a day in the training window.

Features for a v1 model:

```
Per-txn      amount, hour-of-day, merchant_category, is_cross_border
Per-user     txn_count_7d, amount_sum_7d, distinct_merchants_30d,
             time_since_last_txn, avg_amount_30d
Per-device   txn_count_1d, distinct_users_30d
Per-merchant chargeback_rate_90d, approval_rate_7d
```

All aggregates are computed in the stream with a fixed window and a fixed definition, and the same definition is used for training (point-in-time correct) and serving. Point-in-time correctness — i.e., "would this aggregate have been available at the moment of this historical transaction?" — is the hidden complexity that a senior candidate names explicitly.

### 2.4 Step 4 — Model

```
v0 baseline: logistic regression on 10–20 engineered features
v1 target:   gradient-boosted trees (XGBoost/LightGBM) on the full feature set
v2 (if v1 plateaus): sequence model over the user's recent transactions
```

Why in this order: the baseline exists so every subsequent model is judged against something reproducible. Trees are the right choice for tabular + mixed-type features and are cheap to serve; a sequence model is only reached for if the trees plateau on a signal that is temporal in nature.

Cold start is called out: a new user has no `_7d` aggregates. The serving contract specifies that missing aggregates resolve to a sentinel (`-1`, not `0`) so the model learns "unknown" as a distinct category. This is enforced in the shared feature transform, not left to ad-hoc nulls.

Explainability is sized to the audience. The analyst UI needs **per-decision reason codes** ("device seen in 42 accounts in 7 days"), not full SHAP waterfalls. Reason codes are produced by a thin layer on top of the model (top-k feature contributions, mapped to an analyst-readable template), checked into the same repo as the model, and versioned alongside it.

### 2.5 Step 5 — Serving

```
         ┌──────────────┐         ┌───────────────────┐
checkout│ API gateway  │────────▶│  fraud-score svc  │────▶ score, reason codes
         └──────────────┘         └────────┬──────────┘
                                           │
                                           ▼
                                  ┌───────────────────┐
                                  │ online feature    │  (sub-ms lookups)
                                  │ store             │
                                  └───────────────────┘
```

Deployment contract:

- p99 latency budget at the service: 80ms (leaves 70ms of headroom under the 150ms auth budget).
- Model loaded in-process; no network hop to a separate model server unless the model's memory footprint forces it.
- One model version per pod. Rolling deploys swap pods, not weights-in-place.
- Config-driven thresholds: the score-to-"hold for review" threshold is a config value, not compiled into the model. This is what lets analyst capacity (10k/day) be managed independently of model training.
- Shadow mode and canary mode are first-class. A new model is deployed to shadow (gets traffic, output is logged but not acted on) for ≥ 7 days before any canary. Canary is 1% of traffic for ≥ 3 days before full rollout. Criteria for promotion are written down **before** the shadow starts (see the A/B exercise).

### 2.6 Step 6 — Monitoring (the part juniors skip)

Four layers, each with its own on-call response:

| Layer | What is measured | Alert |
|---|---|---|
| System | QPS, p99 latency, error rate, feature store miss rate | Paging; on-call engineer |
| Data | Feature freshness, input distribution (KS/PSI vs. training), missing-feature rate | Paging; feature owner |
| Model | Score distribution, calibration drift, precision/recall on 24h analyst-labeled slice | Ticket; ML team |
| Business | Holds/day vs. capacity, chargeback dollars, FP rate, analyst-acceptance rate | Weekly review; product + ML |

The alert list — not the chart list — is the deliverable. A chart that doesn't page does not exist operationally.

Label latency is the trickiest part and is called out explicitly. Analyst labels close the loop within 24h and drive short-horizon recall estimates. Chargeback labels close the loop at 60–120 days and drive the quarterly revenue impact review. The design must show both, and must show how a model is *not* retrained on chargeback-adjusted labels for transactions that are still inside the dispute window.

### 2.7 Step 7 — Risk register

This is the section senior interviewers listen for. Risks are organized against the two public frameworks named in this module's source policy.

NIST AI Risk Management Framework (MAP → MEASURE → MANAGE → GOVERN):

- `MAP` — Context. The system holds human transactions for manual review; false positives inconvenience customers, false negatives cost chargeback dollars.
- `MEASURE` — Metrics. Precision-at-K and calibration are measured daily on analyst labels; AUC is tracked on the chargeback slice quarterly.
- `MANAGE` — Response. Feature-level drift alert triggers a retraining run; a model that fails canary gating is rolled back automatically.
- `GOVERN` — Policies. Reason-code audit log retained for 2 years; model card updated on every promotion.

OWASP Machine Learning Security Top 10 — mitigations:

- `ML01:2023 Input Manipulation Attack` — Rate-limit per-device; anomaly detector on feature vectors (not just score) to catch inputs crafted to look benign.
- `ML02:2023 Data Poisoning Attack` — Labels from analysts pass a four-eyes review for a sampled 1%; label-source tagged on every row.
- `ML06:2023 AI Supply Chain Attacks` — Model binary built in CI from a pinned commit; model card logs the training dataset SHA, code SHA, and library versions.
- `ML08:2023 Model Skewing` — Training set rebalancing is logged with class ratios; feedback loop on analyst labels is sampled, not full — otherwise the model learns to predict what analysts review, not what is actually fraud.

MITRE ATLAS is referenced as the broader threat model if the interviewer pushes on adversarial ML (e.g., `AML.T0043 Craft Adversarial Data`). The design does not claim to defend against a sophisticated adversary; it claims to detect their presence via the drift and anomaly monitors in step 6.

### 2.8 Step 8 — What this design does NOT do

Said out loud, on purpose:

- Auto-blocking transactions. The output is always a *hold for review*. Auto-block would require a separate approval workflow and a higher-bar precision target.
- Real-time retraining. Models retrain nightly on a rolling 30-day window; online learning is explicitly out of scope for v1.
- Cross-tenant personalization. The model is one global model; per-merchant fine-tuning is a v2 question.
- Multilingual reason codes. Analyst UI is English-only in v1.

Naming non-goals is how a candidate demonstrates scope discipline. The interviewer learns more from what you cut than from what you add.

## 3. Validation steps

A walkthrough is validated by rehearsal, not by code. Run these in order:

1. **Timebox rehearsal.** Deliver the full walkthrough (steps 1–8) in 35 minutes to a stopwatch. If step 1 took more than 7 minutes or step 6 was skipped, the balance is wrong and the design will read as junior.
2. **Peer interview.** Ask a peer to interrupt at step 4 with "why not deep learning?" and at step 6 with "how do you know the labels are correct?" Both are stock interviewer pushbacks. A candidate who cannot answer them in two sentences each has not internalized the framework.
3. **Metric drift check.** On the whiteboard, cross out the training loss row in the step-2 table and ask: can the rest of the design still improve the business metric? If yes, the design is over-indexed on offline metrics; if no, the monitoring layer is missing.
4. **Risk register vs. frameworks.** For each risk listed in step 7, point to its NIST AI RMF function and its OWASP ML Top 10 category (or explain why it maps to neither). Risks that map to neither are either invented or belong in a different design — remove them.
5. **Non-goals read-aloud.** Read step 8 out loud. Every item should feel slightly uncomfortable to cut. If no item is uncomfortable, the non-goals are strawmen and scope discipline will not read as credible.
6. **Cold-write re-derivation.** 48 hours after the first rehearsal, redo the walkthrough on a new prompt (recommendation, ranking, forecasting) using the same 7 steps. The method is only internalized if it transfers.

## 4. Rubric / review checklist

Weighted, `/30` points total. Mirrors the interviewer's scorecard.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Clarifying questions | 3 | ≥ 5 questions covering user, cost of FP/FN, scale, latency, PII before any architecture |
| 2 | Objective vs. metric vs. loss | 3 | Three layers named separately; candidate calls out at least one way they can disagree |
| 3 | Data subsystems | 4 | Historical, online, label stores each named with one failure mode and one mitigation |
| 4 | Point-in-time correctness | 2 | Explicitly stated; feature-store design uses one transform definition for train and serve |
| 5 | Model progression | 2 | Baseline → v1 → v2 ladder; candidate names the exit criterion between each step |
| 6 | Serving contract | 3 | p99 budget sized against business constraint; shadow → canary → A/B written down |
| 7 | Monitoring across four layers | 4 | System / data / model / business each has at least one metric and one named alert |
| 8 | Label latency handled | 2 | Short-horizon (analyst) and long-horizon (chargeback) labels both addressed |
| 9 | Risk register with public frameworks | 4 | ≥ 4 risks mapped to NIST AI RMF functions and/or OWASP ML Top 10 categories |
| 10 | Non-goals | 2 | ≥ 3 explicit non-goals; each one plausibly in scope for v2 |
| 11 | Pacing | 1 | Finishes in 35 ± 5 minutes; no step takes > 10 minutes |

Items 1, 7, and 9 are mandatory: a walkthrough missing clarifying questions, missing the monitoring layer, or missing a risk register cannot score above 18. Those three omissions are the most common reasons strong-on-paper candidates fail ML system design loops.

## 5. Common mistakes

- **Diagram-first.** The candidate draws a four-box architecture in the first two minutes without asking a single clarifying question. Every later choice is then defended against assumptions the interviewer did not make. Clarify first.
- **One metric to rule them all.** "I'll optimize AUC" is treated as sufficient. AUC is a model-ranking metric, not a business metric, and a design that stops at AUC has no answer for "the model's AUC went up and revenue went down."
- **Skipping monitoring.** The candidate builds a training pipeline and a serving service and then declares done. The system cannot know when it is broken in production; the design is incomplete.
- **Treating training-serving skew as "just use the same code."** The claim is right; the implementation is hard. A senior answer names the feature-store boundary and the shared transform library. See exercise-02.
- **Risk hand-waving.** "We will also make sure it's fair and secure" with no categories and no mitigations. Use NIST AI RMF and OWASP ML Top 10 categories out loud — they signal that the candidate reads the field, not just the blog.
- **Over-committing to deep learning.** For tabular fraud at 2k QPS, gradient-boosted trees usually win on both accuracy and serving cost. Reaching for a transformer in step 4 reads as resume-driven rather than problem-driven.
- **Forgetting cold start.** The design lives or dies on what the model does for a user it has never seen. "Return neutral and let a rule fire" is an acceptable answer; silence is not.
- **Online learning in v1.** Tempting; expensive to get right. A senior design calls out that online learning is v2 and explains the two blockers (label latency, poisoning blast radius).
- **Ignoring label quality.** Analyst labels are not ground truth; they are "analyst decisions given the UI and workload of that week." The design must say how the model avoids learning the analyst instead of the fraud signal (sampling, held-out audits).
- **No non-goals.** Everything is in scope, nothing is cut. Scope discipline is a senior signal; its absence reads as junior even when the content is strong.

## 6. References

- NIST AI Risk Management Framework 1.0 — the `MAP`, `MEASURE`, `MANAGE`, `GOVERN` functions used to structure the step-7 risk register. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — category list (`ML01` through `ML10:2023`) used to name adversarial, poisoning, and supply-chain risks in step 7. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — adversary tactics and techniques for ML systems; referenced as the escalation framework when an interviewer probes threat modeling beyond OWASP's categories. <https://atlas.mitre.org/>
