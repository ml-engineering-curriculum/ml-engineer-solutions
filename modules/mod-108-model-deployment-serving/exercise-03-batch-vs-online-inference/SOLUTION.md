# exercise-03 — Batch vs Online Inference · SOLUTION

Paired exercise: `lessons/mod-108-model-deployment-serving/exercises/exercise-03-batch-vs-online-inference.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to pick between **batch** inference (score many records offline on a schedule, write to a store) and **online** inference (score a single record in a request/response path) for a set of product scenarios, and defend the choice. The failure mode most learners fall into is reaching for whichever pattern they built most recently — online when they just finished exercise-01, batch when they were last in a Spark notebook. This solution gives a decision procedure and applies it to the four canonical scenarios.

Design priorities, in order:

1. **Pick the pattern from the SLO, not the stack.** Four questions decide it: how fresh must the prediction be when it is consumed, how fast must the response be, what is the request volume and distribution, and is the input available in real time at all?
2. **Freshness is the lead question.** A churn score consumed once a day by an outbound email campaign does not need millisecond inference; computing it online wastes money and makes the system more fragile. A fraud score consumed in a checkout flow cannot wait for a nightly job no matter how cheap batch is.
3. **The prediction store is a first-class artifact for batch.** Batch output is only useful when a downstream can read it by key with low latency. A "batch job that writes to S3 and hopes" is not deployment; the output must land in a serving layer (feature store, Redis, DynamoDB, warehouse with a sub-second index).
4. **Online inference inherits every reliability property of a web service.** Timeouts, retries, backpressure, circuit breakers — everything in SRE's online-service playbook applies the moment you choose online.
5. **Micro-batching and caching are the compromises.** Most "online" systems can push p99 down and cost down by batching inside the request path (10–50 ms windows) or by caching predictions for stable features. Call these out explicitly — they are not cheating, they are the correct answer for a wide middle band.
6. **Governance is identical.** Both patterns must log the input, the prediction, the model version, and a request/batch ID. NIST AI RMF `MEASURE 2.3` / `MEASURE 2.6` makes no distinction; the OWASP `ML05:2023 Model Theft` / `ML06:2023 AI Supply Chain Attacks` audit trail is the same shape.

| Dimension | Batch favours | Online favours |
|---|---|---|
| Freshness tolerance | Hours to days | Seconds or less |
| Per-request latency target | No real-time user-facing requirement | Tight (p99 < a few hundred ms) |
| Input availability | Known ahead of time (nightly dump, user base snapshot) | Produced at request time (session, cart, click) |
| Request volume | Millions of records with high overlap (same users scored daily) | Low-to-mid QPS with high input variety |
| Cost sensitivity | Dominated by batch job cost ($/run) | Dominated by served QPS cost ($/req) |
| Failure consequence of staleness | Low (yesterday's score ≈ today's) | High (last-minute signal is the point) |
| Model size / resource profile | Large models that cannot fit online SLO | Small-to-medium, warm-model-in-memory feasible |

## 2. Worked implementation

The exercise's four scenarios (names paraphrased from the learning file's framing; see the paired exercise for exact phrasing).

### 2.1 Scenario A — nightly churn score for a weekly email campaign

**Decision: batch.**

- Freshness tolerance: 7 days (the email ships weekly). A score computed Sunday night and consumed Monday-through-Sunday is fit for purpose.
- Latency target: none at the consumption point (the email service reads from a table).
- Input availability: user base is a known snapshot; the feature table is populated by the nightly ETL.
- Request volume: ~10M users, every user scored. Online would be ~10M inferences across one week concentrated around the send window.
- Cost: batch scores all users in minutes on a Spark/Dask job once a week. Online would burn warm pods 24/7 to serve a request pattern that is bursty and avoidable.

Design:

```
feature_warehouse (batch ETL)
   └─▶ nightly Airflow DAG
         ├─▶ load features for all active users
         ├─▶ score with registered model `churn@champion`
         └─▶ write (user_id, churn_prob, model_version, scored_at) to BigQuery
                └─▶ campaign tool reads by user_id
```

Governance: every row includes `model_version` and `scored_at`; a daily SELECT COUNT(*) per `model_version` is the drift-canary.

### 2.2 Scenario B — fraud score at checkout

**Decision: online.**

- Freshness tolerance: milliseconds. The signal of fraud is the current session (IP velocity, cart composition, device fingerprint). A score computed last night is useless.
- Latency target: p99 ≤ 150 ms (the checkout flow has its own SLO; fraud is a dependency of it).
- Input availability: features are produced *during* the request (current cart, current device). Some features come from a low-latency store (user history); they must be fetched inside the request.
- Request volume: ~200 QPS peak, < 10 QPS trough. Not batch-shaped.
- Cost: ~17M requests/day at modest resource; batch here is not even a choice — the input does not exist in batch form.

Design:

```
checkout service ──(gRPC)──▶ fraud-predictor (FastAPI/Triton)
                               │
                               ├─▶ feature store get_online(user_id)  [< 10ms]
                               ├─▶ model.predict(features)            [< 30ms]
                               └─▶ return {score, model_version, rid} [< 5ms]
```

Design budget is p99 < 150 ms = 10 (store) + 30 (predict) + 5 (serialize) + 100 (safety / GC / LB). Anything over that budget must either shrink the model, warm a cache, or push features precomputed to the online store.

### 2.3 Scenario C — personalised recommendations on the home feed

**Decision: hybrid — batch pre-compute + online re-rank.**

- Freshness tolerance: the *candidate set* refreshes daily; the *ranking* refreshes within the session.
- Latency target: p99 ≤ 200 ms for the full page.
- Input availability: user affinity vectors from a nightly batch; current session signals (last click, scroll depth) at request time.
- Request volume: every pageload — millions of QPS at peak for a large consumer surface.

Design:

```
nightly batch
   └─▶ for each user: write top-K candidate items + affinity vector to KV store
online path (pageload)
   ├─▶ fetch user's K candidates + vector from KV store   [~5ms]
   ├─▶ fetch 2–3 session signals                          [~5ms]
   ├─▶ re-rank with small model                            [~20ms]
   └─▶ return ranked list                                  [~1ms]
```

The batch stage is where the expensive two-tower candidate model runs over the full catalogue; the online stage runs a thin re-ranker over 500 items. "Pure online" would need the heavy model in the critical path and miss the SLO; "pure batch" would ship stale rankings and lose to any competitor who re-ranks live.

### 2.4 Scenario D — demand forecast for warehouse replenishment

**Decision: batch.**

- Freshness tolerance: daily (orders ship with 24–48h lead time).
- Latency target: none at the decision point; the replenishment planner runs overnight.
- Input availability: sales aggregates, seasonality calendars, inventory levels — all warehouse-shaped.
- Request volume: ~500k SKUs × ~200 warehouses = 100M combinations, daily.
- Cost: online would be renting warm pods to serve a request shape that happens once a day.

Design: nightly Spark/Dask forecast run → write `(sku_id, warehouse_id, forecast_horizon_h, qty, model_version, run_at)` to the planning warehouse. A single optimisation job reads the table the next morning.

### 2.5 Decision procedure (reusable)

Apply to any new scenario:

1. **Can the input exist before the request?** If no → online (skip the rest).
2. **How stale is "too stale" at consumption?** If > 1 hour → batch is on the table; if < 1 minute → online; between — see 3.
3. **Is the consumer a request/response surface?** If yes → online or hybrid; if a job/analytics surface → batch.
4. **Volume × overlap.** If most records are scored repeatedly with little change → batch + cache beats online.
5. **Resource profile of the model.** If the model does not fit inside the latency budget on sane hardware → batch pre-compute + online re-rank.
6. **Operational cost of staleness vs operational cost of online.** The honest comparison — batch job USD per run vs served QPS USD per month — is what breaks ties.

## 3. Validation steps

A reviewer uses these to check a learner's answer against each scenario.

1. **Freshness fit.** For each scenario, the learner's chosen pattern must satisfy the freshness tolerance. A batch choice that leaves predictions stale by > the tolerance is wrong, regardless of cost.
2. **Latency budget written out.** For an online choice, the answer includes a per-component latency budget that sums under the SLO.
3. **Prediction storage named.** A batch choice names the serving layer (warehouse, feature store, KV); "write to S3" with no downstream index is not a complete answer.
4. **Request-path features declared.** An online choice lists which features are fetched at request time and which come from a store. A missing feature-availability analysis is the most common silent gap.
5. **Hybrid considered.** For scenarios that fall in the middle (Scenario C), the answer must at least mention batch pre-compute + online re-rank and defend why it was or was not chosen.
6. **Governance equivalence.** Both patterns in the answer include: model version on every prediction, a correlation ID (request ID for online, batch run ID for batch), and a prediction log that can be joined to outcomes.
7. **Cost sanity check.** For each pattern chosen, the answer includes an order-of-magnitude cost reasoning (QPS × cost/req, or batch rows × cost/row). A choice made with no cost argument is a guess.
8. **Failure mode named.** For online: what happens if the feature store times out? For batch: what happens if the nightly job fails before the campaign? An answer with no graceful-degradation path is incomplete.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Correct pattern per scenario | 6 | 1.5 each for the four scenarios — the pattern choice is defensible against freshness + latency; a "both work" answer scores half |
| 2 | Decision procedure | 3 | Explicit list of questions the learner used to decide — not a per-scenario hand-wave |
| 3 | Latency budget (online) | 2 | For every online/hybrid choice, a per-component budget that sums under the SLO |
| 4 | Prediction storage (batch) | 2 | For every batch choice, the serving layer is named and the write cadence is specified |
| 5 | Feature availability analysis | 2 | For every online choice, request-time vs store features are enumerated |
| 6 | Hybrid awareness | 1 | The hybrid pattern is considered for at least one scenario and accepted or rejected with reasons |
| 7 | Governance parity | 2 | Model version, correlation ID, and prediction logging called out for *both* patterns, not just online |
| 8 | Cost order-of-magnitude | 1 | A cost figure (even rough) ties the choice to dollars, not just vibes |
| 9 | Failure mode & graceful degradation | 1 | At least one failure path per chosen pattern with a stated behaviour |

Items 1 and 2 are mandatory. A learner who gets the four scenarios right by instinct but cannot state *why* has not learned the thing the exercise teaches — the decision procedure is the artefact, the four answers are the application.

## 5. Common mistakes

- **"Online is modern, batch is legacy."** Both patterns are current. Choosing online for a weekly email campaign burns money and multiplies failure modes for no freshness gain.
- **"Batch is cheaper."** True only when the overlap is high. Scoring 10M users nightly when the campaign emails 10k of them is batch running 1000× the useful work. Cost is a function of pattern *and* consumer footprint.
- **Online without a latency budget.** "FastAPI + GPU" is not a design; the design is "feature fetch 10 ms + predict 30 ms + serialize 5 ms within a p99 150 ms SLO." Without the budget, the first time a feature fetch gets slow, the SLO breaks and nobody knows why.
- **Batch without a serving layer.** Writing predictions to `s3://.../scores/2026-11-05/*.parquet` and expecting the application to read it directly is not a delivery mechanism. A production batch pipeline terminates at an indexed store the consumer can query by key.
- **Treating freshness as constant.** "Freshness: real-time" is almost never true. The honest number is "at most X hours old at consumption"; most ML scores live at 1 hour, 24 hours, or 7 days of tolerance. Pin the number.
- **Ignoring the feature store.** Online inference without an online feature store usually ends up recomputing features from the raw request, which skews from the training distribution and causes train–serve skew — the single largest source of production ML regressions.
- **Micro-batch as "neither."** A 10 ms server-side batch window inside an online path is still online inference; it just amortises the model-call cost. Call it what it is and measure the added tail latency.
- **One governance story for online, none for batch.** Batch jobs lose prediction audit logs more often than online services do — nobody wires up the equivalent of `request_id` and `model_version` on every row. Match the online audit trail row-for-row in the batch output table.
- **Ignoring backfills.** A batch pattern needs an answer to "can we re-score all users with the new model today?" An online pattern needs an answer to "what happens if the model artefact was wrong — can we replay the last day's predictions through the new model?" Both are real asks.
- **Assuming the model is portable.** A model trained with batch features (aggregated windows, slow-moving signals) may not have the signals present at request time, and vice versa. If online and batch inference use different feature sets, they are different models in production, not one model deployed two ways.

## 6. References

- Google Cloud Architecture Center — *MLOps: continuous delivery and automation pipelines in machine learning*, which contrasts online and batch serving topologies and feature-store integration. <https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
- Microsoft Learn — *Machine learning operations (MLOps) for Python*, pattern catalogue for batch vs real-time endpoints on Azure ML. <https://learn.microsoft.com/azure/architecture/ai-ml/guide/mlops-python>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (performance across deployment contexts), `MEASURE 2.6` (deployed system monitoring), and `MANAGE 2.1` (documented provenance). The framework is pattern-agnostic by design; the same controls apply to both batch and online. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft` and `ML06:2023 AI Supply Chain Attacks`; the audit log that mitigates both is required for batch and online alike. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — tactic view covers both batch and online inference as attack surfaces; the data flow differs but the mitigations are shared. <https://atlas.mitre.org/>
