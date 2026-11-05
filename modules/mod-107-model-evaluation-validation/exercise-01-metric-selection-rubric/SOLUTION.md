# exercise-01 — Metric Selection Rubric · SOLUTION

Paired exercise: `lessons/mod-107-model-evaluation-validation/exercises/exercise-01-metric-selection-rubric.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to produce a defensible **metric selection rubric**: a short document that, for a specific ML task, declares which metric(s) the model will be judged on, why, at what threshold, and how ties are broken. The failure mode this rubric exists to prevent is the one every team hits eventually — a model that ships because "accuracy went up," then quietly underperforms on the only slice the business cares about.

The rubric must answer, in order:

1. **What decision does the model actually drive?** The metric follows the decision, not the other way round. A ranker, a thresholded classifier, and a probability feed into three different decision shapes and therefore three different metrics.
2. **What are the costs of each error class?** False positives and false negatives are almost never symmetric. Writing down the cost ratio — even as a rough order of magnitude — is what turns "F1 is balanced" from a tautology into a choice.
3. **What is the prevalence / base rate?** Accuracy on a 99:1 imbalance is uninformative. The rubric must name a prevalence-robust primary metric (PR-AUC, recall at fixed precision, cost-weighted loss) for any task with class imbalance above roughly 10:1.
4. **Which slices matter, and how are they aggregated?** A single scalar hides slice regressions. The rubric names the slices (by product, region, device, protected attribute) and whether slice performance is averaged, min-pooled, or gated separately.
5. **What is the operating point?** Threshold-free metrics (AUC, PR-AUC) rank models; thresholded metrics (precision/recall) decide shipping. The rubric must fix the operating point *before* model comparison, otherwise the winning model is chosen after the fact.
6. **What goes on the model card / evaluation report?** NIST AI RMF `MEASURE 2.3` expects every documented metric to have a definition, a data slice, a confidence interval, and a date. "AUC = 0.87" alone is not a measurement.

This rubric is one artifact in the larger NIST AI RMF `MEASURE 2.3` / `MEASURE 2.11` loop (metrics are defined, re-measured, and compared across versions). It is also the first line of defence against OWASP ML Top 10 `ML02:2023 Data Poisoning Attack` and `ML05:2023 Model Theft` symptoms — a drop on the right slice is often the first observable signal.

| Decision | Alternative | Why this one |
|---|---|---|
| Primary metric fixed *before* training starts | Pick after seeing results | Choosing after-the-fact is a classic researcher-degrees-of-freedom leak; the model that "won" was selected on the metric it happened to be best at |
| Separate "primary" vs "guardrail" metrics | Single scalar | Primary answers "which model is better?"; guardrails catch "did we silently regress on fairness, latency, calibration?" |
| Operating point declared as (threshold, precision, recall) triple | Just a threshold | Thresholds move with calibration; the triple is self-describing and survives recalibration |
| Confidence intervals via bootstrap, reported on every metric | Point estimates | A 0.87 ± 0.02 vs 0.88 ± 0.02 comparison tells the truth; "0.87 vs 0.88" does not |
| Slice metrics reported with sample size | Slice metrics alone | A slice of n=12 is noise; the sample size is part of the measurement |
| Cost matrix written down even when rough | "We'll figure out costs later" | Any FP/FN ratio you can defend is better than the implicit 1:1 the default F1 bakes in |

## 2. Worked implementation: rubric template + two filled examples

### 2.1 The template

Every metric rubric in the project follows this structure. Keep it short — one page per task.

```markdown
# Metric Rubric — <task name>

## 1. Decision
- What does this model output drive? (ranking / filter / score shown to user / automated action)
- Who acts on the output? (another model / a human / a service)
- What is the cost of the wrong decision, in business terms?

## 2. Task framing
- Task type: binary classification / multiclass / regression / ranking / retrieval / generation.
- Prevalence / base rate of the positive class (or label distribution).
- Volume of decisions per day (sets the statistical floor for CIs).

## 3. Primary metric
- Name, precise definition (library + function + arguments), unit.
- Why this one, given sections 1 and 2.
- Target value or shipping bar.
- Confidence interval method (bootstrap n=…, percentile).

## 4. Operating point (thresholded tasks only)
- Threshold: fixed value OR "choose threshold that yields precision ≥ X on dev".
- Expected (threshold, precision, recall, prevalence) on the current dev set.
- Recalibration policy when prevalence shifts.

## 5. Guardrail metrics
- Metric | Direction | Hard floor (ship-block) | Soft floor (review)
- Include at least: fairness-across-slices, calibration, latency, cost.

## 6. Slices
- Slice | Definition | Min sample size | Report as (per-slice / worst-slice / weighted)
- Protected attributes handled per NIST AI RMF MEASURE 2.11.

## 7. Baselines to beat
- Trivial baseline (majority class / mean).
- Rule-based baseline (if one exists in production today).
- Previous model version (champion).

## 8. Reporting
- Section of the model card where this metric appears.
- Frequency of re-measurement in production.
- Who signs off on the number (role, not person).
```

### 2.2 Filled example A — fraud filter, high-stakes imbalanced classifier

```markdown
# Metric Rubric — Payments Fraud Filter v3

## 1. Decision
- Output drives an automated hold on the transaction (score > threshold → review queue).
- Reviewer (human) acts on the hold within 30 s SLA.
- Cost of false positive: angry legitimate customer, ~$8 chargeback-support cost, retention risk.
- Cost of false negative: average $240 chargeback loss + fine exposure.
- Rough cost ratio FN:FP ≈ 30:1.

## 2. Task framing
- Binary classification; label = confirmed-fraud within 60 days.
- Prevalence on live traffic: 0.6 % (highly imbalanced).
- ~4 M decisions/day.

## 3. Primary metric
- **Recall at precision = 0.90**, computed on the held-out week.
- Library: `sklearn.metrics.precision_recall_curve`, pick max recall where precision ≥ 0.90.
- Why: the reviewer queue size is capped; we can afford at most 10 % false positives in the holds. Within that constraint, catching more real fraud is the only axis that matters.
- Shipping bar: ≥ 0.72 (current champion is 0.69).
- CI: percentile bootstrap, n = 2 000 resamples of transactions grouped by account_id.

## 4. Operating point
- Threshold is chosen per deploy to hit precision ≥ 0.90 on the dev week.
- Expected triple on dev: (threshold=0.84, precision=0.91, recall=0.73, prevalence=0.6 %).
- Recalibration: isotonic re-fit when weekly observed precision drifts ±5 pts for two consecutive weeks.

## 5. Guardrail metrics
| Metric | Direction | Hard floor | Soft floor |
|---|---|---|---|
| Recall on worst-performing issuer bank (min across banks ≥ 5 k txns) | ↑ | ≥ 0.55 | ≥ 0.65 |
| Calibration (expected calibration error, 10 bins) | ↓ | ≤ 0.04 | ≤ 0.02 |
| Serving p95 latency | ↓ | ≤ 40 ms | ≤ 25 ms |
| Reviewer queue size (FP count / day) | ↓ | ≤ 24 000 | ≤ 18 000 |

## 6. Slices
- Issuer bank (top 20), card type (credit/debit/prepaid), channel (card-present / CNP / 3DS),
  country, high-value flag (txn ≥ $500). Minimum n = 5 000 per slice.
- Reported as per-slice table **and** worst-slice recall as a scalar on the model card.
- No protected-attribute slicing; decision is on transactions, not persons. Documented per NIST AI RMF MEASURE 2.11.

## 7. Baselines to beat
- Majority-class baseline: recall = 0.
- Rules engine (today's production): recall = 0.44 at precision 0.90.
- Champion model v2: recall = 0.69 at precision 0.90.

## 8. Reporting
- Appears in model card sections: "Intended use", "Metrics", "Known failure modes".
- Re-measured hourly on the last 7 days of production traffic; daily summary posted to the fraud-ops channel.
- Sign-off: Head of Payments Risk.
```

### 2.3 Filled example B — ranker, revenue-sensitive retrieval

```markdown
# Metric Rubric — Product Search Ranker v1.4

## 1. Decision
- Output is a ranked list of up to 50 SKUs shown to a shopper.
- Shopper clicks (or abandons); downstream decisions are human.
- Cost of a bad ranking: abandonment + lost revenue on the session.
- No single FP/FN; graded via click/purchase relevance labels 0–3.

## 2. Task framing
- Learning-to-rank over query–document pairs.
- ~90 M ranked queries/day; eval set is 40 k sampled queries with human-graded top-20.
- Prevalence not applicable; relevance distribution reported instead.

## 3. Primary metric
- **nDCG@10**, computed on the human-graded eval set.
- Library: `sklearn.metrics.ndcg_score` with `k=10` and gains = `2^rel - 1`.
- Why: shoppers see the top of the list; nDCG weights higher positions and matches revenue elasticity to top-K.
- Shipping bar: +0.5 pp over champion with 95 % bootstrap CI not crossing zero.
- CI: paired bootstrap at the query level, n = 10 000.

## 4. Operating point
- N/A — threshold-free metric for ranking.

## 5. Guardrail metrics
| Metric | Direction | Hard floor | Soft floor |
|---|---|---|---|
| nDCG@10 on long-tail queries (query freq < 10/mo) | ↑ | ≥ champion − 0.3 pp | ≥ champion |
| Mean reciprocal rank on single-product-intent queries | ↑ | ≥ champion − 1 pp | ≥ champion |
| Items-with-zero-impressions-over-24h (freshness) | ↓ | ≤ champion + 10 % | ≤ champion |
| p99 serving latency | ↓ | ≤ 120 ms | ≤ 90 ms |

## 6. Slices
- Query head/torso/tail (by frequency decile), query language, device (mobile/desktop),
  category (electronics, apparel, grocery…). Minimum n = 1 000 queries per slice.
- Reported as a table + worst-decile nDCG on the model card.

## 7. Baselines to beat
- Random order baseline: nDCG@10 ≈ 0.11.
- BM25 baseline: nDCG@10 = 0.31.
- Champion v1.3: nDCG@10 = 0.46.

## 8. Reporting
- Appears in model card sections: "Metrics", "Slicing analysis".
- Re-measured weekly on fresh judged eval set.
- Sign-off: Search Product Lead.
```

The two examples sit at opposite ends of the metric-selection space on purpose: a thresholded classifier with asymmetric costs, and a threshold-free ranker with graded relevance. Any new task should map onto one of these shapes before the rubric is filled in.

## 3. Validation steps

A completed rubric is reviewed, not run. Use this checklist in a pull request review of the rubric document.

1. **Decision is concrete.** "Improve recommendations" fails; "move user to a 50-item ranked list" passes. If you cannot name the UI or API that consumes the output, stop and go back to section 1.
2. **Primary metric name is executable.** `sklearn.metrics.ndcg_score(..., k=10)` passes; "F1-ish" fails. A second engineer must be able to reproduce the metric without asking questions.
3. **Prevalence-aware.** If the positive class is < 10 % and the primary metric is `accuracy` or raw `F1`, reject the rubric. Use PR-AUC, recall-at-fixed-precision, or a cost-weighted metric.
4. **Operating point fixed before comparison.** The threshold (or the rule for picking it) is written down before any candidate model is scored. If you change the rule to make the new model win, the comparison is invalid.
5. **Guardrails are non-empty.** At least one fairness/slice metric, one calibration metric, and one latency/cost metric. A rubric with only the primary metric cannot catch silent regressions.
6. **Every number has a sample size and a CI.** `0.73 (bootstrap 95 % CI [0.71, 0.75], n = 2 000 resamples, grouped by account_id)` is a measurement. `0.73` alone is not.
7. **Slices named, not implied.** The slice list is explicit, each with a minimum sample size. "By country" fails; "country ∈ {top 10 by volume}, min n = 5 000" passes.
8. **Baselines include the current production system.** If you cannot beat the current rules engine or the champion model, you cannot ship. The rubric must name both.
9. **Sign-off role.** A human role is named — "Head of X," not "Alice." Rubrics outlive people.

If any check fails, the rubric is rejected; the model is not evaluated against an unsigned rubric.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Decision stated concretely | 2 | Names the consumer of the output and the action taken; cost of error quantified at least to an order of magnitude |
| 2 | Task framing | 1 | Task type, prevalence (or distribution), decision volume all present |
| 3 | Primary metric fit for prevalence | 3 | Accuracy / plain F1 rejected when imbalance > 10:1; a prevalence-robust alternative is chosen with justification |
| 4 | Metric is reproducible | 2 | Library, function, and arguments named; two engineers would compute the same number |
| 5 | Operating point fixed in advance | 2 | Threshold or rule-for-threshold set before candidate models are scored |
| 6 | Guardrail metrics present | 3 | ≥ 1 fairness/slice, ≥ 1 calibration, ≥ 1 latency/cost guardrail, each with hard & soft floors |
| 7 | Slice list with sample sizes | 2 | Slice names and min-n explicit; aggregation rule (per-slice / worst / weighted) stated |
| 8 | Baselines named | 2 | Trivial baseline + production system + previous champion all listed with current numbers |
| 9 | Confidence intervals specified | 2 | CI method named (e.g., percentile bootstrap, n=…) and attached to the primary metric |
| 10 | Sign-off role present | 1 | A role (not a person) owns the number |

Items 3, 5, and 6 are mandatory floors: a submission missing any of them cannot score above 12 / 20 regardless of the rest. These are the three failure modes that most often ship a broken model past the rubric stage.

## 5. Common mistakes

- **Accuracy on imbalanced data.** A churn model on 2 % prevalence gets 98 % accuracy by predicting "no churn" for everyone. The rubric must reject accuracy as the primary metric whenever prevalence < ~10 % (or > ~90 %).
- **Metric picked after seeing results.** Running three models, noticing which metric model C wins on, and declaring that metric primary is a researcher-degrees-of-freedom leak. Fix the metric *first*, then compare.
- **Single-scalar reporting.** AUC = 0.87 with no CI, no slice breakdown, and no operating-point triple cannot be reasoned about. A future reviewer cannot tell whether the model is better than last week, let alone on the slice that matters.
- **F1 as a "balanced" default.** F1 implicitly weights precision and recall equally, which is a cost choice, not a neutral one. If the cost ratio is 30:1, use Fβ with β chosen from the ratio, or recall-at-fixed-precision.
- **Confusing ranker with classifier.** Using accuracy on a top-K recommendation is a type error — the model outputs a permutation, not a label. Ranking tasks take nDCG / MAP / MRR; classification tasks take precision/recall/calibration.
- **Threshold chosen on the test set.** Fixing the threshold on the same data you report precision/recall on inflates both. Use a dev set (or a time-separated slice) for threshold selection; report on untouched test data.
- **Slice metrics without minimum sample size.** A slice with n = 15 will swing wildly run-to-run. Setting `min_n` per slice is the mechanical defence against false-positive fairness alerts.
- **No calibration guardrail on probability outputs.** A model used as a probability (for expected-value math downstream) must be calibrated, not just discriminative. ECE / Brier belong in the guardrails.
- **Protected attributes as model inputs by default.** If the rubric declares protected-attribute slices, declare separately whether those attributes are used as features; NIST AI RMF `MAP 1.1` / `MEASURE 2.11` expects that distinction to be explicit.
- **Rubric lives in a Slack thread.** The rubric belongs in the repository next to the training code, versioned in git. If the rubric changes, the change shows up in `git log` next to the model code that was evaluated under it.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (metric documentation, including definition, slice, CI, and date) and `MEASURE 2.11` (fairness and bias measurement across identified subpopulations). <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF 1.0 — `MAP 1.1` (intended use and context of use must be documented) grounds "what decision does the model drive." <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` and `ML05:2023 Model Theft` describe failure modes whose earliest signals are slice-level metric regressions, not aggregate scores. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0031 Erode ML Model Integrity` for the adversary lens on why guardrail and slice metrics matter alongside the primary metric. <https://atlas.mitre.org/>
- scikit-learn — authoritative definitions of `precision_recall_curve`, `ndcg_score`, `roc_auc_score`, `brier_score_loss`, and calibration tooling. <https://scikit-learn.org/stable/modules/model_evaluation.html>
