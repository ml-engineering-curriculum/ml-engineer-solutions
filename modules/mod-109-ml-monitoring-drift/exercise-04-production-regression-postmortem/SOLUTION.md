# exercise-04 — Production Regression Post-mortem · SOLUTION

Paired exercise: `lessons/mod-109-ml-monitoring-drift/exercises/exercise-04-production-regression-postmortem.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to produce a blameless post-mortem for a production ML regression — the model's quality (AUC, calibration, or business metric) degraded in production, the on-call was paged, and the system recovered. A good post-mortem for an ML incident differs from a classic SRE post-mortem in three ways:

1. **"Root cause" rarely means one commit.** ML regressions are usually the interaction of a model trained on distribution D, a data pipeline that drifted to distribution D′, and a monitoring system that noticed too late. The post-mortem names *each* contributor.
2. **The scoreboard runs behind reality.** Labelled metrics (AUC, F1, revenue) are available after the label horizon (hours to months). The incident is usually detected by proxies (drift, score shift, downstream funnel) and only *confirmed* by labelled metrics days later. The timeline reflects both.
3. **Action items are policy-level, not just code-level.** Fixing the specific regression is table stakes. The durable fix is to move a trigger, tighten a gate, change a reference distribution, or shorten a label horizon — things that make the same shape of incident a non-event next time.

Design priorities for the write-up, in order:

1. **Blameless framing.** Name systems and decisions, not people. "The gate was configured with `auc_lift_min_pp=0.0`, so a 0-point candidate passed" — not "Alex configured the gate."
2. **Timeline in UTC with evidence links.** Every entry is wall-clock, timezone-explicit, and links to the dashboard/log/ticket that recorded it. A timeline with no links is folklore.
3. **Impact quantified end-to-end.** Users affected, requests mispredicted, revenue delta, downstream SLO breaches. "Model quality dropped" is not an impact.
4. **Contributing factors, not single root cause.** List them, rate them (primary / contributing / latent). The latent ones are where the next incident will come from.
5. **Action items are owned, dated, sized.** Each is assigned to a team, has a due date, and is labelled `preventive` / `detective` / `responsive` — preventive items stop the shape of incident, detective items catch it faster next time, responsive items make recovery cheaper.

This design sits inside the NIST AI RMF `MANAGE 4.1` (post-deployment monitoring and response) and `MANAGE 2.4` (incident response planning for AI systems). It is also the operational mirror of OWASP ML Top 10 `ML02:2023 Data Poisoning Attack` — the same forensic rigour applies whether distribution shift is organic or adversarial, and the post-mortem is the record that lets a future reviewer tell the two apart.

## 2. Worked implementation (post-mortem)

> The content below is a fully written example incident for `churn-v3` on **2026-10-21**. The learner's submission should match the structure and detail level — the specific numbers, dates, and services are illustrative of the format. All systems, people, and metric values below are fictional and used solely as exercise context.

### 2.1 Summary

Between **2026-10-21 02:14 UTC** and **2026-10-21 18:40 UTC** the production `churn-v3` model mispredicted the "likely-to-churn" audience for the weekly retention email. Precision at the top decile dropped from the training baseline of **0.71** to **0.42** for ~16.5 hours. The downstream retention campaign sent discount offers to **~320 000 non-churning users** (false positives) and missed **~41 000 churning users** (false negatives). Estimated gross-margin impact: **$84 000** (confidence interval ~±20%, derived from the campaign-attributed lift model). The direct cause was an upstream data-pipeline change that silently nulled the `support_tickets_30d` feature; the model encoded `NaN` as "0 tickets" and therefore predicted most users as low-risk. The system was recovered by rolling back to `churn-v2`.

### 2.2 Impact

| Dimension | Value | Source |
|---|---|---|
| Duration (from degradation to recovery) | 16h 26m | Grafana `ml-service/churn-v3`, 2026-10-21 |
| Users mispredicted (false negatives) | ~41 000 | `analytics.email_outcomes_20261021`, ticket `OPS-5811` |
| Users mispredicted (false positives) | ~320 000 | same |
| Downstream campaign cost (discount offers to FPs) | $62 000 | `finance.campaign_cost_20261021` |
| Estimated retention revenue lost (FNs) | $22 000 | `analytics.retention_impact_model_v2` |
| Pages fired | 1 (`FeatureOutOfRangeSpike`, 03:31 UTC) | Alertmanager, 2026-10-21 |
| Pages that should have fired earlier | 2 (`FeatureNullRate`, `ScoreDistributionDrift`) | this incident |

Severity: **SEV-2** (business metric impacted; no data loss, no privacy incident).

### 2.3 Timeline (UTC)

| Time | Event | Evidence |
|---|---|---|
| 2026-10-20 23:50 | `data-platform` deploys `tickets-pipeline@v4.2.0`, which changes the join key for the support-tickets fact table from `user_id` to `canonical_user_id`. Downstream `features-service` is not updated. | `platform` PR #2812 |
| 2026-10-21 00:00 | Hourly feature job runs; `support_tickets_30d` returns NULL for all rows because the join misses. The feature-service response fills NULL with `0`. | `features-service` logs, 00:03 UTC |
| 02:14 | `churn-v3` hourly batch scoring runs; 99.4% of users receive a `support_tickets_30d=0` feature value. Model scores collapse toward "low risk" (median score drops from 0.31 to 0.12). | `ml_prediction_score` p50 series, Grafana |
| 03:31 | Alert `FeatureOutOfRangeSpike` fires: `support_tickets_30d` null rate crosses 50%. On-call acknowledges. | Alertmanager `APL-10021` |
| 03:35 | On-call opens `features-service` logs, sees nulls, pages `data-platform` on-call. | Pager log |
| 04:12 | Data-platform confirms the pipeline change at 23:50 and the missing update on `features-service`. Fix PR drafted. | Slack #ml-oncall |
| 04:48 | Decision: do not hot-patch the pipeline during peak-send window; flag the issue and let the morning campaign send. (This decision is identified in §2.5 as a contributing factor.) | Slack thread |
| 09:30 | Retention campaign batch ships using corrupted scores; ~360 k emails sent. | `campaign-service` logs |
| 12:05 | `ml_rolling_auc_7d` labelled metric catches up: top-decile precision recorded as 0.42 (baseline 0.71). Second page fires. | Grafana `ml-quality` board |
| 12:30 | Incident escalated to SEV-2. ML on-call initiates registry rollback. | Incident channel |
| 17:50 | Rollback alias `production → churn-v2` applied. Next scoring batch uses v2. | Registry audit log |
| 18:40 | Score distribution returns to within baseline band. Incident declared mitigated. | Grafana |
| 2026-10-23 | Labelled recovery confirmed: top-decile precision on 2026-10-22 scoring is 0.70 (baseline 0.71). | `analytics.email_outcomes_20261022` |

### 2.4 Root-cause analysis

#### 2.4.1 The proximate chain

1. The data-platform deploy changed the join key.
2. The feature pipeline silently returned NULL for `support_tickets_30d`.
3. The feature-service NULL-handler coerced NULL to `0`.
4. The model treats `support_tickets_30d=0` as "no support activity" — a strong negative signal for churn.
5. The daily campaign shipped the scores before the labelled quality check could catch the regression.

#### 2.4.2 Contributing factors (ranked)

| # | Factor | Kind | Rating |
|---|---|---|---|
| 1 | `features-service` silently coerces NULL to `0` instead of flagging | Technical debt | Primary |
| 2 | `data-platform` deploys do not notify downstream feature owners | Process | Primary |
| 3 | The drift job's threshold on `support_tickets_30d` null rate was WATCH, not DRIFT, until 50% — the alert fired, but too late to block the 09:30 campaign | Configuration | Contributing |
| 4 | Campaign scoring and campaign send are the same job — there is no quality-gate step between scoring and sending | Architecture | Contributing |
| 5 | Decision at 04:48 to let the morning send proceed before confirming score quality | Decision | Contributing |
| 6 | Labelled metric (`ml_rolling_auc_7d`) has a label-delay of ~8 hours after the campaign send, so it cannot catch a regression in time to block the same-day send | Latent | Latent |
| 7 | There is no shadow-scoring canary between feature-service and the model | Latent | Latent |

The ratings matter: fixing only the Primary factors would have reduced the impact but not eliminated it. Fixing the Latent factors is what makes the shape of the incident unreachable, not just unlikely.

### 2.5 What went well

- The `FeatureOutOfRangeSpike` alert fired at 03:31, which is the alert that this exercise's `exercise-01` and `exercise-02` solutions are designed to produce. Without the per-feature null-rate counter, no alert would have fired until the labelled metric at 12:05 — an additional 8.5 hours of impact.
- The rollback took **5 minutes 40 seconds** from decision to applied (17:50–12:30 wall-clock is dominated by human decision time, not the technical rollback). The registry-alias rollback pattern from `exercise-03` worked exactly as designed.
- Blameless framing held throughout the incident channel. Every message named a system or a decision, not a person.

### 2.6 What went badly

- The drift job's null-rate threshold was *set*, but the policy allowed the campaign to send *while the alert was live*. Alerts without automated gating of downstream consumers are advisory at best.
- The decision at 04:48 UTC to let the morning send proceed was made in Slack, by a single on-call engineer, under time pressure, with no documented runbook for "feature regression during pre-campaign window." The decision was reasonable given the information available; the absence of a runbook is the failure.
- The labelled quality metric took 8+ hours to register the regression. For this product, that is longer than the campaign's publish window. Labelled metrics are retrospective and cannot be the primary control.

### 2.7 Action items

Each item is owned, dated (today = 2026-11-05 for scheduling), and typed. "Preventive" stops the shape; "detective" catches it faster; "responsive" cuts recovery time.

| ID | Action | Owner | Due | Type |
|---|---|---|---|---|
| AI-1 | Replace NULL→0 coercion in `features-service` with explicit propagation of a `missing` sentinel; the model treats it as an unknown category/value, not as `0`. Add a feature contract test. | `platform-ml` | 2026-11-19 | Preventive |
| AI-2 | Add "notify downstream feature consumers" as a required field in the data-platform deploy checklist; wire the notification to `#ml-oncall`. | `data-platform` | 2026-11-12 | Preventive |
| AI-3 | Tighten the drift job policy for `support_tickets_30d` null rate: 5% → WATCH, 20% → DRIFT (today: 25% → WATCH, 50% → DRIFT). | `ml-platform` | 2026-11-12 | Detective |
| AI-4 | Add a quality gate between the scoring job and the campaign job: if any of (`drift verdict in {DRIFT}`, `null-rate > 20%`, `score distribution p50 shift > 0.1 vs 7d baseline`) is true, the campaign job stops and pages. | `ml-platform` | 2026-11-26 | Preventive |
| AI-5 | Write runbook `ml-ops/runbooks/feature-regression-pre-campaign.md` with decision tree: when to hotfix, when to roll back, when to delay the campaign. | `ml-oncall` | 2026-11-19 | Responsive |
| AI-6 | Add shadow-scoring canary: `features-service` responds to a 1% synthetic probe every 15 minutes; the model scores the probe; the score distribution for the probe is compared to a frozen reference. Alert on >2σ shift. | `ml-platform` | 2026-12-10 | Detective |
| AI-7 | Build a label-free proxy for top-decile precision (e.g., calibration of the predicted score against the previous 7 days of scores), to shorten the current 8-hour label-delay feedback loop. | `ml-platform` | 2026-12-10 | Detective |
| AI-8 | Rollback automation: on `FeatureOutOfRangeSpike` *and* a confirming `ScoreDistributionDrift` within 30 minutes, auto-revert the registry alias with a human-in-the-loop confirmation step. | `ml-platform` | 2027-01-07 | Responsive |

### 2.8 Lessons

1. **A fallback value is a decision.** `features-service` chose to return `0` for a NULL; that decision made a broken pipeline look like a healthy pipeline. The model inherited the mislabeling and ran with it.
2. **Alerts without gates are advisory.** The null-rate alert fired on time. The campaign shipped anyway, because nothing was wired to stop it. Detection and control are different problems and must both be solved.
3. **The labelled metric is not the primary signal.** It is the confirmation. For products with same-day action loops, label-delay is longer than the action window, so leading indicators (drift, score distribution, null rates) are the only controls that can act in time.
4. **Rollback cheapness has compounding value.** A fast rollback (seconds, via registry alias) turns an SEV-1 into an SEV-2. This is the single highest-leverage investment in ML ops.
5. **Blameless post-mortems produce action items; blaming post-mortems produce silence.** Every finding here names a system or a configured threshold — not a human — which is why Section 2.7 has 8 action items instead of 0.

## 3. Validation steps

For the learner's submitted post-mortem:

1. **Timeline check** — every row has a UTC timestamp and a link to an evidence source (dashboard panel, log search, ticket, chat thread). No "around 3am"; no undated rows.
2. **Impact check** — impact is quantified in at least three dimensions (duration, user count, business metric). "Model quality dropped" alone fails.
3. **Blameless check** — zero human names in the Factors and Action Items sections. Systems, decisions, and configurations only.
4. **Factor rating check** — every contributing factor is rated Primary / Contributing / Latent; the ratings are justified in prose.
5. **Action item check** — each action has an owner (team, not person), a due date, and a type (preventive / detective / responsive). A post-mortem with only `responsive` items has accepted that the incident will recur.
6. **Leading-indicator check** — the post-mortem names at least one leading indicator (drift, null rate, score distribution) that would have fired earlier, and the action items include at least one `detective` item that tightens or adds one.
7. **Follow-through** — all action items are tracked in the team's issue tracker with the post-mortem linked as the parent. A post-mortem that is not tracked is a document, not a commitment.
8. **Review and close** — a post-mortem review meeting is held within 10 business days; owners confirm the plan; the document is marked as reviewed with a date and attendee list.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Summary with duration, direct cause, business impact | 2 | One paragraph; a reader who stops here knows what happened and how bad it was |
| 2 | Quantified impact (≥3 dimensions) | 2 | Users / requests / revenue / SLO / page count — at least three with numbers and sources |
| 3 | Timeline in UTC with evidence links | 3 | Every row timestamped and linked; no gaps longer than 2h without an entry |
| 4 | Contributing factors (not single root cause) | 3 | ≥3 factors, each rated Primary/Contributing/Latent with a one-line justification |
| 5 | "What went well" and "what went badly" are both present | 1 | Both sections exist; "went well" is not empty |
| 6 | Action items owned, dated, typed | 3 | Each has team owner, due date, and type (preventive/detective/responsive); at least one of each type |
| 7 | Leading-indicator strategy | 2 | At least one leading indicator named; at least one action item shortens the detection horizon |
| 8 | Rollback mechanics discussed | 1 | Rollback method, time-to-rollback, and whether it's being made faster |
| 9 | Blameless framing | 2 | No personal names in findings; systems and decisions, not people |
| 10 | Tracked and reviewed | 1 | Action items in issue tracker; review date set |

Items 4 and 6 are mandatory: a post-mortem that identifies only one root cause and lists only remedial fixes has not learned from the incident.

## 5. Common mistakes

- **Naming people.** "Alex merged the pipeline change" becomes "nobody reviews pipeline changes when Alex is on-call." The blameless rewrite is "the pipeline-deploy process does not require ML-side review" — which is actionable.
- **Single root cause.** ML regressions are system interactions. The commit is a trigger; the latent factors are the reason the trigger was dangerous. Name both.
- **Timeline without evidence links.** "At 03:30 the alert fired" is a claim. "At 03:30 the alert fired [Alertmanager APL-10021]" is a record. Six months later, only the record is usable.
- **Impact as "quality degraded".** Quantify to users, requests, revenue, SLO. If the data is not available, write "impact is estimated at X; the data-platform team is adding Y to make the next estimate measurable."
- **Only responsive action items.** "Improve the runbook so rollback is faster" is useful; it does not prevent the incident. A preventive action item is one that would have made the shape of the incident impossible.
- **Action items without owners or dates.** An action item without an owner is a wish. Issue tracker + due date + team owner is the minimum.
- **Marking the post-mortem "done" when the doc is written.** The post-mortem is done when the action items are closed. Review-date and tracking are what close the loop.
- **Skipping "what went well."** Leaving the section blank treats the response as all failure; it produces a worse incident next time because the things that worked (the alert that fired, the rollback that worked) are not reinforced.
- **Treating label-delayed metrics as the primary detector.** In same-day action loops, the labelled metric arrives after the damage. The post-mortem must name a leading indicator and an action item that tightens it; otherwise the next incident recurs for the same duration.

## 6. References

- Google SRE Workbook — Postmortem Culture: Learning from Failure (blameless post-mortem framing and checklist). <https://sre.google/workbook/postmortem-culture/>
- NIST AI Risk Management Framework 1.0 — `MANAGE 4.1` (post-deployment monitoring and response) and `MANAGE 2.4` (incident response planning). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack`; the forensic pattern is the same for organic drift and adversarial manipulation, and the post-mortem is the record that distinguishes them. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0020 Poison Training Data`; threat-model counterpart to naturally-occurring distribution shift. <https://atlas.mitre.org/>
