# exercise-04 — Build vs Buy, and Delegating to Peer Specialist Tracks · SOLUTION

Paired exercise: `lessons/mod-110-ml-systems-design/exercises/exercise-04-build-vs-buy-and-delegate.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to produce a defensible build-vs-buy decision for an ML subsystem *and* to decide which pieces of adjacent work should be delegated to peer specialist tracks (data engineering, platform/SRE, security, MLOps/infra). The deliverable is a **decision memo** — not an architecture diagram — that a staff engineer or engineering manager could sign off on.

Design priorities, in order:

1. **Decide on the right unit.** Build-vs-buy is a per-*component* question, not a per-*system* question. A fraud scoring system might build the model and buy the feature store; a recommender might buy the embedding store and build the ranker. One-answer-per-system decisions almost always destroy value.
2. **Total cost of ownership over multiple years, not sticker price.** The real cost of "buy" is integration, vendor lock, and a recurring line item. The real cost of "build" is team-years and the opportunity cost of what the team is not building. Both have to be in the same unit (dollars, team-months, or risk-adjusted probability of success) to compare.
3. **Critical-path risk is asymmetric.** A component on the critical path to revenue cannot have its delivery date be hostage to an external roadmap. Some components should be bought *even when cheaper to build*, and some should be built *even when a vendor is adequate*, purely because of risk.
4. **Delegate by expertise, not by availability.** A piece of work gets delegated to a specialist track because that team's experience makes their answer durably better — not because the ML team is temporarily overloaded. Delegation driven by capacity creates the next incident.
5. **Keep an interface, not a dependency.** Whether a component is built, bought, or delegated, it meets the rest of the system through a stable interface you define. If the interface is weak, swapping build for buy (or vendor A for vendor B) in year 2 is a rewrite.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Per-component decision, not per-system | One build/buy call per project | Prevents forcing a bad answer on components where the other side is better |
| 3-year TCO with named inputs | Qualitative "cheaper in the long run" | TCO forces hidden costs (integration, exit, retraining) onto the page |
| Explicit "irreversibility cost" | Reversibility assumed free | Vendor lock-in and migration costs dominate the year-3 math more often than feature parity |
| Delegate by expertise + interface, keep ownership of the outcome | Delegate by capacity | The ML team remains accountable for the system even when it does not build every piece; interface ownership is what makes that possible |
| Signed decision memo committed to repo | Hallway-chat decision | Forces the tradeoffs onto a page that a reviewer can argue with |
| Named exit plan for every "buy" | Buy without exit plan | Year-3 vendor renegotiation leverage comes from knowing — and having rehearsed — the exit |

## 2. Implementation — worked memo on the fraud scoring system, component-by-component

The template below is a complete, signable decision memo for a fictional but realistic set of components. Section headings match what a reviewer actually looks for.

### 2.1 Context

> Our team (6 engineers: 4 ML, 2 platform) owns the fraud scoring system. Target launch is 6 months. Transactions: ~100M/day. We need to decide, per component, whether we build, buy, or delegate. The components under decision are: feature store, model training platform, model serving, drift monitoring, labeling tooling, reason-code UI for analysts. Peer tracks available for delegation: data engineering (owns ingestion and warehouse), platform/SRE (owns k8s + observability), security (owns IAM, secret management, threat modeling), MLOps infra (central team, owns the shared tracking server and model registry).

### 2.2 Decision inputs, per component

A single table is the reviewer's north star. Every column is an input the reviewer can push on.

| Component | 3-yr TCO build | 3-yr TCO buy | Critical path? | Differentiating? | Reversibility of buy | Decision |
|---|---|---|---|---|---|---|
| Feature store (online + offline) | ~2.5 eng-yr | ~$180k/yr license + 0.5 eng-yr integration | Yes | No | Moderate (schema is ours) | **Buy** |
| Model training pipeline (orchestrator + experiment tracking) | ~0.5 eng-yr if we reuse MLOps infra | ~$60k/yr for managed MLflow-equivalent | No | No | High | **Delegate to MLOps infra** |
| Model serving (low-latency scoring service) | ~1 eng-yr | ~$120k/yr for managed serving + 0.4 eng-yr integration | Yes | Partially (latency SLO) | Moderate | **Build** |
| Drift & data-quality monitoring | ~0.8 eng-yr | ~$75k/yr | No | No | High | **Buy** |
| Labeling tooling (analyst UI + label store) | ~1.5 eng-yr | ~$0 for OSS core + 0.6 eng-yr customization | Yes | Yes (reason codes drive precision) | High (it is our UI) | **Build on OSS core** |
| Drift pipeline feature-join | ~0.3 eng-yr (same transforms as training) | n/a | No | No | n/a | **Delegate to data eng** |
| Reason-code generation + model card | ~0.3 eng-yr | n/a (bespoke) | Yes | Yes (regulatory artifact) | n/a | **Build** |
| IAM, secret management, encryption-at-rest, threat model | n/a | n/a | Yes | No | n/a | **Delegate to security** |

Values are honest estimates for this scenario; a real memo would footnote each with how it was arrived at and what the error bar is.

### 2.3 Decision rationale, per component

Each row of the table gets a paragraph. The rationale is the memo's substance; the table is the index.

**Feature store — buy.** Building a low-latency online store that stays consistent with the offline store is a two-engineer-year commitment with a non-trivial failure tail (see exercise-02 on training-serving skew). The shortlist of commercial and OSS-with-support options reliably serve <5ms p99 lookups at our scale. The component is critical path but not differentiating — customers do not care which feature store we use; they care that fraud decisions are fast and correct. Buy; keep the schema and feature-definition DSL in our repo so the exit plan is a port-of-schema, not a rewrite-from-scratch.

**Training pipeline — delegate to MLOps infra.** The central MLOps team already runs a tracking server and model registry, and a shared pipeline reduces duplicated effort across ML teams. Delegation here is by expertise (MLOps runs this every day; we do not) and by shared surface (tracking server, registry) — not by capacity. Our side of the interface is: our training script calls a well-defined CLI/SDK, logs to the shared tracking URI, and registers artifacts under our project namespace. The MLOps team owns uptime and cross-team compatibility; we own what we log.

**Model serving — build.** p99 latency of 80 ms end-to-end leaves very little budget for a general-purpose serving platform. A single-tenant service that loads the model in-process is cheaper to operate at our scale and gives us full control over the latency budget (which is the one SLO we cannot renegotiate). A managed serving platform is reconsidered if (a) our model's memory footprint grows beyond what in-process can hold, or (b) we need multi-model routing. Rebuild cost is captured.

**Drift and data-quality monitoring — buy.** Monitoring is not a differentiator for fraud; it is a table-stakes operational practice. A commercial drift tool with pre-built checks on PSI, KS, missing-rate, and schema conformance gets us to monitored-production faster than hand-rolling the same checks. Exit plan: all drift signals also land on our Prometheus/Grafana stack, so if the vendor becomes unacceptable, our dashboards do not go dark during migration.

**Labeling tooling — build on OSS core.** This is where fraud-scoring differentiates: reason codes, fields analysts can bulk-action on, label-source tracking (analyst vs. chargeback), four-eyes review workflow. OSS projects cover the core (task assignment, UI shell) but none support our analyst flow out of the box. Build the customization layer on OSS; do not build the task-assignment primitive. The build is scoped and the differentiation justifies the eng-month cost.

**Drift pipeline feature-join — delegate to data eng.** The join between serving logs and analyst labels is a warehouse-side ELT step, not a serving-side feature engineering step. Data engineering owns the warehouse and has built this kind of join for three other teams. Delegating by expertise. Our side of the interface is a documented schema for serving logs and a documented schema for the labels; data eng owns the SQL and the SLA.

**Reason codes and model cards — build.** Reason codes are a product requirement (analysts cannot action a score without one) and the model card is a regulatory artifact. Both are written in the ML team's language (features, thresholds, categories). No vendor substitutes for owning this.

**Security (IAM, secrets, threat modeling) — delegate to security.** The security team runs the IAM substrate, the secret manager, and the review process. Delegation by expertise and by regulatory requirement (security review sign-off cannot be done by the ML team). Our side of the interface: use the sanctioned IAM roles, pull secrets via the sanctioned client, and attend the threat modeling session. Security owns the threat model's completeness; we own documenting the ML-specific threats (adversarial inputs, data poisoning) and mapping them against OWASP ML Top 10 and NIST AI RMF categories.

### 2.4 Interface contracts, per delegation and buy

For every "buy" and every "delegate," the memo commits an interface that is **ours**:

| Decision | Interface contract (owned by us) | SLA owned by them |
|---|---|---|
| Buy feature store | Feature definition DSL + schema in `features/` | <5ms p99 read, 99.95% availability |
| Delegate training pipeline | `train()` script contract + log schema | Pipeline runs/day; registry uptime |
| Buy drift monitoring | Signal export (Prometheus endpoint, drift signals also exported) | Alert routing; dashboard uptime |
| Delegate drift feature-join | Serving log schema + label schema | Warehouse ETL freshness |
| Delegate security | List of ML-specific threats + OWASP ML Top 10 mappings | IAM review; secret rotation |

Interface contracts are the single most important artifact for keeping optionality. If the feature-store vendor becomes unacceptable in year 2, the migration is "re-implement the DSL against the new vendor" — not "redesign the whole feature architecture."

### 2.5 Risk register

- **Vendor lock on feature store.** Mitigation: feature definitions in a vendor-neutral DSL, stored in our repo. Rehearse quarterly a migration to the OSS equivalent, on a small shard.
- **MLOps infra queue time.** Delegating the training pipeline means we are sometimes behind another team's deploy. Mitigation: documented SLA with MLOps infra; a 1-week break-glass path to run the pipeline on our own cluster.
- **Data eng freshness slips.** The drift feature-join is only as fresh as the warehouse. Mitigation: freshness alert with ML as a secondary on-call, so a slow warehouse pages someone before the drift dashboard quietly goes stale.
- **Security review as a bottleneck.** The security team can block launch. Mitigation: start threat modeling in week 2 of the project, not week 20. Attach a draft ML risk register (OWASP ML Top 10 + NIST AI RMF categories) to the first security meeting.
- **Build budget overrun on labeling tool.** The OSS core may need more customization than estimated. Mitigation: a concrete week-6 checkpoint with a go/no-go on whether to pivot to a different OSS core or accept a larger build.

### 2.6 Exit plans

Every "buy" decision has a committed exit plan. "What would we do if this vendor doubled their price or shut down in 180 days?"

- **Feature store exit.** DSL + schema is ours. The exit is: stand up the OSS equivalent on our k8s; port the DSL compiler to its API; backfill history from the warehouse. Estimated 2–3 eng-months. Rehearsed on a 1% shard every 6 months.
- **Drift monitoring exit.** All signals land on our Prometheus/Grafana via a parallel exporter. The exit is: re-point the alert rules; retire the vendor UI. Estimated 2–3 eng-weeks.
- **Managed services from MLOps infra.** Not strictly a vendor — but a cross-team dependency with the same shape. The exit is: run the tracking server and registry in our own namespace. Estimated 2 eng-months.

A buy-without-exit-plan is not a buy; it is a mortgage.

### 2.7 Non-goals

- We are not building our own model registry. The MLOps infra one is the standard; a second registry fragments provenance.
- We are not building the warehouse. Data engineering owns the warehouse, and reproducing it on our side would be an incident factory.
- We are not building a general-purpose vector database. If a v2 model requires embedding lookup, we re-open build/buy at that time.

### 2.8 Signoff

Signed off by: ML lead (us), MLOps infra tech lead, data engineering tech lead, security engineering lead, platform/SRE tech lead. Each signoff is specific to the interface *their* track owns. The memo is committed to the repo; subsequent changes to any row require a PR with reviewers from the signatory list.

## 3. Validation steps

A memo is validated by review and by rehearsal. Run these in order.

1. **Red-team the TCOs.** Have a peer (ideally from the MLOps infra or data eng track) attack every number in the §2.2 table. Any TCO estimate that cannot be defended within a factor of 2 is wrong; refine or replace with a range.
2. **Test the critical-path column.** For every component marked "critical path = Yes," ask: if the vendor or peer track goes down for 7 days, does the fraud system stop making correct decisions? If "no," it is not actually critical path and the decision should be reconsidered.
3. **Interface stress-test.** For each delegated component, write the exact schema and SLA. A reviewer from the other track must be able to implement to that interface without talking to us. If they cannot, the interface is leaky and the delegation will leak back onto the ML team.
4. **Exit-plan rehearsal.** Pick one "buy" row and actually run the exit plan on staging. For the feature store, that means standing up the OSS equivalent on a 1% shard and verifying the DSL compiles against its API. Exits that have never been rehearsed are decorative.
5. **Signoff dry run.** Present the memo to each signatory with a 10-minute slot; collect pushback; revise. A memo whose signatories first see it in the final review is more likely to produce a "we need another meeting" than a signature.
6. **Decision reversal.** For each "build" decision, write a one-paragraph description of the conditions under which the decision would reverse to "buy," and vice versa. Decisions that cannot be reversed under any input are not decisions; they are beliefs.
7. **Security threat register.** Confirm that the delegated-to-security row produces an actual threat register covering OWASP ML Top 10 categories relevant to the system (`ML01`, `ML02`, `ML06`, `ML08` for a fraud scorer) and the NIST AI RMF `MAP` / `GOVERN` functions. Missing categories are a signal that security has not yet been engaged.

## 4. Rubric / review checklist

Weighted, `/24` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Per-component decomposition | 3 | ≥ 5 components, each decided independently |
| 2 | TCO with named inputs | 3 | 3-year horizon; build and buy costs in the same unit; estimates footnoted |
| 3 | Critical-path flag per component | 2 | Each row marked critical / not; critical-path decisions explicitly justified |
| 4 | Differentiator flag per component | 2 | Each row marked differentiating / not; aligns with build decisions |
| 5 | Delegation by expertise | 3 | Each delegated row names the peer track and the expertise it brings, not just availability |
| 6 | Interface contracts | 4 | Every "buy" and "delegate" has a schema + SLA + owner of both sides |
| 7 | Exit plans | 3 | Every "buy" has a named, estimated, and rehearsable exit plan |
| 8 | Risk register | 2 | ≥ 3 risks tied to this memo's decisions; each has a mitigation and an owner |
| 9 | Non-goals | 1 | ≥ 2 explicit non-goals; each one plausible as a future pivot |
| 10 | Signoff | 1 | Named signatories per interface; memo committed to repo |

Items 2, 6, and 7 are mandatory: a memo without TCO arithmetic, without interface contracts, or without exit plans cannot score above 14. Those three are the ingredients that convert "build vs. buy" from a hallway debate into a reviewed, defensible decision.

## 5. Common mistakes

- **One decision for the whole system.** "We're a build shop" or "we buy everything." The real answer is per-component, and good systems mix the two.
- **Sticker-price comparisons.** "The vendor costs $75k/yr and the team is $400k/yr, so we build." Missing: integration cost, operational cost of the built thing, risk-adjusted probability of delivery. Compare 3-year TCO with integration included.
- **Delegating by capacity.** "The MLOps team has bandwidth, so let them own it." When bandwidth changes, the ownership question re-opens. Delegate by expertise; if expertise is not there, do not delegate at all.
- **No interface contract.** The delegated or bought component arrives, and the ML team discovers mid-integration that the schema is close to what they needed but not the same. Interface first, not last.
- **No exit plan.** A buy decision made in month 1 looks great; in month 18, the vendor raises prices 3x and migration is an unbudgeted project. Exits are rehearsed; otherwise they are a claim without evidence.
- **Confusing "critical path" with "important."** Critical path is "the system cannot function without it on the hour." Important is "we care about this a lot." The two often overlap but not always, and the decisions differ. Critical path tolerates higher cost for reliability; important tolerates higher cost for differentiation.
- **Signing off without the signatories.** The memo is committed by the ML lead alone; the other tracks are not consulted; three months later, security re-opens the IAM assumption and derails launch. Every interface's other-side owner signs off at memo time.
- **Building because "it's a fun engineering problem."** The memo should pass the "if the problem were boring, would we still build?" check. If the answer is no, buy or delegate.
- **Buying because "it's been done before."** Some boring components are also differentiating when integrated well (reason codes on fraud holds, for example). Being boring in isolation does not make a component non-differentiating in context.
- **No risk register.** The memo reads as all upside. The year-2 pain comes from the risks that were not named in year 0. Every decision is accompanied by what could go wrong with it.

## 6. References

- NIST AI Risk Management Framework 1.0 — `GOVERN 1.1` (establishing roles, responsibilities, and lines of communication) and `GOVERN 3.2` (mechanisms for human oversight) are the framework-level grounding for the delegation-by-expertise pattern and the per-interface signoff model used in §2.4 and §2.8. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — the §2.5 risk register and §3 step 7 security review are structured around OWASP's category list; `ML06:2023 AI Supply Chain Attacks` specifically motivates the exit-plan practice in §2.6 (a vendor is a supply chain dependency). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — referenced by the security delegation in §2.3 as the threat model's upper layer; `AML.T0010 ML Supply Chain Compromise` is the attack class that every "buy" decision increases exposure to, and that every exit plan reduces the blast radius of. <https://atlas.mitre.org/>
