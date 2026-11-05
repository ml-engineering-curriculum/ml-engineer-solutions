# exercise-04 — Fairness and Robustness Slices · SOLUTION

Paired exercise: `lessons/mod-107-model-evaluation-validation/exercises/exercise-04-fairness-and-robustness-slices.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to extend the evaluation harness with two kinds of per-slice measurement:

- **Fairness slices** — performance on subpopulations defined by protected attributes or their proxies, compared against a reference group with explicit, named disparity metrics.
- **Robustness slices** — performance under realistic distribution shifts (input perturbations, subpopulation shifts, time shifts) that the model will see in deployment but that the random test split hides.

The two share mechanics (slice, score, CI, gate) but answer different questions: fairness asks "is the error concentrated on groups the system harms?" and robustness asks "does the error grow when the world moves off the training distribution?" Both are first-class ship gates in this solution — a candidate that regresses on either is blocked.

Design priorities, in order:

1. **Slices declared before scoring.** Slice definitions — including the reference group for fairness — are checked into the repo *before* the candidate model runs. Choosing the slice after seeing the numbers is a researcher-degrees-of-freedom leak.
2. **Reference group must be named.** Disparity ratios are *per-reference*; "women vs. unspecified" is different from "women vs. men." The rubric from exercise-01 names the reference group, this harness consumes it.
3. **Disparity has three forms, and the choice is a policy decision.** `demographic_parity_difference` (equal selection rate), `equal_opportunity_difference` (equal TPR), and `equalized_odds_difference` (equal TPR and FPR) encode different fairness axioms. The one the team picks is documented, not inferred from code.
4. **Robustness checks three kinds of shift separately.** Covariate shift (feature distribution moves, labels do not), subpopulation shift (segment mix changes), label shift (base rate changes). A single "robust / not robust" scalar hides which shift is biting.
5. **Gates are two-stage: alert + block.** Soft threshold produces a reviewer alert; hard threshold blocks the release. Both are declared in the rubric from exercise-01 and enforced in CI.
6. **Reports include sample size, CI, and reference-group comparison for every slice.** A disparity of 0.08 on n=40 is noise; on n=400 000 it is a finding. The report must show both.

This design maps to NIST AI RMF `MEASURE 2.11` (performance across subpopulations), `MEASURE 2.6` (safety in realistic contexts), and `MANAGE 2.1` (documented limitations). The OWASP ML Top 10 `ML04:2023 Membership Inference Attack` and `ML05:2023 Model Theft` surfaces interact: a model that overperforms on a membership-dense slice is more exposed to inference attacks there. MITRE ATLAS `AML.T0043 Craft Adversarial Data` motivates the robustness-under-perturbation checks.

| Choice | Alternative | Why this one |
|---|---|---|
| Slice definitions in `slices.yaml` in the repo | Defined in the notebook that generates the report | Config in the repo is diffable, review-able, and versioned alongside the model |
| Explicit `reference_group` per protected attribute | Max-over-all pairwise disparities | Pairwise maxima inflate with the number of groups; the reference encodes an intentional baseline |
| Three disparity metrics reported side-by-side | A single "fairness score" | Collapsing to one number hides the trade-off the team actually made |
| Robustness via held-out time slice + perturbation suite + subpopulation reweighting | A single OOD benchmark | Each method isolates a different shift; a benchmark score alone cannot distinguish them |
| Bootstrap CIs grouped by entity id, not row | Row-level bootstrap | Row-level CIs underestimate variance when the same user contributes many rows |
| Hard gates in CI | "Review in Slack" | A policy that requires human ship-approval on every deploy will be bypassed under pressure |

## 2. Worked implementation

### 2.1 Project layout

```
fairness/
├── __init__.py
├── config.py           # SliceConfig, DisparityConfig loaded from slices.yaml
├── slices.yaml         # declarative slice + reference + gate config
├── metrics.py          # per-slice metrics, demographic parity, equal opportunity, equalized odds
├── robustness.py       # perturbation suite + subpopulation reweight + time-slice eval
├── gates.py            # soft + hard thresholds, CI check
├── report.py           # renders fairness.md and robustness.md
└── tests/
    ├── test_reference_disparity.py
    └── test_gates_block_on_violation.py
```

### 2.2 `slices.yaml` — the policy, checked into the repo

```yaml
task: payments_fraud_v3

fairness:
  attributes:
    - name: issuer_bank
      slices:
        - value: BankA
        - value: BankB
        - value: BankC
        - value: BankQ   # the known worst-slice from prior error analysis
      reference_group: BankA
      min_n: 5000
      metric: equal_opportunity_difference
      gate:
        soft_abs: 0.05   # review
        hard_abs: 0.10   # block
    - name: country_tier
      slices:
        - value: tier_1
        - value: tier_2
        - value: tier_3
      reference_group: tier_1
      min_n: 5000
      metric: equalized_odds_difference
      gate:
        soft_abs: 0.05
        hard_abs: 0.10

robustness:
  perturbations:
    - name: amount_noise_1pct
      kind: numeric_noise
      column: amount_usd
      sigma_pct: 1.0
    - name: missing_last_login
      kind: feature_drop
      column: last_login_days
    - name: country_shift
      kind: resample
      distribution: {tier_1: 0.5, tier_2: 0.3, tier_3: 0.2}  # eval-set distribution is 0.7/0.2/0.1
  time_slices:
    - name: last_week
      window_days: 7
    - name: launch_week
      window_days: 7
      offset_days: 180
  gate:
    auc_drop_soft: 0.02
    auc_drop_hard: 0.05
```

Keeping the config declarative means the policy — which subpopulations, what reference, what counts as a block — is reviewable without reading Python. When the policy changes, the diff shows what changed.

### 2.3 `config.py` — parse the policy

```python
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Gate:
    soft_abs: float
    hard_abs: float


@dataclass(frozen=True)
class FairnessAttribute:
    name: str
    slice_values: list[str]
    reference_group: str
    min_n: int
    metric: str          # "demographic_parity_difference" | "equal_opportunity_difference" | "equalized_odds_difference"
    gate: Gate


@dataclass(frozen=True)
class RobustnessGate:
    auc_drop_soft: float
    auc_drop_hard: float


def load(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)
```

### 2.4 `metrics.py` — three disparity metrics, same shape

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class SlicePerformance:
    slice_value: str
    n: int
    selection_rate: float
    tpr: float
    fpr: float
    auc: float


def _rates(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> tuple[float, float, float, float]:
    if len(np.unique(y_true)) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    selection_rate = float(np.mean(y_pred))
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, proba))
    return selection_rate, tpr, fpr, auc


def slice_performance(df: pd.DataFrame, attr: str, value: str,
                      y_col: str, proba_col: str, threshold: float) -> SlicePerformance | None:
    sub = df[df[attr] == value]
    if len(sub) == 0:
        return None
    y = sub[y_col].values.astype(int)
    p = sub[proba_col].values
    y_pred = (p >= threshold).astype(int)
    sel, tpr, fpr, auc = _rates(y, y_pred, p)
    return SlicePerformance(slice_value=value, n=len(sub), selection_rate=sel,
                            tpr=tpr, fpr=fpr, auc=auc)


def demographic_parity_difference(groups: list[SlicePerformance], reference: str) -> dict[str, float]:
    ref = next(g for g in groups if g.slice_value == reference)
    return {g.slice_value: g.selection_rate - ref.selection_rate
            for g in groups if g.slice_value != reference}


def equal_opportunity_difference(groups: list[SlicePerformance], reference: str) -> dict[str, float]:
    ref = next(g for g in groups if g.slice_value == reference)
    return {g.slice_value: g.tpr - ref.tpr
            for g in groups if g.slice_value != reference}


def equalized_odds_difference(groups: list[SlicePerformance], reference: str) -> dict[str, dict[str, float]]:
    ref = next(g for g in groups if g.slice_value == reference)
    return {
        g.slice_value: {"tpr_diff": g.tpr - ref.tpr, "fpr_diff": g.fpr - ref.fpr}
        for g in groups if g.slice_value != reference
    }
```

Three functions, one shape. The caller picks the one that matches `metric` in the config. The metric choice is a *policy* — demographic parity makes sense when outcomes are an allocation (loan approvals); equal opportunity makes sense when the goal is parity of error on the positive class (medical diagnosis). The code does not pick for you.

### 2.5 `robustness.py` — three kinds of shift, three measurements

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class RobustnessResult:
    name: str
    baseline_auc: float
    shifted_auc: float
    delta_auc: float


def numeric_noise(df: pd.DataFrame, column: str, sigma_pct: float,
                  rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    sigma = out[column].std() * sigma_pct / 100.0
    out[column] = out[column] + rng.normal(0, sigma, size=len(out))
    return out


def feature_drop(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.NA
    return out


def resample_to(df: pd.DataFrame, by: str, target: dict[str, float],
                rng: np.random.Generator) -> pd.DataFrame:
    """Subpopulation shift: resample df so df[by] follows `target` distribution."""
    pieces = []
    n_target = len(df)
    for value, frac in target.items():
        pool = df[df[by] == value]
        if len(pool) == 0:
            continue
        pieces.append(pool.sample(int(n_target * frac), replace=True, random_state=int(rng.integers(0, 2**31))))
    return pd.concat(pieces).sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


def evaluate_perturbation(df: pd.DataFrame, predict, y_col: str, name: str,
                          shift_fn) -> RobustnessResult:
    base = roc_auc_score(df[y_col], predict(df))
    shifted = shift_fn(df)
    after = roc_auc_score(shifted[y_col], predict(shifted))
    return RobustnessResult(name=name, baseline_auc=float(base),
                            shifted_auc=float(after), delta_auc=float(after - base))
```

- `numeric_noise` simulates measurement error that was absent in training (sensor recalibration, upstream rounding).
- `feature_drop` simulates a feature pipeline outage. If AUC falls off a cliff when one feature is missing, the model is dangerously brittle to any upstream incident.
- `resample_to` is the canonical subpopulation-shift test: hold the rows the same, change the mixture. A model that depended on the eval-set mixture will degrade; a model that generalises will not.
- Time-slice evaluation (defined by `time_slices` in YAML) is covariate+label shift combined — the honest answer to "does last week look like the training window?"

### 2.6 `gates.py` — two-stage enforcement

```python
from dataclasses import dataclass


@dataclass
class GateDecision:
    pass_: bool
    violations: list[str]
    warnings: list[str]


def check_fairness(disparities: dict[str, float], soft: float, hard: float) -> GateDecision:
    violations, warnings = [], []
    for group, diff in disparities.items():
        if abs(diff) >= hard:
            violations.append(f"{group}: |Δ| = {abs(diff):.3f} ≥ hard {hard}")
        elif abs(diff) >= soft:
            warnings.append(f"{group}: |Δ| = {abs(diff):.3f} ≥ soft {soft}")
    return GateDecision(pass_=not violations, violations=violations, warnings=warnings)


def check_robustness(results, soft: float, hard: float) -> GateDecision:
    violations, warnings = [], []
    for r in results:
        drop = -r.delta_auc  # delta is new - old; a drop is negative delta
        if drop >= hard:
            violations.append(f"{r.name}: AUC drop {drop:.3f} ≥ hard {hard}")
        elif drop >= soft:
            warnings.append(f"{r.name}: AUC drop {drop:.3f} ≥ soft {soft}")
    return GateDecision(pass_=not violations, violations=violations, warnings=warnings)
```

Exit-coded gates — a non-empty `violations` list fails CI. The warnings go to a review channel but do not block; this keeps "borderline regression" out of the ship-blocker category without hiding it.

### 2.7 `tests/test_reference_disparity.py`

```python
import numpy as np
import pandas as pd

from fairness.metrics import slice_performance, equal_opportunity_difference


def test_zero_disparity_when_groups_identical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": ["a"] * 500 + ["b"] * 500,
        "y": rng.integers(0, 2, size=1000),
        "p": rng.random(1000),
    })
    perfs = [
        slice_performance(df, "group", "a", "y", "p", threshold=0.5),
        slice_performance(df, "group", "b", "y", "p", threshold=0.5),
    ]
    disp = equal_opportunity_difference(perfs, reference="a")
    assert abs(disp["b"]) < 0.08  # noise band on n=500


def test_disparity_grows_with_biased_model():
    rng = np.random.default_rng(1)
    y_a = rng.integers(0, 2, size=1000)
    y_b = rng.integers(0, 2, size=1000)
    # Model is accurate on group a, near-random on group b -> TPR gap
    p_a = np.where(y_a == 1, 0.9, 0.1) + rng.normal(0, 0.05, size=1000)
    p_b = rng.random(1000)
    df = pd.DataFrame({
        "group": ["a"] * 1000 + ["b"] * 1000,
        "y": np.concatenate([y_a, y_b]),
        "p": np.concatenate([p_a, p_b]),
    })
    perfs = [
        slice_performance(df, "group", "a", "y", "p", threshold=0.5),
        slice_performance(df, "group", "b", "y", "p", threshold=0.5),
    ]
    disp = equal_opportunity_difference(perfs, reference="a")
    assert disp["b"] < -0.2  # group b has substantially lower TPR than a
```

The point of the second test is to confirm the sign and magnitude of the disparity when the model *is* biased — a harness that reports 0 everywhere is worse than useless.

### 2.8 Example `fairness.md` (filled)

```markdown
# Fairness Evaluation — Payments Fraud Filter v3-candidate

Reference groups: issuer_bank=BankA, country_tier=tier_1
Policy: equal_opportunity_difference (fairness) with soft 0.05 / hard 0.10

| Attribute | Slice | n | TPR | TPR − ref | 95 % CI on Δ | Status |
|---|---|---:|---:|---:|---|---|
| issuer_bank | BankA (ref) | 142 010 | 0.77 | 0.00 | — | ref |
| issuer_bank | BankB | 118 553 | 0.74 | −0.03 | [−0.04, −0.02] | OK |
| issuer_bank | BankC | 86 221 | 0.72 | −0.05 | [−0.06, −0.04] | SOFT |
| issuer_bank | BankQ | 14 212 | 0.61 | −0.16 | [−0.19, −0.13] | HARD |
| country_tier | tier_1 (ref) | 220 118 | 0.78 | 0.00 | — | ref |
| country_tier | tier_2 | 94 006 | 0.75 | −0.03 | [−0.04, −0.02] | OK |
| country_tier | tier_3 | 44 772 | 0.69 | −0.09 | [−0.11, −0.07] | SOFT |

Gate: **FAIL** — BankQ exceeds hard threshold. Candidate blocked.
Follow-up: open issue "BankQ TPR regression" and link to cluster A in the error-analysis doc.
```

And `robustness.md`:

```markdown
# Robustness Evaluation — Payments Fraud Filter v3-candidate

Policy: AUC drop soft 0.02 / hard 0.05

| Shift | Baseline AUC | Shifted AUC | ΔAUC | Status |
|---|---:|---:|---:|---|
| amount_noise_1pct | 0.871 | 0.869 | −0.002 | OK |
| missing_last_login | 0.871 | 0.843 | −0.028 | SOFT |
| country_shift (eval→tier-heavy) | 0.871 | 0.818 | −0.053 | HARD |
| time: last_week | 0.871 | 0.862 | −0.009 | OK |
| time: launch_week (6 months ago) | 0.871 | 0.831 | −0.040 | SOFT |

Gate: **FAIL** — country_shift exceeds hard threshold. Model is over-fit to the
eval-set country mix. Options: reweight training by `country_tier`, add country-specific
features, or restrict deployment scope to tier_1.
```

## 3. Validation steps

Run in order.

1. **Static check** — `python -c "from fairness.metrics import slice_performance; from fairness.robustness import evaluate_perturbation"`.
2. **Unit tests** — `pytest fairness/tests -q`. Both reference-disparity tests must pass.
3. **Config lint** — every attribute in `slices.yaml` has a declared `reference_group`, and the reference appears in `slices`. A missing reference is a config bug, not a modelling bug.
4. **Min-n enforcement** — for each slice, confirm that `n < min_n` causes the slice to be reported as "insufficient data" rather than producing a noisy disparity number that triggers a gate.
5. **Gate dry-run** — run the gate on a deliberately-biased synthetic dataset (e.g., group B with half the training signal). Assert the hard gate fires.
6. **Robustness smoke run** — run each perturbation on the candidate. Any perturbation with |ΔAUC| > 0.10 deserves a write-up even if it did not trip the gate; brittleness is a finding.
7. **Time-slice backtest** — pick a time slice `N` days earlier than the training window and re-score. If AUC collapses on older data, the model is over-fit to recent seasonality.
8. **Grouped bootstrap** — when computing CIs, resample at the entity-id level, not the row level. Verify the CI widths are wider than naive row-level CIs on a dataset where entities contribute many rows.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Slice policy checked into repo | 2 | `slices.yaml` (or equivalent) in git; diff-reviewable; versioned with the model |
| 2 | Reference group named explicitly | 2 | Every protected attribute has a declared reference; disparity is always reference-relative |
| 3 | Disparity metric matches the decision shape | 2 | Allocation → demographic parity; error on positives → equal opportunity; both → equalized odds; choice justified |
| 4 | Min sample size per slice | 1 | Slices below `min_n` are skipped, not reported with a bogus number |
| 5 | CIs on every disparity | 2 | Bootstrap CIs (grouped by entity id when applicable) on each slice and each disparity |
| 6 | Three robustness shift types covered | 3 | Covariate perturbation + subpopulation reweight + time-slice all run with results |
| 7 | Two-stage gates (soft + hard) | 2 | Policy has both; CI blocks on hard, alerts on soft |
| 8 | Reports as reviewable markdown | 2 | `fairness.md` and `robustness.md` rendered artifacts; not notebook cells |
| 9 | Self-tests for harness correctness | 2 | Zero-disparity-on-identical-groups test and biased-model-detection test both green |
| 10 | Follow-up hypotheses tracked | 2 | Each gate failure or soft alert has an owner and a ticket; analysis changes the next iteration |

Items 2, 3, and 6 are mandatory. A submission missing any of them cannot score above 11 / 20 regardless of the rest. A fairness report without a reference group is not a fairness report; a disparity metric chosen "by default" encodes a policy nobody signed off on; a robustness report that checks only one shift type cannot distinguish the three failure modes the exercise exists to separate.

## 5. Common mistakes

- **Fairness metric chosen "because scikit-learn / fairlearn defaults to it."** Demographic parity, equal opportunity, and equalized odds encode different fairness definitions and can disagree. The choice is a policy decision; document it.
- **No reference group named.** Max-over-pairs disparities (e.g., worst gap across all pairs) can inflate with the number of groups and lose interpretability. Fix a reference; justify it.
- **Protected attribute used as a feature, unflagged.** NIST AI RMF `MAP 1.1` / `MEASURE 2.11` expects explicit documentation of whether protected attributes are inputs. If they are, that is a design decision, not a line in a CSV.
- **"Our test set is representative" when it is not.** The eval-set subpopulation mix is usually closer to the training distribution than the deployment distribution. Subpopulation reweighting (resample eval to the expected deployment mix) is the direct test.
- **Random row-level bootstrap on grouped data.** If each user contributes 40 rows, row-level resampling underestimates variance — the bootstrap picks correlated samples. Group-level resampling is the correct default for user-level tasks.
- **One robustness benchmark.** A single "robustness score" collapses three failure modes. Run covariate, subpopulation, and label shifts separately and report the deltas.
- **Soft-only gates.** A policy that never blocks will not be respected. Hard gates on the rubric's declared thresholds are what make "we care about fairness" a verifiable claim.
- **Slice list grown from error analysis.** Slices defined after seeing the errors become part of the error story; they are not independent evidence. Keep the fairness slice list in the rubric (exercise-01) and only *add* new slices as separate, dated additions.
- **Ignoring intersectional slices.** A model can pass per-attribute fairness and fail on the intersection (women + older + specific product). If the data supports it, test at least the highest-prior intersections.
- **Thresholds set after the test runs.** Setting the gate to "whatever the candidate just scored minus a bit" nullifies the gate. Thresholds come from the rubric (exercise-01) and precede the candidate.
- **Adversarial perturbation framed as robustness.** Perturbations in this exercise simulate *benign* operational shift (sensor noise, missing feature, mix change). Adversarial robustness (gradient-crafted inputs) belongs in a security evaluation (OWASP ML Top 10 `ML01:2023`, MITRE ATLAS `AML.T0043`) and is a separate harness.

## 6. References

- NIST AI Risk Management Framework 1.0 — `MEASURE 2.11` (performance measured across identified subpopulations) and `MEASURE 2.6` (safety under realistic conditions, including distribution shifts). <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF 1.0 — `MANAGE 2.1` on documenting known limitations and failure modes surfaced by slice and robustness evaluation. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML04:2023 Membership Inference Attack` on why slices with extreme per-slice performance deserve privacy scrutiny; `ML05:2023 Model Theft` on the operational risk of over-fit subpopulation performance. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0043 Craft Adversarial Data` describes the threat model that motivates separating benign robustness checks from adversarial robustness checks; `AML.T0031 Erode ML Model Integrity` for the longer arc of silent slice regression. <https://atlas.mitre.org/>
- scikit-learn — metrics used here (`confusion_matrix`, `roc_auc_score`) and the bootstrap pattern documented in the user guide. <https://scikit-learn.org/stable/modules/model_evaluation.html>
