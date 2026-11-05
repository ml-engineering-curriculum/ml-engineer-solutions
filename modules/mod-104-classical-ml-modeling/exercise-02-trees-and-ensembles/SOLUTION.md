# exercise-02 — Trees And Ensembles · SOLUTION

Paired exercise: `lessons/mod-104-classical-ml-modeling/exercises/exercise-02-trees-and-ensembles.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to model a tabular dataset with a decision tree, then show that an ensemble reduces error in a way that is explainable in terms of bias and variance. The reference solution works through three models in increasing complexity and ties each result back to what the ensemble is actually doing:

1. **Single `DecisionTreeClassifier`** — fit to a shallow depth, then unbounded. This exposes how a single tree overfits when given enough depth.
2. **`RandomForestClassifier`** — bagging over trees, each one grown on a bootstrap sample with a random feature subset. Averaging reduces variance; bias is roughly unchanged.
3. **`GradientBoostingClassifier` / `HistGradientBoostingClassifier`** — sequentially fit shallow trees on residuals. Reduces bias; variance must be controlled with regularisation.

The design decisions that matter:

- **Compare on the same splits.** All three models are evaluated with the same `StratifiedKFold`, so the comparison is about model, not luck.
- **OOB score on the forest.** `oob_score=True` gives a near-free validation estimate that lets us sanity-check the K-fold number.
- **Depth as the interpretability lever, not accuracy.** The single tree is reported *also* at `max_depth=3` for readability, even if the headline tree is deeper.
- **Feature importance is reported in two ways.** MDI (built-in `.feature_importances_`) *and* permutation importance. MDI is biased toward high-cardinality features; permutation importance on a held-out set is the honest one.

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| Stratified K-fold | Plain K-fold | Target is imbalanced; stratification keeps each fold's class ratio near the global one |
| `n_estimators=500` on the forest | Default `100` | Variance of forest predictions decays as `1/n`; 500 is the first point where OOB stops improving on typical tabular data |
| `HistGradientBoostingClassifier` over plain `GradientBoostingClassifier` | Classic `GradientBoostingClassifier` | Histogram-based boosting is 10–100× faster, handles missing values natively, and is the scikit-learn-blessed path forward |
| Permutation importance on *test* set, not train | Importance on train | Importance on train reflects fit; importance on test reflects generalisation — which is what a reviewer actually wants |
| Fixed `random_state` everywhere | None | Reruns are deterministic; "flaky importance" bugs are almost always missing seeds |
| Report both MDI and permutation importance | Only one | MDI is biased for high-cardinality features; showing both exposes the bias to the reviewer |

## 2. Worked implementation

### 2.1 Project layout

```
trees_and_ensembles/
├── __init__.py
├── data.py
├── models.py          # three model factories
├── evaluate.py        # shared CV + metrics
├── importance.py      # MDI + permutation importance
└── run.py
```

### 2.2 Shared evaluation

```python
# trees_and_ensembles/evaluate.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss


@dataclass
class ModelResult:
    name: str
    cv_auc_mean: float
    cv_auc_std: float
    test_auc: float
    test_logloss: float
    estimator: BaseEstimator


def stratified_cv(random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def cv_auc(est: BaseEstimator, X, y) -> tuple[float, float]:
    cv = stratified_cv()
    scores = cross_val_score(est, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())
```

One `stratified_cv()` factory, one seed, used for every model. The entire comparison hinges on the folds being identical across models.

### 2.3 Three model factories

```python
# trees_and_ensembles/models.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def single_tree(max_depth: int | None = None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )


def random_forest(n_estimators: int = 500) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )


def hist_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
```

Key points, each tied to a predictable failure mode:

- **`min_samples_leaf`** is the single most effective regulariser for trees. Default (`1`) lets the tree memorise outliers; raising it to 5–20 removes most of the pathological overfit.
- **`max_features="sqrt"`** is what makes a Random Forest a *random* forest rather than bagged trees. Without it, every tree is too correlated and the variance reduction collapses.
- **`class_weight="balanced"`** instead of resampling. On an imbalanced target, the split criterion is otherwise dominated by the majority class.
- **`early_stopping=True`** on the booster. Without it, you have no principled stopping rule and the model silently overfits past the validation minimum.

### 2.4 Fit + compare on the same folds

```python
# trees_and_ensembles/run.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

from .evaluate import cv_auc, ModelResult
from .models import single_tree, random_forest, hist_gbm
from .importance import both_importances


def main(data_path: str, target: str, out: str) -> None:
    df = pd.read_parquet(data_path)
    y = df[target].astype("int64")
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "tree_depth3":  single_tree(max_depth=3),
        "tree_unbounded": single_tree(max_depth=None),
        "random_forest": random_forest(),
        "hist_gbm": hist_gbm(),
    }

    results: list[ModelResult] = []
    for name, est in candidates.items():
        mean_, std_ = cv_auc(est, X_train, y_train)
        est.fit(X_train, y_train)
        y_proba = est.predict_proba(X_test)[:, 1]
        results.append(ModelResult(
            name=name,
            cv_auc_mean=mean_,
            cv_auc_std=std_,
            test_auc=float(roc_auc_score(y_test, y_proba)),
            test_logloss=float(log_loss(y_test, y_proba)),
            estimator=est,
        ))

    winner = max(results, key=lambda r: r.cv_auc_mean)
    importances = both_importances(winner.estimator, X_test, y_test)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "winner": winner.name,
        "comparison": [
            {"name": r.name, "cv_auc_mean": r.cv_auc_mean, "cv_auc_std": r.cv_auc_std,
             "test_auc": r.test_auc, "test_logloss": r.test_logloss}
            for r in results
        ],
        "oob_score": getattr(winner.estimator, "oob_score_", None),
        "importance_top10_mdi": importances["mdi"].head(10).to_dict(),
        "importance_top10_permutation": importances["permutation"].head(10).to_dict(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="artifacts/exercise-02")
    args = p.parse_args()
    main(args.data, args.target, args.out)
```

### 2.5 Importance — both views

```python
# trees_and_ensembles/importance.py
import pandas as pd
from sklearn.inspection import permutation_importance


def both_importances(est, X_test, y_test) -> dict[str, pd.Series]:
    mdi = pd.Series(getattr(est, "feature_importances_", []), index=X_test.columns)
    mdi = mdi.sort_values(ascending=False)

    perm = permutation_importance(
        est, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1, scoring="roc_auc",
    )
    perm_s = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    return {"mdi": mdi, "permutation": perm_s}
```

Why report both:

- **MDI** (mean decrease in impurity) is cheap — scikit-learn stores it from training — but it is **biased toward high-cardinality features**. A column with 10,000 distinct values can look important purely because it creates many split opportunities, not because it predicts anything.
- **Permutation importance on the test set** is honest: shuffle the column, see how much AUC drops. A non-informative feature has a drop near zero regardless of cardinality.

If the two rankings disagree sharply at the top, the reviewer should suspect the dataset, not the model.

### 2.6 What the bias–variance story should look like

Expected on a typical tabular benchmark (e.g., adult income, credit default):

| Model | Train AUC | Test AUC | Bias? | Variance? |
|---|---|---|---|---|
| `tree_depth3` | ~0.80 | ~0.79 | High | Low |
| `tree_unbounded` | ~1.00 | ~0.78 | Low | Very high |
| `random_forest` | ~0.99 | ~0.86 | Low | Low — averaging over 500 trees cancels the variance |
| `hist_gbm` | ~0.92 | ~0.88 | Lower still (sequential fit on residuals) | Controlled by `learning_rate` and `early_stopping` |

The pattern a learner must be able to narrate:

- The depth-3 tree is **underfit** (train and test both low).
- The unbounded tree is **overfit** (train perfect, test worse than the shallow tree).
- The forest's test AUC jumps *without changing the single-tree bias*; it only cancels the variance of the single tree.
- The booster improves further by lowering bias through sequential fitting.

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **Import check** — `python -c "import trees_and_ensembles.run"`.
2. **Unit: same folds for all models** — `stratified_cv()` called with the same seed returns the same split indices; assert `list(cv.split(X, y))` is byte-identical across calls.
3. **Unit: forest OOB vs CV** — on a synthetic dataset, `oob_score_` should be within ±0.02 of the mean CV AUC. If they diverge, the forest is too small (bump `n_estimators`).
4. **Unit: tree overfit signal** — fit `tree_unbounded` on a dataset with duplicate rows; `train_score - test_score` must exceed 0.1. Confirms depth control is actually a lever.
5. **Unit: boosted early stopping fires** — fit `hist_gbm()` on a tiny dataset; assert `n_iter_ < max_iter`. If early stopping is silently disabled, you have no principled stopping rule.
6. **Determinism** — run the orchestrator twice; `report.json` is byte-identical.
7. **MDI vs permutation sanity check** — inject a column of `np.random.randn(len(X))` into `X_train`; MDI may rank it mid-pack (cardinality bias), permutation importance must put it near zero. This is the test that says "yes, your permutation importance is wired up."

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Three families compared (tree / forest / boosting) | 3 | All three fitted; same splits; same metric |
| 2 | Stratified splits | 2 | Target imbalance acknowledged; `StratifiedKFold` or `stratify=` used |
| 3 | Overfit demonstrated on single tree | 2 | Train/test gap reported on an unbounded tree; shallow-tree comparison present |
| 4 | Variance reduction demonstrated on forest | 2 | Forest test score strictly beats the best single tree; OOB reported alongside CV |
| 5 | Bias reduction demonstrated on booster | 2 | Booster beats forest on CV *or* explains clearly why not on this dataset |
| 6 | Regularisation levers set deliberately | 2 | `min_samples_leaf`, `max_features`, `learning_rate` explicitly chosen (not defaults) |
| 7 | Permutation importance alongside MDI | 2 | Both reported; disagreement flagged if present |
| 8 | Early stopping on booster | 2 | `early_stopping=True`, `n_iter_no_change` set, `n_iter_` recorded |
| 9 | Reproducibility | 1 | Single `random_state` propagated; two runs produce identical report |
| 10 | Model card / lineage | 2 | `report.json` contains data path, seeds, model versions — supports NIST AI RMF `MAP 1.6` provenance expectation |

Partial credit: missing item 3 or 4 caps the submission at 14 — those are the items that demonstrate understanding, not just scikit-learn call fluency.

## 5. Common mistakes

- **Comparing models on different folds.** Different `cv` objects → different splits → the "better" model may just have gotten easier validation rows. Use one `StratifiedKFold` with a fixed seed for every model.
- **Reading MDI as "the model uses this feature."** MDI is split-count-weighted impurity decrease during training. A randomly-generated high-cardinality column can be ranked high. Always pair with permutation importance.
- **Permutation importance on train.** Measures memorisation, not predictive power. Always on a held-out set.
- **`n_estimators=100` by default.** On many datasets the forest is still noisy at 100 trees. Add trees until OOB stops improving — `n_estimators=500` is a safer default for tabular work.
- **No `class_weight` or resampling on imbalanced targets.** The tree's split criterion is dominated by the majority class; the decision boundary is pushed into minority-class territory. Set `class_weight="balanced"` or stratified-oversample on the training set.
- **Unbounded tree without discussing why.** A `max_depth=None` single tree is a learning device, not a production model. State it, don't ship it.
- **Boosting with no early stopping.** Without `early_stopping` the only levers are `max_iter` and `learning_rate`, and both are chosen blindly. Early stopping with a validation split makes "when to stop" data-driven.
- **Random Forest with `max_features=None`.** This bags fully-correlated trees; variance reduction is far smaller than the sqrt-features default. Decorrelation is the entire point.
- **Confusing bagging with boosting.** Bagging reduces variance (parallel, independent trees on bootstrapped data). Boosting reduces bias (sequential, each tree fits residuals). A learner who gets this wrong will tune the wrong knob when the model underperforms.

## 6. References

- scikit-learn — User Guide, "Decision Trees", "Random Forests", and "Histogram-Based Gradient Boosting". <https://scikit-learn.org/stable/modules/tree.html>, <https://scikit-learn.org/stable/modules/ensemble.html>
- scikit-learn — "Permutation feature importance" and the cautionary note on MDI bias for high-cardinality features. <https://scikit-learn.org/stable/modules/permutation_importance.html>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.5` (document model performance and limitations) and `MAP 1.6` (training-data provenance). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` and `ML05:2023 Model Theft` for the broader threat model around disclosing feature importances. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0024 Exfiltration via ML Inference API` for why over-disclosing importance or decision paths can be an information-leak attack surface. <https://atlas.mitre.org/>
