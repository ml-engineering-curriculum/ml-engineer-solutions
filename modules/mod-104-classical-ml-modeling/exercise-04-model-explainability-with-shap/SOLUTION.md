# exercise-04 — Model Explainability With SHAP · SOLUTION

Paired exercise: `lessons/mod-104-classical-ml-modeling/exercises/exercise-04-model-explainability-with-shap.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to explain a tuned tree-based classifier with SHAP and present both global and local explanations that a reviewer (data scientist, PM, compliance partner) could actually act on. The reference solution frames the work around three ideas:

1. **Use the right explainer for the model.** `shap.TreeExplainer` is exact for tree ensembles and orders of magnitude cheaper than model-agnostic explainers. Using `KernelExplainer` on a tree model is a textbook "worked but was 100× slower and less accurate" mistake.
2. **Explain predictions on a *held-out* sample.** Explaining training rows is explaining memorisation. Explanations should be computed on the same slice of data a reviewer would look at in production.
3. **Report both global and local views, and name the assumptions.** Mean |SHAP| gives the global feature ranking; waterfall and force plots cover individual predictions. The report calls out that SHAP assumes feature independence in its aggregation step, which breaks on highly correlated features.

The explainability artefacts are the *output* of this exercise; the *input* is a trained model from exercise-02 or exercise-03. The exercise is not "train a model with SHAP" — it is "explain a model you already trust."

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| `shap.TreeExplainer` on the LightGBM/XGBoost/RF model | `KernelExplainer` | Exact for trees, ~1000× faster, no background-set choice to get wrong |
| `feature_perturbation="tree_path_dependent"` | `"interventional"` with a background set | Exact Shapley values for tree models without requiring a background distribution; interventional needs a representative background that is hard to pick |
| Explanations on test rows | Training rows | A production explanation is for an unseen prediction; training-row explanations are a different question (model internals, not predictions) |
| Mean \|SHAP\| for global ranking | Built-in `feature_importances_` (MDI) | MDI is biased toward high-cardinality features; mean \|SHAP\| is the model-output-attribution ranking |
| Waterfall for *selected* individual rows (TP/FP/FN/TN) | Random rows | A reviewer wants to see explanations for the decisions that *matter*: a confident wrong prediction, a confident right one, a borderline case. Random rows waste attention |
| Save SHAP values to parquet alongside plots | Plots only | Plots are not reproducible or diff-able; the numeric values are the artefact, plots are a projection |
| Permutation importance alongside SHAP | SHAP only | They answer different questions — "how much does the model *rely* on this feature?" vs "how much does the feature *change the output* for this prediction?" Divergence is informative |

## 2. Worked implementation

### 2.1 Project layout

```
explain/
├── __init__.py
├── load.py            # load the exercise-02 / -03 model + its test slice
├── global_.py         # mean |SHAP| ranking, summary plot
├── local.py           # per-row waterfall/force plots on selected rows
├── interactions.py    # pairwise SHAP interaction values (optional)
└── run.py             # orchestration entry point
```

### 2.2 Load model + held-out slice

```python
# explain/load.py
from dataclasses import dataclass
import joblib
import pandas as pd


@dataclass(frozen=True)
class Loaded:
    model: object
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]


def load(model_path: str, test_parquet: str, target: str) -> Loaded:
    model = joblib.load(model_path)
    df = pd.read_parquet(test_parquet)
    y_test = df[target].astype("int64")
    X_test = df.drop(columns=[target])
    return Loaded(model=model, X_test=X_test, y_test=y_test,
                  feature_names=list(X_test.columns))
```

The model and the test split both come from upstream. SHAP is a lens on an existing model, not a new pipeline.

### 2.3 Global explanation — mean |SHAP|

```python
# explain/global_.py
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def compute_shap(model, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    return explainer(X)


def global_report(sv: shap.Explanation, out_dir: Path) -> pd.DataFrame:
    # Binary classifier: sv.values is (n_samples, n_features) for the positive class
    # (newer SHAP returns (n_samples, n_features, n_classes); normalise to 2D).
    vals = np.asarray(sv.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    mean_abs = np.abs(vals).mean(axis=0)

    ranking = (
        pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_dir / "global_ranking.csv", index=False)

    plt.figure()
    shap.summary_plot(vals, sv.data, feature_names=sv.feature_names,
                      plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    return ranking
```

What this gives the reviewer:

- `global_ranking.csv` — the diff-able ranking. Two different models on the same test slice can now be compared numerically.
- `summary_plot.png` — one plot that encodes both ranking and direction. A learner should be able to read "feature X above the line pushes the prediction up; below the line, down."

### 2.4 Local explanations — on *chosen* rows, not random ones

```python
# explain/local.py
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def pick_rows(y_true: pd.Series, y_proba: np.ndarray, k: int = 2) -> dict[str, list[int]]:
    y_pred = (y_proba >= 0.5).astype(int)
    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    def top_k(mask: np.ndarray, score: np.ndarray, descending: bool) -> list[int]:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        order = np.argsort(-score[idx] if descending else score[idx])
        return idx[order][:k].tolist()

    return {
        "most_confident_tp": top_k(tp_mask, y_proba, descending=True),
        "most_confident_fp": top_k(fp_mask, y_proba, descending=True),
        "most_confident_fn": top_k(fn_mask, y_proba, descending=False),
        "borderline": np.argsort(np.abs(y_proba - 0.5))[:k].tolist(),
    }


def local_report(sv: shap.Explanation, y_true, y_proba, out_dir: Path) -> dict:
    picks = pick_rows(y_true.to_numpy(), np.asarray(y_proba))
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded: dict = {}
    for label, rows in picks.items():
        for r in rows:
            plt.figure()
            shap.plots.waterfall(sv[r], max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / f"waterfall__{label}__row_{r}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
        recorded[label] = rows
    return recorded
```

Why these four classes of row:

- **Most confident TP / TN** — "when the model is right and sure, what did it look at?"
- **Most confident FP / FN** — "when the model is wrong and sure, what led it astray?" This is where the real learnings live.
- **Borderline** — "on the fence — what tiny signal would have flipped the decision?"

Explaining a uniformly-sampled test row is almost never useful; the signal is in the extremes.

### 2.5 Orchestrator

```python
# explain/run.py
import argparse
import json
from pathlib import Path

import numpy as np

from .load import load
from .global_ import compute_shap, global_report
from .local import local_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="joblib-serialised sklearn/LightGBM model")
    p.add_argument("--test", required=True, help="parquet of held-out test rows + target")
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="artifacts/exercise-04")
    args = p.parse_args()

    loaded = load(args.model, args.test, args.target)
    sv = compute_shap(loaded.model, loaded.X_test)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = global_report(sv, out_dir)

    # Prediction probabilities for the positive class (handles sklearn + LightGBM booster).
    predict_proba = getattr(loaded.model, "predict_proba", None)
    if predict_proba is not None:
        y_proba = predict_proba(loaded.X_test)[:, 1]
    else:
        y_proba = loaded.model.predict(loaded.X_test)
    picks = local_report(sv, loaded.y_test, y_proba, out_dir)

    # Persist raw SHAP values so plots are reproducible from numbers.
    np.savez_compressed(
        out_dir / "shap_values.npz",
        values=np.asarray(sv.values),
        base_values=np.asarray(sv.base_values),
        data=np.asarray(sv.data),
        feature_names=np.asarray(sv.feature_names),
    )

    report = {
        "top_10_features_by_mean_abs_shap": ranking.head(10).to_dict(orient="records"),
        "explained_rows": picks,
        "n_test_rows": int(len(loaded.X_test)),
        "explainer": "TreeExplainer",
        "feature_perturbation": "tree_path_dependent",
        "model_output": "raw",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

### 2.6 Reading the output

A reviewer looking at the artefacts should be able to answer, in order:

1. "Which features drive this model globally?" → `global_ranking.csv` + `summary_plot.png`.
2. "When the model is confidently wrong, what is it seeing?" → `waterfall__most_confident_fp_*.png`, `waterfall__most_confident_fn_*.png`.
3. "On a borderline case, which features tip the decision?" → `waterfall__borderline_*.png`.
4. "Are the explanations stable if the pipeline is rerun?" → `shap_values.npz` is byte-identical across runs when the model and seed are fixed.

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **Import check** — `python -c "import explain.run"`.
2. **Unit: explainer matches model** — on a LightGBM booster, assert `shap.TreeExplainer(model)` constructs without falling back to `KernelExplainer`. Easy to miss when the model is loaded from a generic sklearn wrapper.
3. **Unit: SHAP values are additive** — for a random test row `i`, assert `base_value + sum(shap_values[i]) ≈ model.predict(raw_output)[i]` within 1e-6. This is the SHAP contract; if it fails, the explainer is misconfigured (likely `model_output` mismatch).
4. **Unit: global ranking matches `.feature_importances_` loosely** — the top-5 of mean |SHAP| should share at least 3 features with the top-5 of MDI. Large divergence is fine (the point of showing both) but zero overlap usually means a feature-name alignment bug.
5. **Determinism** — rerun the orchestrator; `shap_values.npz` and `global_ranking.csv` are byte-identical.
6. **Unit: correct class in multi-class case** — on a 3-class model, assert the code paths to extract "positive class" SHAP values pick the right class index, not silently default to class 0.
7. **Plot sanity check** — `summary_plot.png` exists, is non-empty, and contains at least one glyph per feature. Catches the `plt.close()`-before-`savefig` bug that produces blank plots.
8. **Row-pick semantics** — with a synthetic `(y_true, y_proba)` where all rows are TP, `most_confident_fp` must be `[]`. Confirms the masking logic handles empty subsets.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Correct explainer for the model class | 2 | `TreeExplainer` on tree models; not `KernelExplainer` |
| 2 | Explanations on a held-out slice | 2 | SHAP values computed on test rows, not training rows |
| 3 | Global ranking persisted as numbers *and* a plot | 2 | `global_ranking.csv` + `summary_plot.png`; CSV is the canonical artefact |
| 4 | Local explanations on *selected* rows | 2 | TP / FP / FN / borderline picks explained with waterfall or force plots |
| 5 | Raw SHAP values saved | 2 | `shap_values.npz` (or equivalent) saved; plots are regeneratable from numbers |
| 6 | SHAP additivity check documented | 2 | At least one unit test or printed assertion that `base + sum = prediction` within float tolerance |
| 7 | Correlated-feature caveat called out | 2 | Report or README notes that SHAP's aggregation assumes independence; highly correlated features split credit in a way that misleads the ranking |
| 8 | Multi-class handling is explicit | 1 | Binary vs multi-class code path is clearly separated; class index is documented |
| 9 | Determinism | 1 | Two runs produce identical `shap_values.npz` and `global_ranking.csv` |
| 10 | Model card linkage | 2 | Explanation artefacts referenced from a model card entry — supports NIST AI RMF `MEASURE 2.9` (explainability/interpretability requirements) |
| 11 | Threat-model awareness | 2 | Report notes that over-disclosing explanations widens the model-inversion / model-theft surface — connects to OWASP ML Top 10 `ML05:2023 Model Theft` and MITRE ATLAS `AML.T0024 Exfiltration via ML Inference API` |

Partial credit: missing item 7 or 11 caps the submission at 14 — those are the items that distinguish "ran SHAP" from "deployed SHAP responsibly."

## 5. Common mistakes

- **Using `KernelExplainer` on a tree model.** `TreeExplainer` is both exact and ~1000× faster. `KernelExplainer` also requires a "background set" whose choice silently biases the explanation.
- **Explaining the training set.** Those rows are inside the model's memorised region. Production explanations are for unseen inputs; use the test slice (or a fresh slice drawn from the same distribution).
- **Reading SHAP ranking as causal.** Mean |SHAP| ranks features by **contribution to model output**, not causal effect on the target. "Removing feature X will drop performance by Y" requires an ablation, not a SHAP number.
- **`model_output="raw"` with a probability-space story.** For binary classifiers, raw output is log-odds; interpreting a "+0.3 SHAP contribution" as "+30% probability" is wrong. Either switch to `model_output="probability"` (approximate, slower) or keep raw and describe contributions in log-odds.
- **Ignoring correlated features.** SHAP splits credit across correlated columns roughly proportionally; a reviewer who reads "feature A matters 10 and feature B matters 5" may miss that B is a noisy copy of A and the two together explain 15. Document this.
- **Random local explanations.** A learner samples 10 random rows and plots waterfalls. None of them are interesting. Pick the four quadrants (confident TP/FP, confident FN, borderline).
- **Plots-only output.** `.png` files cannot be diffed, cannot be regenerated with different cosmetics, cannot be checked for numeric equivalence across model versions. Persist the raw values.
- **Multi-class silently collapsed to class 0.** Newer SHAP returns `(n, m, k)` arrays; indexing `[:, :, 1]` is fine for binary but wrong for any other number of classes. State the class explicitly.
- **Over-disclosing explanations.** A public endpoint that returns SHAP values alongside each prediction gives an attacker a noise-free gradient to reconstruct the model. In a production setting, explanations are an internal artefact unless policy explicitly allows otherwise.

## 6. References

- SHAP (Lundberg et al.) — official docs, in particular `TreeExplainer` and the additivity property. <https://shap.readthedocs.io/en/latest/>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.9` (document explainability and interpretability) and `MAP 3.4` (documenting system intent for downstream use). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft` for the information-disclosure risk of over-exposing explanations at an inference endpoint. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0024 Exfiltration via ML Inference API` as the threat model under which explanation endpoints must be rate-limited and authenticated. <https://atlas.mitre.org/>
- scikit-learn — "Permutation feature importance" (the honest-generalisation companion to SHAP's model-attribution view). <https://scikit-learn.org/stable/modules/permutation_importance.html>
