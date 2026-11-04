# SOLUTION — Exercise 02: scikit-learn pipeline design

> Reference solution for
> [`lessons/mod-102-python-ml-toolchain/exercises/exercise-02-sklearn-pipeline-design.md`](https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-02-sklearn-pipeline-design.md).
> Read it only after attempting your own submission.

## 1. Solution overview

The exercise is an **implementation drill** on the single discipline
that catches more ML production bugs than any other: build a
`Pipeline` + `ColumnTransformer` so that every preprocessing step is
`fit` on the training fold only, then prove it by showing that the
leaky alternative scores higher.

The reference solution picks the **Adult (OpenML) dataset** for the
walkthrough — binary classification on a tabular, mixed-type dataset
with real missing values. The structure mirrors the five parts of the
brief:

| File | Role |
| --- | --- |
| `data.py` | `load()` returns `(X, y, numeric_cols, categorical_cols)` and performs the split *first*. |
| `pipeline.py` | `make_pipeline()` builds the baseline `ColumnTransformer` → `LogisticRegression`. |
| `transformers.py` | `RareCategoryGrouper`: per-column keep-sets learned at `fit`. |
| `leakage_demo.py` | 5-fold CV score, leaky vs. correct imputer placement. |
| `tests/test_*.py` | Four `RareCategoryGrouper` tests + one no-leakage grep test. |

Three non-obvious decisions:

- **Split before you look.** `load()` returns the split tuple; any
  inspection — dtypes, `value_counts`, `isna().mean()` — is done
  *after* the test set has been set aside. This is the only
  discipline that makes the leakage demo fair later. See
  [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
- **`remainder="drop"`** on the `ColumnTransformer`. Silent passthrough
  is the most common cause of "why is there a datetime column in my
  logistic regression". See
  [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
- **`RareCategoryGrouper` before `OneHotEncoder`.** Bucketing rare
  levels drops the one-hot width from hundreds to tens on Adult and
  stabilises CV variance meaningfully. See the sklearn
  [pipelines guide](https://scikit-learn.org/stable/modules/compose.html#pipeline).

## 2. Worked answer or implementation

### Part A — `data.py` (split first, then look)

```python
# data.py
from __future__ import annotations
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector


def load(random_state: int = 0, test_size: float = 0.2):
    """Load Adult, split FIRST, return train/test + column groups.

    The test split is treated as a locked box after this function
    returns. Any inspection of dtypes / distributions below is done
    on the train split only.
    """
    adult = fetch_openml("adult", version=2, as_frame=True)
    X, y = adult.data, adult.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Column groups derived from X_TRAIN only. Do not look at X_test.
    num_sel = make_column_selector(dtype_include="number")
    cat_sel = make_column_selector(dtype_include=["object", "category", "string"])
    numeric_cols = num_sel(X_train)
    categorical_cols = cat_sel(X_train)

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
```

### Part B — `pipeline.py` (baseline)

```python
# pipeline.py
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from transformers import RareCategoryGrouper


def make_preprocessor(numeric_cols, categorical_cols, *, min_count: int = 10):
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_count=min_count)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",          # explicit — see REPORT
        verbose_feature_names_out=True,
    )


def make_pipeline(numeric_cols, categorical_cols, *, C: float = 1.0):
    preprocess = make_preprocessor(numeric_cols, categorical_cols)
    preprocess.set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=0, C=C)),
        ]
    )
```

Fit + score:

```python
from data import load
from pipeline import make_pipeline

X_train, X_test, y_train, y_test, num_cols, cat_cols = load()
pipe = make_pipeline(num_cols, cat_cols)
pipe.fit(X_train, y_train)

# Peek at the first five transformed rows (set_output="pandas")
print(pipe[:-1].transform(X_train.head()))

baseline = pipe.score(X_test, y_test)
print(f"baseline test accuracy: {baseline:.4f}")
```

### Part C — `transformers.py` (`RareCategoryGrouper`)

```python
# transformers.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Map per-column categories seen fewer than ``min_count`` times
    at fit time into a single ``"__rare__"`` bucket. Unseen values
    at transform time also map to ``"__rare__"``.

    Attributes
    ----------
    keep_sets_ : list[set]
        Per-column set of category values whose fit-time count is
        >= ``min_count``.
    n_features_in_ : int
        Number of input columns seen at fit.
    feature_names_in_ : np.ndarray | None
        Column names if fit was called on a DataFrame; else None.
    """

    _RARE = "__rare__"

    def __init__(self, min_count: int = 10):
        self.min_count = min_count

    def fit(self, X, y=None):
        X = self._to_frame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy()
        keep: list[set] = []
        for col in X.columns:
            counts = X[col].astype("object").value_counts(dropna=True)
            keep.append(set(counts[counts >= self.min_count].index))
        self.keep_sets_ = keep
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["keep_sets_"])
        X = self._to_frame(X).copy()
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        for i, col in enumerate(X.columns):
            keep = self.keep_sets_[i]
            s = X[col].astype("object")
            # NaN and any value not in keep -> "__rare__".
            X[col] = s.where(s.isin(keep), self._RARE)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(list(input_features))
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        return np.array([f"x{i}" for i in range(self.n_features_in_)])

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr)
```

Behaviour notes:

- Trailing-underscore attributes (`keep_sets_`, `n_features_in_`,
  `feature_names_in_`) match the scikit-learn
  [developer API conventions](https://scikit-learn.org/stable/developers/develop.html).
- `fit` returns `self`; `check_is_fitted` enforces the pre-fit raise.
- `handle_unknown="ignore"` on the downstream `OneHotEncoder` means
  a `"__rare__"` bucket that happened to never appear at fit is
  still safe — it just encodes to the all-zeros row.

`check_estimator` note: for an input-type-restricted transformer,
several of the sklearn conformance tests legitimately fail
(e.g. the numeric-only input probes). The scikit-learn docs recommend
either implementing `__sklearn_tags__` to declare the constraint or
documenting the skipped checks — both are accepted answers.

### Part D — `GridSearchCV`

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "preprocess__num__impute__strategy": ["median", "mean"],
    "preprocess__cat__rare__min_count": [5, 10, 50],
    "model__C": [0.1, 1.0, 10.0],
}
search = GridSearchCV(
    pipe, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, refit=True
)
search.fit(X_train, y_train)

print("best params :", search.best_params_)
print("best CV acc :", search.best_score_)
print("test acc    :", search.best_estimator_.score(X_test, y_test))
```

Expected output on Adult (seed 0) is in the ballpark of
`best_params_ ≈ {num median, cat min_count 10, C 1.0}`, CV accuracy
around 0.85, and a held-out test accuracy within 0.005 of the CV
score. The exact numbers depend on the OpenML snapshot and are not
worth memorising — the *magnitude of the CV ↔ test gap* is what the
rubric checks.

**Report answer.** `model__C` is the dominant hyperparameter on Adult;
`min_count` matters more for one-hot width than for accuracy; the
imputer strategy barely moves the needle on this dataset.

### Part E — `leakage_demo.py`

```python
# leakage_demo.py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def numeric_only():
    adult = fetch_openml("adult", version=2, as_frame=True)
    num = adult.data.select_dtypes("number").copy()
    return num, adult.target


def leaky_score(X: pd.DataFrame, y: pd.Series) -> float:
    """Fit the imputer on the FULL dataset before CV. BAD."""
    imp = SimpleImputer(strategy="mean").fit(X)
    X_imp = pd.DataFrame(imp.transform(X), columns=X.columns, index=X.index)
    clf = Pipeline(
        [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(clf, X_imp, y, cv=cv, scoring="accuracy").mean()


def correct_score(X: pd.DataFrame, y: pd.Series) -> float:
    """Fit the imputer *inside* the pipeline. GOOD."""
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()


if __name__ == "__main__":
    X, y = numeric_only()
    print(f"leaky   : {leaky_score(X, y):.6f}")
    print(f"correct : {correct_score(X, y):.6f}")
```

Mechanism paragraph for `REPORT.md`:

> In the leaky regime, the imputer computes column means from every
> row in `X` — including the rows that then become the held-out
> fold in each CV split. The imputed values on the fold therefore
> incorporate information from that fold itself. In the correct
> regime, the imputer is fit *inside* each CV train fold, so it
> only ever sees the training rows and the held-out fold is
> genuinely unseen. The leak is tiny per row (a column mean shifts
> by at most 1/n) but it is consistent across every imputed entry,
> and it correlates with the target, so the classifier picks it up.

### Tests

```python
# tests/test_transformers.py
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from transformers import RareCategoryGrouper


def _toy():
    return pd.DataFrame(
        {"c": ["a"] * 20 + ["b"] * 20 + ["c"] * 2 + ["d"] * 2},
        dtype="object",
    )


def test_rare_values_mapped():
    tx = RareCategoryGrouper(min_count=10).fit(_toy())
    out = tx.transform(_toy())
    assert set(out["c"].unique()) == {"a", "b", "__rare__"}


def test_common_values_passthrough():
    tx = RareCategoryGrouper(min_count=10).fit(_toy())
    out = tx.transform(pd.DataFrame({"c": ["a", "b"]}))
    assert out["c"].tolist() == ["a", "b"]


def test_unseen_values_become_rare():
    tx = RareCategoryGrouper(min_count=10).fit(_toy())
    out = tx.transform(pd.DataFrame({"c": ["never_seen", np.nan]}))
    assert out["c"].tolist() == ["__rare__", "__rare__"]


def test_check_is_fitted_raises_before_fit():
    tx = RareCategoryGrouper()
    with pytest.raises(NotFittedError):
        check_is_fitted(tx, attributes=["keep_sets_"])
```

```python
# tests/test_pipeline.py — sanity on the end-to-end pipeline.
from data import load
from pipeline import make_pipeline


def test_pipeline_fits_and_scores():
    X_train, X_test, y_train, y_test, num, cat = load()
    pipe = make_pipeline(num, cat).fit(X_train, y_train)
    assert 0.7 < pipe.score(X_test, y_test) < 0.95


def test_no_fit_transform_touches_X_test():
    # Scan the real project checkout, not pytest's tmp dir. This file
    # lives at <project>/tests/test_pipeline.py, so parent.parent is
    # the project root; `tmp_path / ".." / ".."` would land in pytest's
    # basetemp and read zero project files.
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    files = [p for p in project_root.rglob("*.py") if ".venv" not in p.parts]
    assert files, f"no project .py files found under {project_root}"
    # Narrow heuristic: no co-mention of fit_transform and X_test.
    for path in files:
        for line in path.read_text().splitlines():
            assert not (
                "fit_transform" in line and "X_test" in line
            ), f"leaky call in {path}: {line.strip()}"
```

### `REPORT.md` — reference answers

- **Split-before-look.** Done in `data.py`.
- **`remainder=`.** `"drop"` — the brief lists only two column groups,
  and silent passthrough of any new column would land it unscaled in
  the logistic regression.
- **Effect of `RareCategoryGrouper` on one-hot width.** On Adult with
  `min_count=10`, the one-hot block shrinks from ~100 columns to ~35.
- **CV ↔ test gap.** Within 0.005 absolute. No tuning optimism visible
  at this grid size; a 100-point grid on a smaller dataset would
  likely show it.
- **Leaky vs. correct CV.** Leaky typically ~0.001–0.003 higher than
  correct on Adult; the paragraph above explains the mechanism.

## 3. Validation steps

```bash
uv sync
uv run pytest -q                   # all transformer + pipeline tests green
uv run python leakage_demo.py      # prints leaky and correct scores
uv run python -c "from pipeline import make_pipeline; print('ok')"
```

Spot checks:

1. `grep -n "fit_transform.*X_test\|X_test.*fit_transform"` → empty.
2. `grep -n "remainder=" pipeline.py` → the explicit choice, not the default.
3. `grep -n "set_output" pipeline.py` → called before `fit`.
4. `pipe[:-1].transform(X_train.head())` returns a DataFrame with
   verbose column names (`num__age`, `cat__workclass_Private`, …).

## 4. Rubric or review checklist

| Dimension | Looking for | Weight |
| --- | --- | --- |
| Pre-split discipline | `X_test` never named in a `fit`/`fit_transform` call | 15 |
| `ColumnTransformer` shape | Two branches, explicit `remainder=`, `set_output("pandas")` | 15 |
| `RareCategoryGrouper` correctness | Trailing-`_` attributes; pre-fit raise; DataFrame + ndarray input | 20 |
| `GridSearchCV` wiring | Nested param names resolve; `refit=True`; test-score reported | 15 |
| Leakage demo | Both numbers reported; mechanistic paragraph (not just numbers) | 20 |
| Transformer tests | All four tests from the brief present and passing | 10 |
| Report | Answers all five report prompts honestly | 5 |

Binary review checklist:

- [ ] `data.py` splits before any inspection.
- [ ] `ColumnTransformer` sets `remainder=` explicitly.
- [ ] `set_output(transform="pandas")` is called; feature names look readable.
- [ ] `RareCategoryGrouper` has `keep_sets_`, `n_features_in_`, returns `self` from `fit`.
- [ ] `check_is_fitted` protects `transform`.
- [ ] `GridSearchCV` runs, best params reported, best estimator scored on held-out test.
- [ ] `leakage_demo.py` prints leaky >= correct (small gap) on Adult.
- [ ] No line in the source co-mentions `fit_transform` and `X_test`.

## 5. Common mistakes

1. **Fitting the imputer on `X` before splitting.** The exact bug the
   leakage demo is designed to reveal. The pipeline fixes it
   automatically — the whole point of this exercise is to feel why.
2. **Using `remainder="passthrough"` by default.** Silently keeps every
   column the `ColumnTransformer` did not name. The scikit-learn docs
   call this out as the most common `ColumnTransformer` mistake.
3. **Hard-coding column names.** Breaks when the schema changes.
   `make_column_selector(dtype_include=...)` is robust.
4. **Forgetting `handle_unknown="ignore"` on `OneHotEncoder`.** The
   default raises on unseen levels at predict time, which is exactly
   when you cannot afford it.
5. **Setting `n_jobs=-1` in a notebook.** On some platforms joblib's
   worker pool collides with the kernel. Run `GridSearchCV` from a
   script.
6. **`fit_transform` on the test set.** The one unforgivable sin.
   The `test_no_fit_transform_touches_X_test` test exists specifically
   as a grep-based lint.
7. **Dropping the `check_is_fitted` call in `transform`.** You lose
   the `NotFittedError` protection; wiring the transformer into a
   `Pipeline` that is never fit will silently return garbage instead
   of raising.
8. **`sparse_output=True` on `OneHotEncoder` without handling.** A
   downstream DataFrame-returning step will refuse. For this exercise,
   `sparse_output=False` is fine on Adult's ~35 one-hot columns.

## 6. References

Local exercise context:

- Learning exercise —
  <https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-02-sklearn-pipeline-design.md>
- Companion chapters — `05-sklearn-estimators-and-transformers.md`,
  `06-pipelines-and-columntransformer.md`.
- Builds on exercise-01's feature matrix (optional input).

Official project documentation:

- scikit-learn user guide — <https://scikit-learn.org/stable/user_guide.html>
- Pipelines and composite estimators —
  <https://scikit-learn.org/stable/modules/compose.html#pipeline>
- `ColumnTransformer` —
  <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>
- `SimpleImputer` —
  <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>
- `OneHotEncoder` —
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>
- `GridSearchCV` —
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>
- `train_test_split` —
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>
- Developing estimators —
  <https://scikit-learn.org/stable/developers/develop.html>
- OpenML "Adult" dataset page —
  <https://www.openml.org/d/1590>
