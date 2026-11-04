# SOLUTION — Exercise 04: Testing ML code

> Reference solution for
> [`lessons/mod-102-python-ml-toolchain/exercises/exercise-04-testing-ml-code.md`](https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-04-testing-ml-code.md).
> Read it only after attempting your own submission.

## 1. Solution overview

The exercise is an **implementation drill** on the four layers of
tests that together let you ship ML code: unit → property → golden
→ CI. Each layer catches a different failure mode, and the point is
to feel which test catches what by deliberately breaking each one.

The reference solution extends the `ml_toolkit` project from
exercise-03 and uses the `RareCategoryGrouper` from exercise-02 as
the system-under-test. The deliverable adds:

```
ml_toolkit/
├── src/ml_toolkit/
│   ├── transformers.py                # RareCategoryGrouper (typed)
│   └── pipeline.py                    # build_pipeline(...)
├── tests/
│   ├── test_transformer.py            # Part A — five unit axes
│   ├── test_properties.py             # Part B — hypothesis invariant
│   ├── test_golden.py                 # Part C — pinned predictions
│   └── data/
│       └── pipeline_predictions_golden.npy
├── scripts/regen_goldens.py
└── .github/workflows/ci.yml           # Part D — lint + types + coverage
```

Three non-obvious decisions:

- **Pin every random seed, then checksum with `assert_allclose`.**
  Bit-exact equality across platforms is fragile; `rtol=1e-6,
  atol=1e-8` catches real regressions without flaking on macOS vs.
  Linux BLAS backends. See
  [`numpy.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html).
- **Golden regeneration is gated behind an env var.** A plain
  `pytest` run never writes the golden file; only
  `REGENERATE_GOLDENS=1 pytest` does. Prevents the "test broke,
  updated the golden" anti-pattern.
- **Coverage gate on `src/` only, not on tests.** `tests/` is where
  coverage *comes from* — gating on it makes coverage self-reporting.
  `--cov=src --cov-fail-under=90` is the right shape.

## 2. Worked answer or implementation

### Part A — `tests/test_transformer.py` (five unit axes)

```python
# tests/test_transformer.py
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ml_toolkit.transformers import RareCategoryGrouper


def _toy():
    return pd.DataFrame(
        {"c": ["a"] * 20 + ["b"] * 20 + ["c"] * 2 + ["d"] * 2},
        dtype="object",
    )


# Axis 1 — correctness on a known input
def test_fit_transform_known_output():
    out = RareCategoryGrouper(min_count=10).fit_transform(_toy())
    expected = pd.DataFrame(
        {"c": ["a"] * 20 + ["b"] * 20 + ["__rare__"] * 4},
        dtype="object",
    )
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected.reset_index(drop=True)
    )


# Axis 2 — fit is idempotent
def test_fit_is_idempotent():
    tx = RareCategoryGrouper(min_count=10)
    once = tx.fit(_toy()).keep_sets_
    twice = tx.fit(_toy()).keep_sets_
    assert once == twice


# Axis 3 — pre-fit transform raises
def test_pre_fit_raises():
    tx = RareCategoryGrouper()
    with pytest.raises(NotFittedError):
        check_is_fitted(tx, attributes=["keep_sets_"])


# Axis 4 — train/test discipline: transforming test does not mutate state
def test_state_unchanged_by_transforming_test_data():
    tx = RareCategoryGrouper(min_count=10).fit(_toy())
    before = [s.copy() for s in tx.keep_sets_]
    unseen = pd.DataFrame({"c": ["never_seen", "also_new"]}, dtype="object")
    _ = tx.transform(unseen)
    after = tx.keep_sets_
    assert before == after


# Axis 5 — clone drops fitted state
def test_clone_drops_state():
    tx = RareCategoryGrouper(min_count=10).fit(_toy())
    cloned = clone(tx)
    assert cloned.min_count == 10
    assert not hasattr(cloned, "keep_sets_")
```

### Part B — `tests/test_properties.py` (Hypothesis)

```python
# tests/test_properties.py
import string
import pandas as pd
from hypothesis import given, settings, strategies as st

from ml_toolkit.transformers import RareCategoryGrouper


alphabet = st.sampled_from(list(string.ascii_lowercase[:6]))


@settings(max_examples=50, deadline=None)
@given(
    rows=st.lists(alphabet, min_size=1, max_size=200),
    min_count=st.integers(min_value=1, max_value=20),
)
def test_output_values_subset_of_train_plus_rare(rows, min_count):
    """For any DataFrame and min_count, every value in the transformed
    output is either a fit-time category or the rare sentinel."""
    X = pd.DataFrame({"c": rows}, dtype="object")
    tx = RareCategoryGrouper(min_count=min_count).fit(X)
    kept = tx.keep_sets_[0] | {"__rare__"}

    # Also probe values never seen at fit.
    probe = pd.DataFrame(
        {"c": rows + ["z_novel_a", "z_novel_b"]}, dtype="object"
    )
    out = tx.transform(probe)
    assert set(out["c"].unique()).issubset(kept)
```

Why a subset invariant: it is *structural*, not numerical, so it
survives Hypothesis's shrinking without tolerance tuning. If the
assertion fails, the transformer is broken, not the test.

### Part C — `tests/test_golden.py` (pinned end-to-end predictions)

```python
# tests/test_golden.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from ml_toolkit.pipeline import build_pipeline


GOLDEN = Path(__file__).parent / "data" / "pipeline_predictions_golden.npy"


def _tiny_dataset():
    rng = default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "income": rng.normal(50_000, 15_000, n),
            "country": rng.choice(["US", "GB", "DE", "FR", "JP"], n),
            "job": rng.choice(["eng", "sales", "ops", "exec", "misc"], n),
        }
    )
    # Inject NaNs to exercise the imputer branch.
    df.loc[rng.choice(n, 10, replace=False), "age"] = np.nan
    y = (df["income"] > 55_000).astype(int).to_numpy()
    return df, y


def _fit_predict() -> np.ndarray:
    X, y = _tiny_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    pipe = build_pipeline(
        numeric_cols=["age", "income"],
        categorical_cols=["country", "job"],
        C=1.0,
        min_count=5,
    )
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def test_pipeline_predictions_match_golden():
    probs = _fit_predict()
    if os.environ.get("REGENERATE_GOLDENS") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.save(GOLDEN, probs)
        pytest.skip("golden regenerated")
    if not GOLDEN.exists():
        pytest.skip(
            "golden file missing; run `REGENERATE_GOLDENS=1 pytest` to create it"
        )
    golden = np.load(GOLDEN)
    np.testing.assert_allclose(probs, golden, rtol=1e-6, atol=1e-8)
```

Companion regenerator script (makes the deliberate act of
regenerating easy to find in `git blame`):

```python
# scripts/regen_goldens.py
"""Regenerate all golden files. Run deliberately as part of a PR
that intentionally changes the pipeline — not to make a red test go
green."""
from __future__ import annotations
import os
import subprocess
import sys


def main() -> int:
    os.environ["REGENERATE_GOLDENS"] = "1"
    return subprocess.call([sys.executable, "-m", "pytest", "tests/test_golden.py"])


if __name__ == "__main__":
    sys.exit(main())
```

The break-and-restore ritual:

```bash
# 1. Baseline: tests green.
uv run pytest -q

# 2. Break it: change the numeric imputer from median to mean.
sed -i.bak 's/strategy="median"/strategy="mean"/' src/ml_toolkit/pipeline.py
uv run pytest tests/test_golden.py          # fails with an assert_allclose diff

# 3. Restore and re-verify.
mv src/ml_toolkit/pipeline.py.bak src/ml_toolkit/pipeline.py
uv run pytest tests/test_golden.py          # green again
```

### Part D — `.github/workflows/ci.yml`

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv python install
      - run: uv sync --frozen
      - name: ruff check
        run: uv run ruff check .
      - name: ruff format check
        run: uv run ruff format --check .
      - name: mypy
        run: uv run mypy src tests
      - name: pytest with coverage
        run: uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=90
```

The three deliberate failure drills:

1. **Lint failure.** Add `import os` to a file that does not use it.
   Commit. CI red at `ruff check`: `F401 'os' imported but unused`.
   Revert.
2. **Type failure.** Change `standardise(X: npt.NDArray[np.floating])`
   to accept `X: list` at a call site. Commit. CI red at `mypy`:
   `Argument 1 to "standardise" has incompatible type "list[float]";
   expected "ndarray[Any, dtype[floating[Any]]]"`. Revert.
3. **Golden failure.** Change `min_count=5` to `min_count=50` in
   `build_pipeline`'s default. Commit. CI red at `pytest`:
   `assert_allclose: Max absolute difference: 0.07...`. Revert.

Each failure in `REPORT.md` is annotated with which check caught it
and how clearly. The lint error is the loudest (one line, exact
rule name). The mypy error is the most useful (points at the exact
callsite). The golden error is the most alarming by design —
"predictions changed" is a lot louder than "unused import".

### Supporting source

```python
# src/ml_toolkit/pipeline.py
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from ml_toolkit.transformers import RareCategoryGrouper


def build_pipeline(
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    C: float = 1.0,
    min_count: int = 10,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_count=min_count)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=1000, random_state=0, C=C)),
        ]
    )
```

`transformers.RareCategoryGrouper` is taken verbatim from exercise-02
with full type hints added for `mypy --strict`.

### `REPORT.md` — reference answer

- **What unit tests miss that the golden catches.** The unit tests
  pin the behaviour of each component in isolation; the golden
  pins their *composition* on realistic data. A refactor that keeps
  each unit test green but changes a default (e.g. `min_count`
  default, `C` default, imputer strategy) silently moves every
  prediction. The golden catches it.
- **What the golden misses that the unit tests catch.** The golden
  only exercises one input — the tiny fixed dataset. A bug that
  only shows up on a specific column dtype or an edge-case input
  slips past it. Unit + property tests cover that surface.
- **Coverage on `src/` ≥ 90%.** Measured on `transformers.py` and
  `pipeline.py` (the business logic). The small `features.py`
  module from exercise-03 is still tested there.
- **Deliberate failures.** The lint, type, and golden breaks were
  all tried and reverted; the CI run URLs (or screenshots) are
  pasted into the report, each annotated with which step went red.

## 3. Validation steps

```bash
# Full local run.
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=90

# Regenerate the golden intentionally (only when the pipeline changed on purpose).
REGENERATE_GOLDENS=1 uv run pytest tests/test_golden.py
```

Spot checks:

1. `git ls-files tests/data/` includes `pipeline_predictions_golden.npy`.
2. `git log --diff-filter=A -- tests/data/pipeline_predictions_golden.npy`
   turns up a commit whose message explicitly names the pipeline
   change that produced it.
3. `uv run pytest -q` runs under ~10 seconds on a laptop (tests are
   tiny by construction).
4. The `check_estimator` conformance suite — if wired — is marked
   `@pytest.mark.slow` and only runs under `pytest -m slow`.

## 4. Rubric or review checklist

| Dimension | Looking for | Weight |
| --- | --- | --- |
| Unit tests (A) | All five axes present and passing; test names describe the axis | 20 |
| Property test (B) | ≥ 1 Hypothesis test on a structural invariant; `max_examples ≥ 50` | 15 |
| Golden test (C) | `.npy` committed; `assert_allclose` tolerances explicit; regeneration gated | 20 |
| Break-and-restore | All three deliberate failures tried; `REPORT.md` names which check caught each | 10 |
| CI quality gates | `ruff check` + `ruff format --check` + `mypy` + `pytest --cov` | 15 |
| Coverage | `--cov-fail-under=90` on `src/`; not chasing 100% | 10 |
| Report | Mechanistic answer on what each layer catches | 10 |

Binary review checklist:

- [ ] Five unit-test axes (correctness, idempotency, pre-fit raise,
      train/test discipline, clone) all present and passing.
- [ ] Hypothesis property test passes without a counterexample.
- [ ] Golden `.npy` is committed and loaded by `assert_allclose`.
- [ ] Golden regeneration is gated behind `REGENERATE_GOLDENS=1`.
- [ ] CI runs ruff + ruff format + mypy + pytest with coverage.
- [ ] `--cov-fail-under=90` on `src/`.
- [ ] Three deliberate failures and their recovered states captured
      in `REPORT.md`.

## 5. Common mistakes

1. **Updating the golden to make a red test green.** Defeats the
   whole purpose. Golden regeneration must be a deliberate act in
   the same PR as the code change, with a commit message that
   explains what changed and why.
2. **`np.testing.assert_array_equal` instead of `assert_allclose`.**
   Floating-point equality fails intermittently on different BLAS
   backends. `rtol=1e-6, atol=1e-8` is the standard tolerance.
3. **Hypothesis strategies with no shape bound.** A `data_frames`
   strategy with no `columns=` or `rows=` constraint can generate
   10k-row test cases and make CI time out. Cap at 200 rows / 10
   columns for this scale.
4. **A giant golden file.** If the golden is multiple megabytes,
   the test set is too big — the golden should load in milliseconds
   and diff on failure in a human-readable way.
5. **No type hints on the system under test.** `mypy --strict`
   trips on untyped `.fit` returns. Add
   `# type: ignore[no-any-return]` only at the specific sklearn
   call sites, not module-wide.
6. **CI uses `uv sync` without `--frozen`.** CI silently updates the
   lockfile and the drift-detection test from exercise-03 breaks.
7. **Coverage gate on `tests/`.** Coverage is reported *by* the tests;
   coverage *on* the tests measures nothing useful. `--cov=src`.
8. **Chasing 100% coverage.** Encourages test theatre (asserting on
   things the implementation does trivially). 90% is the sweet spot;
   above that is diminishing returns.
9. **Not pinning `random_state` on the model or the split.** Makes
   the golden test flake. Every randomness source (`default_rng`,
   `train_test_split`, model `random_state`) must be seeded.
10. **Forgetting to seed BLAS / OMP for exact bit equality.** Don't
    try. `rtol=1e-6` is the correct answer; chasing bit-equality
    across hardware is a rabbit hole.

## 6. References

Local exercise context:

- Learning exercise —
  <https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-04-testing-ml-code.md>
- Companion chapter — `08-testing-ml-code.md`.
- Builds on exercise-02 (`RareCategoryGrouper`) and exercise-03
  (`ml_toolkit` project skeleton).

Official project documentation:

- pytest — <https://docs.pytest.org/en/stable/>
- pytest-cov — <https://pytest-cov.readthedocs.io/>
- Hypothesis — <https://hypothesis.readthedocs.io/en/latest/>
- Hypothesis NumPy / Pandas strategies —
  <https://hypothesis.readthedocs.io/en/latest/numpy.html>
- NumPy testing (`assert_allclose`) —
  <https://numpy.org/doc/stable/reference/routines.testing.html>
- scikit-learn developer guide (`check_estimator`, `clone`) —
  <https://scikit-learn.org/stable/developers/develop.html>
- scikit-learn `check_is_fitted` —
  <https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html>
- ruff — <https://docs.astral.sh/ruff/>
- mypy — <https://mypy.readthedocs.io/en/stable/>
- GitHub Actions — <https://docs.github.com/en/actions>
- `astral-sh/setup-uv` action —
  <https://github.com/astral-sh/setup-uv>
