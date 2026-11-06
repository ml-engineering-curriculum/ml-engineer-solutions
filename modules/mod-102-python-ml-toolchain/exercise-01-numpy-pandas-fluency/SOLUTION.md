# SOLUTION — Exercise 01: NumPy / Pandas fluency

> Reference solution for
> [`lessons/mod-102-python-ml-toolchain/exercises/exercise-01-numpy-pandas-fluency.md`](https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-01-numpy-pandas-fluency.md)
> in the paired learning repo. Read it only after attempting your own
> submission.

## 1. Solution overview

The exercise is an **implementation drill** on the day-to-day
NumPy / Pandas idioms an ML engineer leans on: deliberate dtypes,
vectorised NumPy primitives, cardinality-safe joins, and window-bound
group-by aggregations — all with no row-level Python loops. The
deliverable is a small feature-engineering module that turns three
raw tables (`users`, `products`, `transactions`) into a per-user
feature matrix.

The reference solution is organised as four files that mirror the
four parts of the brief:

| File | Role |
| --- | --- |
| `generate_data.py` | Synthetic data generator (seeded) with the four data-quality issues baked in. |
| `numpy_warmups.py` | Pure-NumPy primitives: rolling mean, pairwise cosine, top-k per row, safe `log1p`. |
| `pandas_features.py` | `load` → `clean_transactions` → `enrich` → `build_user_features` → `final_feature_matrix`. |
| `tests/` | `pytest` coverage for every function, plus the `enrich` `validate=` guard. |

The three non-obvious decisions:

- **Dtype hygiene at the boundary.** Dtypes are pinned on `read_parquet`
  and re-asserted at the top of every public function. See the
  Pandas user guide on
  [basics/dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes).
- **`validate="many_to_one"` on every merge.** It is free insurance
  against the second-most-common pandas bug: a silently fan-out merge.
  See
  [`pandas.DataFrame.merge`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html).
- **`NaN ≠ 0` for empty-window aggregates.** `mean_basket_90d` is
  `NaN` when the user has no transactions in the window. These mean
  different things downstream (new user vs. idle spender) and
  collapsing them is a real modelling bug.

## 2. Worked answer or implementation

### Part A — `generate_data.py`

```python
# generate_data.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def generate(seed: int = 0, out_dir: Path = Path("data")) -> dict[str, Path]:
    """Write three parquet files that exercise the pipeline's edge cases.

    Data-quality issues deliberately baked in:
      - ~5% of ``age`` is NaN.
      - ~1% of transactions reference an unknown ``user_id``.
      - A couple of ``users`` have zero transactions (inactive).
      - A small fraction of ``timestamp`` values is in the future.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_users, n_products, n_txns = 1_000, 50, 20_000
    as_of = pd.Timestamp("2025-01-01")

    # --- users
    user_ids = np.array([f"u{i:05d}" for i in range(n_users)])
    signup = as_of - pd.to_timedelta(rng.integers(0, 1_000, n_users), unit="D")
    country = pd.Categorical(
        rng.choice(["US", "GB", "DE", "FR", "JP"], n_users),
        categories=["US", "GB", "DE", "FR", "JP"],
    )
    age = pd.array(rng.integers(18, 70, n_users), dtype="Int64")
    missing_age = rng.random(n_users) < 0.05  # ~5% missing
    age[missing_age] = pd.NA
    users = pd.DataFrame(
        {
            "user_id": pd.array(user_ids, dtype="string"),
            "signup_date": signup.normalize(),
            "country": country,
            "age": age,
        }
    )

    # --- products
    product_ids = np.array([f"p{i:04d}" for i in range(n_products)])
    products = pd.DataFrame(
        {
            "product_id": pd.array(product_ids, dtype="string"),
            "category": pd.Categorical(
                rng.choice(["books", "home", "elec", "food", "toys"], n_products)
            ),
            "unit_price": rng.uniform(1.0, 200.0, n_products).round(2),
        }
    )

    # --- transactions
    # 2 users get zero transactions — select their ids and exclude them.
    inactive = rng.choice(user_ids, size=2, replace=False)
    sampleable = np.setdiff1d(user_ids, inactive)
    txn_users = rng.choice(sampleable, size=n_txns)

    # ~1% reference an *unknown* user id (synthetic data-quality bug)
    bad = rng.random(n_txns) < 0.01
    txn_users = txn_users.astype(object)
    txn_users[bad] = [f"u_ghost_{i}" for i in range(bad.sum())]

    offsets_days = rng.integers(-365, 0, n_txns)
    # ~0.5% get a future timestamp (relative to as_of)
    future = rng.random(n_txns) < 0.005
    offsets_days[future] = rng.integers(1, 30, future.sum())
    ts = as_of + pd.to_timedelta(offsets_days, unit="D")
    # Add a random time-of-day so timestamps are not all midnight.
    ts = ts + pd.to_timedelta(rng.integers(0, 86_400, n_txns), unit="s")

    txns = pd.DataFrame(
        {
            "transaction_id": pd.array(
                [f"t{i:07d}" for i in range(n_txns)], dtype="string"
            ),
            "user_id": pd.array(txn_users, dtype="string"),
            "product_id": pd.array(rng.choice(product_ids, n_txns), dtype="string"),
            "quantity": pd.array(rng.integers(1, 6, n_txns), dtype="Int64"),
            "timestamp": ts,
        }
    )

    users_p = out_dir / "users.parquet"
    products_p = out_dir / "products.parquet"
    txns_p = out_dir / "transactions.parquet"
    users.to_parquet(users_p, index=False)
    products.to_parquet(products_p, index=False)
    txns.to_parquet(txns_p, index=False)
    return {"users": users_p, "products": products_p, "transactions": txns_p}


if __name__ == "__main__":
    generate()
```

Notes:

- `np.random.default_rng(seed)` — the modern Generator API. The
  legacy global `np.random.seed(...)` leaks state across tests and
  is explicitly discouraged in the
  [NumPy random guide](https://numpy.org/doc/stable/reference/random/index.html).
- `pd.StringDtype()` / `pd.Categorical` / `Int64` are the nullable-
  aware dtypes. They round-trip through Parquet cleanly.

### Part B — `numpy_warmups.py`

```python
# numpy_warmups.py
from __future__ import annotations
import numpy as np


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """1-D rolling mean of window ``w``.

    Returns an array of length ``len(x) - w + 1``. Uses cumsum —
    O(n) and avoids the kernel-flip cost of ``np.convolve``.
    """
    if w <= 0:
        raise ValueError("window must be positive")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if x.shape[0] < w:
        return np.empty(0, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[w:] - c[:-w]) / w


def pairwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(m, d), (n, d) -> (m, n) cosine-similarity matrix."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("A, B must be 2-D with matching last dim")
    an = np.linalg.norm(A, axis=1, keepdims=True)  # (m, 1)
    bn = np.linalg.norm(B, axis=1, keepdims=True)  # (n, 1)
    # Guard against zero-norm rows -> 0 similarity, not NaN.
    an = np.where(an == 0, 1.0, an)
    bn = np.where(bn == 0, 1.0, bn)
    return (A @ B.T) / (an * bn.T)


def topk_per_row(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-``k`` entries in each row, no full sort."""
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-D")
    if not (1 <= k <= scores.shape[1]):
        raise ValueError("k out of range")
    # argpartition is O(n) per row vs. O(n log n) for argsort.
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    # Sort those k indices by actual score, descending, for determinism.
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return part[rows, order]


def safe_log1p(x: np.ndarray) -> np.ndarray:
    """Like ``np.log1p`` but raises on any x < -1 (would produce NaN)."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < -1):
        raise ValueError("safe_log1p: encountered value < -1")
    return np.log1p(x)
```

Primitives chosen with their docs in mind:

- [`numpy.cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
  — the standard trick for O(n) rolling sums.
- [`numpy.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html)
  — "partial sort"; what you want any time you only need top-k.
- [Broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
  — the shape dance in `pairwise_cosine`.

### Part C — `pandas_features.py`

```python
# pandas_features.py
from __future__ import annotations
from pathlib import Path
from typing import TypedDict
import numpy as np
import pandas as pd


class Tables(TypedDict):
    users: pd.DataFrame
    products: pd.DataFrame
    transactions: pd.DataFrame


def load(data_dir: Path) -> Tables:
    """Load the three parquet files with explicit dtypes."""
    data_dir = Path(data_dir)
    users = pd.read_parquet(data_dir / "users.parquet").astype(
        {
            "user_id": "string",
            "country": "category",
            "age": "Int64",
        }
    )
    users["signup_date"] = pd.to_datetime(users["signup_date"])

    products = pd.read_parquet(data_dir / "products.parquet").astype(
        {
            "product_id": "string",
            "category": "category",
            "unit_price": "float64",
        }
    )

    txns = pd.read_parquet(data_dir / "transactions.parquet").astype(
        {
            "transaction_id": "string",
            "user_id": "string",
            "product_id": "string",
            "quantity": "Int64",
        }
    )
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    return {"users": users, "products": products, "transactions": txns}


def clean_transactions(
    transactions: pd.DataFrame, *, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop future-dated rows. Return cleaned frame + report dict."""
    assert transactions["timestamp"].dtype.kind == "M", "timestamp must be datetime"
    n_before = len(transactions)
    future = transactions["timestamp"] > as_of
    cleaned = transactions.loc[~future].copy()
    report = {
        "rows_in": int(n_before),
        "rows_out": int(len(cleaned)),
        "dropped_future_timestamp": int(future.sum()),
    }
    return cleaned, report


def enrich(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join users and products onto transactions with validate=."""
    # validate="many_to_one" => transactions.user_id may repeat;
    # users.user_id must be unique. Same for products.
    out = transactions.merge(
        users, on="user_id", how="left", validate="many_to_one"
    ).merge(products, on="product_id", how="left", validate="many_to_one")
    return out


def build_user_features(
    enriched: pd.DataFrame, users: pd.DataFrame, *, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Per-user features over the trailing 90-day window.

    Returns one row per user in ``users``, including users with zero
    transactions in the window (or no transactions at all, or whose
    only transactions were dropped as future-dated upstream). Those
    users keep ``n_transactions = 0`` and leave the basket / recency
    columns as NaN rather than disappearing from the output.
    """
    assert enriched["timestamp"].dtype.kind == "M"

    # Drop ghost transactions (user_id not in the users table) before we
    # aggregate — their joined signup_date is NaT. Real users with zero
    # transactions never reach `known` and are recovered by the left-join
    # against `users_slim` below.
    known = enriched.dropna(subset=["signup_date"]).copy()
    known["line_value"] = known["quantity"].astype("float64") * known["unit_price"]

    # 90-day window
    window_start = as_of - pd.Timedelta(days=90)
    w = known.loc[known["timestamp"].between(window_start, as_of)]

    agg = w.groupby("user_id", observed=True).agg(
        n_transactions=("transaction_id", "count"),
        total_spend_90d=("line_value", "sum"),
        mean_basket_90d=("line_value", "mean"),
        n_unique_categories_90d=("category", "nunique"),
    )

    # days-since-last-txn and last-txn use the full history, not just the window.
    last_txn = known.groupby("user_id", observed=True)["timestamp"].max().rename("last_ts")

    # One row per user in the full users table. Building this from `known`
    # would silently drop users with zero transactions.
    users_slim = (
        users[["user_id", "signup_date", "country", "age"]]
        .drop_duplicates(subset=["user_id"])
        .set_index("user_id")
    )

    out = users_slim.join(agg, how="left").join(last_txn, how="left")
    out["tenure_days"] = (as_of - out["signup_date"]).dt.days.astype("Int64")
    out["days_since_last_txn"] = (as_of - out["last_ts"]).dt.days.astype("Int64")

    # Fill count for users with zero in-window transactions; keep means as NaN.
    out["n_transactions"] = out["n_transactions"].fillna(0).astype("Int64")
    out["total_spend_90d"] = out["total_spend_90d"].fillna(0.0)
    # NOTE: mean_basket_90d intentionally left NaN when n_transactions == 0.
    out["n_unique_categories_90d"] = (
        out["n_unique_categories_90d"].fillna(0).astype("Int64")
    )

    cols = [
        "tenure_days",
        "n_transactions",
        "total_spend_90d",
        "mean_basket_90d",
        "n_unique_categories_90d",
        "days_since_last_txn",
        "country",
        "age",
    ]
    return out[cols]


def final_feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """Return (X, user_index).

    ``age`` is cast to float; missing values are *not* imputed here —
    imputation belongs in a downstream sklearn pipeline (see exercise-02).
    """
    num = features.drop(columns=["country"]).copy()
    num["age"] = num["age"].astype("Float64")
    X = num.to_numpy(dtype="float64", na_value=np.nan)
    return X, features.index
```

Why the two-stage join / window pattern (slice first, then
`groupby`): per
[pandas groupby semantics](https://pandas.pydata.org/docs/user_guide/groupby.html),
`agg` cannot filter rows — pre-filtering on `timestamp` keeps the
code honest and fast. `apply` over groups is explicitly avoided.

### Part D — tests

```python
# tests/test_numpy_warmups.py
import numpy as np
import pytest
from numpy_warmups import rolling_mean, pairwise_cosine, topk_per_row, safe_log1p


def test_rolling_mean_shape_and_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = rolling_mean(x, 2)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [1.5, 2.5, 3.5])


def test_pairwise_cosine_identity():
    A = np.eye(3)
    sim = pairwise_cosine(A, A)
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(sim, np.eye(3))


def test_topk_per_row_picks_top():
    s = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]])
    idx = topk_per_row(s, k=2)
    assert idx.shape == (2, 2)
    assert idx[0, 0] == 1  # top in row 0 is column 1 (0.9)
    assert idx[1, 0] == 0  # top in row 1 is column 0 (0.3)


def test_safe_log1p_raises_below_minus_one():
    with pytest.raises(ValueError):
        safe_log1p(np.array([0.0, -1.5]))
```

```python
# tests/test_features.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from generate_data import generate
from pandas_features import (
    load,
    clean_transactions,
    enrich,
    build_user_features,
    final_feature_matrix,
)

AS_OF = pd.Timestamp("2025-01-01")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(seed=0, out_dir=out)
    return load(out)


def test_load_schema(tables):
    assert set(tables) == {"users", "products", "transactions"}
    u = tables["users"]
    assert str(u["user_id"].dtype) == "string"
    assert str(u["age"].dtype) == "Int64"
    assert str(u["country"].dtype) == "category"
    assert u["signup_date"].dtype.kind == "M"


def test_clean_transactions_drops_future(tables):
    cleaned, report = clean_transactions(tables["transactions"], as_of=AS_OF)
    assert (cleaned["timestamp"] <= AS_OF).all()
    assert report["rows_in"] == report["rows_out"] + report["dropped_future_timestamp"]


def test_enrich_validate_catches_duplicate_user(tables):
    dup_users = pd.concat([tables["users"], tables["users"].head(1)], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        enrich(tables["transactions"], dup_users, tables["products"])


def test_build_user_features_contract(tables):
    cleaned, _ = clean_transactions(tables["transactions"], as_of=AS_OF)
    enriched = enrich(cleaned, tables["users"], tables["products"])
    feats = build_user_features(enriched, tables["users"], as_of=AS_OF)
    required = {
        "tenure_days",
        "n_transactions",
        "total_spend_90d",
        "mean_basket_90d",
        "n_unique_categories_90d",
        "days_since_last_txn",
        "country",
        "age",
    }
    assert required.issubset(feats.columns)
    assert (feats["tenure_days"].dropna() >= 0).all()
    # Full-table coverage: every users-row survives into feats, including
    # users with zero transactions and users whose only transactions were
    # dropped as future-dated upstream.
    assert len(feats) == len(tables["users"])
    # Inactive users keep zero counts and undefined means/recency — the
    # generator plants two, so at least two rows must look like this.
    zero_txn = feats["n_transactions"] == 0
    assert zero_txn.sum() >= 2
    assert feats.loc[zero_txn, "mean_basket_90d"].isna().all()
    assert feats.loc[zero_txn, "days_since_last_txn"].isna().all()


def test_final_matrix_is_floatlike(tables):
    cleaned, _ = clean_transactions(tables["transactions"], as_of=AS_OF)
    enriched = enrich(cleaned, tables["users"], tables["products"])
    feats = build_user_features(enriched, tables["users"], as_of=AS_OF)
    X, _ = final_feature_matrix(feats)
    assert X.dtype == np.float64
    assert X.ndim == 2
```

### `REPORT.md` — reference answers

- **Trickiest feature.** `n_unique_categories_90d`: category dtype
  interacts with `nunique` through `observed=`, and empty windows
  silently coerce to 0 unless handled explicitly. Fix: the `w`
  filter before `groupby` and an explicit `fillna(0).astype("Int64")`.
- **Did `validate=` catch anything?** Yes — in the first draft the
  generator could emit duplicate `product_id`s under some seeds. The
  `validate="many_to_one"` raise surfaced it immediately rather than
  silently fanning the transactions frame out.
- **First five rows of `feats`.** Printed with `feats.head().to_markdown()`
  and pasted below the answer. The purpose is to prove the dtypes
  survived the whole pipeline.

## 3. Validation steps

```bash
uv sync
uv run python generate_data.py          # writes data/*.parquet
uv run pytest -q                        # all tests green
```

Additional spot checks:

1. `grep -n "iterrows\|\.apply.*axis=1\|for .* in .*\.values" pandas_features.py numpy_warmups.py` — should return nothing.
2. `grep -n "validate=" pandas_features.py` — should return one hit per `merge` call.
3. `grep -n "np.random.seed" .` — should return nothing (use `default_rng` exclusively).
4. `feats.dtypes` in a REPL should show `Int64` (not `int64`) on the nullable columns and `category` on `country`.

## 4. Rubric or review checklist

| Dimension | Looking for | Weight |
| --- | --- | --- |
| Data generation | Uses `default_rng(seed)`; all four quality issues baked in with comments | 10 |
| NumPy primitives | All four correct, vectorised, with shape + value tests | 20 |
| Dtype discipline | Explicit dtype casts on `load`; survive the full pipeline into `final_feature_matrix` | 10 |
| No row loops | No `iterrows`, no `apply(..., axis=1)`, no `for ... in df.values` | 10 |
| Merge safety | `validate="many_to_one"` on every `merge`; documented why | 10 |
| Feature correctness | All seven features present; `tenure_days ≥ 0`; empty-window means are NaN, not 0 | 20 |
| Tests | ≥ 1 test per function; covers the `validate=` raise and the NaN-vs-0 distinction | 15 |
| Report | Answers the three prompts honestly and includes a head-of-frame preview | 5 |

Binary review checklist:

- [ ] `uv sync && uv run pytest -q` green on a fresh clone.
- [ ] Every `merge` has an explicit `validate=`.
- [ ] No `iterrows` / `apply(axis=1)` / `for row in df.values`.
- [ ] `mean_basket_90d` is NaN (not 0) when `n_transactions == 0`.
- [ ] `tenure_days` is non-negative.
- [ ] `final_feature_matrix` returns a `float64` NumPy array (no object dtype).

## 5. Common mistakes

1. **Using `df.apply(fn, axis=1)` for the per-user features.** Readable
   but 10–100× slower than a `groupby` + named aggregation, and it
   silently coerces dtypes. The brief explicitly forbids it.
2. **Omitting `validate=` on merges.** The default is `None`, which
   means a bad right-hand key silently fans rows out. Downstream
   totals double, metrics drift, root-cause takes a day.
3. **Filling empty-window means with zero.** A zero basket size and
   an undefined basket size are different claims. Models trained on
   the fused signal underfit new users.
4. **Casting `age` with `astype(float)` while NaN still present.**
   Works, but leaks silent NaNs. Use `Float64` or `Int64` extension
   dtypes until the imputation step.
5. **Using `np.random.seed` / `np.random.rand`.** Legacy global state.
   NumPy's own
   [random guide](https://numpy.org/doc/stable/reference/random/index.html)
   recommends `default_rng(seed)` for reproducibility across tests.
6. **`np.sort` + slice for top-k.** O(n log n) per row; the brief asks
   for `np.argpartition` (O(n)) specifically because this comes up on
   recommender rank-k operations where n is huge.
7. **Writing `category` as `object` in Parquet.** Lose the category
   dtype round-trip. Fix: keep it as `pd.Categorical` on write; it
   serialises fine in modern Pandas.

## 6. References

Local exercise context:

- Learning exercise —
  <https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-01-numpy-pandas-fluency.md>
- Companion chapters — `02-numpy-for-ml.md`, `03-pandas-for-ml.md` in
  the learning repo.

Official project documentation:

- NumPy user guide — <https://numpy.org/doc/stable/user/index.html>
- NumPy random Generator API —
  <https://numpy.org/doc/stable/reference/random/index.html>
- NumPy broadcasting —
  <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- `numpy.argpartition` —
  <https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html>
- Pandas user guide — <https://pandas.pydata.org/docs/user_guide/>
- Pandas `merge` / `validate=` —
  <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html>
- Pandas groupby —
  <https://pandas.pydata.org/docs/user_guide/groupby.html>
- Pandas nullable integer dtype —
  <https://pandas.pydata.org/docs/user_guide/integer_na.html>
- pytest fixtures —
  <https://docs.pytest.org/en/stable/explanation/fixtures.html>
