# SOLUTION — Exercise 03: Reproducible environment setup

> Reference solution for
> [`lessons/mod-102-python-ml-toolchain/exercises/exercise-03-reproducible-env-setup.md`](https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-03-reproducible-env-setup.md).
> Read it only after attempting your own submission.

## 1. Solution overview

The exercise is an **infrastructure drill**: produce the envelope
around a trivial Python package so that any downstream project in
this curriculum rebuilds bit-for-bit the same environment, twice —
once locally, once in a container (or a second machine) — and so
that CI refuses to install a drifted lockfile.

The reference solution uses **`uv`** (Astral's packaging toolchain).
`poetry` is a fully acceptable alternative and the shape of the files
is the same. The deliverable looks like:

```
ml_toolkit/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── Dockerfile
├── README.md
├── REPORT.md
├── .github/workflows/ci.yml
├── src/ml_toolkit/
│   ├── __init__.py
│   └── features.py
└── tests/
    └── test_features.py
```

Three non-obvious decisions:

- **`--locked` is the CI contract.** `uv sync` without a flag will
  *silently update* the lockfile; `uv sync --frozen` goes the other
  way and installs from whatever `uv.lock` happens to say even when
  `pyproject.toml` has drifted away from it. `uv sync --locked`
  refuses to proceed when the lock is stale, which is the behaviour
  CI needs. `uv run` is passed `--locked` too, so a later step cannot
  quietly re-resolve. See the
  [uv lockfile docs](https://docs.astral.sh/uv/concepts/projects/sync/#syncing-the-environment).
- **Pin the interpreter minor version, not just major.** `.python-version`
  contains `3.11` (or `3.12`). Pinning to a range (e.g. `>=3.11`)
  guarantees you *will* get a different binary on some CI runner
  someday, with no code change. See the
  [uv Python guide](https://docs.astral.sh/uv/guides/install-python/).
- **`src/` layout, not flat.** Importing `ml_toolkit` from the test
  suite then only resolves against the installed package, not the
  working tree — which is the whole point of a packaged project
  (and is the recommendation from
  [PyPA packaging guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)).

## 2. Worked answer or implementation

### `pyproject.toml`

```toml
[project]
name = "ml-toolkit"
version = "0.1.0"
description = "Tiny ML utilities for the mod-102 reproducibility exercise."
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "ruff>=0.4",
    "mypy>=1.9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ml_toolkit"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src", "tests"]
```

Build it with the tool, not by hand:

```bash
uv init --lib ml_toolkit
cd ml_toolkit
uv add 'numpy>=1.26' 'pandas>=2.1' 'scikit-learn>=1.4'
uv add --dev 'pytest>=8' 'ruff>=0.4' 'mypy>=1.9'
```

### `.python-version`

A one-line file. No quotes. No comments.

```
3.11
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/

# Tooling caches
.venv/
.mypy_cache/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/

# OS / editor
.DS_Store
.idea/
.vscode/
```

### `src/ml_toolkit/features.py`

```python
# src/ml_toolkit/features.py
from __future__ import annotations
import numpy as np
import numpy.typing as npt


def standardise(X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Zero-mean, unit-variance each column of X."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std
```

### `tests/test_features.py`

```python
# tests/test_features.py
import numpy as np
from ml_toolkit.features import standardise


def test_standardise_zero_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(1000, 3))
    Z = standardise(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-8)


def test_standardise_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(1000, 3))
    Z = standardise(X)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-8)
```

### `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# Install uv from the official distroless image (pinned digest in a real project).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency contract first for layer caching.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Now copy the source and install the project itself.
COPY src ./src
COPY tests ./tests
RUN uv sync --frozen

CMD ["uv", "run", "pytest", "-q"]
```

Build + run + diff:

```bash
docker build -t ml_toolkit:test .
docker run --rm ml_toolkit:test                       # tests pass inside
docker run --rm ml_toolkit:test uv pip freeze > /tmp/docker.txt
uv pip freeze > /tmp/local.txt
diff /tmp/local.txt /tmp/docker.txt                   # expect no output
```

### `.github/workflows/ci.yml`

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

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Install Python pinned in .python-version
        run: uv python install

      - name: Install dependencies (fail if lockfile drifts)
        run: uv sync --locked

      - name: Lint
        run: uv run --locked ruff check .

      - name: Type-check
        run: uv run --locked mypy src tests

      - name: Tests
        run: uv run --locked pytest -q
```

### The deliberate-break demo

Edit one line of `pyproject.toml` by hand — bump `numpy>=1.26` to
`numpy>=2.0` without regenerating the lockfile — commit, push. CI
fails at `uv sync --locked` with a message of roughly:

```
error: The lockfile at `uv.lock` needs to be updated, but `--locked`
was provided. To update the lockfile, run `uv lock`.
```

That is the enforcement working as intended. Revert the edit; CI
goes green again. Paste the failing-run URL into `REPORT.md`.

### `REPORT.md` — reference answers

- **Interpreter choice.** 3.11, because it is the oldest version
  still receiving active security fixes at the time of writing and
  the whole curriculum targets it. (3.12 is an equally defensible
  choice; the rubric only cares that a *version* is pinned.)
- **`uv pip freeze` local vs. container.** Identical. (If you see a
  diff on `pywin32` or platform markers, you are running the local
  freeze on macOS/Linux vs. a Linux container — that's expected and
  `uv.lock` records the per-platform resolution.)
- **Lockfile drift CI run.** Link / screenshot of the failing run,
  with the `--locked` error message annotated.

## 3. Validation steps

```bash
# Rebuild from scratch, no cache.
rm -rf .venv
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked mypy src tests

# Container path.
docker build -t ml_toolkit:test .
docker run --rm ml_toolkit:test

# Drift-detection sanity: temporary hand-edit must fail --locked.
sed -i.bak 's/numpy>=1.26/numpy>=2.0/' pyproject.toml
uv sync --locked && echo "LOCKFILE DRIFT NOT CAUGHT" || echo "ok: locked refused"
mv pyproject.toml.bak pyproject.toml
```

Expected: every command exits zero on the happy path; the drift
block prints `ok: locked refused`.

Additional spot checks:

1. `git ls-files` includes `pyproject.toml`, `uv.lock`,
   `.python-version`, `.gitignore`, `Dockerfile`, `.github/workflows/ci.yml`,
   and the `src/`, `tests/` trees — and nothing in `.venv/`,
   `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`.
2. `grep -rn "pip install" .` returns no hits in the source tree or
   scripts — every dependency change went through `uv add`.
3. `cat .python-version` is a single line, no quotes.
4. `uv lock --check` exits zero.

## 4. Rubric or review checklist

| Dimension | Looking for | Weight |
| --- | --- | --- |
| Project scaffolding | `src/` layout; package imports from `ml_toolkit` | 10 |
| Dependency pinning | `pyproject.toml` + lockfile committed; `.python-version` single line | 15 |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `dist/`, `*.egg-info/` | 10 |
| Test / lint / type baseline | `pytest`, `ruff`, `mypy` all green on fresh sync | 15 |
| Dockerfile | Starts from slim image, installs uv, copies lockfile, `uv sync --frozen`, runs tests | 15 |
| Reproducibility proof | `uv pip freeze` local == container (or justified diff) | 10 |
| CI happy path | Workflow runs sync + lint + mypy + pytest; passes on main | 15 |
| CI enforcement | Hand-bumped dep without lockfile regenerate fails `uv sync --locked`; failing run documented | 10 |

Binary review checklist:

- [ ] `pyproject.toml`, lockfile, `.python-version`, `.gitignore` all committed.
- [ ] `.venv/` is not committed.
- [ ] `uv sync --locked && uv run --locked pytest -q` green on a fresh clone.
- [ ] Container build produces an installed-package list identical to the local one.
- [ ] CI runs on push *and* PR.
- [ ] CI fails on lockfile drift, with the error message captured in `REPORT.md`.
- [ ] No `pip install …` calls appear in shell history / scripts.

## 5. Common mistakes

1. **Committing `.venv/`.** The `.gitignore` is present but the
   developer `git add .`'d before adding it. `git rm -r --cached
   .venv` and re-commit.
2. **`uv sync` (or `uv sync --frozen`) in CI instead of `uv sync
   --locked`.** Bare `uv sync` silently rewrites `uv.lock`;
   `--frozen` goes the other way and installs from a stale lock
   without complaining. Only `--locked` actually fails CI when
   `pyproject.toml` has drifted from `uv.lock`. Pair it with
   `uv run --locked` so downstream steps cannot re-resolve either.
3. **Pinning Python as `>=3.11`.** A range, not a pin. CI will pick
   whichever is default; six months later your dev box and CI runner
   diverge. Use a single minor in `.python-version`.
4. **No `src/` layout.** Imports resolve against the working tree,
   not the installed package. Tests pass even if the package is
   broken. PyPA's layout-discussion doc explains this at length.
5. **`pip install` sneaking in.** Any `pip install foo` call made to
   "quickly try something" bypasses the lockfile and leaves a
   package in the venv that is not recorded anywhere. Delete `.venv`
   and re-sync.
6. **Docker's `COPY . .` first.** Every source edit invalidates the
   dependency layer cache. Copy the lockfile first, `uv sync`, then
   copy the source.
7. **`uv run` on the host but bare `pytest` in the container.** Means
   the container is running against the system Python, not the
   project's venv. Always use `uv run` (or activate the venv).
8. **Forgetting `uv python install` in CI.** On GitHub-hosted runners
   the pinned minor may not be pre-installed; `uv python install`
   fetches it according to `.python-version`.

## 6. References

Local exercise context:

- Learning exercise —
  <https://github.com/ai-infra-curriculum/ml-engineer-learning/blob/main/lessons/mod-102-python-ml-toolchain/exercises/exercise-03-reproducible-env-setup.md>
- Companion chapter — `07-reproducible-python-environments.md`.

Official project / standard documentation:

- Python Packaging User Guide (src vs. flat layout) —
  <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>
- `pyproject.toml` specification (PEP 621) —
  <https://packaging.python.org/en/latest/specifications/pyproject-toml/>
- uv — <https://docs.astral.sh/uv/>
- uv project workflow —
  <https://docs.astral.sh/uv/concepts/projects/>
- uv lockfile / `uv sync --locked` —
  <https://docs.astral.sh/uv/concepts/projects/sync/>
- uv Python installer —
  <https://docs.astral.sh/uv/guides/install-python/>
- Poetry (alternative) — <https://python-poetry.org/docs/>
- GitHub Actions — <https://docs.github.com/en/actions>
- `astral-sh/setup-uv` action —
  <https://github.com/astral-sh/setup-uv>
- Docker best practices (layer caching) —
  <https://docs.docker.com/develop/develop-images/dockerfile_best-practices/>
- ruff — <https://docs.astral.sh/ruff/>
- mypy — <https://mypy.readthedocs.io/en/stable/>
- pytest — <https://docs.pytest.org/en/stable/>
