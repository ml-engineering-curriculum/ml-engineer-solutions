# exercise-02 — Data and Model Versioning · SOLUTION

Paired exercise: `lessons/mod-106-experiment-tracking-reproducibility/exercises/exercise-02-data-and-model-versioning.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to wire up a training workflow where every model artifact is linked to a specific data version and a specific code version. "Linked" means: given a model ID, a reviewer can — without asking the author — recover the exact dataset bytes and git SHA used to train it.

The reference solution uses:

- **[DVC](https://dvc.org/doc)** for data versioning (content-addressed storage with a pointer file committed to git).
- **MLflow Model Registry** for model versioning (stage transitions, aliases, lineage back to the training run).
- **MLflow tags** as the join — the training run carries `data.dvc_sha`, `data.path`, and `git.sha` tags; the registered model points at that run.

Design priorities, in order:

1. **Data lives in object storage, not git.** Git refuses to be a blob store above ~100 MB; DVC puts the blob in S3/GCS/Azure and leaves a tiny `.dvc` pointer behind. The pointer *is* the version.
2. **The model registry is append-only.** Model versions are never edited; new training produces a new version, and stage transitions (`Staging` → `Production`) are recorded as events. This is the OWASP ML Top 10 `ML05:2023 Model Theft` and `ML06:2023 AI Supply Chain Attacks` control surface — the registry is where "which model is in prod, trained from what?" gets answered under incident response.
3. **One source of truth per asset class.** Data = DVC. Model = MLflow Registry. Code = git. Overlapping systems (e.g., also tracking data in the MLflow artifact store) produce drift — someone will update one and not the other.
4. **Aliases, not hard-coded version numbers.** Downstream services ask the registry for `models:/churn/champion`, not `models:/churn/14`. Promotions update the alias atomically; rollbacks flip it back without a code deploy.
5. **Lineage must survive deletion.** When data or code is deleted (GDPR/retention), the *link* must survive as a tombstoned pointer — otherwise audit asks "what trained v14?" and nobody can answer. This aligns with NIST AI RMF `MAP 1.6` / `MANAGE 2.1` on documented provenance.

| Choice | Alternative | Why this one |
|---|---|---|
| DVC with S3 remote | git-LFS | LFS is fine for a few files; DVC's pipeline DAG + stage-aware caching scales with the project |
| MLflow Registry | Hand-rolled "latest" symlink / database | Registry has stage transitions, aliases, and lineage as first-class concepts |
| Aliases (`@champion`, `@challenger`) over stages | Hard-coded version numbers | Flip prod traffic without a deploy; A/B via two aliases |
| Semantic version on registered model | Monotonic integer only | MLflow provides integers automatically; add a semver tag for human-readable release notes |
| Data hash recomputed at train time | Trust the `.dvc` file alone | Catches the case where someone edited the file in place without `dvc add` |
| Model signature stored with every version | Signature only on "important" versions | Serving loads from the registry at random; every version must be servable |

## 2. Worked implementation

### 2.1 Repository layout

```
repo/
├── data/
│   ├── raw/
│   │   └── orders.parquet         # not in git; tracked by DVC
│   └── raw/orders.parquet.dvc     # tiny pointer; in git
├── dvc.yaml                       # pipeline stages
├── dvc.lock                       # content hashes of inputs and outputs
├── .dvc/config                    # remote (S3/GCS) config
├── pipelines/
│   ├── build_features.py
│   └── train_and_register.py
└── .gitignore                     # /data/raw/orders.parquet
```

### 2.2 DVC — `dvc.yaml`

```yaml
stages:
  features:
    cmd: python -m pipelines.build_features
    deps:
      - data/raw/orders.parquet
      - pipelines/build_features.py
    outs:
      - data/features/orders.parquet

  train:
    cmd: python -m pipelines.train_and_register
    deps:
      - data/features/orders.parquet
      - pipelines/train_and_register.py
    params:
      - params.yaml:
          - train
    outs:
      - models/latest/MLmodel:
          cache: false        # the real artifact lives in MLflow, not DVC
    metrics:
      - metrics.json:
          cache: false
```

Why this shape:

- `deps:` is the input contract. Change a dep's content and `dvc repro` reruns the stage; leave it alone and the stage is a cache hit.
- `params:` is the *knob* contract. Change `params.yaml → train.learning_rate` and only the `train` stage reruns.
- `cache: false` on the MLmodel file is deliberate: the MLflow server owns the real artifact; this file is a placeholder DVC uses to know the stage completed.

### 2.3 `params.yaml` — single source of hyperparameters

```yaml
train:
  n_estimators: 200
  max_depth: 3
  learning_rate: 0.05
  random_state: 42

register:
  model_name: churn-baseline
  min_val_auc: 0.80          # gate: do not register a weaker model
```

A hyperparameter changed in `params.yaml` is picked up by DVC, by MLflow (`log_params`), and by the registration gate — all from one file. Hyperparameters duplicated across code, config, and DVC are the root cause of "I thought I turned that off."

### 2.4 Reading the DVC pointer from Python

```python
# pipelines/versioning.py
import hashlib
import subprocess
from pathlib import Path
import yaml


def dvc_pointer(path: Path) -> dict:
    """Return the committed DVC pointer for a data file."""
    pointer = Path(f"{path}.dvc")
    with pointer.open() as f:
        meta = yaml.safe_load(f)
    out = meta["outs"][0]
    return {
        "path": out["path"],
        "md5": out["md5"],
        "size": out["size"],
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    sha = out.stdout.strip()
    return f"{sha}-dirty" if dirty else sha
```

The DVC pointer stores an MD5 (DVC's internal content hash). Recomputing a local SHA-256 at train time is cheap and catches the case where somebody edited the file in place and forgot to run `dvc add`.

### 2.5 Training + registration

```python
# pipelines/train_and_register.py
import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .versioning import dvc_pointer, file_sha256, git_sha


DATA = Path("data/features/orders.parquet")
PARAMS_PATH = Path("params.yaml")


def main() -> None:
    cfg = yaml.safe_load(PARAMS_PATH.read_text())
    tp = cfg["train"]
    rp = cfg["register"]

    df = pd.read_parquet(DATA)
    y = df["churned_90d"].astype("int8")
    x = df.drop(columns=["churned_90d"])
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=tp["random_state"],
    )

    model = GradientBoostingClassifier(
        n_estimators=tp["n_estimators"],
        max_depth=tp["max_depth"],
        learning_rate=tp["learning_rate"],
        random_state=tp["random_state"],
    )
    model.fit(x_tr, y_tr)
    val_proba = model.predict_proba(x_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))

    mlflow.set_experiment(rp["model_name"])
    with mlflow.start_run() as active:
        pointer = dvc_pointer(Path("data/raw/orders.parquet"))
        mlflow.set_tags({
            "git.sha": git_sha(),
            "data.path": pointer["path"],
            "data.dvc_md5": pointer["md5"],
            "data.size_bytes": str(pointer["size"]),
            "features.sha256": file_sha256(DATA),
        })
        mlflow.log_params(tp)
        mlflow.log_metric("val_auc", val_auc)

        signature = infer_signature(x_tr.head(5), y_tr.head(5))
        info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_tr.head(5),
        )

        Path("metrics.json").write_text(json.dumps({"val_auc": val_auc}))

        if val_auc < rp["min_val_auc"]:
            print(f"val_auc={val_auc:.4f} < gate={rp['min_val_auc']}; not registering")
            return

        client = MlflowClient()
        try:
            client.create_registered_model(rp["model_name"])
        except mlflow.exceptions.RestException:
            pass  # already exists

        mv = client.create_model_version(
            name=rp["model_name"],
            source=f"runs:/{active.info.run_id}/model",
            run_id=active.info.run_id,
        )
        client.set_model_version_tag(
            rp["model_name"], mv.version, "data.dvc_md5", pointer["md5"],
        )
        client.set_model_version_tag(
            rp["model_name"], mv.version, "git.sha", git_sha(),
        )
        client.set_registered_model_alias(
            rp["model_name"], "challenger", mv.version,
        )
        print(f"registered {rp['model_name']} v{mv.version} as @challenger")


if __name__ == "__main__":
    main()
```

The gate (`if val_auc < rp["min_val_auc"]: return`) is the single most valuable line in this file. Without it, a broken training run (data bug, label leak fix that collapses AUC to 0.5) still produces a "latest" model version that a careless `@champion` promotion could ship to prod. With it, the registry only contains models that cleared a minimum bar.

### 2.6 Loading by alias at inference time

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/churn-baseline@champion")
predictions = model.predict(input_df)
```

The serving code never names a version number. Promotion is:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("churn-baseline", "champion", version="14")
```

That one call shifts all `@champion` traffic. A rollback is the same call with the old version. The downstream service — load balancer, batch scorer, whatever — restarts or pulls on its own cadence; the registry does not need to coordinate with it.

### 2.7 Reproducing a specific model version

```bash
# Find the lineage
mlflow runs search --filter-string "tags.mlflow.modelVersion = '14'"
# -> gives run_id R

# Check out the code
git checkout $(mlflow runs describe --run-id R --output-format json | jq -r '.data.tags."git.sha"')

# Check out the data
dvc pull data/raw/orders.parquet

# Re-run the pipeline
dvc repro train
```

The whole point of the exercise is that those four steps work without the model author in the room. If step 2 or 3 requires tribal knowledge, the lineage has failed.

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **DVC init sanity** — `dvc status`. Must report "pipelines up to date" or name specific stale stages. An error here means the DVC remote is not reachable, which cascades into every downstream step.
2. **Round-trip a data version** — `dvc add data/raw/orders.parquet && dvc push && rm data/raw/orders.parquet && dvc pull`. The restored file must match the pointer's MD5. Catches a broken remote config.
3. **Clean run registers** — on a clean tree with `val_auc ≥ gate`, running `dvc repro train` must (a) produce `models/latest/MLmodel`, (b) produce a new MLflow run, (c) register a new model version, (d) set the `@challenger` alias. Verify in the UI.
4. **Gate blocks a weak run** — temporarily set `train.learning_rate: 0.0`, rerun. The run must land in MLflow with `val_auc` logged, but **no new registered model version** should be created.
5. **Lineage lookup** — pick a registered version; from its tags, recover the git SHA and `data.dvc_md5`. Check out that SHA, `dvc pull`, and rerun. The new run's `val_auc` must match the original within tolerance.
6. **Dirty tree is marked** — make an uncommitted change, rerun the pipeline. The new registered version's `git.sha` tag must end in `-dirty`. Dirty-tree runs are allowed (sometimes necessary) but must be *marked* so reviewers can filter them out.
7. **Alias flip** — `set_registered_model_alias("churn-baseline", "champion", "14")`. A serving process loading `models:/churn-baseline@champion` must pick up v14 on next load without a code change.
8. **Negative: in-place edit** — edit `data/raw/orders.parquet` bytes without `dvc add`. Running the pipeline must either (a) fail at the feature stage if `dvc repro` is gatekeeping, or (b) at minimum, log a `features.sha256` on the training run that differs from a reviewer's recomputation.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Data not in git | 2 | Raw data file is in `.gitignore`; only the `.dvc` pointer is committed |
| 2 | Pipeline DAG | 2 | `dvc.yaml` declares `deps`, `outs`, and `params` for each stage; `dvc repro` is a no-op on a clean tree |
| 3 | Params as single source | 2 | Hyperparameters live in `params.yaml` and are read by both DVC and the training script |
| 4 | Model Registry use | 3 | Model is registered via `create_model_version`; versions accrue on re-registration; UI shows lineage back to the training run |
| 5 | Registration gate | 3 | A run below the metric threshold logs to MLflow but does **not** create a registered version |
| 6 | Lineage tags on the version | 2 | Registered model version carries `data.dvc_md5` and `git.sha` tags, not just the run |
| 7 | Aliases for promotion | 2 | Serving code loads `models:/<name>@champion`, not a hard-coded version number; promotion is an alias flip |
| 8 | Signature on every version | 2 | Every registered version has a usable `signature` and `input_example`; `pyfunc.load_model` + `predict` round-trips |
| 9 | Documented reproduction path | 2 | A README snippet / runbook describes the 4-step "recover a specific version" path and it works end to end |

Items 4 and 5 are mandatory. A submission that registers *every* run — including the broken ones — has built a graveyard, not a registry, and cannot score above 13.

## 5. Common mistakes

- **Committing data to git.** Works until the file exceeds 100 MB; then your history is corrupted, and `git clone` takes an hour. Use DVC (or git-LFS) from the start.
- **Letting the DVC pointer drift from the file.** Someone edits `data/raw/orders.parquet` in place, forgets `dvc add`, pushes. The next pull returns the old content; training silently uses the *right* bytes locally and the *wrong* bytes in CI. Mitigate by rehashing the actual file at train time and logging it as a run tag (`features.sha256` in the code above).
- **Registering every run, including failures.** The registry becomes a landfill. Gate on at least one metric threshold.
- **Hard-coding model versions in serving code.** Promotions now require a deploy, and rollbacks require a deploy *under incident pressure*. Use aliases.
- **One alias per environment, no "challenger" alias.** A single `production` alias makes A/B testing by alias impossible. Keep `champion` and `challenger` at minimum.
- **Registering without a signature.** The registry happily hosts an unsigned model; serving loads it and accepts any input shape at all. The next step after `log_model` is *always* `signature=`.
- **Deleting old runs to "clean up."** The run is the only place `data.dvc_md5` lives for that version. Once deleted, a registered model points at a 404. Treat runs as append-only for anything that got registered.
- **Mixing data in both DVC and MLflow artifacts.** Two systems claim to be the source of truth; they drift; audit cannot answer "which bytes trained v14?" Pick one — DVC for data, MLflow for models.
- **No "dirty" marker on uncommitted runs.** The git SHA alone is not enough; a learner must detect and announce the dirty state, or all reproducibility claims have a loophole.

## 6. References

- DVC — Data versioning, pipelines (`dvc.yaml`), and the remote storage model. <https://dvc.org/doc>
- MLflow — Model Registry concepts, aliases (introduced in MLflow 2.x) vs. legacy stage transitions. <https://mlflow.org/docs/latest/model-registry.html>
- NIST AI Risk Management Framework 1.0 — `MAP 1.6` on documented data provenance and `MANAGE 2.1` on retained lineage through the model lifecycle. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML05:2023 Model Theft` (registry access control) and `ML06:2023 AI Supply Chain Attacks` (code + data provenance as the audit trail under compromise). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` and `AML.T0018 Backdoor ML Model` for the threat model that model versioning is defending against. <https://atlas.mitre.org/>
