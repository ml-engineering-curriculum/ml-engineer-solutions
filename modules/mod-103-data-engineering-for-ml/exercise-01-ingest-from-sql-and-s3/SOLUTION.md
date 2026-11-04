# exercise-01 — Ingest From SQL and S3 · SOLUTION

Paired exercise: `lessons/mod-103-data-engineering-for-ml/exercises/exercise-01-ingest-from-sql-and-s3.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to build a Python ingestion step that pulls two sources into a single training-ready parquet dataset:

- a transactional **SQL** source (orders, customers) via SQLAlchemy
- a semi-structured **S3** source (JSON event logs) via `boto3` / `s3fs`

The reference solution is structured so that each source is a **pure extract function** returning a `pandas.DataFrame`, and a top-level `run_ingest` function joins and writes a partitioned parquet dataset. Design priorities, in order:

1. **Determinism and replay** — every run writes to a dated partition and records an ingest manifest (row counts, source watermark, content hash).
2. **Incremental, not full** — SQL extract is watermarked on `updated_at`; S3 extract filters by object prefix date.
3. **Fail loud on upstream shape change** — column lists and dtypes are asserted before write, so a downstream training job never silently sees a renamed column.
4. **Credentials via environment, never code** — align with the NIST AI RMF `MAP 1.6` / `MANAGE 2.1` emphasis on documented, auditable data provenance (NIST AI RMF 1.0).

Decision rationale summary:

| Choice | Alternative considered | Why this one |
|---|---|---|
| SQLAlchemy Core + server-side cursor | ORM query | Streams rows for tables larger than RAM; ORM materialises Python objects we never need |
| `pandas.read_sql` with `chunksize` | `read_sql` single-shot | Bounded memory on tables > a few hundred MB |
| Parquet output, partitioned by `ingest_date` | CSV | Preserves dtypes, column-pruned reads, 5–10× smaller on typical tabular data |
| Watermark table in the warehouse, not a local file | Local JSON | A local file disappears with the container; a watermark row survives redeploy |
| Explicit schema assertion before write | Implicit (hope for the best) | Catches upstream rename/drop at ingest, not at train time (relates to OWASP ML Top 10 `ML02:2023 Data Poisoning` — unvalidated inputs are the attack surface) |

## 2. Worked implementation

### 2.1 Project layout

```
ingest/
├── __init__.py
├── config.py          # env parsing, no secrets in code
├── sources/
│   ├── sql_orders.py  # SQL extract
│   └── s3_events.py   # S3 extract
├── schema.py          # expected columns and dtypes
├── manifest.py        # write ingest manifest
└── run.py             # orchestration entry point
```

### 2.2 `config.py`

```python
import os
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class IngestConfig:
    sql_url: str
    s3_bucket: str
    s3_prefix: str
    output_root: str
    ingest_date: date

    @classmethod
    def from_env(cls) -> "IngestConfig":
        return cls(
            sql_url=os.environ["INGEST_SQL_URL"],
            s3_bucket=os.environ["INGEST_S3_BUCKET"],
            s3_prefix=os.environ.get("INGEST_S3_PREFIX", "events/"),
            output_root=os.environ.get("INGEST_OUTPUT_ROOT", "./data/warehouse"),
            ingest_date=date.fromisoformat(os.environ["INGEST_DATE"]),
        )
```

Secrets stay in the environment (or a secret manager). The dataclass is frozen so no downstream code can mutate the config mid-run — this is the single, auditable record of what ran.

### 2.3 SQL extract — streamed, watermarked

```python
# ingest/sources/sql_orders.py
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

ORDERS_SQL = text(
    """
    SELECT  order_id,
            customer_id,
            order_total_cents,
            currency,
            status,
            created_at,
            updated_at
    FROM    orders
    WHERE   updated_at >= :since
      AND   updated_at <  :until
    ORDER BY updated_at
    """
)


def extract_orders(sql_url: str, since: datetime, until: datetime) -> pd.DataFrame:
    engine = create_engine(sql_url, pool_pre_ping=True)
    with engine.connect().execution_options(stream_results=True) as conn:
        chunks = pd.read_sql(
            ORDERS_SQL,
            conn,
            params={"since": since, "until": until},
            chunksize=50_000,
        )
        df = pd.concat(chunks, ignore_index=True)

    df["order_total_usd"] = (df["order_total_cents"] / 100.0).astype("float64")
    df = df.drop(columns=["order_total_cents"])
    return df
```

Key points, each tied to a predictable failure mode:

- `stream_results=True` + `chunksize` means a 50M-row table does not OOM the pod.
- Watermark is **half-open** `[since, until)` so a row updated exactly at `until` is picked up by the *next* run, never twice and never zero times.
- Monetary columns stay integer cents at the DB layer; the float conversion is explicit and local. This avoids the classic "sum of floats drifts from sum of cents" bug.

### 2.4 S3 extract — partition-aware

```python
# ingest/sources/s3_events.py
import json
from datetime import date
import boto3
import pandas as pd


def _object_keys(bucket: str, prefix: str, on_date: date) -> list[str]:
    client = boto3.client("s3")
    full_prefix = f"{prefix.rstrip('/')}/dt={on_date.isoformat()}/"
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl"):
                keys.append(obj["Key"])
    return sorted(keys)


def extract_events(bucket: str, prefix: str, on_date: date) -> pd.DataFrame:
    client = boto3.client("s3")
    rows: list[dict] = []
    for key in _object_keys(bucket, prefix, on_date):
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        for line in body.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame.from_records(rows)
    return df
```

Why this shape:

- The paginator avoids the silent cap at 1,000 keys you get from a bare `list_objects_v2` call.
- Sorting keys gives a stable read order — handy when diffing two ingest runs during an investigation.
- The partition prefix (`dt=YYYY-MM-DD`) is the contract between the producer and this ingest; it makes "pull just today" an O(1) list operation, not a full-bucket scan.

### 2.5 Schema assertion

```python
# ingest/schema.py
import pandas as pd

ORDERS_SCHEMA: dict[str, str] = {
    "order_id": "int64",
    "customer_id": "int64",
    "currency": "string",
    "status": "string",
    "created_at": "datetime64[ns, UTC]",
    "updated_at": "datetime64[ns, UTC]",
    "order_total_usd": "float64",
}


def assert_schema(df: pd.DataFrame, expected: dict[str, str], name: str) -> None:
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    if missing or extra:
        raise ValueError(
            f"{name}: schema drift. missing={sorted(missing)} extra={sorted(extra)}"
        )
    for col, dtype in expected.items():
        actual = str(df[col].dtype)
        if actual != dtype:
            raise TypeError(f"{name}.{col}: expected {dtype}, got {actual}")
```

A learner will be tempted to skip this. Don't — this guard is the only thing between an upstream rename (`order_total` → `total_amount`) and a silent "all training examples now have NaN label" incident. It is the ingest-time expression of OWASP ML Top 10 `ML02:2023 Data Poisoning`: the control is "validate inputs before they reach training," and schema drift is the lowest-cost attack vector (intentional or accidental).

### 2.6 Manifest

```python
# ingest/manifest.py
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def write_manifest(
    df: pd.DataFrame, path: Path, source: str, watermark: str
) -> dict:
    content = df.to_parquet(index=False)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "source": source,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "watermark": watermark,
        "sha256": digest,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
```

Every partition gets a sibling `_manifest.json`. This is the artifact the training job reads to decide "is this the version I tested against?" — a prerequisite for the model-card data provenance expected by NIST AI RMF `MAP 1.6`.

### 2.7 Orchestrator

```python
# ingest/run.py
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from .config import IngestConfig
from .manifest import write_manifest
from .schema import ORDERS_SCHEMA, assert_schema
from .sources.s3_events import extract_events
from .sources.sql_orders import extract_orders


def run_ingest(cfg: IngestConfig) -> None:
    day_start = datetime.combine(cfg.ingest_date, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    orders = extract_orders(cfg.sql_url, since=day_start, until=day_end)
    assert_schema(orders, ORDERS_SCHEMA, "orders")

    events = extract_events(cfg.s3_bucket, cfg.s3_prefix, cfg.ingest_date)

    joined = orders.merge(
        events.groupby("order_id", as_index=False).agg(
            event_count=("event_id", "count"),
            first_event_at=("event_at", "min"),
        ),
        how="left",
        on="order_id",
        validate="one_to_one",
    )

    out_dir = Path(cfg.output_root) / f"ingest_date={cfg.ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out_dir / "orders_enriched.parquet", index=False)

    write_manifest(
        joined,
        out_dir / "_manifest.json",
        source="sql+s3",
        watermark=day_end.isoformat(),
    )


if __name__ == "__main__":
    run_ingest(IngestConfig.from_env())
```

`validate="one_to_one"` on the merge turns "duplicate orders" from a quiet correctness bug into a loud failure. The exit code is what the scheduler (Airflow/Prefect/cron) keys on, so silent correctness bugs must be converted into loud failures to be useful.

## 3. Validation steps

Run these in order. Each has a named failure mode it exists to catch.

1. **Static import check** — `python -c "import ingest.run"`. Catches missing dependencies and import-time typos before you spend time on real data.
2. **Unit: schema assertion** — feed `assert_schema` a DataFrame with a renamed column; assert it raises `ValueError`.
3. **Unit: watermark half-open** — call `extract_orders` twice with contiguous windows `[00:00, 12:00)` and `[12:00, 24:00)`; the union of row sets must equal a single-shot `[00:00, 24:00)` call with no duplicates.
4. **Integration: SQLite + fake S3** — point `INGEST_SQL_URL` at a SQLite file preloaded with 5 orders, point the S3 extract at a `moto` or `localstack` fake bucket, run `run_ingest`, assert the parquet exists and the manifest row count matches.
5. **Idempotence** — run the orchestrator twice with the same `INGEST_DATE`. The second run must overwrite the partition and produce an identical SHA-256 in the manifest.
6. **Negative path: upstream rename** — rename `order_total_usd` → `order_total` in the extract, re-run; the pipeline must fail at `assert_schema`, not silently write a column-missing parquet.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Separation of concerns | 2 | Extracts return DataFrames; orchestrator does join + write; no SQL strings in `run.py` |
| 2 | Secrets handling | 2 | No credentials, URLs, or bucket names hard-coded; all via env or secret manager |
| 3 | Streamed SQL read | 2 | `chunksize` or server-side cursor; a reviewer running `EXPLAIN` on the query sees an index-friendly `updated_at` filter |
| 4 | Watermark correctness | 3 | Half-open interval; two contiguous runs produce disjoint, exhaustive row sets; watermark persisted outside the container filesystem |
| 5 | S3 pagination | 2 | Uses a paginator, not a bare `list_objects_v2`; handles > 1000 keys |
| 6 | Schema assertion | 3 | Fails on missing, extra, or wrong-dtype columns; raises *before* write |
| 7 | Join safety | 2 | `validate=` argument on the merge; one row per primary key documented and enforced |
| 8 | Manifest / provenance | 2 | Every partition has a manifest with row count, watermark, and content hash |
| 9 | Idempotence | 2 | Re-running the same date overwrites deterministically; manifest hash is stable across identical inputs |

Partial credit: a submission missing items 4 or 6 cannot score above 14 — those are the two items that cause real downstream incidents.

## 5. Common mistakes

- **Closed watermark interval** (`updated_at <= :until`). A row updated at exactly the boundary is picked up twice (today and tomorrow) *or* zero times depending on how `until` is computed. Always use half-open `[since, until)`.
- **`list_objects_v2` without a paginator.** The AWS SDK caps a single call at 1,000 keys; the ingest silently truncates on busy days. Fixing this later is a correctness incident, not a performance tune.
- **`pd.read_sql` with no chunksize.** Fine on toy data, OOM on real tables. Add it on day one, not after the first page.
- **Writing CSV.** Floats round-trip as strings, dtypes are rediscovered on read, dates become objects. Parquet is non-negotiable for ML pipelines.
- **No merge `validate=`.** A duplicate `order_id` from a backfill silently inflates row counts. The merge call is where to catch it.
- **Extracting then transforming inside the extract function.** Extract should return raw-but-typed data. Business logic (e.g., currency normalisation across markets) belongs one layer up where it can be tested without a database.
- **Treating missing S3 objects as success.** If the producer failed, zero events today is not the same as zero events happening — distinguish "no partition exists" (upstream broken, fail) from "partition exists and is empty" (legitimate quiet day, proceed).

## 6. References

- NIST AI Risk Management Framework 1.0 — `MAP 1.6` (document provenance) and `MANAGE 2.1` (data lineage and accountability). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` as the control motivation for schema validation at ingest. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` for the broader threat model that data sources sit within. <https://atlas.mitre.org/>
