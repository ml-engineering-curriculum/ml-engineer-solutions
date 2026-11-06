# exercise-01 — FastAPI Prediction Service · SOLUTION

Paired exercise: `lessons/mod-108-model-deployment-serving/exercises/exercise-01-fastapi-prediction-service.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to turn a trained classifier into an HTTP prediction service using [FastAPI](https://fastapi.tiangolo.com/). "A service" is not `uvicorn app:app` with one endpoint — it is a long-running process that must survive a bad request, a corrupted model file, a slow downstream, a restart during deploy, and still produce predictions whose inputs and outputs are auditable.

Design priorities, in order:

1. **The model loads once, at startup, into a known-good state.** A per-request `joblib.load` is both slow (cold-start latency on every call) and unsafe (a corrupted file turns every request into a 500). Lifespan hook fails fast; the pod never starts if the model is bad.
2. **Request and response schemas are the contract.** Pydantic models mirror the MLflow signature inferred at training time. Any payload the model has not been promised to handle is rejected at the edge with 422, before it reaches the inference code. This is the OWASP ML Top 10 `ML01:2023 Input Manipulation Attack` surface — the schema is the first defense.
3. **Prediction is an isolated function.** `predict(payload: Features) -> Prediction` has no `Request` object, no logger, no FastAPI symbols. It is unit-testable without a client.
4. **`/healthz` vs `/readyz` mean different things.** `/healthz` says "the process is alive"; `/readyz` says "the model is loaded and I can serve traffic." Kubernetes rolling deploys depend on this distinction — conflating them causes traffic to land on a pod whose model has not finished loading.
5. **Every response carries a model version and a request ID.** Downstream monitoring joins on these two fields; without them, drift and incident response are both guesswork. This is NIST AI RMF `MEASURE 2.3` / `MEASURE 2.6` on traceable model outputs.
6. **No model internals in error bodies.** A stack trace in a 500 body leaks the model class, library versions, and often the file layout — material for MITRE ATLAS `AML.T0005 Discover ML Model Family`. Errors are structured, logged with a correlation ID, and returned opaque.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Lifespan hook loads the model | Lazy load on first request | First request pays the cold-start; a probe or health check races the request; worse tail latency and worse failure mode |
| Pydantic model mirrors MLflow signature | Accept `dict[str, Any]` | The signature is the contract; typing the handler makes 422 automatic and `openapi.json` useful |
| Separate `/healthz` and `/readyz` | One `/health` endpoint | K8s `livenessProbe` restarts the pod, `readinessProbe` only pulls it out of the service — the semantics are different |
| `/predict` accepts one record; `/predict/batch` accepts a list | Batch-only | Online path should not pay list-serialisation overhead for a single call; batch exists for internal callers |
| Return `model_version` and `request_id` in body + headers | Body only, or headers only | Body for client code, headers for load balancers / log pipelines that do not parse bodies |
| Loguru / stdlib logging in JSON | Print statements | Structured logs are what the aggregator joins on; prints hit stdout as free text |
| `uvicorn --workers N` behind a reverse proxy | Uvicorn directly on `:443` | Production serving wants TLS termination, request limits, and connection pooling upstream; uvicorn should only speak HTTP/1.1 to localhost |

## 2. Worked implementation

### 2.1 Project layout

```
service/
├── __init__.py
├── config.py          # model URI, worker count, log level — all env-driven
├── schemas.py         # Pydantic request/response models
├── model.py           # load + predict; no FastAPI imports
├── app.py             # FastAPI app, lifespan, routes, middleware
└── main.py            # python -m service.main for local dev
tests/
├── test_schemas.py
├── test_model.py
└── test_app.py
pyproject.toml
```

### 2.2 `config.py` — env is the deployment knob

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceConfig:
    model_uri: str
    model_version: str
    log_level: str
    max_batch_size: int

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            model_uri=os.environ["MODEL_URI"],           # fail fast if unset
            model_version=os.environ.get("MODEL_VERSION", "unknown"),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "256")),
        )
```

`MODEL_URI` has no default on purpose: a missing model configuration should crash at import time, not after the first request lands on an empty service. `MAX_BATCH_SIZE` caps the memory a single caller can commit — the service's "slowloris batch" bound.

### 2.3 `schemas.py` — the contract

```python
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid")   # reject unknown fields

    tenure_months: int = Field(ge=0, le=600)
    monthly_charges: float = Field(ge=0.0, le=10_000.0)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    has_phone_service: bool
    num_support_tickets_30d: int = Field(ge=0, le=1_000)


class Prediction(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    label: Literal[0, 1]
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    items: list[Features] = Field(min_length=1, max_length=256)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
```

`extra="forbid"` is load-bearing. Without it, a client sending `{"tenure_months": 12, "ssn": "..."}` silently drops the unexpected field and the service never notices it was handed PII. `Literal` on `contract_type` turns the Pydantic validator into an allow-list — `ML01:2023 Input Manipulation` becomes a 422, not an inference. The bounds (`ge`, `le`, `max_length`) protect downstream resources; a payload of `10_000_000` records would otherwise OOM the pod.

### 2.4 `model.py` — pure inference

```python
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .schemas import Features, Prediction


@dataclass
class Model:
    estimator: object            # anything with predict_proba
    feature_order: list[str]
    version: str

    @classmethod
    def load(cls, uri: str, version: str) -> "Model":
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(f"model URI not found: {uri}")
        bundle = joblib.load(path)
        return cls(
            estimator=bundle["estimator"],
            feature_order=bundle["feature_order"],
            version=version,
        )

    def _frame(self, items: list[Features]) -> pd.DataFrame:
        rows = [i.model_dump() for i in items]
        df = pd.DataFrame(rows)
        return df[self.feature_order]            # reorder, raise KeyError if any missing

    def predict(self, items: list[Features], request_ids: list[str]) -> list[Prediction]:
        if len(items) != len(request_ids):
            raise ValueError("request_ids must align with items")
        x = self._frame(items)
        proba = self.estimator.predict_proba(x)[:, 1]
        return [
            Prediction(
                probability=float(p),
                label=int(p >= 0.5),
                model_version=self.version,
                request_id=rid,
            )
            for p, rid in zip(proba, request_ids)
        ]
```

Zero FastAPI imports. The `feature_order` lookup is explicit — relying on `DataFrame` column order inferred from dict insertion is how a Python-minor-version upgrade silently reorders features and the model's AUC quietly collapses in prod.

### 2.5 `app.py` — the service

```python
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import ServiceConfig
from .model import Model
from .schemas import BatchRequest, BatchResponse, Features, Prediction


logger = logging.getLogger("service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ServiceConfig.from_env()
    logging.basicConfig(level=cfg.log_level, format="%(message)s")
    app.state.cfg = cfg
    app.state.model = Model.load(cfg.model_uri, cfg.model_version)
    app.state.ready = True
    logger.info(
        '{"event":"model_loaded","version":"%s","uri":"%s"}',
        cfg.model_version, cfg.model_uri,
    )
    try:
        yield
    finally:
        app.state.ready = False
        logger.info('{"event":"shutdown"}')


app = FastAPI(title="churn-predictor", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = rid
    response: Response = await call_next(request)
    response.headers["x-request-id"] = rid
    response.headers["x-model-version"] = app.state.cfg.model_version
    return response


@app.exception_handler(ValidationError)
async def _on_validation(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "detail": exc.errors(),
                 "request_id": request.state.request_id},
    )


@app.exception_handler(Exception)
async def _on_unhandled(request: Request, exc: Exception):
    logger.exception(
        '{"event":"unhandled","request_id":"%s"}', request.state.request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error",
                 "request_id": request.state.request_id},
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready", "model_version": app.state.cfg.model_version}


@app.post("/predict", response_model=Prediction)
async def predict(features: Features, request: Request):
    [pred] = app.state.model.predict([features], [request.state.request_id])
    return pred


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(body: BatchRequest, request: Request):
    cfg: ServiceConfig = app.state.cfg
    if len(body.items) > cfg.max_batch_size:
        raise HTTPException(status_code=413, detail="batch too large")
    rids = [f"{request.state.request_id}-{i}" for i in range(len(body.items))]
    preds = app.state.model.predict(body.items, rids)
    return BatchResponse(predictions=preds)
```

Why these bits earn their lines:

- **`lifespan`** runs the model load exactly once; if it raises, uvicorn exits non-zero and Kubernetes restarts the pod rather than serving 500s.
- **`request_id_middleware`** honours a client-supplied `x-request-id` (useful for tracing across services) but generates one if absent — logs always have a join key.
- **`/healthz`** returns 200 unconditionally; it only tests that the event loop is alive. **`/readyz`** returns 503 until the lifespan hook finishes — the signal Kubernetes waits for before adding the pod to the service.
- **Error bodies are opaque**: `"internal_error"` + `request_id`. The operator finds the full trace by the request ID in the logs; the attacker finds nothing.

### 2.6 Running it

```bash
# 1. Train and save bundle (shape expected by model.py)
python - <<'PY'
import joblib
from sklearn.ensemble import GradientBoostingClassifier
# ... fit `clf` on your dataset ...
joblib.dump(
    {"estimator": clf, "feature_order": [
        "tenure_months", "monthly_charges", "contract_type",
        "has_phone_service", "num_support_tickets_30d",
    ]},
    "artifacts/churn-v1.joblib",
)
PY

# 2. Serve
MODEL_URI=artifacts/churn-v1.joblib \
MODEL_VERSION=v1.0.3 \
LOG_LEVEL=INFO \
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 2
```

## 3. Validation steps

Run these in order; each exists to catch one real failure mode.

1. **Import + lifespan boot** — `uvicorn service.app:app --port 8000` with a valid `MODEL_URI` must emit the `model_loaded` log line before `Uvicorn running on`. If the order is reversed, readiness logic is wrong.
2. **Unset `MODEL_URI`** — start with no env; the process must exit non-zero within one second. A service that boots without a model is a liveness-check pass with a 100% error rate.
3. **Golden request** — `curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"tenure_months":12,"monthly_charges":55.0,"contract_type":"month-to-month","has_phone_service":true,"num_support_tickets_30d":2}' | jq`. Expect `probability` in `[0,1]`, `label` in `{0,1}`, `model_version` equal to the env, non-empty `request_id`.
4. **Response headers** — the same call must include `x-request-id` and `x-model-version` headers. Missing either breaks downstream tracing.
5. **Unknown field** — add `"ssn":"123"` to the payload; expect 422 with `error=invalid_request` and no mention of the model. Catches `extra="forbid"` being off.
6. **Out-of-range** — `"tenure_months": -1`; expect 422. Catches missing bounds.
7. **Wrong enum** — `"contract_type": "weekly"`; expect 422. Catches `Literal` being replaced with `str`.
8. **Batch cap** — POST `/predict/batch` with 300 items; expect 413. Catches missing batch bound.
9. **Readiness gate** — in a fresh container, `curl /readyz` within the lifespan should return 503; after boot, 200. `curl /healthz` is 200 throughout. Catches the common bug of pointing both probes at the same handler.
10. **Load (quick)** — `hey -z 30s -c 20 -m POST -H 'content-type: application/json' -d '{...}' http://localhost:8000/predict`. p99 should be flat under sustained concurrency; if it ramps, the model is being reloaded per request.
11. **Crash isolation** — hand-craft a request that triggers a model error (e.g., inject a column type mismatch by changing feature_order). Expect 500 with an opaque body and a logged stack trace. The pod must not crash.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Model lifecycle | 3 | Loaded once in lifespan; `MODEL_URI` unset → process exits non-zero |
| 2 | Pydantic schema | 3 | `extra="forbid"`, bounds and `Literal` on constrained fields, request + response models |
| 3 | `/healthz` vs `/readyz` | 2 | Semantically distinct; readiness is 503 during lifespan |
| 4 | Error handling | 2 | 422 for schema errors; 500 opaque with request ID; no stack traces in responses |
| 5 | Request IDs + model version | 2 | Both in response body *and* `x-*` headers; honours client-supplied `x-request-id` |
| 6 | Batch endpoint with cap | 2 | Separate endpoint, enforced `MAX_BATCH_SIZE`, 413 when exceeded |
| 7 | Separation of concerns | 2 | `model.py` has no FastAPI imports; unit-testable without a client |
| 8 | Structured logging | 2 | JSON (or key=value) logs with `request_id`; load + shutdown + unhandled error events |
| 9 | Env-driven config | 2 | Model URI, version, log level, batch cap all from env; no hard-coded paths |

Items 1, 2, 3, and 4 are mandatory. A service missing any one of them cannot score above 10: an unguarded model load, an unvalidated request, or a leaky error body makes everything else moot.

## 5. Common mistakes

- **Loading the model inside the handler.** Every request pays the deserialisation cost (10–200 ms for a small sklearn model, seconds for a torch checkpoint), and a corrupted file produces a 100% error rate instead of a crash loop. Load in `lifespan`.
- **`extra="allow"` on the request schema.** The service accepts unknown fields silently — PII, attack payloads, or just typoed column names that make it to inference as `NaN`. OWASP ML Top 10 `ML01:2023` starts here.
- **One `/health` endpoint.** Kubernetes `livenessProbe` and `readinessProbe` have different consequences: one restarts the pod, the other pulls it from service. A single endpoint forces the operator to choose which behaviour to break.
- **Returning the raw exception.** `return {"error": str(exc)}` leaks class names, library paths, and sometimes dataset filenames — `AML.T0005 Discover ML Model Family` reconnaissance for free.
- **Mutable request model.** Pydantic validation rejects bad input; then the handler does `features.tenure_months = ...` and defeats it. Treat validated input as immutable.
- **Returning a `numpy.float32`.** FastAPI/Pydantic will coerce silently in v2 but some older clients will choke; always cast to `float` / `int` at the edge. Also keeps the OpenAPI schema honest.
- **No `response_model`.** The handler returns a `Prediction` and it works — until a refactor adds an internal debug field and it leaks into the public body. `response_model` is the outbound schema; it should always be set.
- **Logging to stdout as plain text.** Aggregators parse JSON; free-text logs lose their request ID in a `grep` pipeline and incident response grinds to a halt.
- **Trusting the model file.** `joblib.load` executes pickled code. If the artifact path is under attacker control (e.g., URL fetched from user input), the service is one request from RCE. Treat `MODEL_URI` as privileged configuration, not input.

## 6. References

- FastAPI — official docs, lifespan events, dependency injection, request validation. <https://fastapi.tiangolo.com/>
- Pydantic v2 — `model_config`, `Field` constraints, `Literal`, `extra="forbid"`. <https://docs.pydantic.dev/latest/>
- Uvicorn — ASGI server, worker model, graceful shutdown. <https://www.uvicorn.org/>
- OWASP Machine Learning Security Top 10 — `ML01:2023 Input Manipulation Attack` (why the Pydantic schema is the defense), `ML05:2023 Model Theft` (why error bodies don't name the model). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0005 Discover ML Model Family` (reconnaissance via stack traces and library hints in responses). <https://atlas.mitre.org/>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (traceable outputs), `MEASURE 2.6` (deployed system monitoring) — why `model_version` and `request_id` must be in every response. <https://www.nist.gov/itl/ai-risk-management-framework>
