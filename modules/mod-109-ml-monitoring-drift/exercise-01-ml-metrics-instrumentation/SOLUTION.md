# exercise-01 — ML Metrics Instrumentation · SOLUTION

Paired exercise: `lessons/mod-109-ml-monitoring-drift/exercises/exercise-01-ml-metrics-instrumentation.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to instrument a prediction service so that operators can answer, without opening the model, three questions: **is the service healthy, is the model healthy, and is the model's input shaped like what it was trained on?** Health dashboards that only show request-per-second and p99 latency pass the first question and silently fail the other two — that is the gap this exercise closes.

Design priorities, in order:

1. **Three metric families, not one.** Service health (request rate, errors, latency), model health (score distribution, prediction class mix, confidence), and input health (per-feature missingness, type violations, out-of-range counts). Each family answers a different question and is read by a different responder.
2. **Labels are a schema, not a free-for-all.** Every metric is labelled by `model_name` and `model_version` at minimum, and by request-scoped dimensions (route, slice) only when the cardinality is bounded. Unbounded labels (user id, request id) cause metric cardinality explosion and have to be moved to logs or traces.
3. **Export the shape of the feature, not the feature itself.** Raw feature values are PII-adjacent and high-cardinality. Histograms, quantiles, and null counts are the right shape for monitoring; raw values belong in a sampled prediction log, not in the metrics endpoint.
4. **Instrument once at the boundary.** The predict handler owns the metrics. Instrumenting inside the model wrapper breaks when the model is swapped; instrumenting inside middleware misses the prediction-value signal.
5. **Correlate, don't collapse.** Metrics give the time series, logs give the row, traces give the span. The instrumentation emits all three keyed on a shared `request_id`, so a spike on the dashboard lands a responder in the exact log line in one click.

This design sits inside the NIST AI RMF `MEASURE 2.4` / `MANAGE 4.1` expectation that deployed AI systems are continuously monitored for performance and context drift, and the OWASP ML Top 10 `ML05:2023 Model Theft` / `ML02:2023 Data Poisoning Attack` surfaces where anomalous input volumes and distributions are often the first external signal.

| Choice | Alternative considered | Why this one |
|---|---|---|
| Prometheus client with `/metrics` pull endpoint | Push to StatsD / direct cloud SDK | Pull keeps the service free of outbound dependencies; scraping is the ops standard and every dashboarding stack can read it |
| Histogram for score and latency | Single gauge "last value" | A gauge collapses to the most recent request; a histogram preserves the distribution and powers p50/p95/p99 queries |
| Per-feature `_null_total` counter + `_value` histogram | One big "input OK" boolean | The boolean tells you *something* is wrong, the per-feature metric tells you *which feature* without opening a debugger |
| Model version as a label on every metric | Separate metric names per version | Label-based queries (`sum by (model_version)(...)`) make rollouts and A/B readable; metric-name-per-version makes dashboards unmaintainable |
| Sampled prediction log with `request_id` | Log every prediction | Full logging at 10 krps overwhelms the log pipeline; 1% sampling keyed on `hash(request_id)` keeps forensic power with linear cost |
| OpenTelemetry trace from gateway → feature fetch → model → response | Metrics only | Traces attribute the latency; without them, "p99 doubled" is a dashboard observation with no owner |

## 2. Worked implementation

### 2.1 Project layout

```
service/
├── __init__.py
├── metrics.py         # metric definitions; imported once at process start
├── schema.py          # feature contract + validation
├── prediction_log.py  # sampled structured log
├── app.py             # FastAPI handler; the one place that writes metrics
└── model.py           # loads the model; knows nothing about metrics
```

### 2.2 `metrics.py` — the metric contract

```python
from prometheus_client import Counter, Histogram, Gauge

MODEL_LABELS = ("model_name", "model_version")

predictions_total = Counter(
    "ml_predictions_total",
    "Count of prediction requests served.",
    MODEL_LABELS + ("route", "outcome"),
)

prediction_latency_seconds = Histogram(
    "ml_prediction_latency_seconds",
    "End-to-end latency of /predict, including feature fetch and scoring.",
    MODEL_LABELS + ("route",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

prediction_score = Histogram(
    "ml_prediction_score",
    "Model output score distribution (probability or regression value).",
    MODEL_LABELS,
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

prediction_class_total = Counter(
    "ml_prediction_class_total",
    "Count of predictions per output class after threshold.",
    MODEL_LABELS + ("predicted_class",),
)

feature_null_total = Counter(
    "ml_feature_null_total",
    "Count of requests where a given feature arrived null.",
    MODEL_LABELS + ("feature",),
)

feature_value = Histogram(
    "ml_feature_value",
    "Per-feature value distribution for numeric features (sampled).",
    MODEL_LABELS + ("feature",),
)

feature_out_of_range_total = Counter(
    "ml_feature_out_of_range_total",
    "Count of requests where a feature violated its trained range or category set.",
    MODEL_LABELS + ("feature", "kind"),  # kind = numeric | categorical
)

model_info = Gauge(
    "ml_model_info",
    "Static info about the loaded model (always 1, used as a label carrier).",
    MODEL_LABELS + ("trained_at", "code_sha"),
)
```

Two deliberate choices are visible here. First, **outcome** is a bounded label on `predictions_total` (`success`, `validation_error`, `model_error`, `timeout`) — enough to compute an error rate, few enough that cardinality stays sane. Second, there is no `user_id` label on any metric. Per-user slicing is a log-and-trace question, not a metrics question.

### 2.3 `schema.py` — the feature contract

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class NumericFeature:
    name: str
    min: float
    max: float
    sample_rate: float = 0.1  # fraction of requests to record in `feature_value`

@dataclass(frozen=True)
class CategoricalFeature:
    name: str
    allowed: frozenset[str]

FeatureSpec = NumericFeature | CategoricalFeature

CHURN_V3_FEATURES: tuple[FeatureSpec, ...] = (
    NumericFeature(name="tenure_months", min=0.0, max=240.0),
    NumericFeature(name="monthly_spend", min=0.0, max=10_000.0),
    NumericFeature(name="support_tickets_30d", min=0.0, max=200.0),
    CategoricalFeature(
        name="plan_tier",
        allowed=frozenset({"free", "basic", "pro", "enterprise"}),
    ),
)
```

The schema is derived from the training set — the numeric ranges are the `[min, max]` observed during training, and the categorical set is the categories the model has weights for. This is not input validation in the "accept or reject" sense; it is drift-adjacent instrumentation. A request with `plan_tier="trial"` is served (the model will predict *something*), and a counter is incremented so a responder can see new categories arriving before AUC drops.

### 2.4 `prediction_log.py` — the sampled structured log

```python
import hashlib
import json
import logging
import time

logger = logging.getLogger("prediction")

def should_sample(request_id: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(request_id.encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return bucket < rate

def log_prediction(
    *,
    request_id: str,
    model_name: str,
    model_version: str,
    features: dict,
    score: float,
    predicted_class: str,
    latency_s: float,
    sample_rate: float = 0.01,
) -> None:
    if not should_sample(request_id, sample_rate):
        return
    logger.info(json.dumps({
        "ts": time.time(),
        "request_id": request_id,
        "model_name": model_name,
        "model_version": model_version,
        "features": features,
        "score": score,
        "predicted_class": predicted_class,
        "latency_s": latency_s,
    }))
```

Sampling is deterministic on `request_id`: the same request is either always logged or never logged, which means a responder tailing logs and grepping for a `request_id` from a user complaint will either find the row or know definitively it was not sampled. Random sampling breaks that property.

### 2.5 `app.py` — the one place that writes metrics

```python
import time
import uuid
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app

from .metrics import (
    predictions_total, prediction_latency_seconds, prediction_score,
    prediction_class_total, feature_null_total, feature_value,
    feature_out_of_range_total, model_info,
)
from .model import MODEL, MODEL_NAME, MODEL_VERSION, CODE_SHA, TRAINED_AT
from .prediction_log import log_prediction, should_sample
from .schema import CHURN_V3_FEATURES, NumericFeature, CategoricalFeature

app = FastAPI()
app.mount("/metrics", make_asgi_app())

LABELS = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION}
model_info.labels(**LABELS, trained_at=TRAINED_AT, code_sha=CODE_SHA).set(1)


def _observe_input(request_id: str, features: dict) -> None:
    for spec in CHURN_V3_FEATURES:
        val = features.get(spec.name)
        if val is None:
            feature_null_total.labels(**LABELS, feature=spec.name).inc()
            continue
        if isinstance(spec, NumericFeature):
            if not (spec.min <= val <= spec.max):
                feature_out_of_range_total.labels(
                    **LABELS, feature=spec.name, kind="numeric"
                ).inc()
            if should_sample(request_id, spec.sample_rate):
                feature_value.labels(**LABELS, feature=spec.name).observe(float(val))
        elif isinstance(spec, CategoricalFeature):
            if val not in spec.allowed:
                feature_out_of_range_total.labels(
                    **LABELS, feature=spec.name, kind="categorical"
                ).inc()


@app.post("/predict")
def predict(payload: dict):
    request_id = payload.get("request_id") or str(uuid.uuid4())
    features = payload.get("features", {})
    route = "/predict"
    start = time.perf_counter()
    outcome = "success"
    score = None
    predicted_class = None
    try:
        _observe_input(request_id, features)
        try:
            score = float(MODEL.predict_proba(features))
        except (KeyError, ValueError):
            outcome = "validation_error"
            raise HTTPException(status_code=422, detail="invalid features")
        except Exception:
            outcome = "model_error"
            raise HTTPException(status_code=500, detail="model error")
        predicted_class = "churn" if score >= 0.5 else "retain"
        prediction_score.labels(**LABELS).observe(score)
        prediction_class_total.labels(
            **LABELS, predicted_class=predicted_class
        ).inc()
        return {"request_id": request_id, "score": score, "predicted_class": predicted_class}
    finally:
        elapsed = time.perf_counter() - start
        prediction_latency_seconds.labels(**LABELS, route=route).observe(elapsed)
        predictions_total.labels(**LABELS, route=route, outcome=outcome).inc()
        if score is not None:
            log_prediction(
                request_id=request_id, model_name=MODEL_NAME,
                model_version=MODEL_VERSION, features=features, score=score,
                predicted_class=predicted_class, latency_s=elapsed,
            )
```

Three details earn their complexity:

- The `try/finally` ensures `predictions_total` and `prediction_latency_seconds` are written *even on exception*, so error-rate and p99 panels never go blind under failure.
- `outcome` is a bounded enum. Setting `outcome = "500"` straight from the HTTP status would include every future status code the service might ever return, which is how label cardinality creeps up over months.
- `_observe_input` is called before the model. If the model raises on malformed input, the feature-level counters still recorded *which* feature was malformed, which is the signal a responder needs.

### 2.6 Alerting rules (Prometheus syntax)

```yaml
groups:
- name: ml-service
  rules:
  - alert: PredictionErrorRateHigh
    expr: |
      sum(rate(ml_predictions_total{outcome!="success"}[5m]))
        / sum(rate(ml_predictions_total[5m])) > 0.02
    for: 10m
    annotations:
      summary: "Error rate above 2% for 10m."

  - alert: PredictionLatencyP99High
    expr: |
      histogram_quantile(0.99,
        sum by (le, model_version)(rate(ml_prediction_latency_seconds_bucket[5m]))
      ) > 0.5
    for: 10m
    annotations:
      summary: "p99 latency above 500ms for 10m (per model_version)."

  - alert: FeatureOutOfRangeSpike
    expr: |
      sum by (feature, model_version)(
        rate(ml_feature_out_of_range_total[5m])
      ) > 1
    for: 15m
    annotations:
      summary: "Feature {{ $labels.feature }} out-of-range rate > 1/s."

  - alert: ScoreDistributionDrift
    expr: |
      abs(
        histogram_quantile(0.5, sum by (le)(rate(ml_prediction_score_bucket[1h])))
        - histogram_quantile(0.5, sum by (le)(rate(ml_prediction_score_bucket[7d])))
      ) > 0.1
    for: 1h
    annotations:
      summary: "Median predicted score shifted by >0.1 vs 7-day baseline."
```

The score-distribution alert is intentionally conservative: a 1-hour shift of 0.1 in median score is noticeable without being noisy, and it is one of the few alerts that fires *without ground-truth labels*, which is what makes it a drift tripwire. The drift-detection exercise (`exercise-02`) upgrades this heuristic to a proper statistical test.

## 3. Validation steps

1. **Static check** — `python -c "from service.app import app"` loads the module, which registers every metric. Catches typos in label names before load testing.
2. **Smoke scrape** — start the service and `curl http://localhost:8000/metrics`. The response must include every metric name declared in `metrics.py`, each with a `HELP` and `TYPE` line.
3. **Happy-path prediction** — POST a valid request, re-scrape. `ml_predictions_total{outcome="success"}` must increment by exactly 1; `ml_prediction_latency_seconds_count` by exactly 1; `ml_prediction_score_count` by exactly 1.
4. **Invalid-input prediction** — POST a request with `plan_tier="trial"`. The request must still be served, `ml_feature_out_of_range_total{feature="plan_tier",kind="categorical"}` must increment, and the sampled prediction log must contain the row if sampled.
5. **Error-path prediction** — POST a malformed request (missing a required field). `ml_predictions_total{outcome="validation_error"}` must increment, `ml_prediction_latency_seconds_count` must *also* increment (the finally block is doing its job), and `ml_prediction_score_count` must *not*.
6. **Cardinality audit** — after a synthetic load of 10 000 requests, scrape and run `curl -s /metrics | grep -c '^ml_predictions_total{'`. Expected value: `routes * outcomes * versions`, not thousands. If the count is large, a per-request label snuck in.
7. **Alert rule unit tests** — use `promtool test rules` against a fixed time-series fixture that includes a 3% error-rate window and a 60-second p99 spike. The two corresponding alerts must fire, no others.
8. **Trace parity** — enable OpenTelemetry export and verify one span per `/predict` request appears in the trace backend with `request_id` as an attribute matching the sampled prediction log.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Three metric families present | 3 | Service, model, and input families each have ≥1 metric; none missing |
| 2 | Model name + version as labels | 2 | Every metric carries `model_name` and `model_version`; version is read from the loaded model, not hard-coded |
| 3 | Latency histogram, not gauge | 2 | Histogram with buckets covering expected SLO; p99 is queryable, not just "last value" |
| 4 | Score distribution captured | 2 | Score logged as histogram; median and tail are queryable per version |
| 5 | Per-feature input health | 3 | Missingness counter and out-of-range counter per feature; numeric features have a sampled value histogram |
| 6 | Bounded label cardinality | 2 | No request-id, user-id, or free-form string as a metric label; `outcome` is a fixed enum |
| 7 | Metrics written on error | 2 | `try/finally` (or equivalent) ensures counters and latency fire even when the handler raises |
| 8 | Sampled prediction log | 2 | Structured log, deterministic sampling on `request_id`, includes features + score + version |
| 9 | Alerting rules | 2 | At least error-rate, latency-p99, and one model-signal (score drift or feature-OOR) alert; tested with `promtool test rules` |

Items 2 and 6 are mandatory: a submission missing either cannot score above 12. Metrics without `model_version` cannot distinguish rollouts; unbounded label cardinality kills the Prometheus server and is a production outage waiting to happen.

## 5. Common mistakes

- **Instrumenting inside the model wrapper.** Metrics live in the handler, not the model. Otherwise swapping the model (the whole point of model versioning) silently drops the metrics with it.
- **Using a gauge for latency.** A gauge shows the last request's latency, which is noise. Histograms are what Prometheus's `histogram_quantile` consumes to produce p50/p95/p99.
- **Raw feature values on `/metrics`.** Metrics are world-readable within the ops stack and get scraped into long-term storage. Emit distributions (histograms, counts), not values, and send the raw features to a sampled log.
- **Per-request-id labels.** Each unique label value is a new time series. 10 krps for an hour is 36 M series — enough to wedge a Prometheus server. High-cardinality dimensions go into logs and traces.
- **No metric on error path.** Writing `prediction_count.inc()` only after the model succeeds leaves the error rate invisible. The `finally` block is load-bearing.
- **Hard-coded thresholds in the handler.** `score >= 0.5` for the churn/retain cutoff should be a config-driven value from the model artifact — otherwise two models with different calibration can't share the same handler.
- **One monster `ml_feature_info` metric.** A single metric with one label per feature forces every feature to share a cardinality budget; separate counters per feature keep each one cheap.
- **Alerting on single data points.** Rules without a `for:` clause fire on every transient blip. 10–15 minutes is the floor for most "something is wrong" alerts; immediate-fire is reserved for hard SLOs.
- **No model_info gauge.** Without a metric that labels the running build, a reviewer in two months cannot say which version produced the dashboard they are looking at.

## 6. References

- Prometheus — Metric types (counter/gauge/histogram/summary) and the semantics of `_count` / `_bucket` / `_sum`. <https://prometheus.io/docs/concepts/metric_types/>
- Prometheus — Instrumentation best practices, including label cardinality and the "finally" pattern for error paths. <https://prometheus.io/docs/practices/instrumentation/>
- OpenTelemetry — Specification for metrics, logs, and traces, and the `request_id` / `trace_id` correlation story. <https://opentelemetry.io/docs/specs/otel/>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.4` (ongoing monitoring of AI systems in production) and `MANAGE 4.1` (post-deployment monitoring plans). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — `ML02:2023 Data Poisoning Attack` and `ML05:2023 Model Theft` as attack surfaces where input-distribution anomalies are the first external signal. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0024 Exfiltration via ML Inference API` and related tactics; prediction-volume and score-distribution metrics are the detection surface. <https://atlas.mitre.org/>
