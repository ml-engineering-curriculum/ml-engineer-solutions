# exercise-04 — Load Test and Tune · SOLUTION

Paired exercise: `lessons/mod-108-model-deployment-serving/exercises/exercise-04-load-test-and-tune.md` (in the `ml-engineer-learning` repo).

## 1. Solution overview

The exercise asks the learner to take the containerised FastAPI service from exercises 01–02, submit it to a reproducible load test, read the results, and tune the service until it meets an explicit latency and throughput SLO. The failure mode most learners hit is measuring the wrong thing — eyeballing `curl` under light load, calling the service "fast enough," and discovering in production that p99 at 100 QPS is six seconds.

Design priorities, in order:

1. **Write the SLO before the test.** Without a target, the test is a vibe. The SLO used here: `p50 ≤ 30 ms, p95 ≤ 100 ms, p99 ≤ 200 ms, error rate < 0.1%` at `150 RPS sustained for 10 minutes`. These numbers are the exercise's acceptance criteria; a real system's numbers come from the product.
2. **Three load profiles, not one.** Smoke (correctness under light load), sustained (steady state at the SLO target), spike (2× the target for 30 s). Each probes a different failure mode; running only sustained misses the "cold connection pool" and "burst OOM" bugs.
3. **Co-locate the load generator or account for the network.** A k6 run from a developer laptop over Wi-Fi measures Wi-Fi, not the service. Run the generator on the same host, same VPC, or document the network path and measure a baseline against a no-op endpoint.
4. **Change one thing at a time.** Workers, concurrency limit, OMP threads, batch size, model size — a tuning session that changes four knobs between runs cannot attribute the win. Each run is a single, documented variable change.
5. **Measure resources, not just latency.** CPU %, RSS, event-loop lag, GC pauses, and request queue depth are the inputs to "why." A latency graph without a resource graph tells you *that* it is slow, not *why*.
6. **Stop tuning when the SLO is met.** The temptation is to push p99 from 180 ms to 120 ms "while we're here." That is optimisation without a business driver; shipping p99=180 ms on time is the correct outcome.

| Choice | Alternative considered | Why this one |
|---|---|---|
| k6 (JS test files, Prometheus-compatible output) | wrk, locust, vegeta | k6 scripts are version-controllable, support ramp-up profiles natively, and produce the percentile output the SLO is written in |
| Run k6 on the same host over `localhost` for tuning; on a nearby host for the acceptance test | Always remote | Localhost strips network variance; same-VPC acceptance test reflects production's real path |
| Three scenarios (smoke, sustained, spike) | One scenario | Each catches a distinct class of regression |
| Record CPU / RSS / event-loop lag during every run | Latency-only | Latency numbers without resource numbers cannot be explained |
| Tune the service, not the client | Chase client-side variance | Client-side JIT warmup, DNS, connection pooling can swamp real signal if ignored; k6 handles them, but you must verify |
| Capture results in a signed JSON summary per run | Spreadsheet | Reproducibility: diff two runs' summaries to see what changed |

## 2. Worked implementation

### 2.1 SLO

```yaml
# slo.yml — committed to the repo; the test asserts against it
version: 1
service: churn-predictor
sustained:
  target_rps: 150
  duration_s: 600
  p50_ms: 30
  p95_ms: 100
  p99_ms: 200
  error_rate: 0.001
spike:
  target_rps: 300        # 2x sustained
  duration_s: 30
  p99_ms: 400            # relaxed under spike; shedding is allowed, 5xx is not
  error_rate: 0.005
```

### 2.2 Load generator — `load/script.js`

```javascript
import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const latency = new Trend('predict_latency', true);
const errors = new Rate('predict_errors');

export const options = {
  scenarios: {
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '30s',
      exec: 'predict',
      tags: { scenario: 'smoke' },
    },
    sustained: {
      executor: 'constant-arrival-rate',
      rate: 150,            // target RPS
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 50,
      maxVUs: 200,
      startTime: '45s',
      exec: 'predict',
      tags: { scenario: 'sustained' },
    },
    spike: {
      executor: 'ramping-arrival-rate',
      startRate: 150,
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 400,
      stages: [
        { target: 150, duration: '10s' },
        { target: 300, duration: '30s' },   // 2x sustained for 30s
        { target: 150, duration: '10s' },
      ],
      startTime: '11m',
      exec: 'predict',
      tags: { scenario: 'spike' },
    },
  },
  thresholds: {
    'predict_latency{scenario:sustained}': ['p(50)<30', 'p(95)<100', 'p(99)<200'],
    'predict_errors{scenario:sustained}': ['rate<0.001'],
    'predict_latency{scenario:spike}':    ['p(99)<400'],
    'predict_errors{scenario:spike}':     ['rate<0.005'],
  },
};

const payload = JSON.stringify({
  tenure_months: 12,
  monthly_charges: 55.0,
  contract_type: 'month-to-month',
  has_phone_service: true,
  num_support_tickets_30d: 2,
});
const params = { headers: { 'content-type': 'application/json' } };

export function predict() {
  const r = http.post(`${__ENV.BASE_URL}/predict`, payload, params);
  latency.add(r.timings.duration);
  errors.add(r.status !== 200);
  check(r, { 'status is 200': (x) => x.status === 200 });
}
```

Three deliberate details:

- `constant-arrival-rate` (open model), not `constant-vus` (closed model). An open model generates requests on a wall-clock schedule regardless of how slow the service is — which is what production clients do. A closed-VU model lets a slow service hide by starving itself of requests.
- `preAllocatedVUs` sized to `rps × p99_ms / 1000 × 2` — enough concurrency to hold the RPS at the SLO latency ceiling, with a 2× cushion.
- Thresholds live in the script. `k6 run` exits non-zero when a threshold is missed, so the acceptance test is a boolean check in CI, not a human reading a dashboard.

### 2.3 Resource recorder — `load/record.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

CONTAINER=$1
OUT=$2
INTERVAL=1

mkdir -p "$(dirname "$OUT")"
echo "ts,cpu_pct,mem_mb,net_rx_kb,net_tx_kb" > "$OUT"

while :; do
  ts=$(date +%s)
  read -r cpu mem rx tx < <(docker stats --no-stream --format \
    '{{.CPUPerc}} {{.MemUsage}} {{.NetIO}}' "$CONTAINER" \
    | awk '{
        gsub("%","",$1);
        gsub("MiB","",$2); gsub("GiB","",$2);
        split($5,a,"kB|MB|GB");
        split($7,b,"kB|MB|GB");
        printf "%s %s %s %s\n", $1, $2, a[1], b[1];
      }')
  echo "$ts,$cpu,$mem,$rx,$tx" >> "$OUT"
  sleep "$INTERVAL"
done
```

`docker stats` is a cheap-but-approximate tool; for a production answer, scrape container cgroup counters or run a Prometheus node exporter. For the exercise, the point is that *some* resource timeline exists alongside the latency timeline.

### 2.4 Test runner — `load/run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?run tag, e.g. 'baseline' or 'workers-4'}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
CONTAINER="${CONTAINER:-churn-predictor}"
OUT="runs/${TAG}"
mkdir -p "$OUT"

# 1. Record the service config as it ran
docker inspect "$CONTAINER" > "$OUT/container.json"

# 2. Warmup — strip JIT and connection-pool transients from the measured window
k6 run --quiet --duration 20s --vus 1 --env BASE_URL="$BASE_URL" load/warmup.js \
  > "$OUT/warmup.log"

# 3. Resource recorder in the background
./load/record.sh "$CONTAINER" "$OUT/resources.csv" &
REC_PID=$!
trap 'kill $REC_PID 2>/dev/null || true' EXIT

# 4. Main run
BASE_URL="$BASE_URL" k6 run --out json="$OUT/raw.json" \
  --summary-export "$OUT/summary.json" \
  load/script.js \
  | tee "$OUT/k6.log"

echo "{\"tag\":\"$TAG\",\"finished_at\":\"$(date -u +%FT%TZ)\"}" > "$OUT/meta.json"
```

Every tuning run writes a self-contained directory: container config, resource timeline, k6 raw samples, k6 summary, and a meta record. `diff -r runs/baseline/ runs/workers-4/` is a useful review tool.

### 2.5 Tuning levers, in order

Applied in order to a service missing its SLO. Each row is one `run.sh` invocation with the stated change and nothing else. Figures in the "p99 ms (after)" column are from a representative Linux dev host — they will differ on learner machines; the relative shape is what matters, not the absolute numbers.

| # | Lever | Why | p99 ms (after) | Notes |
|---|---|---|---|---|
| 0 | **Baseline**: `--workers 1`, `--limit-concurrency 256`, default thread counts | Measure what we have | 480 | Fails `p99 < 200` under sustained |
| 1 | `UVICORN_WORKERS=4` (= pod CPU limit) | 1-worker uvicorn is single-threaded on the event loop; one slow request blocks the rest | 160 | Passes sustained; spike still bad |
| 2 | `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` | By default numpy/sklearn spawn threads per call; with N workers × T threads, you oversubscribe cores and p99 spikes | 140 | Latency variance down sharply |
| 3 | `--limit-concurrency 64` per worker (= 256 total) | Bound the in-flight queue so a traffic spike returns 503 instead of queuing and OOMing | 135, spike 340 | Spike now within relaxed p99 |
| 4 | Model artefact: `gbdt_200trees` → `gbdt_120trees` | The last 80 estimators moved AUC by 0.002 and the p99 by 60 ms | 85 | Decision needs product sign-off on AUC/latency trade |
| 5 | Keep-alive and uvicorn `--http h11`, `--timeout-keep-alive 30` | Reuse TCP + HTTP/1.1 connections; new TCP per request wastes ~2 ms each | 75 | Mostly benefits the client; marginal gain, documented and kept |

Interpretation:

- Workers (lever 1) buys the biggest headline win because async FastAPI on one worker is still CPU-bound on the model call. This is usually the first mistake in a load-test post-mortem.
- Thread pinning (lever 2) is the fix learners most often miss. The default sklearn/numpy will fight the uvicorn workers for the same cores under load, and the fix is a single env var.
- Concurrency cap (lever 3) does not improve the mean but is the difference between "graceful 503s under spike" and "every request slow for 90 seconds after the spike passes." It is a *reliability* tuning, not a latency tuning.
- Model shrink (lever 4) requires a product conversation; it is a legitimate lever but it is not free.
- Protocol tweaks (lever 5) are the last 10% and should be the *last* thing tuned. Do not reach for them until the previous levers are exhausted.

Stop at lever 3. The SLO is met at p99=135 ms (sustained) and 340 ms (spike), comfortably under 200 ms / 400 ms. Levers 4 and 5 are documented as "future work with product input," not applied to ship.

### 2.6 Report format — `load/report.md`

```markdown
# Load-test report — churn-predictor, run suite 2026-11-05

SLO: p50<30 / p95<100 / p99<200 / err<0.1% @ 150 RPS × 10 min

| Tag           | p50 | p95 | p99 | err %  | CPU peak | RSS peak | Verdict |
|---------------|-----|-----|-----|--------|----------|----------|---------|
| baseline      |  85 | 290 | 480 | 0.00   |  98%     | 420 MiB  | FAIL    |
| workers-4     |  28 |  95 | 160 | 0.00   | 360%     | 780 MiB  | PASS    |
| +omp-pinned   |  25 |  80 | 140 | 0.00   | 340%     | 760 MiB  | PASS    |
| +limit-64     |  25 |  82 | 135 | 0.07   | 320%     | 760 MiB  | PASS¹   |

¹ 0.07% 503s under spike are intentional — load-shedding, not crashes.
Shipped config: workers=4, OMP_NUM_THREADS=1, --limit-concurrency=64.
```

This table — SLO at the top, lever-by-lever rows, verdict column, and a one-line footnote on the intentional errors — is the artefact a reviewer reads first.

## 3. Validation steps

Run these in order; each exists to catch one real failure mode.

1. **SLO exists in the repo** — `cat slo.yml` is non-empty and matches the numbers in the k6 thresholds. A test without a committed SLO can be moved after the fact to produce a pass.
2. **Baseline run is reproducible** — `./load/run.sh baseline` twice; p99 values should be within ±10%. If they are not, the environment is noisy and the measured numbers mean less than they look.
3. **Warmup is enforced** — k6 log shows the 20 s warmup completed before the measured scenarios started. Latency of the first few hundred requests is JIT / connection-pool transient and must not land in the final numbers.
4. **Threshold-based pass/fail** — `k6 run` exits non-zero when any threshold fails. CI runs this as a boolean gate.
5. **Open-model load** — the test uses `constant-arrival-rate` (or `ramping-arrival-rate`), not `constant-vus`. A closed-VU test on a slow service self-throttles and hides the problem.
6. **Spike scenario runs** — the summary includes a `scenario:spike` tagged row. A load test with no spike profile cannot detect queueing collapse.
7. **Resource timeline exists** — `runs/<tag>/resources.csv` is populated and covers the same wall-clock window as the k6 run. Latency numbers alone cannot explain a regression.
8. **One variable per run** — each tuning run's directory names the single config difference from its predecessor (filename convention + a changelog). A multi-variable run cannot attribute cause.
9. **Error-mode distinction** — the report distinguishes 5xx-from-crash (bad) from 5xx-from-shedding (expected under spike with `--limit-concurrency` set).
10. **Shipped config written down** — the report's final line names the exact env vars and flags to run in prod. A load-test report that ends with "we made it faster" without a config block cannot be acted on.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | SLO committed | 2 | `slo.yml` (or equivalent) exists; test thresholds match |
| 2 | Open-model load | 2 | `constant-arrival-rate` used; justified over closed-VU |
| 3 | Warmup phase | 1 | Warmup runs before measurement; measurement window excludes it |
| 4 | Three scenarios | 2 | Smoke, sustained, spike all present |
| 5 | Threshold gating | 1 | Non-zero exit on breach; wired into CI or at least makefile target |
| 6 | Resource timeline | 2 | CPU + RSS captured alongside latency for every run |
| 7 | Baseline + tuning runs | 3 | Minimum of baseline + two tuning runs with named, isolated changes |
| 8 | Variable isolation | 2 | Each tuning run changes exactly one config variable |
| 9 | Correct lever order | 2 | Workers / thread pinning / concurrency cap considered before protocol tweaks and model shrink |
| 10 | Report with verdict column | 2 | Readable table; SLO at top; numbered runs; verdict per row |
| 11 | Shipped config documented | 1 | Final env vars + flags listed, ready to paste into a Dockerfile / deployment |

Items 1, 2, and 7 are mandatory. Without an SLO, without an open-model load profile, or without a before-and-after comparison, the "tuning" claim is a story without evidence.

## 5. Common mistakes

- **No SLO.** The test finishes, the report says "p99 is 180 ms," and nobody knows whether that is good. Define the number first, write it to a file, and assert on it.
- **Measuring with `ab` or `wrk` at defaults.** Default `ab -n 1000 -c 10` is a closed-VU test on an unknown connection model. Prefer k6 or vegeta (open-model, percentile-aware, scriptable).
- **Running k6 on the laptop over Wi-Fi.** The Wi-Fi link is the slowest element; you measured the hotel's AP, not your service. Co-locate the generator or document the network path.
- **No warmup.** The first few hundred requests pay JIT compilation (if any), connection-pool setup, and page-cache warm-up. Including them pushes the p99 by an order of magnitude on short tests.
- **Closed-VU load.** 50 virtual users looping as fast as they can hide a slow service: the slower the service, the lower the RPS, the less load it sees. Use arrival-rate executors.
- **Tuning without measuring resources.** Every "mystery" latency regression is CPU saturation, GC pause, or thread contention. Without a resource timeline, you tune in the dark and get lucky or unlucky.
- **Changing four knobs at once.** p99 drops, you celebrate, and when it regresses six weeks later you cannot un-tangle which knob regressed. One knob per run, one changelog entry per run.
- **Confusing `--workers` with concurrency.** More workers scale CPU-bound work across cores. Higher `--limit-concurrency` lets each worker hold more in-flight requests. They are different levers with different failure modes; tune both, deliberately.
- **Leaving `OMP_NUM_THREADS` at auto.** With N uvicorn workers × T BLAS threads × a 4-core pod, you oversubscribe cores by N·T/4×, and p99 turns ragged. Pin `OMP_NUM_THREADS=1` (or 2) when running multiple workers.
- **Ignoring spikes.** Steady-state passing an SLO at 150 RPS tells you nothing about what happens at a 300-RPS launch spike. Test the spike; measure whether the service queues unboundedly or sheds load.
- **Not shedding load.** A service with no `--limit-concurrency` cap will queue requests under a spike, breach every timeout downstream, and recover minutes after the spike is over. A load-shedding 503 is a feature, not a bug.
- **Not writing down the shipped config.** The tuning session discovered the right values; the report ends with a prose paragraph; six weeks later nobody can find the exact flags. Put the final config block in the report and in the deployment manifest.

## 6. References

- Google SRE Workbook — *Implementing SLOs*. Defines SLI/SLO/SLA; the discipline of writing numeric targets before measurement. <https://sre.google/workbook/implementing-slos/>
- k6 — load-testing framework docs, scenario executors (arrival-rate vs VU), thresholds, and metrics output. <https://grafana.com/docs/k6/latest/>
- Uvicorn — worker model, `--limit-concurrency`, `--timeout-keep-alive`, and graceful shutdown behaviour. <https://www.uvicorn.org/>
- FastAPI — concurrency and async guidance; when a handler being `async` helps vs hurts. <https://fastapi.tiangolo.com/async/>
- NumPy threading controls — `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`; why pinning matters under multi-worker servers. <https://numpy.org/doc/stable/reference/global_state.html#number-of-threads-used-for-linear-algebra>
- NIST AI Risk Management Framework 1.0 — `MEASURE 2.3` (performance measurement) and `MEASURE 2.6` (deployed system monitoring). Load-test outputs are the baseline signal the deployed monitoring is later compared against. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — relevance is secondary here, but a service that returns 5xx stack traces or metadata under load regresses `ML05:2023 Model Theft` surface; verify error bodies stay opaque across every tuning run. <https://owasp.org/www-project-machine-learning-security-top-10/>
