# exercise-02 — Dockerise the Service · SOLUTION

Paired exercise: `lessons/mod-108-model-deployment-serving/exercises/exercise-02-dockerise-the-service.md` (in the `ml-engineer-learning` repo).

## 2. Solution overview

The exercise asks the learner to turn the FastAPI service from `exercise-01` into a container image that can be scheduled by Kubernetes, scanned by a supply-chain gate, and run without root. "Works on my laptop" is not the bar — the image must boot deterministically, expose the signals Kubernetes expects, and leave no attack surface a 1-MiB runtime does not need.

Design priorities, in order:

1. **Multi-stage build.** Build dependencies (gcc, headers, pip caches) exist only in the builder stage. The runtime stage copies the virtual environment and the application — nothing else. OWASP ML Top 10 `ML06:2023 AI Supply Chain Attacks` and MITRE ATLAS `AML.T0010 ML Supply Chain Compromise` both start with "what is in the image" — a minimal runtime is the cheapest reduction in attack surface available.
2. **Pin the base image by digest, not by tag.** `python:3.12-slim` moves; `python:3.12-slim@sha256:...` does not. A build that succeeds today must produce the same image six months from now, or the signature on it is a lie.
3. **Run as a non-root user with a read-only root filesystem.** A container that writes to `/` is a container whose exploit lands in a writable binary directory. The service needs only its temp dir writable.
4. **Dependencies before code.** The Dockerfile copies `pyproject.toml` / `requirements.txt` first and installs; only then does it copy the application. This keeps the dependency layer cached across every code-only change — a change to `app.py` should not reinstall torch.
5. **Model artifact is a build arg or a runtime mount, not committed to the repo.** Baking a 1 GB `.joblib` into the image is reasonable for immutable deploys; mounting it at runtime is correct when the same code must serve several versions. The solution shows both and names the trade.
6. **Signals and health checks match Kubernetes contracts.** `CMD ["uvicorn", ...]` runs as PID 1 and must forward `SIGTERM`; graceful shutdown lets in-flight requests finish before the container dies. The `HEALTHCHECK` is scoped to the readiness endpoint, not liveness — K8s defines liveness and does not read `HEALTHCHECK`.

| Choice | Alternative considered | Why this one |
|---|---|---|
| `python:3.12-slim-bookworm` pinned by digest | `python:3.12`, `python:3.12-alpine` | `-slim` keeps glibc (manylinux wheels work); Alpine uses musl and silently breaks sklearn/torch wheels |
| Multi-stage with `venv` | Single-stage + `--no-cache-dir` | Multi-stage drops gcc, headers, and the pip cache entirely — half the image size and no compiler in prod |
| Non-root `appuser` (uid 10001) | Default root | Root in a container is root on the host if a CVE drops |
| `--read-only` root filesystem | Writable rootfs | Forces the service to declare what it writes; prevents post-exploit tool-dropping |
| Model baked in (image per version) | Model mounted at `/model` | For immutable deploys with a versioned image registry; variant A documented, variant B shown |
| `uvicorn` directly as `CMD` | `gunicorn -k uvicorn.workers.UvicornWorker` | Uvicorn 0.23+ handles graceful shutdown and `--workers` itself; one fewer process to reason about |
| `HEALTHCHECK` on `/readyz` | On `/healthz` or absent | Docker's healthcheck is for local orchestration (Compose, Swarm); readiness is the right signal — "ready to take traffic" |
| OCI labels (git SHA, build time, version) | No labels | Provenance that survives tagging; `docker inspect` is how incident response finds "what shipped" |
| `.dockerignore` matching `.gitignore` | No ignore file | Prevents `.venv/`, `.git/`, `__pycache__/` from leaking into the build context and the image |

## 2. Worked implementation

### 2.1 Project layout (additions over exercise-01)

```
service/                     # unchanged from exercise-01
pyproject.toml
requirements.txt             # locked output of `pip-compile pyproject.toml`
Dockerfile
.dockerignore
docker-compose.yml           # dev convenience; not prod
scripts/
├── build.sh
└── smoke.sh
```

### 2.2 `.dockerignore` — build-context hygiene

```
.git
.gitignore
.venv
__pycache__
*.pyc
*.pyo
.pytest_cache
.ruff_cache
.mypy_cache
.coverage
htmlcov
tests
docs
.env
.env.*
*.log
artifacts/           # if model is mounted; drop this if model is baked in
```

Anything the Dockerfile does not need must not reach the daemon. A 2 GB `.venv/` or an accidentally-committed `.env` file in the context is both a build slowdown and, often, a secret leak path.

### 2.3 `requirements.txt` — locked, hashable

```
fastapi==0.115.0 --hash=sha256:...
uvicorn[standard]==0.30.6 --hash=sha256:...
pydantic==2.9.2 --hash=sha256:...
scikit-learn==1.5.2 --hash=sha256:...
pandas==2.2.3 --hash=sha256:...
joblib==1.4.2 --hash=sha256:...
numpy==1.26.4 --hash=sha256:...
```

Hashes are not decoration; `pip install --require-hashes -r requirements.txt` refuses any wheel the hash does not match. That is how OWASP ML Top 10 `ML06:2023` is enforced at build time — a compromised PyPI mirror cannot swap `scikit-learn` for a trojaned clone without changing the hash, and the build fails. Generate with `uv pip compile --generate-hashes pyproject.toml -o requirements.txt` (or `pip-compile`).

### 2.4 `Dockerfile` — multi-stage, pinned, non-root

```dockerfile
# syntax=docker/dockerfile:1.7

###############################################################################
# Builder — compiles wheels and populates a venv. Nothing from this stage ships.
###############################################################################
FROM python:3.12.5-slim-bookworm@sha256:a5e4b3c9d9... AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time system deps (compilers for scikit-learn / numpy wheels that need source builds)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential=12.9 \
      gcc=4:12.2.0-3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first — maximises cache hits when application code changes
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --require-hashes --no-deps -r requirements.txt

# Application code
COPY pyproject.toml ./
COPY service/ ./service/
RUN pip install --no-deps -e .

###############################################################################
# Runtime — minimal, non-root, read-only rootfs.
###############################################################################
FROM python:3.12.5-slim-bookworm@sha256:a5e4b3c9d9... AS runtime

ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG MODEL_VERSION=unknown

LABEL org.opencontainers.image.title="churn-predictor" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/your-org/churn-predictor" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_VERSION="${MODEL_VERSION}"

# Minimal runtime deps only (tini for signal handling; curl for HEALTHCHECK)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tini=0.19.0-1 \
      curl=7.88.1-10+deb12u7 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --shell /usr/sbin/nologin --no-create-home app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /build/service /app/service

# Optional: bake a specific model version into the image (variant A).
# Omit this COPY and mount at runtime via `-v /path:/model:ro` (variant B).
COPY --chown=app:app artifacts/churn-v1.joblib /model/churn.joblib
ENV MODEL_URI=/model/churn.joblib

USER app
WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=2s --start-period=20s --retries=3 \
    CMD curl --fail --silent --max-time 1 http://127.0.0.1:8000/readyz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "service.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*", \
     "--timeout-keep-alive", "5", \
     "--limit-concurrency", "256"]
```

Why each block earns its lines:

- **`FROM ... @sha256:...`** — the digest is the only reference that cannot be moved under us. A Dockerfile that passes CI today and silently installs a new base image next week is not reproducible.
- **`PIP_NO_CACHE_DIR=1`** in the builder keeps the pip cache out of the venv; the actual cache is held by BuildKit's `--mount=type=cache,target=/root/.cache/pip`, which does not land in any layer.
- **`--require-hashes --no-deps`** — `--require-hashes` fails the build if the lock and the wheel disagree; `--no-deps` prevents a transitive dependency from being resolved live and sidestepping the lock.
- **Two `COPY` steps for code** — the venv lands first and is cached independently; the app lands after and is the only layer that invalidates on code change. A 500 ms code tweak rebuilds in seconds, not minutes.
- **`tini`** as PID 1 — reaps zombies and forwards `SIGTERM` to uvicorn so graceful shutdown works under Kubernetes termination. Uvicorn handles signals correctly itself, but tini is cheap insurance for subprocesses.
- **`--proxy-headers --forwarded-allow-ips=*`** — only honour `X-Forwarded-*` headers behind an ingress that you trust. In a public-facing deploy, set `--forwarded-allow-ips` to the ingress CIDR.
- **`--limit-concurrency`** — caps in-flight requests per worker; past this uvicorn returns 503 instead of queueing unbounded and OOMing the pod.
- **`HEALTHCHECK` targets `/readyz`** — because a container that is alive but not ready should fail the healthcheck. Kubernetes uses its own probes and ignores `HEALTHCHECK` metadata; this is for Compose, Swarm, and `docker ps` legibility.

### 2.5 `scripts/build.sh` — provenance in every tag

```bash
#!/usr/bin/env bash
set -euo pipefail

GIT_SHA=$(git rev-parse --short=12 HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  GIT_SHA="${GIT_SHA}-dirty"
fi
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
MODEL_VERSION="${MODEL_VERSION:-unknown}"

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --build-arg GIT_SHA="${GIT_SHA}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg MODEL_VERSION="${MODEL_VERSION}" \
  --tag "churn-predictor:${MODEL_VERSION}" \
  --tag "churn-predictor:${GIT_SHA}" \
  --provenance=true \
  --sbom=true \
  .
```

`--provenance=true --sbom=true` emits SLSA-style attestations and an SBOM into the image index. That is the paper trail a scanner (Trivy, Grype) uses to flag `scikit-learn==1.5.2` the day a CVE drops, and the artefact your supply-chain gate verifies before promoting to prod.

### 2.6 `docker-compose.yml` — for local dev only

```yaml
services:
  predictor:
    build:
      context: .
      args:
        MODEL_VERSION: dev
    image: churn-predictor:dev
    read_only: true
    tmpfs:
      - /tmp:size=64m
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    environment:
      MODEL_URI: /model/churn.joblib
      MODEL_VERSION: dev
      LOG_LEVEL: DEBUG
      MAX_BATCH_SIZE: "32"
    volumes:
      - ./artifacts/churn-v1.joblib:/model/churn.joblib:ro
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/readyz"]
      interval: 5s
      timeout: 2s
      retries: 3
```

`read_only: true` + `tmpfs:/tmp` is the production-mirroring posture; `cap_drop: ALL` + `no-new-privileges` is the one that catches most "the container escaped" classes before they happen.

## 3. Validation steps

Run these in order; each exists to catch one real failure mode.

1. **Build is reproducible** — run `./scripts/build.sh` twice without changes and compare `docker image inspect --format '{{.Id}}' churn-predictor:<sha>`. The content digest must match (byte-for-byte reproducibility is harder; image content digest is the minimum bar).
2. **Non-root** — `docker run --rm churn-predictor:<sha> id` prints `uid=10001(app) gid=10001(app)`. If this says `uid=0`, the Dockerfile's `USER app` line is missing or overridden.
3. **Read-only root** — `docker run --rm --read-only churn-predictor:<sha> touch /proof` must fail with `Read-only file system`. If it succeeds, the compose file is lying and prod will have a writable rootfs.
4. **No compiler in runtime** — `docker run --rm churn-predictor:<sha> which gcc` prints nothing and exits 1. If it finds gcc, the multi-stage split failed.
5. **Smoke run** — `./scripts/smoke.sh` starts the container, waits for `/readyz` to return 200, POSTs the golden request from exercise-01, asserts a 200 with a valid body, then stops the container. Catches regressions in the app contract.
6. **`HEALTHCHECK` latches** — `docker inspect --format '{{json .State.Health}}'` reports `"Status": "healthy"` once the lifespan hook finishes, and `"Status": "unhealthy"` if you kill the python process. Catches a healthcheck pointed at `/healthz` instead of `/readyz`.
7. **Graceful shutdown** — start an open batch request, then `docker stop <id>`. The request should either complete or return a clean 503, not a connection reset. Catches a missing `tini` or a signal-ignoring CMD.
8. **Image size** — `docker image ls` reports the runtime image. Rule of thumb: FastAPI + sklearn bundle lands around 250–350 MB; a value above 800 MB usually means the builder stage leaked (compilers, caches, or site-packages for the wrong Python).
9. **Supply-chain scan** — `trivy image churn-predictor:<sha> --severity HIGH,CRITICAL` reports zero high-or-critical CVEs on a fresh base pin. Any finding on `python-slim` means the digest pin is stale.
10. **SBOM emitted** — `docker buildx imagetools inspect churn-predictor:<sha> --format '{{json .SBOM}}'` is non-empty. This is what a prod gate (cosign + policy-controller) will verify.
11. **Labels carry provenance** — `docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'` prints the git SHA. Incident response reads this when "what commit shipped?" is urgent.

## 4. Rubric / review checklist

Weighted, `/20` points total.

| # | Dimension | Weight | Passing criterion |
|---|---|---|---|
| 1 | Multi-stage build | 3 | Runtime stage has no compilers, no pip cache, no build tools; image size matches the heuristic |
| 2 | Base image pinned by digest | 2 | `FROM ... @sha256:...`; both stages pinned to the same digest |
| 3 | Dependency pinning with hashes | 2 | `requirements.txt` has hashes; install uses `--require-hashes` |
| 4 | Non-root user | 2 | Explicit `useradd`, `USER app`; `docker run id` confirms uid≠0 |
| 5 | Read-only rootfs compatible | 2 | Nothing in the service writes outside `/tmp`; compose uses `read_only: true` |
| 6 | Signal / shutdown handling | 2 | `tini` as ENTRYPOINT (or documented alternative); graceful shutdown on `docker stop` |
| 7 | `HEALTHCHECK` on `/readyz` | 1 | Correct endpoint, sane interval + start-period |
| 8 | OCI labels | 1 | `org.opencontainers.image.{revision,version,created,source}` populated from build args |
| 9 | `.dockerignore` | 1 | `.git`, `.venv`, `__pycache__`, tests, local caches excluded from context |
| 10 | Config via env | 1 | `MODEL_URI`, `MODEL_VERSION`, `LOG_LEVEL`, `MAX_BATCH_SIZE` all overridable at `docker run` |
| 11 | Supply-chain attestations | 2 | `--provenance=true --sbom=true`; Trivy scan clean at HIGH/CRITICAL on fresh pin |
| 12 | Model-artefact strategy stated | 1 | README names "baked" vs "mounted" and which the image uses, with rationale |

Items 1, 2, 3, and 4 are mandatory. An image with the compiler in it, a floating tag, unhashed dependencies, or a root user cannot score above 10; those are the issues a security review will block on, not polish items.

## 5. Common mistakes

- **Floating tag on the base image.** `FROM python:3.12-slim` passes CI today, pulls a new patch tomorrow, and the lock file is no longer the truth. Pin by digest; `docker buildx imagetools inspect python:3.12-slim` prints the current one.
- **`pip install -r requirements.txt` without hashes.** The lock is advisory, not enforced; a compromised mirror or a maliciously typo-squatted dep lands silently. `--require-hashes` makes the lock load-bearing.
- **`COPY . .` as the first instruction.** Every code change invalidates the dependency install layer. Install dependencies first, copy the app last.
- **Running as root.** `WORKDIR /app && CMD ["uvicorn", ...]` with no `USER` leaves PID 1 as root. A CVE in a transitive dep (and there is always one eventually) becomes a root-in-container exploit.
- **`CMD` as a shell form.** `CMD uvicorn ...` runs under `/bin/sh -c`, which does not forward `SIGTERM`. Use the exec form (`CMD ["uvicorn", ...]`) or front it with `tini` / `dumb-init`.
- **Alpine for ML workloads.** Alpine ships musl libc; many `scikit-learn` / `torch` / `numpy` wheels are manylinux-only, and pip falls back to source builds that may or may not succeed. Use `-slim` (glibc Debian) unless you have a specific reason not to.
- **Baking secrets.** `ARG DATABASE_URL=...` or `COPY .env .` writes secrets into layers; they persist in the image history even after a later `RUN rm`. Secrets are runtime config, not build config.
- **`HEALTHCHECK` on `/healthz`.** The liveness-vs-readiness distinction from exercise-01 collapses here: a container that is alive but has not finished loading is reported as healthy, Kubernetes puts it in service, and the first requests all 503.
- **`--workers 8` on a 1-core limit.** Workers each take a full copy of the model in RAM and compete for the same CPU; more workers past the pod's CPU quota slows every request. Size workers to `min(cpu_limit, 2 * cpu_limit)` and measure.
- **Not setting `PYTHONUNBUFFERED=1`.** Logs sit in the Python stdout buffer and never flush until the container stops. The aggregator sees "silent" pods and the operator debugs blind.
- **Trusting `latest`.** `docker run churn-predictor:latest` in prod is a pointer that an attacker who controls your registry can rewrite. Deploy by digest (`@sha256:...`) or by immutable tag (`${GIT_SHA}`).
- **Giant model, no mount.** Baking a 5 GB model into the image makes every pod cold-start pull 5 GB. Mount from object storage (S3 / GCS / CSI) or a shared read-only volume, and use the baked-in pattern only for small (< 500 MB) artefacts.

## 6. References

- Dockerfile reference and best-practices (multi-stage, build secrets, BuildKit, cache mounts). <https://docs.docker.com/reference/dockerfile/> and <https://docs.docker.com/build/building/best-practices/>
- Docker BuildKit — `--provenance`, `--sbom`, cache mounts, platform matrix. <https://docs.docker.com/build/attestations/>
- OCI image-spec — `org.opencontainers.image.*` label conventions. <https://github.com/opencontainers/image-spec/blob/main/annotations.md>
- NIST SP 800-190 — Application Container Security Guide (non-root, read-only rootfs, minimal base images). <https://csrc.nist.gov/publications/detail/sp/800-190/final>
- OWASP Machine Learning Security Top 10 — `ML06:2023 AI Supply Chain Attacks` (why pinned, hashed, scanned images are the baseline). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — `AML.T0010 ML Supply Chain Compromise` — the container is one of the links in that chain; the Dockerfile is the primary mitigation surface. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework 1.0 — `MANAGE 2.1` (documented provenance across the ML lifecycle) as justification for OCI labels and SBOMs. <https://www.nist.gov/itl/ai-risk-management-framework>
