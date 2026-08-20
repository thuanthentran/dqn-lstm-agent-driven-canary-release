# checkoutservice

Google Cloud microservices-demo checkout service — extended with a **fault injection layer** for canary-release benchmarking research.

---

## Running locally

```bash
dep ensure --vendor-only   # restore vendor/ dependencies (legacy dep tool)
# OR
go mod download            # preferred — uses go.mod / go.sum
go build -o checkoutservice .
```

---

## Chaos Fault Injection

A `chaos` sub-package was added to enable **reproducible, parameterized fault injection** controlled entirely via **environment variables**. No source-code changes are required between experiment runs — only env-var values change.

### Design goals

| Goal | How it is achieved |
|---|---|
| **Fair comparison** | Both Rule-Based and RL Agent controllers observe the same Prometheus metrics produced by the same fault sequence |
| **Reproducibility** | A fixed `CHAOS_RANDOM_SEED` guarantees identical random sequences across repeated runs |
| **Zero overhead when disabled** | All variables default to "off"; `CHAOS_ENABLED=false` makes every call a no-op |
| **Paper transparency** | Fault logic lives in `chaos/chaos.go` — readable, reviewable, citable |

### Architecture

```
gRPC request arrives
        │
        ▼
 chaosInterceptor  ←─── registered via grpc.ChainUnaryInterceptor in main.go
        │
        ├─ 1. sampleLatency()  → time.Sleep(delay)
        ├─ 2. Bernoulli trial  → return gRPC error (if rand < error_rate)
        │
        ▼
 handler(ctx, req)  ← original PlaceOrder / health-check logic
```

### Environment variables reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `CHAOS_ENABLED` | bool | `false` | **Master switch.** Must be `"true"` to activate any fault. |
| `CHAOS_RANDOM_SEED` | int64 | `42` | RNG seed. Fix this across all runs of the same scenario for reproducibility. |
| **Latency** | | | |
| `CHAOS_LATENCY_MS_MIN` | float64 | `0` | Lower bound of injected latency (ms). |
| `CHAOS_LATENCY_MS_MAX` | float64 | `0` | Upper bound (ms). Set to `0` to disable latency injection. |
| `CHAOS_LATENCY_DIST` | string | `lognormal` | Sampling distribution: `none` · `fixed` · `uniform` · `normal` · `lognormal` |
| **Error Rate** | | | |
| `CHAOS_ERROR_RATE` | float64 | `0.0` | Probability `[0.0, 1.0]` that a request returns a gRPC error. |
| `CHAOS_ERROR_CODE` | string | `UNAVAILABLE` | gRPC status code returned on injected errors. Any standard gRPC code name. |
| **Resource pressure** | | | |
| `CHAOS_CPU_PERCENT` | int | `0` | Target CPU load from a background busy-loop goroutine (0–100). |
| `CHAOS_MEM_ALLOC_MB` | int | `0` | MB to allocate on startup and hold forever (baseline memory leak). |

### Distribution reference

| `CHAOS_LATENCY_DIST` | Parameters used | Best for |
|---|---|---|
| `none` / `0 max` | — | Disable latency |
| `fixed` | midpoint = (min+max)/2 | Deterministic baseline / CPU & memory scenarios |
| `uniform` | Uniform[min, max] | Simple, easy to reason about |
| `normal` | μ=(min+max)/2, σ=(max−min)/6 | Symmetric jitter, minimal outliers |
| `lognormal` | Fitted to [min, max] 5th–95th percentile | **Recommended** — closest to real production tail-latency |

### Scenario quick-reference (5 benchmark scenarios)

```bash
# S1 — High Latency Ramp  (use inject_fault.py to step through these)
CHAOS_ENABLED=true CHAOS_LATENCY_DIST=lognormal \
  CHAOS_LATENCY_MS_MIN=10 CHAOS_LATENCY_MS_MAX=100   # t=0  baseline
  CHAOS_LATENCY_MS_MIN=50 CHAOS_LATENCY_MS_MAX=300   # t=2m ramp
  CHAOS_LATENCY_MS_MIN=200 CHAOS_LATENCY_MS_MAX=800  # t=4m warning
  CHAOS_LATENCY_MS_MIN=500 CHAOS_LATENCY_MS_MAX=1500 # t=6m critical

# S2 — CPU Spike
CHAOS_ENABLED=true CHAOS_CPU_PERCENT=70

# S3 — Memory Leak (static baseline at pod startup)
CHAOS_ENABLED=true CHAOS_MEM_ALLOC_MB=200

# S4 — Error Rate Burst  (toggle CHAOS_ERROR_RATE between steps)
CHAOS_ENABLED=true CHAOS_ERROR_RATE=0.8 CHAOS_ERROR_CODE=UNAVAILABLE

# S5 — Cascading Failure (combined: latency + error on canary pod)
CHAOS_ENABLED=true \
  CHAOS_LATENCY_DIST=lognormal CHAOS_LATENCY_MS_MIN=100 CHAOS_LATENCY_MS_MAX=500 \
  CHAOS_ERROR_RATE=0.4 CHAOS_ERROR_CODE=UNAVAILABLE
```

### Kubernetes — adding env vars to a canary pod

In `gitops/releases/checkoutservice-values.yaml`, add under `env:`:

```yaml
env:
  # ... existing vars ...
  - name: CHAOS_ENABLED
    value: "true"
  - name: CHAOS_RANDOM_SEED
    value: "42"
  - name: CHAOS_LATENCY_DIST
    value: "lognormal"
  - name: CHAOS_LATENCY_MS_MIN
    value: "50"
  - name: CHAOS_LATENCY_MS_MAX
    value: "300"
  - name: CHAOS_ERROR_RATE
    value: "0.0"
```

> **Important**: Only the **canary** revision should have `CHAOS_ENABLED=true`.  
> The **stable** revision must always run with `CHAOS_ENABLED=false` (default) to provide a clean baseline for metric comparison.

---

## Files changed

| File | Change |
|---|---|
| `chaos/chaos.go` | **New** — fault injection package (Config, Apply, UnaryServerInterceptor, distributions) |
| `main.go` | Added `chaos` import; `chaos.LoadFromEnv()` call; `grpc.ChainUnaryInterceptor(chaosCfg.UnaryServerInterceptor())` on the gRPC server |
| `README.md` | This documentation |

## Trigger CI Build