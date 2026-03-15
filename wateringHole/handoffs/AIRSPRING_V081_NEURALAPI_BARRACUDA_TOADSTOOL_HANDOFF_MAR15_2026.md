# airSpring V0.8.1 — neuralAPI + BarraCuda/ToadStool Evolution Handoff

> **Date**: March 15, 2026
> **From**: airSpring
> **To**: barraCuda, ToadStool, coralReef, biomeOS
> **License**: AGPL-3.0-or-later
> **Covers**: neuralAPI integration, barracuda usage analysis, evolution opportunities

---

## Executive Summary

- airSpring v0.8.1 is the first Spring implementing all 3 neuralAPI primal enhancements (structured metrics, operation dependencies, cost estimates)
- We consume **34 barracuda primitives** across 6 domains (ops, linalg, stats, pde, optimize, special) — all stable, all upstream
- **Zero local WGSL shaders** — Write→Absorb→Lean complete since v0.7.2
- Our neuralAPI integration patterns are reusable by other Springs and by barracuda/ToadStool themselves
- 847 lib tests, 41 capabilities, 0 clippy pedantic warnings

---

## Part 1: BarraCuda Usage Summary (What We Consume)

### By Domain

| barracuda Domain | Primitives Used | airSpring Module(s) | Notes |
|-----------------|----------------|---------------------|-------|
| `ops` | `batched_elementwise_f64` (ops 0-19), `fused_map_reduce_f64`, `variance_f64_wgsl`, `moving_window_stats`, `kriging_f64`, `autocorrelation_f64_wgsl`, `bio::diversity_fusion` | `gpu::*` (25 modules) | Core GPU dispatch — all upstream, all stable |
| `optimize` | `brent`, `brent_gpu`, `nelder_mead`, `multi_start` | `eco::richards`, `gpu::van_genuchten`, `gpu::isotherm`, `gpu::infiltration` | VG inverse, Richards PDE, isotherm fitting |
| `pde` | `richards`, `richards_gpu`, `crank_nicolson` | `gpu::richards` | Richards unsaturated flow — airSpring contributed upstream (S40) |
| `stats` | `bootstrap`, `jackknife`, `diversity`, `normal`, `pearson_correlation`, `regression::fit_linear`, `rmse`, `metrics` | `gpu::jackknife`, `gpu::bootstrap`, `eco::*` | Uncertainty quantification + validation |
| `special` | `gamma::regularized_gamma_p`, `gamma::ln_gamma` | `eco::drought_index` | SPI drought index — leaned to upstream |
| `linalg` | `ridge::ridge_regression` | `eco::correction` | Bias correction |
| `device` | `WgpuDevice`, `PrecisionRoutingAdvice`, `Fp64Strategy`, `GpuDriverProfile`, `F64BuiltinCapabilities` | `gpu::device_info`, all GPU modules | Hardware discovery + precision routing |
| `validation` | `ValidationHarness`, `exit_no_gpu`, `gpu_required` | `validation.rs`, all `validate_*` bins | Shared validation framework |
| `tolerances` | `check`, `Tolerance` | `tolerances.rs` | Shared tolerance checking |
| `shaders::provenance` | `SpringDomain` | cross-spring benchmarks | Cross-spring shader provenance tracking |

### Stability Assessment

All barracuda APIs we use have been stable across 3+ releases (0.3.3→0.3.5+HEAD).
No breaking changes encountered. `PrecisionRoutingAdvice` was the most significant
API evolution (added v0.3.3) and we adopted it immediately.

**Verdict: barracuda is production-quality infrastructure for airSpring.**

---

## Part 2: What airSpring Contributed Upstream

| Contribution | barracuda Module | ToadStool Sprint | Status |
|-------------|-----------------|-----------------|--------|
| Richards PDE solver | `pde::richards` | S40 | **Absorbed** |
| Stats metrics re-exports | `stats::metrics` | S64 | **Absorbed** |
| SCS-CN runoff | `ops` op=17 | S66 | **Absorbed** |
| Stewart/Makkink/Turc/Hamon/Blaney-Criddle ET₀ | `ops` ops 14-16, 19 | S66 | **Absorbed** |
| Yield response | `ops` op=18 | S66 | **Absorbed** |
| `pow_f64` fractional exponent fix | `shaders` | TS-001 | **Merged** |
| Reduce buffer N≥1024 fix | `ops` | TS-004 | **Merged** |
| acos precision boundary fix | `shaders` | TS-003 | **Merged** |

**All local GPU ops absorbed.** airSpring has zero local WGSL shaders.
The Write→Absorb→Lean cycle is complete and proven.

---

## Part 3: neuralAPI Integration Patterns (Reusable by All Primals)

### Enhancement 1: Structured Metrics (5 min effort)

```rust
// Passive logging — biomeOS Pathway Learner scrapes these
eprintln!(
    "[metrics] primal_id={PRIMAL_NAME} operation={op} latency_ms={ms:.2} success={ok}"
);

// Active reporting — when BIOMEOS_METRICS_SOCKET is set
if let Ok(socket_path) = std::env::var("BIOMEOS_METRICS_SOCKET") {
    // fire-and-forget JSON to Unix socket
}
```

**barraCuda action**: Add structured logging to `barracuda::device::WgpuDevice::submit()` and key ops. biomeOS can then learn GPU dispatch latencies across Springs.

### Enhancement 2: Operation Dependencies (30 min effort)

```json
{
  "science.et0": ["weather_data"],
  "science.vpd": ["temperature_data", "humidity_data"],
  "science.gdd": ["temperature_data"]
}
```

biomeOS auto-detects that `vpd` and `gdd` are independent and can run them in parallel.

**barraCuda action**: Declare dependencies for key ops (e.g., `batched_elementwise_f64` requires `device + input_buffer`, `fused_map_reduce` requires `device + input_buffer`). This helps biomeOS batch GPU submissions.

### Enhancement 3: Cost Estimates (1 hr effort)

```json
{
  "science.et0": { "latency_ms": 0.5, "cpu": "low", "memory_bytes": 256 },
  "compute.execute": { "latency_ms": 5.0, "cpu": "high", "memory_bytes": 65536 }
}
```

**ToadStool action**: Expose cost estimates per compute workload. biomeOS scheduler can then optimize GPU vs CPU routing based on actual measured costs.

---

## Part 4: Evolution Opportunities for barraCuda

### Priority 1: GPU Provenance Tracking

airSpring's `ipc/provenance.rs` records GPU compute steps in the provenance DAG:

```rust
record_gpu_step(&session_id, &serde_json::json!({
    "shader": "et0_fao56",
    "device": "rtx4070",
    "precision": "f64_native",
    "input_elements": 10000,
    "latency_ms": 2.3,
}));
```

**barraCuda opportunity**: Add a `provenance` feature flag to `WgpuDevice` that automatically emits structured provenance events for every shader dispatch. Springs would get GPU provenance "for free" without manual recording.

### Priority 2: neuralAPI-Aware Dispatch

Currently `PrecisionRoutingAdvice` routes based on hardware. With neuralAPI cost estimates, biomeOS could influence routing:

- High-latency GPU op on busy device → route to CPU
- Low-latency op batch → fuse into single GPU submission
- Cross-Spring op → route to the Spring with warm device

**barraCuda opportunity**: Accept a `DispatchHint` from biomeOS that biases precision routing based on Pathway Learner observations.

### Priority 3: Streaming Pipeline Provenance

airSpring's `gpu::seasonal_pipeline` chains ops (ET₀→Kc→WB→Yield). Each stage could emit provenance vertices automatically if barracuda tracked pipeline stage boundaries.

**barraCuda opportunity**: Add optional pipeline-level provenance hooks to `UnidirectionalPipeline` and `gpu_step()`.

---

## Part 5: Evolution Opportunities for ToadStool

### 1. `compute.provenance` Capability

When ToadStool exposes a `compute.provenance` capability, Springs can merge GPU execution traces with experiment DAGs automatically. airSpring's `provenance.complete` already expects this:

```json
{
  "gpu_trace": "${TOADSTOOL_TRACE_ID}",
  "shader_chain": ["et0_fao56", "kc_climate", "water_balance"],
  "precision": "f64_native",
  "device": "rtx4070"
}
```

### 2. Pathway Learner Metrics from ToadStool

ToadStool processes GPU workloads for multiple Springs. Structured metrics from ToadStool would let biomeOS learn:
- Which workloads benefit from GPU vs CPU
- Optimal batch sizes per shader family
- Device warmup patterns

### 3. Graph-Based Compute Orchestration

airSpring's `airspring_provenance_pipeline.toml` shows how graphs orchestrate multi-step compute. ToadStool could evolve to accept graph-based workload descriptions instead of individual `compute.execute` calls.

---

## Part 6: coralReef Integration Status

airSpring is pinned to coralReef Phase 10. Current integration:
- `barracuda::device::WgpuDevice` uses coralReef for adapter discovery
- NVK (Mesa Vulkan) path validated on Titan V (24/24 PASS)
- Zero-output detection + CPU fallback implemented
- `BARRACUDA_GPU_ADAPTER` env var for multi-GPU selection

**No new coralReef requirements from v0.8.1.** neuralAPI integration is above the GPU layer.

---

## Part 7: Cross-Spring Learnings

### What Works Well

1. **`batched_elementwise_f64`** is the workhorse — 20 of our 25 GPU modules use it
2. **`PrecisionRoutingAdvice`** eliminated all our precision-related GPU bugs
3. **`ValidationHarness`** shared across all Springs — excellent for hotSpring-pattern validation
4. **Write→Absorb→Lean** cycle works — 6 local ops absorbed, zero local WGSL remaining

### What Could Be Better

1. **No structured metrics in barracuda** — we had to add metrics at the primal level
2. **No pipeline-level provenance** — we track GPU steps manually in `record_gpu_step()`
3. **No `DispatchHint`** from biomeOS — precision routing is purely hardware-based
4. **No `compute.provenance`** from ToadStool — GPU traces are airSpring-local

### Patterns Other Springs Should Adopt

1. `auto_record_provenance()` in dispatch — provenance becomes automatic
2. `ecology.experiment` high-level method — single call for full provenance lifecycle
3. `NestGateProvider` 3-tier routing — graceful degradation to standalone
4. `capability.list` with deps + costs — biomeOS can learn from you immediately

---

## Action Items

| # | Owner | Action | Priority |
|---|-------|--------|----------|
| 1 | barraCuda | Add structured metrics to `WgpuDevice::submit()` | High |
| 2 | barraCuda | Add `provenance` feature flag for automatic GPU provenance | Medium |
| 3 | barraCuda | Accept `DispatchHint` for biomeOS-influenced routing | Medium |
| 4 | ToadStool | Expose `compute.provenance` capability | High |
| 5 | ToadStool | Add structured metrics to `compute.execute` | High |
| 6 | ToadStool | Evolve to graph-based workload descriptions | Low |
| 7 | coralReef | No new requirements from airSpring v0.8.1 | — |
| 8 | All Springs | Adopt neuralAPI 3 enhancements (metrics, deps, costs) | High |

---

*AGPL-3.0-or-later — airSpring v0.8.1 (March 15, 2026)*
