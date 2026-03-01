# airSpring V038 — Complete GPU Rewire + Cross-Spring Benchmark

**Date:** March 1, 2026
**From:** airSpring team
**To:** ToadStool / BarraCuda team
**airSpring version:** v0.5.6+
**ToadStool HEAD:** `1dd7e338` (S70+++)

---

## Executive Summary

airSpring has completed the full GPU rewire against ToadStool S70+:

1. **All 6 batched elementwise ops** (0, 1, 5, 6, 7, 8) dispatch via GPU-first paths
2. **Exp 057 validation**: 26/26 checks pass (GPU↔CPU parity, scaling, seasonal pipeline)
3. **Cross-spring benchmark**: 35/35 pass (was 30/30) with ops 5-8 GPU validation
4. **GPU stats module**: Wired `linear_regression_f64` and `matrix_correlation_f64` (neuralSpring S69) for sensor calibration regression and multi-variate soil analysis
5. **640 lib tests** (was 636), zero clippy warnings, 22.7× Rust vs Python

---

## What Changed

### 1. Exp 057 — GPU Ops 5-8 Rewire Validation + Benchmark

New binary `validate_gpu_rewire_benchmark` validates all 6 batched ops against CPU baselines with timing benchmarks and cross-spring provenance tracking.

**Key results (Titan V, NVK driver):**

| Operation | N | GPU (ms) | CPU (ms) | Parity |
|-----------|---|----------|----------|--------|
| FAO-56 ET₀ (op=0) | 1,000 | 1.92 | 0.09 | exact |
| SensorCal (op=5) | 2,000 | 1.46 | 0.00 | < 1e-15 |
| Hargreaves (op=6) | 2,000 | 1.67 | 0.10 | < 0.10 (NVK acos polyfill) |
| Kc Climate (op=7) | 2,000 | 1.76 | 0.02 | < 1e-15 |
| DualKc Ke (op=8) | 1,000 | 1.89 | — | [0, 1.5) valid |
| Seasonal (153d) | 153 | 3.34 | 0.02 | ET₀ < 1%, yield < 5% |

**GPU throughput scaling** (Hargreaves op=6):
- N=100: 64K items/s
- N=1,000: 636K items/s
- N=10,000: 4.5M items/s
- N=50,000: 11.5M items/s

At N≤2000, GPU overhead (~1.5ms dispatch) dominates — CPU wins. At N>10K, GPU wins decisively. The crossover is ~5K items for element-wise ops.

### 2. GPU Stats Module (`gpu::stats`)

Wired `barracuda::ops::stats_f64` (neuralSpring S69 → ToadStool absorption):

- **`sensor_regression_gpu()`**: Batched polynomial OLS for raw counts → VWC calibration
- **`soil_correlation_gpu()`**: Pearson correlation matrix for multi-variate soil data
- **`predict_vwc()`**: Apply fitted coefficients to predict VWC

Tests skip gracefully on drivers without `enable f64` WGSL support (NVK).

### 3. Updated Cross-Spring Benchmark

`bench_cross_spring` now includes 35 benchmarks (was 30):
- +5 new: SensorCal GPU, Hargreaves GPU, Kc Climate GPU, DualKc Ke GPU, op=5-8 parity sweep
- Updated provenance table with 4 new entries (stats_f64, seasonal_pipeline, brent_f64, hydrology GPU-first)

### 4. Provenance Table Update

`PROVENANCE` in `device_info.rs` now tracks 19 entries (was 15), including:

| Shader | Origin | airSpring Use |
|--------|--------|---------------|
| `batched_elementwise_f64.wgsl` | multi-spring convergence | 6 ops GPU-first (was 2) |
| `hydrology (CPU→GPU kernel)` | airSpring | GPU-first since v0.5.6 |
| `stats_f64 (GPU statistics)` | neuralSpring S69 | Sensor regression, soil analysis |
| `seasonal_pipeline.wgsl (fused)` | airSpring → ToadStool S70+ | Future: fused pipeline |
| `brent_f64.wgsl (root-finding)` | airSpring → ToadStool S70+ | Future: GPU VG inverse |

---

## Cross-Spring Shader Evolution

The `batched_elementwise_f64.wgsl` shader is a convergence point for the entire ecoPrimals ecosystem:

```
hotSpring  → math_f64.wgsl (exp, log, sin, cos, acos, pow)
           → df64_core (double-float f32 pairs, ~48-bit)
           → Universal precision S67-68 (f64 canonical)
wetSpring  → diversity batch patterns (Shannon/Simpson)
           → moving_window_f64, kriging_f64
neuralSpring → BatchedElementwiseF64 orchestrator
             → ValidationHarness (S58), stats_f64 GPU
airSpring  → ops 5-8 domain (SensorCal, HG, Kc, DualKc)
           → hydrology stats → barracuda::stats::hydrology
groundSpring → MC ET₀ xoshiro + Box-Muller GPU RNG

ToadStool S70+: unified absorption from ALL Springs
```

---

## What ToadStool Should Know

### 1. NVK `acos_f64` Polyfill Drift

The NVK driver's polyfill `acos` accumulates error through:
`sunset_hour_angle = acos(-tan(lat) * tan(declination))`
→ `Ra = f(ws, lat, declination)`
→ `ET₀ = 0.0023 * (T + 17.8) * Ra^0.5 * (Tmax - Tmin)`

Per-point error up to ~0.07 mm/day. Seasonal aggregate stays within 0.36%. Native f64 drivers achieve < 0.001 mm/day.

### 2. `stats_f64` Compilation Issue on NVK

`linear_regression_f64.wgsl` and `matrix_correlation_f64.wgsl` fail shader compilation on NVK because `enable f64` is not supported. These work on native f64 drivers (Titan V with proprietary driver, A100, etc.).

### 3. Remaining airSpring GPU Gaps (Tier B)

| Gap | Blocker | ToadStool Action |
|-----|---------|------------------|
| MC ET₀ GPU | No `mc_et0_propagate_f64.wgsl` | Write shader + executor |
| Batch Nelder-Mead | Serial NM only | Multi-start parallel NM shader |
| VG inverse GPU | `brent_f64.wgsl` line 49 bug | Fix and add Rust executor |
| Water Balance stateful | Sequential day-over-day state | Wire via `StatefulPipeline` |
| Seasonal pipeline fused | WGSL exists, no Rust executor | Write `SeasonalPipelineF64` executor |
| Pedotransfer GPU | No shader | Polynomial evaluation shader |
| Cover crop diversity GPU | No batch shader | Extend `fused_map_reduce_f64` |

### 4. `StreamingPipeline` for Seasonal Pipeline

`StreamingPipeline` from `staging::pipeline` could chain ET₀ → Kc → WB → Yield as separate stages with `StageLink` intermediate buffers. However, the fused `seasonal_pipeline.wgsl` is a better target for single-dispatch performance.

---

## Cumulative State

| Metric | Value |
|--------|-------|
| airSpring version | v0.5.6+ |
| Barracuda lib tests | **640** |
| metalForge lib tests | **57** |
| Validation binaries | **61** |
| Exp 055 GPU pipeline checks | **78/78** |
| Exp 057 GPU rewire checks | **26/26** |
| Cross-spring benchmarks | **35/35** |
| metalForge mixed-hardware checks | **104** |
| Rust vs Python speedup | **22.7×** (18/18 parity) |
| GPU ops wired | 6 element-wise + 2 stats + reduce + stream + kriging |
| Seasonal pipeline | GPU Stages 1-2 (ET₀ + Kc), CPU Stages 3-4 |
| ToadStool HEAD | `1dd7e338` (S70+++) |
| Clippy (pedantic) | **0 warnings** |

---

## Recommended ToadStool Actions

1. **Fix `brent_f64.wgsl` line 49** — enables airSpring VG inverse GPU path
2. **Write `SeasonalPipelineF64` Rust executor** — the WGSL is ready
3. **Add `mc_et0_propagate_f64.wgsl`** — Box-Muller + FAO-56 fused for uncertainty
4. **Absorb `sensor_regression_gpu` pattern** — polynomial OLS for IoT sensor fleets
5. **Consider workgroup size tuning** — `seasonal_pipeline.wgsl` uses `@workgroup_size(1)`, consider 64 or 256 for throughput
