# airSpring V053 — ToadStool/BarraCuda Absorption Guide & Evolution Intelligence

**Date**: 2026-03-02
**From**: airSpring v0.6.8 (77 experiments, 846 lib + 61 forge, 86 binaries)
**To**: ToadStool/BarraCuda team
**ToadStool Pin**: S87 (`2dc26792`)
**Direction**: airSpring → ToadStool (unidirectional)

---

## Executive Summary

This handoff consolidates everything the ToadStool/BarraCuda team needs to absorb
airSpring's remaining local work and evolve the shared infrastructure. It covers:

1. **6 pending WGSL ops** (V051 proposal — ops 14-19 for `batched_elementwise_f64`)
2. **Cross-spring evolution intelligence** from Exp 077 (32/32 provenance benchmark)
3. **GPU driver insights** from live hardware testing (RTX 4070 + Titan V)
4. **What airSpring learned** about ToadStool's architecture that guides future evolution
5. **Precision and performance data** from CPU↔GPU benchmarks

---

## Part 1: Pending Absorption — 6 Local WGSL Ops → f64 Canonical

### 1.1 Current State

airSpring ships `local_elementwise.wgsl` — a single f32 compute shader with 6
element-wise operations dispatched via `gpu::local_dispatch::LocalElementwise`.
This is a stopgap: the math works (Exp 075 validates CPU/GPU parity) but precision
is f32 (~7 decimal digits) rather than the f64 canonical (~15 digits) that ToadStool
provides through `compile_shader_universal`.

### 1.2 Proposed Ops 14-19 for `batched_elementwise_f64.wgsl`

| Proposed Op | Name | Formula | Stride | Helper Functions Needed |
|:-----------:|------|---------|:------:|------------------------|
| 14 | SCS-CN Runoff | `Q = (P − λS)² / (P − λS + S)`, `S = 25400/CN − 254` | 3 | None (pure arithmetic) |
| 15 | Stewart Yield | `Ya/Ymax = 1 − Ky × (1 − ETa/ETc)` | 2 | None (pure arithmetic) |
| 16 | Makkink ET₀ | `0.61 × (Δ/(Δ+γ)) × Rs/λ − 0.12` | 3 | `atm_pressure`, `psychrometric`, `sat_vp`, `vp_slope` |
| 17 | Turc ET₀ | `0.013 × T/(T+15) × (23.8856Rs + 50)`, humidity branch | 3 | None (pure arithmetic + branch) |
| 18 | Hamon PET | `0.1651 × N × ρsat(T)` | 3 | `daylight_hr(lat, doy)` — needs `acos_f64` from `math_f64.wgsl` |
| 19 | Blaney-Criddle | `p × (0.46T + 8.13)`, daylight fraction | 3 | `daylight_hr(lat, doy)` — same as op 18 |

### 1.3 Helper Function Mapping to Existing ToadStool Infrastructure

All helper functions already exist in ToadStool:

| Local Helper | ToadStool Equivalent | Source |
|-------------|---------------------|--------|
| `atm_pressure(elev)` | FAO-56 Eq. 7 — already in `batched_elementwise_f64` op=0 preamble | airSpring V039 |
| `psychrometric(P)` | FAO-56 Eq. 8 — already in op=0 | airSpring V039 |
| `sat_vp(T)` | `exp_f64()` in `math_f64.wgsl` | hotSpring precision |
| `vp_slope(T)` | Derivative of sat_vp — already in op=0 | airSpring V039 |
| `daylight_hr(lat, doy)` | Needs `acos_f64()` from `math_f64.wgsl` + `sin_f64()` | hotSpring precision |

The `daylight_hr` function (FAO-56 Eq. 34) is the only one not already present in the
elementwise preamble. It computes sunrise hour angle → day length for latitude-dependent
ET₀ methods. Formula:

```
dr = 1 + 0.033 * cos(2π * doy / 365)
δ  = 0.4093 * sin(2π * doy / 365 − 1.39)
ωs = acos(-tan(lat) * tan(δ))
N  = 24 * ωs / π
```

### 1.4 Precision Characteristics

Current f32 precision (local shader):

| Op | Max |Δ| vs CPU f64 | Acceptable? |
|----|---------------------|------------|
| SCS-CN Runoff | < 0.01 mm | Yes for event-scale, tight for design |
| Stewart Yield | < 0.001 | Yes |
| Makkink ET₀ | < 0.002 mm/day | Yes for daily, marginal for interannual |
| Turc ET₀ | < 0.002 mm/day | Yes for daily |
| Hamon PET | < 0.002 mm/day | Yes for daily |
| Blaney-Criddle | < 0.003 mm/day | Marginal — daylight trig accumulates |

After f64 absorption: all ops would match CPU within floating-point ULP (~1e-15),
consistent with ops 0-13 behavior.

### 1.5 Validation Resources

airSpring provides complete test infrastructure for validating the absorbed ops:

| Resource | Location | What It Does |
|----------|----------|-------------|
| `Exp 075` binary | `barracuda/src/bin/validate_local_gpu.rs` | CPU/GPU parity for all 6 ops, edge cases, batch scaling |
| `Exp 077` binary | `barracuda/src/bin/validate_cross_spring_provenance.rs` | CPU vs GPU timing benchmark with provenance tracking |
| Python controls | `control/scs_curve_number/`, `control/yield_response/`, `control/makkink/`, `control/turc/`, `control/hamon/`, `control/blaney_criddle/` | Ground truth from papers |
| Benchmark JSONs | `control/*/benchmark_*.json` | Digitized paper values with provenance |

---

## Part 2: Cross-Spring Evolution Intelligence (Exp 077)

### 2.1 What We Learned from the Provenance Benchmark

Exp 077 (`validate_cross_spring_provenance`, 32/32 PASS) benchmarked CPU vs GPU for
every GPU-wired operation and tracked shader provenance across 5 springs. Key findings:

#### Precision Lineage Validation

The following hotSpring-originated precision functions are exercised transitively
by airSpring's agricultural operations, validating their correctness under real-world
parameter ranges:

| Function | Origin | airSpring Usage |
|----------|--------|----------------|
| `erf_f64` | hotSpring nuclear | VG inverse confidence intervals |
| `gamma_f64` | hotSpring spectral | Pedotransfer incomplete gamma |
| `norm_cdf/ppf` | hotSpring → barracuda::stats | MC ET₀ parametric CI |
| `shannon_entropy` | wetSpring diversity | Soil/tissue diversity indices |
| `simpson_index` | wetSpring diversity | Ecological evenness |
| `ridge_regression` | wetSpring ESN | Sensor calibration |
| `nelder_mead` | neuralSpring optimizer | Isotherm fitting (Langmuir, Freundlich) |

#### GPU Driver Observations

| Driver | GPU | Observation | Impact |
|--------|-----|-------------|--------|
| NVK/Mesa 24.3.4 | Titan V (GV100) | `bootstrap_mean_f64.wgsl` panics on NaN sentinel `bitcast<vec2<u32>>(f64(0.0) / f64(0.0))` | Jackknife/bootstrap/diversity GPU skipped on NVK |
| NVK/Mesa 24.3.4 | Titan V | All other f64 shaders work fine | NaN sentinel is the only issue |
| nvidia 570.86.16 | RTX 4070 | All shaders compile and dispatch correctly | Full GPU stack operational |

**Recommendation**: The NaN sentinel pattern in `bootstrap_mean_f64.wgsl` could use
an alternative NaN encoding that doesn't trigger NVK validation. The `0.0 / 0.0`
division-based NaN works on proprietary NVIDIA drivers but fails Mesa's stricter
validation pipeline.

### 2.2 Cross-Spring Shader Provenance Map

This map documents which spring's innovations flow through which ToadStool primitives
into airSpring's domain operations:

```
hotSpring (precision)
├── df64_core.wgsl ──────────────→ All f64 GPU dispatch (consumer GPUs)
├── math_f64.wgsl ───────────────→ ET₀ (pow, exp, log, trig), VG (pow), Richards
├── df64_transcendentals.wgsl ───→ erf, gamma → MC CI, pedotransfer
└── complex_f64.wgsl ────────────→ Available for spectral soil analysis

wetSpring (bio/environmental)
├── kriging_f64.wgsl ────────────→ Spatial interpolation (100 stations → grid)
├── diversity_fusion_f64.wgsl ───→ Shannon/Simpson/Pielou (soil + tissue)
├── moving_window_f64.wgsl ──────→ IoT sensor stream smoothing
└── ridge regression ────────────→ Sensor calibration (SoilWatch 10)

neuralSpring (ML/optimization)
├── nelder_mead_gpu.wgsl ────────→ Isotherm fitting (Langmuir/Freundlich)
├── ValidationHarness ───────────→ All 86 validation binaries
├── brent_f64.wgsl ──────────────→ VG inverse θ→h, Green-Ampt infiltration
└── L-BFGS ──────────────────────→ Available for smooth soil parameter estimation

groundSpring (uncertainty)
├── jackknife_mean_f64.wgsl ─────→ ET₀ and WB uncertainty quantification
├── bootstrap_mean_f64.wgsl ─────→ Confidence intervals (NVK workaround needed)
├── mc_et0_propagate_f64.wgsl ───→ Monte Carlo ET₀ propagation (xoshiro + Box-Muller)
└── batched_multinomial ─────────→ Resampling for bootstrap

airSpring (precision agriculture) → ToadStool absorption
├── seasonal_pipeline.wgsl ──────→ Fused ET₀→Kc→WB→stress (S72 absorbed)
├── Richards PDE (S40 absorbed) ─→ VG-Mualem + Picard + CN + Thomas
├── BatchedStatefulF64 (S83) ────→ GPU-resident ping-pong state
├── Stats metrics (S64) ─────────→ MAE, RMSE, R², etc.
└── local_elementwise.wgsl ──────→ ** PENDING: ops 14-19 **
```

---

## Part 3: What airSpring Learned (Guides ToadStool Evolution)

### 3.1 Architecture Insights

1. **`ComputeDispatch` is excellent**: All 144 ops using the builder pattern work cleanly
   from airSpring's perspective. The fluent API eliminates boilerplate and reduces errors.

2. **Universal precision works**: airSpring's local f32 shader confirmed that f32 is adequate
   for daily-scale agricultural computations but accumulates error over long chains
   (seasonal pipeline, 153-day simulations). The f64 canonical approach with per-device
   precision selection is the right design.

3. **`BatchedEncoder` opportunity**: airSpring's seasonal pipeline currently dispatches
   ET₀, Kc, and WB as separate GPU operations. `BatchedEncoder` (single `queue.submit()`
   for multi-op) could reduce dispatch overhead by ~20× (matching Exp 072's finding of
   19.7× reduction when fusing 4 stages).

4. **`UnidirectionalPipeline` for AtlasStream**: airSpring's 80-year, 100-station atlas
   pipeline currently uses CPU-chained processing with per-batch GPU dispatch. The
   `UnidirectionalPipeline` pattern (fire-and-forget GPU streaming) would eliminate all
   CPU→GPU round-trips for the atlas workload.

### 3.2 What Works Perfectly

| Primitive | airSpring Experience |
|-----------|---------------------|
| `batched_elementwise_f64` (ops 0-13) | Rock solid. 25 Tier A modules validated. |
| `BrentGpu` | Converges for all practical VG Se fractions. |
| `RichardsGpu` | GPU Picard solver matches CPU within ULP. |
| `FusedMapReduceF64` | TS-004 fully resolved, GPU N≥1024 works. |
| `KrigingF64` | 100 stations → 500 targets in 25µs. |
| `MovingWindowStats` | IoT stream smoothing at 33M elem/s. |
| `ValidationHarness` | Used in all 86 binaries without issues. |

### 3.3 Suggestions for ToadStool Evolution

| Suggestion | Rationale | Priority |
|-----------|-----------|----------|
| NaN sentinel alternative for NVK | `bootstrap_mean_f64.wgsl` panics on Mesa NVK | MEDIUM |
| `daylight_hr_f64` helper in op preamble | Needed for ops 18-19 (Hamon, Blaney-Criddle) | LOW (only for absorption) |
| `SeasonalPipelineF64` adoption guide | airSpring wants to adopt but needs migration example | LOW |
| Document `BatchedEncoder` usage pattern | airSpring could fuse seasonal pipeline stages | LOW |

---

## Part 4: Performance Data

### 4.1 CPU vs GPU Throughput (Exp 077)

| Operation | CPU Throughput | GPU Available | Cross-Spring Origin |
|-----------|---------------|--------------|-------------------|
| FAO-56 ET₀ | 10M/s | `batched_elementwise_f64` op=0 | airSpring → hotSpring precision |
| Water Balance | 162M days/s | `batched_elementwise_f64` op=1 | airSpring → multi-spring |
| Hargreaves ET₀ | — | `HargreavesBatchGpu` dedicated | airSpring → hotSpring precision |
| Kc Climate | — | `batched_elementwise_f64` op=7 | FAO-56 Eq. 62 |
| VG θ(h) | 37.7M/s | `batched_elementwise_f64` op=9 | hotSpring precision + wetSpring bio |
| GDD | — | `batched_elementwise_f64` op=12 | airSpring phenology |
| Pedotransfer | — | `batched_elementwise_f64` op=13 | Saxton-Rawls 2006 |
| Seasonal Pipeline | 6.8M field-days/s | Fused multi-stage | All springs |

### 4.2 Validation Counts

| Category | Count | Status |
|----------|:-----:|--------|
| Python baselines | 1,237 | ALL PASS |
| Rust lib tests | 846 | ALL PASS |
| Forge tests | 61 | ALL PASS |
| Validation binaries | 86 | ALL PASS |
| Cross-spring evolution bench | 146/146 | PASS (S87) |
| Exp 077 provenance bench | 32/32 | PASS |
| Cross-spring rewire | 68/68 | PASS (5 springs) |
| GPU math portability | 46/46 | PASS (13 modules) |
| Clippy pedantic | 0 warnings | Clean |
| Cargo-deny | Clean | AGPL-3.0-or-later |

---

## Part 5: Recommended Absorption Sequence

1. **Ops 14-15 first** (SCS-CN, Stewart): Pure arithmetic, no helpers needed, trivial
2. **Op 17 next** (Turc): Pure arithmetic with humidity branch
3. **Ops 16, 18-19 last** (Makkink, Hamon, BC): Need FAO-56 preamble helpers + daylight

After absorption, airSpring updates `local_elementwise.wgsl` references to
`batched_elementwise_f64` ops 14-19, removes the local dispatch engine for these ops,
and leans on upstream. The "Write → Absorb → Lean" cycle completes.

---

## Appendix: airSpring's Complete GPU Module Inventory

| Module | ToadStool Op/Shader | Origin | Tier |
|--------|-------------------|--------|------|
| `gpu::et0` | `batched_elementwise_f64` op=0 | airSpring V039 | A |
| `gpu::water_balance` | `batched_elementwise_f64` op=1 | airSpring V039 | A |
| `gpu::sensor_calibration` | `batched_elementwise_f64` op=5 | Dong 2024, S70 | A |
| `gpu::hargreaves` | `HargreavesBatchGpu` | FAO-56, S70 | A |
| `gpu::kc_climate` | `batched_elementwise_f64` op=7 | FAO-56, S70 | A |
| `gpu::dual_kc` | `batched_elementwise_f64` op=8 | airSpring V039, S70 | A |
| `gpu::van_genuchten` | `batched_elementwise_f64` ops 9-10 | hotSpring + wetSpring, S79 | A |
| `gpu::thornthwaite` | `batched_elementwise_f64` op=11 | airSpring climate, S79 | A |
| `gpu::gdd` | `batched_elementwise_f64` op=12 | airSpring phenology, S79 | A |
| `gpu::pedotransfer` | `batched_elementwise_f64` op=13 | Saxton-Rawls, S79 | A |
| `gpu::jackknife` | `JackknifeMeanGpu` | groundSpring → S71 | A |
| `gpu::bootstrap` | `BootstrapMeanGpu` | groundSpring → S71 | A |
| `gpu::diversity` | `DiversityFusionGpu` | wetSpring → S70 | A |
| `gpu::kriging` | `KrigingF64` | wetSpring spatial | A |
| `gpu::reduce` | `FusedMapReduceF64` | wetSpring + airSpring TS-004 | A |
| `gpu::stream` | `MovingWindowStats` | wetSpring S28+ | A |
| `gpu::richards` | `pde::richards` + `RichardsGpu` | airSpring S40 + S83 | A |
| `gpu::isotherm` | `nelder_mead` + `multi_start` | neuralSpring optimizer | A |
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` | groundSpring xoshiro | A |
| `gpu::infiltration` | `BrentGpu` GA residual | neuralSpring → S83 | A |
| `gpu::stats` | `linear_regression_f64` + `matrix_correlation_f64` | neuralSpring S69 | A |
| `gpu::runoff` | `local_elementwise.wgsl` op=0 | **LOCAL (pending op=14)** | A-local |
| `gpu::yield_response` | `local_elementwise.wgsl` op=1 | **LOCAL (pending op=15)** | A-local |
| `gpu::simple_et0` | `local_elementwise.wgsl` ops 2-5 | **LOCAL (pending ops 16-19)** | A-local |
| `gpu::local_dispatch` | `LocalElementwise` (wgpu direct) | airSpring v0.6.8 | Engine |
| `gpu::seasonal_pipeline` | Chains ops 0→7→1→yield | All springs | Pipeline |
| `gpu::atlas_stream` | `UnidirectionalPipeline` target | Multi-year regional | Pipeline |

---

*V053 — airSpring v0.6.8, 77 experiments, ToadStool S87 synced.*
*Previous handoffs: V052 (S87 sync), V051 (local GPU + ops 14-19), V050 (evolution).*
