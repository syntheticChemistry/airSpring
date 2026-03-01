# airSpring → ToadStool / BarraCuda Evolution Handoff V039

**Date**: March 1, 2026
**From**: airSpring (biomeGate)
**To**: ToadStool / BarraCuda evolution team
**Covers**: airSpring v0.5.6 — complete S70+ rewire, GPU benchmark, cross-spring provenance
**Supersedes**: V038 (archived)
**ToadStool Pin**: `1dd7e338` (S70+++)
**License**: AGPL-3.0-or-later

---

## Executive Summary

1. **All 6 batched elementwise ops** (0, 1, 5, 6, 7, 8) rewired to GPU-first dispatch
2. **GPU stats module** wired (`linear_regression_f64`, `matrix_correlation_f64` from neuralSpring S69)
3. **Exp 057**: 26/26 GPU rewire validation + benchmark checks pass
4. **20/20 CPU parity** vs Python (17.9× speedup — now includes SensorCal + Kc adj)
5. **35/35 cross-spring benchmarks** (was 30/30), **640 lib tests** (was 636), zero clippy warnings
6. Full cross-spring provenance documented: hotSpring, wetSpring, neuralSpring, airSpring, groundSpring

---

## Part 1: How airSpring Uses BarraCuda

### Upstream Dependencies (consumed from ToadStool)

| Module | API | airSpring Use | Wired Since |
|--------|-----|---------------|:-----------:|
| `ops::batched_elementwise_f64` | `BatchedElementwiseF64::fao56_et0_batch()` | ET₀ computation (N stations) | S54 |
| `ops::batched_elementwise_f64` | `execute(&data, n, Op::WaterBalance)` | Daily depletion update | S54 |
| `ops::batched_elementwise_f64` | `execute(&data, n, Op::SensorCalibration)` | SoilWatch 10 VWC (Dong Eq.5) | S70+ |
| `ops::batched_elementwise_f64` | `execute(&data, n, Op::HargreavesEt0)` | Temperature-only ET₀ (FAO-56 Eq.52) | S70+ |
| `ops::batched_elementwise_f64` | `execute(&data, n, Op::KcClimateAdjust)` | Kc climate adjustment (FAO-56 Eq.62) | S70+ |
| `ops::batched_elementwise_f64` | `execute(&data, n, Op::DualKcKe)` | Soil evaporation Ke (FAO-56 Ch 7/11) | S70+ |
| `ops::stats_f64` | `linear_regression(device, x, y, b, n, k)` | Sensor calibration OLS regression | S69 |
| `ops::stats_f64` | `matrix_correlation(device, data, n, p)` | Multi-variate soil analysis | S69 |
| `shaders::science` | `batched_elementwise_f64.wgsl` | All 6 element-wise ops | S54→S70+ |
| `shaders::stats` | `linear_regression_f64.wgsl` | Batched OLS on GPU | S69 |
| `shaders::stats` | `matrix_correlation_f64.wgsl` | Pearson correlation on GPU | S69 |
| `device::WgpuDevice` | `new_f64_capable()` | GPU device acquisition | S42 |
| `device::probe` | `F64BuiltinCapabilities` | Precision probing | S67 |
| `stats::hydrology` | `hargreaves_et0`, `crop_coefficient`, `soil_water_balance` | CPU reference implementations | S66 |

### Local Implementations (airSpring-specific, for ToadStool to absorb)

| Module | Primitives | Domain | Absorption Priority |
|--------|------------|--------|:-------------------:|
| `gpu::seasonal_pipeline` | ET₀→Kc→WB→Yield chained | 153-day growing season | **High** (fused shader exists) |
| `gpu::atlas_stream` | Multi-station batch unified | Regional crop water atlas | Medium |
| `gpu::mc_et0` | MC uncertainty bands | Parametric CI for ET₀ | Medium |
| `eco::anderson` | θ→S_e→p_c→z→d_eff→QS regime | Soil-moisture coupling | Low (different from physics Anderson) |

---

## Part 2: What to Absorb

### 2a. Immediate: Fused Seasonal Pipeline Executor (High Priority)

`seasonal_pipeline.wgsl` exists in ToadStool (S70+) but has no Rust executor. airSpring
currently chains ops 0→7→1→yield across 4 GPU submissions. A single-dispatch executor
would eliminate 3 CPU round-trips per day.

**Shader**: `shaders/science/seasonal_pipeline.wgsl`
**Uniform**: `SeasonalParams { cell_count, day_of_year, stage_length, day_in_stage, kc_prev, kc_next, taw_default, raw_fraction, field_capacity }`
**Bindings**: `cell_weather[9*N]`, `cell_output[5*N]`, `params` (uniform)
**ToadStool action**: Write `SeasonalPipelineF64` executor with `execute(device, weather, params) -> Vec<SeasonalOutput>`

### 2b. Immediate: Fix `brent_f64.wgsl` Line 49 (High Priority)

airSpring needs GPU VG inverse for Richards PDE acceleration. The `brent_f64.wgsl`
shader (op=0: VG inverse, op=1: Green-Ampt) has a bug on line 49 that prevents correct
convergence. Once fixed, airSpring can write a `BatchedBrentF64` executor.

### 2c. Medium: MC ET₀ Propagate Shader

airSpring's `gpu::mc_et0` module has a CPU Monte Carlo path (Box-Muller + FAO-56) but
no GPU shader. A fused `mc_et0_propagate_f64.wgsl` would accelerate uncertainty
quantification (N=5000+ samples per station-day).

### 2d. Medium: `UnidirectionalPipeline` for Atlas Stream

airSpring's `gpu::atlas_stream` currently batches ET₀ for all stations in one GPU submission.
The full pipeline (ET₀ → Kc → WB → Yield) could use `staging::UnidirectionalPipeline` for
streaming without CPU readback between stages.

### 2e. Low: Batch Nelder-Mead for Isotherm Fitting

airSpring uses `nelder_mead.wgsl` for serial isotherm fitting. A batched multi-start
NM shader would accelerate global optimization for large isotherm datasets.

---

## Part 3: Evolution Discoveries

### 3a. Cross-Spring Shader Provenance

The `batched_elementwise_f64.wgsl` shader is a convergence point for the entire
ecoPrimals ecosystem:

```
hotSpring   → math_f64.wgsl: exp_f64, log_f64, sqrt, sin_f64, cos_f64, acos_f64, pow_f64
            → df64_core: double-float f32 pairs (~48-bit) for consumer GPUs
            → Universal precision (S67-68): f64 canonical → Df64/f32/f16 compile targets
wetSpring   → diversity batch patterns: Shannon/Simpson → fused_map_reduce
            → moving_window_f64: IoT stream smoothing
            → kriging_f64: soil moisture spatial interpolation
neuralSpring → BatchedElementwiseF64: orchestrator pattern (batch, dispatch, readback)
             → ValidationHarness (S58): structured pass/fail with tracing
             → stats_f64 (S69): linear regression + correlation matrix on GPU
airSpring   → ops 5-8 domain logic: SensorCal, Hargreaves, KcClimate, DualKcKe
            → hydrology stats → barracuda::stats::hydrology (CPU reference)
            → seasonal pipeline concept → seasonal_pipeline.wgsl (WGSL implementation)
groundSpring → MC ET₀: xoshiro256++ + Box-Muller GPU RNG
             → rawr_mean: robust averaging
```

### 3b. NVK `acos_f64` Polyfill Precision

NVK polyfill `acos` accumulates error through Hargreaves:
`ws = acos(-tan(lat) · tan(δ))` → `Ra = f(ws, lat, δ)` → `ET₀ = 0.0023·(T+17.8)·√(Ra)·ΔT`

Per-point error: up to ~0.07 mm/day. Seasonal aggregate: < 0.36%. Native f64 drivers: < 0.001.

### 3c. GPU Throughput Crossover

At batch sizes N≤2000, GPU dispatch overhead (~1.5ms) dominates — CPU wins. At N>5000
GPU wins decisively. At N=50K: 11.5M items/s. This means GPU is optimal for:
- Regional atlas (100+ stations × 365 days)
- Monte Carlo (5000+ samples)
- Seasonal batch (153 days × many fields)

### 3d. `stats_f64` WGSL Compilation on NVK

`linear_regression_f64.wgsl` and `matrix_correlation_f64.wgsl` fail on NVK because
`enable f64` is not supported in the shader parser. Works on native f64 drivers. airSpring
tests catch the panic and skip gracefully.

---

## Part 4: Paper Validation Controls

All airSpring experiments use open data and published literature:

| Data Source | Papers | Access |
|-------------|:------:|--------|
| FAO-56 (Allen et al. 1998) | 10+ | Open literature |
| Open-Meteo ERA5 | 5 | Free API, 80-year archive |
| USDA NASS | 2 | Public portal |
| USDA SCAN | 1 | Public portal |
| AmeriFlux | 1 | Public portal |
| NCBI SRA (16S) | 1 | Public GenBank |
| Published tables | 40+ | Open literature |

**Control substrate matrix**: 20/20 Python↔Rust parity benchmarks at 17.9× geometric speedup.

---

## Part 5: Current State

| Metric | V038 | V039 |
|--------|------|------|
| Experiments | 57 | 57 |
| Python checks | 1237 | 1237 |
| Barracuda lib tests | 640 | 640 |
| metalForge lib tests | 57 | 57 |
| Validation binaries | 62 | 62 |
| CPU vs Python parity | 18/18 (22.7×) | **20/20 (17.9×)** |
| GPU pipeline (Exp 055) | 78/78 | 78/78 |
| GPU rewire (Exp 057) | 26/26 | 26/26 |
| Cross-spring benchmarks | 35/35 | 35/35 |
| Mixed-hardware (Exp 056) | 104/104 | 104/104 |
| Tier A GPU orchestrators | 17 | 17 |
| Tier B GPU gaps | 7 | 7 |
| Seasonal pipeline | GPU Stages 1-2 | GPU Stages 1-2 |
| Clippy pedantic | 0 warnings | 0 warnings |
| ToadStool HEAD | `1dd7e338` | `1dd7e338` |

---

## Part 6: Remaining Evolution Gaps

| Gap | Priority | Blocker | ToadStool Action |
|-----|:--------:|---------|------------------|
| Fused seasonal pipeline | **P0** | No Rust executor for `seasonal_pipeline.wgsl` | Write `SeasonalPipelineF64` |
| Brent root-finding | **P0** | Bug on L49 of `brent_f64.wgsl` | Fix and add executor |
| MC ET₀ GPU | **P1** | No `mc_et0_propagate_f64.wgsl` | Write fused shader |
| Atlas `UnidirectionalPipeline` | **P1** | Not wired | Provide example pattern |
| Water balance stateful | **P2** | Sequential day-over-day state | Wire via `StatefulPipeline` |
| Batch Nelder-Mead | **P2** | Serial NM only | Multi-start parallel shader |
| Pedotransfer GPU | **P3** | No shader | Polynomial evaluation shader |
| Cover crop diversity GPU | **P3** | No batch shader | Extend `fused_map_reduce_f64` |

---

## Closing

This is a unidirectional handoff — no response expected. Everything needed to evolve
is included: what we consume, what we built, what we learned, and what we need next.
The full cross-spring provenance chain (paper → Python → Barracuda CPU → Barracuda GPU →
metalForge cross-system) is validated end-to-end.

airSpring will continue building locally and will wire new ToadStool primitives as they
become available.
