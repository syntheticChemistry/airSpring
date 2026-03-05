# GPU Promotion Map — airSpring BarraCuda

SPDX-License-Identifier: AGPL-3.0-or-later

**Last updated**: 2026-03-05 (v0.7.0, barraCuda 0.3.3 / wgpu 28, fused Welford + Pearson, f64 canonical universal precision)
**Sources**: `EVOLUTION_READINESS.md`, `gpu/evolution_gaps.rs`, `BARRACUDA_REQUIREMENTS.md`

---

## Tier Definitions

| Tier | Definition |
|------|------------|
| **A** | GPU-integrated — ToadStool f64 primitive wired, validated, GPU-first (or GPU N≥1024) |
| **A-local** | GPU-local — airSpring `local_elementwise.wgsl` (f32), ToadStool f64 absorption pending |
| **B** | Orchestrator with CPU fallback — ToadStool primitive exists but ops not yet absorbed |
| **C** | CPU-only — needs new WGSL shader or is inherently serial/non-GPU |

---

## gpu/ Modules

| Module | Tier | WGSL Shader | Pipeline Stage | Blocker |
|--------|------|-------------|----------------|---------|
| `gpu::et0` | **A** | `batched_elementwise_f64` (op=0) | FAO-56 PM ET₀ | — |
| `gpu::water_balance` | **A** | `batched_elementwise_f64` (op=1) | Daily water balance | — |
| `gpu::sensor_calibration` | **A** | `batched_elementwise_f64` (op=5) | SoilWatch VWC | — (S70+) |
| `gpu::hargreaves` | **A** | `HargreavesBatchGpu` | Hargreaves ET₀ | — (S70+) |
| `gpu::kc_climate` | **A** | `batched_elementwise_f64` (op=7) | Kc climate adjustment | — (S70+) |
| `gpu::dual_kc` | **A** | `batched_elementwise_f64` (op=8) | Dual Kc (Ke batch) | — (S70+) |
| `gpu::van_genuchten` | **A** | `batched_elementwise_f64` (op=9,10) | VG θ/K batch | — (S79) |
| `gpu::thornthwaite` | **A** | `batched_elementwise_f64` (op=11) | Monthly ET₀ | — (S79) |
| `gpu::gdd` | **A** | `batched_elementwise_f64` (op=12) | GDD batch | — (S79) |
| `gpu::pedotransfer` | **A** | `batched_elementwise_f64` (op=13) | Saxton-Rawls | — (S79) |
| `gpu::kriging` | **A** | `kriging_f64` | Spatial interpolation | — |
| `gpu::reduce` | **A** | `fused_map_reduce_f64` | Seasonal stats (N≥1024) | — |
| `gpu::stream` | **A** | `moving_window_f64` | IoT stream smoothing | — |
| `gpu::richards` | **A** | `pde_richards.wgsl` | Richards PDE | — |
| `gpu::infiltration` | **A** | `brent_f64.wgsl` | Green-Ampt infiltration | — (S83) |
| `gpu::isotherm` | **A** | `nelder_mead.wgsl` | Isotherm fitting | — |
| `gpu::mc_et0` | **A** | `mc_et0_propagate_f64.wgsl` | MC uncertainty | — |
| `gpu::jackknife` | **A** | `JackknifeMeanGpu` | Jackknife CI | — (S79) |
| `gpu::bootstrap` | **A** | `BootstrapMeanGpu` | Bootstrap CI | — (S79) |
| `gpu::diversity` | **A** | `DiversityFusionGpu` | Diversity indices | — (S79) |
| `gpu::stats` | **A** | `linear_regression_f64` + `matrix_correlation_f64` | Sensor regression | — |
| `gpu::runoff` | **A-local** | `local_elementwise.wgsl` (op=0) | SCS-CN runoff | ToadStool f64 absorption |
| `gpu::yield_response` | **A-local** | `local_elementwise.wgsl` (op=1) | Stewart yield | ToadStool f64 absorption |
| `gpu::simple_et0` | **A-local** | `local_elementwise.wgsl` (ops 2-5) | Makkink/Turc/Hamon/BC | ToadStool f64 absorption |
| `gpu::local_dispatch` | — | `local_elementwise.wgsl` | wgpu dispatch engine | Infrastructure module |
| `gpu::seasonal_pipeline` | **B** | Chained ops 0→7→1→yield | End-to-end season | Fused GPU pipeline |
| `gpu::atlas_stream` | **B** | `UnidirectionalPipeline` (pending) | Regional atlas | GPU streaming primitive |
| `gpu::device_info` | — | N/A | Precision probing | Documentation module |
| `gpu::evolution_gaps` | — | N/A | Living roadmap | Documentation module |

---

## eco/ Modules → GPU Path

| Module | GPU Tier | GPU Wrapper | Blocker |
|--------|----------|-------------|---------|
| `eco::evapotranspiration` (PM) | **A** | `gpu::et0` | — |
| `eco::evapotranspiration` (HG) | **A** | `gpu::hargreaves` | — |
| `eco::evapotranspiration` (Makkink/Turc/Hamon/BC) | **A-local** | `gpu::simple_et0` | ToadStool f64 absorption |
| `eco::water_balance` | **A** | `gpu::water_balance` | — |
| `eco::richards` | **A** | `gpu::richards` | — |
| `eco::van_genuchten` | **A** | `gpu::van_genuchten` | — |
| `eco::infiltration` | **A** | `gpu::infiltration` | — |
| `eco::isotherm` | **A** | `gpu::isotherm` | — |
| `eco::soil_moisture` | **A** | `gpu::kriging` | — |
| `eco::correction` | **A** | CPU `linalg::ridge` | — (small N, no GPU benefit) |
| `eco::diversity` | **A** | `gpu::diversity` | — |
| `eco::sensor_calibration` | **A** | `gpu::sensor_calibration` | — |
| `eco::dual_kc` | **A** | `gpu::dual_kc` | — |
| `eco::crop` (Kc adj) | **A** | `gpu::kc_climate` | — |
| `eco::thornthwaite` | **A** | `gpu::thornthwaite` | — |
| `eco::crop` (GDD) | **A** | `gpu::gdd` | — |
| `eco::runoff` | **A-local** | `gpu::runoff` | ToadStool f64 absorption |
| `eco::yield_response` | **A-local** | `gpu::yield_response` | ToadStool f64 absorption |
| `eco::anderson` | **C** | — | Needs new WGSL shader |
| `eco::et0_ensemble` | **C** | — | CPU ensemble logic |
| `eco::solar` | — | Used by ET₀ | Supporting module |

---

## ToadStool WGSL Shaders (airSpring-relevant)

| Shader | Location | Used By | Status |
|--------|----------|---------|--------|
| `batched_elementwise_f64.wgsl` | `shaders/science/` | et0, water_balance, sensor_cal, HG, kc, dual_kc, VG, thornthwaite, gdd, pedotransfer | Ops 0-13 wired |
| `kriging_f64.wgsl` | `shaders/interpolation/` | kriging | Wired |
| `fused_map_reduce_f64.wgsl` | `shaders/reduce/` | reduce | Wired |
| `moving_window_f64.wgsl` | `shaders/stats/` | stream | Wired |
| `pde_richards.wgsl` | `shaders/pde/` | richards | Wired |
| `brent_f64.wgsl` | `shaders/optimize/` | infiltration (GA residual) | Wired (S83) |
| `nelder_mead.wgsl` | `shaders/optimize/` | isotherm | Wired |
| `mc_et0_propagate_f64.wgsl` | `shaders/bio/` | mc_et0 | Wired |
| `linear_regression_f64.wgsl` | `shaders/stats/` | stats | Wired |
| `matrix_correlation_f64.wgsl` | `shaders/stats/` | stats | Wired |
| `local_elementwise.wgsl` | `barracuda/src/shaders/` | runoff, yield, simple_et0 | **Local** (airSpring f32, ToadStool f64 pending) |

---

## Blocker Summary

| Blocker | Modules | Resolution |
|---------|---------|------------|
| ToadStool f64 absorption of local ops | runoff, yield_response, simple_et0 (6 ops) | Absorb `local_elementwise.wgsl` into canonical f64 `batched_elementwise_f64.wgsl` ops 14-19 |
| Fused GPU seasonal pipeline | seasonal_pipeline | Fuse ops 0→7→1→yield in single dispatch |
| `UnidirectionalPipeline` | atlas_stream | Implement fire-and-forget GPU streaming |
| Anderson shader | anderson | New WGSL for θ→S_e→d_eff→QS coupling |

---

## Tier Counts

| Tier | Count | Evolution Path |
|------|-------|----------------|
| **A** | 21 | GPU-first via ToadStool f64 |
| **A-local** | 6 | GPU-local f32 WGSL → ToadStool f64 absorption |
| **B** | 2 | Pipeline/streaming GPU evolution |
| **C** | 2 | Needs new shaders or architectural decision |
