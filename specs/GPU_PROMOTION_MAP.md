# GPU Promotion Map — airSpring BarraCuda

SPDX-License-Identifier: AGPL-3.0-or-later

**Last updated**: 2026-02-27 (v0.5.2, ToadStool S68+ `e96576ee`)
**Sources**: `EVOLUTION_READINESS.md`, `gpu/evolution_gaps.rs`, `BARRACUDA_REQUIREMENTS.md`

---

## Tier Definitions

| Tier | Definition |
|------|------------|
| **A** | GPU-integrated — primitive wired, validated, GPU-first (or GPU N≥1024) |
| **B** | Orchestrator with CPU fallback — ToadStool primitive exists but ops not yet absorbed |
| **C** | CPU-only — needs new WGSL shader or is inherently serial/non-GPU |

---

## gpu/ Modules

| Module | Tier | WGSL Shader | Pipeline Stage | Blocker |
|--------|------|-------------|----------------|---------|
| `gpu::et0` | **A** | `batched_elementwise_f64` (op=0) | FAO-56 PM ET₀ | — |
| `gpu::water_balance` | **A** | `batched_elementwise_f64` (op=1) | Daily water balance | — |
| `gpu::kriging` | **A** | `kriging_f64` | Spatial interpolation | — |
| `gpu::reduce` | **A** | `fused_map_reduce_f64` | Seasonal stats (N≥1024) | — |
| `gpu::stream` | **A** | `moving_window_f64` | IoT stream smoothing | — |
| `gpu::richards` | **A** | `van_genuchten_f64` (θ/K eval) | Richards PDE | — |
| `gpu::isotherm` | **A** | CPU `nelder_mead` | Isotherm fitting | `NelderMeadGpu` exists upstream but not batch-wired |
| `gpu::sensor_calibration` | **B** | `batched_elementwise_f64` (op=5) | SoilWatch VWC | ToadStool op=5 absorption |
| `gpu::hargreaves` | **B** | `batched_elementwise_f64` (op=6) | Hargreaves ET₀ | ToadStool op=6 absorption |
| `gpu::kc_climate` | **B** | `batched_elementwise_f64` (op=7) | Kc climate adjustment | ToadStool op=7 absorption |
| `gpu::dual_kc` | **B** | `batched_elementwise_f64` (op=8) | Dual Kc (Ke batch) | ToadStool op=8 absorption |
| `gpu::mc_et0` | **B** | `mc_et0_propagate_f64` (exists) | MC uncertainty | GPU dispatch not wired |
| `gpu::seasonal_pipeline` | **B** | Chained ops 0→7→1→yield | End-to-end season | CPU-chained; needs ops 5–8 absorbed |
| `gpu::atlas_stream` | **B** | `UnidirectionalPipeline` (pending) | Regional atlas | Needs GPU streaming primitive |
| `gpu::device_info` | — | N/A | Precision probing | Documentation module |
| `gpu::evolution_gaps` | — | N/A | Living roadmap | Documentation module |

---

## eco/ Modules → GPU Path

| Module | GPU Tier | GPU Wrapper | Blocker |
|--------|----------|-------------|---------|
| `eco::evapotranspiration` (PM) | **A** | `gpu::et0` | — |
| `eco::evapotranspiration` (HG) | **B** | `gpu::hargreaves` | ToadStool op=6 |
| `eco::water_balance` | **A** | `gpu::water_balance` | — |
| `eco::richards` | **A** | `gpu::richards` | — |
| `eco::van_genuchten` | **A** | via `gpu::richards` | — |
| `eco::isotherm` | **A** | `gpu::isotherm` | — |
| `eco::soil_moisture` | **A** | `gpu::kriging` | — |
| `eco::correction` | **A** | CPU `linalg::ridge` | — (small N, no GPU benefit) |
| `eco::diversity` | **A** | CPU `stats::diversity` | `DiversityFusionGpu` upstream, not wired |
| `eco::sensor_calibration` | **B** | `gpu::sensor_calibration` | ToadStool op=5 |
| `eco::dual_kc` | **B** | `gpu::dual_kc` | ToadStool op=8 |
| `eco::crop` (Kc adj) | **B** | `gpu::kc_climate` | ToadStool op=7 |
| `eco::anderson` | **C** | — | Needs new WGSL shader |
| `eco::thornthwaite` | **C** | — | Monthly; low parallelism |
| `eco::yield_response` | **C** | — | No batch primitive |
| `eco::et0_ensemble` | **C** | — | CPU ensemble logic |
| `eco::crop` (GDD) | **C** | — | Domain data; serial scan |
| `eco::solar` | — | Used by ET₀ | Supporting module |

---

## ToadStool WGSL Shaders (airSpring-relevant)

| Shader | Location | Used By | Status |
|--------|----------|---------|--------|
| `batched_elementwise_f64.wgsl` | `shaders/science/` | et0, water_balance | Ops 0,1 wired; 5–8 pending |
| `kriging_f64.wgsl` | `shaders/interpolation/` | kriging | Wired |
| `fused_map_reduce_f64.wgsl` | `shaders/reduce/` | reduce | Wired |
| `moving_window_f64.wgsl` | `shaders/stats/` | stream | Wired |
| `van_genuchten_f64.wgsl` | `shaders/science/` | richards | Wired (θ/K eval) |
| `crank_nicolson_f64.wgsl` | `shaders/pde/` | richards (CN cross-val) | Available |
| `mc_et0_propagate_f64.wgsl` | `shaders/bio/` | mc_et0 | Exists; dispatch not wired |
| `bray_curtis_f64.wgsl` | `shaders/math/` | diversity (potential) | Available; not wired |
| `diversity_fusion_f64.wgsl` | `shaders/bio/` | diversity (potential) | Available; not wired |

---

## Blocker Summary

| Blocker | Modules | Resolution |
|---------|---------|------------|
| ToadStool ops 5–8 | sensor_cal, hargreaves, kc_climate, dual_kc | Add cases to `batched_elementwise_f64.wgsl` |
| MC ET₀ GPU dispatch | mc_et0 | Wire existing `mc_et0_propagate_f64.wgsl` |
| `UnidirectionalPipeline` | atlas_stream | Implement fire-and-forget GPU streaming |
| Anderson shader | anderson | New WGSL for θ→S_e→d_eff→QS coupling |
| `NelderMeadGpu` batch | isotherm | Wire upstream `NelderMeadGpu` for batch fitting |

---

## Tier Counts

| Tier | Count | Evolution Path |
|------|-------|----------------|
| **A** | 11 | Ready for sovereign pipeline |
| **B** | 9 | Blocked on ToadStool absorption → automatic A when resolved |
| **C** | 5+ | Needs new shaders or architectural decision |
