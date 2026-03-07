# GPU Promotion Map — airSpring BarraCuda

SPDX-License-Identifier: AGPL-3.0-or-later

**Last updated**: 2026-03-07 (v0.7.3, barraCuda 0.3.3+ / wgpu 28, PrecisionRoutingAdvice wired, upstream provenance registry integrated)
**Sources**: `EVOLUTION_READINESS.md`, `gpu/evolution_gaps.rs`, `BARRACUDA_REQUIREMENTS.md`

---

## Tier Definitions

| Tier | Definition |
|------|------------|
| **A** | GPU-integrated — `BatchedElementwiseF64` or dedicated shader, validated, GPU-first |
| **B** | Orchestrator with CPU fallback — upstream primitive exists but ops not yet absorbed |
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
| `gpu::runoff` | **A** | `batched_elementwise_f64` (op=17) | SCS-CN runoff | — (absorbed v0.7.2) |
| `gpu::yield_response` | **A** | `batched_elementwise_f64` (op=18) | Stewart yield | — (absorbed v0.7.2) |
| `gpu::simple_et0` | **A** | `batched_elementwise_f64` (ops 14-16, 19) | Makkink/Turc/Hamon/BC | — (absorbed v0.7.2) |
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
| `eco::evapotranspiration` (Makkink/Turc/Hamon/BC) | **A** | `gpu::simple_et0` | — (absorbed v0.7.2) |
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
| `eco::runoff` | **A** | `gpu::runoff` | — (absorbed v0.7.2) |
| `eco::yield_response` | **A** | `gpu::yield_response` | — (absorbed v0.7.2) |
| `eco::anderson` | **C** | — | Needs new WGSL shader |
| `eco::et0_ensemble` | **C** | — | CPU ensemble logic |
| `eco::solar` | — | Used by ET₀ | Supporting module |

---

## ToadStool WGSL Shaders (airSpring-relevant)

| Shader | Location | Used By | Status |
|--------|----------|---------|--------|
| `batched_elementwise_f64.wgsl` | `shaders/science/` | et0, water_balance, sensor_cal, HG, kc, dual_kc, VG, thornthwaite, gdd, pedotransfer, makkink, turc, hamon, scs-cn, stewart, blaney-criddle | Ops 0-19 wired |
| `kriging_f64.wgsl` | `shaders/interpolation/` | kriging | Wired |
| `fused_map_reduce_f64.wgsl` | `shaders/reduce/` | reduce | Wired |
| `moving_window_f64.wgsl` | `shaders/stats/` | stream | Wired |
| `pde_richards.wgsl` | `shaders/pde/` | richards | Wired |
| `brent_f64.wgsl` | `shaders/optimize/` | infiltration (GA residual) | Wired (S83) |
| `nelder_mead.wgsl` | `shaders/optimize/` | isotherm | Wired |
| `mc_et0_propagate_f64.wgsl` | `shaders/bio/` | mc_et0 | Wired |
| `linear_regression_f64.wgsl` | `shaders/stats/` | stats | Wired |
| `matrix_correlation_f64.wgsl` | `shaders/stats/` | stats | Wired |
| ~~`local_elementwise.wgsl`~~ | ~~`barracuda/src/shaders/`~~ | ~~runoff, yield, simple_et0~~ | **RETIRED** — absorbed into `batched_elementwise_f64.wgsl` ops 14-19 (v0.7.2) |

---

## Blocker Summary

| Blocker | Modules | Resolution |
|---------|---------|------------|
| ~~Local ops f64 absorption~~ | ~~runoff, yield_response, simple_et0~~ | **RESOLVED** (v0.7.2) — all 6 ops absorbed into `BatchedElementwiseF64` ops 14-19 |
| Fused GPU seasonal pipeline | seasonal_pipeline | Fuse ops 0→7→1→yield in single dispatch |
| `UnidirectionalPipeline` | atlas_stream | Implement fire-and-forget GPU streaming |
| Anderson shader | anderson | New WGSL for θ→S_e→d_eff→QS coupling |

---

## Tier Counts

| Tier | Count | Evolution Path |
|------|-------|----------------|
| **A** | 24 | GPU-first via `BatchedElementwiseF64` or dedicated shader |
| **B** | 2 | Pipeline/streaming GPU evolution |
| **C** | 2 | Needs new shaders or architectural decision |

---

## Sovereign Compute Evolution Roadmap

The ecoPrimals ecosystem is evolving toward a fully sovereign Rust GPU stack,
eliminating C dependencies from the shader compilation path:

| Phase | Component | Status | What It Does |
|-------|-----------|--------|--------------|
| 1 | **barraCuda DF64** | Complete | Double-float f32-pair workaround for f64 on consumer GPUs |
| 2 | **coralReef** | Phase 10 (sovereign compiler) | Sovereign Rust GPU compiler — WGSL/SPIR-V → native GPU binary |
| 3 | coralReef multi-vendor | In Progress | NVIDIA (SM70+), AMD (RDNA), Intel (Xe) backends |
| 4 | coralReef f64 lowering | In Progress | DFMA-based transcendentals for native f64 |
| 5 | **coralDriver** | Planned | Userspace GPU driver (pure Rust) |
| 6 | **coralGpu** | Planned | Unified Rust GPU abstraction |

**coralReef** (`ecoPrimals/coralReef/`): Sovereign Rust GPU compiler (replaces coralNAK).
Turns WGSL/SPIR-V input into native GPU binaries with multi-vendor backend support.
Integrates with barraCuda via `shader.compile.*` JSON-RPC methods through ToadStool proxy.
Once complete, replaces naga + wgpu's shader path with sovereign Rust compilation that
correctly emits f64 transcendentals (fixing the root cause that DF64 works around).

Pipeline flow: `WGSL → naga → SPIR-V → coralReef (optimize, legalize, encode) → Native GPU`
