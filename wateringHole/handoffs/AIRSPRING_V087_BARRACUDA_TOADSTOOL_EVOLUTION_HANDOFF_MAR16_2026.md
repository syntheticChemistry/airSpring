# airSpring V0.8.7 — barraCuda / toadStool Evolution Handoff

**Date**: March 16, 2026
**From**: airSpring V0.8.7
**To**: barraCuda, toadStool, coralReef, sibling springs
**License**: AGPL-3.0-or-later

---

## Executive Summary

- airSpring consumes **75+ distinct barraCuda functions/types** across 11 modules
- **20 GPU ops** via `BatchedElementwiseF64` (ops 0–19), all upstream
- **11 specialized GPU primitives** (BrentGpu, RichardsGpu, KrigingF64, etc.)
- **Zero local WGSL shaders** — Write→Absorb→Lean complete
- **3 toadStool IPC methods** via typed `compute_dispatch` client
- **4 upstream absorption candidates** for barraCuda to consider

## barraCuda Consumption Inventory

### Modules Used (by import count)

| Module | ~Imports | Key Types/Functions |
|--------|----------|---------------------|
| `stats` | 45+ | mean, variance, std_dev, pearson, spearman, rmse, mbe, bootstrap_ci, percentile, norm_ppf, fit_linear/quadratic/exponential/logarithmic, shannon, simpson, bray_curtis, chao1, pielou, moving_window_stats_f64, hargreaves_et0_batch, crop_coefficient |
| `ops` | 35+ | BatchedElementwiseF64, FusedMapReduceF64, VarianceF64, CorrelationF64, MovingWindowStats, KrigingF64, AutocorrelationF64, DiversityFusionGpu, JackknifeMeanGpu, BootstrapMeanGpu |
| `device` | 25+ | WgpuDevice, test_pool::tokio_block_on, PrecisionRoutingAdvice, F64BuiltinCapabilities, probe_f64_builtins, Fp64Strategy, GpuDriverProfile |
| `optimize` | 15+ | brent, BrentGpu, nelder_mead, multi_start_nelder_mead |
| `validation` | 10+ | ValidationHarness, exit_no_gpu, gpu_required |
| `pde` | 8+ | solve_richards, RichardsGpu, CrankNicolsonConfig, HeatEquation1D, SoilParams |
| `tolerances` | 5+ | Tolerance, check (4 domain submodules: atmospheric, soil, gpu, instrument) |
| `shaders::provenance` | 5+ | shaders_consumed_by, evolution_report, cross_spring_matrix, SpringDomain, ShaderRecord |
| `linalg` | 5+ | tridiagonal_solve, ridge_regression |
| `pipeline` | 2 | StatefulPipeline, BatchedStatefulF64 |
| `special` | 1 | regularized_gamma_p |
| `error` | 1 | BarracudaError |

### GPU Ops (all via `BatchedElementwiseF64`)

| Op | Domain | Module |
|----|--------|--------|
| 0 | FAO-56 PM ET₀ | `gpu/et0.rs` |
| 1 | Water balance | `gpu/water_balance.rs` |
| 5 | Sensor calibration | `gpu/sensor_calibration.rs` |
| 6 | Hargreaves ET₀ | `gpu/hargreaves.rs` |
| 7 | Kc climate adjustment | `gpu/kc_climate.rs` |
| 8 | Dual crop coefficient | `gpu/dual_kc.rs` |
| 9–10 | Van Genuchten θ(h)/K(h) | `gpu/van_genuchten.rs` |
| 11 | Thornthwaite PET | `gpu/thornthwaite.rs` |
| 12 | Growing degree days | `gpu/gdd.rs` |
| 13 | Pedotransfer | `gpu/pedotransfer.rs` |
| 14 | Makkink ET₀ | `gpu/simple_et0.rs` |
| 15 | Turc ET₀ | `gpu/simple_et0.rs` |
| 16 | Hamon PET | `gpu/simple_et0.rs` |
| 17 | SCS-CN runoff | `gpu/runoff.rs` |
| 18 | Yield response | `gpu/yield_response.rs` |
| 19 | Blaney-Criddle ET₀ | `gpu/simple_et0.rs` |

### Specialized GPU Primitives (beyond BatchedElementwiseF64)

| Primitive | Shader | Usage |
|-----------|--------|-------|
| `BrentGpu` | `brent_f64.wgsl` | VG inverse, Green-Ampt infiltration |
| `RichardsGpu` | `pde_richards` | Richards PDE (Picard iteration) |
| `FusedMapReduceF64` | `fused_map_reduce_f64.wgsl` | Seasonal reduction |
| `VarianceF64` | `variance_f64.wgsl` | Atlas drift detection |
| `CorrelationF64` | `correlation_f64.wgsl` | Cross-station analysis |
| `MovingWindowStats` | `moving_window_stats.wgsl` | Streaming IoT |
| `KrigingF64` | `kriging_f64.wgsl` | Spatial interpolation |
| `AutocorrelationF64` | `autocorrelation_f64.wgsl` | Temporal analysis |
| `BootstrapMeanGpu` | `bootstrap_mean_f64.wgsl` | Confidence intervals |
| `JackknifeMeanGpu` | `jackknife_mean_f64.wgsl` | Leave-one-out estimation |
| `DiversityFusionGpu` | `diversity_fusion_f64.wgsl` | Shannon/Simpson/Bray-Curtis batch |

## Write → Absorb → Lean Status

**All complete.** Zero local WGSL shaders remain.

| Module | Absorbed Into | Status |
|--------|---------------|--------|
| ValidationRunner | `barracuda::validation::ValidationHarness` | Leaning |
| van_genuchten | `barracuda::pde::richards::SoilParams` | Leaning |
| isotherm NM | `barracuda::optimize::nelder_mead` | Leaning |
| StatefulPipeline | `barracuda::pipeline::stateful` | Leaning |
| BatchedStatefulF64 | `barracuda::pipeline::batched_stateful` | Available |
| BrentGpu | `barracuda::optimize::brent_gpu` | Available |
| RichardsGpu | `barracuda::pde::richards_gpu` | Available |
| 6 metalForge modules | `barracuda::stats::*` + `barracuda::eco::*` | Leaning |
| 6 local ops (14–19) | `BatchedElementwiseF64` | Leaning |

## toadStool Integration

### Current: Typed `compute_dispatch` Client

| Method | Purpose |
|--------|---------|
| `compute.dispatch.submit` | Submit GPU workload → `job_id` |
| `compute.dispatch.result` | Poll result by `job_id` |
| `compute.dispatch.capabilities` | Query available compute |

Discovery: `biomeos::discover_primal_socket(primal_names::TOADSTOOL)` or
`AIRSPRING_COMPUTE_PRIMAL` env override.

### Validation: `validate_toadstool_dispatch` binary verifies dispatch path.

## Upstream Absorption Candidates

Patterns airSpring pioneered or refined that barraCuda could absorb:

| Candidate | What | Benefit |
|-----------|------|---------|
| `parse_capabilities` | 4-format capability discovery (flat, object, nested, double-nested) | Used by 4+ springs; standardize in `barracuda::biomeos` |
| `IpcError` | Structured IPC error type (ConnectionFailed, Timeout, RpcError, DeserializationFailed) | Replace ad-hoc error handling in all springs |
| `extract_rpc_error` | `fn(Value) -> Option<(i64, String)>` | Centralize JSON-RPC error extraction |
| Named physical constants | 55+ FAO-56, SCS-CN, Saxton-Rawls constants | Validate domain-specific precision |

## Evolution Requests

### For barraCuda

| Request | Priority | Context |
|---------|----------|---------|
| `TensorSession` f64 support | High | airSpring seasonal pipeline would benefit from fused multi-op f64 sessions |
| `compile_shader_df64_streaming` | Medium | neuralSpring V24 request; airSpring would use for atlas-scale streaming |
| `NpuDispatch` trait | Low | Replace local `npu.rs` with upstream abstraction |
| `batched_multinomial.wgsl` | Low | groundSpring V10 request; airSpring rarefaction use case |

### For toadStool

| Request | Priority | Context |
|---------|----------|---------|
| Confirm `compute.dispatch.submit` response format | High | airSpring expects `{"job_id": "..."}` |
| Songbird `http.request` latency headers | Medium | SongbirdHttpProvider needs timeout control |
| Provenance Registry indexing | Low | `toadstool.provenance` cross-spring flow matrix |

## What Worked Well

- **Write→Absorb→Lean** cycle prevented scope bloat — all 6 local ops retired cleanly
- **Dependency injection** for biomeOS socket discovery eliminated `set_var`/`remove_var` safety issues
- **Centralized tolerances** (60 constants, 4 submodules) made parity verification systematic
- **Zero-panic validation** (47 binaries) prevented CI noise from panic backtraces
- **Proptest fuzz** caught no bugs but provides regression safety for IPC infrastructure

## Next Steps (V0.8.8 Candidates)

- Wire GPU orchestrators to optionally route through `compute_dispatch::submit()`
  with local `WgpuDevice` fallback
- Expand `extract_rpc_error()` usage to all IPC call sites
- Profile `compute_gpu()` at N=100K+ for atlas-scale crossover analysis
- ISSUE-008: ET₀ GPU promotion (Thornthwaite, Makkink, Turc, Hamon → Tier A)
- ISSUE-013: Content convergence — document ET₀ convergence candidates

## Superseded Handoffs (archived)

- `AIRSPRING_V085_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR16_2026.md` → `archive/`
- `AIRSPRING_V085_CROSS_SPRING_ABSORPTION_HANDOFF_MAR16_2026.md` → `archive/`
- `AIRSPRING_V086_DEEP_EXECUTION_HANDOFF_MAR16_2026.md` → `archive/`

---
*ScyBorg Provenance Trio: AGPL-3.0-or-later (code) + ORC (game mechanics) + CC-BY-SA 4.0 (creative content)*
