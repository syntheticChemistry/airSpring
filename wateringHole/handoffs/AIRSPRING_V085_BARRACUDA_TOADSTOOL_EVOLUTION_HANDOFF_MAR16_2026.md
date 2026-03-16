# airSpring V0.8.5 — barraCuda / toadStool Evolution Handoff

**Date**: 2026-03-16
**From**: airSpring (ecological & agricultural sciences niche)
**To**: barraCuda team, toadStool team, All Springs
**License**: AGPL-3.0-or-later
**Scope**: Comprehensive inventory of airSpring's barracuda consumption, upstream absorption candidates, evolution requests, and cross-spring learnings

---

## Executive Summary

airSpring v0.8.5 consumes 20 `BatchedElementwiseF64` ops (0-19), 11 specialized GPU primitives, 6 optimization/PDE modules, 5 stats modules, and the validation harness. All local WGSL shaders are absorbed upstream — zero local shaders remain. The Write→Absorb→Lean cycle is **complete** for all GPU ops. This handoff documents what airSpring learned that is relevant to barraCuda/toadStool evolution.

---

## Part 1: barraCuda Primitive Consumption

### BatchedElementwiseF64 Ops (20/20 upstream)

| Op | Domain | airSpring Module | Provenance |
|----|--------|------------------|------------|
| 0 | FAO-56 PM ET₀ | `gpu::et0` | hotSpring `pow_f64` fix (TS-001) |
| 1 | Water balance step | `gpu::water_balance` | Multi-spring shared |
| 5 | Sensor calibration | `gpu::sensor_calibration` | Dong et al. 2024 |
| 6 | Hargreaves-Samani ET₀ | `gpu::hargreaves` | FAO-56 Eq. 52 |
| 7 | Kc climate adjustment | `gpu::kc_climate` | FAO-56 Eq. 62 |
| 8 | Dual Kc Ke | `gpu::dual_kc` | FAO-56 Ch 7 |
| 9 | Van Genuchten θ(h) | `gpu::van_genuchten` | hotSpring precision |
| 10 | Van Genuchten K(h) | `gpu::van_genuchten` | Mualem 1976 |
| 11 | Thornthwaite ET₀ | `gpu::thornthwaite` | S79 absorption |
| 12 | Growing degree days | `gpu::gdd` | S79 absorption |
| 13 | Pedotransfer | `gpu::pedotransfer` | Saxton-Rawls 2006 |
| 14 | Makkink ET₀ | `gpu::simple_et0` | v0.7.2 absorption |
| 15 | Turc ET₀ | `gpu::simple_et0` | v0.7.2 absorption |
| 16 | Hamon PET | `gpu::simple_et0` | v0.7.2 absorption |
| 17 | SCS-CN runoff | `gpu::runoff` | NEH-4 |
| 18 | Stewart yield response | `gpu::yield_response` | FAO-56 Table 24 |
| 19 | Blaney-Criddle ET₀ | `gpu::simple_et0` | USDA TP-96 |

### Specialized GPU Primitives

| Primitive | airSpring Use | Notes |
|-----------|--------------|-------|
| `FusedMapReduceF64` | Seasonal aggregation (`gpu::reduce`) | N≥1024 threshold |
| `VarianceF64` | Seasonal statistics | Welford algorithm |
| `MovingWindowStats` | IoT stream smoothing (`gpu::stream`) | Real-time filtering |
| `KrigingF64` | Spatial interpolation (`gpu::kriging`) | Ordinary kriging |
| `AutocorrelationF64` | ET₀ temporal persistence | Lag-1 ACF |
| `CorrelationF64` | Cross-variable correlation | Pearson r |
| `BootstrapMeanGpu` | Bootstrap CI (`gpu::bootstrap`) | Deterministic seed |
| `JackknifeMeanGpu` | Jackknife variance (`gpu::jackknife`) | LOO estimate |
| `DiversityFusionGpu` | Shannon/Simpson diversity (`gpu::diversity`) | Cross-spring: wetSpring bio |
| `BrentGpu` | VG θ→h inversion, Green-Ampt infiltration | GPU root-finding |
| `RichardsGpu` | 1D Richards PDE solver | Picard iteration |

### Optimization & PDE

| Module | airSpring Use |
|--------|--------------|
| `optimize::brent` | VG θ→h CPU path |
| `optimize::brent_gpu::BrentGpu` | Green-Ampt GPU infiltration |
| `optimize::nelder_mead` | Isotherm fitting |
| `optimize::lbfgs` | Pipeline type checks |
| `pde::richards` | 1D Richards CPU |
| `pde::richards_gpu::RichardsGpu` | 1D Richards GPU (Picard) |
| `pde::crank_nicolson` | CN diffusion scheme |

### Stats & Special Functions

| Module | airSpring Use |
|--------|--------------|
| `stats::normal::norm_ppf` | MC ET₀ uncertainty, SPI drought |
| `stats::regression::fit_linear` | Lysimeter validation |
| `stats::pearson_correlation` | Cross-station correlation |
| `stats::jackknife` / `stats::bootstrap` | Confidence intervals |
| `stats::hydrology` | Pipeline type checks |
| `special::gamma::regularized_gamma_p` | SPI gamma CDF |

### Infrastructure

| Module | airSpring Use |
|--------|--------------|
| `validation::ValidationHarness` | All 91 validation binaries |
| `device::WgpuDevice` | All GPU orchestrators |
| `device::driver_profile::PrecisionRoutingAdvice` | Cross-spring modern validation |
| `device::probe::F64BuiltinCapabilities` | Device probing |
| `shaders::provenance` | Cross-spring provenance tracking |
| `pipeline::StatefulPipeline` | Day-over-day water balance |
| `pipeline::batched_stateful::BatchedStatefulF64` | GPU-resident ping-pong state |
| `nautilus` | Evolutionary reservoir computing |
| `spectral::anderson` | Anderson coupling |
| `error::BarracudaError` | Error conversion |

---

## Part 2: Write → Absorb → Lean Status

**Complete.** All local GPU ops have been absorbed upstream:

| Phase | What | Status |
|-------|------|--------|
| S40 | `van_genuchten` → `barracuda::pde::richards` | **Leaning** |
| S59 | `ValidationRunner` → `ValidationHarness` | **Leaning** |
| S62 | Isotherm NM → `barracuda::optimize::nelder_mead` | **Leaning** |
| S64-S66 | 6 metalForge modules → upstream | **Leaning** (6/6) |
| S70+ | Ops 5-8 GPU-first → `BatchedElementwiseF64` | **Leaning** |
| S79 | Ops 9-13 + jackknife/bootstrap/diversity | **Leaning** |
| S80 | `StatefulPipeline` → `barracuda::pipeline` | **Leaning** |
| S83 | `BatchedStatefulF64` → `barracuda::pipeline` | **Available** |
| v0.7.2 | Ops 14-19, `local_dispatch` retired | **Leaning** |

**Zero local WGSL shaders.** airSpring has no `.wgsl` files — all shaders are upstream in barraCuda.

---

## Part 3: Upstream Absorption Candidates

These patterns emerged from airSpring (and are now used by 3-5 springs). They are candidates for centralization in barraCuda or a shared ecosystem crate.

### 3.1 `parse_capabilities` — Dual-Format Capability Discovery

airSpring, groundSpring, neuralSpring, ludoSpring, and healthSpring all independently implement capability list parsing that handles both string arrays (`["health", "compute.dispatch"]`) and object arrays (`[{"name": "health", "version": "1.0"}]`). airSpring's implementation is in `biomeos.rs`.

**Recommendation**: Absorb into `barracuda::biomeos::parse_capabilities` or a shared `biomeos-common` crate.

### 3.2 `IpcError` — Structured IPC Error Type

airSpring evolved `rpc::send()` from `Option<Value>` to `Result<Value, IpcError>` with variants: `ConnectionFailed`, `Timeout`, `RpcError`, `DeserializationFailed`. This same pattern exists in groundSpring and neuralSpring.

**Recommendation**: Absorb into `barracuda::ipc::IpcError` or a shared IPC crate.

### 3.3 `SongbirdHttpProvider` — Tower Atomic HTTP Pattern

airSpring and neuralSpring both route HTTP through Songbird's `http.request` capability via JSON-RPC IPC. This eliminates the `ureq`/`ring` C dependency chain entirely.

**Recommendation**: Consider a shared `songbird-http` crate or module in barracuda for springs that need standalone HTTP.

### 3.4 Named Physical Constants

airSpring extracted ~55 named constants with source citations (FAO-56, SCS-CN, AMC Hawkins, Saxton-Rawls). These are domain-specific but overlap with groundSpring's agricultural constants.

**Recommendation**: Consider `barracuda::constants::fao56::*` for shared agricultural constants used by multiple springs.

---

## Part 4: Evolution Requests to barraCuda

### 4.1 `TensorSession` f64 Support (Priority: High)

airSpring's seasonal pipeline chains ops 0→7→1→yield through CPU orchestration. With `TensorSession` f64, these could be fused into a single GPU dispatch — significant latency reduction for atlas-scale workloads.

**Current state**: `TensorSession` is f32-only. airSpring uses `BatchedElementwiseF64` per-op dispatch.

### 4.2 `compile_shader_df64_streaming` (Priority: Medium)

Simpler API for DF64 shader compilation. Currently requires manual emulation setup. A `compile_shader_df64_streaming(source, entry_point)` helper would reduce boilerplate.

### 4.3 NPU Primitives (Priority: Low)

airSpring has a local `npu/` module for AKD1000 inference. If `barracuda::npu` or `barracuda::nn` emerges, airSpring can absorb.

---

## Part 5: Evolution Requests to toadStool

### 5.1 `compute.dispatch` Wiring (Priority: High)

airSpring's GPU orchestrators dispatch directly through `wgpu` via `WgpuDevice`. They are ready to route through toadStool's `compute.dispatch` for live hardware selection. This would enable:
- Multi-GPU dispatch (RTX 4070 + Titan V)
- Hardware-aware precision routing
- Cross-system workload migration

**Current**: Direct `wgpu::Device` access.
**Target**: `rpc::send(toadstool_socket, "compute.dispatch", params)`.

### 5.2 Songbird `http.request` Enhancements (Priority: Medium)

neuralSpring's `SongbirdHttp` expects `save_to` (download to file) and `content_length` in Songbird responses. Ensure Songbird's `http.request` handler supports these for large downloads.

### 5.3 Provenance Registry Cross-Spring Indexing (Priority: Low)

airSpring uses `barracuda::shaders::provenance` for cross-spring tracking (32/32 checks in Exp 077). Consider cross-spring indexing in the provenance registry so toadStool can query "which springs contributed to this shader."

---

## Part 6: Cross-Spring Learnings

### What Worked

1. **Write→Absorb→Lean**: The discipline of implementing locally, validating against papers, then handing off to barraCuda eliminated all duplicate math. airSpring has zero local shaders.

2. **Feature-gated C deps**: Making `ureq` optional behind `standalone-http` allowed the default build to stay ecoBin-compliant while maintaining a development path. When Songbird matured, the migration was straightforward.

3. **`ValidationHarness` from upstream**: Using barraCuda's `ValidationHarness` (not a local copy) meant all 91 binaries automatically got upstream improvements (tolerance centralization, JSON provenance, structured output).

4. **`PrecisionRoutingAdvice`**: Hardware-aware precision routing from groundSpring V84 / toadStool S128 worked correctly across all 4 airSpring hardware targets (RTX 4070, Titan V, AKD1000, i9-12900K).

5. **Cross-spring diversity**: Importing `DiversityFusionGpu` from wetSpring bio and `BrentGpu` from optimization demonstrated that barraCuda primitives compose across domains.

### What Didn't Work

1. **Direct `wgpu` access in orchestrators**: Works but bypasses toadStool's hardware intelligence. Should evolve to `compute.dispatch`.

2. **`TensorSession` f32-only**: airSpring's core domain is f64 (agricultural physics). Couldn't use fused pipelines.

3. **Capability parsing fragmentation**: 5 springs implementing the same dual-format parsing logic independently. Should be centralized.

### Ecosystem Patterns Now Standard

| Pattern | Origin | Adopted By |
|---------|--------|------------|
| Zero-panic validation (`let...else`) | groundSpring V109 | airSpring, healthSpring |
| `#[expect(reason)]` | wetSpring V122 | airSpring, groundSpring, neuralSpring |
| Tower Atomic HTTP (Songbird IPC) | neuralSpring V108 | airSpring |
| Dual-format capability discovery | ludoSpring | airSpring, groundSpring, healthSpring |
| `IpcError` structured type | airSpring/groundSpring | neuralSpring |
| Named physical constants | groundSpring V109 | airSpring |
| `DispatchOutcome` biomeOS alignment | biomeOS | airSpring |

---

## Part 7: Quality Metrics

| Metric | Value |
|--------|-------|
| barraCuda version | v0.3.5 (wgpu 28, DF64 precision tier) |
| BatchedElementwiseF64 ops consumed | 20 (ops 0-19) |
| Specialized GPU primitives | 11 |
| Local WGSL shaders | **0** |
| Library tests | 865 |
| Integration tests | 285 |
| Validation binaries | 91 |
| `cargo check` warnings | 58 (pre-existing `missing_docs`) |
| C dependencies | **0** |
| `#[allow()]` in production | **0** |
| Cross-spring provenance records | 32/32 (5 springs) |
| CPU-GPU parity modules | 21/21 |

---

*AGPL-3.0-or-later · ScyBorg Provenance Trio*
