# airSpring V052 — ToadStool S87 Sync & Revalidation

**Date**: 2026-03-02
**From**: airSpring v0.6.8 (Eastgate)
**To**: ToadStool/BarraCuda team
**ToadStool Pin**: S87 (`2dc26792`)
**Previous Pin**: S86 (`2fee1969`)
**Direction**: airSpring → ToadStool (unidirectional)

---

## Executive Summary

airSpring synced from ToadStool S86 to S87 with **zero breakage**. All 846 lib tests
pass, all 61 forge tests pass, 0 clippy pedantic warnings. The S86→S87 delta (66 files,
1,417 insertions) includes gpu_helpers refactoring, FHE shader arithmetic fixes, async-trait
reclassification, unsafe audit, and hardware_verification hardening — none of which affect
airSpring's API surface (ops 0-13 + BrentGpu + RichardsGpu + all GPU orchestrators stable).

This handoff also serves as a status update on what ToadStool has evolved since airSpring
last synced, and what remains pending for ToadStool to absorb.

---

## Part 1: What Changed (S86 → S87)

### 1.1 ToadStool S87 Highlights (relevant to airSpring)

| Change | Impact on airSpring |
|--------|-------------------|
| `gpu_helpers.rs` refactored (663L → 3 submodules) | None — internal to sparse linalg |
| FHE shader arithmetic fixes (NTT/INTT/pointwise) | None — airSpring doesn't use FHE |
| `TODO(afit)` → `NOTE(async-dyn)` (75 instances) | None — documentation reclassification |
| Unsafe audit (60+ sites documented) | None — already safe wrappers |
| `hardware_verification` 13/13 pass | Good — validates airSpring's hardware path |
| MatMul shape validation | Good — protects airSpring's linalg usage |
| `BarracudaError::is_device_lost()` | Good — better error handling for GPU paths |

### 1.2 S80-S86 Evolution (absorbed since last sync)

These were already available at S86 but worth documenting for completeness:

| Session | Key Deliverable | airSpring Usage |
|---------|----------------|-----------------|
| S80 | Nautilus absorption (7 files, 22 tests) | `AirSpringBrain` via `barracuda::nautilus` |
| S80 | BatchedEncoder (fused multi-op GPU pipeline) | Available for seasonal pipeline optimization |
| S80 | Batch Nelder-Mead GPU | `gpu::isotherm` multi-start fitting |
| S80 | StatefulPipeline<S> + WaterBalanceState | `gpu::seasonal_pipeline` GPU step dispatch |
| S80 | GpuDriverProfile workarounds | NVK Taylor-series sin/cos preamble |
| S83 | BrentGpu + RichardsGpu + L-BFGS | `gpu::infiltration` + `gpu::richards` |
| S83 | BatchedStatefulF64 | Water balance day-over-day tracking |
| S84-86 | ComputeDispatch: 111→144 ops | Future: migrate airSpring dispatch patterns |
| S87 | Device-lost recovery improvements | Better error propagation in GPU paths |

---

## Part 2: Revalidation Results

### 2.1 Test Suite (S87 HEAD `2dc26792`)

| Suite | Result |
|-------|--------|
| `barracuda` lib tests | **846 passed**, 0 failed |
| `metalForge` lib tests | **61 passed**, 0 failed |
| `cargo clippy --pedantic` (barracuda) | **0 warnings** |
| `cargo clippy --pedantic` (forge) | **0 warnings** |
| `cargo build` | **Clean** (0 errors, 0 warnings) |

### 2.2 API Stability

All airSpring-consumed APIs remain stable across S86→S87:

| API | Status |
|-----|--------|
| `barracuda::ops::batched_elementwise_f64` (ops 0-13) | **Stable** |
| `barracuda::ops::kriging_f64` | **Stable** |
| `barracuda::ops::fused_map_reduce_f64` | **Stable** |
| `barracuda::pde::richards_gpu` | **Stable** |
| `barracuda::optimize::brent_gpu` | **Stable** |
| `barracuda::optimize::nelder_mead` | **Stable** |
| `barracuda::device::WgpuDevice` | **Stable** |
| `barracuda::validation::ValidationHarness` | **Stable** |
| `barracuda::nautilus` | **Stable** |
| `barracuda::stats::hydrology::fao56_et0` | **Stable** |

---

## Part 3: Pending — airSpring V051 Absorption

The V051 handoff (airSpring v0.6.8 → ToadStool S86) proposed 6 new `batched_elementwise_f64`
ops (14-19) for ToadStool to absorb. These remain pending:

| Proposed Op | Name | airSpring Local Op | Status |
|-------------|------|-------------------|--------|
| 14 | SCS-CN Runoff | `local_elementwise.wgsl` op=0 | **Pending** |
| 15 | Stewart Yield | `local_elementwise.wgsl` op=1 | **Pending** |
| 16 | Makkink ET₀ | `local_elementwise.wgsl` op=2 | **Pending** |
| 17 | Turc ET₀ | `local_elementwise.wgsl` op=3 | **Pending** |
| 18 | Hamon PET | `local_elementwise.wgsl` op=4 | **Pending** |
| 19 | Blaney-Criddle | `local_elementwise.wgsl` op=5 | **Pending** |

All helper functions (`sat_vp`, `vp_slope`, `daylight_hr`, `atm_pressure`, `psychrometric`)
already exist in ToadStool's DF64 transcendental library. Absorption is composition,
not new development. See V051 handoff for full details.

Once absorbed, airSpring will:
1. Switch `gpu::runoff/yield_response/simple_et0` from `LocalElementwise` to `BatchedElementwiseF64`
2. Update `metalForge` workloads from `ShaderOrigin::Local` to `ShaderOrigin::Absorbed`
3. Tighten validation tolerances from f32 to f64

---

## Part 4: ToadStool Evolution Highlights (S68 → S87)

For the record, here's the full scope of ToadStool's evolution since airSpring's
initial deep integration:

### Precision Architecture (S68)
- **Zero f32-only shaders** — all 844 WGSL shaders are f64 canonical
- **Dual-layer universal precision**: `op_preamble` (per-shader) + `df64_rewrite` (naga IR)
- **DF64 transcendentals**: 15 functions (exp, log, pow, sin, cos, tan, asin, acos, atan, atan2, sqrt, cbrt, gamma, erf, erfc)

### Cross-Spring Absorption (S69-S76)
- 5 spring handoffs absorbed (196 handoff files reviewed)
- 30+ new WGSL shaders + dispatch wired
- airSpring ops 0-13 fully integrated
- Seasonal pipeline, BrentGpu, RichardsGpu, StatefulPipeline

### Deep Debt Evolution (S70-S87)
- `anyhow` → `thiserror` (all 30+ crates)
- `chrono` → `std::time` (28 crates)
- `pollster` → `tokio_block_on`
- `serde_yaml` → `serde_yaml_ng`
- `async-trait` → native AFIT (5 crates, 75 remaining as conscious arch decision)
- `libc` → `rustix` (akida-driver)
- 37+ god files refactored (all < 1000 lines)
- ComputeDispatch: 144 ops migrated (~139 remaining)
- Zero production stubs/mocks, zero blind unwraps, zero hardcoded IPs

### Quality State (S87)
- 2,866 barracuda tests, 5,500+ workspace tests
- 0 clippy warnings, 0 fmt diffs
- ~60+ unsafe blocks (all documented with SAFETY comments)
- 36 crates with `#![deny(unsafe_code)]`

---

## Part 5: Handoff Chain

| Version | Scope | Status |
|---------|-------|--------|
| V051 | v0.6.8 local GPU + NUCLEUS routing + absorption handoff | Current (ops 14-19 pending) |
| **V052** | **ToadStool S87 sync + revalidation** | **Current** |

---

*airSpring v0.6.8 — ToadStool S87 (`2dc26792`). 846 lib + 61 forge tests, 0 clippy.
All APIs stable. 6 local WGSL ops pending absorption (proposed ops 14-19).*
