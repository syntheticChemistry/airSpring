# SPDX-License-Identifier: AGPL-3.0-or-later

# airSpring → BarraCuda/ToadStool: V0.6.9 Absorption Targets + Evolution Learnings

**Date:** 2026-03-05
**From:** airSpring v0.6.9
**To:** BarraCuda/ToadStool team
**Covers:** V0.6.8 deep debt audit (3 rounds) + V0.6.9 f64-canonical local compute
**Supersedes:** V068 Deep Debt ToadStool Handoff (Mar 04)
**License:** AGPL-3.0-or-later

---

## Executive Summary

airSpring v0.6.9 is fully synced with standalone BarraCuda 0.3.1. 852 lib + 33
integration + 62 forge tests pass, 0 clippy warnings (pedantic+nursery), 95.66%
line coverage, zero unsafe code, zero mocks in production. Three rounds of deep
debt audit completed: provenance normalization, structured error handling,
streaming I/O, SubmitParams refactoring, BarraCuda primitive wiring, and
env-configurable RPC timeout. 78 experiments, 1237/1237 Python baselines with
full commit-level provenance.

**Absorption targets for BarraCuda** (highest impact first):
1. **6 f64-canonical local WGSL ops** — `local_elementwise_f64.wgsl` ready for upstream absorption
2. **`compile_shader_universal()` validation results** — confirmed f64→f32 downcast correctness for 6 agricultural ops
3. **NVK/Mesa f64 reliability finding** — 10% dispatch failure on Titan V NVK worth tracking in hardware matrix

---

## Part 1: Absorption Targets

### 1.1 Local WGSL Shader (6 ops) → BarraCuda Absorption Candidate

airSpring wrote `local_elementwise_f64.wgsl` as a single multi-op WGSL compute
shader with 6 element-wise operations. These are **not yet in BarraCuda** — they
use `wgpu` directly via `gpu::local_dispatch::LocalElementwise`.

| Op | Domain | Function | Inputs | Status |
|----|--------|----------|--------|--------|
| 0 | SCS-CN runoff | `scs_cn_runoff(P, CN, Ia_ratio)` | P(mm), CN, Ia ratio | Validated CPU + GPU |
| 1 | Stewart yield | `yield_ratio_single(Ky, ETa/ETc)` | Ky, ratio, — | Validated CPU + GPU |
| 2 | Makkink ET₀ | `makkink_et0(T, Rs, elev)` | T(°C), Rs(MJ), elev(m) | Validated CPU + GPU |
| 3 | Turc ET₀ | `turc_et0(T, Rs, RH)` | T(°C), Rs(MJ), RH(%) | Validated CPU + GPU |
| 4 | Hamon PET | `hamon_pet(T, lat, DOY)` | T(°C), lat(rad), DOY | Validated CPU + GPU |
| 5 | Blaney-Criddle ET₀ | `blaney_criddle(T, lat, DOY)` | T(°C), lat(rad), DOY | Validated CPU + GPU |

**Absorption path**: These 6 ops could become `batched_elementwise_f64` ops 14–19,
following the existing pattern for ops 0–13. Alternatively, BarraCuda could absorb
the `local_elementwise_f64.wgsl` shader directly and compile it via
`compile_shader_universal()`.

**Source files:**
- WGSL: `barracuda/src/shaders/local_elementwise_f64.wgsl`
- Dispatcher: `barracuda/src/gpu/local_dispatch.rs` (460 lines production + 190 lines tests)
- CPU references: `eco::runoff::scs_cn_runoff`, `eco::yield_response::yield_ratio_single`, `eco::simple_et0::{makkink,turc,hamon,blaney_criddle}`

### 1.2 compile_shader_universal() Validation

airSpring validated that `compile_shader_universal()` correctly compiles f64-canonical
WGSL to f32 on consumer GPUs while preserving f64 on pro GPUs. Key findings:

| Metric | F64 (Titan V) | F32 (RTX 4070) |
|--------|---------------|----------------|
| SCS-CN relative error | < 1e-10 | < 1e-3 |
| Stewart yield relative error | < 1e-12 | < 1e-4 |
| Makkink ET₀ relative error | < 1e-8 | < 2e-3 |
| Turc ET₀ relative error | < 1e-8 | < 2e-3 |
| Hamon PET relative error | < 1e-6 | < 5e-3 |
| Blaney-Criddle ET₀ relative error | < 1e-6 | < 5e-3 |

All within documented, justified tolerances for agricultural science (FAO-56 needs
~6 significant digits; f32 gives ~7).

### 1.3 NVK/Mesa f64 Reliability

Titan V via NVK/Mesa (not proprietary NVIDIA driver) shows ~10% f64 dispatch failure
rate. Failures are silent (wgpu returns zeros). This is a driver maturity issue, not
a hardware limitation. Recommendation: document in BarraCuda hardware compatibility
matrix, add `is_device_lost()` check after dispatch for NVK paths.

---

## Part 2: What airSpring Leans On (BarraCuda Primitives Consumed)

airSpring consumes **25 Tier A GPU modules** + **6 GPU-universal local ops** from BarraCuda:

| Category | Primitives | Count |
|----------|-----------|-------|
| `batched_elementwise_f64` | ops 0–13 (ET₀, WB, sensor cal, Hargreaves, Kc, dual Kc, VG, Thornthwaite, GDD, pedotransfer) | 14 |
| Dedicated GPU shaders | kriging, fused_map_reduce, moving_window, richards, nelder_mead, mc_et0, jackknife, bootstrap, diversity, linear_regression, matrix_correlation | 11 |
| CPU primitives | ridge_regression, brent optimizer, diversity, validation harness, stats (pearson, spearman, bootstrap_ci, variance, std_dev) | 8 |
| Local f64-canonical | `compile_shader_universal` dispatch (6 ops) | 6 |
| Pipeline | seasonal_pipeline, atlas_stream, UnidirectionalPipeline | 3 |

**Total**: 42 BarraCuda touchpoints. Zero local math duplication (variance/std_dev
wired to `barracuda::stats::correlation` in deep debt audit round 3).

---

## Part 3: Deep Debt Audit Learnings (Relevant to BarraCuda Evolution)

### 3.1 Provenance Normalization

All 47 benchmark JSONs normalized to `"_provenance"` key (with leading underscore)
for consistent parsing. Python baseline scripts now dynamically fetch git commit
hash for reproducibility. Recommendation: BarraCuda validation harness could enforce
`_provenance` convention.

### 3.2 Structured Error Handling for Validation Binaries

Introduced `json_f64_required()` helper that calls `std::process::exit(1)` with
clear error message instead of panicking. Follows hotSpring pattern: validation
binaries should never panic — explicit pass/fail with exit code 0/1.

### 3.3 SubmitParams Pattern for wgpu Dispatch

Replaced 8-argument `submit_and_copy()` with `SubmitParams` struct. This is a
reusable pattern for any wgpu compute dispatch with >5 parameters. BarraCuda's
own dispatch functions could benefit from this pattern.

### 3.4 env-Configurable RPC Timeout

`BIOMEOS_RPC_TIMEOUT_SECS` env var (default 5s) for JSON-RPC socket timeout.
All biomeOS socket paths are already env-driven (`BIOMEOS_SOCKET_DIR`,
`XDG_RUNTIME_DIR`, `BIOMEOS_FAMILY_ID`). This completes the capability-based
configuration of inter-primal communication.

### 3.5 Streaming JSON I/O

Replaced `read_to_string` + `from_str` with `from_reader(BufReader::new(file))`
for benchmark JSON loading. Zero-copy where possible. Recommendation: BarraCuda
validation harness could standardize this pattern.

---

## Part 4: What's NOT in airSpring (Gaps for BarraCuda)

| Gap | Status | Notes |
|-----|--------|-------|
| **Kokkos GPU parity** | Not started | No Kokkos baselines. Documented in `specs/BARRACUDA_REQUIREMENTS.md`. Candidate: ET₀ batch as entry point. |
| **Python CPU benchmarks for BarraCuda** | Partial | `bench_cpu_vs_python` covers 21 algorithms (14.5× geometric mean). No standalone BarraCuda-only Python benchmark suite. |
| **WGSL shader unit tests** | Via CPU parity | Shaders validated via CPU↔GPU parity, not standalone WGSL test harness. |
| **f16 / bf16 precision** | Not explored | Agricultural science needs ≥f32. f16/bf16 irrelevant for this domain. |

---

## Part 5: Action Items

### BarraCuda team actions:
1. **Absorb 6 local ops** as `batched_elementwise_f64` ops 14–19 (or dedicated shader)
2. **Add NVK f64 reliability note** to hardware compatibility matrix
3. **Consider `json_f64_required` pattern** for validation harness
4. **Consider `SubmitParams` pattern** for wgpu dispatch functions
5. **Consider `_provenance` convention** for benchmark JSON standard

### airSpring actions (post-absorption):
1. Delete `local_elementwise_f64.wgsl` and `gpu::local_dispatch` (lean on upstream)
2. Rewire ops 14–19 through `batched_elementwise_f64` (same pattern as ops 0–13)
3. Update cross-spring evolution benchmark

---

## Appendix: File Manifest

| File | Lines | Purpose |
|------|-------|---------|
| `src/shaders/local_elementwise_f64.wgsl` | ~120 | f64-canonical WGSL (6 ops) |
| `src/gpu/local_dispatch.rs` | 660 | LocalElementwise dispatcher + tests |
| `src/validation.rs` | ~220 | json_f64, json_f64_required, ValidationHarness init |
| `src/rpc.rs` | ~400 | JSON-RPC 2.0 with env-configurable timeout |
| `src/gpu/reduce.rs` | ~250 | Wired to barracuda::stats::correlation |
| `src/eco/richards.rs` | ~550 | Richards PDE (absorbed upstream S40) |
