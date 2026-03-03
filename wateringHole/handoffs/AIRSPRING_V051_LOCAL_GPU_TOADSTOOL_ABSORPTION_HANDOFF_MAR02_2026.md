# airSpring V051 — Local GPU Compute Evolution & ToadStool Absorption Handoff

**Date**: 2026-03-02
**From**: airSpring v0.6.8 (Eastgate)
**To**: ToadStool/BarraCuda team
**ToadStool Pin**: S86
**Direction**: airSpring → ToadStool (unidirectional)

---

## Executive Summary

airSpring v0.6.8 evolved 6 element-wise ecological operations from CPU-only placeholders
to **live GPU compute** via a local WGSL shader (`local_elementwise.wgsl`), dispatched
through `wgpu` directly from the `gpu::local_dispatch::LocalElementwise` engine. This
handoff documents the work done, the mathematical formulas implemented, the precision
characteristics, and what ToadStool needs to absorb into canonical f64 shaders.

Additionally, `metalForge` workloads expanded from 21 to 27, and NUCLEUS mesh routing
was validated end-to-end (Exp 076: 60/60 PASS including PCIe P2P bypass modeling).

---

## Part 1: What We Built

### 1.1 Local WGSL Shader: `local_elementwise.wgsl`

**Location**: `barracuda/src/shaders/local_elementwise.wgsl`

A unified f32 compute shader with 6 element-wise operations selected by a `params.op`
uniform. Workgroup size 256. Three input arrays (`in_a`, `in_b`, `in_c`) → one output array.

| Op | Name | Equation | Inputs |
|----|------|----------|--------|
| 0 | SCS-CN Runoff | `Q = (P − Ia)² / (P − Ia + S)`, `S = 25400/CN − 254` | a=P(mm), b=CN, c=ia_ratio |
| 1 | Stewart Yield | `Ya/Ymax = 1 − Ky × (1 − ETa/ETc)` | a=Ky, b=ETa/ETc, c=unused |
| 2 | Makkink ET₀ | `0.61 × (Δ/(Δ+γ)) × Rs/λ − 0.12` | a=T(°C), b=Rs(MJ), c=elev(m) |
| 3 | Turc ET₀ | `0.013 × T/(T+15) × (23.8856Rs + 50)`, humidity branch | a=T(°C), b=Rs(MJ), c=RH(%) |
| 4 | Hamon PET | `0.1651 × N × ρsat(T)` | a=T(°C), b=lat(rad), c=doy |
| 5 | Blaney-Criddle | `p × (0.46T + 8.13)`, `p` from daylight fraction | a=T(°C), b=lat(rad), c=doy |

**Helper functions** implemented in-shader:
- `atm_pressure(elev)` — FAO-56 Eq. 7
- `psychrometric(P)` — FAO-56 Eq. 8
- `sat_vp(T)` — FAO-56 Eq. 11
- `vp_slope(T)` — FAO-56 Eq. 13
- `daylight_hr(lat, doy)` — FAO-56 Eq. 34 (sunrise hour angle → day length)

### 1.2 Dispatch Engine: `gpu::local_dispatch::LocalElementwise`

**Location**: `barracuda/src/gpu/local_dispatch.rs`

- Compiles `local_elementwise.wgsl` into a `wgpu::ComputePipeline` at construction time
- `dispatch(op, a, b, c) → Result<Vec<f64>>` handles:
  - f64 → f32 narrowing (CPU side)
  - GPU buffer allocation via `create_buffer_init`
  - Bind group creation (4 storage + 1 uniform)
  - Compute dispatch with `div_ceil` workgroup count
  - GPU → CPU readback via map_async
  - f32 → f64 widening (CPU side)
- Uses `barracuda::device::WgpuDevice` for the underlying `wgpu::Device` and `wgpu::Queue`

### 1.3 GPU Orchestrators Updated

| Module | Struct | LocalOp | Status |
|--------|--------|---------|--------|
| `gpu::runoff` | `GpuRunoff` | `ScsCnRunoff` | **GPU-local** |
| `gpu::yield_response` | `GpuYieldResponse` | `StewartYield` | **GPU-local** |
| `gpu::simple_et0` | `GpuSimpleEt0` | `Makkink`, `Turc`, `Hamon`, `BlaneyCriddle` | **GPU-local** |

Each orchestrator has `compute_cpu()` (f64) and `compute_gpu()` (f32 via `LocalElementwise`)
with unit tests validating CPU-GPU parity within f32 tolerance.

### 1.4 metalForge Workload Expansion

6 new `EcoWorkload` definitions with `ShaderOrigin::Local`:
- `scs_cn_batch`, `stewart_yield_batch`, `makkink_et0_batch`, `turc_et0_batch`, `hamon_pet_batch`, `blaney_criddle_et0_batch`

Total: **27 ecological workloads** (21 existing + 6 local).

---

## Part 2: What ToadStool Should Absorb

### 2.1 Six New `batched_elementwise_f64` Ops (Proposed: ops 14–19)

The 6 operations in `local_elementwise.wgsl` are pure element-wise arithmetic —
exactly the pattern `batched_elementwise_f64.wgsl` already handles for ops 0–13.

**Proposed mapping:**

| ToadStool Op | airSpring Local Op | Name | Complexity |
|--------------|--------------------|------|------------|
| 14 | 0 | SCS-CN Runoff | 2 inputs + ratio → 1 output, branching on (P − Ia > 0) |
| 15 | 1 | Stewart Yield | 2 inputs → 1 output, trivial |
| 16 | 2 | Makkink ET₀ | 3 inputs, needs `sat_vp`, `vp_slope`, `atm_pressure`, `psychrometric` |
| 17 | 3 | Turc ET₀ | 3 inputs, humidity branch (RH < 50 correction) |
| 18 | 4 | Hamon PET | 3 inputs, needs `daylight_hr`, `sat_vp` |
| 19 | 5 | Blaney-Criddle | 3 inputs, needs `daylight_hr` |

**Key precision note**: The local shader uses f32. ToadStool's `batched_elementwise_f64`
uses df64 emulation for full f64 precision. The helper functions (`sat_vp`, `vp_slope`,
`daylight_hr`) are already implemented in ToadStool's `df64_transcendentals` family.
Absorbing these ops means wiring the existing df64 helper functions to the new op cases.

### 2.2 Helper Functions Already Available in ToadStool

| airSpring Local Helper | ToadStool Equivalent | Notes |
|------------------------|---------------------|-------|
| `sat_vp(T)` | `df64_exp` in FAO-56 formula | 0.6108 × exp(17.27T/(T+237.3)) |
| `vp_slope(T)` | `df64_exp` derivative | 4098 × e_s / (T+237.3)² |
| `atm_pressure(elev)` | `df64_pow` | 101.3 × ((293−0.0065z)/293)^5.26 |
| `psychrometric(P)` | trivial multiply | 0.000665 × P |
| `daylight_hr(lat, doy)` | `df64_acos`, `df64_tan` | FAO-56 Eq. 34 |

All transcendental helpers exist in ToadStool's df64 library. Absorption should be
straightforward composition rather than new shader development.

### 2.3 Precision Characteristics

| Operation | f32 local tolerance | Expected f64 tolerance | Notes |
|-----------|--------------------|-----------------------|-------|
| SCS-CN | 0.5 mm | 1e-6 mm | Pure arithmetic, no transcendentals |
| Stewart | 0.001 | 1e-10 | Trivial multiply |
| Makkink | 0.15 mm/d | 1e-4 mm/d | exp() in sat_vp drives f32 error |
| Turc | 0.15 mm/d | 1e-4 mm/d | Similar to Makkink |
| Hamon | 0.1 mm/d | 1e-4 mm/d | exp() + daylight trig |
| Blaney-Criddle | 0.1 mm/d | 1e-4 mm/d | Daylight trig only |

Once absorbed into f64 ops, tolerances tighten by 6+ orders of magnitude.

---

## Part 3: Validation State

### Exp 075: `validate_local_gpu` (barracuda)

- 6 ops validated: CPU (f64) vs GPU (f32) parity
- Batch scaling tests: N=1, 100, 10000
- Edge cases: zero precipitation, zero stress, extreme temperatures
- All checks PASS within f32 precision tolerances

### Exp 076: `validate_nucleus_routing` (metalForge)

- 60/60 checks PASS
- 27 workloads capability-routed (6 new local ops included)
- 7-stage mixed-hardware pipeline: NPU→GPU→GPU→GPU→GPU→GPU→CPU
- PCIe P2P bypass: GPU↔NPU direct transfer (no CPU roundtrip)
- NUCLEUS atomics: Tower (crypto/mesh), Node (compute/GPU), Nest (storage/provenance)
- Multi-node cross-hop routing validated

### Test Counts (v0.6.8)

| Suite | Count |
|-------|-------|
| Barracuda lib | 846 |
| metalForge lib | 61 |
| Barracuda binaries | 80 |
| metalForge binaries | 5 |
| Experiments | 76 |
| clippy pedantic warnings | 0 |

---

## Part 4: Learnings for ToadStool Evolution

### 4.1 Local Shader Pattern

The `local_elementwise.wgsl` + `LocalElementwise` pattern demonstrates that springs
can prototype GPU compute locally using `wgpu` directly, validate against CPU baselines,
and then hand off to ToadStool for canonical f64 absorption. This is the "Write locally
→ Absorb upstream → Lean" cycle in action at the shader level.

**Recommendation**: ToadStool could provide a `LocalShaderKit` or similar scaffolding
to standardize this pattern across springs, reducing boilerplate in the `dispatch()` method.

### 4.2 Element-Wise Op Convention

The 6 new ops all follow the `(a, b, c) → result` pattern with a `params.op` switch.
This maps perfectly to `batched_elementwise_f64.wgsl`'s existing architecture. No new
shader infrastructure needed — just new `case` branches in the main switch.

### 4.3 Precision Ladder

The f32 → f64 promotion path is clean:
1. airSpring validates math correctness with f32 local shaders (wide tolerance)
2. ToadStool absorbs into df64 canonical shader (tight tolerance)
3. airSpring updates `ShaderOrigin::Local` → `ShaderOrigin::Absorbed` in metalForge workloads
4. Tolerances tighten automatically in validation binaries

### 4.4 metalForge `ShaderOrigin` Update

When ToadStool absorbs ops 14–19, the following metalForge workloads should update:
```
scs_cn_batch:          ShaderOrigin::Local → ShaderOrigin::Absorbed
stewart_yield_batch:   ShaderOrigin::Local → ShaderOrigin::Absorbed
makkink_et0_batch:     ShaderOrigin::Local → ShaderOrigin::Absorbed
turc_et0_batch:        ShaderOrigin::Local → ShaderOrigin::Absorbed
hamon_pet_batch:       ShaderOrigin::Local → ShaderOrigin::Absorbed
blaney_criddle_et0_batch: ShaderOrigin::Local → ShaderOrigin::Absorbed
```

---

## Part 5: Handoff Chain

| Version | Scope | Status |
|---------|-------|--------|
| V050 | v0.6.6 full evolution handoff (14 contributed, 25 consumed) | Superseded by V051 |
| **V051** | **v0.6.8 local GPU + NUCLEUS routing + absorption handoff** | **Current** |

---

## Part 6: Files Changed (v0.6.7 → v0.6.8)

| File | Change |
|------|--------|
| `barracuda/src/shaders/local_elementwise.wgsl` | **NEW** — 6-op WGSL compute shader |
| `barracuda/src/gpu/local_dispatch.rs` | **NEW** — `LocalElementwise` wgpu dispatch |
| `barracuda/src/gpu/mod.rs` | Added `pub mod local_dispatch`, updated module doc |
| `barracuda/src/gpu/runoff.rs` | `GpuRunoff` → GPU-local via `LocalElementwise` |
| `barracuda/src/gpu/yield_response.rs` | `GpuYieldResponse` → GPU-local |
| `barracuda/src/gpu/simple_et0.rs` | `GpuSimpleEt0` → GPU-local (4 methods) |
| `barracuda/src/gpu/evolution_gaps.rs` | Updated roadmap: 6 ops GPU-local, v0.6.8 section |
| `barracuda/src/bin/validate_local_gpu.rs` | **NEW** — Exp 075 |
| `barracuda/Cargo.toml` | v0.6.8, +wgpu, +bytemuck |
| `metalForge/forge/src/workloads.rs` | 6 new local workloads (27 total) |
| `metalForge/forge/src/bin/validate_nucleus_routing.rs` | **NEW** — Exp 076 |
| `metalForge/forge/Cargo.toml` | Added validate_nucleus_routing binary |

---

*airSpring v0.6.8 — 76 experiments, 846 lib + 61 forge tests, 85 binaries,
25 Tier A + 6 GPU-local, 27 metalForge workloads, ToadStool S86 synced.
Pure Rust + BarraCuda. AGPL-3.0-or-later.*
