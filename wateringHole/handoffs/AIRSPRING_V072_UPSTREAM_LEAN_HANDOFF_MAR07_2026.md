<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# airSpring V0.7.2 — Full Upstream Lean Handoff

**Date**: March 7, 2026
**From**: airSpring V0.7.2
**To**: barraCuda (P0), toadStool (P1), coralReef (P2), all Springs (P3)
**License**: AGPL-3.0-or-later
**Supersedes**: AIRSPRING_V071_DEEP_DEBT_NVK_TOADSTOOL_HANDOFF_MAR07_2026.md

---

## Executive Summary

Write → Absorb → Lean cycle **complete** for all 6 local GPU ops. airSpring
now fully leans on `BatchedElementwiseF64` (ops 14-19) from barraCuda HEAD.
The `local_dispatch` module, `LocalElementwise` dispatcher, and both local WGSL
shaders have been retired. Zero local shader code remains.

**844 lib tests, 0 failures.** All quality gates pass.

---

## 1. What Changed

### Retired Infrastructure

| Component | Lines | Status |
|-----------|-------|--------|
| `gpu/local_dispatch.rs` | 658 | **DELETED** — all consumers rewired |
| `shaders/local_elementwise_f64.wgsl` | 166 | **DELETED** — absorbed upstream |
| `shaders/local_elementwise.wgsl` | 107 | **DELETED** — absorbed upstream |

### Rewired Modules

| Module | Before | After |
|--------|--------|-------|
| `gpu::runoff` | `LocalElementwise` + `LocalOp::ScsCnRunoff` | `BatchedElementwiseF64` + `Op::ScsCnRunoff` (op=17) |
| `gpu::yield_response` | `LocalElementwise` + `LocalOp::StewartYield` | `BatchedElementwiseF64` + `Op::StewartYieldWater` (op=18) |
| `gpu::simple_et0` (Makkink) | `LocalElementwise` + `LocalOp::MakkinkEt0` | `BatchedElementwiseF64` + `Op::MakkinkEt0` (op=14) |
| `gpu::simple_et0` (Turc) | `LocalElementwise` + `LocalOp::TurcEt0` | `BatchedElementwiseF64` + `Op::TurcEt0` (op=15) |
| `gpu::simple_et0` (Hamon) | `LocalElementwise` + `LocalOp::HamonPet` | `BatchedElementwiseF64` + `Op::HamonEt0` (op=16) |
| `gpu::simple_et0` (BC) | `LocalElementwise` + `LocalOp::BlaneyCriddleEt0` | `BatchedElementwiseF64` + `Op::BlaneyCriddleEt0` (op=19) |

### Input Packing Changes

The local `LocalElementwise::dispatch(op, a, b, c)` used 3 separate buffers.
The upstream `BatchedElementwiseF64::execute(data, n, op)` uses a flattened
stride array. Input parameter order matches upstream `cpu_ref.rs`:

| Op | Stride | Layout |
|----|--------|--------|
| SCS-CN (17) | 3 | `[P, CN, Ia_ratio]` |
| Stewart (18) | 2 | `[Ky, ETa_ETc_ratio]` |
| Makkink (14) | 3 | `[Rs, T_mean, elevation]` |
| Turc (15) | 3 | `[Rs, T_mean, RH_mean]` |
| Hamon (16) | 2 | `[T_mean, daylight_hours]` — daylight pre-computed on CPU |
| BC (19) | 2 | `[T_mean, daylight_hours]` — daylight pre-computed on CPU |

### Hamon Formula Divergence

The upstream `BatchedElementwiseF64` Hamon shader implements the Hamon (1963)
ASCE formulation (`PET = 13.97 × D² × Pt`). airSpring's `eco::simple_et0::hamon_pet`
uses the Lu et al. (2005) version with different coefficients. Both are valid
Hamon implementations. GPU parity tests compare against the upstream formula.

---

## 2. Updated Documentation

| File | Change |
|------|--------|
| `gpu/mod.rs` | Module table: runoff/yield/simple_et0 → "GPU-first (BatchedElementwiseF64)" |
| `gpu/evolution_gaps.rs` | Shader mapping table updated to ops 14-19. v0.7.2 section added. |
| `EVOLUTION_READINESS.md` | Tier A counts updated. `local_dispatch` retired. coralReef noted. |
| `specs/GPU_PROMOTION_MAP.md` | A-local tier eliminated. 24 Tier A modules. coralNAK→coralReef. |
| `validate_local_gpu.rs` | Rewritten: upstream `BatchedElementwiseF64` dispatch only. |
| `validate_cross_spring_evolution.rs` | Rewritten: upstream dispatch with corrected provenance. |
| `validate_cross_spring_provenance.rs` | Provenance table updated (ops 14-19). |

---

## 3. Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** |
| `cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery` | **PASS** — 0 warnings |
| `cargo doc --no-deps` | **PASS** |
| `cargo test --lib` | **844 pass**, 0 fail |
| `#![forbid(unsafe_code)]` | Enforced |
| `LocalElementwise` / `LocalOp` references | **0** in src/ |
| Local WGSL shader files | **0** remaining |

---

## 4. For barraCuda Team

### Confirmed Working Ops

airSpring v0.7.2 validates these upstream ops via GPU parity tests:

| Op | Validated | Tolerance |
|----|-----------|-----------|
| 14 (Makkink) | 500+ elements | rel 5e-3, abs 0.01 |
| 15 (Turc) | 500+ elements | rel 5e-3, abs 0.01 |
| 16 (Hamon) | 365 elements (full year) | rel 1e-2, abs 0.02 |
| 17 (SCS-CN) | 1000+ elements | rel 1e-3 |
| 18 (Stewart) | 500+ elements | rel 1e-4 |
| 19 (BC) | 365 elements (full year) | rel 1e-2, abs 0.02 |

### Hamon Formulation Note

The upstream Hamon (Op 16) uses the Hamon (1963) ASCE formulation.
airSpring's CPU reference uses Lu et al. (2005). These produce ~12% different
results for the same inputs. This is expected — both are valid Hamon variants.
If upstream ever unifies Hamon implementations, airSpring can adopt.

---

## 5. For coralReef Team

airSpring's `local_elementwise_f64.wgsl` was previously in coralReef's test
corpus. With its retirement, the test corpus reference should be updated to
`batched_elementwise_f64.wgsl` from barraCuda.

---

## 6. Evolution Status

```
Write → Absorb → Lean: COMPLETE (all 6 ops)

Local shaders:     0 (was 2)
Local dispatch:    0 modules (was 1)
Upstream ops used: 20 (0-19)
Tier A modules:    24
Tier A-local:      0 (was 6, eliminated)
```

---

## 7. Next Steps

1. **TensorContext optimization**: `BatchedElementwiseF64` could benefit from
   pooled buffers when dispatching many small batches (available upstream).
2. **DF64 precision tier**: Consumer GPU users get DF64 (~48-bit) via
   `Fp64Strategy::Hybrid` — no airSpring code changes needed.
3. **Seasonal pipeline fusion**: Ops 0→7→1→yield fused in single dispatch
   (Tier B, pending upstream `seasonal_pipeline.wgsl`).
