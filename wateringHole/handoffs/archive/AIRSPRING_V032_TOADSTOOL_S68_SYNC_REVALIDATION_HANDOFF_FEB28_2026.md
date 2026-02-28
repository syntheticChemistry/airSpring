# AIRSPRING V032 — ToadStool S68 Sync + Revalidation

**Date**: February 28, 2026
**From**: airSpring v0.5.2
**To**: ToadStool / BarraCuda / biomeOS / metalForge teams
**Covers**: V031 → V032
**Direction**: airSpring → ToadStool (unidirectional)
**License**: AGPL-3.0-or-later

---

## Executive Summary

Full review of ToadStool S50–S68+ evolution (29 commits, 779 files, +21,891/−13,831 lines).
All airSpring imports verified against ToadStool `e96576ee` — **zero breaking changes**.
Revalidated entire suite: 584/584 lib tests, 33/33 cross-validation, 1498/1498 atlas,
48/48 validation binaries, 31/31 metalForge, 0 clippy warnings.

---

## ToadStool S50–S68 Evolution Review

### Key milestones absorbed by airSpring

| Session | Evolution | airSpring Impact |
|---------|-----------|-----------------|
| S50-S56 | Deep audit, cross-spring absorption, idiomatic Rust, +193 tests | Stability, code quality |
| S57 | +47 tests, println→tracing, 4,224+ core tests | Quality infrastructure |
| S58-S59 | DF64, Fp64Strategy, ridge, ValidationHarness, Anderson correlated | Precision probing, validation |
| S60-S63 | DF64 FMA, sovereign compiler, SPIR-V passthrough, CN f64 GPU shader | Richards PDE evolution |
| S64-S65 | Stats absorption from all Springs, smart refactoring, doc cleanup | metrics, diversity leaning |
| S66 | Cross-spring absorption, 8 SoilParams, P0 BGL fix | **All 6 metalForge modules absorbed** |
| S67 | Universal precision doctrine — "math is universal, precision is silicon" | Architecture alignment |
| S68 | **296 f32-only shaders removed** — ZERO f32-only, all f64 canonical | Pure math shaders |
| S68+ | GPU device-lost resilience, root doc cleanup, archive stale scripts | Stability |

### Precision Architecture (S67–S68)

ToadStool now has a dual-layer universal precision system:

1. **Layer 1 (op_preamble)**: Abstract math ops (`op_add`, `op_mul`, `Scalar`) resolve per `Precision` variant (F16/F32/F64/Df64). Shaders using these ops compile to any precision from one source.

2. **Layer 2 (naga IR rewrite)**: `df64_rewrite.rs` parses f64 WGSL via naga, replaces f64 infix ops with DF64 bridge functions. Automatic consumer-GPU acceleration.

| Pipeline | API | Use |
|----------|-----|-----|
| F64 native | `compile_shader_f64()` | Titan V, A100 (1:2 f64:f32) |
| DF64 hybrid | `compile_shader_df64()` | RTX 3090, 4070 (~9.9× vs native f64) |
| Universal | `compile_shader_universal()` | One source → any target |
| Op-based | `compile_op_shader(src, precision)` | Abstract ops + preamble injection |

airSpring currently uses `compile_shader_f64()` via `BatchedElementwiseF64` — this remains correct and stable. DF64 path available when ToadStool evolves `BatchedElementwiseF64` to accept a precision parameter.

### Shader Inventory

- 700 WGSL shaders total (497 f32 downcast, 182 native f64, 21 DF64)
- ZERO f32-only shaders (all f64 canonical, downcast for f32 targets)
- 122 dedicated shader tests (unit + e2e + chaos + fault)

---

## airSpring Changes (V031 → V032)

### Binary Registration Fix

- Registered `validate_gpu_math` in `Cargo.toml` (Exp 047, 46/46 checks)
- Registered `validate_ncbi_16s_coupling` in `Cargo.toml` (Exp 048, 29/29 checks)
- Both were building and passing but not formally registered

### Clippy Fix

- Fixed 2 `manual_clamp` warnings in `validate_ncbi_16s_coupling.rs`

### Documentation Updates

- `EVOLUTION_READINESS.md`: Updated to 48 experiments, 1144 Python, 57 binaries, S68 full review
- `ABSORPTION_MANIFEST.md`: Updated quality section with current metrics + ToadStool sync status

---

## Revalidation Results

| Check | Result |
|-------|--------|
| `cargo test --lib` | **584/584 PASS** |
| `cargo clippy --all-targets` | **0 warnings** (barracuda + metalForge) |
| `cross_validate` | **33/33 PASS** (Python↔Rust) |
| `validate_atlas` | **1498/1498 PASS** (Michigan Crop Water Atlas) |
| `validate_et0` | **31/31 PASS** (FAO-56 PM) |
| `validate_water_balance` | **13/13 PASS** (FAO-56 Ch 8) |
| `validate_richards` | **15/15 PASS** (VG+Picard+CN) |
| `validate_biochar` | **14/14 PASS** (Langmuir/Freundlich) |
| `validate_anderson` | **104/104 PASS** (Soil-moisture Anderson coupling) |
| `validate_ncbi_16s_coupling` | **29/29 PASS** (16S + θ + Anderson) |
| `validate_diversity` | **22/22 PASS** (Shannon, Simpson, Bray-Curtis, Chao1) |
| `validate_hargreaves` | **24/24 PASS** (Hargreaves-Samani) |
| `validate_et0_ensemble` | **17/17 PASS** (6-method ensemble) |
| metalForge `cargo test` | **31/31 PASS** |

---

## Handoff Sync Gap

ToadStool has processed airSpring through **V009** (S66 absorption). Pending:

| Range | Content |
|-------|---------|
| V010-V019 | S60-S68 sync, stats rewiring, universal precision, Priestley-Taylor, Thornthwaite, GDD, pedotransfer, atlas scale |
| V020-V025 | Cross-spring evolution, ET₀ ensemble, Richards coupling, bias correction, biomeOS Neural API, debt resolution |
| V026-V029 | CPU↔GPU parity, metalForge dispatch, Titan V live, ToadStool absorption, universal precision sync |
| V030-V031 | Anderson coupling, 25.9→26.3× CPU benchmark, GPU math portability (46/46), metalForge fixes |
| **V032** | ToadStool S68 full review, binary registration, revalidation |

**Action for ToadStool**: Process V010–V032 at next cross-spring absorption wave. Key items:
- Tier B ops 5-8 shaders (sensor_cal, hargreaves, kc_climate, dual_kc) — ready for absorption
- Anderson coupling primitives (coupling_chain, coupling_series, classify_regime)
- MC ET₀ propagation shader (`mc_et0_propagate_f64.wgsl`)
- Exp 047 GPU math portability validates all 13 GPU modules (CPU↔GPU identical)
- Exp 048 NCBI 16S coupling extends Anderson to real-world metagenomics

---

## Metrics Summary

| Metric | Value |
|--------|-------|
| Experiments | 48 |
| Python baselines | 1144 |
| Rust lib tests | 584 |
| Validation binaries | 48 (+ 3 bench + 2 utility) |
| Cross-validation | 33/33 (Python↔Rust) |
| Atlas checks | 1498/1498 |
| GPU math portability | 46/46 (13 modules) |
| metalForge dispatch | 29/29 + 21/21 routing |
| CPU speedup | 26.3× (8/8 parity) |
| Clippy warnings | 0 |
| ToadStool sync | S68+ (`e96576ee`) |
