# airSpring → ToadStool Handoff V012: S65 Primitive Rewiring + CN f64 Integration

**Date**: February 26, 2026
**From**: airSpring (Precision Agriculture — v0.4.4, 643 tests, 18 binaries)
**To**: ToadStool / BarraCuda core team + all Springs
**Supersedes**: V011 (retained in archive)
**ToadStool PIN**: `17932267` (S65 — 774 WGSL shaders)
**License**: AGPL-3.0-or-later

---

## What Changed in V012

### Rewiring to New ToadStool S60–S65 Primitives

1. **Crank-Nicolson f64 WIRED** — `pde::crank_nicolson::CrankNicolson1D` is now f64
   with `WGSL_CRANK_NICOLSON_F64` GPU shader (previously documented as f32-only).
   airSpring wired `BatchedRichards::solve_cn_diffusion()` for linearised diffusion
   cross-validation against the full nonlinear Richards solver.

2. **Optimizer inventory expanded + wired** — `brent` now wired for VG pressure
   head inversion (`inverse_van_genuchten_h()`), `norm_ppf` wired for MC ET₀
   parametric confidence intervals (`McEt0Result::parametric_ci()`).
   Additional availability documented: `bfgs`, `bisect`, `newton`, `secant`,
   `BatchedBisectionGpu`, `ResumableNelderMead`, `adaptive_penalty`.

3. **Richards PDE promoted to Tier A** — Was Tier B ("needs wiring"), now fully
   integrated: `gpu::richards::BatchedRichards` wraps upstream + CN cross-validation.

4. **evolution_gaps.rs** fully updated — 23 entries (11 Tier A, 11 Tier B, 1 Tier C).
   All stale references corrected (CN f32→f64, optimizer list expanded, brent+norm_ppf wired).

### Quality Verification

```
$ cargo fmt --check    → no diff
$ cargo clippy -- -D warnings    → 0 warnings
$ cargo doc --no-deps  → 0 warnings
$ cargo test (barracuda)
  464 lib tests PASS
  126 integration tests PASS (8 GPU-dispatch tests SKIP via catch_unwind)
$ cargo test (metalForge/forge)
   53 tests PASS
  ─────────────────────
  643 total PASS, 0 FAIL
$ cargo llvm-cov --lib  → 96.81% line, 97.58% function coverage
```

All 11 Python baselines re-run and confirmed zero drift.
33/33 cross-validation values match (tol=1e-5).

---

## P0 — Sovereign Compiler GPU Dispatch (Still Open)

`BatchedElementwiseF64` GPU dispatch still panics at `pipeline.get_bind_group_layout(0)`
after S60–S65 SPIR-V path. airSpring guards with `catch_unwind` → SKIP.

**Blocks**: ET₀ GPU (op=0), water balance GPU (op=1), MC ET₀ kernel, any
Spring using `BatchedElementwiseF64`.

---

## P1 — Remaining Open Items

| # | Item | Since | Status |
|:-:|------|:-----:|--------|
| 2 | ~~`crank_nicolson_f64` shader~~ | V007 | **RESOLVED** — now f64 + GPU shader! airSpring wired. |
| 3 | Named VG constants in `pde::richards` | V007 | Still open — 8 Carsel & Parrish soil presets |
| 4 | Preallocation in `pde::richards` | V007 | Still open — Picard buffers outside solve loop |
| 5 | Re-export `spearman_correlation` in `stats/mod.rs` | V008 | Still open — fn exists at `stats::correlation::spearman_correlation` |
| N2 | Absorb `forge::regression` (4 models, 11 tests) | V010 | Still open |
| N3 | Absorb `forge::hydrology` (4 functions, 13 tests) | V010 | Still open |
| N4 | Absorb `forge::moving_window_f64` (CPU f64, 7 tests) | V010 | Still open |
| N5 | Absorb `forge::isotherm` (Langmuir/Freundlich fits, 5 tests) | V012 | **NEW** |

---

## P2 — Future Wiring Opportunities (Expanded)

| Upstream Capability | airSpring Use Case |
|--------------------|--------------------|
| `DiversityFusionGpu` | Large-N cover crop diversity assessment |
| `BatchedMultinomialGpu` | Soil microbiome rarefaction at scale |
| `df64_transcendentals.wgsl` | VG curve double-double precision |
| Dual Kc GPU shader (op=8) | Batched multi-field crop coefficient |
| `bfgs` | Multi-adsorbate isotherm fitting (3+ parameters) |
| `BatchedBisectionGpu` | Soil water potential inversion at scale |
| `bisect` / `newton` | Additional root-finding (field capacity from texture) |
| `ResumableNelderMead` | Long-running global isotherm search checkpointing |
| `OdeSystem` + `BatchedOdeRK4` | Dynamic soil-plant-atmosphere continuum models |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001–V009 | 2026-02-25 | (see archived handoffs) |
| V010 | 2026-02-26 | ToadStool S60–S65 sync: stats rewired upstream |
| V011 | 2026-02-26 | Full cross-spring rewiring, absorption roadmap |
| **V012** | **2026-02-26** | **S65 primitive rewiring: CN f64 wired, brent+norm_ppf wired, Richards→Tier A, 11 Tier A, 643 tests** |

---

*End of V012 handoff. Direction: airSpring → ToadStool (unidirectional).
All 643 tests pass against ToadStool HEAD `17932267`. 11 Tier A wired modules.
P0 blocker: sovereign compiler GPU dispatch regression (unchanged).
3 metalForge modules + 1 isotherm module ready for absorption.
Next: pure GPU workload validation once sovereign compiler is fixed.*
