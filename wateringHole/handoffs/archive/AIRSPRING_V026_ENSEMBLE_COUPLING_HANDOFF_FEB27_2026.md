# airSpring V026 — ET₀ Ensemble, Pedotransfer-Richards Coupling & Bias Correction

**Date**: 2026-02-27
**From**: airSpring (v0.4.15)
**To**: ToadStool / BarraCuda, biomeOS, wetSpring
**License**: AGPL-3.0-or-later
**Covers**: Experiments 037-039
**Supersedes**: V025 (biomeOS Neural API)

---

## Executive Summary

- **Exp 037** — ET₀ Ensemble Consensus: combines 6 daily ET₀ methods into a
  data-adaptive weighted consensus (Python 9/9, Rust 17/17 PASS)
- **Exp 038** — Pedotransfer → Richards Coupling: Saxton-Rawls soil texture →
  Van Genuchten parameters → implicit Richards 1D PDE (Python 29/29, Rust 32/32 PASS)
- **Exp 039** — ET₀ Bias Correction: quantifies systematic bias of simplified
  methods relative to PM and computes linear correction factors across 4 climate
  scenarios (Python 24/24, Rust 24/24 PASS)
- **Total**: 39 experiments, 946 Python + 515 Rust lib + 1024 validation checks
- **Cross-primal issues**: 10 issues documented in shared
  `wateringHole/SPRING_EVOLUTION_ISSUES.md` for biomeOS/ToadStool/Springs

---

## Part 1: New barracuda Capabilities

### `et0_ensemble` (evapotranspiration.rs)

New public API for multi-method ET₀ consensus:

```rust
pub struct EnsembleInput { /* all fields Optional except elevation/latitude/doy */ }
pub struct EnsembleResult {
    pub consensus: f64,  // equal-weight mean (mm/day)
    pub spread: f64,     // max - min method range
    pub n_methods: u8,   // count of contributing methods
    pub pm: f64, pub pt: f64, pub hargreaves: f64,
    pub makkink: f64, pub turc: f64, pub hamon: f64,
}
pub fn et0_ensemble(input: &EnsembleInput) -> EnsembleResult;
```

**GPU promotion candidate**: The ensemble is 6 independent element-wise
operations followed by a reduction. Maps to `BatchedElementwise` + `FusedMapReduce`.

### Pedotransfer-Richards Pipeline

Demonstrated pipeline: `saxton_rawls()` → `VanGenuchtenParams` → `solve_richards_1d()`.
No new API surface — existing functions compose cleanly. Validates that the
implicit Richards solver produces physically correct wetting/drainage dynamics
from pedotransfer-derived parameters.

### Bias Correction Factors

Correction factors from 4-climate calibration:

| Method | Factor | Avg corrected error |
|--------|--------|---------------------|
| Hargreaves | 0.430 | 16.7% |
| Makkink | 1.489 | 17.3% |
| Turc | 1.077 | 8.9% |
| Hamon | 12.919 | 18.2% |

Turc has the lowest bias (closest to PM); Hamon has the largest systematic
underestimation.

---

## Part 2: Cross-Primal Issues

Created `wateringHole/SPRING_EVOLUTION_ISSUES.md` (10 issues) in the shared
ecosystem wateringHole. Key issues for each team:

### biomeOS action items
- **ISSUE-001**: Add `[domains.ecology]` to capability registry
- **ISSUE-003**: Standardize `capability.call` parameter format
- **ISSUE-004**: Test deployment graphs end-to-end
- **ISSUE-007**: Document "Spring as Provider" registration pattern
- **ISSUE-010**: Define cross-primal time series exchange format

### ToadStool action items
- **ISSUE-002**: Ship sync Neural API client (or feature-flag async)
- **ISSUE-006**: Absorb NPU substrate model from multiple Springs
- **ISSUE-008**: GPU promotion for 4 ET₀ methods (Tier B → A)
- **ISSUE-009**: Standardize benchmark JSON schema

### All Springs
- **ISSUE-005**: Safe UID discovery via `/proc/self/status`

---

## Part 3: Quality Metrics

| Metric | Value |
|--------|-------|
| Total experiments | 39 |
| Python checks | 946/946 PASS |
| Rust lib tests | 515 PASS |
| Rust validation checks | 1024 PASS |
| Atlas checks | 1393 PASS |
| Clippy errors | 0 |
| Clippy warnings | 0 (from new code) |
| Unsafe blocks | 0 |
| External C deps | 0 |

---

## Recommended Next Steps

1. **ToadStool**: Promote `et0_ensemble` to GPU via `BatchedElementwise` + `FusedMapReduce`
2. **biomeOS**: Review ISSUE-001 (ecology domain) and ISSUE-007 (Spring provider pattern)
3. **airSpring**: Live test on tower node once biomeOS adds ecology domain
4. **wetSpring**: Review ISSUE-010 for cross-primal θ(t) → diversity pipeline format
