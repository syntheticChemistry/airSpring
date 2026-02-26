# airSpring → ToadStool Handoff V015: S66 Sync — All metalForge Absorbed

**Date**: February 26, 2026
**From**: airSpring v0.4.5 — 16 experiments, 474 Python + 719 Rust checks, 21 binaries
**To**: ToadStool / BarraCuda core team
**ToadStool PIN**: `045103a7` (S66 — cross-spring absorption wave)
**Supersedes**: V013 (archived — all absorption items resolved), V014 (retained for experiment GPU roadmap)

---

## Summary

ToadStool S66 absorbed **all four pending metalForge modules** (regression, hydrology,
moving_window_f64, metrics expansion) plus re-exported `spearman_correlation`,
added 8 named `SoilParams` constants, `mae`/`hill`/`monod`, `shannon_from_frequencies`,
and `rawr_mean`. airSpring has pulled S66, validated against it (643 cargo tests +
17 validation binaries = 473 binary checks, all PASS), and updated all provenance docs.

**metalForge is now fully absorbed.** The forge crate remains as a fossil record
(53 tests) but is not in the dependency graph.

---

## What S66 Resolved (V013/V014 Items)

| V013/V014 Item | S66 Resolution | R-ID |
|----------------|----------------|------|
| N2: Named VG soil type constants | 8 `SoilParams` constants (Carsel & Parrish 1988) | R-S66-006 |
| N3: `spearman_correlation` re-export | Now re-exported from `stats::correlation` | R-S66-005 |
| N4: `forge::regression` → `barracuda::linalg` | Absorbed into `barracuda::stats::regression` | R-S66-001 |
| N5: `forge::hydrology` → `barracuda::eco` | Absorbed into `barracuda::stats::hydrology` | R-S66-002 |
| (implicit): `forge::moving_window_f64` | Absorbed into `barracuda::stats::moving_window_f64` | R-S66-003 |
| (implicit): `metrics::mae` | Added to `barracuda::stats::metrics` | R-S66-036 |

---

## airSpring Leaning Status (Post-S66)

### Already Delegating to Upstream

| airSpring Module | Upstream Primitive | Since |
|-----------------|-------------------|:-----:|
| `testutil::stats::rmse` | `barracuda::stats::rmse` | S64 |
| `testutil::stats::mbe` | `barracuda::stats::mbe` | S64 |
| `testutil::stats::pearson_r` | `barracuda::stats::pearson_correlation` | S64 |
| `testutil::stats::spearman_r` | `barracuda::stats::correlation::spearman_correlation` | S66 |
| `testutil::bootstrap` | `barracuda::stats::bootstrap::bootstrap_ci` | S64 |
| `eco::diversity` | `barracuda::stats::diversity::*` | S64 |
| `gpu::mc_et0::parametric_ci` | `barracuda::stats::normal::norm_ppf` | v0.4.4 |
| `eco::richards::inverse_van_genuchten_h` | `barracuda::optimize::brent` | v0.4.4 |
| `gpu::richards::BatchedRichards` | `barracuda::pde::richards::solve_richards` | v0.4.0 |
| `gpu::isotherm::fit_*_nm` | `barracuda::optimize::nelder_mead` | v0.4.1 |
| `gpu::et0::BatchedEt0` | `barracuda::ops::batched_elementwise_f64` | v0.3.0 |
| `gpu::kriging` | `barracuda::ops::kriging_f64` | v0.3.6 |
| `gpu::reduce` | `barracuda::ops::fused_map_reduce_f64` | v0.3.6 |
| `gpu::stream` | `barracuda::ops::moving_window_stats` | v0.3.6 |
| `eco::correction::fit_ridge` | `barracuda::linalg::ridge::ridge_regression` | v0.3.10 |
| `validation::ValidationHarness` | `barracuda::validation::ValidationHarness` | v0.3.6 |
| `validate_lysimeter::rmse` | `barracuda::stats::rmse` | **V015** (was local) |

### Intentionally Local (Domain-Specific)

| Module | Reason | Upstream Equivalent |
|--------|--------|---------------------|
| `eco::correction::fit_*` | Typed `ModelType` enum + `FittedModel` for sensor correction | `stats::regression::FitResult` (generic) |
| `eco::evapotranspiration::hargreaves_et0` | FAO-56-validated `(tmin, tmax, ra)` param order | `stats::hydrology::hargreaves_et0` (`(ra, tmax, tmin)`) |
| `testutil::stats::nash_sutcliffe` | Returns 1.0 for constant-obs perfect match | `stats::metrics::nash_sutcliffe` (returns 0.0) |
| `testutil::stats::index_of_agreement` | Returns 1.0 for constant-obs perfect match | `stats::metrics::index_of_agreement` (returns 0.0) |
| `testutil::stats::r_squared` | Pearson r² (r × r) | `stats::metrics::r_squared` (SS-based = NSE) |

---

## Remaining Open Items

| ID | Item | Priority | Status |
|----|------|:--------:|--------|
| N1 | `BatchedElementwiseF64` GPU dispatch panic (P0) | P0 | **Blocked** (unchanged) |
| N6 | Lysimeter `mass_to_et` utility for `barracuda::eco` | P3 | Open |
| N7 | OAT sensitivity utility for `barracuda::stats` | P3 | Open |
| N8 | Scheduling irrigation trigger API for `barracuda::eco` | P3 | Open |

Items N2-N5 from V013 are **closed** (resolved by S66).

---

## Quality Gates (v0.4.5, ToadStool S66)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | **Clean** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** |
| `cargo test --workspace` | **643** cargo tests PASS |
| Validation binaries | **17/17** PASS (473 checks) |
| Python controls | **474/474** PASS (16 experiments) |
| Cross-validation | **75/75** match (tol=1e-5) |
| Coverage | **96.81%** lines (lib) |
| `unsafe` blocks | **0** |
| `unwrap()` in lib | **0** |
| metalForge status | **Fully absorbed** (6/6 upstream) |

---

*airSpring v0.4.5 — ToadStool S66 synced. 16 experiments, 474/474 Python, 719 Rust
checks, 21 binaries. All metalForge absorbed upstream. Pure Rust + BarraCuda.
AGPL-3.0-or-later.*
