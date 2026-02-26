# Absorption Manifest — airSpring → barracuda

**Date**: February 26, 2026 (updated v0.4.5 — ToadStool S66 absorption complete)
**Source**: `metalForge/forge/` (airspring-forge v0.2.0)
**Target**: `barracuda` (ToadStool crate)
**Absorption Status**: 6/6 absorbed upstream — ALL modules absorbed as of S66

---

## Absorption Process (Write → Absorb → Lean)

Following the pattern established by hotSpring:

1. **Write**: Implement primitives in pure Rust inside `metalForge/forge/`
2. **Validate**: Test against published benchmarks and Python baselines
3. **Hand off**: Document in `ABSORPTION_MANIFEST.md` with signatures, tests, provenance
4. **Absorb**: ToadStool copies into `barracuda::stats::*` or `barracuda::ops::*`
5. **Lean**: airSpring rewires to `use barracuda::*`, deletes local code
6. **Verify**: Run `validate_all` to confirm suites still pass

---

## Absorbed (All Complete)

### 1. Statistical Agreement Metrics → `barracuda::stats::metrics` (**ABSORBED** — S64 + S66)

| Function | Upstream Location | R-ID |
|----------|-------------------|------|
| `rmse` | `barracuda::stats::metrics::rmse` | S64 |
| `mbe` | `barracuda::stats::metrics::mbe` | S64 |
| `mae` | `barracuda::stats::metrics::mae` | R-S66-036 |
| `nash_sutcliffe` | `barracuda::stats::metrics::nash_sutcliffe` | S64 |
| `index_of_agreement` | `barracuda::stats::metrics::index_of_agreement` | S64 |
| `r_squared` | `barracuda::stats::metrics::r_squared` | S64 |
| `hit_rate` | `barracuda::stats::metrics::hit_rate` | S64 |
| `mean` | `barracuda::stats::metrics::mean` | S64 |
| `percentile` | `barracuda::stats::metrics::percentile` | S64 |
| `dot` | `barracuda::stats::metrics::dot` | S64 |
| `l2_norm` | `barracuda::stats::metrics::l2_norm` | S64 |
| `hill` | `barracuda::stats::metrics::hill` | R-S66-038 |
| `monod` | `barracuda::stats::metrics::monod` | R-S66-038 |

**airSpring leaning**: `testutil::stats::rmse` and `mbe` delegate to upstream.
`nash_sutcliffe` and `index_of_agreement` keep local edge-case convention (1.0 for
constant-observation perfect match vs upstream 0.0 division guard).

### 2. Analytical Regression → `barracuda::stats::regression` (**ABSORBED** — R-S66-001)

| Function | Upstream Location |
|----------|-------------------|
| `fit_linear` | `barracuda::stats::regression::fit_linear` |
| `fit_quadratic` | `barracuda::stats::regression::fit_quadratic` |
| `fit_exponential` | `barracuda::stats::regression::fit_exponential` |
| `fit_logarithmic` | `barracuda::stats::regression::fit_logarithmic` |
| `fit_all` | `barracuda::stats::regression::fit_all` |
| `FitResult` | `barracuda::stats::regression::FitResult` |

**airSpring leaning**: `eco::correction` keeps domain-specific `FittedModel` with
typed `ModelType` enum and sensor correction evaluation functions. Upstream
`FitResult` uses `model: &'static str` — different return type serves a different use case.
airSpring documented provenance in `eco::correction` module doc.

### 3. CPU f64 Moving Window Statistics → `barracuda::stats::moving_window_f64` (**ABSORBED** — R-S66-003)

| Function | Upstream Location |
|----------|-------------------|
| `moving_window_stats_f64` | `barracuda::stats::moving_window_f64::moving_window_stats_f64` |
| `MovingWindowResultF64` | `barracuda::stats::moving_window_f64::MovingWindowResultF64` |

**airSpring leaning**: `gpu::stream::StreamSmoother` already uses upstream
`ops::moving_window_stats` (GPU path). CPU f64 path now available upstream too.

### 4. Hydrology Primitives → `barracuda::stats::hydrology` (**ABSORBED** — R-S66-002)

| Function | Upstream Location |
|----------|-------------------|
| `hargreaves_et0` | `barracuda::stats::hydrology::hargreaves_et0` |
| `hargreaves_et0_batch` | `barracuda::stats::hydrology::hargreaves_et0_batch` |
| `crop_coefficient` | `barracuda::stats::hydrology::crop_coefficient` |
| `soil_water_balance` | `barracuda::stats::hydrology::soil_water_balance` |

**airSpring leaning**: `eco::evapotranspiration::hargreaves_et0` keeps local
FAO-56-validated implementation with `(tmin, tmax, ra)` parameter convention
(upstream uses `(ra, t_max, t_min)`). Documented provenance in module doc.

### 5. Van Genuchten Soil Hydraulics → `barracuda::pde::richards` (**ABSORBED** — pre-S64)

**Status**: Fully absorbed. airSpring uses `barracuda::pde::richards::SoilParams`.
S66 added 8 named soil type constants (R-S66-006): `SANDY_LOAM`, `SILT_LOAM`,
`CLAY_LOAM`, `SAND`, `CLAY`, `LOAM`, `SILTY_CLAY_LOAM`, `LOAMY_SAND`.

### 6. Isotherm Models → `barracuda::optimize` (**WIRED** — v0.4.1)

**Status**: Fitting uses `barracuda::optimize::nelder_mead` and `multi_start_nelder_mead`.
Linearized initial guess functions remain local in `eco::isotherm` (domain-specific).

---

## Additional S66 Absorptions (Not from metalForge)

| Feature | Upstream Location | R-ID |
|---------|-------------------|------|
| `spearman_correlation` re-export | `barracuda::stats::correlation::spearman_correlation` | R-S66-005 |
| `rawr_mean` (RAWR bootstrap) | `barracuda::stats::bootstrap::rawr_mean` | R-S66-004 |
| `shannon_from_frequencies` | `barracuda::stats::diversity::shannon_from_frequencies` | R-S66-037 |
| 8 named `SoilParams` constants | `barracuda::pde::richards::{SAND, LOAM, ...}` | R-S66-006 |

---

## metalForge Status

The `metalForge/forge/` crate is now **vestigial** — all 6 modules have upstream
equivalents in barracuda. The crate is retained as a fossil record with provenance:
53 tests validate the original implementations that were absorbed upstream.

The forge crate is NOT in airSpring's dependency graph (no `use airspring_forge::`).
It can be archived or removed at any time without affecting the build.

---

## Quality

```
cargo fmt   — clean
cargo clippy --all-targets — zero warnings (pedantic)
cargo test  — 53/53 pass (forge), 643/643 pass (barracuda workspace)
```
