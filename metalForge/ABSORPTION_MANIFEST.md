# Absorption Manifest — airSpring → barracuda

**Date**: February 28, 2026 (updated v0.5.3 — V034 active handoff, forge evolved to mixed hardware dispatch, 18 workloads, 29/29 cross-system)
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
equivalents in barracuda. The crate is retained as a fossil record with provenance
and as the staging area for cross-system routing validation (18 eco workloads).

The forge crate validates dispatch routing (GPU > NPU > CPU) and cross-system
hardware probing. 31 tests covering substrate discovery + capability routing + 4 Tier B local workloads.

---

## Tier B Local Workloads (v0.5.2)

4 new workloads added for Tier B GPU orchestrators pending ToadStool absorption:

| Workload | Op | Description | Substrate |
|----------|-----|-------------|-----------|
| `SensorCalibration` | 5 | SoilWatch 10 VWC calibration | CPU (pending GPU) |
| `HargreavesBatch` | 6 | Hargreaves-Samani ET₀ | CPU (pending GPU) |
| `KcClimateAdjust` | 7 | FAO-56 Eq. 62 climate correction | CPU (pending GPU) |
| `DualKcBatch` | 8 | Ke evaporation layer | CPU (pending GPU) |

These workloads auto-activate GPU dispatch upon ToadStool absorption of ops 5-8.

---

## Quality

```
cargo fmt   — clean
cargo clippy --all-targets — zero warnings (pedantic)
cargo test  — 31/31 pass (forge), 584/584 pass (barracuda lib)
validate_*  — 48/48 PASS (barracuda) + 3 bench binaries (30/30 cross-spring benchmarks)
metalForge cross-system routing — 29/29 PASS (18 workloads × dispatch checks)
ToadStool sync: S68+ (e96576ee) — universal precision, 700 WGSL, 6-Spring provenance
```
