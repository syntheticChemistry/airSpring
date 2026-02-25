# metalForge — airSpring Upstream Contributions

**Date**: February 25, 2026
**Crate**: `airspring-forge` v0.2.0
**License**: AGPL-3.0-or-later

---

## Philosophy

metalForge is where airSpring stages domain primitives for upstream
absorption into `barracuda` (ToadStool).  Following hotSpring's pattern:

```
Write locally → Validate against benchmarks → Hand off to ToadStool → Absorb → Lean on upstream
```

Springs don't import each other.  They contribute to the shared compute
infrastructure through metalForge, and the ToadStool team absorbs what's
general enough for cross-domain use.

## What's Here

### `forge/` — Rust crate (`airspring-forge`)

Four absorption-ready modules extracted from airSpring's validated
pipeline.  Pure Rust, zero dependencies, 40/40 tests pass.

| Module | Functions | Upstream target | Status |
|--------|-----------|----------------|--------|
| `metrics` | `rmse`, `mbe`, `nash_sutcliffe`, `index_of_agreement`, `coefficient_of_determination` | `barracuda::stats::metrics` | Ready |
| `regression` | `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `fit_all` + `FitResult::predict()` | `barracuda::stats::regression` | Ready |
| `moving_window_f64` | `moving_window_stats` (mean, variance, min, max) | `barracuda::ops::moving_window_stats_f64` | Ready |
| `hydrology` | `hargreaves_et0`, `hargreaves_et0_batch`, `crop_coefficient`, `soil_water_balance` | `barracuda::ops::hydrology` | Ready |

### Provenance

All implementations are validated against published benchmarks:

- **Metrics**: Dong et al. (2020) soil sensor calibration (36/36), FAO-56
  real data pipeline (918 station-days, R²=0.967), 65/65 Python-Rust
  cross-validation
- **Regression**: Dong et al. (2020) four-model correction suite, validated
  against scipy `curve_fit` outputs.  `FitResult::predict()` follows the
  `RidgeResult::predict()` pattern from `barracuda::linalg::ridge`
- **Moving window f64**: CPU f64 complement to upstream f32 GPU path
  (wetSpring S28+ `moving_window.wgsl`)
- **Hydrology**: Hargreaves & Samani (1985), FAO-56 (Allen et al. 1998),
  918 station-days, cross-validated with Python ETo library

## Relationship to hotSpring's metalForge

| | hotSpring metalForge | airSpring metalForge |
|--|----------------------|----------------------|
| **Focus** | Hardware characterization, substrate discovery, capability dispatch | Statistical metrics, regression, hydrology, signal processing |
| **Upstream target** | `barracuda::device::unified` | `barracuda::stats::*`, `barracuda::ops::*` |
| **Crate** | `hotspring-forge` | `airspring-forge` |
| **Dependencies** | barracuda, wgpu, tokio | None (pure Rust) |
| **Modules** | substrate, probe, inventory, dispatch, bridge | metrics, regression, moving_window_f64, hydrology |
| **Tests** | Hardware probing, bridge seam | 40 unit tests, numerical correctness |

## Cross-Spring Absorption Candidates

These airSpring patterns may benefit other springs:

| Pattern | Used by | Potential |
|---------|---------|-----------|
| RMSE / MBE / NSE / IA | airSpring, groundSpring | Universal validation metrics |
| Analytical curve fitting + `predict()` | airSpring (soil calibration) | Any domain with empirical fits |
| CPU f64 moving window | airSpring (IoT sensor streams) | Complement to GPU f32 path |
| Hargreaves ET₀ + Kc interpolation | airSpring (precision agriculture) | Environmental modeling |
| Soil water balance | airSpring (irrigation scheduling) | Hydrology domains |
| `ValidationRunner` pattern | airSpring | Shared validation harness |
| `len_f64()` utility | All springs | Already in barracuda candidates |

## Quality

```
cargo fmt   — clean
cargo clippy --all-targets — zero warnings (pedantic)
cargo test  — 40/40 pass (9 metrics + 11 regression + 7 moving_window + 13 hydrology)
```
