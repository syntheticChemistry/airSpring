# metalForge — airSpring Upstream Contributions

**Date**: February 26, 2026
**Crate**: `airspring-forge` v0.2.0 (vestigial — all modules absorbed upstream)
**License**: AGPL-3.0-or-later

---

## Philosophy

metalForge is where airSpring staged domain primitives for upstream
absorption into `barracuda` (ToadStool). Following hotSpring's pattern:

```
Write locally → Validate against benchmarks → Hand off to ToadStool → Absorb → Lean on upstream
```

**Status: COMPLETE.** All 6 metalForge modules have been absorbed upstream.
The forge crate remains as a fossil record (53 tests) but is no longer
in the active dependency graph. airSpring now leans on upstream primitives.

## What's Here

### `forge/` — Rust crate (`airspring-forge`)

Six modules, all absorbed upstream. Pure Rust, zero dependencies, 53/53 tests pass.

| Module | Functions | Absorbed Into | When |
|--------|-----------|--------------|------|
| `metrics` | `rmse`, `mbe`, `nash_sutcliffe`, `index_of_agreement`, `coefficient_of_determination` | `barracuda::stats::metrics` | **S64** |
| `regression` | `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `fit_all` + `FitResult::predict()` | `barracuda::stats::regression` | **S66** (R-S66-001) |
| `moving_window_f64` | `moving_window_stats` (mean, variance, min, max) | `barracuda::stats::moving_window_f64` | **S66** (R-S66-003) |
| `hydrology` | `hargreaves_et0`, `hargreaves_et0_batch`, `crop_coefficient`, `soil_water_balance` | `barracuda::stats::hydrology` | **S66** (R-S66-002) |
| `van_genuchten` | VG retention, conductivity, capacity | `barracuda::pde::richards::SoilParams` | **S40+S66** |
| `isotherm` | Langmuir/Freundlich linearized fits | `barracuda::eco::isotherm` (via NM) | **S64** |

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
| **Tests** | Hardware probing, bridge seam | 53 unit tests, numerical correctness |

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
cargo test  — 53/53 pass (11 metrics + 12 regression + 7 moving_window + 13 hydrology + 5 van_genuchten + 5 isotherm)
```
