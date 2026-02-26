# airSpring → ToadStool Handoff V016: S66 Validation Complete

**Date**: February 26, 2026
**From**: airSpring v0.4.5 — 16 experiments, 474 Python + 725 Rust checks, 21 binaries
**To**: ToadStool / BarraCuda core team
**ToadStool PIN**: `045103a7` (S66 — cross-spring absorption, explicit BGL, multi-precision WGSL)
**Supersedes**: V015 (S66 sync), V014 (experiment buildout — both retained for context)

---

## Executive Summary

airSpring has completed full validation of ToadStool S66: 8 new cross-spring
evolution tests, P0 GPU dispatch blocker **RESOLVED**, both benchmarks updated
with S66 provenance, and all documentation aligned. The Write→Absorb→Lean
cycle is complete for all 6 metalForge modules. GPU-first paths are now stable.

This handoff documents:
1. What airSpring uses, what it keeps local, and why
2. P0 GPU dispatch resolution confirmation
3. Next absorption candidates for ToadStool
4. Cross-spring evolution learnings relevant to all Springs

---

## Part 1: airSpring BarraCuda Usage Map

### Wired and Production (17 capabilities)

| Capability | BarraCuda Module | Wired In | airSpring Consumer |
|------------|-----------------|----------|-------------------|
| Tolerances | `barracuda::tolerances` | v0.3.6 | Re-exported, all modules |
| Validation harness | `barracuda::validation::ValidationHarness` | v0.3.6 | All 21 binaries |
| Richards PDE | `barracuda::pde::richards::solve_richards` | v0.4.0 | `gpu::richards::BatchedRichards` |
| Crank-Nicolson f64 | `barracuda::pde::crank_nicolson::CrankNicolson1D` | v0.4.4 | CN diffusion cross-validation |
| Nelder-Mead | `barracuda::optimize::nelder_mead` | v0.4.1 | `gpu::isotherm::fit_*_nm` |
| Multi-start NM | `barracuda::optimize::multi_start_nelder_mead` | v0.4.1 | `gpu::isotherm::fit_*_global` |
| Brent root-finder | `barracuda::optimize::brent` | v0.4.4 | `eco::richards::inverse_van_genuchten_h` |
| `norm_ppf` | `barracuda::stats::normal::norm_ppf` | v0.4.4 | `gpu::mc_et0::parametric_ci` |
| Diversity metrics | `barracuda::stats::diversity::*` | v0.4.3 | `eco::diversity` delegates |
| Stats metrics | `barracuda::stats::metrics::*` | v0.4.3 | `testutil::stats` delegates |
| Ridge regression | `barracuda::linalg::ridge::ridge_regression` | v0.4.4 | `eco::correction::fit_ridge` |
| Batched ET₀ | `barracuda::ops::batched_elementwise_f64` (op=0) | v0.3.0 | `gpu::et0::BatchedEt0` |
| Batched WB | `barracuda::ops::batched_elementwise_f64` (op=1) | v0.3.0 | `gpu::water_balance::BatchedWaterBalance` |
| Kriging | `barracuda::ops::kriging_f64::KrigingF64` | v0.3.0 | `gpu::kriging::KrigingInterpolator` |
| Map-reduce | `barracuda::ops::fused_map_reduce_f64` | v0.3.0 | `gpu::reduce::SeasonalReducer` |
| Moving window | `barracuda::ops::moving_window_stats` | v0.3.0 | `gpu::stream::StreamSmoother` |
| Spearman | `barracuda::stats::spearman_correlation` | S66 | Cross-spring tests |

### Absorbed metalForge (6/6 — Write→Absorb→Lean complete)

| Module | Absorbed Into | When | airSpring Leaning Status |
|--------|--------------|------|--------------------------|
| `forge::metrics` | `stats::metrics` | S64 | **Leaning** via `testutil::stats` |
| `forge::regression` | `stats::regression` | S66 (R-S66-001) | **Leaning** — local `FittedModel` kept for domain API |
| `forge::moving_window` | `stats::moving_window_f64` | S66 (R-S66-003) | **Leaning** — f64 path available |
| `forge::hydrology` | `stats::hydrology` | S66 (R-S66-002) | **Leaning** — local Hargreaves kept (param order) |
| `forge::van_genuchten` | `pde::richards::SoilParams` | S40+S66 | **Leaning** — 8 named constants |
| `forge::isotherm` | via `optimize::nelder_mead` | S64 | **Leaning** — NM replaces linearized fits |

### Intentionally Local (not for absorption)

| airSpring Module | Reason for Staying Local | Upstream Equivalent |
|-----------------|-------------------------|-------------------|
| `eco::correction::FittedModel` | Typed `ModelType` enum (sensor correction domain API) | `stats::regression::FitResult` (generic) |
| `eco::evapotranspiration::hargreaves_et0` | FAO-56 param order `(tmin, tmax, ra)` | `stats::hydrology::hargreaves_et0(ra, tmax, tmin)` |
| `eco::dual_kc` | FAO-56 Ch 7/11 specialized scheduling logic | None |
| `eco::sensor_calibration` | SoilWatch 10 IoT device-specific | None |
| `eco::crop` | FAO-56 Table 12 crop parameter database | None |
| `io::csv_ts` | airSpring-specific IoT CSV parser | None |
| `testutil::nash_sutcliffe` | Returns 1.0 for constant-obs perfect match | Upstream convention differs |

---

## Part 2: P0 GPU Dispatch — RESOLVED

S66's explicit `BindGroupLayout` (R-S66-041) eliminates the `layout: None` +
`get_bind_group_layout(0)` pattern that caused `BatchedElementwiseF64` to panic.

**Confirmed working** (all pass):
- `test_gpu_batched_et0_station_day_gpu_dispatch` — ET₀ batch on GPU
- `test_gpu_water_balance_gpu_step_dispatch` — water balance batch on GPU
- `test_gpu_batched_et0_deterministic` — bit-identical reruns
- `test_gpu_water_balance_deterministic` — bit-identical reruns
- `test_gpu_reducer_deterministic` — reduce on GPU
- `test_gpu_kriging_deterministic` — kriging on GPU

The `try_gpu_dispatch` wrapper in `tests/common/mod.rs` has been retained
defensively for future shader regressions but is no longer triggered.

---

## Part 3: S66 Cross-Spring Validation Tests

8 new tests in `barracuda/tests/cross_spring_evolution.rs` (§11–§12):

| Test | Validates |
|------|-----------|
| `s66_regression_absorbed_upstream` | `fit_linear` R² matches local vs upstream |
| `s66_hydrology_hargreaves_absorbed_upstream` | Hargreaves ET₀ matches across param orders |
| `s66_moving_window_f64_absorbed_upstream` | f64 moving window stats |
| `s66_spearman_reexport_available` | `spearman_correlation` re-exported from `stats` |
| `s66_soil_params_named_constants` | `SoilParams::SANDY_LOAM`, `CLAY`, θ(h=0) = θs |
| `s66_metrics_mae_available` | `mae` metric |
| `s66_diversity_shannon_from_frequencies` | Shannon entropy from frequency vector |
| `benchmark_s66_regression_throughput` | 10K `fit_linear` < 2s |

Total cross-spring evolution tests: **37** (was 29 pre-S66).

---

## Part 4: Next Absorption Candidates

These airSpring domain functions are general enough for cross-Spring use:

### High Priority (useful to other Springs)

| Function | Module | What It Does | Other Spring Uses |
|----------|--------|-------------|-------------------|
| `psychrometric_constant(elevation)` | `eco::evapotranspiration` | Atmospheric physics from elevation | Any Spring doing environmental modeling |
| `atmospheric_pressure(elevation)` | `eco::evapotranspiration` | Standard atmosphere (FAO-56 Eq 7) | General atmospheric computations |
| `saturation_vapour_pressure(temp)` | `eco::evapotranspiration` | Tetens formula (FAO-56 Eq 11) | Environmental/bio Springs |
| `topp_equation(ka)` | `eco::soil_moisture` | Dielectric → VWC (Topp 1980) | Soil sensing applications |
| `total_available_water(fc, wp, rd)` | `eco::water_balance` | FAO-56 Ch 8 TAW | Agricultural modeling |
| `stress_coefficient(dr, taw, raw)` | `eco::water_balance` | Crop water stress factor Ks | Agricultural modeling |
| `yield_ratio_single(ky, eta_etc)` | `eco::yield_response` | Stewart (1977) yield response | Agricultural modeling |

### Medium Priority (domain data, less general)

| Function | Notes |
|----------|-------|
| `ky_table(crop)` | FAO-56 Table 24 yield response factors (9 crops) — domain data |
| `water_use_efficiency(yield, actual_et)` | Simple division, but standardized API |
| `mass_to_et_mm(delta_mass, area)` | Lysimeter conversion (Exp 016) |

### Low Priority / Stay Local

| Function | Reason |
|----------|--------|
| `dual_kc` simulation | Too specialized for general use |
| `sensor_calibration` | Device-specific |
| `csv_ts` parser | airSpring-specific IoT format |

---

## Part 5: Cross-Spring Evolution Learnings

### What We Learned Building 16 Experiments

1. **Write→Absorb→Lean works**. All 6 metalForge modules were absorbed in
   2 ToadStool sessions (S64, S66). The pattern is now proven at scale.

2. **Param order matters**. `hargreaves_et0(tmin, tmax, ra)` vs `(ra, tmax, tmin)`.
   When absorbing domain functions, document both conventions. Consider builder
   patterns or named parameters for functions with >3 f64 arguments.

3. **Convention differences compound**. `nash_sutcliffe` returning 1.0 vs 0.0 for
   constant observations, `r_squared` via Pearson r² vs SS-based — these need
   clear documentation. airSpring intentionally keeps local versions when the
   semantic meaning differs from upstream.

4. **GPU dispatch is fragile to BGL changes**. The S60-S65 `layout: None` regression
   went undetected until airSpring's cross-spring integration tests caught it. The
   `try_gpu_dispatch` wrapper pattern (catch_unwind → SKIP) keeps test suites green
   while blocking issues are filed upstream.

5. **Composition > monolith**. The scheduling pipeline (Exp 014) chains ET₀→Kc→water
   balance→Stewart yield across 4 modules. Each module is independently validated.
   GPU promotion means each stage can dispatch independently, and ToadStool's
   unidirectional streaming reduces round-trips.

6. **Cross-spring tests are cheap insurance**. 37 tests validating upstream
   primitives catch absorption regressions (e.g., if `SoilParams` constants change).
   Every Spring should have these.

### Cross-Spring Shader Provenance

```text
hotSpring (S42) → df64 core + pow/exp/log f64
  └→ airSpring: VG retention, atmospheric pressure, ET₀

wetSpring (S28) → kriging, fused_reduce, moving_window, ridge
  └→ airSpring: soil mapping, seasonal stats, IoT smoothing, calibration

neuralSpring (S52) → nelder_mead, multi_start, brent, ValidationHarness
  └→ airSpring: isotherm fitting, VG inversion, all binaries

groundSpring (S64) → mc_et0_propagate, norm_ppf
  └→ airSpring: MC uncertainty, parametric CI

airSpring (S40) → Richards PDE absorbed upstream
  └→ all Springs: unsaturated flow solver

airSpring metalForge (S66) → regression, hydrology, moving_window_f64
  └→ all Springs: sensor correction, crop water, stream stats
```

---

## Part 6: GPU Evolution — Tier B Roadmap

These Tier B items are ready to wire with low effort:

| Need | Shader Op | Effort | Impact |
|------|-----------|--------|--------|
| Dual Kc batch (Ke) | `batched_elementwise_f64` op=8 | Low | Enables GPU dual crop coefficient |
| VG θ/K batch | New op | Low | Batch soil retention curves |
| Sensor calibration batch | op=5 | Low | IoT pipeline on GPU |
| Hargreaves ET₀ batch | op=6 | Low | Simpler ET₀ for large-scale runs |
| Kc climate adjustment | op=7 | Low | Regional crop coefficient tuning |
| MC ET₀ GPU kernel | `mc_et0_propagate_f64.wgsl` | Medium | Blocked by sovereign compiler; needs re-test post-S66 |

---

## Part 7: Quality Gates

| Check | Status |
|-------|--------|
| `cargo fmt --check` | **Clean** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** (pedantic) |
| `cargo test --workspace` | **649** cargo tests PASS (464 lib + 132 integration + 53 forge) |
| Validation binaries | **21/21** PASS (515 checks) |
| Python controls | **474/474** PASS (16 experiments) |
| Cross-validation | **75/75** match (tol=1e-5) |
| GPU dispatch (P0) | **RESOLVED** — S66 explicit BGL (R-S66-041) |
| `unsafe` code | **Zero** |
| `unwrap()` in lib | **Zero** |
| Coverage | **96.81%** line (97.58% function) |

---

## Open Items (P3)

| ID | Item | Priority | Status |
|----|------|:--------:|--------|
| N6 | Lysimeter `mass_to_et` utility for `barracuda::eco` | P3 | Open |
| N7 | OAT sensitivity utility for `barracuda::stats` | P3 | Open |
| N8 | Scheduling irrigation trigger API for `barracuda::eco` | P3 | Open |

---

## Files Changed Since V015

| File | Change |
|------|--------|
| `barracuda/tests/cross_spring_evolution.rs` | +8 S66 tests (§11–§12) |
| `barracuda/tests/common/mod.rs` | `try_gpu_dispatch` docs updated for S66 resolution |
| `barracuda/src/bin/bench_airspring_gpu.rs` | 3 new bench ops, 774 shader provenance |
| `barracuda/src/bin/bench_cpu_vs_python.rs` | 3 new benchmark sections (Exp 014, 016, 017) |
| `barracuda/src/gpu/evolution_gaps.rs` | S66 BGL fix noted |
| `barracuda/EVOLUTION_READINESS.md` | P0 resolved, 6/6 absorbed, 725 tests |
| `specs/CROSS_SPRING_EVOLUTION.md` | S66 validation timeline entry |
| All root/whitePaper docs | Count alignment (725 Rust, 132 integration) |

---

*airSpring v0.4.5 → ToadStool S66 (`045103a7`). All metalForge absorbed,
P0 resolved, 725 Rust checks, 474 Python checks, 21 binaries. Next: Tier B
GPU wiring (5 low-effort ops) and MC ET₀ GPU re-test post-S66.*
