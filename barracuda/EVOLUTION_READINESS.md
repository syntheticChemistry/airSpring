# airSpring BarraCuda — Evolution Readiness

**Last Updated**: February 26, 2026 (v0.4.8 — 491 tests, 97.45% coverage)
**ToadStool PIN**: S68 (universal f64, ValidationHarness tracing, LazyLock shader constants)
**Handoff**: V022 (Thornthwaite + GDD + pedotransfer — 27 binaries, 570 validation + 1393 atlas checks)
**License**: AGPL-3.0-or-later

---

## Write → Absorb → Lean Status

airSpring follows the same pattern as hotSpring and wetSpring: implement locally,
validate against papers, hand off to ToadStool/BarraCuda, lean on upstream.

### Already Absorbed (Lean)

| Module | Absorbed Into | When | Status |
|--------|--------------|------|--------|
| `ValidationRunner` | `barracuda::validation::ValidationHarness` | S59 | **Leaning** — all 27 binaries use upstream |
| `van_genuchten` | `barracuda::pde::richards::SoilParams` | S40 | **Leaning** — `gpu::richards` bridges to upstream |
| `isotherm NM` | `barracuda::optimize::nelder_mead` | S62 | **Leaning** — `gpu::isotherm` bridges to upstream |

### Absorbed Upstream (6/6 metalForge modules — Write→Absorb→Lean complete)

| Module | Absorbed Into | When | Status |
|--------|--------------|------|--------|
| `forge::metrics` | `barracuda::stats::metrics` | S64 | **LEANING** — `testutil::stats` delegates |
| `forge::regression` | `barracuda::stats::regression` | S66 (R-S66-001) | **LEANING** — `eco::correction` keeps domain `FittedModel` |
| `forge::moving_window` | `barracuda::stats::moving_window_f64` | S66 (R-S66-003) | **LEANING** — `gpu::stream` f64 path available |
| `forge::hydrology` | `barracuda::stats::hydrology` | S66 (R-S66-002) | **LEANING** — `eco::evapotranspiration` keeps FAO-56 param order |
| `forge::isotherm` | `barracuda::eco::isotherm` (was local) | S64 | **LEANING** — `gpu::isotherm` delegates via NM |
| `forge::van_genuchten` | `barracuda::pde::richards::SoilParams` | S40+S66 | **LEANING** — 8 named constants (R-S66-006) |

See `metalForge/ABSORPTION_MANIFEST.md` for full signatures and validation details.

### Stays Local (domain-specific)

| Module | Reason |
|--------|--------|
| `eco::dual_kc` | FAO-56 Ch 7/11 domain logic — too specialized for barracuda |
| `eco::sensor_calibration` | SoilWatch 10 specific — domain consumer |
| `eco::crop` | FAO-56 Table 12 crop database, GDD, kc_from_gdd — domain data |
| `eco::evapotranspiration` | Thornthwaite monthly ET₀ — domain consumer |
| `eco::soil_moisture` | Saxton-Rawls pedotransfer (θs/θr/Ks from texture) — domain consumer |
| `io::csv_ts` | airSpring-specific IoT CSV parser |
| `testutil::generators` | Synthetic IoT data for airSpring tests |

---

## GPU Evolution Tiers

### Tier A: Integrated (11 modules — GPU primitive wired, validated)

| airSpring Module | BarraCuda Primitive | Status |
|-----------------|--------------------|----|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **INTEGRATED** |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **WIRED** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **WIRED** |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | **WIRED** (+ CN f64 cross-val) |
| `gpu::isotherm::fit_*_nm/global` | `optimize::nelder_mead` + `multi_start` | **WIRED** |
| `eco::diversity` | `stats::diversity` (Shannon, Simpson, Bray-Curtis) | **LEANING** (S64) |
| `gpu::mc_et0::parametric_ci` | `stats::normal::norm_ppf` | **WIRED** — hotSpring precision lineage |
| `eco::richards::inverse_van_genuchten_h` | `optimize::brent` | **WIRED** — neuralSpring optimizer lineage |

### Tier B: Upstream Exists, Needs Domain Wiring (11 items, 4 wired)

| Need | Closest Primitive | Effort |
|------|-------------------|:------:|
| Dual Kc batch (Ke) | `batched_elementwise_f64` (op=8) | Low |
| VG θ/K batch | `batched_elementwise_f64` (new op) | Low |
| Batch Nelder-Mead GPU | `NelderMeadGpu` | Medium |
| Sensor batch calibration | `batched_elementwise_f64` (op=5) | Low |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | Low |
| Kc climate adjustment | `batched_elementwise_f64` (op=7) | Low |
| Tridiagonal solve batch | `linalg::tridiagonal_solve_f64` | Low |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Low |
| m/z tolerance search | `batched_bisection_f64.wgsl` (wetSpring) | Low |
| Crank-Nicolson PDE | `pde::crank_nicolson::CrankNicolson1D` (f64 + GPU shader!) | **WIRED** |
| BFGS optimizer | `optimize::bfgs` | Low |
| Brent VG inverse | `optimize::brent` | **WIRED** |
| bisect/Newton/secant | `optimize::{bisect, newton, secant}` | Low |
| Batched bisection GPU | `optimize::BatchedBisectionGpu` | Low |

### Tier C: Needs New Primitive (1 item)

| Need | Description |
|------|-------------|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs (not GPU) |

---

## ToadStool S42–S66 Evolution (180+ commits)

ToadStool underwent massive evolution since S42. Key milestones:

| Session | What Changed | Impact on airSpring |
|---------|-------------|---------------------|
| S42 | Rename BarraCUDA → BarraCuda, 19 new WGSL shaders | Naming alignment |
| S46 | Cross-project absorption: lattice QCD, MD transport, bio ODE | New ODE primitives |
| S49 | Shader-first architecture, 13 f32→f64 evolutions | Better f64 coverage |
| S51 | CG shaders, ESN NPU, generic ODE, CPU solver | `solve_f64_cpu()`, `OdeSystem` trait |
| S52 | 18 absorptions, unified_hardware, tolerances, provenance | Infrastructure primitives |
| S54 | **TS-001/003/004 resolved**, baseCamp primitives, 5 WGSL | Our bugs fixed |
| S56 | Final absorptions, idiomatic Rust | All 46 items complete |
| S57 | +47 tests, coverage push | 4,224+ core tests |
| S58-S59 | df64, Fp64Strategy, ridge, ValidationHarness | Cross-spring quality |
| S60 | DF64 FMA, transcendentals, CN fix, Cholesky SPD | Math precision |
| S61-63 | Sovereign compiler, SPIR-V passthrough, `CrankNicolson1D` **f64** | **CN now f64!** |
| S64 | Stats absorption (metrics, diversity from Springs), `chrono` removed | Diversity leaning |
| S65 | Smart refactoring, dead code removal, doc cleanup | Stabilization |
| S66 | **Cross-spring absorption**: regression, hydrology, moving_window_f64, spearman re-export, 8 SoilParams, mae/hill/monod, multi-precision WGSL | **All metalForge absorbed** |

## Upstream Capabilities — Wired and Available

### Wired (using in production)

| Capability | Module | Wired In | Status |
|-----------|--------|----------|--------|
| `barracuda::tolerances` | `tolerances` | v0.3.6 | **LEANING** — re-exported |
| `barracuda::validation::ValidationHarness` | `validation` | v0.3.6 | **LEANING** — all 27 binaries (incl. validate_atlas, 1393 checks) |
| `pde::richards::solve_richards` | `pde` | v0.4.0 | **WIRED** — `gpu::richards` |
| `pde::crank_nicolson::CrankNicolson1D` | `pde` | v0.4.4 | **WIRED** — CN f64 diffusion cross-val |
| `optimize::nelder_mead` | `optimize` | v0.4.1 | **WIRED** — isotherm fitting |
| `optimize::multi_start_nelder_mead` | `optimize` | v0.4.1 | **WIRED** — global isotherm search |
| `stats::diversity::*` | `stats` | v0.4.3 | **LEANING** — `eco::diversity` delegates |
| `stats::metrics::*` | `stats` | v0.4.3 | **LEANING** — `testutil::stats` delegates |
| `stats::normal::norm_ppf` | `stats` | v0.4.4 | **WIRED** — `McEt0Result::parametric_ci()` |
| `optimize::brent` | `optimize` | v0.4.4 | **WIRED** — `inverse_van_genuchten_h()` θ→h inversion |

### Available (not yet needed)

| Capability | Module | Added In | Potential Use |
|-----------|--------|----------|---------------|
| `FusedMapReduceF64::dot(a, b)` | `ops` | S51 | GPU dot product convenience |
| `barracuda::provenance` | `provenance` | S52 | 12 `ProvenanceTag` consts for origin tracking |
| `solve_f64_cpu()` | `linalg::solve` | S51 | Gaussian elimination + partial pivoting |
| `GpuSessionBuilder` | `session` | S52 | Pre-warmed GPU sessions |
| `OdeSystem` + `BatchedOdeRK4` | `numerical` | S51 | Generic ODE with WGSL template |
| `NelderMeadGpu` | `optimize` | S52+ | GPU-resident NM (5-50 params, not cost-effective for 2-param) |
| `ResumableNelderMead` | `optimize` | S52+ | Checkpoint/resume for long-running optimizers |
| `bfgs` | `optimize` | S52+ | Quasi-Newton with gradient (smooth objectives) |
| `bisect` | `optimize` | S52+ | Robust bracketed root-finding |
| `newton` / `secant` | `optimize` | S52+ | Derivative-based root-finding |
| `BatchedBisectionGpu` | `optimize` | S52+ | GPU-parallel batched root-finding |
| `adaptive_penalty` | `optimize` | S52+ | Constrained optimization with penalty |
| `unified_hardware` | `unified_hardware` | S52 | `HardwareDiscovery`, `ComputeScheduler` — metalForge target |
| `chi2_decomposed` | `stats` | S52 | Chi-squared goodness-of-fit |
| `spectral_density` | `stats` | S57 | RMT spectral analysis |
| `normal::norm_cdf` | `stats` | S52+ | Normal cumulative distribution |
| `spearman_correlation` | `stats::correlation` | S66 (R-S66-005) | Rank correlation — **now re-exported** from `stats/mod.rs` |

---

## Quality Gates

| Check | Status |
|-------|--------|
| `cargo fmt --check` | **Clean** |
| `cargo clippy -- -D warnings` | **0 warnings** (pedantic + nursery via `[lints.clippy]`) |
| `cargo doc --no-deps` | **Builds**, 0 warnings |
| `cargo test` | **491 total** (lib + integration + doc-tests) |
| `cargo llvm-cov --lib` | **97.45%** line coverage (97.58% functions) |
| `unsafe` code | **Zero** |
| `unwrap()` in lib | **Zero** (all in `#[cfg(test)]` or validation-binary JSON helpers) |
| Files > 1000 lines | **Zero** (max src: 800 lines `eco/richards.rs`, max test: 698 lines `eco_integration.rs`) |
| Validation binaries | **27/27 PASS** (570 checks + 1393 atlas) |
| GPU dispatch (P0 blocker) | **RESOLVED** — S66 explicit BGL (R-S66-041) |
| Cross-validation | **75/75 MATCH** (tol=1e-5) |

---

## Cross-Spring Provenance

| Primitive | Origin Spring | What airSpring Gets |
|-----------|--------------|---------------------|
| `pow_f64`, `exp_f64`, `log_f64` | hotSpring | VG retention, atmospheric pressure |
| `kriging_f64`, `fused_map_reduce` | wetSpring | Spatial interpolation, seasonal aggregation |
| `moving_window_stats` | wetSpring | IoT stream smoothing |
| `ridge_regression` | wetSpring | Sensor correction pipeline |
| `nelder_mead`, `multi_start` | neuralSpring | Isotherm fitting |
| `ValidationHarness` | neuralSpring | All 27 validation binaries |
| `norm_ppf` (Moro 1995) | hotSpring | MC ET₀ parametric confidence intervals |
| `brent` (Brent 1973) | neuralSpring | VG pressure head inversion (θ→h) |
| `pde::richards` | airSpring → upstream | 1D Richards equation (absorbed S40) |
| `stats::regression` | airSpring metalForge → upstream | Sensor correction fitting (absorbed S66) |
| `stats::hydrology` | airSpring metalForge → upstream | Hargreaves ET₀, batch (absorbed S66) |
| `stats::moving_window_f64` | airSpring metalForge → upstream | f64 stream statistics (absorbed S66) |

### airSpring Contributions Back

| Fix | Impact | Commit |
|-----|--------|--------|
| TS-001: `pow_f64` fractional exponent | All Springs using VG/exponential math | S54 (H-011) |
| TS-003: `acos` precision boundary | All Springs using trig in f64 shaders | S54 (H-012) |
| TS-004: reduce buffer N≥1024 | All Springs using `FusedMapReduceF64` | S54 (H-013) |
| Richards PDE | airSpring → `pde::richards` (S40) | upstream |
