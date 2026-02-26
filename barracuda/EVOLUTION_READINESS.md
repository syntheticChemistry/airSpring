# airSpring BarraCuda — Evolution Readiness

**Last Updated**: February 25, 2026 (v0.4.2 — 433 lib + 115 integration)
**ToadStool PIN**: `02207c4a` (S62+ — 170 commits, 46 absorptions, 4,224+ core tests, 758 WGSL shaders)
**License**: AGPL-3.0-or-later

---

## Write → Absorb → Lean Status

airSpring follows the same pattern as hotSpring and wetSpring: implement locally,
validate against papers, hand off to ToadStool/BarraCuda, lean on upstream.

### Already Absorbed (Lean)

| Module | Absorbed Into | When | Status |
|--------|--------------|------|--------|
| `ValidationRunner` | `barracuda::validation::ValidationHarness` | S59 | **Leaning** — all 18 binaries use upstream |
| `van_genuchten` | `barracuda::pde::richards::SoilParams` | S40 | **Leaning** — `gpu::richards` bridges to upstream |
| `isotherm NM` | `barracuda::optimize::nelder_mead` | S62 | **Leaning** — `gpu::isotherm` bridges to upstream |

### Ready for Absorption (4 metalForge modules)

| Module | Target | Tests | Provenance |
|--------|--------|:-----:|------------|
| `forge::metrics` | `barracuda::stats::metrics` | 11 | Dong 2020, 918 station-days |
| `forge::regression` | `barracuda::stats::regression` | 11 | Dong 2020 sensor corrections |
| `forge::moving_window` | `barracuda::ops::moving_window_stats_f64` | 7 | IoT sensor smoothing (f64) |
| `forge::hydrology` | `barracuda::ops::hydrology` | 13 | FAO-56, Hargreaves 1985 |

See `metalForge/ABSORPTION_MANIFEST.md` for full signatures and validation details.

### Stays Local (domain-specific)

| Module | Reason |
|--------|--------|
| `eco::dual_kc` | FAO-56 Ch 7/11 domain logic — too specialized for barracuda |
| `eco::sensor_calibration` | SoilWatch 10 specific — domain consumer |
| `eco::crop` | FAO-56 Table 12 crop database — domain data |
| `io::csv_ts` | airSpring-specific IoT CSV parser |
| `testutil::generators` | Synthetic IoT data for airSpring tests |

---

## GPU Evolution Tiers

### Tier A: Integrated (8 modules — GPU primitive wired, validated)

| airSpring Module | BarraCuda Primitive | Status |
|-----------------|--------------------|----|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **INTEGRATED** |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **WIRED** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **WIRED** |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | **WIRED** |
| `gpu::isotherm::fit_*_nm/global` | `optimize::nelder_mead` + `multi_start` | **WIRED** |

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
| Crank-Nicolson f64 | `ops::crank_nicolson` (**f32 only — needs f64**) | Medium |
| BFGS optimizer | `optimize::bfgs` | Low |

### Tier C: Needs New Primitive (1 item)

| Need | Description |
|------|-------------|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs (not GPU) |

---

## ToadStool S42–S62 Evolution (170 commits)

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
| S62 | BandwidthTier, PeakDetectF64, infrastructure | Performance primitives |

## Upstream Capabilities Available (Not Yet Wired)

These exist in ToadStool/BarraCuda but airSpring hasn't needed them yet:

| Capability | Module | Added In | Potential Use |
|-----------|--------|----------|---------------|
| `FusedMapReduceF64::dot(a, b)` | `ops` | S51 | GPU dot product convenience |
| `barracuda::tolerances` | `tolerances` | S52 | Centralized tolerance with justification |
| `barracuda::provenance` | `provenance` | S52 | 12 `ProvenanceTag` consts for origin tracking |
| `solve_f64_cpu()` | `linalg::solve` | S51 | Gaussian elimination + partial pivoting |
| `GpuSessionBuilder` | `session` | S52 | Pre-warmed GPU sessions |
| `OdeSystem` + `BatchedOdeRK4` | `numerical` | S51 | Generic ODE with WGSL template |
| `NelderMeadGpu` | `optimize` | S52+ | GPU-resident NM (5-50 params) |
| `crank_nicolson` | `ops`/`pde` | S46+ | Implicit PDE solver (**f32 only** — Richards needs f64) |
| `unified_hardware` | `unified_hardware` | S52 | `HardwareDiscovery`, `ComputeScheduler` — metalForge target |
| `bfgs` | `optimize` | S52+ | Quasi-Newton with gradient |
| `adaptive_penalty` | `optimize` | S52+ | Constrained optimization with penalty |
| `chi2_decomposed` | `stats` | S52 | Chi-squared goodness-of-fit |
| `spectral_density` | `stats` | S57 | RMT spectral analysis |
| `normal::norm_cdf/ppf` | `stats` | S52+ | Normal distribution CDF/quantile |
| `spearman_correlation` | `stats::correlation` | S52+ | Rank correlation (fn exists, not re-exported from mod) |

---

## Quality Gates

| Check | Status |
|-------|--------|
| `cargo fmt --check` | **Clean** |
| `cargo clippy -- -D warnings` | **0 warnings** (pedantic via `[lints.clippy]`) |
| `cargo doc --no-deps` | **Builds**, 0 warnings |
| `cargo test` | **601 total** (433 lib + 115 integration + 53 forge) |
| `cargo llvm-cov --lib` | **97.55%** line coverage |
| `unsafe` code | **Zero** |
| `unwrap()` in lib | **Zero** (all in `#[cfg(test)]`) |
| Files > 1000 lines | **Zero** (max: 845 lines) |
| Validation binaries | **16/16 PASS** (341 checks) |
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
| `ValidationHarness` | neuralSpring | All 16 validation binaries |
| `pde::richards` | airSpring → upstream | 1D Richards equation (absorbed S40) |

### airSpring Contributions Back

| Fix | Impact | Commit |
|-----|--------|--------|
| TS-001: `pow_f64` fractional exponent | All Springs using VG/exponential math | S54 (H-011) |
| TS-003: `acos` precision boundary | All Springs using trig in f64 shaders | S54 (H-012) |
| TS-004: reduce buffer N≥1024 | All Springs using `FusedMapReduceF64` | S54 (H-013) |
| Richards PDE | airSpring → `pde::richards` (S40) | upstream |
