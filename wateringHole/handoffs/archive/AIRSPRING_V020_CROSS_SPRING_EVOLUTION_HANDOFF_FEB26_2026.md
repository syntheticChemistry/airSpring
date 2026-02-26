# airSpring V020 — Cross-Spring Evolution & Absorption Handoff

**Date**: 2026-02-26
**From**: airSpring v0.4.6
**To**: ToadStool / BarraCuda team
**ToadStool pin**: S68 (`f0feb226`)
**License**: AGPL-3.0-or-later
**Covers**: Full cross-spring evolution lineage through S68, validated benchmarks, and evolution recommendations

**airSpring**: 608 Rust tests + 1354 atlas checks, 22 binaries, 47 cross-spring evolution tests, 0 clippy errors

---

## Executive Summary

airSpring has completed its S68 rewiring and now exercises **14 upstream BarraCuda primitives**
across 5 cross-spring shader families. This handoff documents:

1. **What airSpring learned** — precision requirements, scale bottlenecks, and domain patterns
   that generalize across springs
2. **What ToadStool should absorb next** — domain-validated patterns ready for upstream promotion
3. **Cross-spring evolution map** — which springs contributed what, and how the ecosystem benefits
4. **Benchmarks** — modern S68 stack performance on real agricultural workloads

This is a comprehensive evolution handoff, not a sync notification. It supersedes V018 (atlas)
and V019 (S68 sync) as the current active handoff.

---

## Part 1: Cross-Spring Shader Evolution — Who Helped Whom

The ToadStool ecosystem now has **774+ WGSL shaders** from **46+ cross-spring absorptions**.
Each spring contributed domain-specific GPU primitives that benefit the entire ecosystem:

### hotSpring → Precision Foundation

| Contribution | Impact on airSpring | Impact on Ecosystem |
|-------------|--------------------|--------------------|
| `df64_core.wgsl` — double-float arithmetic | Enables f64-quality GPU computation for ET₀, Richards PDE | Foundation for ALL f64 GPU paths in every spring |
| `math_f64.wgsl` — pow, exp, log, trig in f64 | TS-001 pow_f64 fix came from airSpring's atmospheric pressure test | Shared by 4 springs for transcendental evaluation |
| S67-S68 "math is universal, precision is silicon" | Validated by airSpring: 608 tests, zero numerical regression | 334+ shaders evolved to f64-canonical |
| Lanczos spectral, Hermite/Laguerre polys | Available for future soil spectroscopy | Nuclear physics → environmental spectral analysis |

### wetSpring → Bio/Stats Infrastructure

| Contribution | Impact on airSpring | Impact on Ecosystem |
|-------------|--------------------|--------------------|
| `kriging_f64.wgsl` | `gpu::kriging::KrigingInterpolator` — soil moisture spatial interpolation | Spatial statistics for any georeferenced data |
| `fused_map_reduce_f64.wgsl` | `gpu::reduce::SeasonalReducer` — GPU aggregation for N≥1024 | Single-dispatch map+reduce (TS-004 fix from airSpring) |
| `moving_window.wgsl` | `gpu::stream::StreamSmoother` — IoT sensor smoothing | Environmental monitoring stream analytics |
| Shannon/Simpson/Bray-Curtis diversity | `eco::diversity` wired for crop diversity metrics | Biodiversity analytics for all bio springs |
| Ridge regression | `eco::correction::fit_ridge` — regularized sensor calibration | ESN-origin regression available to all springs |

### neuralSpring → Optimization & Validation

| Contribution | Impact on airSpring | Impact on Ecosystem |
|-------------|--------------------|--------------------|
| `nelder_mead` / `multi_start_nelder_mead` | `gpu::isotherm` — nonlinear isotherm fitting | Derivative-free optimization for any spring |
| `ValidationHarness` | 22 binaries with structured pass/fail | Standard validation pattern across all springs |
| Pairwise metrics (Jaccard, L2, Hamming) | Available for sensor similarity analysis | Cross-domain distance metrics |

### groundSpring → Uncertainty Quantification

| Contribution | Impact on airSpring | Impact on Ecosystem |
|-------------|--------------------|--------------------|
| `mc_et0_propagate_f64.wgsl` | `gpu::mc_et0` — Monte Carlo ET₀ uncertainty bands | Input uncertainty propagation template |
| `batched_multinomial` | Available for stochastic crop modeling | Multinomial sampling primitive |
| `rawr_mean` bootstrap | `testutil::bootstrap_rmse` wired | Robust bootstrap mean estimation |

### airSpring → Contributions Back to ToadStool

| What We Gave | Absorbed In | Benefit |
|-------------|------------|---------|
| **TS-001** pow_f64 fix (fractional exponents returned 0.0) | S54 | Fixed atmospheric pressure calc; benefits all springs using `pow_f64` |
| **TS-003** acos/sin precision (low-order polynomial → full math_f64) | S54 | Solar geometry precision; benefits any angular computation |
| **TS-004** FusedMapReduceF64 buffer conflict for N≥1024 | S54 | Stabilized GPU reduce for all springs |
| stats metrics (RMSE, MBE, NSE, R², IA, hit_rate) | S64 | `barracuda::stats::metrics` — shared validation metrics |
| regression (fit_linear, fit_quadratic, fit_exponential) | S66 | `barracuda::stats::regression` — from metalForge absorption |
| hydrology (hargreaves_et0, thornthwaite, solar_declination) | S66 | `barracuda::stats::hydrology` — environmental hydrology |
| moving_window_f64 | S66 | `barracuda::stats::moving_window_f64` — precision stream stats |
| Richards PDE (van Genuchten-Mualem) | S40 | `barracuda::pde::richards` — vadose zone flow simulation |

---

## Part 2: Current BarraCuda Wiring (14 Primitives)

| airSpring Module | BarraCuda Primitive | Spring Origin | Status |
|-----------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | Multi-spring | **GPU-FIRST** |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | Multi-spring | **GPU-STEP** |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | wetSpring | **INTEGRATED** |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | wetSpring | **GPU N≥1024** |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | wetSpring S28+ | **WIRED** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN | **WIRED** |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | airSpring S40 | **WIRED** |
| `gpu::isotherm::fit_*_nm` | `optimize::nelder_mead` | neuralSpring S62 | **WIRED** |
| `validation::ValidationHarness` | `barracuda::validation` | neuralSpring | **ABSORBED** |
| `testutil::r_squared` | `stats::pearson_correlation` | Shared | **WIRED** |
| `testutil::spearman_r` | `stats::spearman_correlation` | Shared S66 | **WIRED** |
| `testutil::bootstrap_rmse` | `stats::bootstrap_ci` | hotSpring+groundSpring | **WIRED** |
| `eco::diversity::*` | `stats::diversity::*` | wetSpring S64 | **WIRED** |
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` CPU mirror | groundSpring S64 | **WIRED** |

---

## Part 3: S68 Benchmark Results (Release Mode)

All benchmarks run on the modern S68 stack with universal f64 precision:

| Benchmark | N | Time | Throughput | Cross-Spring Lineage |
|-----------|---|------|------------|---------------------|
| ET₀ (FAO-56) | 200 stations × 150 days | < 5s | 6K+ station-days/s | hotSpring pow_f64 + airSpring elementwise |
| Richards PDE | 10 × (50 nodes, 0.1d) | < 10s | 1+ sim/s (debug) | airSpring → S40, hotSpring df64 |
| Regression f64 | 100 points linear + quadratic | < 1ms | Exact to 1e-10 | airSpring metalForge → S66 |
| Hargreaves ET₀ | Single point | < 1ms | Deterministic | airSpring metalForge → S66 |
| Shannon/Simpson diversity | 4-species community | < 1ms | Stable to 0.01 | wetSpring → S64 |
| Moving window | 50 points, window=5 | < 1ms | All finite/non-negative | airSpring IoT → S66 |
| Brent optimizer | Quadratic minimum | < 1ms | Converges to 1e-8 | neuralSpring lineage |
| Spearman correlation | 20-point monotonic | < 1ms | ρ = 1.0 ± 1e-10 | Cross-spring metric |
| Bootstrap CI | 100 points, 1000 resamples | < 100ms | Valid intervals | hotSpring + groundSpring |
| Full test suite | 608 tests + 1354 atlas | < 15s | All pass | 5-spring ecosystem |

---

## Part 4: What ToadStool Should Evolve Next

### High Priority — Ready for Absorption

These patterns emerged from airSpring's real-data validation and would benefit all springs:

1. **`tracing-subscriber` documentation for `ValidationHarness`**
   S68 migrated `ValidationHarness` from `println!` to `tracing::info!`. Any spring consuming
   `ValidationHarness` now needs `tracing-subscriber` + `init_tracing()`. Document this
   requirement in `barracuda::validation` module docs or provide a convenience re-export.

2. **Batch Hargreaves ET₀ GPU op**
   `hargreaves_et0(ra, tmax, tmin)` is a simple 3-input formula suitable for
   `batched_elementwise_f64.wgsl` as **op=6**. Simpler than Penman-Monteith (no humidity/wind).
   airSpring validated CPU precision; GPU batch would accelerate atlas-scale workloads.

3. **Batch sensor calibration GPU op**
   SoilWatch 10 VWC calibration as `batched_elementwise_f64.wgsl` **op=5**.
   airSpring has CPU-validated calibration curves ready for GPU promotion.

### Medium Priority — Evolution Opportunities

4. **Crank-Nicolson cross-validation for Richards**
   `pde::crank_nicolson::CrankNicolson1D` is now f64 + GPU shader (S62). airSpring's
   Richards solver uses implicit Euler + Thomas. CN cross-validation would provide
   independent verification of both solvers — useful for all springs with PDE needs.

5. **RK45 adaptive ODE for soil dynamics**
   `numerical::rk45_solve` (Dormand-Prince) is available upstream. airSpring could
   wire it for dynamic soil moisture models and biochar kinetics. wetSpring could
   use it for population dynamics; hotSpring for coupled ODE systems.

6. **Sparse solvers for large kriging systems**
   `linalg::sparse::BiCGStab` and `CG` GPU solvers are available. Current kriging
   uses dense LU — sparse solvers would enable 1000+ station interpolation.

### Low Priority — Future

7. **HTTP/JSON data client** — Not GPU, but standardizing Open-Meteo / NOAA CDO
   ingestion would benefit any spring doing real-data validation.

---

## Part 5: Lessons Learned (Relevant to ToadStool Evolution)

1. **CSV I/O is the bottleneck, not compute**
   At 100-station atlas scale, parsing 15,300 station-days of CSV takes ~70% of wall time.
   The FAO-56 pipeline itself is <1ms per station. GPU acceleration ROI is highest when
   data is already in GPU memory. Consider: columnar binary format as a `barracuda::io` primitive.

2. **API rate limiting needs cooperative backoff**
   Open-Meteo API returns HTTP 429 at ~50 requests/hour. Our `download_atlas_80yr.py`
   implements exponential backoff (30s → 10min). If ToadStool ever adds data ingestion
   primitives, cooperative rate limiting should be first-class.

3. **`ValidationHarness` at 1354 checks scales well**
   The harness handles 1354 checks across 104 stations with no performance issues.
   The `tracing::info!` migration is cleaner than `println!` and allows filtering.
   Consider adding `ValidationHarness::to_json()` for machine-readable CI output.

4. **Cross-spring tests are documentation**
   Our 47 cross-spring evolution tests in `cross_spring_evolution.rs` document WHICH
   spring contributed EACH primitive. This is living provenance that compiles. Other
   springs should adopt this pattern.

5. **metalForge → upstream absorption cycle is complete**
   airSpring's metalForge contained 6 modules. All 6 were absorbed upstream by S66:
   regression (R-S66-001), hydrology (R-S66-002), moving_window_f64 (R-S66-003),
   metrics (S64), diversity (S64 from wetSpring), bootstrap (S64 from groundSpring).
   The Write→Absorb→Lean cycle works. metalForge is now vestigial.

---

## Part 6: Quality Gates

| Gate | Value |
|------|-------|
| ToadStool pin | S68 (`f0feb226`) |
| `cargo test` | 608 PASS (464 lib + 47 cross-spring + 97 integration) |
| Atlas checks | 1354/1354 PASS (104 stations, 10 crops) |
| `cargo clippy` | 0 errors (pedantic + nursery) |
| `cargo fmt` | Clean |
| Cross-spring tests | 47/47 PASS (§1-§14) |
| Release benchmarks | 9/9 PASS (< 5s total) |
| P0 blockers | None |

---

## Part 7: File Manifest

| File | Purpose |
|------|---------|
| `barracuda/tests/cross_spring_evolution.rs` | 47 tests with provenance comments (§1-§14) |
| `barracuda/src/gpu/evolution_gaps.rs` | Living gap roadmap: 11 Tier A, 11 Tier B, 1 Tier C |
| `specs/CROSS_SPRING_EVOLUTION.md` | 774+ shader provenance, usage matrix, timeline |
| `barracuda/EVOLUTION_READINESS.md` | Absorption status, quality gates |
| `metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed, metalForge vestigial |

---

*airSpring v0.4.6 → ToadStool S68. 14 BarraCuda primitives wired, 47 cross-spring
evolution tests documenting 5-spring shader lineage. 608 Rust tests + 1354 atlas checks.
Write→Absorb→Lean cycle complete. AGPL-3.0-or-later.*
