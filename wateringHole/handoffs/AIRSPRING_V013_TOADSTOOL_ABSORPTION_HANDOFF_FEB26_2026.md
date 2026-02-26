# airSpring → ToadStool Handoff V013: Absorption Candidates + Cross-Spring Learnings

**Date**: February 26, 2026
**From**: airSpring (Precision Agriculture — v0.4.4, 643 tests, 18 binaries, 375 validation checks)
**To**: ToadStool / BarraCuda core team
**Supersedes**: V012 (retained in archive)
**ToadStool PIN**: `17932267` (S65 — 774 WGSL shaders)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring has completed its deep rewiring to ToadStool S65 primitives. This handoff
documents **what airSpring can contribute back** to the shared BarraCuda crate,
**what patterns we validated** that other Springs should adopt, and **what we learned**
about cross-spring evolution that is relevant to ToadStool's ongoing development.

11 Tier A modules wired. 643 tests pass. 96.81% library coverage. Zero unsafe code.
69x geometric mean CPU speedup over Python. All 15 validation binaries pass (375 checks).

---

## Part 1: Absorption Candidates for ToadStool

### Ready Now (metalForge modules, 53 tests)

| Module | Tests | Signatures | Target | Notes |
|--------|:-----:|------------|--------|-------|
| `forge::regression` | 11 | `linear_fit`, `quadratic_fit`, `exponential_fit`, `logarithmic_fit` | `barracuda::stats::regression` | Dong 2020 sensor correction curves |
| `forge::hydrology` | 13 | `penman_monteith_et0`, `hargreaves_et0`, `solar_declination`, `sunset_hour_angle` | `barracuda::ops::hydrology` | FAO-56 standard equations |
| `forge::moving_window_f64` | 7 | `moving_mean_f64`, `moving_std_f64`, `moving_min_f64`, `moving_max_f64` | `barracuda::ops::moving_window_stats_f64` | CPU f64 complement to existing f32 GPU |
| `forge::isotherm` | 5 | `fit_langmuir_linear`, `fit_freundlich_linear`, `langmuir_predict`, `freundlich_predict` | `barracuda::ops::isotherm` | Linearized isotherm fitting |

**Post-absorption**: airSpring rewires `eco::correction`, `eco::evapotranspiration`,
`io::csv_ts`, and `eco::isotherm` to delegate to upstream.

### Validated Patterns Worth Absorbing

| Pattern | What airSpring Proved | Candidate |
|---------|----------------------|-----------|
| VG θ→h inversion via `brent` | Monotone retention curves converge in <10 iterations | Add `inverse_van_genuchten_h` to `pde::richards` |
| Parametric MC CI via `norm_ppf` | z-score CI matches empirical ±3% at N=5000 | Document as `stats::normal` usage pattern |
| Stewart yield response | Single/multi-stage yield loss from water stress | `barracuda::ops::yield_response` (agri-specific) |
| Dual Kc partitioning | Soil evaporation (Ke) + transpiration (Kcb) separation | Domain-specific, stays in airSpring |
| CW2D media parameters | Richards solver works with extreme VG params (gravel, organic) | Add Carsel & Parrish presets to `pde::richards::SoilParams` |

### Previously Absorbed (Leaning)

| Module | Absorbed Into | Session | Status |
|--------|--------------|---------|--------|
| `ValidationRunner` | `barracuda::validation::ValidationHarness` | S59 | **Leaning** — all 16 binaries |
| `van_genuchten` | `barracuda::pde::richards::SoilParams` | S40 | **Leaning** — via `gpu::richards` |
| `isotherm NM` | `barracuda::optimize::nelder_mead` | S62 | **Leaning** — via `gpu::isotherm` |
| `stats::metrics` | `barracuda::stats::metrics` | S64 | **Leaning** — rmse, mbe, NSE, IA, R² |

---

## Part 2: Cross-Spring Learnings

### What We Learned About Primitive Evolution

1. **Brent >> Newton for bounded monotone functions**
   VG retention is smooth and strictly monotone on (−∞, 0). Brent converges in
   <10 iterations with guaranteed correctness. Newton needs derivative guards and
   can overshoot near saturation. Other Springs with monotone inversion problems
   (e.g., wetSpring enzyme kinetics, hotSpring equation of state) should prefer
   `brent` over hand-rolled Newton-Raphson.

2. **`norm_ppf` enables hybrid MC analysis**
   Parametric CI (mean ± z·σ) and empirical percentiles (sorted quantiles) tell
   different stories. At N=100 parametric is more stable; at N=10,000 empirical
   captures non-normality. Both should be reported. The `norm_ppf` API is clean
   and fast (negligible overhead vs MC sampling itself).

3. **Crank-Nicolson f64 is production-ready**
   The S62+ `CrankNicolson1D` with f64 + GPU shader produces physically correct
   diffusion profiles. Cross-validated against airSpring's implicit Euler + Picard
   Richards solver — both converge to same steady state within 1% for linear
   diffusion. ToadStool should promote CN f64 documentation (currently it's
   buried in `pde::crank_nicolson` without examples).

4. **Multi-start NM (LHS) is the sweet spot for 2-param fitting**
   For Langmuir/Freundlich (2 parameters), single NM from linearized LS guess
   gives R²=0.9995. Multi-start (8×LHS) finds the same optimum but proves
   it's global. `NelderMeadGpu` is overkill for <5 parameters — cost of GPU
   dispatch exceeds optimization time. Other Springs should use CPU NM for
   small-parameter problems and reserve GPU NM for 5-50 parameter problems.

5. **Population variance vs sample variance matters in MC**
   MC draws are the *entire population*, not a sample. Using ÷n (population
   variance) instead of ÷(n-1) (sample variance) gives correct uncertainty bands.
   `barracuda::stats::correlation::variance` uses ÷(n-1) — ToadStool should
   consider adding `population_variance` or documenting the distinction.

### Cross-Spring Shader Usage (from airSpring's perspective)

| Shader Family | Origin | airSpring Use | Other Springs Should Know |
|---------------|--------|---------------|---------------------------|
| `math_f64.wgsl` | hotSpring | VG retention, atmospheric pressure | Foundation for all f64 GPU math |
| `kriging_f64.wgsl` | wetSpring | Soil moisture spatial interpolation | Clean API, works well at 5-20 sensors |
| `fused_map_reduce_f64.wgsl` | wetSpring | Seasonal ET₀/WB aggregation | TS-004 fix critical for N≥1024 |
| `moving_window.wgsl` | wetSpring | IoT sensor smoothing (24h window) | 33M elem/sec, good for time series |
| `nelder_mead.wgsl` | neuralSpring | Isotherm fitting | Use CPU NM for <5 params |
| `mc_et0_propagate_f64.wgsl` | groundSpring | MC uncertainty propagation | Box-Muller + xoshiro128** |

---

## Part 3: P0 Blocker (Unchanged)

`BatchedElementwiseF64` GPU dispatch still panics at `pipeline.get_bind_group_layout(0)`
after S60–S65 SPIR-V path. airSpring guards with `catch_unwind` → SKIP.

**Blocks**: ET₀ GPU (op=0), water balance GPU (op=1), MC ET₀ kernel, any
Spring using `BatchedElementwiseF64`.

---

## Part 4: Remaining Open Items

| # | Item | Since | Status |
|:-:|------|:-----:|--------|
| 3 | Named VG constants in `pde::richards` | V007 | Still open — 8 Carsel & Parrish soil presets |
| 4 | Preallocation in `pde::richards` | V007 | Still open — Picard buffers outside solve loop |
| 5 | Re-export `spearman_correlation` in `stats/mod.rs` | V008 | Still open |
| N2 | Absorb `forge::regression` (4 models, 11 tests) | V010 | Ready |
| N3 | Absorb `forge::hydrology` (4 functions, 13 tests) | V010 | Ready |
| N4 | Absorb `forge::moving_window_f64` (CPU f64, 7 tests) | V010 | Ready |
| N5 | Absorb `forge::isotherm` (linearized fits, 5 tests) | V012 | Ready |
| N6 | Add `population_variance` to `stats` | V013 | **NEW** — MC draws need ÷n not ÷(n-1) |
| N7 | Document CN f64 usage examples in `pde::crank_nicolson` | V013 | **NEW** |

---

## Part 5: Benchmark Results (v0.4.4)

### CPU Throughput

| Computation | Throughput | Speedup vs Python |
|-------------|-----------|:-----------------:|
| FAO-56 ET₀ (10K) | 12.7M/s | 20x |
| VG θ(h) retention (100K) | 38.1M/s | 83x |
| Richards PDE (50 nodes) | 3,620/s | 502x |
| Yield single-stage (100K) | 1.08B/s | 81x |
| Isotherm NM 8×LHS | 36.5K/s | — |
| MC ET₀ (10K samples) | 4.2M samples/s | — |
| Brent VG inverse (1K) | 1.4M–3.1M/s | — |

### Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt` | Clean |
| `cargo clippy -D warnings` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `cargo test` | 643 total (464 lib + 126 integration + 53 forge) |
| `cargo llvm-cov --lib` | 96.81% lines, 97.58% functions |
| Validation binaries | 15/15 PASS (375 checks) |
| Cross-validation | 75/75 MATCH (tol=1e-5) |
| `unsafe` code | Zero |
| `unwrap()` in lib | Zero |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001–V009 | 2026-02-25 | (see archived handoffs) |
| V010 | 2026-02-26 | ToadStool S60–S65 sync: stats rewired upstream |
| V011 | 2026-02-26 | Full cross-spring rewiring, absorption roadmap |
| V012 | 2026-02-26 | S65 primitive rewiring: CN f64, brent+norm_ppf, 11 Tier A |
| **V013** | **2026-02-26** | **Absorption candidates + cross-spring learnings for ToadStool team** |

---

*End of V013 handoff. Direction: airSpring → ToadStool (unidirectional).
643 tests pass against ToadStool HEAD `17932267`. 4 metalForge modules ready
for absorption (36 tests). 5 validated patterns documented. 2 new ToadStool
evolution suggestions (population_variance, CN f64 docs).
P0 blocker: sovereign compiler GPU dispatch regression (unchanged).*
