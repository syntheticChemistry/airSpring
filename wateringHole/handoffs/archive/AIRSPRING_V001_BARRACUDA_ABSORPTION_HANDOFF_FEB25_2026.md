# airSpring V001: BarraCuda Evolution + Absorption Handoff

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — Ecological & Agricultural Sciences)
**To**: ToadStool/BarraCuda core team
**airSpring Version**: 0.3.8 (293 tests, 123 validation checks, 65/65 cross-validation)
**ToadStool HEAD**: `02207c4a` (S62+)
**License**: AGPL-3.0-or-later

---

## Part 1: What airSpring Uses from BarraCuda

### GPU Orchestrators (6 wired, all operational)

| airSpring Module | BarraCuda Primitive | Usage | Provenance |
|-----------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | Batched FAO-56 ET₀ for N station-days | hotSpring `pow_f64` fix (TS-001) |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | Batched depletion update per day | Multi-spring shared shader |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | Soil moisture spatial interpolation | wetSpring interpolation |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64::FusedMapReduceF64` | Seasonal sum/mean/max/min for N≥1024 | wetSpring, airSpring TS-004 fix |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats::MovingWindowStats` | IoT sensor stream smoothing (24h window) | wetSpring S28+ environmental |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | Regularized sensor calibration | wetSpring ESN calibration |

### Stats & Validation (5 primitives)

| Primitive | airSpring Integration |
|-----------|-----------------------|
| `stats::pearson_correlation` | R² in `testutil::r_squared` |
| `stats::correlation::spearman_correlation` | Nonparametric validation in `testutil::spearman_r` |
| `stats::bootstrap::bootstrap_ci` | RMSE uncertainty in `testutil::bootstrap_rmse` |
| `stats::correlation::std_dev` | Cross-validation integration tests |
| `validation::ValidationHarness` | All 6 validation binaries (absorbed from neuralSpring S59) |

### Benchmark Results (CPU baselines, `--release`)

| Operation | N | Time (µs) | Throughput | Shader |
|-----------|---|-----------|------------|--------|
| ET₀ (FAO-56) | 10,000 | 795 | 12.6M ops/sec | `batched_elementwise_f64.wgsl` |
| Reduce (seasonal) | 100,000 | 251 | 399M elem/sec | `fused_map_reduce_f64.wgsl` |
| Stream smooth | 8,760 (24h) | 270 | 32.4M elem/sec | `moving_window.wgsl` |
| Kriging | 20→500 | 26 | — | `kriging_f64.wgsl` |
| Ridge regression | 5,000 | 48 | R²=1.000 | (CPU-only) |

---

## Part 2: What airSpring Contributed Upstream

### Bug Fixes (3, all resolved in ToadStool)

| ID | Summary | Severity | Impact |
|----|---------|:--------:|--------|
| **TS-001** | `pow_f64` returns 0.0 for fractional exponents | Critical | All Springs using `batched_elementwise_f64` |
| **TS-003** | `acos_simple`/`sin_simple` low-order approximations | Low | Precision at boundary values |
| **TS-004** | `FusedMapReduceF64` buffer conflict for N≥1024 | High | All Springs using `fused_map_reduce_f64` |

These were discovered through airSpring's domain validation (ET₀ atmospheric
pressure calculation exposed TS-001; seasonal statistics at N=1024 exposed TS-004).
The fixes are in ToadStool commit `0c477306` and benefit all Springs.

### Domain Knowledge Contributions

1. **FAO-56 as a GPU test case**: The Penman-Monteith equation exercises `exp`, `pow`,
   `sqrt`, `sin`, `cos`, `acos`, `atan2` in f64 — excellent shader coverage.
2. **Kriging validation**: Ordinary kriging with LU decomposition and spherical/exponential
   variograms tested against known analytical solutions.
3. **Mass conservation invariant**: Water balance simulations provide a strict
   conservation check (total in = total out + storage change, error < 1e-10 mm).
4. **IoT stream patterns**: Real sensor data patterns (diurnal temperature cycles,
   slowly-depleting soil moisture) exercise `MovingWindowStats` with realistic inputs.

---

## Part 3: What airSpring Needs Next (Absorption Targets)

### Tier B — Ready to Wire (upstream primitive exists)

| Need | BarraCuda Primitive | airSpring Purpose |
|------|--------------------|----|
| Sensor batch calibration | `batched_elementwise_f64.wgsl` (op=5) | Batch SoilWatch 10 VWC calibration |
| Hargreaves ET₀ batch | `batched_elementwise_f64.wgsl` (op=6) | Simpler ET₀ (no humidity/wind) |
| Kc climate adjustment | `batched_elementwise_f64.wgsl` (op=7) | FAO-56 Eq. 62 crop coefficient |
| Nonlinear curve fitting | `optimize::nelder_mead`, `NelderMeadGpu` | Correction equation fitting |
| m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring (wetSpring) |

### Tier B (new) — Richards PDE (PROMOTED from Tier C)

| Need | BarraCuda Primitive | airSpring Purpose |
|------|--------------------|----|
| 1D Richards equation | `pde::richards::solve_richards` | Unsaturated soil water flow (van Genuchten-Mualem) |
| Tridiagonal solve | `linalg::tridiagonal_solve_f64` | Implicit PDE time-stepping |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Dynamic soil moisture models |

**Note (v0.3.8):** ToadStool now provides a complete Richards equation solver
with van Genuchten-Mualem constitutive relations, Picard iteration,
Crank-Nicolson time-stepping, and Thomas spatial solver. airSpring needs to
wire `SoilParams` from `eco::soil_moisture` textures and validate against
HYDRUS benchmarks (Experiment 006).

### Tier C — Needs New Primitive

| Need | Description | Complexity |
|------|-------------|:---------:|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs | Low — not GPU |

---

## Part 4: metalForge Absorption Candidates

The `metalForge/forge/` crate (18 tests) contains airSpring-specific primitives
staged for upstream absorption into `barracuda`. These are CPU-only and cover
statistical metrics and regression.

### Metrics Module (absorption target: `barracuda::stats::metrics`)

| Function | Description | Tests |
|----------|-------------|:-----:|
| `rmse` | Root Mean Square Error | 3 |
| `mbe` | Mean Bias Error | 2 |
| `nash_sutcliffe` | Nash-Sutcliffe Efficiency | 3 |
| `index_of_agreement` | Willmott's Index of Agreement | 3 |

These are domain-standard hydrology/meteorology metrics. BarraCuda's `stats`
module already has `pearson_correlation`, `bootstrap_ci`, etc. — these would
be natural additions.

### Regression Module (absorption target: `barracuda::stats::regression`)

| Function | Description | Tests |
|----------|-------------|:-----:|
| `fit_linear` | Linear least squares (y = ax + b) | 2 |
| `fit_quadratic` | Quadratic normal equations (y = ax² + bx + c) | 2 |
| `fit_exponential` | Log-linearized exponential (y = a·exp(bx)) | 1 |
| `fit_logarithmic` | Log-linearized logarithmic (y = a·ln(x) + b) | 1 |
| `fit_all` | Best-of-four model selection | 1 |

These complement the existing `barracuda::linalg::ridge::ridge_regression`.
Together they form a complete curve fitting toolkit for calibration workflows.

---

## Part 5: Cross-Spring Evolution Insights

### What We Learned About Shader Precision

1. **f64 is essential for agriculture**: ET₀ calculations involve `P = 101.3 × ((293-0.0065z)/293)^5.26` —
   fractional exponents on small differences. f32 loses 2-3 significant digits.
   hotSpring's df64 arithmetic is the foundation.

2. **GPU-CPU agreement must be exact**: Our cross-validation harness demands 1e-5
   agreement between Python, Rust CPU, and Rust GPU. Any shader precision issue
   (like TS-001/003) becomes visible immediately.

3. **Mass conservation is a strict invariant**: Water balance provides an independent
   check on floating-point accumulation. Over a 153-day season with 918 station-days
   of real data, our mass balance error is < 1e-10 mm. This exercises GPU reduce
   precision at scale.

### What We Learned About Ecosystem Design

1. **The Write → Absorb → Lean pattern works**: We wrote `ValidationRunner` locally,
   ToadStool absorbed it as `ValidationHarness`, and we now lean on upstream. Same
   pattern for metrics (forge → barracuda::stats).

2. **Cross-spring shaders are genuinely useful**: wetSpring's `kriging_f64` and
   `moving_window_stats` solved real airSpring problems. We didn't duplicate — we
   wired. This is the ecosystem working as designed.

3. **Bug reports from domain science are high-value**: TS-001 (pow_f64) was invisible
   to synthetic tests but immediately visible when computing atmospheric pressure
   from real elevation data. Domain validation catches different bugs than unit tests.

### Recommendations for ToadStool Evolution

1. **Consider absorbing `eco::correction::fit_ridge` pattern**: The design matrix
   construction (x, 1.0 intercept column) and goodness-of-fit reporting are
   reusable across Springs. A `barracuda::stats::regression` module would serve
   all Springs doing calibration work.

2. **Consider absorbing hydrology metrics**: NSE, IA, MBE are standard in earth
   sciences and complement the existing Pearson/Spearman/bootstrap suite. They're
   pure functions with no dependencies.

3. **The `batched_elementwise_f64` op-code pattern scales well**: We've used op=0
   (ET₀) and op=1 (water balance). Adding ops 5-7 for sensor calibration, Hargreaves,
   and Kc adjustment would demonstrate the pattern's extensibility. The shader's
   `switch(op)` structure is clean.

4. **Richards equation is the next frontier**: 1D unsaturated flow PDE would benefit
   from `ops::crank_nicolson` + tridiagonal solver. This is a high-complexity but
   high-impact primitive — every Spring doing PDE work would benefit.

---

## Summary

airSpring demonstrates that domain-validated agricultural science (FAO-56, Dong lab)
runs correctly on the BarraCuda/ToadStool GPU stack. The ecosystem's cross-spring
shader evolution (608 WGSL shaders, 46 absorptions) provides genuine value — we use
5 shared shaders contributed by other Springs and discovered 3 bugs that improved
precision for everyone.

**airSpring by the numbers:**
- 5 papers reproduced, 142/142 Python + 123/123 Rust checks
- 253 tests (175 unit + 76 integration + 2 doc-tests)
- 65/65 Python↔Rust cross-validation match (tol=1e-5)
- 918 real station-days (Open-Meteo ERA5, 6 Michigan stations)
- 6 GPU orchestrators wired to BarraCuda primitives
- 17 evolution gaps tracked (8 Tier A, 8 Tier B, 1 Tier C) — Richards PDE promoted C→B
- 3 bug fixes contributed upstream (TS-001, TS-003, TS-004)
- 18 metalForge tests staged for absorption

The path to Penny Irrigation: a $200 IoT sensor node, free Open-Meteo weather data,
and a $600 GPU running BarraCuda's f64 shaders — sovereign precision irrigation for
any farmer, anywhere, with no institutional access required.

---

*AGPL-3.0-or-later. airSpring v0.3.8, ToadStool HEAD `02207c4a`.
Richards PDE promoted C→B — upstream `pde::richards::solve_richards` available.
metalForge candidates (metrics, regression, hydrology, moving_window_f64) NOT yet absorbed.*
