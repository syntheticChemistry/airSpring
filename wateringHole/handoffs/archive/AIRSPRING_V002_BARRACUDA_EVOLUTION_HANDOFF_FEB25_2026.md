# airSpring V002: BarraCuda Evolution + Deep Debt + Absorption Handoff

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — Ecological & Agricultural Sciences)
**To**: ToadStool/BarraCuda core team
**airSpring Version**: 0.3.10 (279 tests, 287 validation checks, 65/65 cross-validation)
**ToadStool HEAD**: `02207c4a` (S62+)
**License**: AGPL-3.0-or-later
**Supersedes**: V001 (archived)

---

## Executive Summary

airSpring v0.3.10 adds FAO-56 dual crop coefficient (Ch 7), cover crop species (5),
no-till mulch reduction (Ch 11), regional ET₀ intercomparison, and a GPU orchestrator
for M-field batched dual Kc. CPU benchmarks prove Rust advantage: 12.7M ET₀/s,
59M dual Kc/s, 64M mulched Kc/s. Deep debt audit confirms: zero unsafe code, zero
mocks in production, all external deps essential, capability-based station discovery
replacing hardcoded values, idiomatic JSON helpers replacing 58 raw `.unwrap()` calls.

**By the numbers:**
- 8 papers reproduced, 306/306 Python + 287/287 Rust checks
- 279 Rust tests + 40 forge tests = 319 tests total
- 65/65 Python↔Rust cross-validation match (tol=1e-5)
- 918 real station-days (Open-Meteo ERA5, 6 Michigan stations)
- 7 GPU orchestrators wired to BarraCuda primitives
- 18 evolution gaps tracked (8 Tier A, 9 Tier B, 1 Tier C)
- 3 bug fixes contributed upstream (TS-001, TS-003, TS-004)
- 40 metalForge tests staged for absorption

---

## Part 1: What's New Since V001

### New Modules

| Module | Description | Checks | CPU Throughput |
|--------|-------------|:------:|----------------|
| `eco::dual_kc` | FAO-56 Ch 7 dual crop coefficient (Kcb+Ke) | 61/61 | 59M days/s |
| `eco::dual_kc` (cover crops) | 5 species: cereal rye, crimson clover, winter wheat, hairy vetch, tillage radish | 40/40 | 64M days/s |
| `eco::dual_kc` (mulched_ke) | No-till mulch reduction (5 residue levels, FAO-56 Ch 11) | included | included |
| `gpu::dual_kc` | M-field batched dual Kc orchestrator (CPU path, Tier B → GPU) | 4 tests | — |
| `testutil::pearson_r` | Raw Pearson correlation for cross-station analysis | — | — |
| `validation::{json_str,json_field,json_array}` | Idiomatic JSON helpers replacing raw `.unwrap()` | — | — |

### New Validation Binaries

| Binary | Checks | Scope |
|--------|:------:|-------|
| `validate_dual_kc` | 61/61 | FAO-56 Ch 7 Kcb+Ke, 10 crops, 11 soils, integration scenarios |
| `validate_cover_crop` | 40/40 | Cover crops, mulch factors, Islam et al., no-till savings |
| `validate_regional_et0` | 61/61 | 6-station intercomparison, Pearson r, CV, geographic consistency |
| `bench_cpu_vs_python` | — | CPU throughput benchmarks (ET₀, dual Kc, mulched Kc) |

### Capability-Based Discovery

`validate_real_data` evolved from hardcoded station list to dynamic discovery:
1. Check `AIRSPRING_STATIONS` env var (comma-separated)
2. Discover stations from CSV filenames in `data_dir`
3. Fall back to `DEFAULT_STATIONS` only if both empty

This pattern should be standard for all Springs. Primal code only has self-knowledge
and discovers other primals at runtime.

---

## Part 2: What airSpring Uses from BarraCuda

### GPU Orchestrators (7 wired, all operational)

| airSpring Module | BarraCuda Primitive | Usage | Provenance |
|-----------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | Batched FAO-56 ET₀ for N station-days | hotSpring `pow_f64` fix (TS-001) |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | Batched depletion update per day | Multi-spring shared shader |
| `gpu::dual_kc::BatchedDualKc` | CPU path (Tier B → pending shader) | M-field batched Ke computation | airSpring v0.3.10 |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | Soil moisture spatial interpolation | wetSpring interpolation |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64::FusedMapReduceF64` | Seasonal sum/mean/max/min for N≥1024 | wetSpring, airSpring TS-004 fix |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats::MovingWindowStats` | IoT sensor stream smoothing (24h window) | wetSpring S28+ environmental |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | Regularized sensor calibration | wetSpring ESN calibration |

### Stats & Validation (6 primitives)

| Primitive | airSpring Integration |
|-----------|-----------------------|
| `stats::pearson_correlation` | R² in `testutil::r_squared` + raw `testutil::pearson_r` |
| `stats::correlation::spearman_correlation` | Nonparametric validation in `testutil::spearman_r` |
| `stats::bootstrap::bootstrap_ci` | RMSE uncertainty in `testutil::bootstrap_rmse` |
| `stats::correlation::std_dev` | Cross-validation integration tests |
| `validation::ValidationHarness` | All 10 validation binaries (absorbed from neuralSpring S59) |

### CPU Benchmark Results (`--release`)

| Operation | N | Throughput | Notes |
|-----------|---|------------|-------|
| ET₀ (FAO-56 PM) | 100,000 | 12.7M station-days/s | `bench_cpu_vs_python` |
| Dual Kc (Kcb+Ke) | 100,000 | 59M days/s | FAO-56 Ch 7 partitioning |
| Mulched Kc | 100,000 | 64M days/s | No-till mulch reduction |
| Reduce (seasonal) | 100,000 | 399M elem/sec | `bench_airspring_gpu` |
| Stream smooth | 8,760 | 32.4M elem/sec | 24h sliding window |
| Kriging | 20→500 | — | Spatial interpolation |
| Ridge regression | 5,000 | R²=1.000 | CPU-only |

---

## Part 3: What airSpring Contributed Upstream

### Bug Fixes (3, all resolved)

| ID | Summary | Severity | Impact |
|----|---------|:--------:|--------|
| **TS-001** | `pow_f64` returns 0.0 for fractional exponents | Critical | All Springs using `batched_elementwise_f64` |
| **TS-003** | `acos_simple`/`sin_simple` low-order approximations | Low | Precision at boundary values |
| **TS-004** | `FusedMapReduceF64` buffer conflict for N≥1024 | High | All Springs using `fused_map_reduce_f64` |

### Domain Knowledge

1. **Dual Kc as a GPU test case**: The FAO-56 dual crop coefficient exercises
   conditional branching (stage 1 vs stage 2 evaporation), clamping (Kc_max),
   and multi-field independence — excellent for batch shader design.

2. **Cover crop diversity**: 5 species with distinct Kcb curves (0.15–0.30)
   test the elementwise shader's ability to handle heterogeneous per-field
   coefficients in batch mode.

3. **No-till mulch physics**: Mulch factor (0.25–1.0) multiplies Ke, reducing
   soil evaporation. The interaction between mulch and Kc_max capping reveals
   edge cases where `ETc` appears identical despite different underlying components.

4. **Capability-based discovery**: Station lists should not be hardcoded.
   The filesystem-first, env-var-override, default-fallback pattern is reusable
   across all Springs for any discoverable configuration.

---

## Part 4: What airSpring Needs Next

### Tier B — Ready to Wire (GPU orchestrator exists, pending shader)

| Need | BarraCuda Primitive | airSpring Purpose | Priority |
|------|--------------------|----|:---:|
| **Dual Kc batch (Ke)** | `batched_elementwise_f64` (op=8) | GPU Ke computation for M-field batching | **HIGH** — orchestrator already wired |
| Sensor batch calibration | `batched_elementwise_f64` (op=5) | Batch SoilWatch 10 VWC calibration | Medium |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | Simpler ET₀ (no humidity/wind) | Low |
| Kc climate adjustment | `batched_elementwise_f64` (op=7) | FAO-56 Eq. 62 crop coefficient | Low |
| Nonlinear curve fitting | `optimize::nelder_mead`, `NelderMeadGpu` | Correction equation fitting | Medium |
| m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring (wetSpring) | Low |

### Tier B — PDE & Numerics (upstream primitives exist)

| Need | BarraCuda Primitive | airSpring Purpose |
|------|--------------------|----|
| 1D Richards equation | `pde::richards::solve_richards` | Unsaturated soil water flow (van Genuchten-Mualem) |
| Tridiagonal solve | `linalg::tridiagonal_solve_f64` | Implicit PDE time-stepping |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Dynamic soil moisture models |

### Tier C — Needs New Primitive

| Need | Description | Complexity |
|------|-------------|:---------:|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs | Low — not GPU |

---

## Part 5: Deep Debt Audit Results

### External Dependencies

All external dependencies in `Cargo.toml` are essential:

| Dependency | Purpose | Rust Alternative? |
|------------|---------|:-:|
| `serde` + `serde_json` | JSON benchmark parsing | No — ecosystem standard |
| `csv` | Station data loading | No — `io::csv_ts` handles domain streaming |
| `chrono` | Date parsing for real data | Possible (manual DOY), but chrono is standard |
| `wgpu` (via barracuda) | GPU dispatch | No — core of ToadStool |

**Verdict**: No dependency bloat. All deps serve clear roles.

### Unsafe Code

Zero `unsafe` blocks in the entire airSpring codebase. All Rust code is safe.
Performance comes from:
- Release-mode optimizations (`--release`)
- Batch processing (amortize overhead over N items)
- Zero-copy JSON parsing where possible
- BarraCuda GPU dispatch for large N

### Mocks in Production

Zero mocks in production code. All mocks are isolated to:
- `#[cfg(test)]` modules in library code
- Integration test files (`tests/`)
- Validation binaries use real algorithms + real benchmarks

### Hardcoded Values → Capability-Based

| Before | After | File |
|--------|-------|------|
| `STATIONS` array (6 hardcoded) | `RuntimeConfig::discover_stations()` | `validate_real_data.rs` |
| Column name `et0_fao56_mm` | Dynamic column lookup | `validate_regional_et0.rs` |

Remaining hardcoded items (acceptable):
- FAO-56 physical constants (Stefan-Boltzmann, etc.) — these are constants of nature
- Benchmark JSON paths (compile-time `include_str!`) — these are test fixtures
- Default tolerance values (1e-5) — configurable via the validation harness

### Large File Cohesion Audit

| File | Lines | Verdict | Rationale |
|------|:-----:|:-------:|-----------|
| `eco::dual_kc` | 777 | **Keep** | Single domain concept (FAO-56 Ch 7), tightly coupled stages |
| `eco::evapotranspiration` | 695 | **Keep** | FAO-56 function chain, each fn depends on predecessors |
| `eco::correction` | 578 | **Keep** | Unified fitting interface, shared error types |
| `eco::soil_moisture` | 450+ | **Keep** | Single domain (Topp eq family + hydraulic properties) |

All large files maintain strong domain cohesion. Splitting would scatter
related functions across files without improving navigability.

### Idiomatic Rust Evolution

| Before | After | Impact |
|--------|-------|--------|
| 58 raw `.unwrap()` calls | `json_str()`, `json_field()`, `json_array()` helpers | Descriptive panic messages |
| `return;` after `v.finish()` | Removed unreachable code | Clean compiler output |
| Old `DailyEt0Input` field names | Updated to current struct API | Type-safe construction |

---

## Part 6: metalForge Absorption Candidates

The `metalForge/forge/` crate (40 tests) contains airSpring-specific primitives
staged for upstream absorption into `barracuda`.

### Metrics Module → `barracuda::stats::metrics`

| Function | Description | Tests |
|----------|-------------|:-----:|
| `rmse` | Root Mean Square Error | 3 |
| `mbe` | Mean Bias Error | 2 |
| `nash_sutcliffe` | Nash-Sutcliffe Efficiency | 3 |
| `index_of_agreement` | Willmott's Index of Agreement | 3 |

### Regression Module → `barracuda::stats::regression`

| Function | Description | Tests |
|----------|-------------|:-----:|
| `fit_linear` | Linear least squares (y = ax + b) | 2 |
| `fit_quadratic` | Quadratic normal equations | 2 |
| `fit_exponential` | Log-linearized exponential | 1 |
| `fit_logarithmic` | Log-linearized logarithmic | 1 |
| `fit_all` | Best-of-four model selection | 1 |

### Hydrology Module → `barracuda::eco` or new domain crate

| Function | Description | Tests |
|----------|-------------|:-----:|
| Richards 1D (staged) | Unsaturated flow — waiting on upstream wiring | — |

### Moving Window → already absorbed

`moving_window_f64` functionality is now provided by upstream
`barracuda::ops::moving_window_stats`. metalForge version deprecated.

---

## Part 7: Cross-Spring Evolution Insights

### What We Learned About Dual Kc on GPU

The dual crop coefficient is an excellent candidate for `batched_elementwise_f64`
because:
1. **Per-field independence**: M fields can be processed in parallel with zero
   inter-field dependencies. The `BatchedDualKc` orchestrator already expresses this.
2. **Ke computation is the bottleneck**: The soil evaporation coefficient involves
   conditional logic (stage 1 vs stage 2), clamping, and division. This is a
   single elementwise operation per field per timestep.
3. **Mulch factor is a simple multiplier**: `Ke_mulch = Ke × mulch_factor` adds
   negligible GPU overhead but doubles the agronomic value.

**Recommendation**: Add op=8 to `batched_elementwise_f64.wgsl` for Ke computation.
Inputs: `ET0, Kcb, Kc_max, few, De_prev, TEW, REW, P, I, mulch_factor`.
Output: `Ke, De_new`. The switch-case pattern in the shader is clean.

### What We Learned About Cover Crops

Cover crop diversity (5 species × 5 residue levels = 25 configurations) is a
natural batch workload. A farmer running 50 fields with different cover crop
histories would dispatch 50 parallel Ke computations per timestep.

This is the Penny Irrigation use case: $200 sensor + Open-Meteo + $600 GPU
running batched dual Kc for a whole farm cooperative.

### What We Learned About Capability-Based Design

Hardcoded station lists break when:
- A new station is added to the data directory
- A station file is temporarily unavailable
- The same binary runs on different farms

The filesystem-first discovery pattern solves this:
```
fn discover() -> Vec<Station> {
    if let Ok(env) = std::env::var("STATIONS") { parse(env) }
    else if let Ok(files) = std::fs::read_dir(data_dir) { scan(files) }
    else { DEFAULTS }
}
```

This pattern should become a BarraCuda utility: `barracuda::config::discover<T>()`.

### Recommendations for ToadStool Evolution

1. **Absorb `eco::dual_kc` Ke computation as op=8**: The math is well-defined
   (FAO-56 Eqs 69-74), extensively tested (61+40+4 checks), and naturally
   batch-parallel. airSpring's `gpu::dual_kc` orchestrator is ready to wire.

2. **Absorb hydrology metrics (forge::metrics)**: NSE, IA, MBE are standard
   earth science metrics. They're pure functions, no dependencies, 11 tests.

3. **Absorb regression toolkit (forge::regression)**: Linear, quadratic,
   exponential, logarithmic, best-of-four selection. Complements existing
   `linalg::ridge::ridge_regression`. 7 tests.

4. **Consider a `config::discover` utility**: The filesystem-first, env-override,
   default-fallback pattern is reusable across all Springs. A generic
   `discover_resources<T>()` would reduce boilerplate.

5. **Consider `pearson_r` in stats**: airSpring added `testutil::pearson_r` wrapping
   `stats::pearson_correlation` for direct use. If other Springs need raw Pearson r
   (not R²), it could be a first-class export.

6. **Richards equation wiring**: The PDE solver is Tier B (upstream available).
   airSpring will wire it when ready. The key integration point is translating
   `eco::soil_moisture` textures (Ksat, θs, θr, α, n) to `SoilParams` for the
   van Genuchten-Mualem constitutive model.

---

## Part 8: Compute Pipeline — CPU → GPU → metalForge

### Current State

```
Layer 1 (CPU):     ██████████████████████████████████████████ COMPLETE (287/287)
Layer 2 (GPU):     ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7/16 wired
Layer 3 (metalForge): ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ staged (40 tests)
```

### Pathway to Pure GPU

1. **Now**: All 8 papers validated via Python + Rust CPU. CPU baselines are the
   correctness reference for everything downstream.

2. **Next**: Wire Tier B GPU primitives (dual Kc batch op=8, sensor calibration
   op=5, Hargreaves op=6, Kc adjustment op=7). Each GPU result must match CPU
   baseline within tolerance.

3. **Then**: ToadStool streaming (unidirectional) reduces dispatch overhead.
   Batch M fields × T timesteps in a single GPU submission. This is where
   GPU throughput exceeds CPU for practical workloads.

4. **Finally**: metalForge mixed hardware — CPU for control flow and small-N
   operations, GPU for batch math, future NPU for real-time IoT inference.

### Validation Chain

```
Paper benchmark → Python control → Rust CPU → Rust GPU → metalForge
       ↕               ↕              ↕           ↕          ↕
   digitized      306/306 PASS    287/287     match CPU    match GPU
```

Every layer must reproduce the same answers. This is the sovereign compute
promise: you can verify the math at every stage.

---

## Summary

airSpring v0.3.10 demonstrates that precision agriculture computation (FAO-56
dual Kc, cover crops, no-till mulch, regional ET₀) runs correctly and fast on
the BarraCuda/ToadStool stack. The deep debt audit confirms clean code: zero
unsafe, zero production mocks, minimal hardcoding, idiomatic Rust throughout.

The path forward is clear: GPU-accelerate the validated CPU modules, then
demonstrate mixed-hardware scheduling via metalForge. The dual Kc batch
orchestrator is wired and waiting for ToadStool's op=8 shader.

The path to Penny Irrigation: a $200 IoT sensor node, free Open-Meteo weather
data, and a $600 GPU running BarraCuda's f64 shaders — sovereign precision
irrigation for any farmer, anywhere, with no institutional access required.

---

*AGPL-3.0-or-later. airSpring v0.3.10, ToadStool HEAD `02207c4a`.
18 evolution gaps (8A+9B+1C). Deep debt audit: zero unsafe, zero production mocks,
capability-based discovery, idiomatic JSON helpers. metalForge candidates
(metrics, regression, hydrology) NOT yet absorbed.*
