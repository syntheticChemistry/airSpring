# airSpring V003: GPU Wiring + Absorption + Evolution Handoff

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — Ecological & Agricultural Sciences)
**To**: ToadStool/BarraCuda core team
**airSpring Version**: 0.4.0 (319 tests, 75/75 cross-validation, 8 GPU orchestrators)
**ToadStool HEAD**: `02207c4a` (S62+)
**License**: AGPL-3.0-or-later
**Supersedes**: V002 (archived)

---

## Executive Summary

airSpring v0.4.0 wires three new upstream primitives (Richards PDE, Nelder-Mead
optimization, tridiagonal solver), adds two new GPU orchestrator modules
(`gpu::richards`, `gpu::isotherm`), expands cross-validation from 65→75 values,
stages two new metalForge absorption modules, and adds CPU benchmarks for all
new scientific modules.

**By the numbers:**
- 11 papers reproduced, 344/344 Python + 319 Rust tests
- 75/75 Python↔Rust cross-validation match (tol=1e-5)
- 8 GPU orchestrators wired to BarraCuda primitives (was 7)
- 53 metalForge tests staged for absorption (6 modules, 2 already absorbed upstream)
- 20 evolution gaps tracked (8 Tier A, 11 Tier B, 1 Tier C)
- 3 bug fixes contributed upstream (TS-001, TS-003, TS-004) — all resolved
- CPU benchmarks: 12.7M ET₀/s, 36.5M VG θ/s, 57M Langmuir fits/s

---

## Part 1: What's New Since V002

### New Modules

| Module | Description | Tests | CPU Throughput |
|--------|-------------|:-----:|----------------|
| `eco::richards` | 1D Richards equation, van Genuchten-Mualem, implicit Euler + Picard | 6 | 3,600 sims/s |
| `eco::isotherm` | Langmuir/Freundlich models, linearized LS + 1D grid | 10 | 57M Langmuir fits/s |
| `gpu::richards` | Bridges `eco::richards` ↔ `barracuda::pde::richards` | 4 | — |
| `gpu::isotherm` | Bridges `eco::isotherm` ↔ `barracuda::optimize::nelder_mead` | 6 | — |

### New Validation Binaries

| Binary | Checks | Scope |
|--------|:------:|-------|
| `validate_richards` | 15/15 | VG retention/conductivity, infiltration, drainage, mass balance, steady-state flux |
| `validate_biochar` | 14/14 | Langmuir/Freundlich fitting, R², RL factor, residuals |
| `validate_long_term_wb` | 11/11 | 60-year Wooster OH, ET₀ range, mass balance, climate trends |

### Cross-Validation Expansion (65→75 values)

Added to both `scripts/cross_validate.py` and `barracuda/src/bin/cross_validate.rs`:
- Van Genuchten θ: h=0, h=-10, h=-100 for sand (3 values)
- Van Genuchten K: h=0, h=-10 for sand (2 values)
- Langmuir predictions: Ce=1,10,50,100 (4 values)
- Freundlich predictions: Ce=1,10,50,100 (4 values)
- Langmuir separation factor: RL at C0=100 (1 value)

All 10 new values match Python↔Rust within 1e-5.

---

## Part 2: What airSpring Uses from BarraCuda (8 orchestrators)

### GPU Orchestrators

| airSpring Module | BarraCuda Primitive | Status | Notes |
|-----------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** | hotSpring pow_f64 fix |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** | Multi-spring shared |
| `gpu::dual_kc::BatchedDualKc` | CPU path (Tier B → pending shader) | CPU ready | airSpring v0.3.10 |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **Integrated** | wetSpring |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** | TS-004 fix |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **Wired** | wetSpring S28+ |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | **Wired** (v0.4.0) | Crank-Nicolson + Picard |
| `gpu::isotherm::fit_*_nm` | `optimize::nelder_mead` | **Wired** (v0.4.0) | Nonlinear LS |

### Stats & Validation (6 primitives, unchanged)

| Primitive | airSpring Integration |
|-----------|-----------------------|
| `stats::pearson_correlation` | R² in `testutil::r_squared` + raw `testutil::pearson_r` |
| `stats::spearman_correlation` | Nonparametric validation in `testutil::spearman_r` |
| `stats::bootstrap_ci` | RMSE uncertainty in `testutil::bootstrap_rmse` |
| `stats::std_dev` | Cross-validation integration tests |
| `linalg::ridge::ridge_regression` | `eco::correction::fit_ridge` (sensor calibration) |
| `validation::ValidationHarness` | All 16 binaries |

### CPU Benchmarks (v0.4.0 additions marked with *)

| Operation | N | Throughput | Binary |
|-----------|---|------------|--------|
| ET₀ (FAO-56 PM) | 1M | 12.2M station-days/s | `bench_cpu_vs_python` |
| Dual Kc (Kcb+Ke) | 3650 | 59M days/s | `bench_cpu_vs_python` |
| Mulched Kc | 3650 | 64M days/s | `bench_cpu_vs_python` |
| *VG θ retention | 100K | 36.5M evals/s | `bench_cpu_vs_python` |
| *Richards 1D (20 nodes) | 10 steps | 3,618 sims/s | `bench_cpu_vs_python` |
| *Langmuir fit (9 pts) | 9 | 57M fits/s | `bench_cpu_vs_python` |
| *Freundlich fit (9 pts) | 9 | 1.2M fits/s | `bench_cpu_vs_python` |

---

## Part 3: What airSpring Contributed Upstream

### Bug Fixes (3, all resolved — same as V002)

| ID | Summary | Severity | Impact |
|----|---------|:--------:|--------|
| **TS-001** | `pow_f64` returns 0.0 for fractional exponents | Critical | All Springs |
| **TS-003** | `acos_simple`/`sin_simple` low-order approximations | Low | Precision |
| **TS-004** | `FusedMapReduceF64` buffer conflict for N≥1024 | High | All Springs |

### Domain Knowledge (NEW in v0.4.0)

1. **Richards PDE unit mismatch**: airSpring's `eco::richards` uses Ks in cm/day
   (agronomic convention), upstream `pde::richards` uses cm/s (physics convention).
   `gpu::richards::to_barracuda_params()` bridges this with `ks / 86_400.0`.
   Recommendation: document expected units prominently in `SoilParams` doc comment.

2. **Implicit Euler vs Crank-Nicolson**: airSpring's local solver (implicit Euler +
   under-relaxation ω=0.2) and upstream's solver (Crank-Nicolson) produce different
   transient solutions but converge to the same steady state. Both are physically
   correct; the difference is method accuracy vs stability tradeoff.

3. **Linearized vs Nelder-Mead isotherm fitting**: Linearized LS (Ce/qe vs Ce) is
   100x faster but can produce different parameters than nonlinear LS for
   ill-conditioned data. The Nelder-Mead path through `barracuda::optimize` matches
   scipy.curve_fit more closely. Recommendation: `NelderMeadGpu` would enable
   batch fitting of 1000s of isotherms simultaneously for field mapping.

4. **Van Genuchten as a GPU primitive**: The θ(h) and K(h) functions are pure
   element-wise operations (no data dependencies between nodes). Perfect for
   `batched_elementwise_f64` as a new op. airSpring computes 36.5M evals/s on CPU;
   GPU batch would scale linearly with N for spatial soil moisture mapping.

---

## Part 4: What airSpring Needs Next

### Tier B — Ready to Wire (GPU orchestrator exists, pending shader/refinement)

| Need | BarraCuda Primitive | airSpring Purpose | Priority |
|------|--------------------|----|:---:|
| **Dual Kc batch (Ke)** | `batched_elementwise_f64` (op=8) | GPU Ke computation for M-field batching | **HIGH** |
| **VG θ/K batch** | `batched_elementwise_f64` (new op) | Element-wise van Genuchten for spatial mapping | **HIGH** |
| Sensor batch calibration | `batched_elementwise_f64` (op=5) | Batch SoilWatch 10 VWC | Medium |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | Simpler ET₀ (no humidity/wind) | Low |
| Kc climate adjustment | `batched_elementwise_f64` (op=7) | FAO-56 Eq. 62 | Low |
| **Batch Nelder-Mead** | `NelderMeadGpu` | Batch isotherm fitting for spatial mapping | Medium |

### Tier B — PDE & Numerics

| Need | BarraCuda Primitive | Status |
|------|--------------------|----|
| Richards PDE (GPU) | `pde::richards` WGSL shader | Wired on CPU; WGSL van_genuchten_f64 exists |
| Tridiagonal solve (GPU) | `linalg::tridiagonal_solve_f64` | Available; useful for batched PDE |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Available for dynamic soil models |

### Tier C — Needs New Primitive

| Need | Description |
|------|-------------|
| HTTP/JSON data client | Open-Meteo, NOAA CDO APIs (not GPU, but needed) |

---

## Part 5: metalForge Absorption Candidates (6 modules)

### Already Absorbed Upstream

| Module | metalForge Location | Upstream Location | Status |
|--------|--------------------|----|---|
| `van_genuchten` | `forge/src/van_genuchten.rs` | `barracuda::pde::richards::SoilParams` | **ABSORBED** |
| `isotherm` (fitting) | `forge/src/isotherm.rs` | `barracuda::optimize::nelder_mead` | **WIRED** via gpu::isotherm |

### Ready for Absorption (4 modules, 43 tests)

| Module | Signature | Tests | Provenance |
|--------|-----------|:-----:|------------|
| `metrics` | `rmse(o,s) → Result<f64>`, `mbe`, `nse`, `ia`, `r2` | 11 | Dong 2020 + FAO-56 |
| `regression` | `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic` → `FitResult` | 11 | Dong 2020 corrections |
| `moving_window_f64` | `moving_window_stats(data, window) → Option<...>` | 7 | wetSpring S28+ f64 |
| `hydrology` | `hargreaves_et0`, `crop_coefficient`, `soil_water_balance` | 13 | FAO-56 Ch 8 |

**Absorption target layout**:
- `metrics` → `barracuda::stats::metrics`
- `regression` → `barracuda::stats::regression`
- `moving_window_f64` → `barracuda::ops::moving_window_stats_f64`
- `hydrology` → `barracuda::ops::hydrology`

See `metalForge/ABSORPTION_MANIFEST.md` for full signatures, provenance, and rewiring plan.

---

## Part 6: Compute Pipeline Per Paper

Every completed experiment follows the evolution path. Status:

| Exp | Paper | Python | BarraCuda CPU | BarraCuda GPU | metalForge |
|:---:|-------|:------:|:-------------:|:-------------:|:----------:|
| 001 | FAO-56 PM ET₀ | 64/64 | 31/31 | `BatchedEt0` **GPU-FIRST** | — |
| 002 | Dong 2020 soil sensors | 36/36 | 26/26 | `fit_ridge` wired | — |
| 003 | Dong 2024 IoT irrigation | 24/24 | 11/11 | `StreamSmoother` wired | — |
| 004 | FAO-56 Ch 8 water balance | 18/18 | 13/13 | `BatchedWaterBalance` **GPU-STEP** | — |
| 005 | Real data 918 days | R²=0.967 | 23/23 | All orchestrators | Future |
| 006 | Richards equation | 14/14 | 15/15 | `BatchedRichards` **wired** | Future |
| 007 | Biochar isotherms | 14/14 | 14/14 | `fit_*_nm` **wired** | Future |
| 009 | FAO-56 Ch 7 dual Kc | 63/63 | 61/61 | `BatchedDualKc` Tier B | Future |
| 010 | Regional ET₀ 6 stations | 61/61 | 61/61 | `BatchedEt0` at scale | Future |
| 011 | Cover crops + no-till | 40/40 | 40/40 | `BatchedDualKc` + mulch | Future |
| 015 | 60-year water balance | 10/10 | 11/11 | `BatchedEt0` + `BatchedWB` | Future |

**Evolution path for each paper**: Python control → BarraCuda CPU (validated) → BarraCuda GPU (wired) → metalForge (mixed hardware dispatch).

---

## Part 7: Lessons Learned for ToadStool Evolution

### Unit Convention Recommendation

Different scientific domains use different units for the same quantity:
- Hydraulic conductivity: cm/day (agronomy), cm/s (physics), m/s (SI)
- Pressure head: cm (agronomy), Pa (physics), bar (engineering)

**Recommendation**: Document expected units in every `SoilParams`-like struct's
doc comments. Consider adding unit conversion functions to the `pde` module,
or adopting a units crate pattern for compile-time safety.

### Batch VG as a New Op

Van Genuchten θ(h) and K(h) are embarrassingly parallel:
- No inter-element dependencies
- Pure arithmetic (pow, clamp, division)
- Perfect for `batched_elementwise_f64` as op=9/10
- 36.5M CPU evals/s → expect 1B+ GPU evals/s
- Enables real-time 3D soil moisture visualization

### NelderMeadGpu for Batch Fitting

`gpu::isotherm` currently uses CPU Nelder-Mead. For field-scale mapping
(fitting isotherms at 100+ sampling points), `NelderMeadGpu` batch dispatch
would be ideal. Each fitting problem is independent — perfect for GPU.

### Absorption Priority

If ToadStool can absorb one metalForge module, `metrics` is the highest impact:
every Spring uses RMSE, MBE, NSE, IA, R². The signatures are stable, tests
are comprehensive (11), and the validation provenance (Dong 2020 + FAO-56)
is solid. After that, `hydrology` provides Hargreaves ET₀ which benefits
from GPU batch dispatch at op=6.

---

## Part 8: Cross-Spring Evolution Insights

### What airSpring Learned from Other Springs

| From | What | How airSpring Uses It |
|------|------|----------------------|
| hotSpring | Write → Absorb → Lean pattern | Exact pattern for metalForge |
| hotSpring | TS-001 pow_f64 fix | Enables atmospheric pressure in GPU ET₀ |
| wetSpring | kriging_f64 | Spatial soil moisture interpolation |
| wetSpring | moving_window_stats | IoT sensor stream smoothing |
| wetSpring | ridge_regression | Sensor calibration pipeline |
| neuralSpring | ValidationHarness | All 16 binaries |
| neuralSpring | Surrogate MLP | Future: fast inner loop for irrigation scheduling |

### What Other Springs Can Learn from airSpring

1. **Capability-based discovery** (station lists via env var → filesystem → default)
2. **Cross-validation harness** (75 intermediate values, JSON diff)
3. **Multi-decade open data** (Open-Meteo ERA5 60+ years, zero cost)
4. **Linearized + NM two-stage fitting** (fast initial guess → robust refinement)
5. **Van Genuchten as universal soil primitive** (reusable across wetSpring, groundSpring)

---

*End of V003 handoff. Direction: airSpring → ToadStool (unidirectional).
Supersedes V002 (archived). Next handoff: V004 after GPU validation of Richards
and full metalForge mixed-hardware demonstration.*
