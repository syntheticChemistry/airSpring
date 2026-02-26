# airSpring → ToadStool Handoff V009: Full Evolution & Absorption Roadmap

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — v0.4.2, 601 tests, 18 binaries, 69x CPU speedup)
**To**: ToadStool / BarraCuda core team + all Springs
**Supersedes**: V008 (archived)
**ToadStool PIN**: `02207c4a` (S62+ — 170 commits, 46 absorptions, 4,224+ core tests, 758 WGSL shaders)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring is the ecological sciences validation Spring — precision agriculture,
irrigation science, soil physics, and environmental systems. This handoff
documents our complete BarraCuda integration state, what we've contributed
back, what ToadStool should absorb next, and cross-spring evolution learnings
relevant to the entire ecosystem.

**By the numbers:**

| Metric | Value |
|--------|-------|
| Total Rust tests | **601** (433 lib + 115 integration + 53 forge) |
| Library coverage | **97.55%** (llvm-cov) |
| Python baselines | **400/400 PASS** (13 experiments incl. yield response + CW2D) |
| Validation binaries | **18/18 PASS** (439 quantitative checks) |
| Cross-validation | **75/75 MATCH** (Python↔Rust, tol=1e-5) |
| GPU orchestrators | **8** wired to ToadStool primitives |
| BarraCuda primitives consumed | **14** |
| ToadStool issues discovered + resolved | **4** (TS-001 through TS-004, all resolved S54) |
| Upstream contributions | Richards PDE (S40), 3 bug fixes (S54) |
| Zero unsafe / zero unwrap in lib / zero clippy warnings / zero files >1000 lines |

**Key findings since V008:**
- Corrected stale shader counts across all docs (608→**758** actual WGSL shaders)
- All 601 tests confirmed passing against ToadStool HEAD `02207c4a`
- Cross-spring evolution test suite validates provenance chains
- 4 metalForge modules remain ready for absorption (42 tests, pure arithmetic)

---

## Part 1: What airSpring Contributed to the Ecosystem

### 1.1 Upstream Contributions (Already Absorbed)

| Contribution | Absorbed Into | Session | Impact |
|-------------|--------------|---------|--------|
| Richards equation (VG-Mualem PDE) | `barracuda::pde::richards` | S40 | Any Spring doing unsaturated flow gets this |
| `pow_f64` fractional exponent fix (TS-001) | `math_f64.wgsl` | S54 (H-011) | All Springs using exponential math |
| `acos` precision boundary fix (TS-003) | `math_f64.wgsl` | S54 (H-012) | All Springs using trig in f64 shaders |
| `FusedMapReduceF64` N≥1024 fix (TS-004) | `fused_map_reduce_f64.wgsl` | S54 (H-013) | All Springs using reduce with large buffers |

### 1.2 Domain Learnings for the Ecosystem

**Tolerance patterns**: airSpring's domain (agricultural science) requires
tolerances that look small but have real physical meaning. FAO-56 Table 2.3
rounds saturation vapour pressure to 3 decimals (kPa), so our tolerance of
0.01 kPa isn't arbitrary — it's the publication precision. We created
`barracuda/src/tolerances.rs` (21 named constants) as a pattern other
Springs might adopt. Each constant has a `justification` field explaining
*why* that tolerance, not just *what* it is.

**IoT streaming**: Agricultural sensor data is continuous, high-frequency,
and noisy. Our `io::csv_ts` streaming parser processes 11,300+ records/sec
from SoilWatch 10 sensors. The pattern: stream rows, compute rolling
statistics (24-hr window), flag anomalies — all without buffering the entire
file. The `MovingWindowStats` primitive from wetSpring S28+ is central here.

**Real-data validation**: airSpring validates against 918 real station-days
from 6 Michigan agricultural weather stations (Open-Meteo ERA5, free API).
R²=0.967 against independent ET₀ computation. This is a pattern other
Springs should adopt — synthetic benchmarks prove math; real-data validation
proves the pipeline.

---

## Part 2: BarraCuda Integration Map

### 2.1 Primitives Consumed (14)

| # | Primitive | Module | Origin Spring | airSpring Usage |
|:-:|-----------|--------|--------------|-----------------|
| 1 | `batched_elementwise_f64` (op=0) | `ops` | Multi-spring | ET₀ computation (GPU-FIRST) |
| 2 | `batched_elementwise_f64` (op=1) | `ops` | Multi-spring | Water balance step |
| 3 | `kriging_f64::KrigingF64` | `ops` | wetSpring | Soil moisture spatial interpolation |
| 4 | `fused_map_reduce_f64` | `ops` | wetSpring | Seasonal statistics (min/max/mean/total) |
| 5 | `moving_window_stats` | `ops` | wetSpring S28+ | IoT stream smoothing |
| 6 | `pde::richards::solve_richards` | `pde` | airSpring→upstream S40 | 1D unsaturated flow |
| 7 | `optimize::nelder_mead` | `optimize` | neuralSpring | Isotherm fitting |
| 8 | `optimize::multi_start_nelder_mead` | `optimize` | neuralSpring | Global isotherm fitting (LHS) |
| 9 | `linalg::ridge::ridge_regression` | `linalg` | wetSpring ESN | Sensor calibration correction |
| 10 | `validation::ValidationHarness` | `validation` | neuralSpring S59 | All 16 validation binaries |
| 11 | `stats::pearson_correlation` | `stats` | Shared | Model evaluation |
| 12 | `stats::spearman_correlation` | `stats` | Shared | Rank correlation |
| 13 | `stats::bootstrap_ci` | `stats` | Shared | Confidence intervals |
| 14 | `tolerances::Tolerance` / `check()` | `tolerances` | neuralSpring S52 | 21 domain-specific constants |

### 2.2 GPU Orchestrators (8 Wired)

| Orchestrator | BarraCuda Dispatch | Throughput | Cross-Spring Provenance |
|-------------|-------------------|-----------|------------------------|
| `BatchedEt0` | `fao56_et0_batch()` | 12.5M ops/s @ N=10K | hotSpring `pow_f64` fix enables VG math |
| `BatchedWaterBalance` | `water_balance_batch()` | — | Multi-spring `batched_elementwise_f64` |
| `KrigingInterpolator` | `KrigingF64::interpolate()` | 26 µs/solve | wetSpring joint development |
| `SeasonalReducer` | `FusedMapReduceF64::run()` | 395M elem/s | wetSpring origin + airSpring TS-004 fix |
| `StreamSmoother` | `MovingWindowStats::compute()` | 31.7M elem/s | wetSpring S28+ environmental |
| `BatchedRichards` | `solve_richards()` | 72 sims/s | airSpring→ToadStool S40, hotSpring df64 |
| `fit_*_nm/global` | `nelder_mead()` / `multi_start()` | 175K fits/s | neuralSpring optimizer |
| `fit_ridge` | `ridge_regression()` | R²=1.000 | wetSpring ESN ridge |

### 2.3 Cross-Spring Shader Provenance

airSpring benefits from a shader ecosystem built by 4 Springs:

| Shader Family | Origin | What airSpring Gets | Sessions |
|--------------|--------|---------------------|----------|
| `math_f64.wgsl` (pow, exp, log, trig) | hotSpring | VG retention curves, atmospheric pressure, solar geometry | S18+ |
| `kriging_f64.wgsl` | wetSpring | Soil moisture spatial interpolation | S28+ |
| `fused_map_reduce_f64.wgsl` | wetSpring | Seasonal statistics (min/max/mean/sum) | S28+, fixed S54 |
| `moving_window.wgsl` | wetSpring | IoT 24-hr rolling statistics | S28+ |
| `nelder_mead.wgsl` + `multi_start` | neuralSpring | Isotherm fitting (global optimizer) | S52+ |
| `pde_richards.wgsl` | airSpring→upstream | 1D unsaturated flow (VG-Mualem) | S40 |

18 integration tests in `tests/cross_spring_evolution.rs` validate these provenance chains.

---

## Part 3: Pending Absorption — 4 metalForge Modules

These modules are ready for absorption into BarraCuda. All are pure arithmetic
on `&[f64]`, zero dependencies beyond `std`, fully tested and validated.

### 3.1 `forge::metrics` → `barracuda::stats::metrics`

| Function | Purpose | Validation |
|----------|---------|------------|
| `rmse(obs, sim)` | Root mean square error | Dong 2020, 918 station-days |
| `mbe(obs, sim)` | Mean bias error | Dong 2020, 918 station-days |
| `nash_sutcliffe(obs, sim)` | Nash-Sutcliffe efficiency | Nash & Sutcliffe (1970) |
| `index_of_agreement(obs, sim)` | Willmott's IA | Willmott (1981) |
| `coefficient_of_determination(obs, sim)` | R² (alias of NSE) | Standard |

**Tests**: 11. **Why it matters to all Springs**: Universal model evaluation
metrics. Currently hotSpring implements chi², wetSpring implements IA/NSE,
airSpring implements all five — these should be shared upstream.

### 3.2 `forge::regression` → `barracuda::stats::regression`

| Function | Model | Method |
|----------|-------|--------|
| `fit_linear` | y = a·x + b | Normal equations |
| `fit_quadratic` | y = a·x² + b·x + c | Cramer's rule |
| `fit_exponential` | y = a·exp(b·x) | Log-linearized LS |
| `fit_logarithmic` | y = a·ln(x) + b | Linearized LS |

**Tests**: 11. **Why it matters**: Sensor correction is cross-domain. Any IoT
pipeline in any Spring needs linear/exponential/logarithmic fitting.

### 3.3 `forge::moving_window_f64` → `barracuda::ops::moving_window_stats_f64`

CPU f64 moving window (mean, variance, min, max). Upstream `moving_window_stats`
operates in f32 on GPU; agricultural sensor data needs f64 for sub-degree
temperature and sub-percent soil moisture precision.

**Tests**: 7.

### 3.4 `forge::hydrology` → `barracuda::ops::hydrology`

| Function | Reference |
|----------|-----------|
| `hargreaves_et0(ra, tmax, tmin)` | Hargreaves & Samani (1985) |
| `hargreaves_et0_batch(ra, tmax, tmin)` | Batched convenience |
| `crop_coefficient(kc_prev, kc_next, day, stage_len)` | FAO-56 Ch. 6 |
| `soil_water_balance(theta, precip, irrig, etc, fc)` | FAO-56 Ch. 8 |

**Tests**: 13. **Why it matters**: Climate-driven agriculture primitives useful
for any environmental Spring.

**Total**: 42 tests, pure arithmetic. Full signatures and post-absorption
rewiring plan in `metalForge/ABSORPTION_MANIFEST.md`.

---

## Part 4: Action Items for ToadStool/BarraCuda

### P0 — Blocking

*None. airSpring is not blocked. All 601 tests pass.*

### P1 — High Value

| # | Item | Since | Impact |
|:-:|------|:-----:|--------|
| 1 | **Absorb 4 metalForge modules** (§3) | V006 | 42 tests, pure arithmetic — every Spring benefits from metrics, regression, hydrology |
| 2 | **`crank_nicolson_f64`** | V007 | Exists in f32 (S46+); Richards PDE requires f64 for Picard convergence |
| 3 | **Named constants in `pde::richards`** | V007 | 8 VG constants from airSpring for cross-Spring consistency |
| 4 | **Preallocation in `pde::richards`** | V007 | Picard iteration buffers outside solve loop |
| 5 | **Re-export `spearman_correlation` from `stats/mod.rs`** | V008 | Function exists in `stats::correlation::spearman_correlation` but isn't in the `pub use` block |
| 6 | **`[lints.clippy]` in barracuda Cargo.toml** | V008 | Modern Rust pattern (stable 1.74+). airSpring/forge already migrated. Ecosystem consistency |

### P2 — Nice to Have

| # | Item | Impact |
|:-:|------|--------|
| 7 | Batch PDE dispatch (`pde::richards::solve_batch_gpu`) | M soil columns in parallel |
| 8 | `fma_f64` WGSL instruction | `mul_add()` CPU parity for numerical consistency |
| 9 | Dual Kc GPU shader (`batched_elementwise_f64` op=8) | Multi-field crop coefficients |
| 10 | Sensor calibration batch op | `batched_elementwise_f64` op=5 for SoilWatch 10 |
| 11 | Hargreaves ET₀ batch op | `batched_elementwise_f64` op=6 (simpler than PM) |

### P3 — Research

| # | Item | Impact |
|:-:|------|--------|
| 12 | `unified_hardware` integration | metalForge → `HardwareDiscovery` + `ComputeScheduler` |
| 13 | Surrogate learning | Richards PDE → neural surrogate for real-time irrigation |
| 14 | `OdeSystem` for soil-plant dynamics | Root water uptake via generic ODE (S51 framework) |

---

## Part 5: Cross-Spring Evolution Observations

These observations from airSpring's experience may benefit all Springs and
ToadStool core development:

### 5.1 The `pow_f64` Fix Was the Most Impactful

TS-001 (`pow_f64` fractional exponent) was discovered during van Genuchten
soil retention curve validation. VG uses `h^n` where `n` is typically 1.2–2.0
(non-integer). Before the fix, GPU results diverged by >1% from CPU. After
S54, GPU matches CPU to 1e-10. This fix benefits every Spring doing
exponential math — which is nearly all of them.

### 5.2 Tolerance Centralization Prevents Drift

Before `barracuda::tolerances` (S52), each Spring had its own ad-hoc tolerance
constants scattered across test files. airSpring formalized 21 domain-specific
tolerances with justification fields. When a tolerance changes, the justification
forces you to explain *why*. Recommend all Springs adopt this pattern.

### 5.3 Cross-Validation Catches What Unit Tests Miss

airSpring's 75/75 Python↔Rust cross-validation has caught two issues that
passed all unit tests:
1. A sign error in `slope_vapour_pressure` that canceled in single-point tests
   but accumulated across season simulations
2. A rounding difference in Richards boundary conditions that appeared only
   when comparing 50-node profiles

Recommend every Spring that has Python baselines implement cross-validation
on intermediate values, not just final outputs.

### 5.4 Shader Provenance Matters

When `FusedMapReduceF64` started returning wrong results for N≥1024 (TS-004),
tracing the shader provenance (wetSpring origin → ToadStool absorption → airSpring
consumption) made debugging tractable. Without the provenance chain, we'd have
searched the wrong codebase. Recommend all GPU modules document which Spring
originated the underlying shader.

### 5.5 ToadStool Shader Count: 758 (Not 650)

We counted 758 `.wgsl` files in ToadStool HEAD `02207c4a`. The README says
"650+". Recommend updating ToadStool's own documentation.

---

## Part 6: What "Done" Looks Like for airSpring

airSpring's validation mission is **complete for the current paper set** (11
experiments, 8 Dong publications, FAO-56 + 3 supplemental). Next steps require
new field data from Dong Lab (establishing in 2026):

| Milestone | Trigger | What We Need from ToadStool |
|-----------|---------|---------------------------|
| Multi-sensor network | Dong Lab field data (2026) | `batched_elementwise_f64` ops 5-7 |
| Forecast-integrated scheduling | Open-Meteo forecast API | HTTP/JSON client (Tier C) |
| Pure GPU Richards | Convergence testing | `crank_nicolson_f64` (P1 item #2) |
| Penny Irrigation demo | All above | `unified_hardware` integration |

Until then, airSpring remains a stable consumer of BarraCuda and a source of
domain-specific contributions for the ecosystem.

---

## Part 7: Artifacts

| Document | Location |
|----------|----------|
| This handoff | `wateringHole/handoffs/AIRSPRING_V009_EVOLUTION_HANDOFF_FEB25_2026.md` |
| Previous handoff | `wateringHole/handoffs/archive/AIRSPRING_V008_*.md` |
| Evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| Absorption manifest | `metalForge/ABSORPTION_MANIFEST.md` |
| Cross-spring evolution | `specs/CROSS_SPRING_EVOLUTION.md` |
| Evolution gaps (code) | `barracuda/src/gpu/evolution_gaps.rs` |
| Cross-spring tests (code) | `barracuda/tests/cross_spring_evolution.rs` |
| Tolerances (code) | `barracuda/src/tolerances.rs` |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001 | 2026-02-25 | Initial GPU handoff (v0.3.8) |
| V002 | 2026-02-25 | Dual Kc, cover crops, deep debt (v0.3.10) |
| V003 | 2026-02-25 | Richards + isotherm GPU wiring (v0.4.0) |
| V004 | 2026-02-25 | ToadStool S62 sync, multi-start NM (v0.4.1) |
| V005 | 2026-02-25 | Complete status, GPU integration tests (v0.4.2) |
| V006 | 2026-02-25 | Deep audit pass 1, 96.84% coverage (v0.4.2+) |
| V007 | 2026-02-25 | Lint migration, 97.58% coverage, 555 tests (v0.4.2, archived) |
| V008 | 2026-02-25 | ToadStool S62 sync: revalidated, cross-spring provenance (archived) |
| **V009** | **2026-02-25** | **Full evolution handoff: 758 shaders, updated action items, cross-spring observations** |

---

*End of V009 handoff. Direction: airSpring → ToadStool (unidirectional).
All 601 tests pass against ToadStool HEAD `02207c4a`. 758 WGSL shaders counted.
Next handoff: V010 after metalForge absorption or new field data arrives.*
