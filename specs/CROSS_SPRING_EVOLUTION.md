# Cross-Spring Shader Evolution — airSpring Provenance

**Updated**: February 26, 2026 (v0.4.2, ToadStool HEAD `17932267`)

## Summary

ToadStool's BarraCuda runtime contains **774 WGSL shaders** across 41+ categories,
built through **46 cross-spring absorptions** (sessions S51-S57). Each Spring
contributes domain-specific GPU primitives that benefit the entire ecosystem.

airSpring uses **5 shared shaders** directly and contributed **3 critical fixes**
(TS-001, TS-003, TS-004) that improved precision and stability for all Springs.

---

## Shader Provenance by Spring

### hotSpring — Precision Physics (56 shaders)

Domain: Lattice QCD, nuclear structure, high-fidelity physics simulation.

| Category | Shaders | Examples |
|----------|---------|----------|
| **Lattice QCD** | 20+ | SU3 HMC, Wilson plaquette, CG solver, Dirac staggered fermions, pseudofermion forces, gauge forces, link updates |
| **Science/HFB** | 15+ | Batched HFB (hamiltonian, BCS, wavefunction, potential, density, energy), BCS bisection |
| **Math (df64)** | 8+ | `df64_core`, `math_f64`, `complex_f64`, `su3`, `su3_df64` |
| **Special functions** | 3 | Laguerre, Hermite polynomials |
| **MD transport** | 2 | `heat_current_f64` |

Key contribution: **df64 double-float arithmetic** — the precision foundation that
all Springs depend on for f64-quality GPU computation. Without hotSpring's df64
work, airSpring's ET₀ and water balance GPU paths would not achieve the required
sub-1% agreement with CPU baselines.

Cross-spring benefit to airSpring: `pow_f64` fix (TS-001) originated from
airSpring's ET₀ testing, revealing that fractional exponents returned 0.0.
hotSpring's math infrastructure (`exp_f64`, `log_f64`) provided the fix path.

### wetSpring — Bio & Environmental (25 shaders)

Domain: Metagenomics, phylogenetics, environmental monitoring, ESN.

| Category | Shaders | Examples |
|----------|---------|----------|
| **Bio** | 14 | Smith-Waterman banded, Felsenstein, UniFrac, DADA2, k-mer histogram, taxonomy, dN/dS, ANI, HMM forward, Gillespie SSA |
| **Math** | 4 | Bray-Curtis, Hill diversity, cosine similarity, batch pair reduce |
| **Interpolation** | 1 | `kriging_f64` |
| **Stats** | 1 | `moving_window` |
| **ESN** | 3 | reservoir_update, readout (f32/f64) |
| **Reduce** | 1 | `fused_map_reduce_f64` |

Key contributions to airSpring:
- **`kriging_f64.wgsl`** — spatial interpolation for soil moisture mapping from
  sensor networks. wetSpring uses it for environmental sample sites; airSpring
  wired it as `gpu::kriging::KrigingInterpolator`.
- **`fused_map_reduce_f64.wgsl`** — single-dispatch map+reduce. airSpring's
  TS-004 fix (buffer conflict for N≥1024) stabilized this for all Springs.
- **`moving_window.wgsl`** — sliding window statistics (mean, variance, min, max).
  wetSpring used it for environmental monitoring (S28+); airSpring now wires it
  as `gpu::stream::StreamSmoother` for IoT sensor stream smoothing.
- **`ridge_regression`** — CPU-only ridge from ESN calibration pipeline. airSpring
  wires it as `eco::correction::fit_ridge` for regularized sensor calibration.

### neuralSpring — ML & Optimization (20 shaders)

Domain: Evolutionary optimization, neural surrogates, spectral methods.

| Category | Shaders | Examples |
|----------|---------|----------|
| **Math** | 6 | spatial_payoff, pairwise (Jaccard/L2/Hamming), matmul variants |
| **Linalg** | 2 | symmetrize, laplacian |
| **Sample** | 1 | metropolis |
| **Spectral** | 1 | batch_ipr |
| **ML** | 2 | hmm_forward_log, batch_fitness_eval |
| **Bio** | 4 | swarm_nn_forward, multi_obj_fitness, locus_variance, hill_gate |
| **Numerical** | 2 | hessian_column, rk4_parallel |
| **Stats** | 2 | mean_reduce, histogram |

Key contributions to ecosystem: Multi-head attention decomposition, mixed-hardware
inference infrastructure, and the `nelder_mead` optimizer. airSpring's `gpu::isotherm`
module now wires `nelder_mead` for nonlinear isotherm fitting (v0.4.0).

### airSpring — Precision Agriculture (5 shaders used, 3 fixes contributed)

**Shaders used:**
- `bray_curtis_f64` — sensor similarity metrics
- `kriging_f64` — soil moisture spatial interpolation
- `batched_elementwise_f64` — ET₀ and water balance batch GPU
- `fused_map_reduce_f64` — seasonal aggregation statistics
- `moving_window.wgsl` — IoT sensor stream smoothing

**Fixes contributed upstream:**
- **TS-001** `pow_f64`: fractional exponents returned 0.0 → fixed with
  `exp_f64(exp * log_f64(base))`. Discovered during ET₀ atmospheric pressure calc.
- **TS-003** `acos_simple`/`sin_simple`: low-order polynomial approximations →
  replaced with full-precision `math_f64.wgsl` implementations.
- **TS-004** `FusedMapReduceF64`: buffer usage conflict for N≥1024 → separate
  input/output buffers in partials pipeline.

---

## Cross-Spring Shader Usage Matrix

| Shader | hotSpring | wetSpring | neuralSpring | airSpring | Benefit |
|--------|-----------|-----------|--------------|-----------|---------|
| `kriging_f64` | | x | | x | Spatial interpolation for soil moisture & sample sites |
| `bray_curtis_f64` | | x | | x | Diversity metrics & sensor similarity |
| `batched_elementwise_f64` | x | x | x | x | Unified element-wise ops; TS-001 fix benefits all |
| `fused_map_reduce_f64` | x | x | x | x | Single-dispatch map+reduce; TS-004 fix stabilizes all |
| `moving_window` | | x | | x | Environmental monitoring & IoT stream smoothing |
| `math_f64` | x | x | x | x | Shared f64 math; TS-003 `acos` fix from airSpring |
| `df64_core` | x | x | x | x | Double-float precision foundation (hotSpring origin) |

---

## BarraCuda Primitives Wired into airSpring

| airSpring Module | BarraCuda Primitive | Spring Origin | Status |
|-----------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | Multi-spring | GPU-FIRST |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | Multi-spring | GPU-STEP |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | wetSpring | INTEGRATED |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | wetSpring | GPU N≥1024 |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | wetSpring S28+ | **WIRED** (new) |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN | **WIRED** (new) |
| `gpu::dual_kc::BatchedDualKc` | CPU path (Tier B → GPU pending) | airSpring | **CPU-STEP** |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | airSpring v0.4.0 | **WIRED** (new) |
| `gpu::isotherm::fit_*_nm` | `optimize::nelder_mead` | airSpring v0.4.0 | **WIRED** (new) |
| `validation` | `validation::ValidationHarness` | neuralSpring | ABSORBED |
| `testutil::r_squared` | `stats::pearson_correlation` | Shared | WIRED |
| `testutil::spearman_r` | `stats::spearman_correlation` | Shared | WIRED |
| `testutil::bootstrap_rmse` | `stats::bootstrap_ci` | Shared | WIRED |

---

## Timeline of Cross-Spring Evolution

| Date | Event | Impact on airSpring |
|------|-------|---------------------|
| Jan 2026 | hotSpring df64 core + lattice QCD | Foundation for f64 GPU precision |
| Jan 2026 | wetSpring bio shaders + kriging | `KrigingInterpolator` becomes possible |
| Feb 2026 | neuralSpring MHA + optimization | `nelder_mead` available (Tier B) |
| Feb 16 | Unified handoff — TS-001/002/003/004 resolved | GPU-FIRST ET₀ + water balance |
| Feb 16 | airSpring 3 fixes contributed upstream | Precision + stability for all Springs |
| Feb 22+ | ToadStool S51-S57: 46 cross-spring absorptions | ValidationHarness absorbed |
| Feb 24 | airSpring v0.3.6: ToadStool sync + rewire | BarraCuda rename, 8 Tier A items |
| Feb 24 | airSpring v0.3.6+: MovingWindow + Ridge wired | Stream smoothing + calibration pipeline |
| Feb 25 | airSpring v0.3.7: metalForge v0.2.0 evolution | 4 absorption-ready modules (metrics, regression, moving_window_f64, hydrology) |
| Feb 25 | airSpring v0.3.8: ToadStool deep audit | Richards PDE promoted C→B (upstream solver available), +2 Tier B gaps (tridiag, RK45) |
| Feb 25 | airSpring v0.3.9: Dual Kc + cover crops | FAO-56 Ch 7/11 CPU-validated, 5 cover crop species, no-till mulch |
| Feb 25 | airSpring v0.3.10: GPU dual Kc + benchmarks | BatchedDualKc orchestrator, CPU benchmarks 12.7M ET₀/s |
| Feb 25 | airSpring v0.4.0: Richards + isotherm + 60yr WB | `gpu::richards` wired to pde::richards, `gpu::isotherm` wired to optimize::nelder_mead, 75/75 cross-validation |
| Feb 25 | airSpring v0.4.1: ToadStool S52-S62 sync | multi_start_nelder_mead wired, global isotherm fitting (LHS), 323 tests |
| Feb 25 | airSpring v0.4.2: GPU integration + benchmarks | Richards/isotherm GPU integration tests, cross-spring benchmark suite, 328 tests |

---

## Cross-Spring Shader Provenance — Who Helps Whom

| Spring | Shaders | Contribution to airSpring | Reverse Contribution |
|--------|---------|--------------------------|---------------------|
| **hotSpring** | 56 | df64 core, pow/exp/log/trig f64 — enables VG retention, atmospheric pressure | TS-001 pow_f64 fix (airSpring uncovered) |
| **wetSpring** | 25 | kriging_f64, fused_map_reduce, moving_window, ridge_regression | TS-004 reduce buffer fix (airSpring stabilized for all) |
| **neuralSpring** | 20 | nelder_mead, multi_start_nelder_mead, ValidationHarness | TS-003 acos precision fix (airSpring found boundary issue) |
| **airSpring** | — | Domain consumer | Richards PDE → absorbed upstream (S40) |

774 WGSL shaders in ToadStool, 46 cross-spring absorptions (S51-S57). airSpring uses 5 shader families + contributed 3 critical fixes. Zero shader duplication.

---

## Benchmark Summary (CPU baselines, `--release`, v0.4.2)

| Operation | N | Time (µs) | Throughput | Provenance |
|-----------|---|-----------|------------|------------|
| ET₀ (FAO-56) | 10,000 | 797 | 12.5M ops/sec | hotSpring pow_f64, multi-spring elementwise |
| Reduce (seasonal) | 100,000 | 254 | 395M elem/sec | wetSpring fused_map_reduce |
| Stream smooth | 8,760 (24h) | 276 | 31.7M elem/sec | wetSpring moving_window |
| Kriging | 20→500 | 26 | — | wetSpring kriging_f64 |
| Ridge regression | 5,000 | 48 | R²=1.000 | wetSpring ESN ridge |
| Richards PDE | 50 nodes | 13,930 | 72 sims/sec | airSpring→ToadStool S40, hotSpring df64 |
| VG θ(h) batch | 100,000 | 2,575 | 38.9M evals/sec | hotSpring df64 precision |
| Isotherm (linearized) | 9 pts | 0.1 | 8.3M fits/sec | airSpring eco::isotherm |
| Isotherm (NM 1-start) | 9 pts | 5.7 | 175K fits/sec | neuralSpring nelder_mead |
| Isotherm (NM 8×LHS) | 9 pts | 23.5 | 42.5K fits/sec | neuralSpring multi_start_nelder_mead |

---

## Remaining Evolution Gaps

**Tier B (11 items):** Richards PDE GPU shader (CPU wired via `gpu::richards`),
batch Nelder-Mead (CPU wired via `gpu::isotherm`), VG θ/K batch (new op),
dual Kc batch (op=8, GPU orchestrator wired, pending shader),
sensor calibration batch, Hargreaves batch, Kc climate adjustment,
isotherm batch fitting, tridiagonal solve, adaptive RK45 ODE,
m/z tolerance search.

**Tier C (1 item):** HTTP/JSON data client.

See `gpu::evolution_gaps` module for full structured inventory (20 gaps total).
