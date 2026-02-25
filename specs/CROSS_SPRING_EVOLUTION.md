# Cross-Spring Shader Evolution — airSpring Provenance

**Updated**: February 25, 2026 (v0.3.7, ToadStool HEAD `02207c4a`)

## Summary

ToadStool's BarraCuda runtime contains **608 WGSL shaders** across 41 categories,
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
inference infrastructure, and the `nelder_mead` optimizer that airSpring may wire
for nonlinear curve fitting (currently Tier B in evolution gaps).

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

---

## Benchmark Summary (CPU baselines, `--release`)

| Operation | N | Time (µs) | Throughput | Provenance |
|-----------|---|-----------|------------|------------|
| ET₀ (FAO-56) | 10,000 | 795 | 12.6M ops/sec | hotSpring pow_f64, multi-spring elementwise |
| Reduce (seasonal) | 100,000 | 251 | 399M elem/sec | wetSpring fused_map_reduce |
| Stream smooth | 8,760 (24h) | 270 | 32.4M elem/sec | wetSpring moving_window |
| Kriging | 20→500 | 26 | — | wetSpring kriging_f64 |
| Ridge regression | 5,000 | 48 | R²=1.000 | wetSpring ESN ridge |

---

## Remaining Evolution Gaps

**Tier B (8 items):** 1D Richards equation (PROMOTED from Tier C — upstream
`pde::richards::solve_richards` now available with van Genuchten-Mualem),
sensor calibration batch, Hargreaves batch, Kc climate adjustment,
nonlinear solver (Nelder-Mead), tridiagonal solve, adaptive RK45 ODE,
m/z tolerance search.

**Tier C (1 item):** HTTP/JSON data client.

See `gpu::evolution_gaps` module for full structured inventory.
