# Cross-Spring Shader Evolution — airSpring Provenance

**Updated**: March 7, 2026 (v0.7.3, barraCuda 0.3.3 / wgpu 28)

## Summary

BarraCuda contains **767+ WGSL shaders** (pure math, precision per silicon), built
through **60+ cross-spring absorptions**. Each Spring contributes domain-specific GPU
primitives that benefit the entire ecosystem.

airSpring uses **8 shared shader families** directly (ops 0-19 + uncertainty stack),
contributed **3 critical fixes** (TS-001, TS-003, TS-004), had its **stats metrics
absorbed upstream** (S64), and completed the full Write→Absorb→Lean cycle: all 6 local
WGSL ops absorbed upstream into `BatchedElementwiseF64` (ops 14-19), `local_dispatch`
retired (v0.7.2). `PrecisionRoutingAdvice` wired and upstream provenance registry
integrated (v0.7.3).

v0.7.3: **25 Tier A** orchestrators (ops 0-19 all upstream), 848 lib + 186 forge tests,
`PrecisionRoutingAdvice` for per-hardware f64 dispatch routing. 27 metalForge workloads,
NUCLEUS mesh routing (Exp 076: 60/60). BrentGpu, RichardsGpu, StatefulPipeline.

### Paper 12 — Immunological Anderson Infrastructure (NEW)

airSpring provides computational infrastructure for Paper 12 (Anderson
localization in immunological signaling):

| Module | Paper 12 Application | Status |
|--------|---------------------|--------|
| `eco::tissue` | Cell-type diversity → Anderson disorder W (Pielou evenness mapping) | **NEW** (8 tests) |
| `eco::cytokine` | `CytokineBrain`: 3-head Nautilus reservoir (IL-31 propagation, tissue W, barrier state) | **NEW** (9 tests) |
| `gpu::diversity` | GPU-accelerated Shannon/Simpson/Pielou for tissue cell populations | Existing (v0.6.1) |
| `nautilus` | `AirSpringBrain` pattern reused for `CytokineBrain` | Existing (v0.6.2) |
| `gpu::atlas_stream` | `MonitoredAtlasStream` → `DriftMonitor` pattern for cytokine regime changes | Existing (v0.6.2) |
| `gpu::van_genuchten` | Barrier permeability model (porous media ↔ skin ECM analogy) | Existing (v0.6.1) |

**Dimensional Promotion/Collapse Duality** (Paper 06 ↔ Paper 12):
```
Paper 06 (soil):  Tillage = d collapse (3D→2D) → QS FAILS    → ecosystem loss
Paper 12 (skin):  Scratch = d promotion (2D→3D) → cytokine PROPAGATES → AD amplification
```
Same Anderson physics, opposite direction, opposite outcome. `eco::tissue::barrier_disruption_d_eff()`
quantifies the dimensional promotion from barrier breach fraction.

### bingoCube/nautilus — Evolutionary Reservoir Computing (INTEGRATED)

`ecoPrimals/primalTools/bingoCube/nautilus/` provides feed-forward evolutionary
reservoir computing on BingoCube boards. **Now integrated** as `airspring_barracuda::nautilus`
with a domain-specific `AirSpringBrain` (3-head: ET₀, soil moisture, crop stress).

| Capability | API | airSpring Integration |
|------------|-----|----------------------|
| Evolutionary reservoir | `NautilusShell::evolve_generation()` | `AirSpringBrain::train()` — weather→ET₀/soil/crop prediction |
| Drift monitoring | `DriftMonitor::is_drifting()` | `MonitoredAtlasStream` — regime change detection in 80-year atlas |
| Edge seeding | `EdgeSeeder::seed_boards()` | Focus boards on concept-edge microclimates |
| AKD1000 NPU export | `NautilusShell::export_akd1000_weights()` | ~48µs edge inference for sensor stations |
| Shell transfer/merge | `NautilusShell::continue_from()`, `merge_shell()` | Cross-station learning (station-a → station-b) |
| JSON persistence | `AirSpringBrain::export_json()` | Cross-run bootstrap via serde |
| Concept edge detection | Per-observation MSE threshold | Identifies regime boundaries (drought onset, frost) |

### S60–S71 Cross-Spring Absorption Wave (Feb–Mar 2026)

| Session | What Was Absorbed | Origin Spring | airSpring Impact |
|---------|-------------------|---------------|------------------|
| S60 | DF64 FMA + transcendentals | hotSpring | `df64_core.wgsl` now uses FMA (2 ops vs 17); `df64_transcendentals.wgsl` adds sin/cos/exp/log in double-double |
| S64 | Stats metrics (rmse, mbe, NSE, IA, R², hit\_rate, mean, percentile) | **airSpring** → upstream | Our `testutil::stats` functions now live in `barracuda::stats::metrics` |
| S64 | Ecological diversity (Shannon, Simpson, Chao1, Bray-Curtis, rarefaction) | wetSpring | New `eco::diversity` module wired in airSpring |
| S64 | MC ET₀ propagation shader | groundSpring → `ToadStool` | `mc_et0_propagate_f64.wgsl` available for GPU uncertainty bands |
| S64 | Bio GPU ops (diversity\_fusion, batched\_multinomial) | wetSpring | Available for future large-scale diversity GPU dispatch |
| S61-63 | Sovereign compiler (SPIR-V passthrough) | `ToadStool` core | Regression: breaks `BatchedElementwiseF64` GPU dispatch (P0 fix needed) |
| S65 | Smart refactoring + dead code removal | `ToadStool` core | Cleaner upstream codebase |
| S66 | BindGroupLayout fix (R-S66-041) | `ToadStool` core | P0 GPU dispatch blocker resolved |
| S67-70 | `HargreavesBatchGpu`, `fao56_et0`, `HistogramGpu` | groundSpring | GPU Hargreaves ET₀ + scalar PM available; bit-identical to airSpring local |
| S71 | `JackknifeMeanGpu`, `BootstrapMeanGpu`, `KimuraGpu` | groundSpring + wetSpring | GPU jackknife/bootstrap CI + Kimura fixation probability |
| S71 | Universal precision architecture (`Fp64Strategy`) | hotSpring → core | f32-only shaders removed; precision per silicon (F64, Df64, F32, F16) |
| S71 | 66 `ComputeDispatch` migrations | `ToadStool` core | Clean dispatch API, `BindGroupLayout` consistent |
| S71 | DF64 transcendentals complete (15 functions) | hotSpring | sinh, cosh, tanh, atanh, cbrt, hypot, log1p, expm1 now available |

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

### airSpring — Precision Agriculture (6 shader families used, 3 fixes contributed, stats absorbed)

**Shaders used:**
- `bray_curtis_f64` — sensor similarity + agroecological diversity (via `eco::diversity`)
- `kriging_f64` — soil moisture spatial interpolation
- `batched_elementwise_f64` — ET₀ and water balance batch GPU
- `fused_map_reduce_f64` — seasonal aggregation statistics
- `moving_window.wgsl` — IoT sensor stream smoothing
- `mc_et0_propagate_f64.wgsl` — Monte Carlo ET₀ uncertainty (GPU kernel available, CPU wired)

**Functions absorbed upstream (S64):**
- `rmse`, `mbe`, `nash_sutcliffe`, `r_squared`, `index_of_agreement` → `barracuda::stats::metrics`
- airSpring's `testutil::stats::rmse` and `mbe` now delegate to upstream

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
| `df64_transcendentals` | x | | | *(future)* | sin/cos/exp/log in double-double — hotSpring S60 |
| `mc_et0_propagate_f64` | | | | x | MC uncertainty propagation — groundSpring → S64 |
| `diversity_fusion` | | x | x | *(future)* | Fused Shannon+Simpson+evenness GPU dispatch |

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
| `testutil::{hit_rate,mean,percentile,dot,l2_norm}` | `stats::metrics::*` | airSpring → S64 absorption | **WIRED** (new, v0.4.3) |
| `eco::diversity::*` | `stats::diversity::*` | wetSpring → S64 absorption | **WIRED** (new, v0.4.3) |
| `gpu::mc_et0::mc_et0_cpu` | `mc_et0_propagate_f64.wgsl` (CPU mirror) | groundSpring → S64 | **WIRED** (new, v0.4.3) |

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
| Feb 26 | ToadStool S60-S65 pulled | 234 files changed, df64 FMA, stats absorption, diversity, MC ET₀, sovereign compiler regression |
| Feb 26 | airSpring v0.4.3: Modern rewiring | Stats delegate to upstream, `eco::diversity` wired (wetSpring), `gpu::mc_et0` wired (groundSpring), 571 tests |
| Feb 26 | airSpring v0.4.4: Deep S65 rewiring | `norm_ppf` → parametric CI, `brent` → VG inverse, CN f64 cross-val, 11 Tier A |
| Feb 26 | airSpring v0.4.5: Experiment buildout | 3 new experiments (scheduling, lysimeter, sensitivity), 474/474 Python, 725 Rust checks, 21 binaries |
| Feb 26 | ToadStool S66 pulled | Cross-spring absorption: regression (R-S66-001), hydrology (R-S66-002), moving_window_f64 (R-S66-003), spearman re-export (R-S66-005), 8 SoilParams constants (R-S66-006), mae/hill/monod, multi-precision WGSL |
| Feb 26 | airSpring S66 sync | All metalForge modules absorbed upstream — rewired provenance docs, cleaned ABSORPTION_MANIFEST, evolution_gaps updated |
| Feb 26 | airSpring S66 validation | 8 cross-spring S66 tests, P0 GPU dispatch resolved (R-S66-041 explicit BGL), 3 new GPU bench ops, 3 new CPU bench sections, 662 Rust tests + 1302 atlas |
| Feb 26 | airSpring v0.4.6: Deep audit | R-S66-001/003 wired (correction→regression, stream→moving_window), van_genuchten extracted, clippy nursery enforced, 11 doc-tests, 662 Rust tests + 1302 atlas checks, 97.45% coverage |
| Feb 26 | ToadStool S67-S68 pulled | S67 codified "math is universal, precision is silicon"; S68 evolved 334+ shaders to f64-canonical with `downcast_f64_to_f32()` backward compat. `ValidationHarness` migrated to `tracing::info!` |
| Feb 26 | airSpring S68 sync | Added `tracing-subscriber` for `ValidationHarness` output, wired `init_tracing()` into all 22 binaries, 10 new S68 cross-spring evolution tests (regression, hydrology, diversity, moving_window, Brent, Spearman, bootstrap, atlas pipeline, Richards benchmarks). 608 cargo tests + 1354 atlas checks all pass |
| Feb 26 | airSpring v0.4.7: PT + intercomparison | Priestley-Taylor ET₀ (Exp 019), 3-method intercomparison (Exp 020). 24 binaries, 616 Rust tests + 1393 atlas. Cross-spring test files split: `cross_spring_absorption.rs`, `cross_spring_benchmarks.rs`, `cross_spring_primitives.rs` |
| Feb 26 | airSpring v0.4.8: Thornthwaite + GDD + pedotransfer | Thornthwaite monthly ET₀ (Exp 021), GDD phenology (Exp 022), Saxton-Rawls pedotransfer (Exp 023). 22 experiments, 594/594 Python, 491 Rust tests + 570 validation + 1393 atlas, 27 binaries. V022 handoff |
| Feb 26 | airSpring v0.4.11: NASS yield, forecast, SCAN, multicrop, NPU trilogy, AmeriFlux, Hargreaves, diversity | 32 experiments, 808 Python, 499 Rust tests + 853 validation + 1393 atlas, 38 binaries. AKD1000 NPU live. metalForge mixed hardware |
| Feb 27 | airSpring v0.4.12: Debt resolution, clippy pedantic, tolerance centralization, CI coverage gate, primal self-knowledge | V024 handoff: error type evolution, NPU convergence (3 Springs), barracuda delegation inventory |
| Feb 27 | airSpring v0.5.0: 12 new experiments (Exp 033-044), Titan V GPU live, metalForge live hardware | 44 experiments, 1054 Python, 645 Rust tests + 1393 atlas, 51 binaries. Titan V 24/24 PASS (0.04% seasonal parity). metalForge 5 live substrates. V028 ToadStool absorption handoff |
| Feb 27 | airSpring v0.5.1: Anderson coupling (Exp 045), CPU benchmark, documentation sweep | 45 experiments, 1109 Python, 651 Rust tests + 1393 atlas, 54 binaries. 25.9× Rust-vs-Python (8/8 parity). `eco::anderson` coupling chain (θ→S_e→d_eff→QS). V030 evolution handoff |
| Feb 27 | airSpring v0.5.2: 4 Tier B GPU orchestrators + seasonal pipeline + atlas stream | 584 lib + 31 forge tests, 55 binaries. Ops 5-8 wired (sensor cal, Hargreaves, Kc climate, dual Kc). Seasonal pipeline chained ET₀→Kc→WB→Yield. Atlas stream 73/73 PASS (12 stations, 4800 results). metalForge 18 workloads, 29/29 cross-system. V052 ToadStool/NestGate/biomeOS handoffs |
| Feb 28 | airSpring v0.5.3: Blaney-Criddle + SCS-CN runoff + Green-Ampt infiltration | 618 lib tests, 56 binaries, 51 experiments. 8th ET₀ method, `eco::runoff` + `eco::infiltration` modules. 42+ named constants, zero dead code, capability-based GPU. V034 deep debt handoff |
| Feb 28 | airSpring v0.5.4: Pipeline coupling + inverse problems + season-scale audit | 618 lib tests, 59 binaries, 54 experiments. Coupled runoff-infiltration (292/292), VG inverse via Brent (84/84), full-season WB audit (34/34). V035 handoff: batched Brent GPU candidate |
| Mar 01 | airSpring v0.5.6: ToadStool S70+ complete rewire + GPU benchmark | 640 lib tests, 67 binaries, 57 experiments. Ops 5-8 GPU-first, GPU stats (neuralSpring S69), Exp 057 (26/26), 20/20 CPU parity (17.9×), 35/35 cross-spring benchmarks. V039 handoff: full cross-spring provenance |
| Mar 01 | airSpring v0.5.7: Science extensions + streaming + debt resolution | 641 lib tests, 68 binaries, 58 experiments. Exp 058 climate scenario (46/46), streaming pipeline (GpuPipelined/GpuFused), 21/21 CPU parity (14.5×), Turc named constants, zero magic numbers. V040 handoff |
| Mar 01 | airSpring v0.5.8: NUCLEUS cross-primal pipeline + ecology domain | 641 lib tests, 72 binaries, 63 experiments. NUCLEUS primal (16 caps), ecology domain in biomeOS, capability.call routing, cross-primal forwarding, 28/28 pipeline (Exp 063). Atlas decade 80yr (102/102), NASS real (99/99), NCBI diversity (63/63). V041 handoff |

---

## Cross-Spring Shader Provenance — Who Helps Whom

| Spring | Shaders | Contribution to airSpring | Reverse Contribution |
|--------|---------|--------------------------|---------------------|
| **hotSpring** | 56 | df64 core, pow/exp/log/trig f64 — enables VG retention, atmospheric pressure | TS-001 pow_f64 fix (airSpring uncovered) |
| **wetSpring** | 25 | kriging_f64, fused_map_reduce, moving_window, ridge_regression | TS-004 reduce buffer fix (airSpring stabilized for all) |
| **neuralSpring** | 20 | nelder_mead, multi_start_nelder_mead, ValidationHarness | TS-003 acos precision fix (airSpring found boundary issue) |
| **groundSpring** | — | MC ET₀ propagation shader (S64), uncertainty quantification | — |
| **airSpring** | — | Domain consumer | Richards PDE (S40), stats metrics (S64) absorbed upstream |

845 WGSL shaders in ToadStool, 46+ cross-spring absorptions (S51-S65). airSpring uses 6 shader families + contributed 3 critical fixes + stats metrics absorbed upstream. Zero shader duplication.

---

845 WGSL shaders in ToadStool (S93), 46+ cross-spring absorptions (S51-S68). airSpring metalForge fully absorbed. S93 universal precision validated.

## Benchmark Summary (CPU baselines, `--release`, v0.4.5)

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
| **Alpha diversity** | **100 species** | **< 200** | **> 50K evals/sec** | **wetSpring S64 diversity absorption** |
| **MC ET₀ (uncertainty)** | **10K samples** | **< 5,000,000** | **> 2K propagations/sec** | **groundSpring S64 mc_et0_propagate** |
| **Stats delegation** | **10K vectors ×4** | **< 2,000,000** | **> 2K metric/sec** | **airSpring→upstream S64 absorption** |

---

## Remaining Evolution Gaps

**Tier B (14 items, 9 wired):** Richards PDE GPU shader (CPU wired via `gpu::richards`),
batch Nelder-Mead (CPU wired via `gpu::isotherm`), VG θ/K batch (new op),
dual Kc batch (op=8, **WIRED** v0.5.2), sensor calibration batch (op=5, **WIRED** v0.5.2),
Hargreaves batch (op=6, **WIRED** v0.5.2), Kc climate adjustment (op=7, **WIRED** v0.5.2),
seasonal pipeline (**WIRED** v0.5.2, CPU chained), atlas stream (**WIRED** v0.5.2),
MC ET₀ GPU (**WIRED** v0.5.2), isotherm batch fitting, tridiagonal solve,
adaptive RK45 ODE, m/z tolerance search.

**Tier C (1 item):** HTTP/JSON data client.

See `gpu::evolution_gaps` module for full structured inventory (26 gaps total: 11 Tier A, 14 Tier B (9 wired), 1 Tier C).
