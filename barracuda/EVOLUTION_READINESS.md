# airSpring BarraCuda — Evolution Readiness

**Last Updated**: March 3, 2026 (v0.6.8 — 1132 tests, 86 binaries, 77 experiments, 1237 Python, 30 NUCLEUS capabilities, 67/67 metalForge cross-system, 6 local WGSL shaders)
**barraCuda**: v0.3.1 standalone primal (`ecoPrimals/barraCuda` — 767 WGSL shaders, 957 Rust files, extracted from ToadStool S89)
**ToadStool**: S93 (5,369 tests, 845 WGSL shaders, sovereignty cleanup, DF64 ownership transferred to barraCuda)
**Handoff**: V0.6.8 (barraCuda 0.3.1 standalone rewire + absorption — capability-based discovery, 6 shaders for upstream, deep debt resolved)
**License**: AGPL-3.0-or-later

---

## Write → Absorb → Lean Status

airSpring follows the same pattern as hotSpring and wetSpring: implement locally,
validate against papers, hand off to barraCuda, lean on upstream.

**Architecture (S89+)**: barraCuda is the standalone sovereign math engine —
"math is universal, precision is silicon." ToadStool dispatches across hardware;
barraCuda provides universal math primitives. Springs depend on barraCuda directly,
not through ToadStool.

### Already Absorbed (Lean)

| Module | Absorbed Into | When | Status |
|--------|--------------|------|--------|
| `ValidationRunner` | `barracuda::validation::ValidationHarness` | S59 | **Leaning** — all 82 binaries use upstream |
| `van_genuchten` | `barracuda::pde::richards::SoilParams` | S40 | **Leaning** — `gpu::richards` bridges to upstream |
| `isotherm NM` | `barracuda::optimize::nelder_mead` | S62 | **Leaning** — `gpu::isotherm` bridges to upstream |
| `StatefulPipeline` | `barracuda::pipeline::stateful::StatefulPipeline` | S80 | **Leaning** — `WaterBalanceState` day-over-day |
| `BatchedStatefulF64` | `barracuda::pipeline::batched_stateful` | S83 | **Available** — GPU-resident ping-pong state |
| `BrentGpu` | `barracuda::optimize::brent_gpu` | S83 | **Available** — batched GPU Brent root-finding |
| `RichardsGpu` | `barracuda::pde::richards_gpu` | S83 | **Available** — GPU Picard solver |
| `BatchedNelderMeadGpu` | `barracuda::optimize::batched_nelder_mead_gpu` | S80 | **Available** — parallel NM optimizations |
| `nautilus` | `barracuda::nautilus` | S80 | **Available** — evolutionary reservoir (bingoCube) |

### Absorbed Upstream (6/6 metalForge modules — Write→Absorb→Lean complete)

| Module | Absorbed Into | When | Status |
|--------|--------------|------|--------|
| `forge::metrics` | `barracuda::stats::metrics` | S64 | **LEANING** — `testutil::stats` delegates |
| `forge::regression` | `barracuda::stats::regression` | S66 (R-S66-001) | **LEANING** — `eco::correction` keeps domain `FittedModel` |
| `forge::moving_window` | `barracuda::stats::moving_window_f64` | S66 (R-S66-003) | **LEANING** — `gpu::stream` f64 path available |
| `forge::hydrology` | `barracuda::stats::hydrology` | S66 (R-S66-002) | **LEANING** — `eco::evapotranspiration` keeps FAO-56 param order |
| `forge::isotherm` | `barracuda::eco::isotherm` (was local) | S64 | **LEANING** — `gpu::isotherm` delegates via NM |
| `forge::van_genuchten` | `barracuda::pde::richards::SoilParams` | S40+S66 | **LEANING** — 8 named constants (R-S66-006) |

See `metalForge/ABSORPTION_MANIFEST.md` for full signatures and validation details.

### Stays Local (domain-specific)

| Module | Reason |
|--------|--------|
| `eco::dual_kc` | FAO-56 Ch 7/11 domain logic — too specialized for barracuda |
| `eco::sensor_calibration` | SoilWatch 10 specific — domain consumer |
| `eco::crop` | FAO-56 Table 12 crop database, GDD, kc_from_gdd — domain data |
| `eco::evapotranspiration` | Thornthwaite monthly ET₀, Blaney-Criddle ET₀ — domain consumer |
| `eco::runoff` | SCS-CN curve number — domain consumer |
| `eco::infiltration` | Green-Ampt infiltration — domain consumer |
| `eco::soil_moisture` | Saxton-Rawls pedotransfer (θs/θr/Ks from texture) — domain consumer |
| `io::csv_ts` | airSpring-specific IoT CSV parser |
| `testutil::generators` | Synthetic IoT data for airSpring tests |

---

## GPU Evolution Tiers

### Tier A: Integrated (15 modules — GPU primitive wired or CPU batch ready)

| airSpring Module | BarraCuda Primitive | Status |
|-----------------|--------------------|----|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **INTEGRATED** |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **WIRED** |
| `gpu::infiltration` | `BrentGpu::solve_green_ampt()` (brent_f64.wgsl GA residual) | **WIRED** (S83) |
| `gpu::runoff` | Batched SCS-CN (CPU-vectorised, ToadStool op pending) | **CPU batch** |
| `gpu::yield_response` | Batched Stewart (CPU-vectorised, ToadStool op pending) | **CPU batch** |
| `gpu::simple_et0` | Batched Makkink/Turc/Hamon/Blaney-Criddle (CPU-vectorised, ToadStool ops pending) | **CPU batch** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **WIRED** |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | **WIRED** (+ CN f64 cross-val) |
| `gpu::isotherm::fit_*_nm/global` | `optimize::nelder_mead` + `multi_start` | **WIRED** |
| `eco::diversity` | `stats::diversity` (Shannon, Simpson, Bray-Curtis, matrix, frequencies) | **LEANING** (S64+S66) |
| `gpu::mc_et0::parametric_ci` | `stats::normal::norm_ppf` | **WIRED** — hotSpring precision lineage |
| `eco::richards::inverse_van_genuchten_h` | `optimize::brent` | **WIRED** — neuralSpring optimizer lineage |

### Tier B: Upstream Exists, Needs Domain Wiring (14 items, 9 wired)

| Need | Closest Primitive | Effort | Status |
|------|-------------------|:------:|--------|
| Sensor batch calibration | `batched_elementwise_f64` (op=5) | Low | **WIRED** (v0.5.2) |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | Low | **WIRED** (v0.5.2) |
| Kc climate adjustment | `batched_elementwise_f64` (op=7) | Low | **WIRED** (v0.5.2) |
| Dual Kc batch (Ke) | `batched_elementwise_f64` (op=8) | Low | **WIRED** (v0.5.2) |
| Seasonal pipeline | Chains ops 0→7→1→yield | Low | **WIRED** (v0.5.2, CPU chained) |
| Atlas stream | `UnidirectionalPipeline` (pending) | Low | **WIRED** (v0.5.2, CPU chained) |
| MC ET₀ GPU | `mc_et0_propagate_f64.wgsl` + `norm_ppf` | Low | **WIRED** (v0.5.2) |
| VG θ/K batch | `batched_elementwise_f64` (new op) | Low | |
| Batch Nelder-Mead GPU | `NelderMeadGpu` | Medium | |
| Crank-Nicolson PDE | `pde::crank_nicolson::CrankNicolson1D` (f64 + GPU shader!) | Low | **WIRED** |
| Brent VG inverse | `optimize::brent` | Low | **WIRED** |
| Tridiagonal solve batch | `linalg::tridiagonal_solve_f64` | Low | |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Low | |
| m/z tolerance search | `batched_bisection_f64.wgsl` (wetSpring) | Low | |

### Tier C: Needs New Primitive (1 item)

| Need | Description |
|------|-------------|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs (not GPU) |

---

## ToadStool S42–S68+ Evolution (180+ commits)

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
| S60 | DF64 FMA, transcendentals, CN fix, Cholesky SPD | Math precision |
| S61-63 | Sovereign compiler, SPIR-V passthrough, `CrankNicolson1D` **f64** | **CN now f64!** |
| S64 | Stats absorption (metrics, diversity from Springs), `chrono` removed | Diversity leaning |
| S65 | Smart refactoring, dead code removal, doc cleanup | Stabilization |
| S66 | **Cross-spring absorption** + **P0 fix**: explicit BGL, regression, hydrology, 8 SoilParams | **All metalForge absorbed** |
| S67 | **Universal precision doctrine**: "math is universal — precision is silicon" | Architecture alignment |
| S68 | **296 f32-only shaders removed** — all f64 canonical, `op_preamble()`, `df64_rewrite.rs` naga IR | Pure math shaders |
| S68+ | GPU device-lost resilience, root doc cleanup, archive stale scripts | Stability |

## Upstream Capabilities — Wired and Available

### Wired (using in production)

| Capability | Module | Wired In | Status |
|-----------|--------|----------|--------|
| `barracuda::tolerances` | `tolerances` | v0.3.6 | **LEANING** — re-exported |
| `barracuda::validation::ValidationHarness` | `validation` | v0.3.6 | **LEANING** — all 63 validation binaries (incl. validate_atlas, 1393 checks) |
| `pde::richards::solve_richards` | `pde` | v0.4.0 | **WIRED** — `gpu::richards` |
| `pde::crank_nicolson::CrankNicolson1D` | `pde` | v0.4.4 | **WIRED** — CN f64 diffusion cross-val |
| `optimize::nelder_mead` | `optimize` | v0.4.1 | **WIRED** — isotherm fitting |
| `optimize::multi_start_nelder_mead` | `optimize` | v0.4.1 | **WIRED** — global isotherm search |
| `stats::diversity::*` | `stats` | v0.4.3 | **LEANING** — `eco::diversity` delegates (+ `bray_curtis_matrix`, `shannon_from_frequencies` v0.5.2) |
| `stats::metrics::*` | `stats` | v0.4.3 | **LEANING** — `testutil::stats` delegates |
| `stats::hydrology::hargreaves_et0_batch` | `stats` | v0.5.2 | **WIRED** — `gpu::hargreaves` delegates CPU batch to upstream |
| `stats::hydrology::crop_coefficient` | `stats` | v0.5.2 | **WIRED** — `eco::crop::crop_coefficient_stage` delegates to upstream |
| `stats::normal::norm_ppf` | `stats` | v0.4.4 | **WIRED** — `McEt0Result::parametric_ci()` |
| `optimize::brent` | `optimize` | v0.4.4 | **WIRED** — `inverse_van_genuchten_h()` θ→h inversion |

### Available (not yet needed)

| Capability | Module | Added In | Potential Use |
|-----------|--------|----------|---------------|
| `FusedMapReduceF64::dot(a, b)` | `ops` | S51 | GPU dot product convenience |
| `barracuda::provenance` | `provenance` | S52 | 12 `ProvenanceTag` consts for origin tracking |
| `solve_f64_cpu()` | `linalg::solve` | S51 | Gaussian elimination + partial pivoting |
| `GpuSessionBuilder` | `session` | S52 | Pre-warmed GPU sessions |
| `OdeSystem` + `BatchedOdeRK4` | `numerical` | S51 | Generic ODE with WGSL template |
| `NelderMeadGpu` | `optimize` | S52+ | GPU-resident NM (5-50 params) |
| `ResumableNelderMead` | `optimize` | S52+ | Checkpoint/resume for long-running optimizers |
| `bfgs` | `optimize` | S52+ | Quasi-Newton with gradient (smooth objectives) |
| `bisect` | `optimize` | S52+ | Robust bracketed root-finding |
| `newton` / `secant` | `optimize` | S52+ | Derivative-based root-finding |
| `BatchedBisectionGpu` | `optimize` | S52+ | GPU-parallel batched root-finding |
| `adaptive_penalty` | `optimize` | S52+ | Constrained optimization with penalty |
| `unified_hardware` | `unified_hardware` | S52 | `HardwareDiscovery`, `ComputeScheduler` — metalForge target |
| `chi2_decomposed` | `stats` | S52 | Chi-squared goodness-of-fit |
| `spectral_density` | `stats` | S57 | RMT spectral analysis |
| `normal::norm_cdf` | `stats` | S52+ | Normal cumulative distribution |
| `spearman_correlation` | `stats::correlation` | S66 (R-S66-005) | Rank correlation — **now re-exported** from `stats/mod.rs` |
| `compile_shader_universal` | `shaders` | S68 | One f64 source → F16/F32/F64/Df64 target |
| `Fp64Strategy::Native/Hybrid` | `device` | S58+ | Auto precision per GPU (ratio ≤2.5 → Native, else Hybrid) |
| `probe_f64_builtins` | `device` | S58+ | Hardware f64 builtin capability probing |
| `probe_f64_throughput_ratio` | `device` | S58+ | f64:f32 throughput ratio → F64Tier |
| `UnidirectionalPipeline` | `staging` | S52+ | Fire-and-forget streaming, eliminates round-trip overhead |
| `StatefulPipeline` | `staging` | S52+ | GPU-resident iterative solvers (minimal readback) |
| `MultiDevicePool` | `multi_gpu` | S52+ | Multi-GPU dispatch with load balancing |
| `ShaderTemplate` | `shaders` | S68 | `{{SCALAR}}`/`{{VEC2}}` templated precision-generic shaders |
| `compile_op_shader` | `shaders` | S68 | Inject `op_preamble` for abstract math ops |

---

## Quality Gates

| Check | Status |
|-------|--------|
| `cargo fmt --check` | **Clean** (both crates) |
| `cargo clippy --all-targets -W pedantic` | **0 warnings** (both crates) |
| `cargo doc --no-deps` | **Builds**, 0 warnings |
| `cargo test --workspace` | **1132 passed** (lib + bin + doc + integration) |
| `cargo llvm-cov --lib --summary-only` | **95.11% line** / **95.81% function** coverage |
| barraCuda version | **0.3.1** standalone primal (`ecoPrimals/barraCuda`) |
| `unsafe` code | **Zero** |
| `unwrap()` in lib | **Zero** (all in `#[cfg(test)]` or validation-binary JSON helpers) |
| Files > 1000 lines | **Zero** (max src: 872 `eco/evapotranspiration.rs` after Thornthwaite extraction) |
| Validation binaries | **63 PASS** (barracuda validate_*) + 3 bench (35/35 benchmarks) + 5/5 PASS (forge) |
| NUCLEUS pipeline | **28/28 PASS** (ecology domain, capability.call, cross-primal forwarding) |
| GPU live (Titan V) | **24/24 PASS** (0.04% seasonal parity, `BARRACUDA_GPU_ADAPTER=titan`) |
| metalForge live | **29/29 PASS** (5 substrates, 18 workloads route) |
| Atlas stream (real data) | **73/73 PASS** (12 stations, 4800 crop-year results) |
| GPU dispatch (P0 blocker) | **RESOLVED** — S66 explicit BGL (R-S66-041) |
| try_gpu catch_unwind debt | **REMOVED** — S66+ resolved sovereign compiler regression |
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
| `ValidationHarness` | neuralSpring | All 63 validation binaries |
| `norm_ppf` (Moro 1995) | hotSpring | MC ET₀ parametric confidence intervals |
| `brent` (Brent 1973) | neuralSpring | VG pressure head inversion (θ→h) |
| `pde::richards` | airSpring → upstream | 1D Richards equation (absorbed S40) |
| `stats::regression` | airSpring metalForge → upstream | Sensor correction fitting (absorbed S66) |
| `stats::hydrology` | airSpring metalForge → upstream | Hargreaves ET₀, batch, crop_coefficient (absorbed S66) |
| `stats::moving_window_f64` | airSpring metalForge → upstream | f64 stream statistics (absorbed S66) |
| `stats::diversity::bray_curtis_matrix` | wetSpring → upstream | Full M×M distance matrix for ordination (wired v0.5.2) |
| `stats::diversity::shannon_from_frequencies` | wetSpring → upstream | Pre-normalised Shannon for streaming pipelines (wired v0.5.2) |

### airSpring Contributions Back

| Fix | Impact | Commit |
|-----|--------|--------|
| TS-001: `pow_f64` fractional exponent | All Springs using VG/exponential math | S54 (H-011) |
| TS-003: `acos` precision boundary | All Springs using trig in f64 shaders | S54 (H-012) |
| TS-004: reduce buffer N≥1024 | All Springs using `FusedMapReduceF64` | S54 (H-013) |
| Richards PDE | airSpring → `pde::richards` (S40) | upstream |

---

## Cross-Spring Sync (Feb 27, 2026)

### Sibling Spring Handoff Review

| Handoff | Date | ToadStool Baseline | Key Takeaways for airSpring |
|---------|------|--------------------|-----------------------------|
| wetSpring V61 | Feb 27 | S68 (`f0feb226`) | NPU inference bridge proposed (`barracuda::npu`); power-budget-aware dispatch; 79 ToadStool primitives in use |
| neuralSpring V24 | Feb 27 | S68 (`f0feb226`) | `compile_shader_df64_streaming` proposed; `barracuda::nn` (MLP, LSTM, ESN); two-tier df64 precision validated |
| groundSpring V10 | Feb 25 | S50–S62 | `if let Ok` + CPU fallback pattern (wateringHole standard); `mc_et0_propagate` ready; three-mode CI (local/barracuda/barracuda-gpu) |
| ToadStool S61-63 | Feb 25 | S61–63 | Sovereign compiler; cyclic reduction for n≥2048; maximin LHS O(n); `erfc_deriv` public |

### Pending Upstream Absorptions to Track

| Primitive | Proposed By | Impact on airSpring | Status |
|-----------|-------------|---------------------|--------|
| `barracuda::npu` (NpuDispatch trait) | wetSpring V61 + airSpring V024 | Would replace our local `npu.rs` | Proposed |
| `barracuda::nn` (MLP, LSTM, ESN) | neuralSpring V24 | ML/regime surrogates for crop modeling | Proposed |
| `compile_shader_df64_streaming` | neuralSpring V24 | Simplify df64 shader compilation | Proposed |
| `barracuda::ml::esn` | wetSpring V61 | ESN reservoir for time-series IoT | Proposed |
| `batched_multinomial.wgsl` | groundSpring V10 | Rarefaction for diversity GPU | Proposed |

### S68+ HEAD Sync (e96576ee) — Full Review Feb 28, 2026

ToadStool S50–S68+ represents 29 commits since Feb 25, touching 779 files (+21,891/−13,831 lines).
Reviewed at `e96576ee`. All airSpring imports verified — **zero breaking changes**.

**Major evolution absorbed** (S50–S68):
- S50-S56: Deep audit, cross-spring absorption, idiomatic Rust, coverage push (+193 tests)
- S57: +47 tests, println→tracing migration, coverage to 4,224+ core tests
- S58-S59: DF64, Fp64Strategy, ridge, ValidationHarness absorption (anderson correlated)
- S60-S63: DF64 FMA, sovereign compiler, SPIR-V passthrough, CN f64 GPU shader
- S64-S65: Stats absorption from all Springs, smart refactoring, doc cleanup
- S66: **Cross-spring absorption** — regression, hydrology, 8 SoilParams, P0 BGL fix
- S67: Universal precision doctrine — "math is universal, precision is silicon"
- S68: **296 f32-only shaders removed** — ZERO f32-only, all f64 canonical
  - `op_preamble()` → abstract math ops for precision-parametric shaders
  - `df64_rewrite.rs` → naga IR rewrite: f64 infix → DF64 bridge calls
  - `compile_op_shader(source, precision, label)` → one source, any precision
  - 122 dedicated shader tests (unit + e2e + chaos + fault)
- S68+: GPU device-lost resilience, root doc cleanup, archive stale scripts

**Universal precision architecture** now fully available:
- `compile_shader_universal(source, precision, label)` → one f64 source compiles to F16/F32/F64/Df64
- `compile_op_shader(source, precision, label)` → preamble injection for abstract ops
- `Fp64Strategy::Native` (Titan V, A100) vs `Fp64Strategy::Hybrid` (RTX 4070, consumer GPUs)
- `op_preamble()` → abstract math ops (`op_add`, `op_mul`, `Scalar` type alias) resolve per precision
- `df64_rewrite.rs` → naga IR rewrite transforms f64 infix → DF64 bridge calls
- `downcast_f64_to_df64()` → text-based fallback when naga rewrite unavailable
- 700 WGSL shaders total (497 f32 via downcast, 182 native f64, 21 df64)
- 2,546+ barracuda unit tests, 21,599+ workspace tests

**ToadStool handoff sync gap**: ToadStool has processed airSpring through **V009** (S66 absorption).
airSpring handoffs V010–V031 are pending upstream absorption. V032 created to acknowledge S68 sync.

**airSpring V033 cross-spring rewiring**:
- **Rewired `gpu::hargreaves`** — CPU batch now delegates to `barracuda::stats::hargreaves_et0_batch` (ToadStool S66)
- **Wired `eco::diversity::bray_curtis_matrix`** — full M×M distance matrix for ordination (wetSpring S64)
- **Wired `eco::diversity::shannon_from_frequencies`** — pre-normalised Shannon for streaming 16S (wetSpring S66)
- **Wired `eco::crop::crop_coefficient_stage`** — delegates to `barracuda::stats::crop_coefficient` (airSpring metalForge → S66)
- **Cleaned `gpu::richards::solve_cn_diffusion`** — consolidated SoilParams via `to_barracuda_params()`
- **Expanded `bench_cross_spring` v0.5.2** — 30 benchmarks (was 16), 16 shader provenance entries (was 10), 45 primitives, 6 origin Springs
- **New benchmarks**: Hargreaves batch (365/10K), diversity alpha, Bray-Curtis matrix (20 samples), Shannon frequencies, crop Kc stage (180d), Kc from GDD (corn), Anderson coupling chain (10K θ), Anderson regimes
- **Expanded PROVENANCE** table: added hydrology batch kernel (airSpring), diversity bio kernel (wetSpring), anderson coupling kernel (groundSpring)
- 618 lib tests (+5 from new rewiring), 0 clippy warnings, 0 errors

Prior V032 cleanup:
- Registered `validate_gpu_math` and `validate_ncbi_16s_coupling` in Cargo.toml (were unregistered)
- Fixed 2 clippy `manual_clamp` warnings in `validate_ncbi_16s_coupling.rs`
- Prior V030-V031: removed `try_gpu` catch_unwind, updated docs for universal precision,
  added `gpu::device_info` (Fp64Strategy probing), added `bench_cross_spring` (16→30 benchmarks)

Revalidation: 618/618 tests, 0 clippy, 33/33 cross-validation, 1498/1498 atlas, 46/46 GPU math, 29/29 NCBI 16S, 30/30 benchmarks

### S89–S93 HEAD Sync — barraCuda Standalone Rewire (March 3, 2026)

ToadStool S87→S93 includes the most significant architectural change: **barraCuda extraction**
as a standalone sovereign math engine (S89), followed by sovereignty cleanup (S90–S92) and
DF64 ownership transfer (S93).

**S88**: Cross-spring absorption — `anderson_4d`, `wegner_block_4d`, `SeasonalGpuParams::new()`,
`LbfgsGpu`, new hydrology tolerances.

**S89**: **barraCuda extraction** — 956 Rust files, 767 WGSL shaders extracted from ToadStool
into `ecoPrimals/barraCuda`. `toadstool-core`, `akida-driver` dependencies removed. All Springs
now depend on barraCuda directly. hotSpring validated 716/716 with path swap only.

**S90**: REST API removed (JSON-RPC 2.0 only), `get_socket_path_for_capability()`,
SPDX headers universal, 5,322 tests.

**S92**: Sovereignty deprecations, +47 tests (5,369 total), BearDog strings neutralized.

**S93**: D-DF64 ownership transferred to barraCuda team, 12 stale docs removed, root docs refreshed.

**airSpring rewire**: Updated `barracuda` path from `../../phase1/toadstool/crates/barracuda` →
`../../barraCuda/crates/barracuda` (both `barracuda/Cargo.toml` and `metalForge/forge/Cargo.toml`).
Zero code changes needed — all imports, traits, shader references stable.

Revalidation: 1132/1132 tests, 0 clippy warnings (pedantic), 0 fmt diffs, docs build clean.

---

## Dependency Evolution Analysis (v0.6.8)

### Direct Dependencies

| Crate | Version | C deps? | Purpose | Evolution Path |
|-------|---------|---------|---------|----------------|
| `barracuda` | 0.3.1 (path) | wgpu (vulkan) | GPU primitives, stats, validation | **Core** — standalone primal (`ecoPrimals/barraCuda`) |
| `bingocube-nautilus` | 0.1.0 (path) | None | Evolutionary reservoir computing | **Core** — stays, pure Rust |
| `serde` | 1.0 | None | Brain state serialization | **Stays** — pure Rust, ecosystem standard |
| `serde_json` | 1.0 | None | Benchmark JSON + JSON-RPC | **Stays** — pure Rust, ecosystem standard |
| `tracing-subscriber` | 0.3 | None | Validation output logging | **Stays** — pure Rust, ecosystem standard |
| `ureq` | 3.2 | **ring** (C/asm via rustls) | HTTP client (data providers) | **Evolve** → Songbird (sovereign TLS) |

### Transitive C/Assembly Dependencies

| Crate | Pulled By | C/ASM? | Sovereignty Risk | Evolution |
|-------|-----------|--------|-----------------|-----------|
| `ring` 0.17 | ureq → rustls | **Yes** (C, assembly) | **Medium** — crypto primitives are C/asm | **Evolve**: ureq → Songbird (BearDog pure-Rust TLS 1.3) |
| `wgpu` (via barracuda) | barracuda | Vulkan driver | **Low** — GPU driver is inherently platform-specific | Stays — hardware interface |

### Evolution Path: `ureq` → Songbird Capability

`ureq` is the only dependency pulling C code (`ring` via `rustls`). The evolution:

1. **Current (standalone)**: `ureq` for direct HTTPS to Open-Meteo, NOAA, etc.
2. **Sovereign**: `Songbird` pure-Rust TLS 1.3 via BearDog crypto delegation.
   Route HTTPS through `capability.call("tls", "request", {...})`.
3. **Discovery**: `data::provider::discover_transport()` already selects Songbird
   when `SONGBIRD_SOCKET` is set, falling back to ureq otherwise.

The transport tier is already abstracted — when Tower Atomic is running, all
HTTPS routes through Songbird. No airSpring code changes needed; the dependency
simply becomes unused.

### Audit Results

- `cargo deny check`: **Clean** — all dependencies AGPL/MIT/Apache/BSD licensed
- `#![forbid(unsafe_code)]`: Both crates — no unsafe Rust
- No `openssl`, `reqwest`, or other heavy C dependencies
- Pure Rust stack except `ring` (via ureq→rustls) and GPU drivers (via wgpu)

### Quality Gates (v0.6.8 — barraCuda 0.3.1 rewire)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** (both crates) |
| `cargo clippy --workspace -- -D warnings -W clippy::pedantic` | **PASS** — 0 warnings (both crates) |
| `cargo doc --no-deps` | **PASS** (both crates) |
| `cargo test --workspace` | **1132 PASS** (lib + bin + doc + integration) |
| `cargo llvm-cov --lib --summary-only` | **95.11% line** / **95.81% function** coverage |
| `cargo deny check` | **PASS** |
| SPDX headers | **All .rs files**: `AGPL-3.0-or-later` |
| File size limit | **All files < 1000 lines** (max: 935, bench binary) |
| `#![forbid(unsafe_code)]` | **Both crates** |
| Validation provenance | **All 79 binaries** have script/commit/date or cross-spring provenance |
| Tolerance provenance | **47/47 constants** with mathematical justification + baseline table |
| barraCuda source | **`ecoPrimals/barraCuda/crates/barracuda`** v0.3.1 standalone |
