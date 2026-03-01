// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolution gaps: airSpring CPU→GPU migration roadmap.
//!
//! This module documents the gaps between airSpring's validated CPU pipeline
//! and the GPU-accelerated `BarraCuda` path. It serves as a living roadmap.
//!
//! # Gap Categories
//!
//! - **Integrated**: GPU primitive wired and validated (GPU-first, CPU fallback).
//! - **Ready**: `ToadStool` primitive exists, airSpring just needs to wire it.
//! - **Needs Adaptation**: Shader exists, needs domain customisation.
//! - **Needs Primitive**: No `ToadStool` implementation yet.
//!
//! # Shader Promotion Mapping
//!
//! | Rust Module | GPU Orchestrator | WGSL Shader | Pipeline Stage | Tier |
//! |---|---|---|---|---|
//! | `eco::evapotranspiration` | `gpu::et0` | `batched_elementwise_f64.wgsl` (op=0) | ET₀ computation | A (ready) |
//! | `eco::water_balance` | `gpu::water_balance` | `batched_elementwise_f64.wgsl` (op=1) | Daily water balance | B (needs `BatchedStatefulF64`) |
//! | `eco::dual_kc` | `gpu::dual_kc` | `batched_elementwise_f64.wgsl` (op=8) | Crop coefficient | A (GPU-first, S70+ absorbed) |
//! | `eco::soil_moisture` | `gpu::kriging` | `kriging_f64.wgsl` | Spatial interpolation | A (ready) |
//! | `eco::richards` | `gpu::richards` | `pde_richards.wgsl` | PDE solve | A (ready) |
//! | `eco::isotherm` | `gpu::isotherm` | `nelder_mead.wgsl` | Isotherm fitting | B (needs batch NM) |
//! | `testutil` | `gpu::reduce` | `fused_map_reduce_f64.wgsl` | Seasonal stats | A (ready) |
//! | `eco::sensor_calibration` | `gpu::sensor_calibration` | `batched_elementwise_f64.wgsl` (op=5) | Sensor VWC | A (GPU-first, S70+ absorbed) |
//! | `eco::evapotranspiration` (HG) | `gpu::hargreaves` | `batched_elementwise_f64.wgsl` (op=6) | Hargreaves ET₀ | A (GPU-first, S70+ absorbed) |
//! | `eco::crop` (Kc adj) | `gpu::kc_climate` | `batched_elementwise_f64.wgsl` (op=7) | Kc climate adjust | A (GPU-first, S70+ absorbed) |
//! | `eco::*` (pipeline) | `gpu::seasonal_pipeline` | Chained ops 0→7→1→yield | Seasonal pipeline | **GPU Stages 1-2** (v0.5.6) |
//! | `eco::*` (stream) | `gpu::atlas_stream` | Unified batch + streaming callback | Regional atlas | **GPU-capable** (v0.5.6) |
//! | `eco::*` (stream) | `gpu::seasonal_pipeline` | `Backend::GpuPipelined`, `Backend::GpuFused` | Streaming pipeline | **v0.5.7** |
//! | `eco::*` (MC) | `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` (pending) | MC uncertainty | B (wired, pending shader) |
//! | `io::csv_ts` | `gpu::stream` | `moving_window.wgsl` | Stream smoothing | A (ready) |
//! | `eco::sensor_calibration` (OLS) | `gpu::stats` | `linear_regression_f64.wgsl` | Sensor regression | A (GPU, neuralSpring S69) |
//! | `eco::*` (multi-var) | `gpu::stats` | `matrix_correlation_f64.wgsl` | Soil correlation | A (GPU, neuralSpring S69) |
//!
//! # Current Inventory (March 1, 2026 — v0.5.9, synced to `ToadStool` HEAD `8dc01a37`)
//!
//! `ToadStool` S42–S71: 200+ commits, 50+ cross-spring absorptions, 2,773+ barracuda tests, 671 WGSL shaders.
//! All four airSpring issues (TS-001 through TS-004) resolved in **S54**.
//! P0 GPU dispatch blocker resolved in **S66** (explicit `BindGroupLayout`).
//! S71 synced: DF64 transcendentals complete (15 functions), 66 `ComputeDispatch` ops,
//! `HargreavesBatchGpu`, `JackknifeMeanGpu`, `BootstrapMeanGpu`, `HistogramGpu`, `KimuraGpu`,
//! `fao56_et0` scalar, HMM log-domain dispatch, pure math precision-per-silicon doctrine.
//!
//! ## Universal Precision Architecture (S67–S68)
//!
//! All WGSL shaders are **f64 canonical** — written in f64, compiled to target
//! precision via `compile_shader_universal(source, precision, label)`:
//! - `Precision::F64` → native builtins (Titan V, A100)
//! - `Precision::Df64` → double-float f32-pair ~48-bit (consumer GPUs)
//! - `Precision::F32` → downcast via `downcast_f64_to_f32()` (backward compat)
//! - `Precision::F16` → downcast via `downcast_f64_to_f16()` (edge inference)
//!
//! `Fp64Strategy::Native` vs `Fp64Strategy::Hybrid` is auto-selected per-device
//! by `GpuDriverProfile::fp64_strategy()` based on f64:f32 throughput ratio.
//!
//! Upstream capabilities available (S51–S68+):
//! - S51+: `solve_f64_cpu()`, `GpuSessionBuilder`, `OdeSystem` trait + `BatchedOdeRK4`
//! - S52+: `NelderMeadGpu`, `BatchedBisectionGpu`, `chi2_decomposed`, `FusedMapReduceF64::dot()`
//! - S54+: TS-001–004 resolved, `barracuda::tolerances`, `barracuda::provenance`
//! - S58+: `df64`, `Fp64Strategy`, ridge regression, `ValidationHarness`
//! - S60+: DF64 FMA, `norm_cdf`/`norm_ppf`, `empirical_spectral_density`
//! - S62+: `BandwidthTier`, `PeakDetectF64`, `CrankNicolson1D` (f64 + GPU shader!)
//! - S64: Stats absorption (metrics, diversity, bootstrap from Springs)
//! - S65: Smart refactoring, dead code removal, doc cleanup
//! - S66: **Cross-spring absorption** — regression, hydrology, `moving_window_f64`,
//!   `spearman_correlation` re-export, 8 named `SoilParams` constants, `mae`,
//!   `hill`/`monod`, `shannon_from_frequencies`, `rawr_mean`, multi-precision WGSL.
//!   **P0 fix**: explicit `BindGroupLayout` replaces `layout: None` + `get_bind_group_layout(0)`.
//! - S67: Codified "math is universal — precision is silicon" doctrine
//! - S68: **Universal precision** — 296 f32-only shaders removed, all f64 canonical,
//!   `downcast_f64_to_f32()` for backward compat, `op_preamble()` for abstract math,
//!   `df64_rewrite.rs` naga IR rewrite. `ValidationHarness` migrated to `tracing::info!`.
//! - S68+: GPU device-lost resilience, root doc cleanup, archive stale scripts
//! - S68++–S68+++: Ecosystem audit, AGPL-3 license, chrono elimination, dead code cleanup
//! - S69++: `ComputeDispatch` migration, architecture evolution
//! - S70: Deep debt — modern idiomatic concurrent Rust, archive cleanup
//! - S70+: **Cross-spring absorption** — ops 5–8 (`SensorCal`, `Hargreaves`, `KcClimate`, `DualKc`),
//!   `seasonal_pipeline.wgsl`, `brent_f64.wgsl`, `stats::hydrology`, `stats::jackknife`,
//!   `stats::diversity`, `stats::evolution`, `nn::simple_mlp`, `staging::pipeline`,
//!   new WGSL shaders: `anderson_coupling_f64`, `lanczos_iteration_f64`,
//!   `linear_regression_f64`, `matrix_correlation_f64`
//! - S70++: Sovereignty, architecture, monitoring split, stub evolution
//! - S70+++: Builder refactor, dead code removal, monitoring evolution
//! - S71: **GPU dispatch wiring + sovereignty** — `HargreavesBatchGpu` (science shader),
//!   `JackknifeMeanGpu`, `BootstrapMeanGpu`, `HistogramGpu`, `KimuraGpu` (bio),
//!   `HmmForwardLogF32`/`F64`, `fao56_et0` scalar PM, `df64_transcendentals.wgsl`
//!   (asin, acos, atan, atan2, sinh, cosh, gamma, erf → 15 DF64 functions complete),
//!   `hargreaves_batch_f64.wgsl`, `jackknife_mean_f64.wgsl`, `kimura_fixation_f64.wgsl`,
//!   66 `ComputeDispatch` migrations, 14 reduction+FFT+index `ComputeDispatch` migrations,
//!   all files < 1000 lines, hardcoded primal names → `primals::*` constants,
//!   `jsonrpc_server.rs` 904→628, `network_config/types.rs` split 7 modules,
//!   5 stale examples deleted, 2 stub test files deleted. Pure math + precision per silicon.
//!
//! ## Cross-Spring Shader Provenance (validated in `cross_spring_absorption.rs` §13)
//!
//! | Spring | Domain | Contributed to `ToadStool` |
//! |--------|--------|------------------------|
//! | hotSpring | Nuclear/precision physics | `df64_core`, `math_f64`, `complex_f64`, SU(3), Hermite/Laguerre, Lanczos |
//! | wetSpring | Bio/environmental | Shannon/Simpson/Bray-Curtis, kriging, `moving_window`, Hill, ODE bio, NMF |
//! | neuralSpring | ML/optimization | Nelder-Mead, `ValidationHarness`, pairwise metrics, batch IPR, matmul |
//! | airSpring | Precision agriculture | regression, hydrology, `moving_window_f64`, Richards PDE (S40), TS-001/003/004 fixes |
//! | groundSpring | Uncertainty/stats | MC ET₀ propagation, `batched_multinomial`, `rawr_mean` |
//!
//! **Key evolution since V011**: `pde::crank_nicolson` is now **f64** with
//! `WGSL_CRANK_NICOLSON_F64` GPU shader — previously documented as f32-only.
//! Optimizers expanded: `bfgs`, `brent`, `newton`, `secant`, `bisect` all available.
//! `ResumableNelderMead` adds solver state persistence.
//!
//! ## Tier A: Integrated (GPU primitive wired, validated, GPU-first)
//!
//! | airSpring Module | `ToadStool` Primitive | Status |
//! |-----------------|--------------------|----|
//! | `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** via `fao56_et0_batch()` |
//! | `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** via `water_balance_batch()` |
//! | `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **INTEGRATED** — ordinary kriging via LU |
//! | `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64::FusedMapReduceF64` | **GPU for N≥1024** (TS-004 resolved) |
//! | `validation::ValidationHarness` | `barracuda::validation::ValidationHarness` | **ABSORBED** — leaning on upstream (S59) |
//! | R² / Pearson correlation | `stats::pearson_correlation` | **Already wired** (testutil) |
//! | Variance / std deviation | `stats::correlation::variance`, `std_dev` | **Already wired** (integration tests) |
//! | `gpu::stream::StreamSmoother` | `ops::moving_window_stats::MovingWindowStats` | **WIRED** — `IoT` stream smoothing (wetSpring S28+) |
//! | `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **WIRED** — calibration regression (wetSpring ESN) |
//! | Spearman rank correlation | `stats::correlation::spearman_correlation` | Available, used in testutil |
//! | Bootstrap confidence intervals | `stats::bootstrap::bootstrap_ci` | WIRED (`testutil::bootstrap`) |
//! | Crank-Nicolson PDE (f64) | `pde::crank_nicolson::CrankNicolson1D` | Available — f64 + GPU shader (S62+) |
//! | BFGS quasi-Newton optimizer | `optimize::bfgs` | Available — smooth objective fitting |
//! | Brent VG inverse | `optimize::brent` | **WIRED** — `inverse_van_genuchten_h()` θ→h (v0.4.5) |
//! | `norm_ppf` MC CI | `stats::normal::norm_ppf` | **WIRED** — `McEt0Result::parametric_ci()` (v0.4.5) |
//! | Newton / secant methods | `optimize::newton`, `secant` | Available — derivative-based roots |
//! | Bisection (scalar) | `optimize::bisect` | Available — robust bracketed root |
//! | Batched bisection (GPU) | `optimize::BatchedBisectionGpu` | Available — parallel root-finding |
//! | Normal CDF | `stats::normal::norm_cdf` | Available — cumulative probability |
//! | Chi-squared decomposition | `stats::chi2::chi2_decomposed` | Available (S52+) |
//! | Spectral density / RMT | `stats::spectral_density::empirical_spectral_density` | Available (S57+) |
//! | Resumable Nelder-Mead | `optimize::ResumableNelderMead` | Available — checkpoint/resume |
//!
//! ## Tier B: Upstream Primitive Exists, Needs Domain Wiring
//!
//! | Need | Closest `ToadStool` Primitive | Gap |
//! |------|---------------------------|-----|
//! | 1D Richards equation | `pde::richards::solve_richards` (VG-Mualem, Picard+CN+Thomas) | **PROMOTED to Tier A** — `gpu::richards::BatchedRichards` (v0.4.0) |
//! | Crank-Nicolson cross-val | `pde::crank_nicolson::CrankNicolson1D` (f64 + GPU) | **NEW** — available for Richards CN comparison (was f32-only, now f64) |
//! | Sensor calibration (batch) | `batched_elementwise_f64.wgsl` (custom op) | Add `SoilWatch` 10 as op=5 |
//! | Hargreaves ET₀ (batch) | `batched_elementwise_f64.wgsl` | Add as op=6 (simpler than PM) |
//! | Kc climate adjustment (batch) | `batched_elementwise_f64.wgsl` | Add as op=7 |
//! | Moving window statistics | `ops::moving_window_stats` | **PROMOTED to Tier A** — `gpu::stream::StreamSmoother` |
//! | Nonlinear fitting | `optimize::{nelder_mead, bfgs, NelderMeadGpu}` | **WIRED (NM)** — BFGS available for smooth objectives |
//! | Ridge regression | `linalg::ridge::ridge_regression` | **PROMOTED to Tier A** — `eco::correction::fit_ridge` |
//! | Root-finding (batch GPU) | `optimize::BatchedBisectionGpu` | Cross-spring from `wetSpring` — soil water potential inversion |
//! | Root-finding (scalar) | `optimize::{brent, bisect, newton, secant}` | Available for any scalar root problem |
//! | Tridiagonal solve (batch) | `linalg::tridiagonal_solve_f64`, `ops::cyclic_reduction_f64` | Wired via `pde::richards` (v0.4.0) |
//! | Adaptive ODE (RK45) | `numerical::rk45_solve` (Dormand-Prince) | Available for dynamic soil models |
//! | Batch isotherm fitting | `optimize::multi_start_nelder_mead` | **WIRED** — `gpu::isotherm::fit_batch_global()` (v0.4.1) |
//!
//! ## Tier C: Needs New `ToadStool` Primitives
//!
//! | Need | Description | Complexity |
//! |------|-------------|-----------|
//! | HTTP/JSON data client | Open-Meteo, NOAA CDO APIs | Low — not GPU, but needed |
//!
//! ## Deprecated Patterns (Clean Up)
//!
//! | Pattern | Status | Replacement |
//! |---------|--------|-------------|
//! | `rayon` dependency | **Removed** (v0.2.0) | `ToadStool` GPU dispatch replaces thread pool |
//! | Ad-hoc `String` errors | **Replaced** (v0.2.0) | `AirSpringError` enum |
//! | `HashMap` CSV storage | **Replaced** (v0.2.0) | Columnar `Vec<Vec<f64>>` |
//! | Hardcoded runoff model | **Replaced** (v0.2.0) | `RunoffModel` enum |
//! | CPU-only GPU stubs | **Replaced** (v0.3.0) | GPU-first via `BatchedElementwiseF64` |
//! | Local `ValidationRunner` | **Replaced** (v0.3.6) | `barracuda::validation::ValidationHarness` |
//!
//! ## Shader Precision (All Resolved)
//!
//! All precision issues fixed in `ToadStool` commit `0c477306`:
//! - `pow_f64`: Now uses `exp_f64(exp * log_f64(base))` for non-integer exponents (**TS-001**)
//! - `acos_f64`: Full-precision from `math_f64.wgsl` wired in (**TS-003**)
//! - `sin_f64`: Full-precision from `math_f64.wgsl` wired in (**TS-003**)
//! - `BatchedElementwiseF64` Rust orchestrator: Created (**TS-002**)
//! - `FusedMapReduceF64` buffer conflict: Resolved (**TS-004**)
//!
//! ## Cross-Validation Strategy
//!
//! GPU paths are validated against CPU baselines:
//! 1. CPU validation remains source of truth (515 lib tests, 51 binaries, 1393 atlas checks)
//! 2. GPU results must match CPU within documented tolerance
//! 3. Cross-validation harness (33/33 Python↔Rust) extends to GPU path
//! 4. Each GPU function has a `test_gpu_matches_cpu_*` integration test
//! 5. GPU determinism proven: 4 bit-identical rerun tests (`gpu_determinism.rs`)
//! 6. Titan V live validation: 24/24 GPU parity tests PASS (0.04% seasonal tolerance)
//! 7. S68+ universal precision: f64 canonical shaders compile to any precision target

/// Structured representation of an evolution gap.
#[derive(Debug)]
pub struct EvolutionGap {
    /// Short identifier.
    pub id: &'static str,
    /// Human description of the gap.
    pub description: &'static str,
    /// Current tier (A=integrated/ready, B=adapt, C=new).
    pub tier: Tier,
    /// What `ToadStool` provides (if anything).
    pub toadstool_primitive: Option<&'static str>,
    /// What airSpring needs to do.
    pub action: &'static str,
}

/// Evolution tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// GPU primitive integrated and validated.
    A,
    /// Shader exists, needs adaptation for airSpring domain.
    B,
    /// Needs new `ToadStool` primitive.
    C,
}

/// All known evolution gaps (25+6 entries — 17 Tier A integrated, 7 Tier B + 6 orchestrators, 1 Tier C).
///
/// v0.5.6: Ops 5–8 promoted Tier B→A after `ToadStool` S70+ absorption.
/// `SeasonalPipeline::gpu()` dispatches Stages 1-2 (ET₀ + Kc) to GPU;
/// `AtlasStream::with_gpu()` + `process_streaming()` callback pattern.
/// Synced to `ToadStool` S70+++ (1dd7e338). Universal precision architecture
/// means all GPU dispatch is precision-agnostic: f64 on Titan V, Df64 on consumer
/// GPUs, f32 fallback. S60-S65 sovereign compiler regression **RESOLVED** (S66+).
///
/// Key upstream capabilities (S52-S68+):
/// - `compile_shader_universal(source, precision, label)`: One shader → any precision
/// - `Fp64Strategy::Native/Hybrid`: Auto-selected per-device
/// - `op_preamble()`: Abstract math ops for precision-parametric shaders
/// - `probe_f64_builtins(device)`, `probe_f64_throughput_ratio(device)`: Hardware probing
/// - `NelderMeadGpu`: GPU-resident optimizer (5-50 params)
/// - `BatchedBisectionGpu`: GPU-parallel batched root-finding
/// - `UnidirectionalPipeline`: Fire-and-forget GPU streaming, eliminates round-trip overhead
/// - `StatefulPipeline`: GPU-resident iterative solvers (minimal readback)
/// - `MultiDevicePool`: Multi-GPU dispatch with load balancing
/// - `pde::crank_nicolson::CrankNicolson1D`: **f64** CN PDE solver + GPU shader
/// - `optimize::{bfgs, brent, newton, secant, bisect}`: Full optimizer suite
/// - `optimize::ResumableNelderMead`: Checkpoint/resume for long-running optimization
/// - `optimize::adaptive_penalty`: Constrained optimization with data-driven penalty
/// - `unified_hardware`: `HardwareDiscovery`, `ComputeScheduler`, `MixedSubstrate`
pub const GAPS: &[EvolutionGap] = &[
    // ── Tier A: Integrated (GPU primitive wired and validated) ─────────
    EvolutionGap {
        id: "batched_et0_gpu",
        description: "Batched FAO-56 ET₀ on GPU for N station-days",
        tier: Tier::A,
        toadstool_primitive: Some("ops::batched_elementwise_f64::BatchedElementwiseF64 (op=0)"),
        action: "GPU-FIRST — BatchedEt0::gpu() → fao56_et0_batch() (TS-001/002 resolved)",
    },
    EvolutionGap {
        id: "batched_water_balance_gpu",
        description: "Batched water balance depletion update on GPU",
        tier: Tier::A,
        toadstool_primitive: Some("ops::batched_elementwise_f64::BatchedElementwiseF64 (op=1)"),
        action:
            "GPU-STEP — BatchedWaterBalance::gpu_step() → water_balance_batch() (TS-002 resolved)",
    },
    EvolutionGap {
        id: "kriging_soil_moisture",
        description: "Spatial interpolation of soil moisture from sensor network",
        tier: Tier::A,
        toadstool_primitive: Some("ops::kriging_f64::KrigingF64"),
        action: "INTEGRATED — KrigingInterpolator wraps KrigingF64 (ordinary kriging via LU)",
    },
    EvolutionGap {
        id: "fused_reduce_stats",
        description: "GPU-accelerated batch reductions (sum, max, min) for ET₀ totals",
        tier: Tier::A,
        toadstool_primitive: Some("ops::fused_map_reduce_f64::FusedMapReduceF64"),
        action:
            "INTEGRATED — SeasonalReducer wraps FusedMapReduceF64 (GPU for N≥1024, TS-004 resolved)",
    },
    EvolutionGap {
        id: "bootstrap_uncertainty",
        description: "Bootstrap confidence intervals for ET₀ and water balance",
        tier: Tier::A,
        toadstool_primitive: Some("stats::bootstrap::bootstrap_ci"),
        action: "WIRED (testutil::bootstrap_rmse) — already using barracuda::stats",
    },
    EvolutionGap {
        id: "validation_harness",
        description: "Structured pass/fail validation with exit codes",
        tier: Tier::A,
        toadstool_primitive: Some("barracuda::validation::ValidationHarness"),
        action: "ABSORBED — local ValidationRunner replaced, leaning on upstream (S59)",
    },
    EvolutionGap {
        id: "moving_window_stream",
        description: "Sliding window statistics for IoT sensor stream smoothing",
        tier: Tier::A,
        toadstool_primitive: Some("ops::moving_window_stats::MovingWindowStats"),
        action: "WIRED — gpu::stream::StreamSmoother wraps MovingWindowStats (wetSpring S28+)",
    },
    EvolutionGap {
        id: "ridge_calibration",
        description: "Ridge regression for sensor calibration pipeline",
        tier: Tier::A,
        toadstool_primitive: Some("linalg::ridge::ridge_regression"),
        action: "WIRED — eco::correction::fit_ridge wraps barracuda ridge (wetSpring ESN)",
    },
    EvolutionGap {
        id: "norm_ppf_mc_ci",
        description: "Parametric confidence intervals for MC ET₀ via norm_ppf",
        tier: Tier::A,
        toadstool_primitive: Some("stats::normal::norm_ppf (Moro 1995 rational approx)"),
        action: "WIRED (v0.4.5): McEt0Result::parametric_ci() uses norm_ppf for z-scores \
                 (hotSpring special-function lineage → barracuda::stats S52+)",
    },
    EvolutionGap {
        id: "brent_vg_inverse",
        description: "VG pressure head inversion via Brent root-finding",
        tier: Tier::A,
        toadstool_primitive: Some("optimize::brent (Brent 1973, guaranteed convergence)"),
        action: "WIRED (v0.4.5): eco::richards::inverse_van_genuchten_h() uses brent \
                 for θ→h inversion (neuralSpring optimizer lineage → barracuda::optimize S52+)",
    },
    EvolutionGap {
        id: "dual_kc_batch",
        description: "Batched dual Kc (Ke + ETc) across M fields per timestep",
        tier: Tier::A,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (op=8, stride=9)"),
        action: "GPU-FIRST (v0.5.6): gpu::dual_kc step_gpu() → Op::DualKcKe (S70+ absorbed)",
    },
    EvolutionGap {
        id: "sensor_calibration_batch",
        description: "Batch sensor calibration (SoilWatch 10) via custom op",
        tier: Tier::A,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (op=5, stride=1)"),
        action:
            "GPU-FIRST (v0.5.6): gpu::sensor_calibration → Op::SensorCalibration (S70+ absorbed)",
    },
    EvolutionGap {
        id: "hargreaves_batch",
        description: "Hargreaves ET₀ as batch GPU op (simpler than PM, fewer inputs)",
        tier: Tier::A,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (op=6, stride=4)"),
        action: "GPU-FIRST (v0.5.6): gpu::hargreaves → Op::HargreavesEt0 (S70+ absorbed)",
    },
    EvolutionGap {
        id: "kc_climate_adjust",
        description: "Kc climate adjustment (FAO-56 Eq. 62) as batch GPU op",
        tier: Tier::A,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (op=7, stride=4)"),
        action: "GPU-FIRST (v0.5.6): gpu::kc_climate → Op::KcClimateAdjust (S70+ absorbed)",
    },
    // ── Tier B: Shader exists, needs domain adaptation ────────────────
    EvolutionGap {
        id: "nonlinear_solver",
        description: "Nonlinear least squares for soil calibration curve fitting",
        tier: Tier::B,
        toadstool_primitive: Some(
            "optimize::{nelder_mead, bfgs, NelderMeadGpu, ResumableNelderMead}",
        ),
        action: "NM WIRED (v0.4.1); BFGS available for smooth objectives; \
                 NelderMeadGpu for 5-50 param problems",
    },
    EvolutionGap {
        id: "richards_pde",
        description: "1D Richards equation for unsaturated soil water flow",
        tier: Tier::A,
        toadstool_primitive: Some(
            "pde::richards::solve_richards (van Genuchten-Mualem, Picard + CN + Thomas)",
        ),
        action: "WIRED (v0.4.0): gpu::richards::BatchedRichards wraps barracuda::pde::richards. \
                 pde::crank_nicolson now f64 + GPU shader for CN cross-validation.",
    },
    EvolutionGap {
        id: "tridiagonal_batch",
        description: "Tridiagonal solver for implicit PDE time-stepping",
        tier: Tier::B,
        toadstool_primitive: Some("linalg::tridiagonal_solve_f64, ops::cyclic_reduction_f64"),
        action: "Wired via pde::richards (v0.4.0) — direct use available for other PDE solvers",
    },
    EvolutionGap {
        id: "rk45_adaptive",
        description: "Adaptive RK45 ODE solver for dynamic soil/water models",
        tier: Tier::B,
        toadstool_primitive: Some("numerical::rk45_solve (Dormand-Prince, adaptive step)"),
        action: "Available upstream — wire for soil moisture dynamics, biochar kinetics",
    },
    EvolutionGap {
        id: "isotherm_batch_fitting",
        description: "Batch isotherm fitting via nonlinear optimization",
        tier: Tier::B,
        toadstool_primitive: Some("optimize::nelder_mead, multi_start_nelder_mead, NelderMeadGpu"),
        action: "WIRED (v0.4.1): gpu::isotherm::{fit_*_nm, fit_*_global, fit_batch_global}",
    },
    // ── Tier B (new v0.5.2): Pipeline & streaming orchestrators ──────
    EvolutionGap {
        id: "seasonal_pipeline",
        description: "Chained ET₀→Kc→WB→Yield pipeline (zero CPU round-trips target)",
        tier: Tier::B,
        toadstool_primitive: Some("StreamingPipeline + seasonal_pipeline.wgsl"),
        action: "GPU Stages 1-2 (v0.5.6): SeasonalPipeline::gpu() dispatches ET₀ (op=0) + \
                 Kc (op=7) to GPU; stages 3-4 CPU; fused seasonal_pipeline.wgsl available",
    },
    EvolutionGap {
        id: "atlas_stream",
        description: "Streaming multi-year regional ET₀ for 100+ stations",
        tier: Tier::B,
        toadstool_primitive: Some("UnidirectionalPipeline (fire-and-forget GPU streaming)"),
        action: "GPU+streaming (v0.5.4): AtlasStream::with_gpu() + process_streaming() \
                 callback pattern, true GPU streaming when UnidirectionalPipeline available",
    },
    EvolutionGap {
        id: "mc_et0_gpu",
        description: "Monte Carlo ET₀ uncertainty propagation on GPU",
        tier: Tier::B,
        toadstool_primitive: Some("mc_et0_propagate_f64.wgsl (xoshiro + Box-Muller)"),
        action: "WIRED (v0.5.2): gpu::mc_et0::mc_et0_gpu() — CPU fallback, \
                 GPU activates when WGSL_MC_ET0_PROPAGATE_F64 wired",
    },
    // ── Tier C: Needs new primitive ──────────────────────────────────
    EvolutionGap {
        id: "data_client",
        description: "HTTP/JSON client for Open-Meteo, NOAA CDO APIs",
        tier: Tier::C,
        toadstool_primitive: None,
        action: "Future: not GPU, but needed for automated data ingestion",
    },
];

/// `ToadStool` issues — all **RESOLVED** as of commit `0c477306`.
///
/// These were communicated to the `ToadStool` team and fixed in the
/// February 16, 2026 unified handoff. `ToadStool` has since evolved to
/// `1dd7e338` (S70+++), absorbing cross-spring content from all Springs.
pub const TOADSTOOL_ISSUES: &[ToadStoolIssue] = &[
    ToadStoolIssue {
        id: "TS-001",
        file: "crates/barracuda/src/shaders/science/batched_elementwise_f64.wgsl",
        line: 138,
        severity: "CRITICAL",
        summary: "pow_f64 returns 0.0 for non-integer exponents",
        detail: "The pow_f64 function had a placeholder `return zero;` for \
                 non-integer exponents. Atmospheric pressure P = 101.3 * \
                 ((293 - 0.0065*z) / 293)^5.26 silently computed P = 0.0, \
                 cascading gamma = 0.0 and incorrect ET₀.",
        fix: "RESOLVED: replaced with exp_f64(exp * log_f64(base)) when base > 0",
        blocks: "NONE (was: GPU ET₀ op=0, any shader path using fractional exponents)",
        status: IssueStatus::Resolved,
    },
    ToadStoolIssue {
        id: "TS-002",
        file: "crates/barracuda/src/ops/batched_elementwise_f64.rs",
        line: 0,
        severity: "MEDIUM",
        summary: "No Rust ops module for batched_elementwise_f64",
        detail: "The WGSL shader existed but there was no Rust orchestrator \
                 (ops::batched_elementwise_f64) to create compute pipelines, \
                 pack input buffers, dispatch workgroups, and read back results.",
        fix: "RESOLVED: BatchedElementwiseF64 orchestrator created with fao56_et0_batch() \
              and water_balance_batch() convenience methods",
        blocks: "NONE (was: All GPU dispatch for ET₀ and water balance from Rust)",
        status: IssueStatus::Resolved,
    },
    ToadStoolIssue {
        id: "TS-003",
        file: "crates/barracuda/src/shaders/science/batched_elementwise_f64.wgsl",
        line: 0,
        severity: "LOW",
        summary: "acos_simple and sin_simple use low-order approximations",
        detail: "acos_simple was a 3-term polynomial, sin_simple was 5-term Taylor. \
                 Both were adequate for FAO-56 ET₀ but not general scientific use.",
        fix: "RESOLVED: full math_f64.wgsl acos_f64/sin_f64 wired into batched shader \
              with (zero + literal) pattern for full f64 precision",
        blocks: "NONE (was: precision drift near boundary values)",
        status: IssueStatus::Resolved,
    },
    ToadStoolIssue {
        id: "TS-004",
        file: "crates/barracuda/src/ops/fused_map_reduce_f64.rs",
        line: 0,
        severity: "HIGH",
        summary: "FusedMapReduceF64 GPU dispatch panics on buffer usage conflict",
        detail: "When N >= 1024 (GPU dispatch threshold), the partials pipeline's \
                 second compute pass attempted STORAGE_READ_WRITE on a buffer already \
                 bound as STORAGE_READ in the same dispatch. wgpu panicked with \
                 'Attempted to use buffer with conflicting usages'.",
        fix: "RESOLVED: separate buffers for input (STORAGE_READ) and output \
              (STORAGE_READ_WRITE) in the partials pipeline",
        blocks: "NONE (was: GPU acceleration for arrays N >= 1024)",
        status: IssueStatus::Resolved,
    },
];

/// Status of a `ToadStool` issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueStatus {
    /// Issue is resolved in `ToadStool`.
    Resolved,
    /// Issue is still open.
    Open,
}

/// A discovered issue in `ToadStool` that was communicated upstream.
#[derive(Debug)]
pub struct ToadStoolIssue {
    /// Short issue identifier.
    pub id: &'static str,
    /// File path within the `ToadStool` repository.
    pub file: &'static str,
    /// Approximate line number (0 = file-level).
    pub line: u32,
    /// Severity: CRITICAL, HIGH, MEDIUM, LOW.
    pub severity: &'static str,
    /// One-line summary.
    pub summary: &'static str,
    /// Detailed description.
    pub detail: &'static str,
    /// Fix applied (or suggested).
    pub fix: &'static str,
    /// What this blocked in airSpring.
    pub blocks: &'static str,
    /// Current status.
    pub status: IssueStatus,
}
