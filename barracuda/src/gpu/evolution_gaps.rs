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
//! # Current Inventory (February 25, 2026 — v0.3.10, synced to `ToadStool` HEAD `02207c4a`)
//!
//! All four `ToadStool` issues (TS-001 through TS-004) are **RESOLVED**.
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
//! | Spearman rank correlation | `stats::correlation::spearman_correlation` | Available, not yet used |
//! | Bootstrap confidence intervals | `stats::bootstrap::bootstrap_ci` | Available, not yet used |
//! | Normal distribution | `stats::normal::norm_cdf`, `norm_ppf` | Available, not yet used |
//! | Chi-squared decomposition | `stats::chi2::chi2_decomposed` | Available (new in S52+) |
//! | Spectral density / RMT | `stats::spectral_density::empirical_spectral_density` | Available (new in S57+) |
//!
//! ## Tier B: Upstream Primitive Exists, Needs Domain Wiring
//!
//! | Need | Closest `ToadStool` Primitive | Gap |
//! |------|---------------------------|-----|
//! | 1D Richards equation | `pde::richards::solve_richards` (van Genuchten-Mualem) | **PROMOTED from Tier C** — wire with airSpring soil params |
//! | Sensor calibration (batch) | `batched_elementwise_f64.wgsl` (custom op) | Add `SoilWatch` 10 as op=5 |
//! | Hargreaves ET₀ (batch) | `batched_elementwise_f64.wgsl` | Add as op=6 (simpler than PM) |
//! | Kc climate adjustment (batch) | `batched_elementwise_f64.wgsl` | Add as op=7 |
//! | Moving window statistics | `ops::moving_window_stats` | **PROMOTED to Tier A** — `gpu::stream::StreamSmoother` |
//! | Nonlinear curve fitting | `optimize::nelder_mead`, `NelderMeadGpu` | Wire for correction eq fitting |
//! | Ridge regression | `linalg::ridge::ridge_regression` | **PROMOTED to Tier A** — `eco::correction::fit_ridge` |
//! | m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring from `wetSpring` |
//! | Tridiagonal solve (batch) | `linalg::tridiagonal_solve_f64` | Available for implicit PDE steps |
//! | Adaptive ODE (RK45) | `numerical::rk45_solve` (Dormand-Prince) | Available for dynamic soil models |
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
//! 1. CPU validation remains source of truth (123/123 checks)
//! 2. GPU results must match CPU within documented tolerance
//! 3. Cross-validation harness (65/65 Python↔Rust) extends to GPU path
//! 4. Each GPU function has a `test_gpu_matches_cpu_*` integration test
//! 5. GPU determinism proven: 4 bit-identical rerun tests (`gpu_integration.rs`)

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

/// All known evolution gaps (18 entries — 8 Tier A integrated, 9 Tier B, 1 Tier C).
///
/// Richards PDE promoted C→B: upstream `pde::richards::solve_richards` now available.
/// Tridiagonal and RK45 added as new Tier B capabilities.
/// Dual Kc batch added as Tier B: `gpu::dual_kc` CPU ready, pending shader op.
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
    // ── Tier B: Shader exists, needs domain adaptation ────────────────
    EvolutionGap {
        id: "dual_kc_batch",
        description: "Batched dual Kc (Ke + ETc) across M fields per timestep",
        tier: Tier::B,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (custom op)"),
        action: "GPU orchestrator wired (gpu::dual_kc), CPU validated — add Ke as op=8 in shader",
    },
    EvolutionGap {
        id: "sensor_calibration_batch",
        description: "Batch sensor calibration (SoilWatch 10) via custom op",
        tier: Tier::B,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (custom op)"),
        action: "Add SoilWatch 10 calibration as op=5 in batched shader",
    },
    EvolutionGap {
        id: "hargreaves_batch",
        description: "Hargreaves ET₀ as batch GPU op (simpler than PM, fewer inputs)",
        tier: Tier::B,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl"),
        action: "Add as op=6 — needs only tmax, tmin, Ra (no humidity/wind)",
    },
    EvolutionGap {
        id: "kc_climate_adjust",
        description: "Kc climate adjustment (FAO-56 Eq. 62) as batch GPU op",
        tier: Tier::B,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl"),
        action: "Add as op=7 — function of wind speed and RH_min",
    },
    EvolutionGap {
        id: "nonlinear_solver",
        description: "Nonlinear least squares for soil calibration curve fitting",
        tier: Tier::B,
        toadstool_primitive: Some("optimize::nelder_mead, optimize::NelderMeadGpu"),
        action: "Local analytical fits exist; can upgrade to GPU Nelder-Mead for large batches",
    },
    EvolutionGap {
        id: "richards_pde",
        description: "1D Richards equation for unsaturated soil water flow",
        tier: Tier::B,
        toadstool_primitive: Some(
            "pde::richards::solve_richards (van Genuchten-Mualem, Picard + CN + Thomas)",
        ),
        action: "PROMOTED C→B: upstream solver available — wire with airSpring soil params",
    },
    EvolutionGap {
        id: "tridiagonal_batch",
        description: "Tridiagonal solver for implicit PDE time-stepping",
        tier: Tier::B,
        toadstool_primitive: Some("linalg::tridiagonal_solve_f64, ops::cyclic_reduction_f64"),
        action: "Available upstream — wire when Richards PDE integration begins",
    },
    EvolutionGap {
        id: "rk45_adaptive",
        description: "Adaptive RK45 ODE solver for dynamic soil/water models",
        tier: Tier::B,
        toadstool_primitive: Some("numerical::rk45_solve (Dormand-Prince, adaptive step)"),
        action: "Available upstream — wire for soil moisture dynamics, biochar kinetics",
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
/// `02207c4a` (S62+), absorbing cross-spring content from all Springs.
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
