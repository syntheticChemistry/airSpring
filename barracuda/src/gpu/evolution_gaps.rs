//! Evolution gaps: what airSpring needs from `ToadStool` and what's ready.
//!
//! This module documents the gaps between airSpring's validated CPU pipeline
//! and the GPU-accelerated future. It serves as a living roadmap.
//!
//! # Gap Categories
//!
//! - **Ready**: `ToadStool` primitive exists, airSpring just needs to wire it.
//! - **Needs Orchestrator**: WGSL shader exists but no Rust `ops::` module.
//! - **Needs Primitive**: No `ToadStool` implementation yet.
//! - **Deprecated**: Replaced by better approach.
//!
//! # Current Inventory (February 16, 2026)
//!
//! ## Tier A: Ready to Wire (GPU primitive exists in `ToadStool`)
//!
//! | airSpring Module | `ToadStool` Primitive | Gap |
//! |-----------------|--------------------|----|
//! | `eco::evapotranspiration::daily_et0` | `batched_elementwise_f64.wgsl` (op=0) | Needs Rust orchestrator (`BatchedEt0Gpu`) |
//! | `eco::water_balance` | `batched_elementwise_f64.wgsl` (op=1) | Needs Rust orchestrator |
//! | Soil moisture spatial mapping | `ops::kriging_f64::KrigingF64` | Ready — wire with domain API |
//! | Batch statistical reductions | `ops::fused_map_reduce_f64::FusedMapReduceF64` | Ready — wire for ET₀ totals |
//! | R² / Pearson correlation | `stats::pearson_correlation` | **Already wired** (testutil) |
//! | Variance / std deviation | `stats::correlation::variance`, `std_dev` | **Already wired** (integration tests) |
//! | Spearman rank correlation | `stats::correlation::spearman_correlation` | Available, not yet used |
//! | Bootstrap confidence intervals | `stats::bootstrap::bootstrap_ci` | Available, not yet used |
//! | Normal distribution | `stats::normal::norm_cdf`, `norm_ppf` | Available, not yet used |
//!
//! ## Tier B: Shader Exists, Needs Adaptation
//!
//! | Need | Closest `ToadStool` Primitive | Gap |
//! |------|---------------------------|-----|
//! | Sensor calibration (batch) | `batched_elementwise_f64.wgsl` (custom op) | Add `SoilWatch` 10 as op=5 |
//! | Hargreaves ET₀ (batch) | `batched_elementwise_f64.wgsl` | Add as op=6 (simpler than PM) |
//! | Kc climate adjustment (batch) | `batched_elementwise_f64.wgsl` | Add as op=7 |
//! | m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring from wetSpring |
//!
//! ## Tier C: Needs New `ToadStool` Primitives
//!
//! | Need | Description | Complexity |
//! |------|-------------|-----------|
//! | Nonlinear least squares | `fit_correction_equations()` from Python | Medium — needs Levenberg-Marquardt |
//! | Moving window statistics | `IoT` stream smoothing | Low — sliding window reduce |
//! | 1D Richards equation | Unsaturated flow PDE | High — uses `CgGpu` for tridiagonal |
//! | HTTP/JSON data client | Open-Meteo, NOAA CDO APIs | Low — not GPU, but needed |
//!
//! ## Deprecated Patterns (Clean Up)
//!
//! | Pattern | Status | Replacement |
//! |---------|--------|-------------|
//! | `rayon` dependency | **Removed** (v0.2.0) | Will use `ToadStool` dispatch when GPU-ready |
//! | Ad-hoc `String` errors | **Replaced** (v0.2.0) | `AirSpringError` enum |
//! | `HashMap` CSV storage | **Replaced** (v0.2.0) | Columnar `Vec<Vec<f64>>` |
//! | Hardcoded runoff model | **Replaced** (v0.2.0) | `RunoffModel` enum |
//!
//! ## Shader Precision Notes
//!
//! The `batched_elementwise_f64.wgsl` shader has known precision limitations:
//!
//! 1. **`acos_simple`**: Uses a 3-term approximation — fine for latitude range
//!    (±70°) but loses accuracy near ±1. Evolution: wire `math_f64.wgsl` full
//!    `acos_f64` when available.
//!
//! 2. **`pow_f64`**: Returns 0.0 for non-integer exponents. The atmospheric
//!    pressure calculation (5.26 exponent) will fail on GPU. Evolution: need
//!    full `exp_f64(exp * log_f64(base))` path in shader.
//!
//! 3. **`sin_simple`**: 5-term Taylor series — adequate for ET₀ but not for
//!    general use. Evolution: wire `math_f64.wgsl` full `sin_f64`.
//!
//! ## Cross-Validation Strategy
//!
//! Before any GPU path becomes production:
//! 1. CPU validation remains source of truth (119/119 checks, 162 tests)
//! 2. GPU results must match CPU within documented tolerance
//! 3. Cross-validation harness (65/65 Python↔Rust) extends to GPU path
//! 4. Each GPU function gets a `test_gpu_matches_cpu_*` integration test

/// Structured representation of an evolution gap.
#[derive(Debug)]
pub struct EvolutionGap {
    /// Short identifier.
    pub id: &'static str,
    /// Human description of the gap.
    pub description: &'static str,
    /// Current tier (A=ready, B=adapt, C=new).
    pub tier: Tier,
    /// What `ToadStool` provides (if anything).
    pub toadstool_primitive: Option<&'static str>,
    /// What airSpring needs to do.
    pub action: &'static str,
}

/// Evolution tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// GPU primitive exists, just needs wiring.
    A,
    /// Shader exists, needs adaptation for airSpring domain.
    B,
    /// Needs new `ToadStool` primitive.
    C,
}

/// All known evolution gaps.
pub const GAPS: &[EvolutionGap] = &[
    // ── Tier A: Ready to wire (primitive exists) ─────────────────────
    EvolutionGap {
        id: "batched_et0_gpu",
        description: "Batched FAO-56 ET₀ on GPU for N station-days",
        tier: Tier::A,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (op=0)"),
        action: "WIRED (cpu::gpu::et0::BatchedEt0) — GPU blocked on pow_f64 fix",
    },
    EvolutionGap {
        id: "batched_water_balance_gpu",
        description: "Batched water balance depletion update on GPU",
        tier: Tier::A,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (op=1)"),
        action: "WIRED (gpu::water_balance::BatchedWaterBalance) — GPU needs ops module",
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
        action: "INTEGRATED — SeasonalReducer wraps FusedMapReduceF64 (GPU for N≥1024, see TS-004)",
    },
    EvolutionGap {
        id: "bootstrap_uncertainty",
        description: "Bootstrap confidence intervals for ET₀ and water balance",
        tier: Tier::A,
        toadstool_primitive: Some("stats::bootstrap::bootstrap_ci"),
        action: "WIRED (testutil::bootstrap_rmse) — already using barracuda::stats",
    },
    // ── Tier B: Shader exists, needs ToadStool fix ───────────────────
    EvolutionGap {
        id: "pow_f64_shader",
        description: "pow_f64 returns 0.0 for non-integer exponents (e.g. 5.26)",
        tier: Tier::B,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl:138"),
        action: "TOADSTOOL ISSUE: replace placeholder with exp_f64(exp * log_f64(base))",
    },
    EvolutionGap {
        id: "acos_precision",
        description: "acos_simple 3-term approximation loses accuracy near ±1",
        tier: Tier::B,
        toadstool_primitive: Some("math_f64.wgsl full acos_f64 exists, not wired"),
        action: "TOADSTOOL ISSUE: wire math_f64.wgsl acos_f64 into batched shader",
    },
    EvolutionGap {
        id: "batched_ops_module",
        description: "No Rust ops::batched_elementwise_f64 orchestrator in ToadStool",
        tier: Tier::B,
        toadstool_primitive: Some("batched_elementwise_f64.wgsl (shader only)"),
        action: "TOADSTOOL ISSUE: create Rust orchestrator (pipeline, buffers, dispatch)",
    },
    // ── Tier C: Needs new primitive ──────────────────────────────────
    EvolutionGap {
        id: "nonlinear_solver",
        description: "Nonlinear least squares for soil calibration curve fitting",
        tier: Tier::C,
        toadstool_primitive: None,
        action: "SOLVED locally: pure Rust fit_correction_equations() in eco::correction",
    },
    EvolutionGap {
        id: "richards_pde",
        description: "1D Richards equation for unsaturated soil water flow",
        tier: Tier::C,
        toadstool_primitive: None,
        action: "Future: FD solver using barracuda::ops::linalg for tridiagonal",
    },
    EvolutionGap {
        id: "moving_window",
        description: "Sliding window statistics for IoT sensor stream processing",
        tier: Tier::C,
        toadstool_primitive: None,
        action: "Future: GPU moving average/variance reduction kernel",
    },
];

/// Issues discovered in `ToadStool` that block GPU acceleration.
///
/// These should be communicated to the `ToadStool` team for the next handoff round.
pub const TOADSTOOL_ISSUES: &[ToadStoolIssue] = &[
    ToadStoolIssue {
        id: "TS-001",
        file: "crates/barracuda/src/shaders/science/batched_elementwise_f64.wgsl",
        line: 138,
        severity: "CRITICAL",
        summary: "pow_f64 returns 0.0 for non-integer exponents",
        detail: "The pow_f64 function has a placeholder `return zero;` for \
                 non-integer exponents. Atmospheric pressure P = 101.3 * \
                 ((293 - 0.0065*z) / 293)^5.26 silently computes P = 0.0, \
                 cascading gamma = 0.0 and incorrect ET₀.",
        fix: "Replace line 138 with: if (base > zero) { return exp_f64(exp * log_f64(base)); }",
        blocks: "GPU ET₀ (op=0), any shader path using fractional exponents",
    },
    ToadStoolIssue {
        id: "TS-002",
        file: "crates/barracuda/src/shaders/science/batched_elementwise_f64.wgsl",
        line: 0,
        severity: "MEDIUM",
        summary: "No Rust ops module for batched_elementwise_f64",
        detail: "The WGSL shader exists but there is no Rust orchestrator \
                 (ops::batched_elementwise_f64) to create compute pipelines, \
                 pack input buffers, dispatch workgroups, and read back results.",
        fix: "Create ops/batched_elementwise_f64.rs following the FusedMapReduceF64 pattern",
        blocks: "All GPU dispatch for ET₀ and water balance from Rust",
    },
    ToadStoolIssue {
        id: "TS-003",
        file: "crates/barracuda/src/shaders/science/batched_elementwise_f64.wgsl",
        line: 0,
        severity: "LOW",
        summary: "acos_simple and sin_simple use low-order approximations",
        detail: "acos_simple is a 3-term polynomial, sin_simple is 5-term Taylor. \
                 Both are adequate for FAO-56 ET₀ (latitude ±70°, DOY angles) but \
                 would fail for general scientific use near boundary values.",
        fix: "Wire the full math_f64.wgsl acos_f64/sin_f64 into the batched shader",
        blocks: "Nothing currently (ET₀ precision is within tolerance)",
    },
    ToadStoolIssue {
        id: "TS-004",
        file: "crates/barracuda/src/ops/fused_map_reduce_f64.rs",
        line: 0,
        severity: "HIGH",
        summary: "FusedMapReduceF64 GPU dispatch panics on buffer usage conflict",
        detail: "When N ≥ 1024 (GPU dispatch threshold), the partials pipeline's \
                 second compute pass attempts STORAGE_READ_WRITE on a buffer already \
                 bound as STORAGE_READ in the same dispatch. wgpu panics with \
                 'Attempted to use buffer with conflicting usages'.",
        fix: "Use separate buffers for input (STORAGE_READ) and output \
              (STORAGE_READ_WRITE) in the partials pipeline, or add a barrier \
              between passes",
        blocks: "GPU acceleration for arrays N ≥ 1024 (SeasonalReducer falls back to CPU)",
    },
];

/// A discovered issue in `ToadStool` that needs to be communicated upstream.
#[derive(Debug)]
pub struct ToadStoolIssue {
    /// Short issue identifier.
    pub id: &'static str,
    /// File path within the `ToadStool` repository.
    pub file: &'static str,
    /// Approximate line number (0 = file-level).
    pub line: u32,
    /// Severity: CRITICAL, MEDIUM, LOW.
    pub severity: &'static str,
    /// One-line summary.
    pub summary: &'static str,
    /// Detailed description.
    pub detail: &'static str,
    /// Suggested fix.
    pub fix: &'static str,
    /// What this blocks in airSpring.
    pub blocks: &'static str,
}
