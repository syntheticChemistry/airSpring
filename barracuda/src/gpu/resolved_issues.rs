// SPDX-License-Identifier: AGPL-3.0-or-later
//! Historical `BarraCuda` issues — all **RESOLVED** as of S54+S66.
//!
//! These were communicated upstream and fixed in the February 16, 2026
//! unified handoff. `BarraCuda` (extracted from `ToadStool` in S89) has
//! since evolved to standalone 0.3.1 with universal precision architecture.
//!
//! Retained as a historical record and fossil reference. No open issues remain.

/// Status of a `BarraCuda` issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueStatus {
    /// Issue is resolved in `BarraCuda`.
    Resolved,
    /// Issue is still open.
    Open,
}

/// A discovered issue in `BarraCuda` (pre-extraction, `ToadStool` S54) that was
/// communicated upstream.
#[derive(Debug)]
pub struct BarraCudaIssue {
    /// Short issue identifier.
    pub id: &'static str,
    /// File path within the `BarraCuda` repository.
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

/// All `BarraCuda` issues — all **RESOLVED** as of commit `0c477306`.
pub const BARRACUDA_ISSUES: &[BarraCudaIssue] = &[
    BarraCudaIssue {
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
    BarraCudaIssue {
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
    BarraCudaIssue {
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
    BarraCudaIssue {
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
