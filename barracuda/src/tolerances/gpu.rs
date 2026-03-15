// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU/CPU cross-validation, cross-spring, and NUCLEUS IPC tolerances.

use super::Tolerance;

/// GPU↔CPU cross-validation: f64 shader rounding via DF64 emulation.
///
/// `BarraCuda` TS-001 (`pow_f64` fix, S54) and TS-003 (`acos_f64` fix, S54)
/// established that WGSL f64 shaders achieve ≤1e-5 relative agreement with
/// CPU f64. This tolerance is used for all GPU/CPU comparison tests.
pub const GPU_CPU_CROSS: Tolerance = Tolerance {
    name: "gpu_cpu_cross_validation",
    abs_tol: 1e-5,
    rel_tol: 1e-5,
    justification: "WGSL f64 shader vs CPU f64; BarraCuda TS-001/003 S54 validated",
};

/// Kriging interpolation: small-system linear algebra tolerance.
pub const KRIGING_INTERPOLATION: Tolerance = Tolerance {
    name: "kriging_interpolation",
    abs_tol: 1e-6,
    rel_tol: 1e-6,
    justification: "Kriging weights via matrix solve; small N exact to f64 precision",
};

/// Seasonal reduction (sum, mean, min, max): GPU accumulation tolerance.
pub const SEASONAL_REDUCTION: Tolerance = Tolerance {
    name: "seasonal_reduction",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "FusedMapReduceF64 GPU sum; TS-004 S54 buffer fix (N≥1024)",
};

/// `IoT` stream smoothing: `MovingWindowStats` f32 precision.
pub const IOT_STREAM_SMOOTHING: Tolerance = Tolerance {
    name: "iot_stream_smoothing",
    abs_tol: 0.01,
    rel_tol: 1e-4,
    justification: "MovingWindowStats uses f32 shaders; f32→f64 promotion rounding",
};

/// Analytical functions from upstream barraCuda (erf, gamma, bessel, etc.):
/// f64-exact to machine precision against known mathematical identities.
///
/// Provenance: analytical — erf(1), Γ(5)=24, J₀(0)=1 are mathematical
/// identities with no Python baseline dependency.
pub const CROSS_SPRING_ANALYTICAL: Tolerance = Tolerance {
    name: "cross_spring_analytical",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "Mathematical identities: erf(1), Γ(5)=24, J₀(0)=1 — f64-exact",
};

/// Cross-spring GPU↔CPU parity for absorbed shader ops: DF64 emulation
/// introduces rounding at ~1e-6 for compound operations.
pub const CROSS_SPRING_GPU_CPU: Tolerance = Tolerance {
    name: "cross_spring_gpu_cpu",
    abs_tol: 1e-4,
    rel_tol: 1e-4,
    justification: "DF64 compound ops (exp, pow, log): ~1e-6 per op, chained to 1e-4",
};

/// Cross-spring evolution validation: moderately tight for rewired ops
/// that chain multiple barraCuda primitives.
pub const CROSS_SPRING_EVOLUTION: Tolerance = Tolerance {
    name: "cross_spring_evolution",
    abs_tol: 1e-3,
    rel_tol: 1e-3,
    justification: "Chained rewire (CPU→GPU): accumulates DF64 rounding across 3-5 ops",
};

/// NUCLEUS round-trip: JSON-RPC serialization introduces no numerical error
/// for f64 values that survive IEEE-754 → JSON → IEEE-754 round-trip.
pub const NUCLEUS_ROUNDTRIP: Tolerance = Tolerance {
    name: "nucleus_roundtrip",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "JSON f64 round-trip: IEEE-754 double → serde_json → double is exact to 1e-15",
};

/// NUCLEUS pipeline end-to-end: weather → ET₀ → water balance → yield
/// through JSON-RPC; tolerance accumulates per stage.
pub const NUCLEUS_PIPELINE: Tolerance = Tolerance {
    name: "nucleus_pipeline",
    abs_tol: 1e-6,
    rel_tol: 1e-6,
    justification: "Multi-stage JSON-RPC pipeline: f64 arithmetic per stage, 4 stages max",
};
