// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-specific validation tolerances for agricultural science.
//!
//! Extends the [`barracuda::tolerances`] pattern (centralized, justified) with
//! airSpring-specific constants grounded in FAO-56, HYDRUS, and published
//! Python baselines. Every tolerance here has a physical or numerical
//! justification—no magic numbers.
//!
//! # Cross-Spring Provenance
//!
//! The `Tolerance` struct itself comes from `ToadStool` S52 (M-010), inspired by
//! neuralSpring's validation registry and adopted across all Springs.
//!
//! | Domain | Tolerance | Origin |
//! |--------|-----------|--------|
//! | ET₀ Penman-Monteith | 0.01 mm/day | FAO-56 Table 2 (worked examples) |
//! | Saturation vapor pressure | 0.01 kPa | FAO-56 Table 2.3 (3 decimals) |
//! | Water balance mass | 0.01 mm | FAO-56 Ch 8 conservation law |
//! | Richards PDE | benchmark-specified | HYDRUS numerical convergence |
//! | Isotherm fit | 0.01 mg/g | Adsorption experiment precision |
//! | Kriging interpolation | 1e-6 | Numerical linear algebra (small systems) |
//! | GPU/CPU cross-validation | 1e-5 | f64 shader rounding (hotSpring TS-001 fix) |

pub use barracuda::tolerances::{check, Tolerance};

// ═══════════════════════════════════════════════════════════════════
// ET₀ and atmospheric tolerances (FAO-56)
// ═══════════════════════════════════════════════════════════════════

/// Saturation vapour pressure es(T): FAO-56 Table 2.3 rounds to 3 decimals.
pub const ET0_SAT_VAPOUR_PRESSURE: Tolerance = Tolerance {
    name: "et0_sat_vapour_pressure",
    abs_tol: 0.01,
    rel_tol: 1e-4,
    justification: "FAO-56 Table 2.3: 3 decimal kPa; Tetens equation precision",
};

/// Slope of saturation vapour pressure Δ: FAO-56 Table 2.4, 3 decimal precision.
pub const ET0_SLOPE_VAPOUR: Tolerance = Tolerance {
    name: "et0_slope_vapour_pressure",
    abs_tol: 0.005,
    rel_tol: 1e-4,
    justification: "FAO-56 Table 2.4: 3 decimal kPa/°C; derivative of Tetens",
};

/// Net radiation Rn: FAO-56 Example 18, precision ±0.5 MJ/m²/day due to
/// combined albedo, clear-sky, and longwave radiation chain.
pub const ET0_NET_RADIATION: Tolerance = Tolerance {
    name: "et0_net_radiation",
    abs_tol: 0.5,
    rel_tol: 0.05,
    justification: "FAO-56 Ex 18: Rn chain (albedo→Rns→Rnl→Rn), ±0.5 MJ/m²/day",
};

/// Reference ET₀ (mm/day): FAO-56 worked examples, ±0.01 mm/day for
/// well-instrumented stations.
pub const ET0_REFERENCE: Tolerance = Tolerance {
    name: "et0_reference",
    abs_tol: 0.01,
    rel_tol: 1e-3,
    justification: "FAO-56 Examples 17-19: validated against 3-decimal tables",
};

/// Vapour pressure deficit: ±0.02 kPa, combined ea + es uncertainty.
pub const ET0_VPD: Tolerance = Tolerance {
    name: "et0_vpd",
    abs_tol: 0.02,
    rel_tol: 1e-3,
    justification: "Combined saturation + actual vapour pressure uncertainty",
};

/// Cold-climate ET₀: wider tolerance for extreme conditions where small
/// absolute values amplify relative error.
pub const ET0_COLD_CLIMATE: Tolerance = Tolerance {
    name: "et0_cold_climate",
    abs_tol: 0.5,
    rel_tol: 0.1,
    justification: "Near-zero ET₀ in cold climates; small denominator amplifies error",
};

/// Psychrometric constant γ: ±0.001 kPa/°C, elevation-dependent.
pub const PSYCHROMETRIC_CONSTANT: Tolerance = Tolerance {
    name: "psychrometric_constant",
    abs_tol: 0.001,
    rel_tol: 1e-4,
    justification: "FAO-56 Eq 8: γ = 0.665e-3 × P; elevation precision",
};

// ═══════════════════════════════════════════════════════════════════
// Water balance and soil moisture
// ═══════════════════════════════════════════════════════════════════

/// Mass balance conservation: FAO-56 Ch 8, `P + I - ETc - DP = ΔS`.
pub const WATER_BALANCE_MASS: Tolerance = Tolerance {
    name: "water_balance_mass",
    abs_tol: 0.01,
    rel_tol: 1e-6,
    justification: "FAO-56 Ch 8: conservation law — ΔDr ≤ 0.01 mm per step",
};

/// Stress coefficient Ks: `[0,1]` range, ±0.01 for linear depletion model.
pub const STRESS_COEFFICIENT: Tolerance = Tolerance {
    name: "stress_coefficient",
    abs_tol: 0.01,
    rel_tol: 0.01,
    justification: "FAO-56 Eq 84: Ks = (TAW-Dr)/(TAW-RAW), midpoint precision",
};

/// Soil hydraulic properties: ±0.01 m³/m³ for USDA texture class averages.
pub const SOIL_HYDRAULIC: Tolerance = Tolerance {
    name: "soil_hydraulic",
    abs_tol: 0.01,
    rel_tol: 0.02,
    justification: "USDA texture class θ_FC, θ_WP averages; pedotransfer uncertainty",
};

/// Newton-Raphson convergence for soil moisture roundtrip.
pub const SOIL_ROUNDTRIP: Tolerance = Tolerance {
    name: "soil_roundtrip",
    abs_tol: 0.001,
    rel_tol: 1e-4,
    justification: "Newton-Raphson VWC→dielectric→VWC roundtrip convergence",
};

// ═══════════════════════════════════════════════════════════════════
// Richards equation (PDE)
// ═══════════════════════════════════════════════════════════════════

/// Richards PDE steady-state: HYDRUS benchmark tolerance.
pub const RICHARDS_STEADY: Tolerance = Tolerance {
    name: "richards_steady_state",
    abs_tol: 0.001,
    rel_tol: 0.01,
    justification: "HYDRUS benchmark: steady-state θ(z) profile precision",
};

/// Richards PDE transient: wider tolerance for time-stepping error.
pub const RICHARDS_TRANSIENT: Tolerance = Tolerance {
    name: "richards_transient",
    abs_tol: 0.005,
    rel_tol: 0.02,
    justification: "Picard iteration + implicit Euler; cumulative time-step error",
};

// ═══════════════════════════════════════════════════════════════════
// Isotherm fitting (biochar adsorption)
// ═══════════════════════════════════════════════════════════════════

/// Langmuir/Freundlich parameter fit: Nelder-Mead convergence to benchmark.
pub const ISOTHERM_PARAMETER: Tolerance = Tolerance {
    name: "isotherm_parameter",
    abs_tol: 0.01,
    rel_tol: 0.01,
    justification: "NM simplex convergence tol=1e-8, 5000 iterations max",
};

/// Isotherm model prediction: qe (mg/g) at given Ce.
pub const ISOTHERM_PREDICTION: Tolerance = Tolerance {
    name: "isotherm_prediction",
    abs_tol: 0.1,
    rel_tol: 0.01,
    justification: "Batch adsorption experiment precision: ±0.1 mg/g",
};

// ═══════════════════════════════════════════════════════════════════
// GPU/CPU cross-validation
// ═══════════════════════════════════════════════════════════════════

/// GPU↔CPU cross-validation: f64 shader rounding via DF64 emulation.
///
/// hotSpring TS-001 (`pow_f64` fix, S54) and TS-003 (`acos_f64` fix, S54)
/// established that WGSL f64 shaders achieve ≤1e-5 relative agreement with
/// CPU f64. This tolerance is used for all GPU/CPU comparison tests.
pub const GPU_CPU_CROSS: Tolerance = Tolerance {
    name: "gpu_cpu_cross_validation",
    abs_tol: 1e-5,
    rel_tol: 1e-5,
    justification: "WGSL f64 shader vs CPU f64; hotSpring TS-001/003 S54 validated",
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

// ═══════════════════════════════════════════════════════════════════
// Sensor calibration
// ═══════════════════════════════════════════════════════════════════

/// Polynomial/analytical sensor calibration: exact-match for simple arithmetic.
pub const SENSOR_EXACT: Tolerance = Tolerance {
    name: "sensor_exact",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "Polynomial evaluation and linear regression: f64-exact",
};

/// Irrigation recommendation depth (cm).
pub const IRRIGATION_DEPTH: Tolerance = Tolerance {
    name: "irrigation_depth",
    abs_tol: 0.01,
    rel_tol: 0.01,
    justification: "Depth precision: (FC − VWC) × root_zone_m × 100; ±0.01 cm",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_et0_tolerance_check() {
        assert!(check(2.338, 2.338, &ET0_REFERENCE));
        assert!(check(2.348, 2.338, &ET0_REFERENCE));
        assert!(!check(2.448, 2.338, &ET0_REFERENCE));
    }

    #[test]
    fn test_water_balance_mass_check() {
        assert!(check(0.0, 0.0, &WATER_BALANCE_MASS));
        assert!(check(0.005, 0.0, &WATER_BALANCE_MASS));
        assert!(!check(0.02, 0.0, &WATER_BALANCE_MASS));
    }

    #[test]
    fn test_gpu_cpu_cross_validation() {
        let cpu = 3.88;
        let gpu = 3.880_03;
        assert!(check(gpu, cpu, &GPU_CPU_CROSS));
    }

    #[test]
    fn test_sensor_exact_precision() {
        assert!(check(42.000_000_001, 42.0, &SENSOR_EXACT));
        assert!(!check(42.001, 42.0, &SENSOR_EXACT));
    }

    #[test]
    fn test_richards_steady_tolerance() {
        assert!(check(0.350, 0.351, &RICHARDS_STEADY));
        assert!(!check(0.340, 0.351, &RICHARDS_STEADY));
    }

    #[test]
    fn test_isotherm_parameter_tolerance() {
        assert!(check(10.005, 10.0, &ISOTHERM_PARAMETER));
        assert!(!check(10.5, 10.0, &ISOTHERM_PARAMETER));
    }

    #[test]
    fn test_kriging_tolerance() {
        assert!(check(25.000_000_5, 25.0, &KRIGING_INTERPOLATION));
        assert!(!check(25.001, 25.0, &KRIGING_INTERPOLATION));
    }

    #[test]
    fn test_seasonal_reduction_tolerance() {
        assert!(check(1_000.000_000_005, 1_000.0, &SEASONAL_REDUCTION));
    }

    #[test]
    fn test_cold_climate_wide_tolerance() {
        assert!(check(0.3, 0.1, &ET0_COLD_CLIMATE));
        assert!(!check(1.0, 0.1, &ET0_COLD_CLIMATE));
    }

    #[test]
    fn test_all_tolerances_have_justification() {
        let all = [
            &ET0_SAT_VAPOUR_PRESSURE,
            &ET0_SLOPE_VAPOUR,
            &ET0_NET_RADIATION,
            &ET0_REFERENCE,
            &ET0_VPD,
            &ET0_COLD_CLIMATE,
            &PSYCHROMETRIC_CONSTANT,
            &WATER_BALANCE_MASS,
            &STRESS_COEFFICIENT,
            &SOIL_HYDRAULIC,
            &SOIL_ROUNDTRIP,
            &RICHARDS_STEADY,
            &RICHARDS_TRANSIENT,
            &ISOTHERM_PARAMETER,
            &ISOTHERM_PREDICTION,
            &GPU_CPU_CROSS,
            &KRIGING_INTERPOLATION,
            &SEASONAL_REDUCTION,
            &IOT_STREAM_SMOOTHING,
            &SENSOR_EXACT,
            &IRRIGATION_DEPTH,
        ];
        for tol in &all {
            assert!(!tol.name.is_empty(), "tolerance must have a name");
            assert!(
                !tol.justification.is_empty(),
                "tolerance {} must have justification",
                tol.name
            );
            assert!(tol.abs_tol > 0.0, "{}: abs_tol must be positive", tol.name);
            assert!(tol.rel_tol > 0.0, "{}: rel_tol must be positive", tol.name);
        }
    }
}
