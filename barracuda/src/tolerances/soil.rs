// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soil physics, water balance, PDE, and crop science tolerances.

use super::Tolerance;

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

/// Biochar isotherm mean residual: no systematic bias threshold.
pub const ISOTHERM_MEAN_RESIDUAL: Tolerance = Tolerance {
    name: "isotherm_mean_residual",
    abs_tol: 0.5,
    rel_tol: 0.05,
    justification: "Kumari et al. (2025): mean(|qe_obs - qe_pred|) < 0.5 mg/g for no systematic bias",
};

/// Per-step mass balance: machine-precision conservation check per time step.
pub const WATER_BALANCE_PER_STEP: Tolerance = Tolerance {
    name: "water_balance_per_step",
    abs_tol: 1e-6,
    rel_tol: 1e-10,
    justification: "Per-step conservation check: f64 arithmetic residual < 1e-6 mm",
};

/// Topp equation VWC round-trip: ±0.005 m³/m³ (half-percent volumetric).
pub const TOPP_EQUATION: Tolerance = Tolerance {
    name: "topp_equation",
    abs_tol: 0.005,
    rel_tol: 1e-3,
    justification: "Topp et al. (1980): 0.005 m³/m³ covers polynomial regression residual",
};

/// Analytical computation tolerance: simple arithmetic/polynomial evaluation.
pub const ANALYTICAL_COMPUTATION: Tolerance = Tolerance {
    name: "analytical_computation",
    abs_tol: 0.1,
    rel_tol: 0.01,
    justification: "Generic analytical: covers digitization precision from published tables",
};

/// SCS Curve Number analytical precision: USDA-SCS equation arithmetic.
pub const SCS_CN_ANALYTICAL: Tolerance = Tolerance {
    name: "scs_cn_analytical",
    abs_tol: 0.01,
    rel_tol: 1e-4,
    justification: "SCS-CN Q and S: integer CN → f64 arithmetic yields ±0.01 mm precision",
};

/// Green-Ampt infiltration: Newton iteration convergence for implicit F(t).
pub const GREEN_AMPT_ANALYTICAL: Tolerance = Tolerance {
    name: "green_ampt_analytical",
    abs_tol: 0.001,
    rel_tol: 1e-4,
    justification: "Green-Ampt Newton iteration converges to 1e-12; 0.001 cm covers soil param uncertainty",
};

/// FAO-56 Kc precision: Eq. 72 `Kc_max` and dual Kc component precision.
pub const DUAL_KC_PRECISION: Tolerance = Tolerance {
    name: "dual_kc_precision",
    abs_tol: 0.01,
    rel_tol: 0.01,
    justification: "FAO-56 Eq 72 Kc_max: 2-decimal tabulated values; wind/RH adjustment precision",
};

/// Saxton-Rawls pedotransfer moisture content (θ): regression coefficients
/// have 4 significant digits; exponential intermediate steps preserve to 1e-4.
pub const PEDOTRANSFER_MOISTURE: Tolerance = Tolerance {
    name: "pedotransfer_moisture",
    abs_tol: 1e-4,
    rel_tol: 1e-4,
    justification: "Saxton & Rawls (2006) regression: 4-digit θ precision",
};

/// Saxton-Rawls Ksat: exponential amplification of regression residuals
/// widens tolerance to ±0.5 mm/hr.
pub const PEDOTRANSFER_KSAT: Tolerance = Tolerance {
    name: "pedotransfer_ksat",
    abs_tol: 0.5,
    rel_tol: 0.05,
    justification: "Saxton & Rawls (2006) Ksat: exponential amplification of regression error",
};

/// GDD computation: integer arithmetic on clamped temperatures yields exact results
/// to f64 precision.
pub const GDD_EXACT: Tolerance = Tolerance {
    name: "gdd_exact",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "GDD avg/clamp: f64-exact integer arithmetic (max/min/midpoint)",
};

/// Irrigation recommendation depth (cm).
pub const IRRIGATION_DEPTH: Tolerance = Tolerance {
    name: "irrigation_depth",
    abs_tol: 0.01,
    rel_tol: 0.01,
    justification: "Depth precision: (FC − VWC) × root_zone_m × 100; ±0.01 cm",
};
