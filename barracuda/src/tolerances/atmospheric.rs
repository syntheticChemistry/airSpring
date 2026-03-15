// SPDX-License-Identifier: AGPL-3.0-or-later
//! ET₀ and atmospheric tolerances (FAO-56, simplified methods, cross-method).

use super::Tolerance;

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

/// Thornthwaite heat-index term and exponent: polynomial regression coefficients
/// have 4–5 significant digits; intermediate values converge to 1e-4.
pub const THORNTHWAITE_ANALYTICAL: Tolerance = Tolerance {
    name: "thornthwaite_analytical",
    abs_tol: 1e-4,
    rel_tol: 1e-4,
    justification: "Thornthwaite (1948) polynomial coefficients: 4-digit precision",
};

/// Blaney-Criddle daylight fraction `p`: FAO-24 Table 18 interpolation precision.
pub const BLANEY_CRIDDLE_DAYLIGHT: Tolerance = Tolerance {
    name: "blaney_criddle_daylight",
    abs_tol: 0.015,
    rel_tol: 0.05,
    justification: "FAO-24 Table 18 p values: ±0.015 covers latitude interpolation and solar model",
};

/// Bangkok saturation vapour pressure: wider tolerance for high-temperature range.
pub const ET0_SAT_VAPOUR_PRESSURE_WIDE: Tolerance = Tolerance {
    name: "et0_sat_vapour_pressure_wide",
    abs_tol: 0.02,
    rel_tol: 1e-3,
    justification: "FAO-56 Ex 17 Bangkok: high-T range doubles Tetens rounding to 0.02 kPa",
};

/// Hargreaves vs PM cross-method tolerance (percent).
pub const ET0_CROSS_METHOD_PCT: Tolerance = Tolerance {
    name: "et0_cross_method_pct",
    abs_tol: 25.0,
    rel_tol: 0.0,
    justification: "Literature: Hargreaves vs PM 10-30% divergence; 25% accommodates Great Lakes climate",
};

/// Monte Carlo ET₀ propagation: O(1/√N) sampling noise for N=1000 samples.
///
/// Central limit theorem: `std_error` ≈ σ/√N. For ET₀ with σ ≈ 1 mm/day and
/// N = 1000, expected error ≈ 0.03 mm. Tolerance of 0.5 mm provides ~16σ
/// headroom for worst-case variance amplification through nonlinear chains.
pub const MC_ET0_PROPAGATION: Tolerance = Tolerance {
    name: "mc_et0_propagation",
    abs_tol: 0.5,
    rel_tol: 0.1,
    justification: "O(1/√N) CLT convergence: σ/√1000 ≈ 0.03; 0.5 provides 16σ headroom",
};

/// Rust ↔ Python cross-validation: IEEE-754 rounding at 1e-5.
pub const CROSS_VALIDATION: Tolerance = Tolerance {
    name: "cross_validation",
    abs_tol: 1e-5,
    rel_tol: 1e-5,
    justification: "Rust vs Python f64: IEEE-754 produces ~1e-10 diffs; 1e-5 is conservative",
};

/// Minimum R² for ET₀ model-observation fit.
pub const R2_MINIMUM: Tolerance = Tolerance {
    name: "r2_minimum",
    abs_tol: 0.85,
    rel_tol: 0.0,
    justification: "FAO-56 PM typically R² > 0.90; 0.85 allows for ERA5 reanalysis noise",
};

/// Maximum RMSE (mm/day) for ET₀ validation.
pub const RMSE_MAXIMUM: Tolerance = Tolerance {
    name: "rmse_maximum",
    abs_tol: 1.5,
    rel_tol: 0.0,
    justification: "Doorenbos & Pruitt (1977): ±1.5 mm/day ET₀ measurement uncertainty",
};
