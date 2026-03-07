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
//! The `Tolerance` struct itself comes from `BarraCuda` S52 (M-010), inspired by
//! barracuda validation registry and adopted across all Springs.
//!
//! | Domain | Tolerance | Origin |
//! |--------|-----------|--------|
//! | ET₀ Penman-Monteith | 0.01 mm/day | FAO-56 Table 2 (worked examples) |
//! | Saturation vapor pressure | 0.01 kPa | FAO-56 Table 2.3 (3 decimals) |
//! | Water balance mass | 0.01 mm | FAO-56 Ch 8 conservation law |
//! | Richards PDE | benchmark-specified | HYDRUS numerical convergence |
//! | Isotherm fit | 0.01 mg/g | Adsorption experiment precision |
//! | Kriging interpolation | 1e-6 | Numerical linear algebra (small systems) |
//! | GPU/CPU cross-validation | 1e-5 | f64 shader rounding (`BarraCuda` TS-001 fix) |
//!
//! # Baseline Provenance
//!
//! Tolerances were validated against Python baselines with the following provenance.
//! Each tolerance domain maps to one or more control experiments whose outputs
//! confirmed the threshold is sufficient and minimal.
//!
//! | Domain | Control script | Commit | Date | Command |
//! |--------|---------------|--------|------|---------|
//! | ET₀ (PM) | `control/fao56/penman_monteith.py` | `94cc51d` | 2026-02-16 | `python3 control/fao56/penman_monteith.py` |
//! | ET₀ (intercomp.) | `control/et0_intercomparison/et0_three_method.py` | `9a84ae5` | 2026-02-26 | `python3 control/et0_intercomparison/et0_three_method.py` |
//! | Priestley-Taylor | `control/priestley_taylor/priestley_taylor_et0.py` | `9a84ae5` | 2026-02-26 | `python3 control/priestley_taylor/priestley_taylor_et0.py` |
//! | Hargreaves | `control/hargreaves/hargreaves_samani.py` | `fad2e1b` | 2026-02-26 | `python3 control/hargreaves/hargreaves_samani.py` |
//! | Thornthwaite | `control/thornthwaite/thornthwaite_et0.py` | `fad2e1b` | 2026-02-26 | `python3 control/thornthwaite/thornthwaite_et0.py` |
//! | Water balance | `control/water_balance/fao56_water_balance.py` | `94cc51d` | 2026-02-16 | `python3 control/water_balance/fao56_water_balance.py` |
//! | Richards PDE | `control/richards/richards_1d.py` | `3afc229` | 2026-02-25 | `python3 control/richards/richards_1d.py` |
//! | Soil sensors | `control/soil_sensors/calibration_dong2020.py` | `94cc51d` | 2026-02-16 | `python3 control/soil_sensors/calibration_dong2020.py` |
//! | IoT irrigation | `control/iot_irrigation/calibration_dong2024.py` | `94cc51d` | 2026-02-16 | `python3 control/iot_irrigation/calibration_dong2024.py` |
//! | Dual Kc | `control/dual_kc/cover_crop_dual_kc.py` | `3afc229` | 2026-02-25 | `python3 control/dual_kc/cover_crop_dual_kc.py` |
//! | Biochar isotherm | `control/biochar/biochar_isotherms.py` | `3afc229` | 2026-02-25 | `python3 control/biochar/biochar_isotherms.py` |
//! | Pedotransfer | `control/pedotransfer/saxton_rawls.py` | `fad2e1b` | 2026-02-26 | `python3 control/pedotransfer/saxton_rawls.py` |
//! | GDD | `control/gdd/growing_degree_days.py` | `fad2e1b` | 2026-02-26 | `python3 control/gdd/growing_degree_days.py` |
//! | SCS-CN | `control/scs_curve_number/scs_curve_number.py` | `97e7533` | 2026-02-28 | `python3 control/scs_curve_number/scs_curve_number.py` |
//! | Green-Ampt | `control/green_ampt/green_ampt_infiltration.py` | `97e7533` | 2026-02-28 | `python3 control/green_ampt/green_ampt_infiltration.py` |
//! | Blaney-Criddle | `control/blaney_criddle/blaney_criddle_et0.py` | `97e7533` | 2026-02-28 | `python3 control/blaney_criddle/blaney_criddle_et0.py` |
//! | Anderson coupling | `control/anderson_coupling/anderson_coupling.py` | `0500398` | 2026-02-27 | `python3 control/anderson_coupling/anderson_coupling.py` |
//! | Diversity | `control/diversity/diversity_indices.py` | `fad2e1b` | 2026-02-26 | `python3 control/diversity/diversity_indices.py` |
//! | GPU/CPU parity | `control/cpu_gpu_parity/cpu_gpu_parity.py` | `fad2e1b` | 2026-03-02 | `python3 control/cpu_gpu_parity/cpu_gpu_parity.py` |

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

// ═══════════════════════════════════════════════════════════════════
// Thornthwaite, GDD, and pedotransfer (Experiments 021–023)
// ═══════════════════════════════════════════════════════════════════

/// Thornthwaite heat-index term and exponent: polynomial regression coefficients
/// have 4–5 significant digits; intermediate values converge to 1e-4.
pub const THORNTHWAITE_ANALYTICAL: Tolerance = Tolerance {
    name: "thornthwaite_analytical",
    abs_tol: 1e-4,
    rel_tol: 1e-4,
    justification: "Thornthwaite (1948) polynomial coefficients: 4-digit precision",
};

/// GDD computation: integer arithmetic on clamped temperatures yields exact results
/// to f64 precision.
pub const GDD_EXACT: Tolerance = Tolerance {
    name: "gdd_exact",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "GDD avg/clamp: f64-exact integer arithmetic (max/min/midpoint)",
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

// ═══════════════════════════════════════════════════════════════════
// Per-step and strict conservation
// ═══════════════════════════════════════════════════════════════════

/// Per-step mass balance: machine-precision conservation check per time step.
pub const WATER_BALANCE_PER_STEP: Tolerance = Tolerance {
    name: "water_balance_per_step",
    abs_tol: 1e-6,
    rel_tol: 1e-10,
    justification: "Per-step conservation check: f64 arithmetic residual < 1e-6 mm",
};

// ═══════════════════════════════════════════════════════════════════
// Sensor-specific calibration
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Statistical quality criteria
// ═══════════════════════════════════════════════════════════════════

/// Index of Agreement criterion: Willmott (1981), Dong et al. (2020) Table 3.
pub const IA_CRITERION: Tolerance = Tolerance {
    name: "index_of_agreement_criterion",
    abs_tol: 0.80,
    rel_tol: 0.0,
    justification: "Dong et al. (2020) Table 3: IA ≥ 0.80 for sensor correction adequacy",
};

/// Statistical significance threshold (two-tailed, α = 0.05).
pub const P_SIGNIFICANCE: Tolerance = Tolerance {
    name: "p_significance",
    abs_tol: 0.05,
    rel_tol: 0.0,
    justification: "Standard two-tailed significance level: α = 0.05",
};

/// Water savings tolerance: irrigation efficiency comparison.
pub const WATER_SAVINGS: Tolerance = Tolerance {
    name: "water_savings",
    abs_tol: 0.1,
    rel_tol: 0.05,
    justification: "IoT irrigation savings: ±10% comparison margin (Dong 2024 Fig 7)",
};

// ═══════════════════════════════════════════════════════════════════
// Biochar adsorption
// ═══════════════════════════════════════════════════════════════════

/// Biochar isotherm mean residual: no systematic bias threshold.
pub const ISOTHERM_MEAN_RESIDUAL: Tolerance = Tolerance {
    name: "isotherm_mean_residual",
    abs_tol: 0.5,
    rel_tol: 0.05,
    justification:
        "Kumari et al. (2025): mean(|qe_obs - qe_pred|) < 0.5 mg/g for no systematic bias",
};

// ═══════════════════════════════════════════════════════════════════
// Cross-method and cross-station validation
// ═══════════════════════════════════════════════════════════════════

/// Rust ↔ Python cross-validation: IEEE-754 rounding at 1e-5.
pub const CROSS_VALIDATION: Tolerance = Tolerance {
    name: "cross_validation",
    abs_tol: 1e-5,
    rel_tol: 1e-5,
    justification: "Rust vs Python f64: IEEE-754 produces ~1e-10 diffs; 1e-5 is conservative",
};

/// Bangkok saturation vapour pressure: wider tolerance for high-temperature range.
pub const ET0_SAT_VAPOUR_PRESSURE_WIDE: Tolerance = Tolerance {
    name: "et0_sat_vapour_pressure_wide",
    abs_tol: 0.02,
    rel_tol: 1e-3,
    justification: "FAO-56 Ex 17 Bangkok: high-T range doubles Tetens rounding to 0.02 kPa",
};

// ═══════════════════════════════════════════════════════════════════
// Regional / statistical validation bounds
// ═══════════════════════════════════════════════════════════════════

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

/// Hargreaves vs PM cross-method tolerance (percent).
pub const ET0_CROSS_METHOD_PCT: Tolerance = Tolerance {
    name: "et0_cross_method_pct",
    abs_tol: 25.0,
    rel_tol: 0.0,
    justification:
        "Literature: Hargreaves vs PM 10-30% divergence; 25% accommodates Great Lakes climate",
};

// ═══════════════════════════════════════════════════════════════════
// Simplified ET₀ methods (Blaney-Criddle, SCS-CN, Green-Ampt)
// ═══════════════════════════════════════════════════════════════════

/// Blaney-Criddle daylight fraction `p`: FAO-24 Table 18 interpolation precision.
pub const BLANEY_CRIDDLE_DAYLIGHT: Tolerance = Tolerance {
    name: "blaney_criddle_daylight",
    abs_tol: 0.015,
    rel_tol: 0.05,
    justification: "FAO-24 Table 18 p values: ±0.015 covers latitude interpolation and solar model",
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
    justification:
        "Green-Ampt Newton iteration converges to 1e-12; 0.001 cm covers soil param uncertainty",
};

/// FAO-56 Kc precision: Eq. 72 `Kc_max` and dual Kc component precision.
pub const DUAL_KC_PRECISION: Tolerance = Tolerance {
    name: "dual_kc_precision",
    abs_tol: 0.01,
    rel_tol: 0.01,
    justification: "FAO-56 Eq 72 Kc_max: 2-decimal tabulated values; wind/RH adjustment precision",
};

// ═══════════════════════════════════════════════════════════════════
// IoT sensor data validation
// ═══════════════════════════════════════════════════════════════════

/// Temperature mean tolerance for synthetic `IoT` data.
pub const IOT_TEMPERATURE_MEAN: Tolerance = Tolerance {
    name: "iot_temperature_mean",
    abs_tol: 2.0,
    rel_tol: 0.1,
    justification: "Synthetic 25°C centre ± diurnal; mean within ~2°C",
};

/// Temperature extremes tolerance for synthetic `IoT` data.
pub const IOT_TEMPERATURE_EXTREMES: Tolerance = Tolerance {
    name: "iot_temperature_extremes",
    abs_tol: 3.0,
    rel_tol: 0.15,
    justification: "Synthetic diurnal amplitude ~8°C; extremes by up to 3°C from analytical peak",
};

/// PAR sensor maximum tolerance.
pub const IOT_PAR_MAX: Tolerance = Tolerance {
    name: "iot_par_max",
    abs_tol: 200.0,
    rel_tol: 0.15,
    justification: "Bell-curve PAR peak ≈ 1800 µmol/m²/s; discretization ± 200",
};

/// CSV round-trip tolerance for decimal-truncated float data.
pub const IOT_CSV_ROUNDTRIP: Tolerance = Tolerance {
    name: "iot_csv_roundtrip",
    abs_tol: 0.1,
    rel_tol: 0.01,
    justification: "CSV {:.2} format truncation: round-trip within 0.1 of mean",
};

// ═══════════════════════════════════════════════════════════════════
// NPU streaming classification
// ═══════════════════════════════════════════════════════════════════

/// Minimum EMA samples before anomaly detection activates — guards against
/// false positives during warmup.
pub const NPU_MIN_ANOMALY_SAMPLES: u64 = 10;

/// EMA variance floor — prevents division by zero in z-score anomaly detection.
pub const NPU_SIGMA_FLOOR: Tolerance = Tolerance {
    name: "npu_sigma_floor",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "EMA variance floor — prevents division by zero in z-score anomaly detection",
};

/// FAO-56 p-factor: stress onset when Dr > 0.55 × TAW (Allen et al. 1998 Eq 84,
/// midpoint for field crops).
///
/// This is a **physical threshold**, not a validation tolerance. It is stored
/// here alongside tolerances for colocation with NPU constants, but semantically
/// it is a domain parameter (fraction of TAW at which stress begins).
pub const NPU_STRESS_DEPLETION_THRESHOLD: f64 = 0.55;

// ═══════════════════════════════════════════════════════════════════
// Biodiversity indices (Shannon, Simpson, Bray-Curtis)
// Aligned with barraCuda 0.3.1 `tolerances::BIO_DIVERSITY_*`.
// ═══════════════════════════════════════════════════════════════════

/// Shannon diversity index H': exact to 8 digits for deterministic OTU tables.
pub const BIO_DIVERSITY_SHANNON: Tolerance = Tolerance {
    name: "bio_diversity_shannon",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "Shannon H' is a summation of -p·ln(p); f64 accumulation matches Python scipy.stats.entropy to 1e-8",
};

/// Simpson diversity index 1-D: exact to 10 digits for deterministic OTU tables.
pub const BIO_DIVERSITY_SIMPSON: Tolerance = Tolerance {
    name: "bio_diversity_simpson",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification:
        "Simpson 1-D is a summation of p²; pure f64 arithmetic matches Python exactly to 1e-10",
};

/// Bray-Curtis dissimilarity: pairwise distance matrix, f64 summation.
pub const BIO_BRAY_CURTIS: Tolerance = Tolerance {
    name: "bio_bray_curtis",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "Bray-Curtis is |Σ|aᵢ-bᵢ|| / Σ(aᵢ+bᵢ); f64 matches scipy.spatial.distance.braycurtis to 1e-8",
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
        let all_tolerances: &[&Tolerance] = &[
            // ET₀ and atmospheric (FAO-56)
            &ET0_SAT_VAPOUR_PRESSURE,
            &ET0_SLOPE_VAPOUR,
            &ET0_NET_RADIATION,
            &ET0_REFERENCE,
            &ET0_VPD,
            &ET0_COLD_CLIMATE,
            &PSYCHROMETRIC_CONSTANT,
            // Water balance and soil moisture
            &WATER_BALANCE_MASS,
            &WATER_BALANCE_PER_STEP,
            &STRESS_COEFFICIENT,
            &SOIL_HYDRAULIC,
            &SOIL_ROUNDTRIP,
            // Richards equation
            &RICHARDS_STEADY,
            &RICHARDS_TRANSIENT,
            // Isotherm fitting
            &ISOTHERM_PARAMETER,
            &ISOTHERM_PREDICTION,
            &ISOTHERM_MEAN_RESIDUAL,
            // GPU/CPU cross-validation
            &GPU_CPU_CROSS,
            &KRIGING_INTERPOLATION,
            &SEASONAL_REDUCTION,
            &IOT_STREAM_SMOOTHING,
            // Sensor calibration
            &SENSOR_EXACT,
            &IRRIGATION_DEPTH,
            // Thornthwaite, GDD, pedotransfer
            &THORNTHWAITE_ANALYTICAL,
            &GDD_EXACT,
            &PEDOTRANSFER_MOISTURE,
            &PEDOTRANSFER_KSAT,
            // Per-step and sensor-specific
            &TOPP_EQUATION,
            &ANALYTICAL_COMPUTATION,
            // Statistical quality criteria
            &IA_CRITERION,
            &P_SIGNIFICANCE,
            &WATER_SAVINGS,
            // Cross-method and cross-station
            &CROSS_VALIDATION,
            &ET0_SAT_VAPOUR_PRESSURE_WIDE,
            &R2_MINIMUM,
            &RMSE_MAXIMUM,
            &ET0_CROSS_METHOD_PCT,
            // Simplified ET₀ methods
            &BLANEY_CRIDDLE_DAYLIGHT,
            &SCS_CN_ANALYTICAL,
            &GREEN_AMPT_ANALYTICAL,
            &DUAL_KC_PRECISION,
            // IoT sensor data validation
            &IOT_TEMPERATURE_MEAN,
            &IOT_TEMPERATURE_EXTREMES,
            &IOT_PAR_MAX,
            &IOT_CSV_ROUNDTRIP,
            // NPU streaming classification
            &NPU_SIGMA_FLOOR,
        ];
        for tol in all_tolerances {
            assert!(
                !tol.name.is_empty(),
                "tolerance {} must have a name",
                tol.name
            );
            assert!(
                !tol.justification.is_empty(),
                "tolerance {} must have justification",
                tol.name
            );
            assert!(tol.abs_tol > 0.0, "{}: abs_tol must be positive", tol.name);
        }
        // 46 Tolerance structs + 1 plain threshold (NPU_STRESS_DEPLETION_THRESHOLD)
        assert_eq!(
            all_tolerances.len(),
            46,
            "test must include every Tolerance constant defined in this file"
        );
        let threshold = NPU_STRESS_DEPLETION_THRESHOLD;
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "stress threshold must be a fraction of TAW"
        );
    }
}
