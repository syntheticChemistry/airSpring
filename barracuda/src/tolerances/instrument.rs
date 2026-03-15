// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sensor, `IoT`, NPU, biodiversity, and statistical quality tolerances.

use super::Tolerance;

/// Polynomial/analytical sensor calibration: exact-match for simple arithmetic.
pub const SENSOR_EXACT: Tolerance = Tolerance {
    name: "sensor_exact",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "Polynomial evaluation and linear regression: f64-exact",
};

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
    justification: "Simpson 1-D is a summation of p²; pure f64 arithmetic matches Python exactly to 1e-10",
};

/// Bray-Curtis dissimilarity: pairwise distance matrix, f64 summation.
pub const BIO_BRAY_CURTIS: Tolerance = Tolerance {
    name: "bio_bray_curtis",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "Bray-Curtis is |Σ|aᵢ-bᵢ|| / Σ(aᵢ+bᵢ); f64 matches scipy.spatial.distance.braycurtis to 1e-8",
};

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
