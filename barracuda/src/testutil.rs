//! Validation and test utilities.
//!
//! Synthetic data generators for testing parsers, statistics, and validation
//! binaries. These produce deterministic data with known properties so that
//! computed results can be verified analytically.
//!
//! # Production use
//!
//! This module is NOT for production data ingestion. It exists to support:
//! - Unit tests in [`crate::io::csv_ts`]
//! - Validation binary `validate_iot`
//! - Integration tests in `tests/integration.rs`

use crate::io::csv_ts::TimeseriesData;

/// Generate a synthetic `IoT` sensor CSV dataset for validation and testing.
///
/// Produces deterministic data with known statistical properties:
/// - Temperature: diurnal cycle, mean = 25 °C, amplitude ±8 °C
/// - Soil moisture: slowly decreasing, range 0.10–0.40 m³/m³
/// - PAR: bell curve centered at solar noon, max ≈ 1800 µmol/m²/s
/// - Humidity: inverse of temperature, range 55–85%
///
/// # Analytical properties (for 7+ full days)
///
/// | Column | Mean | Min | Max |
/// |--------|------|-----|-----|
/// | `temperature` | 25.0 | 17.0 | 33.0 |
/// | `humidity` | 70.0 | 55.0 | 85.0 |
/// | `par` | ~464 | 0.0 | 1800.0 |
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn generate_synthetic_iot_data(n_records: usize) -> TimeseriesData {
    use std::f64::consts::PI;

    let column_names = vec![
        "soil_moisture_1".to_string(),
        "soil_moisture_2".to_string(),
        "temperature".to_string(),
        "humidity".to_string(),
        "par".to_string(),
    ];
    let mut timestamps = Vec::with_capacity(n_records);
    let mut cols: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(n_records)).collect();

    for i in 0..n_records {
        let hour = i % 24;
        let day = i / 24;
        timestamps.push(format!("2024-07-{:02}T{hour:02}:00:00", day + 1));

        // Soil moisture: slowly decreasing with daily ET
        let base_sm = 0.002f64.mul_add(-(i as f64), 0.28);
        let sm1 = (0.02f64.mul_add((i as f64).mul_add(0.1, 0.0).sin(), base_sm)).clamp(0.10, 0.40);
        let sm2 = (0.015f64.mul_add((i as f64).mul_add(0.1, 1.0).sin(), base_sm - 0.01))
            .clamp(0.10, 0.40);
        cols[0].push(sm1);
        cols[1].push(sm2);

        // Temperature: diurnal cycle centered at 14:00
        let temp = 8.0f64.mul_add(((hour as f64 - 14.0) * PI / 12.0).cos(), 25.0);
        cols[2].push(temp);

        // Humidity: inverse of temperature
        let rh = (-15.0f64).mul_add(((hour as f64 - 14.0) * PI / 12.0).cos(), 70.0);
        cols[3].push(rh);

        // PAR: bell curve centered at solar noon
        let par = if (6..=20).contains(&hour) {
            1800.0 * (-(((hour as f64 - 13.0) / 3.5).powi(2))).exp()
        } else {
            0.0
        };
        cols[4].push(par);
    }

    TimeseriesData::new(column_names, timestamps, cols)
}

/// Compute Pearson R² between observed and simulated data.
///
/// Uses barracuda's `pearson_correlation` primitive for cross-validation.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] if the barracuda
/// primitive fails (e.g. length mismatch, degenerate data).
#[must_use = "R² value should be checked"]
pub fn r_squared(observed: &[f64], simulated: &[f64]) -> crate::error::Result<f64> {
    let r = barracuda::stats::pearson_correlation(observed, simulated)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
    Ok(r * r)
}

/// Compute Root Mean Square Error (RMSE).
///
/// RMSE = sqrt(Σ(obs - sim)² / n)
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rmse(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    let n = observed.len() as f64;
    let sum_sq: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

/// Compute Mean Bias Error (MBE).
///
/// MBE = Σ(sim - obs) / n
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mbe(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    let n = observed.len() as f64;
    let sum_bias: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| s - o)
        .sum();
    sum_bias / n
}

/// Index of Agreement (Willmott, 1981).
///
/// IA = 1 − Σ(Mᵢ − Pᵢ)² / Σ(|Pᵢ − M̄| + |Mᵢ − M̄|)²
///
/// Ported from the Python baseline (`control/soil_sensors/calibration_dong2020.py`
/// `compute_ia`).  Values range from 0.0 (no agreement) to 1.0 (perfect).
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths or are empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn index_of_agreement(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    assert!(!observed.is_empty(), "Vectors must not be empty");

    let n = observed.len() as f64;
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let numerator: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .sum();

    let denominator: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| ((s - mean_obs).abs() + (o - mean_obs).abs()).powi(2))
        .sum();

    if denominator == 0.0 {
        return 1.0;
    }
    1.0 - numerator / denominator
}

/// Nash-Sutcliffe Efficiency (NSE).
///
/// NSE = 1 − Σ(Obsᵢ − Simᵢ)² / Σ(Obsᵢ − Obs̄)²
///
/// Widely used in hydrology (Nash & Sutcliffe, 1970). NSE = 1.0 is perfect
/// agreement; NSE < 0 means the model is worse than using the mean.
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths or are empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn nash_sutcliffe(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    assert!(!observed.is_empty(), "Vectors must not be empty");

    let n = observed.len() as f64;
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let ss_res: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .sum();

    let ss_tot: f64 = observed.iter().map(|o| (o - mean_obs).powi(2)).sum();

    if ss_tot == 0.0 {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

/// Coefficient of determination (R²) using sum-of-squares method.
///
/// R² = 1 − `SS_res` / `SS_tot`
///
/// Unlike [`r_squared`], which wraps barracuda's Pearson R, this uses the
/// standard regression definition: it can be negative if the model is poor.
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths or are empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn coefficient_of_determination(observed: &[f64], simulated: &[f64]) -> f64 {
    nash_sutcliffe(observed, simulated)
}

// ── BarraCUDA stats integration ─────────────────────────────────────

/// Spearman rank correlation coefficient.
///
/// Wraps [`barracuda::stats::correlation::spearman_correlation`] for
/// nonparametric validation — useful when the relationship between
/// observed and simulated data may not be strictly linear.
///
/// Returns a value in \[-1, 1\].
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "Spearman ρ value should be checked"]
pub fn spearman_r(observed: &[f64], simulated: &[f64]) -> crate::error::Result<f64> {
    barracuda::stats::correlation::spearman_correlation(observed, simulated)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
}

/// Bootstrap confidence interval for RMSE.
///
/// Uses [`barracuda::stats::bootstrap::bootstrap_ci`] to compute a
/// non-parametric confidence interval around the RMSE estimate. This
/// quantifies the uncertainty in our error metric.
///
/// Returns `(lower, upper)` bounds at the specified confidence level.
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[allow(clippy::cast_precision_loss)]
#[must_use = "bootstrap CI should be checked"]
pub fn bootstrap_rmse(
    observed: &[f64],
    simulated: &[f64],
    n_bootstrap: usize,
    confidence: f64,
) -> crate::error::Result<(f64, f64)> {
    assert_eq!(observed.len(), simulated.len());
    // Compute residuals for bootstrap
    let residuals: Vec<f64> = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .collect();

    let ci = barracuda::stats::bootstrap_ci(
        &residuals,
        |data| {
            let n = data.len() as f64;
            if n == 0.0 {
                return 0.0;
            }
            (data.iter().sum::<f64>() / n).sqrt()
        },
        n_bootstrap,
        confidence,
        42, // deterministic seed
    )
    .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;

    Ok((ci.lower, ci.upper))
}

/// Compute sample variance using barracuda's implementation.
///
/// Wraps [`barracuda::stats::correlation::variance`] for consistency
/// with the barracuda ecosystem.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "variance value should be checked"]
pub fn variance(data: &[f64]) -> crate::error::Result<f64> {
    barracuda::stats::correlation::variance(data)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
}

/// Compute sample standard deviation using barracuda's implementation.
///
/// Wraps [`barracuda::stats::correlation::std_dev`].
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "standard deviation value should be checked"]
pub fn std_deviation(data: &[f64]) -> crate::error::Result<f64> {
    barracuda::stats::correlation::std_dev(data)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
}
