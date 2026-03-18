// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atmospheric parameters and vapour pressure for ET₀ calculations.
//!
//! FAO-56 Eqs. 7–8 (pressure, psychrometric), Eqs. 11–17 (vapour pressure),
//! Eq. 47 (wind speed adjustment).

// ── FAO-56 constants ─────────────────────────────────────────────────

/// Psychrometric coefficient (kPa/°C per kPa pressure). FAO-56 Eq. 8.
const PSYCHROMETRIC_COEFF: f64 = 0.665e-3;
/// Sea-level atmospheric pressure (kPa). FAO-56 Eq. 7.
const SEA_LEVEL_PRESSURE_KPA: f64 = 101.3;
/// Standard temperature lapse rate (°C/m). FAO-56 Eq. 7.
const LAPSE_RATE: f64 = 0.0065;
/// Standard base temperature (K). FAO-56 Eq. 7.
const BASE_TEMP_K: f64 = 293.0;
/// Pressure exponent. FAO-56 Eq. 7.
const PRESSURE_EXPONENT: f64 = 5.26;
/// Magnus formula coefficient a. FAO-56 Eq. 11.
const MAGNUS_A: f64 = 0.6108;
/// Magnus formula coefficient b. FAO-56 Eq. 11.
const MAGNUS_B: f64 = 17.27;
/// Magnus formula coefficient c (°C). FAO-56 Eq. 11.
const MAGNUS_C: f64 = 237.3;
/// Vapour pressure slope numerator. FAO-56 Eq. 13.
const VP_SLOPE_NUMERATOR: f64 = 4098.0;

// ── Atmospheric parameters ───────────────────────────────────────────

/// Psychrometric constant γ (kPa/°C).
///
/// γ = 0.665 × 10⁻³ × P
///
/// FAO-56 Eq. 8.
#[must_use]
pub fn psychrometric_constant(pressure_kpa: f64) -> f64 {
    PSYCHROMETRIC_COEFF * pressure_kpa
}

/// Atmospheric pressure from elevation (kPa).
///
/// P = 101.3 × ((293 − 0.0065z) / 293)^5.26
///
/// FAO-56 Eq. 7.
#[must_use]
pub fn atmospheric_pressure(elevation_m: f64) -> f64 {
    SEA_LEVEL_PRESSURE_KPA
        * (LAPSE_RATE.mul_add(-elevation_m, BASE_TEMP_K) / BASE_TEMP_K).powf(PRESSURE_EXPONENT)
}

// ── Vapour pressure functions ────────────────────────────────────────

/// Saturation vapour pressure e°(T) (kPa) at temperature T (°C).
///
/// FAO-56 Eq. 11: e°(T) = 0.6108 × exp(17.27T / (T + 237.3))
#[must_use]
pub fn saturation_vapour_pressure(temp_c: f64) -> f64 {
    MAGNUS_A * ((MAGNUS_B * temp_c) / (temp_c + MAGNUS_C)).exp()
}

/// Slope of saturation vapour pressure curve Δ (kPa/°C).
///
/// FAO-56 Eq. 13: Δ = 4098 × e°(T) / (T + 237.3)²
#[must_use]
pub fn vapour_pressure_slope(temp_c: f64) -> f64 {
    let es = saturation_vapour_pressure(temp_c);
    VP_SLOPE_NUMERATOR * es / (temp_c + MAGNUS_C).powi(2)
}

/// Mean saturation vapour pressure from `Tmin` and `Tmax`.
///
/// FAO-56 Eq. 12: es = \[e°(Tmax) + e°(Tmin)\] / 2
#[must_use]
pub fn mean_saturation_vapour_pressure(tmin: f64, tmax: f64) -> f64 {
    f64::midpoint(
        saturation_vapour_pressure(tmax),
        saturation_vapour_pressure(tmin),
    )
}

/// Actual vapour pressure from dewpoint temperature.
///
/// FAO-56 Eq. 14: ea = e°(Tdew)
#[must_use]
pub fn actual_vapour_pressure_dewpoint(tdew: f64) -> f64 {
    saturation_vapour_pressure(tdew)
}

/// Actual vapour pressure from relative humidity (kPa).
///
/// FAO-56 Eq. 17: ea = \[e°(Tmin) × `RHmax` + e°(Tmax) × `RHmin`\] / 200
#[must_use]
pub fn actual_vapour_pressure_rh(tmin: f64, tmax: f64, rh_min: f64, rh_max: f64) -> f64 {
    let e_tmin = saturation_vapour_pressure(tmin);
    let e_tmax = saturation_vapour_pressure(tmax);
    f64::midpoint(e_tmin * rh_max / 100.0, e_tmax * rh_min / 100.0)
}

// ── Wind speed adjustment ─────────────────────────────────────────────

/// Convert wind speed measured at height `z_m` to the standard 2 m height.
///
/// FAO-56 Eq. 47: u₂ = uz × 4.87 / ln(67.8z − 5.42)
///
/// Most weather stations measure wind at 10 m. The Penman-Monteith equation
/// requires wind at 2 m. This conversion assumes logarithmic wind profile.
///
/// # Errors
///
/// Returns `InvalidInput` if `z_m` ≤ 0.0 (physically impossible measurement height).
pub fn wind_speed_at_2m(uz: f64, z_m: f64) -> crate::error::Result<f64> {
    if z_m <= 0.0 {
        return Err(crate::error::AirSpringError::InvalidInput(format!(
            "measurement height must be positive: {z_m}"
        )));
    }
    Ok(uz * 4.87 / (67.8f64.mul_add(z_m, -5.42)).ln())
}
