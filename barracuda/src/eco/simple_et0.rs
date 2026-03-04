// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simplified ET₀ methods for data-sparse deployments.
//!
//! These methods require fewer inputs than the full FAO-56 Penman-Monteith
//! equation and are appropriate when wind, humidity, or radiation data are
//! unavailable. They trade accuracy for practicality in data-scarce regions,
//! long-term reconstruction, and rapid screening.
//!
//! | Method | Inputs | Use Case |
//! |--------|--------|----------|
//! | Makkink (1957) | T, Rs | Northern Europe (KNMI standard) |
//! | Turc (1961) | T, Rs, RH | Humid / sub-humid climates |
//! | Hamon (1961) | T, N | Temperature-only reconstruction |
//! | Blaney-Criddle (1950) | T, p | Western US irrigation districts |
//!
//! All methods clamp output to ≥ 0.0 (ET₀ is non-negative by definition).

use super::evapotranspiration::{
    atmospheric_pressure, psychrometric_constant, saturation_vapour_pressure, vapour_pressure_slope,
};
use super::solar::daylight_hours;

// ── Hamon constants ─────────────────────────────────────────────────

/// Hamon saturation density coefficient (g/m³ per kPa/K). Lu et al. (2005).
const HAMON_RHO_COEFF: f64 = 216.7;
/// Hamon absolute temperature offset (K). Lu et al. (2005).
const HAMON_TEMP_OFFSET_K: f64 = 273.3;
/// Hamon PET coefficient. Lu et al. (2005).
const HAMON_PET_COEFF: f64 = 0.1651;

// ── Turc humidity correction constants ─────────────────────────────

/// Turc (1961) RH threshold for humidity correction (%). Below this,
/// ET₀ is increased by a dryness factor. Turc (1961) Eq. 2.
const TURC_RH_THRESHOLD_PCT: f64 = 50.0;
/// Turc (1961) humidity correction denominator (%). Turc (1961) Eq. 2.
const TURC_RH_CORRECTION_RANGE: f64 = 70.0;
/// Turc (1961) temperature factor denominator offset (°C).
const TURC_TEMP_DENOM_OFFSET: f64 = 15.0;
/// Turc (1961) empirical coefficient. Turc (1961) Eq. 1.
const TURC_COEFF: f64 = 0.013;

// ── Blaney-Criddle constants ────────────────────────────────────────

/// Blaney-Criddle annual daylight hours (approx 4380 hrs / 100). FAO-24.
const BC_ANNUAL_DAYLIGHT: f64 = 43.80;
/// Blaney-Criddle temperature coefficient. USDA-SCS (1950).
const BC_TEMP_COEFF: f64 = 0.46;
/// Blaney-Criddle offset constant. USDA-SCS (1950).
const BC_OFFSET: f64 = 8.13;

// ── Makkink (1957) ──────────────────────────────────────────────────

/// Makkink (1957) radiation-based ET₀ estimate (mm/day).
///
/// ```text
/// ET₀ = C₁ × (Δ/(Δ+γ)) × Rs/λ + C₂
/// ```
///
/// A radiation-only method requiring solar radiation and temperature.
/// Widely used in the Netherlands (KNMI standard) and Northern Europe.
/// The de Bruin (1987) coefficients C₁=0.61, C₂=−0.12 are standard.
///
/// # Reference
///
/// Makkink GF (1957) J Inst Water Eng 11:277-288.
/// de Bruin HAR (1987) From Penman to Makkink. TNO, The Hague, pp 5-31.
#[must_use]
pub fn makkink_et0(tmean_c: f64, rs_mj: f64, elevation_m: f64) -> f64 {
    const C1: f64 = 0.61;
    const C2: f64 = -0.12;
    const LAMBDA_MJ_KG: f64 = 2.45;
    let pressure = atmospheric_pressure(elevation_m);
    let gamma = psychrometric_constant(pressure);
    let delta = vapour_pressure_slope(tmean_c);
    C1.mul_add(delta / (delta + gamma) * (rs_mj / LAMBDA_MJ_KG), C2)
        .max(0.0)
}

// ── Turc (1961) ─────────────────────────────────────────────────────

/// Turc (1961) temperature-radiation ET₀ estimate (mm/day).
///
/// ```text
/// RH ≥ 50%: ET₀ = 0.013 × T/(T+15) × (23.8846 Rs + 50)
/// RH <  50%: ET₀ × (1 + (50−RH)/70)
/// ```
///
/// Requires temperature, solar radiation, and humidity. The conversion
/// factor 23.8846 converts MJ/m²/day to cal/cm²/day.
///
/// # Reference
///
/// Turc L (1961) Annales Agronomiques 12:13-49.
#[must_use]
pub fn turc_et0(tmean_c: f64, rs_mj: f64, rh_pct: f64) -> f64 {
    const MJ_TO_CAL_CM2: f64 = 23.8846;
    const RADIATION_OFFSET_CAL: f64 = 50.0;
    let denom = tmean_c + TURC_TEMP_DENOM_OFFSET;
    if denom == 0.0 {
        return 0.0;
    }
    let t_factor = tmean_c / denom;
    if t_factor < 0.0 {
        return 0.0;
    }
    let rs_cal = MJ_TO_CAL_CM2.mul_add(rs_mj, RADIATION_OFFSET_CAL);
    let mut et0 = TURC_COEFF * t_factor * rs_cal;
    if rh_pct < TURC_RH_THRESHOLD_PCT {
        et0 *= 1.0 + (TURC_RH_THRESHOLD_PCT - rh_pct) / TURC_RH_CORRECTION_RANGE;
    }
    et0.max(0.0)
}

// ── Hamon (1961) ────────────────────────────────────────────────────

/// Hamon (1961) temperature-based PET estimate (mm/day).
///
/// ```text
/// PET = 0.1651 × N × RHOSAT × KPEC
/// RHOSAT = 216.7 × e_s / (T + 273.3)  [g/m³]
/// ```
///
/// The simplest ET₀ method — requires only temperature and day length.
/// Appropriate for data-sparse deployments and long-term reconstruction
/// from temperature-only records.
///
/// Uses `KPEC = 1.0` per Lu et al. (2005).
///
/// # Reference
///
/// Hamon WR (1961) J Hydraulics Div ASCE 87(HY3):107-120.
/// Lu J, et al. (2005) J Am Water Resour Assoc 41(3):621-633.
#[must_use]
pub fn hamon_pet(tmean_c: f64, day_length_hours: f64) -> f64 {
    if tmean_c < 0.0 || day_length_hours <= 0.0 {
        return 0.0;
    }
    let es = saturation_vapour_pressure(tmean_c);
    let rhosat = HAMON_RHO_COEFF * es / (tmean_c + HAMON_TEMP_OFFSET_K);
    HAMON_PET_COEFF * day_length_hours * rhosat
}

/// Hamon PET from geographic location (computes day length internally).
#[must_use]
pub fn hamon_pet_from_location(tmean_c: f64, latitude_rad: f64, day_of_year: u32) -> f64 {
    if tmean_c < 0.0 {
        return 0.0;
    }
    let n = daylight_hours(latitude_rad, day_of_year);
    hamon_pet(tmean_c, n)
}

// ── Blaney-Criddle (1950) ───────────────────────────────────────────

/// Blaney-Criddle daylight fraction `p` from latitude and day-of-year.
///
/// p = N / 43.80, where N is daylight hours. Total annual daylight ≈ 4380 hrs
/// at any latitude (summer/winter balance). p ≈ 0.274 at equator.
///
/// # Reference
///
/// Doorenbos J, Pruitt WO (1977) FAO Irrigation and Drainage Paper 24, Table 18.
#[must_use]
pub fn blaney_criddle_p(latitude_rad: f64, day_of_year: u32) -> f64 {
    let n = daylight_hours(latitude_rad, day_of_year);
    n / BC_ANNUAL_DAYLIGHT
}

/// Blaney-Criddle (1950) PET estimate (mm/day).
///
/// ET₀ = p × (0.46 × T + 8.13)
///
/// The simplest widely-used PET method — requires only temperature and daylight
/// fraction. Widely used in western US irrigation districts.
///
/// # Arguments
///
/// * `tmean_c` — Mean temperature (°C).
/// * `p` — Blaney-Criddle daylight fraction (use [`blaney_criddle_p`]).
///
/// # Reference
///
/// Blaney HF, Criddle WD (1950) *Determining water requirements in irrigated
/// areas from climatological and irrigation data.* USDA-SCS Tech Paper 96.
#[must_use]
pub fn blaney_criddle_et0(tmean_c: f64, p: f64) -> f64 {
    (p * BC_TEMP_COEFF.mul_add(tmean_c, BC_OFFSET)).max(0.0)
}

/// Blaney-Criddle PET from location (latitude + DOY).
///
/// Convenience wrapper: computes daylight fraction `p` from latitude and DOY,
/// then applies the Blaney-Criddle equation.
///
/// # Arguments
///
/// * `tmean_c` — Mean temperature (°C).
/// * `latitude_rad` — Latitude in radians.
/// * `day_of_year` — Day of year (1–366).
#[must_use]
pub fn blaney_criddle_from_location(tmean_c: f64, latitude_rad: f64, day_of_year: u32) -> f64 {
    let p = blaney_criddle_p(latitude_rad, day_of_year);
    blaney_criddle_et0(tmean_c, p)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_makkink_analytical() {
        let et0 = makkink_et0(20.0, 15.0, 100.0);
        assert!((et0 - 2.438).abs() < 0.01, "Makkink(20,15,100) = {et0}");
        let et0_hot = makkink_et0(30.0, 25.0, 0.0);
        assert!(
            (et0_hot - 4.755).abs() < 0.01,
            "Makkink(30,25,0) = {et0_hot}"
        );
    }

    #[test]
    fn test_makkink_clamp_zero() {
        assert!(
            makkink_et0(20.0, 0.0, 100.0) < f64::EPSILON,
            "Zero Rs → clamped to 0"
        );
    }

    #[test]
    fn test_makkink_monotonicity() {
        let lo = makkink_et0(20.0, 10.0, 100.0);
        let hi = makkink_et0(20.0, 20.0, 100.0);
        assert!(hi > lo, "More radiation → more ET₀: {lo} < {hi}");
    }

    #[test]
    fn test_turc_high_humidity() {
        let et0 = turc_et0(20.0, 15.0, 70.0);
        assert!((et0 - 3.033).abs() < 0.01, "Turc(20,15,70) = {et0}");
    }

    #[test]
    fn test_turc_low_humidity_correction() {
        let et0_humid = turc_et0(30.0, 25.0, 55.0);
        let et0_dry = turc_et0(30.0, 25.0, 40.0);
        assert!(et0_dry > et0_humid, "Drier air → higher ET₀");
    }

    #[test]
    fn test_turc_negative_temp_clamp() {
        assert!(turc_et0(-5.0, 3.0, 90.0) < f64::EPSILON, "Negative T → 0");
    }

    #[test]
    fn test_turc_humidity_boundary_continuity() {
        let at50 = turc_et0(20.0, 15.0, 50.0);
        let at49 = turc_et0(20.0, 15.0, 49.99);
        assert!(
            (at50 - at49).abs() < 0.01,
            "Continuity at RH=50%: {at50} vs {at49}"
        );
    }

    #[test]
    fn test_hamon_analytical() {
        let pet = hamon_pet(20.0, 14.0);
        assert!((pet - 3.993).abs() < 0.01, "Hamon(20,14) = {pet}");
    }

    #[test]
    fn test_hamon_clamp_negative_temp() {
        assert!(hamon_pet(-5.0, 12.0) < f64::EPSILON);
    }

    #[test]
    fn test_hamon_clamp_zero_daylight() {
        assert!(hamon_pet(20.0, 0.0) < f64::EPSILON);
    }

    #[test]
    fn test_hamon_from_location() {
        let pet = hamon_pet_from_location(20.0, 42.0_f64.to_radians(), 172);
        assert!(pet > 3.0 && pet < 6.0, "Hamon at 42°N midsummer: {pet}");
    }

    #[test]
    fn test_hamon_monotonicity() {
        let cool = hamon_pet(10.0, 14.0);
        let warm = hamon_pet(30.0, 14.0);
        assert!(warm > cool, "Warmer → more PET");
        let short = hamon_pet(20.0, 10.0);
        let long = hamon_pet(20.0, 16.0);
        assert!(long > short, "Longer day → more PET");
    }

    #[test]
    fn test_blaney_criddle_equator_25c() {
        let et0 = blaney_criddle_et0(25.0, 0.274);
        assert!((et0 - 5.38).abs() < 0.02, "BC equator 25°C: {et0}");
    }

    #[test]
    fn test_blaney_criddle_negative_clamp() {
        let et0 = blaney_criddle_et0(-20.0, 0.199);
        assert!(et0.abs() < 1e-10, "BC -20°C should be 0: {et0}");
    }

    #[test]
    fn test_blaney_criddle_from_location_summer() {
        let lat = 40.0_f64.to_radians();
        let et0 = blaney_criddle_from_location(25.0, lat, 172);
        assert!(et0 > 4.0 && et0 < 8.0, "BC 40°N summer: {et0}");
    }

    #[test]
    fn test_blaney_criddle_p_equator() {
        let p = blaney_criddle_p(0.0, 172);
        assert!((p - 0.274).abs() < 0.005, "p equator: {p}");
    }

    #[test]
    fn test_blaney_criddle_temperature_monotonic() {
        let p = 0.274;
        let et0_10 = blaney_criddle_et0(10.0, p);
        let et0_20 = blaney_criddle_et0(20.0, p);
        let et0_30 = blaney_criddle_et0(30.0, p);
        assert!(et0_10 < et0_20 && et0_20 < et0_30);
    }
}
