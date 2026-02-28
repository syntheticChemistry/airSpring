// SPDX-License-Identifier: AGPL-3.0-or-later
//! Solar geometry and radiation (FAO-56).
//!
//! Inverse relative distance, declination, sunset hour angle,
//! extraterrestrial radiation, daylight hours, and net radiation components.
//! Used by Penman-Monteith, Hargreaves, Hamon, and other ET₀ methods.

use std::f64::consts::PI;

/// Earth orbit eccentricity coefficient. FAO-56 Eq. 23.
const ECCENTRICITY_COEFF: f64 = 0.033;
/// Days in standard year.
const DAYS_PER_YEAR: f64 = 365.0;
/// Maximum solar declination (radians). FAO-56 Eq. 24.
const MAX_DECLINATION: f64 = 0.409;
/// Solar declination phase offset (radians). FAO-56 Eq. 24.
const DECLINATION_PHASE: f64 = 1.39;
/// Solar constant Gsc (MJ/m²/min). FAO-56 Table 2.7.
const SOLAR_CONSTANT_MJ: f64 = 0.0820;
/// Stefan-Boltzmann constant (MJ/m²/day/K⁴).
const STEFAN_BOLTZMANN: f64 = 4.903e-9;
/// Clear-sky elevation coefficient (per metre). FAO-56 Eq. 37.
const CLEAR_SKY_ELEV_COEFF: f64 = 2.0e-5;
/// Clear-sky base transmissivity. FAO-56 Eq. 37.
const CLEAR_SKY_BASE: f64 = 0.75;
/// Net longwave humidity factor coefficient. FAO-56 Eq. 39.
const LW_HUMIDITY_COEFF: f64 = 0.14;

/// Inverse relative distance Earth–Sun (FAO-56 Eq. 23).
///
/// dr = 1 + 0.033 × cos(2π/365 × J)
#[must_use]
pub fn inverse_rel_distance(day_of_year: u32) -> f64 {
    ECCENTRICITY_COEFF.mul_add((2.0 * PI * f64::from(day_of_year) / DAYS_PER_YEAR).cos(), 1.0)
}

/// Solar declination δ (radians) (FAO-56 Eq. 24).
///
/// δ = 0.409 × sin(2π/365 × J − 1.39)
#[must_use]
pub fn solar_declination(day_of_year: u32) -> f64 {
    MAX_DECLINATION * (2.0 * PI * f64::from(day_of_year) / DAYS_PER_YEAR - DECLINATION_PHASE).sin()
}

/// Sunset hour angle ωs (radians) (FAO-56 Eq. 25).
///
/// ωs = arccos(−tan(φ) × tan(δ))
#[must_use]
pub fn sunset_hour_angle(latitude_rad: f64, declination_rad: f64) -> f64 {
    (-latitude_rad.tan() * declination_rad.tan())
        .clamp(-1.0, 1.0)
        .acos()
}

/// Extraterrestrial radiation Ra (MJ/m²/day) (FAO-56 Eq. 21).
#[must_use]
pub fn extraterrestrial_radiation(latitude_rad: f64, day_of_year: u32) -> f64 {
    let dr = inverse_rel_distance(day_of_year);
    let delta = solar_declination(day_of_year);
    let ws = sunset_hour_angle(latitude_rad, delta);

    (24.0 * 60.0 / PI)
        * SOLAR_CONSTANT_MJ
        * dr
        * (ws * latitude_rad.sin())
            .mul_add(delta.sin(), latitude_rad.cos() * delta.cos() * ws.sin())
}

/// Daylight hours N (FAO-56 Eq. 34).
///
/// N = 24/π × ωs
#[must_use]
pub fn daylight_hours(latitude_rad: f64, day_of_year: u32) -> f64 {
    let delta = solar_declination(day_of_year);
    let ws = sunset_hour_angle(latitude_rad, delta);
    24.0 / PI * ws
}

/// Clear-sky solar radiation Rso (MJ/m²/day) (FAO-56 Eq. 37).
///
/// Rso = (0.75 + 2 × 10⁻⁵ × z) × Ra
#[must_use]
pub fn clear_sky_radiation(elevation_m: f64, ra: f64) -> f64 {
    CLEAR_SKY_ELEV_COEFF.mul_add(elevation_m, CLEAR_SKY_BASE) * ra
}

/// Net shortwave radiation Rns (MJ/m²/day) (FAO-56 Eq. 38).
///
/// Rns = (1 − α) × Rs, where α = 0.23 for hypothetical grass reference.
#[must_use]
pub fn net_shortwave_radiation(rs: f64, albedo: f64) -> f64 {
    (1.0 - albedo) * rs
}

/// Net longwave radiation Rnl (MJ/m²/day) (FAO-56 Eq. 39).
#[must_use]
pub fn net_longwave_radiation(tmin: f64, tmax: f64, ea: f64, rs: f64, rso: f64) -> f64 {
    let tk_min = tmin + 273.16;
    let tk_max = tmax + 273.16;
    let avg_tk4 = f64::midpoint(tk_max.powi(4), tk_min.powi(4));
    let humidity_factor = LW_HUMIDITY_COEFF.mul_add(-ea.sqrt(), 0.34);
    let cloudiness_factor = if rso > 0.0 {
        1.35f64.mul_add((rs / rso).min(1.0), -0.35).max(0.05)
    } else {
        0.05
    };
    STEFAN_BOLTZMANN * avg_tk4 * humidity_factor * cloudiness_factor
}

/// Net radiation Rn (MJ/m²/day) (FAO-56 Eq. 40).
///
/// Rn = Rns − Rnl
#[must_use]
pub fn net_radiation(rns: f64, rnl: f64) -> f64 {
    rns - rnl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraterrestrial_radiation() {
        // FAO-56 Example 8: Sep 3 (DOY 246), lat −22.9°, Ra ≈ 32.2 MJ/m²/day
        let lat_rad = (-22.9_f64).to_radians();
        let ra = extraterrestrial_radiation(lat_rad, 246);
        assert!((ra - 32.2).abs() < 1.5, "Ra: {ra}");
    }

    #[test]
    fn test_daylight_hours() {
        // FAO-56 Example 9: Sep 3 (DOY 246), lat −22.9°, N ≈ 11.7 h
        let lat_rad = (-22.9_f64).to_radians();
        let n = daylight_hours(lat_rad, 246);
        assert!((n - 11.7).abs() < 0.2, "N: {n}");
    }

    #[test]
    fn test_inverse_rel_distance() {
        // DOY 1 (Jan 1): Earth closest to Sun, dr > 1
        let dr_jan = inverse_rel_distance(1);
        assert!(dr_jan > 1.0, "dr at DOY 1: {dr_jan}");
        // DOY 182 (Jul 1): Earth farthest, dr < 1
        let dr_jul = inverse_rel_distance(182);
        assert!(dr_jul < 1.0, "dr at DOY 182: {dr_jul}");
    }

    #[test]
    fn test_solar_declination() {
        // Summer solstice (DOY ~172): δ ≈ +0.409 rad (max)
        let delta_summer = solar_declination(172);
        assert!(delta_summer > 0.3, "δ summer: {delta_summer}");
        // Winter solstice (DOY ~355): δ ≈ −0.409 rad (min)
        let delta_winter = solar_declination(355);
        assert!(delta_winter < -0.3, "δ winter: {delta_winter}");
    }

    #[test]
    fn test_net_shortwave_radiation() {
        // Rs = 20 MJ/m²/day, albedo = 0.23 → Rns = 15.4
        let rns = net_shortwave_radiation(20.0, 0.23);
        assert!((rns - 15.4).abs() < 0.01);
    }

    #[test]
    fn test_net_radiation_identity() {
        // Rn = Rns − Rnl
        let rns = 17.0;
        let rnl = 3.5;
        assert!((net_radiation(rns, rnl) - 13.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_sky_radiation() {
        // Rso = (0.75 + 2e-5 × 100) × 40 = 0.752 × 40 = 30.08
        let rso = clear_sky_radiation(100.0, 40.0);
        assert!((rso - 30.08).abs() < 0.01, "Rso: {rso}");
    }
}
