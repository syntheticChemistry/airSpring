//! FAO-56 Penman-Monteith Reference Evapotranspiration (ET₀).
//!
//! Implements the standard FAO Paper 56 equation (Allen et al., 1998):
//!
//! ```text
//! ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
//! ```
//!
//! This is the foundational calculation for all irrigation scheduling.
//! Every variable has a published derivation in FAO-56 Chapters 2-4.
//!
//! Reference: Allen RG, Pereira LS, Raes D, Smith M (1998)
//! "Crop evapotranspiration — Guidelines for computing crop water requirements"
//! FAO Irrigation and Drainage Paper 56, Rome.

use std::f64::consts::PI;

/// Psychrometric constant γ (kPa/°C).
/// γ = 0.665e-3 * P, where P is atmospheric pressure (kPa).
pub fn psychrometric_constant(pressure_kpa: f64) -> f64 {
    0.665e-3 * pressure_kpa
}

/// Atmospheric pressure from elevation (FAO-56 Eq. 7).
/// P = 101.3 * ((293 - 0.0065*z) / 293)^5.26
pub fn atmospheric_pressure(elevation_m: f64) -> f64 {
    101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0).powf(5.26)
}

/// Saturation vapour pressure es (kPa) at temperature T (°C).
/// FAO-56 Eq. 11: e°(T) = 0.6108 * exp(17.27*T / (T+237.3))
pub fn saturation_vapour_pressure(temp_c: f64) -> f64 {
    0.6108 * ((17.27 * temp_c) / (temp_c + 237.3)).exp()
}

/// Slope of saturation vapour pressure curve Δ (kPa/°C).
/// FAO-56 Eq. 13: Δ = 4098 * e°(T) / (T+237.3)²
pub fn vapour_pressure_slope(temp_c: f64) -> f64 {
    let es = saturation_vapour_pressure(temp_c);
    4098.0 * es / (temp_c + 237.3).powi(2)
}

/// Mean saturation vapour pressure from Tmin and Tmax.
/// FAO-56 Eq. 12: es = [e°(Tmax) + e°(Tmin)] / 2
pub fn mean_saturation_vapour_pressure(tmin: f64, tmax: f64) -> f64 {
    (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin)) / 2.0
}

/// Actual vapour pressure from dewpoint temperature.
/// FAO-56 Eq. 14: ea = e°(Tdew)
pub fn actual_vapour_pressure_dewpoint(tdew: f64) -> f64 {
    saturation_vapour_pressure(tdew)
}

/// Actual vapour pressure from relative humidity.
/// FAO-56 Eq. 17: ea = [e°(Tmin)*RHmax + e°(Tmax)*RHmin] / 200
pub fn actual_vapour_pressure_rh(tmin: f64, tmax: f64, rh_min: f64, rh_max: f64) -> f64 {
    let e_tmin = saturation_vapour_pressure(tmin);
    let e_tmax = saturation_vapour_pressure(tmax);
    (e_tmin * rh_max / 100.0 + e_tmax * rh_min / 100.0) / 2.0
}

/// Inverse relative distance Earth-Sun (FAO-56 Eq. 23).
/// dr = 1 + 0.033 * cos(2π/365 * J)
pub fn inverse_rel_distance(day_of_year: u32) -> f64 {
    1.0 + 0.033 * (2.0 * PI * day_of_year as f64 / 365.0).cos()
}

/// Solar declination (radians) (FAO-56 Eq. 24).
/// δ = 0.409 * sin(2π/365 * J - 1.39)
pub fn solar_declination(day_of_year: u32) -> f64 {
    0.409 * (2.0 * PI * day_of_year as f64 / 365.0 - 1.39).sin()
}

/// Sunset hour angle (radians) (FAO-56 Eq. 25).
/// ωs = arccos(-tan(φ) * tan(δ))
pub fn sunset_hour_angle(latitude_rad: f64, declination_rad: f64) -> f64 {
    (-latitude_rad.tan() * declination_rad.tan())
        .clamp(-1.0, 1.0)
        .acos()
}

/// Extraterrestrial radiation Ra (MJ/m²/day) (FAO-56 Eq. 21).
pub fn extraterrestrial_radiation(latitude_rad: f64, day_of_year: u32) -> f64 {
    let gsc = 0.0820; // Solar constant (MJ/m²/min)
    let dr = inverse_rel_distance(day_of_year);
    let delta = solar_declination(day_of_year);
    let ws = sunset_hour_angle(latitude_rad, delta);

    (24.0 * 60.0 / PI) * gsc * dr
        * (ws * latitude_rad.sin() * delta.sin()
            + latitude_rad.cos() * delta.cos() * ws.sin())
}

/// Daylight hours N (FAO-56 Eq. 34).
/// N = 24/π * ωs
pub fn daylight_hours(latitude_rad: f64, day_of_year: u32) -> f64 {
    let delta = solar_declination(day_of_year);
    let ws = sunset_hour_angle(latitude_rad, delta);
    24.0 / PI * ws
}

/// Clear-sky solar radiation Rso (MJ/m²/day) (FAO-56 Eq. 37).
/// Rso = (0.75 + 2e-5 * z) * Ra
pub fn clear_sky_radiation(elevation_m: f64, ra: f64) -> f64 {
    (0.75 + 2.0e-5 * elevation_m) * ra
}

/// Net shortwave radiation Rns (MJ/m²/day) (FAO-56 Eq. 38).
/// Rns = (1 - α) * Rs, where α = 0.23 for hypothetical grass reference.
pub fn net_shortwave_radiation(rs: f64, albedo: f64) -> f64 {
    (1.0 - albedo) * rs
}

/// Net longwave radiation Rnl (MJ/m²/day) (FAO-56 Eq. 39).
pub fn net_longwave_radiation(tmin: f64, tmax: f64, ea: f64, rs: f64, rso: f64) -> f64 {
    let sigma = 4.903e-9; // Stefan-Boltzmann (MJ/m²/day/K⁴)
    let tk_min = tmin + 273.16;
    let tk_max = tmax + 273.16;
    let avg_tk4 = (tk_max.powi(4) + tk_min.powi(4)) / 2.0;
    let humidity_factor = 0.34 - 0.14 * ea.sqrt();
    let cloudiness_factor = if rso > 0.0 {
        (1.35 * (rs / rso).min(1.0) - 0.35).max(0.05)
    } else {
        0.05
    };
    sigma * avg_tk4 * humidity_factor * cloudiness_factor
}

/// Net radiation Rn (MJ/m²/day) (FAO-56 Eq. 40).
/// Rn = Rns - Rnl
pub fn net_radiation(rns: f64, rnl: f64) -> f64 {
    rns - rnl
}

/// Input parameters for daily ET₀ calculation.
#[derive(Debug, Clone)]
pub struct DailyEt0Input {
    /// Minimum temperature (°C)
    pub tmin: f64,
    /// Maximum temperature (°C)
    pub tmax: f64,
    /// Mean temperature (°C) — if None, uses (tmin+tmax)/2
    pub tmean: Option<f64>,
    /// Solar radiation Rs (MJ/m²/day)
    pub solar_radiation: f64,
    /// Wind speed at 2m height (m/s)
    pub wind_speed_2m: f64,
    /// Actual vapour pressure ea (kPa)
    pub actual_vapour_pressure: f64,
    /// Elevation above sea level (m)
    pub elevation_m: f64,
    /// Latitude (decimal degrees, positive = North)
    pub latitude_deg: f64,
    /// Day of year (1-366)
    pub day_of_year: u32,
}

/// Result of ET₀ calculation.
#[derive(Debug, Clone)]
pub struct Et0Result {
    /// Reference evapotranspiration (mm/day)
    pub et0: f64,
    /// Net radiation Rn (MJ/m²/day)
    pub rn: f64,
    /// Soil heat flux G (MJ/m²/day) — assumed 0 for daily
    pub g: f64,
    /// Slope of vapour pressure curve Δ (kPa/°C)
    pub delta: f64,
    /// Psychrometric constant γ (kPa/°C)
    pub gamma: f64,
    /// Saturation vapour pressure es (kPa)
    pub es: f64,
    /// Vapour pressure deficit (es - ea) (kPa)
    pub vpd: f64,
    /// Extraterrestrial radiation Ra (MJ/m²/day)
    pub ra: f64,
}

/// Compute daily FAO-56 Penman-Monteith reference ET₀.
///
/// FAO-56 Eq. 6:
/// ```text
/// ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
/// ```
pub fn daily_et0(input: &DailyEt0Input) -> Et0Result {
    let tmean = input.tmean.unwrap_or((input.tmin + input.tmax) / 2.0);
    let lat_rad = input.latitude_deg * PI / 180.0;

    // Atmospheric parameters
    let pressure = atmospheric_pressure(input.elevation_m);
    let gamma = psychrometric_constant(pressure);
    let delta = vapour_pressure_slope(tmean);

    // Vapour pressures
    let es = mean_saturation_vapour_pressure(input.tmin, input.tmax);
    let ea = input.actual_vapour_pressure;
    let vpd = es - ea;

    // Radiation
    let ra = extraterrestrial_radiation(lat_rad, input.day_of_year);
    let rso = clear_sky_radiation(input.elevation_m, ra);
    let rns = net_shortwave_radiation(input.solar_radiation, 0.23);
    let rnl = net_longwave_radiation(input.tmin, input.tmax, ea, input.solar_radiation, rso);
    let rn = net_radiation(rns, rnl);

    // Soil heat flux: G ≈ 0 for daily time step (FAO-56)
    let g = 0.0;

    // FAO-56 Eq. 6
    let u2 = input.wind_speed_2m;
    let numerator = 0.408 * delta * (rn - g)
        + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd;
    let denominator = delta + gamma * (1.0 + 0.34 * u2);
    let et0 = numerator / denominator;

    Et0Result {
        et0: et0.max(0.0),
        rn,
        g,
        delta,
        gamma,
        es,
        vpd,
        ra,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturation_vapour_pressure() {
        // FAO-56 Table 2.3: at 20°C, es = 2.338 kPa
        let es = saturation_vapour_pressure(20.0);
        assert!((es - 2.338).abs() < 0.001, "es at 20°C: {}", es);
    }

    #[test]
    fn test_vapour_pressure_slope() {
        // FAO-56 Table 2.4: at 20°C, Δ = 0.1447 kPa/°C
        let delta = vapour_pressure_slope(20.0);
        assert!((delta - 0.1447).abs() < 0.001, "Δ at 20°C: {}", delta);
    }

    #[test]
    fn test_atmospheric_pressure() {
        // FAO-56 Example: sea level → 101.3 kPa
        let p = atmospheric_pressure(0.0);
        assert!((p - 101.3).abs() < 0.1, "P at 0m: {}", p);
        // At 1800m → ~81.8 kPa
        let p1800 = atmospheric_pressure(1800.0);
        assert!((p1800 - 81.8).abs() < 0.5, "P at 1800m: {}", p1800);
    }

    #[test]
    fn test_psychrometric_constant() {
        // FAO-56 Example: at 101.3 kPa, γ = 0.0674 kPa/°C
        let gamma = psychrometric_constant(101.3);
        assert!((gamma - 0.0674).abs() < 0.001, "γ: {}", gamma);
    }

    #[test]
    fn test_extraterrestrial_radiation() {
        // FAO-56 Example 8: Sep 3 (DOY=246), lat=-22.9°
        // Ra = 32.2 MJ/m²/day
        let lat_rad = -22.9 * PI / 180.0;
        let ra = extraterrestrial_radiation(lat_rad, 246);
        assert!((ra - 32.2).abs() < 1.5, "Ra: {}", ra);
    }

    #[test]
    fn test_daylight_hours() {
        // FAO-56 Example 9: Sep 3 (DOY=246), lat=-22.9°
        // N = 11.7 hours
        let lat_rad = -22.9 * PI / 180.0;
        let n = daylight_hours(lat_rad, 246);
        assert!((n - 11.7).abs() < 0.2, "N: {}", n);
    }
}
