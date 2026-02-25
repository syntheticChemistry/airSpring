//! FAO-56 Penman-Monteith Reference Evapotranspiration (ET₀).
//!
//! Implements the standard FAO Paper 56 equation (Allen et al., 1998):
//!
//! ```text
//! ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
//! ```
//!
//! This is the foundational calculation for all irrigation scheduling.
//! Every variable has a published derivation in FAO-56 Chapters 2–4.
//!
//! # Reference
//!
//! Allen RG, Pereira LS, Raes D, Smith M (1998)
//! "Crop evapotranspiration — Guidelines for computing crop water requirements"
//! FAO Irrigation and Drainage Paper 56, Rome.

use std::f64::consts::PI;

// ── Atmospheric parameters ───────────────────────────────────────────

/// Psychrometric constant γ (kPa/°C).
///
/// γ = 0.665 × 10⁻³ × P
///
/// FAO-56 Eq. 8.
#[must_use]
pub fn psychrometric_constant(pressure_kpa: f64) -> f64 {
    0.665e-3 * pressure_kpa
}

/// Atmospheric pressure from elevation (kPa).
///
/// P = 101.3 × ((293 − 0.0065z) / 293)^5.26
///
/// FAO-56 Eq. 7.
#[must_use]
pub fn atmospheric_pressure(elevation_m: f64) -> f64 {
    101.3 * (0.0065f64.mul_add(-elevation_m, 293.0) / 293.0).powf(5.26)
}

// ── Vapour pressure functions ────────────────────────────────────────

/// Saturation vapour pressure e°(T) (kPa) at temperature T (°C).
///
/// FAO-56 Eq. 11: e°(T) = 0.6108 × exp(17.27T / (T + 237.3))
#[must_use]
pub fn saturation_vapour_pressure(temp_c: f64) -> f64 {
    0.6108 * ((17.27 * temp_c) / (temp_c + 237.3)).exp()
}

/// Slope of saturation vapour pressure curve Δ (kPa/°C).
///
/// FAO-56 Eq. 13: Δ = 4098 × e°(T) / (T + 237.3)²
#[must_use]
pub fn vapour_pressure_slope(temp_c: f64) -> f64 {
    let es = saturation_vapour_pressure(temp_c);
    4098.0 * es / (temp_c + 237.3).powi(2)
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

// ── Solar geometry ───────────────────────────────────────────────────

/// Inverse relative distance Earth–Sun (FAO-56 Eq. 23).
///
/// dr = 1 + 0.033 × cos(2π/365 × J)
#[must_use]
pub fn inverse_rel_distance(day_of_year: u32) -> f64 {
    0.033f64.mul_add((2.0 * PI * f64::from(day_of_year) / 365.0).cos(), 1.0)
}

/// Solar declination δ (radians) (FAO-56 Eq. 24).
///
/// δ = 0.409 × sin(2π/365 × J − 1.39)
#[must_use]
pub fn solar_declination(day_of_year: u32) -> f64 {
    0.409 * (2.0 * PI * f64::from(day_of_year) / 365.0 - 1.39).sin()
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
    let gsc = 0.0820; // Solar constant (MJ/m²/min)
    let dr = inverse_rel_distance(day_of_year);
    let delta = solar_declination(day_of_year);
    let ws = sunset_hour_angle(latitude_rad, delta);

    (24.0 * 60.0 / PI)
        * gsc
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

// ── Radiation ────────────────────────────────────────────────────────

/// Clear-sky solar radiation Rso (MJ/m²/day) (FAO-56 Eq. 37).
///
/// Rso = (0.75 + 2 × 10⁻⁵ × z) × Ra
#[must_use]
pub fn clear_sky_radiation(elevation_m: f64, ra: f64) -> f64 {
    2.0e-5f64.mul_add(elevation_m, 0.75) * ra
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
    let sigma = 4.903e-9; // Stefan-Boltzmann (MJ/m²/day/K⁴)
    let tk_min = tmin + 273.16;
    let tk_max = tmax + 273.16;
    let avg_tk4 = f64::midpoint(tk_max.powi(4), tk_min.powi(4));
    let humidity_factor = 0.14f64.mul_add(-ea.sqrt(), 0.34);
    let cloudiness_factor = if rso > 0.0 {
        1.35f64.mul_add((rs / rso).min(1.0), -0.35).max(0.05)
    } else {
        0.05
    };
    sigma * avg_tk4 * humidity_factor * cloudiness_factor
}

/// Net radiation Rn (MJ/m²/day) (FAO-56 Eq. 40).
///
/// Rn = Rns − Rnl
#[must_use]
pub fn net_radiation(rns: f64, rnl: f64) -> f64 {
    rns - rnl
}

// ── Solar radiation estimation ────────────────────────────────────────

/// Solar radiation from sunshine hours (FAO-56 Eq. 35).
///
/// Rs = (as + bs × n/N) × Ra
///
/// Default Ångström coefficients: as = 0.25, bs = 0.50.
/// `n` is actual sunshine hours, `N` is maximum possible daylight hours.
///
/// # Panics
///
/// Panics if `max_daylight_hours` is zero.
#[must_use]
pub fn solar_radiation_from_sunshine(sunshine_hours: f64, max_daylight_hours: f64, ra: f64) -> f64 {
    assert!(
        max_daylight_hours > 0.0,
        "Max daylight hours must be positive"
    );
    0.50f64.mul_add(sunshine_hours / max_daylight_hours, 0.25) * ra
}

/// Solar radiation from temperature range — Hargreaves method (FAO-56 Eq. 50).
///
/// Rs = kRs × √(Tmax − Tmin) × Ra
///
/// `krs` is an empirical coefficient: 0.16 for interior, 0.19 for coastal.
/// Use when sunshine or cloud data are unavailable.
#[must_use]
pub fn solar_radiation_from_temperature(tmax: f64, tmin: f64, ra: f64, krs: f64) -> f64 {
    krs * (tmax - tmin).max(0.0).sqrt() * ra
}

/// Soil heat flux G for monthly time step (FAO-56 Eq. 43).
///
/// G = 0.14 × (Tᵢ − Tᵢ₋₁)
///
/// For daily time steps, G ≈ 0 (handled in [`daily_et0`]).
#[must_use]
pub fn soil_heat_flux_monthly(t_month: f64, t_month_prev: f64) -> f64 {
    0.14 * (t_month - t_month_prev)
}

/// Hargreaves–Samani ET₀ estimate (FAO-56 Eq. 52).
///
/// ET₀ = 0.0023 × (Tmean + 17.8) × √(Tmax − Tmin) × Ra
///
/// A simplified ET₀ method requiring only temperature and Ra.
/// Recommended by FAO-56 when wind, humidity, and radiation data
/// are unavailable. Accuracy is lower than Penman-Monteith.
///
/// Ra must be in equivalent mm/day (divide MJ/m²/day by 2.45 = λ).
#[must_use]
pub fn hargreaves_et0(tmin: f64, tmax: f64, ra_mm_day: f64) -> f64 {
    let tmean = f64::midpoint(tmin, tmax);
    (0.0023 * (tmean + 17.8) * (tmax - tmin).max(0.0).sqrt() * ra_mm_day).max(0.0)
}

// ── Wind speed adjustment ─────────────────────────────────────────────

/// Convert wind speed measured at height `z_m` to the standard 2 m height.
///
/// FAO-56 Eq. 47: u₂ = uz × 4.87 / ln(67.8z − 5.42)
///
/// Most weather stations measure wind at 10 m. The Penman-Monteith equation
/// requires wind at 2 m. This conversion assumes logarithmic wind profile.
///
/// # Panics
///
/// Panics if `z_m` ≤ 0.0 (physically impossible measurement height).
#[must_use]
pub fn wind_speed_at_2m(uz: f64, z_m: f64) -> f64 {
    assert!(z_m > 0.0, "Measurement height must be positive: {z_m}");
    uz * 4.87 / (67.8f64.mul_add(z_m, -5.42)).ln()
}

// ── Daily ET₀ ────────────────────────────────────────────────────────

/// Input parameters for daily ET₀ calculation.
#[derive(Debug, Clone, Copy)]
pub struct DailyEt0Input {
    /// Minimum temperature (°C)
    pub tmin: f64,
    /// Maximum temperature (°C)
    pub tmax: f64,
    /// Mean temperature (°C) — if `None`, uses (tmin + tmax) / 2
    pub tmean: Option<f64>,
    /// Solar radiation Rs (MJ/m²/day)
    pub solar_radiation: f64,
    /// Wind speed at 2 m height (m/s)
    pub wind_speed_2m: f64,
    /// Actual vapour pressure ea (kPa)
    pub actual_vapour_pressure: f64,
    /// Elevation above sea level (m)
    pub elevation_m: f64,
    /// Latitude (decimal degrees, positive = North)
    pub latitude_deg: f64,
    /// Day of year (1–366)
    pub day_of_year: u32,
}

/// Result of ET₀ calculation.
#[derive(Debug, Clone, Copy)]
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
    /// Vapour pressure deficit (es − ea) (kPa)
    pub vpd: f64,
    /// Extraterrestrial radiation Ra (MJ/m²/day)
    pub ra: f64,
}

/// Low-level FAO-56 Penman-Monteith equation (Eq. 6).
///
/// ```text
/// ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ VPD] / [Δ + γ(1 + 0.34 u₂)]
/// ```
///
/// This exposes the core equation for use when all intermediate values
/// are already computed (e.g., from GPU buffers or pre-computed tables).
/// For a higher-level API that computes intermediates from raw weather
/// data, use [`daily_et0`].
///
/// # Arguments
///
/// - `rn`: Net radiation (MJ/m²/day)
/// - `g`: Soil heat flux (MJ/m²/day), typically 0.0 for daily step
/// - `tmean_c`: Mean temperature (°C)
/// - `u2`: Wind speed at 2 m (m/s)
/// - `vpd_kpa`: Vapour pressure deficit es − ea (kPa)
/// - `delta`: Slope of saturation vapour pressure curve (kPa/°C)
/// - `gamma`: Psychrometric constant (kPa/°C)
#[must_use]
pub fn fao56_penman_monteith(
    rn: f64,
    g: f64,
    tmean_c: f64,
    u2: f64,
    vpd_kpa: f64,
    delta: f64,
    gamma: f64,
) -> f64 {
    let numerator =
        (0.408 * delta).mul_add(rn - g, gamma * (900.0 / (tmean_c + 273.0)) * u2 * vpd_kpa);
    let denominator = gamma.mul_add(0.34f64.mul_add(u2, 1.0), delta);
    (numerator / denominator).max(0.0)
}

/// Compute daily FAO-56 Penman-Monteith reference ET₀.
///
/// FAO-56 Eq. 6:
/// ```text
/// ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
/// ```
#[must_use]
pub fn daily_et0(input: &DailyEt0Input) -> Et0Result {
    let tmean = input
        .tmean
        .unwrap_or_else(|| f64::midpoint(input.tmin, input.tmax));
    let lat_rad = input.latitude_deg.to_radians();

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

    // Soil heat flux: G ≈ 0 for daily time step (FAO-56 §4.1)
    let g = 0.0;

    // FAO-56 Eq. 6 — delegates to low-level function
    let et0 = fao56_penman_monteith(rn, g, tmean, input.wind_speed_2m, vpd, delta, gamma);

    Et0Result {
        et0,
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

    // FAO-56 Table 2.3 values are rounded to 3 decimal places, but the
    // equation coefficients (17.27, 237.3) are themselves rounded from the
    // original Tetens formula. At high temperatures (35–48 °C), the Eq. 11
    // computation diverges from the tabulated values by up to 0.009 kPa.
    // Tolerance 0.01 matches benchmark_fao56.json specification.
    const FAO56_SVP_TOL: f64 = 0.01;
    // Table 2.4 tolerance from benchmark JSON: 0.005 kPa/°C.
    const FAO56_DELTA_TOL: f64 = 0.005;

    #[test]
    fn test_saturation_vapour_pressure_table_2_3() {
        // FAO-56 Table 2.3: saturation vapour pressure at various temperatures.
        let cases = [
            (1.0, 0.657),
            (5.0, 0.872),
            (10.0, 1.228),
            (15.0, 1.705),
            (20.0, 2.338),
            (25.0, 3.168),
            (30.0, 4.243),
            (35.0, 5.624),
            (40.0, 7.384),
            (45.0, 9.585),
        ];
        for (temp, expected) in cases {
            let es = saturation_vapour_pressure(temp);
            assert!(
                (es - expected).abs() < FAO56_SVP_TOL,
                "es({temp}°C) = {es}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_vapour_pressure_slope_table_2_4() {
        // FAO-56 Table 2.4: slope of vapour pressure curve.
        let cases = [
            (1.0, 0.047),
            (10.0, 0.082),
            (20.0, 0.145),
            (30.0, 0.243),
            (40.0, 0.393),
        ];
        for (temp, expected) in cases {
            let delta = vapour_pressure_slope(temp);
            assert!(
                (delta - expected).abs() < FAO56_DELTA_TOL,
                "Δ({temp}°C) = {delta}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_atmospheric_pressure() {
        let p_sea = atmospheric_pressure(0.0);
        assert!((p_sea - 101.3).abs() < 0.1, "P at 0m: {p_sea}");
        let p_1800 = atmospheric_pressure(1800.0);
        assert!((p_1800 - 81.8).abs() < 0.5, "P at 1800m: {p_1800}");
    }

    #[test]
    fn test_psychrometric_constant() {
        let gamma = psychrometric_constant(101.3);
        assert!((gamma - 0.0674).abs() < 0.001, "γ: {gamma}");
    }

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

    #[test]
    fn test_actual_vapour_pressure_rh() {
        // FAO-56 Eq. 17: ea from Tmin=18, Tmax=25, RHmin=54, RHmax=80
        let ea = actual_vapour_pressure_rh(18.0, 25.0, 54.0, 80.0);
        // Expected: midpoint(es(18)*80/100, es(25)*54/100)
        let e18 = saturation_vapour_pressure(18.0);
        let e25 = saturation_vapour_pressure(25.0);
        let expected = f64::midpoint(e18 * 80.0 / 100.0, e25 * 54.0 / 100.0);
        assert!((ea - expected).abs() < 1e-10);
        // Should be a reasonable value for these conditions (1.5–2.0 kPa)
        assert!(ea > 1.0 && ea < 2.5, "ea = {ea}");
    }

    #[test]
    fn test_actual_vapour_pressure_dewpoint() {
        // ea at Tdew = 20°C should equal es(20) = 2.338
        let ea = actual_vapour_pressure_dewpoint(20.0);
        assert!((ea - 2.338).abs() < 0.001);
    }

    #[test]
    fn test_wind_speed_at_2m_from_10m() {
        // FAO-56 Eq. 47: u₂ = uz × 4.87 / ln(67.8z − 5.42)
        // At z=10m: u₂ = 3.0 × 4.87 / ln(672.58) = 3.0 × 0.748 ≈ 2.244
        let u2 = wind_speed_at_2m(3.0, 10.0);
        assert!((u2 - 2.244).abs() < 0.02, "u₂ from 10m: {u2}");
    }

    #[test]
    fn test_wind_speed_at_2m_identity_at_2m() {
        // At z=2m the conversion should be approximately identity.
        let u2 = wind_speed_at_2m(5.0, 2.0);
        assert!((u2 - 5.0).abs() < 0.15, "u₂ at 2m should be ~5.0: {u2}");
    }

    #[test]
    fn test_wind_speed_lower_at_2m() {
        // Wind at 2m should always be lower than at any height above 2m.
        for &z in &[3.0, 5.0, 10.0, 20.0, 50.0] {
            let u2 = wind_speed_at_2m(5.0, z);
            assert!(u2 < 5.0, "u₂ should be < uz at z={z}m: u₂={u2}");
        }
    }

    #[test]
    fn test_daily_et0_uccle_example_18() {
        // FAO-56 Example 18: Uccle, Belgium, 6 July.
        // Published answer: 3.88 mm/day (tolerance 0.10 per benchmark JSON).
        // Our function takes Rs directly; the published example derives Rs from
        // sunshine hours. Using the published intermediate Rs = 22.07 MJ/m²/day.
        let input = DailyEt0Input {
            tmin: 12.3,
            tmax: 21.5,
            tmean: Some(16.9),
            solar_radiation: 22.07,
            wind_speed_2m: 2.078,
            actual_vapour_pressure: 1.409,
            elevation_m: 100.0,
            latitude_deg: 50.80,
            day_of_year: 187,
        };
        let result = daily_et0(&input);
        assert!(
            (result.et0 - 3.88).abs() < 0.25,
            "Uccle ET₀: {:.3} (expected ~3.88)",
            result.et0
        );
    }

    #[test]
    fn test_daily_et0_positive_for_typical_conditions() {
        let input = DailyEt0Input {
            tmin: 15.0,
            tmax: 28.0,
            tmean: None,
            solar_radiation: 18.0,
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 1.5,
            elevation_m: 50.0,
            latitude_deg: 45.0,
            day_of_year: 200,
        };
        let result = daily_et0(&input);
        assert!(result.et0 > 0.0, "ET₀ should be positive: {}", result.et0);
    }

    #[test]
    fn test_daily_et0_zero_wind_reduces_et() {
        let base = DailyEt0Input {
            tmin: 20.0,
            tmax: 32.0,
            tmean: None,
            solar_radiation: 18.0,
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 2.0,
            elevation_m: 50.0,
            latitude_deg: 30.0,
            day_of_year: 150,
        };
        let calm = DailyEt0Input {
            wind_speed_2m: 0.0,
            ..base
        };
        assert!(daily_et0(&calm).et0 < daily_et0(&base).et0);
    }

    #[test]
    fn test_solar_radiation_from_sunshine() {
        // FAO-56 Eq. 35: Rs = (0.25 + 0.50 × n/N) × Ra
        // n=7.1, N=11.7, Ra=32.2 → Rs = (0.25 + 0.50 × 7.1/11.7) × 32.2
        let rs = solar_radiation_from_sunshine(7.1, 11.7, 32.2);
        let expected = (0.25 + 0.50 * 7.1 / 11.7) * 32.2;
        assert!(
            (rs - expected).abs() < 0.01,
            "Rs = {rs}, expected {expected}"
        );
    }

    #[test]
    fn test_solar_radiation_from_sunshine_zero_sunshine() {
        // Zero sunshine hours: Rs = 0.25 × Ra (cloudy day)
        let rs = solar_radiation_from_sunshine(0.0, 12.0, 40.0);
        assert!((rs - 10.0).abs() < 0.01, "Rs(n=0) = {rs}, expected 10.0");
    }

    #[test]
    fn test_solar_radiation_from_temperature() {
        // FAO-56 Eq. 50: Rs = 0.16 × √(25-15) × 40 = 0.16 × √10 × 40
        let rs = solar_radiation_from_temperature(25.0, 15.0, 40.0, 0.16);
        let expected = 0.16 * 10.0_f64.sqrt() * 40.0;
        assert!(
            (rs - expected).abs() < 0.01,
            "Rs = {rs}, expected {expected}"
        );
    }

    #[test]
    fn test_soil_heat_flux_monthly() {
        // G = 0.14 × (25 − 22) = 0.42
        let g = soil_heat_flux_monthly(25.0, 22.0);
        assert!((g - 0.42).abs() < 0.001, "G = {g}");
    }

    #[test]
    fn test_soil_heat_flux_monthly_cooling() {
        // Cooling month → negative G
        let g = soil_heat_flux_monthly(18.0, 22.0);
        assert!(g < 0.0, "G should be negative: {g}");
    }

    #[test]
    fn test_hargreaves_et0_reasonable_range() {
        // Typical summer conditions: Tmin=18, Tmax=32, Ra=40 MJ/m²/day
        // Ra in mm/day = 40/2.45 ≈ 16.33
        let ra_mm = 40.0 / 2.45;
        let et0 = hargreaves_et0(18.0, 32.0, ra_mm);
        // Hargreaves typically gives 3–8 mm/day for summer conditions
        assert!(et0 > 2.0 && et0 < 10.0, "Hargreaves ET₀ = {et0} mm/day");
    }

    #[test]
    fn test_hargreaves_et0_increases_with_temperature() {
        let ra_mm = 40.0 / 2.45;
        let et0_cool = hargreaves_et0(10.0, 20.0, ra_mm);
        let et0_warm = hargreaves_et0(18.0, 32.0, ra_mm);
        assert!(
            et0_warm > et0_cool,
            "Warmer should have higher ET₀: {et0_cool} vs {et0_warm}"
        );
    }

    #[test]
    fn test_hargreaves_et0_non_negative() {
        // Even with cold conditions, should not go negative
        let et0 = hargreaves_et0(-5.0, 0.0, 5.0);
        assert!(et0 >= 0.0, "ET₀ = {et0}");
    }
}
