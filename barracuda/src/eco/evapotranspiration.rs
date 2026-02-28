// SPDX-License-Identifier: AGPL-3.0-or-later
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

// Re-export solar geometry and radiation for backward compatibility.
pub use super::et0_ensemble::{et0_ensemble, EnsembleInput, EnsembleResult};
pub use super::solar::{
    clear_sky_radiation, daylight_hours, extraterrestrial_radiation, inverse_rel_distance,
    net_longwave_radiation, net_radiation, net_shortwave_radiation, solar_declination,
    sunset_hour_angle,
};

// ── FAO-56 physical constants ────────────────────────────────────────

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

/// Default Ångström coefficient as. FAO-56 Eq. 35.
const ANGSTROM_AS: f64 = 0.25;
/// Default Ångström coefficient bs. FAO-56 Eq. 35.
const ANGSTROM_BS: f64 = 0.50;

/// Monthly soil heat flux coefficient (MJ/m²/month per °C). FAO-56 Eq. 43.
const SOIL_HEAT_FLUX_COEFF: f64 = 0.14;

/// Hargreaves empirical coefficient. FAO-56 Eq. 52.
const HARGREAVES_COEFF: f64 = 0.0023;
/// Hargreaves temperature offset (°C). FAO-56 Eq. 52.
const HARGREAVES_TEMP_OFFSET: f64 = 17.8;

/// Latent heat conversion: MJ/m²/day → mm/day. FAO-56 (1/λ at 20°C).
const MJ_TO_MM: f64 = 0.408;

/// Hamon saturation density coefficient (g/m³ per kPa/K). Lu et al. (2005).
const HAMON_RHO_COEFF: f64 = 216.7;
/// Hamon absolute temperature offset (K). Lu et al. (2005).
const HAMON_TEMP_OFFSET_K: f64 = 273.3;
/// Hamon PET coefficient. Lu et al. (2005).
const HAMON_PET_COEFF: f64 = 0.1651;

/// Blaney-Criddle annual daylight hours (approx 4380 hrs / 100). FAO-24.
const BC_ANNUAL_DAYLIGHT: f64 = 43.80;
/// Blaney-Criddle temperature coefficient. USDA-SCS (1950).
const BC_TEMP_COEFF: f64 = 0.46;
/// Blaney-Criddle offset constant. USDA-SCS (1950).
const BC_OFFSET: f64 = 8.13;

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
    SEA_LEVEL_PRESSURE_KPA * (LAPSE_RATE.mul_add(-elevation_m, BASE_TEMP_K) / BASE_TEMP_K).powf(PRESSURE_EXPONENT)
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
    ANGSTROM_BS.mul_add(sunshine_hours / max_daylight_hours, ANGSTROM_AS) * ra
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
    SOIL_HEAT_FLUX_COEFF * (t_month - t_month_prev)
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
///
/// # Upstream note (`ToadStool` S66)
///
/// `barracuda::stats::hydrology::hargreaves_et0(ra, t_max, t_min)` provides
/// an equivalent (R-S66-002, absorbed from airSpring metalForge). This local
/// version uses FAO-56 `(tmin, tmax, ra)` parameter order (matching the
/// equation's written form: temperature terms first, radiation last). The
/// upstream version uses `(ra, tmax, tmin)` for consistency with its batch
/// API. Both produce identical results. This local version is retained for
/// validation binary compatibility and FAO-56 code-review legibility.
#[must_use]
pub fn hargreaves_et0(tmin: f64, tmax: f64, ra_mm_day: f64) -> f64 {
    let tmean = f64::midpoint(tmin, tmax);
    (HARGREAVES_COEFF * (tmean + HARGREAVES_TEMP_OFFSET) * (tmax - tmin).max(0.0).sqrt() * ra_mm_day).max(0.0)
}

/// Priestley-Taylor ET₀ estimate (mm/day).
///
/// ```text
/// ET₀_PT = α × 0.408 × (Δ / (Δ + γ)) × (Rn - G)
/// ```
///
/// A radiation-only method requiring net radiation and temperature (no wind
/// or humidity). The coefficient α = 1.26 accounts for the empirical ratio
/// of actual to equilibrium evaporation for well-watered surfaces.
///
/// # Reference
///
/// Priestley CHB, Taylor RJ (1972) "On the assessment of surface heat flux
/// and evaporation using large-scale parameters." *Monthly Weather Review*
/// 100(2): 81-92.
///
/// The 0.408 factor converts MJ/m²/day to mm/day (= 1/λ for water at 20°C).
#[must_use]
pub fn priestley_taylor_et0(rn: f64, g: f64, tmean_c: f64, elevation_m: f64) -> f64 {
    const ALPHA_PT: f64 = 1.26;
    let pressure = atmospheric_pressure(elevation_m);
    let gamma = psychrometric_constant(pressure);
    let delta = vapour_pressure_slope(tmean_c);
    (ALPHA_PT * MJ_TO_MM * (delta / (delta + gamma)) * (rn - g)).max(0.0)
}

/// Compute both Priestley-Taylor and Penman-Monteith ET₀ from the same inputs.
///
/// Returns `(pt_et0, pm_et0, rn)` for cross-validation.
#[must_use]
pub fn daily_et0_pt_and_pm(input: &DailyEt0Input) -> (f64, Et0Result) {
    let pm_result = daily_et0(input);
    let tmean = input
        .tmean
        .unwrap_or_else(|| f64::midpoint(input.tmin, input.tmax));
    let pt = priestley_taylor_et0(pm_result.rn, pm_result.g, tmean, input.elevation_m);
    (pt, pm_result)
}

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
    let denom = tmean_c + 15.0;
    if denom == 0.0 {
        return 0.0;
    }
    let t_factor = tmean_c / denom;
    if t_factor < 0.0 {
        return 0.0;
    }
    let rs_cal = MJ_TO_CAL_CM2.mul_add(rs_mj, 50.0);
    let mut et0 = 0.013 * t_factor * rs_cal;
    if rh_pct < 50.0 {
        et0 *= 1.0 + (50.0 - rh_pct) / 70.0;
    }
    et0.max(0.0)
}

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

// Thornthwaite (1948) monthly ET₀ has moved to `eco::thornthwaite`.
// Re-exported for backward compatibility.
pub use super::thornthwaite::{
    annual_heat_index, mean_daylight_hours_for_month, monthly_heat_index_term,
    thornthwaite_exponent, thornthwaite_monthly_et0, thornthwaite_unadjusted_et0,
};

// ── Blaney-Criddle (1950) PET ─────────────────────────────────────────

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
    let n = super::solar::daylight_hours(latitude_rad, day_of_year);
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

    #[test]
    fn test_priestley_taylor_zero_radiation() {
        let pt = priestley_taylor_et0(0.0, 0.0, 20.0, 0.0);
        assert!((pt).abs() < 1e-10, "PT should be 0 when Rn=0: {pt}");
    }

    #[test]
    fn test_priestley_taylor_negative_rn_clamped() {
        let pt = priestley_taylor_et0(-2.0, 0.0, 15.0, 0.0);
        assert!(
            (pt).abs() < 1e-10,
            "PT should clamp to 0 for negative Rn: {pt}"
        );
    }

    #[test]
    fn test_priestley_taylor_summer_reasonable() {
        let pt = priestley_taylor_et0(15.0, 0.0, 25.0, 0.0);
        assert!(
            pt > 3.0 && pt < 10.0,
            "PT should be 3-10 mm/day for summer Rn=15: {pt}"
        );
    }

    #[test]
    fn test_priestley_taylor_increases_with_rn() {
        let pt_low = priestley_taylor_et0(5.0, 0.0, 20.0, 0.0);
        let pt_high = priestley_taylor_et0(20.0, 0.0, 20.0, 0.0);
        assert!(
            pt_high > pt_low,
            "PT should increase with Rn: {pt_low} → {pt_high}"
        );
    }

    #[test]
    fn test_priestley_taylor_increases_with_temp() {
        let pt_cool = priestley_taylor_et0(15.0, 0.0, 5.0, 0.0);
        let pt_warm = priestley_taylor_et0(15.0, 0.0, 35.0, 0.0);
        assert!(
            pt_warm > pt_cool,
            "PT should increase with temperature (Δ/(Δ+γ) increases): {pt_cool} → {pt_warm}"
        );
    }

    #[test]
    fn test_priestley_taylor_altitude_effect() {
        let pt_sea = priestley_taylor_et0(15.0, 0.0, 25.0, 0.0);
        let pt_high = priestley_taylor_et0(15.0, 0.0, 25.0, 1500.0);
        assert!(
            pt_high > pt_sea,
            "PT should be higher at altitude (lower γ → higher Δ/(Δ+γ)): {pt_sea} → {pt_high}"
        );
    }

    #[test]
    fn test_priestley_taylor_soil_heat_flux() {
        let pt_no_g = priestley_taylor_et0(15.0, 0.0, 25.0, 0.0);
        let pt_with_g = priestley_taylor_et0(15.0, 2.0, 25.0, 0.0);
        assert!(
            pt_no_g > pt_with_g,
            "PT should decrease with positive G (less energy for ET): {pt_no_g} → {pt_with_g}"
        );
    }

    #[test]
    fn test_priestley_taylor_cross_validate_pm() {
        let input = DailyEt0Input {
            tmin: 12.3,
            tmax: 21.5,
            tmean: Some(16.9),
            solar_radiation: 22.07,
            wind_speed_2m: 2.078,
            actual_vapour_pressure: 1.409,
            elevation_m: 100.0,
            latitude_deg: 50.8,
            day_of_year: 187,
        };
        let (pt, pm_result) = daily_et0_pt_and_pm(&input);
        let ratio = pt / pm_result.et0;
        assert!(
            (0.85..=1.25).contains(&ratio),
            "PT/PM ratio should be 0.85-1.25 for humid climate (Uccle): ratio={ratio}"
        );
    }

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
        let et0 = blaney_criddle_from_location(25.0, lat, 172); // June 21
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
