// SPDX-License-Identifier: AGPL-3.0-or-later
//! FAO-56 Penman-Monteith Reference Evapotranspiration (ET₀).
//!
//! Implements the standard FAO Paper 56 equation (Allen et al., 1998):
//!
//! ```text
//! ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
//! ```
//!
//! # Reference
//!
//! Allen RG, Pereira LS, Raes D, Smith M (1998)
//! "Crop evapotranspiration — Guidelines for computing crop water requirements"
//! FAO Irrigation and Drainage Paper 56, Rome.

use crate::eco::solar::{
    clear_sky_radiation, extraterrestrial_radiation, net_longwave_radiation, net_radiation,
    net_shortwave_radiation,
};

use super::atmosphere;

/// Latent heat conversion: MJ/m²/day → mm/day. FAO-56 (1/λ at 20°C).
const MJ_TO_MM: f64 = 0.408;
/// Celsius to Kelvin offset. FAO-56 Eq. 6 uses T+273; 273.15 is the
/// standard conversion for thermodynamic consistency.
const CELSIUS_TO_KELVIN: f64 = 273.15;
/// Penman-Monteith wind term numerator (900). FAO-56 Eq. 6.
const FAO56_PM_WIND_NUMERATOR: f64 = 900.0;
/// Penman-Monteith wind term u₂ coefficient. FAO-56 Eq. 6: γ(1 + 0.34 u₂).
const FAO56_PM_WIND_U2_COEFF: f64 = 0.34;

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
    let numerator = (MJ_TO_MM * delta).mul_add(
        rn - g,
        gamma * (FAO56_PM_WIND_NUMERATOR / (tmean_c + CELSIUS_TO_KELVIN)) * u2 * vpd_kpa,
    );
    let denominator = gamma.mul_add(FAO56_PM_WIND_U2_COEFF.mul_add(u2, 1.0), delta);
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
    let pressure = atmosphere::atmospheric_pressure(input.elevation_m);
    let gamma = atmosphere::psychrometric_constant(pressure);
    let delta = atmosphere::vapour_pressure_slope(tmean);

    // Vapour pressures
    let es = atmosphere::mean_saturation_vapour_pressure(input.tmin, input.tmax);
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

/// Compute both Priestley-Taylor and Penman-Monteith ET₀ from the same inputs.
///
/// Returns `(pt_et0, pm_et0, rn)` for cross-validation.
#[must_use]
pub fn daily_et0_pt_and_pm(input: &DailyEt0Input) -> (f64, Et0Result) {
    let pm_result = daily_et0(input);
    let tmean = input
        .tmean
        .unwrap_or_else(|| f64::midpoint(input.tmin, input.tmax));
    let pt = super::priestley_taylor::priestley_taylor_et0(
        pm_result.rn,
        pm_result.g,
        tmean,
        input.elevation_m,
    );
    (pt, pm_result)
}
