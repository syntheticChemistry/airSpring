// SPDX-License-Identifier: AGPL-3.0-or-later
//! Solar radiation estimation for ET₀.
//!
//! FAO-56 Eqs. 35 (Ångström), 50 (Hargreaves Rs), 43 (soil heat flux).

/// Default Ångström coefficient as. FAO-56 Eq. 35.
const ANGSTROM_AS: f64 = 0.25;
/// Default Ångström coefficient bs. FAO-56 Eq. 35.
const ANGSTROM_BS: f64 = 0.50;
/// Monthly soil heat flux coefficient (MJ/m²/month per °C). FAO-56 Eq. 43.
const SOIL_HEAT_FLUX_COEFF: f64 = 0.14;

/// Solar radiation from sunshine hours (FAO-56 Eq. 35).
///
/// Rs = (as + bs × n/N) × Ra
///
/// Default Ångström coefficients: as = 0.25, bs = 0.50.
/// `n` is actual sunshine hours, `N` is maximum possible daylight hours.
///
/// # Errors
///
/// Returns `InvalidInput` if `max_daylight_hours` ≤ 0.0.
pub fn solar_radiation_from_sunshine(
    sunshine_hours: f64,
    max_daylight_hours: f64,
    ra: f64,
) -> crate::error::Result<f64> {
    if max_daylight_hours <= 0.0 {
        return Err(crate::error::AirSpringError::InvalidInput(
            "max daylight hours must be positive".into(),
        ));
    }
    Ok(ANGSTROM_BS.mul_add(sunshine_hours / max_daylight_hours, ANGSTROM_AS) * ra)
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
/// For daily time steps, G ≈ 0 (handled in [`crate::eco::evapotranspiration::daily_et0`]).
#[must_use]
pub fn soil_heat_flux_monthly(t_month: f64, t_month_prev: f64) -> f64 {
    SOIL_HEAT_FLUX_COEFF.mul_add(t_month, -SOIL_HEAT_FLUX_COEFF * t_month_prev)
}
