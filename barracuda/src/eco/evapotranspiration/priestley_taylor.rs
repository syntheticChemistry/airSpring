// SPDX-License-Identifier: AGPL-3.0-or-later
//! Priestley-Taylor ET₀ (radiation-only method).
//!
//! Priestley & Taylor (1972). No wind or humidity required.

use super::atmosphere;

/// Latent heat conversion: MJ/m²/day → mm/day. FAO-56 (1/λ at 20°C).
const MJ_TO_MM: f64 = 0.408;
/// Priestley-Taylor α coefficient. Priestley & Taylor (1972).
const PRIESTLEY_TAYLOR_ALPHA: f64 = 1.26;

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
    let pressure = atmosphere::atmospheric_pressure(elevation_m);
    let gamma = atmosphere::psychrometric_constant(pressure);
    let delta = atmosphere::vapour_pressure_slope(tmean_c);
    (PRIESTLEY_TAYLOR_ALPHA * MJ_TO_MM * (delta / (delta + gamma)) * (rn - g)).max(0.0)
}
