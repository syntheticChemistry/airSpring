// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hargreaves–Samani ET₀ (FAO-56 Eq. 52).
//!
//! Temperature-only method for data-sparse deployments.

/// Hargreaves empirical coefficient. FAO-56 Eq. 52.
const HARGREAVES_COEFF: f64 = 0.0023;
/// Hargreaves temperature offset (°C). FAO-56 Eq. 52.
const HARGREAVES_TEMP_OFFSET: f64 = 17.8;

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
/// # Upstream note (`BarraCuda` S66)
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
    (HARGREAVES_COEFF
        * (tmean + HARGREAVES_TEMP_OFFSET)
        * (tmax - tmin).max(0.0).sqrt()
        * ra_mm_day)
        .max(0.0)
}
