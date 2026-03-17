// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure FAO-56 equation functions — Eqs. 69, 71–73, 77.

/// FAO-56 Eq. 69: dual crop evapotranspiration.
///
/// `ETc` = (Kcb × Ks + Ke) × ET₀
#[must_use]
pub fn etc_dual(kcb: f64, ks: f64, ke: f64, et0: f64) -> f64 {
    kcb.mul_add(ks, ke) * et0
}

/// FAO-56 Eq. 72: upper limit on evapotranspiration coefficient.
///
/// `Kc_max` = max(1.2 + \[0.04(u₂ − 2) − 0.004(RHmin − 45)\] × (h/3)^0.3,
///              Kcb + 0.05)
#[must_use]
pub fn kc_max(u2: f64, rh_min: f64, h: f64, kcb: f64) -> f64 {
    let h_clamp = h.max(0.001);
    let climate_term =
        (0.04f64.mul_add(u2 - 2.0, -0.004 * (rh_min - 45.0))) * (h_clamp / 3.0).powf(0.3);
    (1.2 + climate_term).max(kcb + 0.05)
}

/// FAO-56 Eq. 73: total evaporable water from the surface soil layer.
///
/// TEW = 1000 × (`θFC` − 0.5 × `θWP`) × Ze (mm)
#[must_use]
pub fn total_evaporable_water(theta_fc: f64, theta_wp: f64, ze: f64) -> f64 {
    1000.0 * theta_wp.mul_add(-0.5, theta_fc) * ze
}

/// FAO-56 Eq. 72: evaporation reduction coefficient.
///
/// Kr = 1.0 when De ≤ REW (stage 1 drying), otherwise
/// Kr = (TEW − De) / (TEW − REW) clamped to \[0, 1\].
#[must_use]
pub fn evaporation_reduction(tew: f64, rew: f64, de: f64) -> f64 {
    if de <= rew {
        return 1.0;
    }
    if tew <= rew {
        return 0.0;
    }
    ((tew - de) / (tew - rew)).clamp(0.0, 1.0)
}

/// FAO-56 Eq. 71: soil evaporation coefficient.
///
/// `Ke` = min(Kr × (`Kc_max` − `Kcb`), `few` × `Kc_max`), bounded ≥ 0.
#[must_use]
pub fn soil_evaporation_ke(kr: f64, kcb: f64, kc_max_val: f64, few: f64) -> f64 {
    let ke = kr.mul_add(kc_max_val, -kr * kcb);
    ke.min(few * kc_max_val).max(0.0)
}

/// FAO-56 Eq. 77 (simplified): daily evaporation layer water balance.
///
/// De,i = De,i−1 − P − I + (Ke × ET₀)/few, clamped to \[0, TEW\].
#[must_use]
pub fn evaporation_layer_balance(
    de_prev: f64,
    precip: f64,
    irrig: f64,
    ke: f64,
    et0: f64,
    few: f64,
    tew: f64,
) -> f64 {
    let evap = if few > 0.001 { ke * et0 / few } else { 0.0 };
    (de_prev - precip - irrig + evap).clamp(0.0, tew)
}
