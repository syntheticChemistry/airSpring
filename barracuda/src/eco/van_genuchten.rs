// SPDX-License-Identifier: AGPL-3.0-or-later
//! Van Genuchten-Mualem soil hydraulic functions (f64, full-featured).
//!
//! Used by `eco::richards` for the 1D Richards equation solver. This module
//! provides θ(h), K(h), C(h), and inverse θ→h with numerical guards for
//! stability in PDE contexts.
//!
//! # References
//!
//! - van Genuchten (1980) SSSA J 44:892-898
//! - Carsel & Parrish (1988) WRR 24:755-769

use barracuda::optimize::brent;

/// Maximum absolute head for VG θ(h) input guard (cm).
const VG_H_ABS_MAX: f64 = 1e4;

/// Overflow guard for (α|h|)^n computation.
const VG_POWF_MAX: f64 = 1e10;

/// Capacity C(h) returned at saturation (h ≥ 0) — small positive value
/// ensures the mixed-form C·∂h/∂t denominator never vanishes.
const SATURATED_CAPACITY: f64 = 1e-6;

/// Minimum absolute head for capacity calculation (cm) — avoids
/// numerical instability as h → 0⁻ in the VG capacity derivative.
const CAPACITY_H_MIN: f64 = 0.1;

/// Minimum capacity allowed from VG derivative.
const CAPACITY_FLOOR: f64 = 1e-10;

/// Maximum capacity (cm⁻¹) — physical upper bound.
const CAPACITY_CEIL: f64 = 1e2;

/// Pressure head lower bound for K(h) and inverse search (cm).
const H_CLIP_MIN: f64 = -10_000.0;

/// Van Genuchten soil parameters.
#[derive(Debug, Clone, Copy)]
pub struct VanGenuchtenParams {
    /// Residual water content (m³/m³).
    pub theta_r: f64,
    /// Saturated water content (m³/m³).
    pub theta_s: f64,
    /// Inverse of air-entry pressure (1/cm).
    pub alpha: f64,
    /// Pore-size distribution index.
    pub n_vg: f64,
    /// Saturated hydraulic conductivity (cm/day).
    pub ks: f64,
}

/// Van Genuchten water retention: θ(h).
///
/// θ = θr + (θs - θr) / [1 + (α|h|)^n]^m   where m = 1 - 1/n
#[must_use]
pub fn van_genuchten_theta(h: f64, theta_r: f64, theta_s: f64, alpha: f64, n_vg: f64) -> f64 {
    if h >= 0.0 {
        return theta_s;
    }
    let h_safe = h.abs().min(VG_H_ABS_MAX);
    let m = 1.0 - 1.0 / n_vg;
    let x = (alpha * h_safe).powf(n_vg).min(VG_POWF_MAX);
    let se = 1.0 / (1.0 + x).powf(m);
    let theta = (theta_s - theta_r).mul_add(se, theta_r);
    theta.clamp(theta_r, theta_s)
}

/// Mualem-van Genuchten hydraulic conductivity: K(h).
///
/// K = Ks × Se^0.5 × [1 - (1 - Se^(1/m))^m]^2
#[must_use]
pub fn van_genuchten_k(h: f64, ks: f64, theta_r: f64, theta_s: f64, alpha: f64, n_vg: f64) -> f64 {
    if h >= 0.0 {
        return ks;
    }
    if h < H_CLIP_MIN {
        return 0.0;
    }
    let m = 1.0 - 1.0 / n_vg;
    let theta = van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg);
    let se = (theta - theta_r) / (theta_s - theta_r);
    if se <= 0.0 {
        return 0.0;
    }
    if se >= 1.0 {
        return ks;
    }
    let term = 1.0 - se.powf(1.0 / m);
    if term <= 0.0 {
        return ks;
    }
    let kr = se.sqrt() * (1.0 - term.powf(m)).powi(2);
    ks * kr.clamp(0.0, 1.0)
}

/// Capacity function C(h) = dθ/dh (needed for mixed form).
#[must_use]
pub fn van_genuchten_capacity(h: f64, theta_r: f64, theta_s: f64, alpha: f64, n_vg: f64) -> f64 {
    if h >= 0.0 {
        return SATURATED_CAPACITY;
    }
    let h_safe = h.abs().clamp(CAPACITY_H_MIN, VG_H_ABS_MAX);
    let m = 1.0 - 1.0 / n_vg;
    let x = (alpha * h_safe).powf(n_vg).min(VG_POWF_MAX);
    let denom = (1.0 + x).powf(m + 1.0);
    if denom <= 0.0 || !denom.is_finite() {
        return SATURATED_CAPACITY;
    }
    let dse_dh = m * n_vg * alpha.powf(n_vg) * h_safe.powf(n_vg - 1.0) / denom;
    let result = (theta_s - theta_r) * dse_dh;
    result.clamp(CAPACITY_FLOOR, CAPACITY_CEIL)
}

/// Inverse Van Genuchten: find pressure head `h` (cm) for a target moisture θ.
///
/// Solves θ(h) − `θ_target` = 0 using Brent's root-finding method
/// (`barracuda::optimize::brent`, neuralSpring optimizer lineage).
///
/// # Cross-Spring Provenance
///
/// | Primitive | Origin | Purpose |
/// |-----------|--------|---------|
/// | `brent` | Brent (1973) via `barracuda::optimize` S52+ | Guaranteed-convergence root-finder |
/// | `van_genuchten_theta` | van Genuchten (1980), airSpring | Forward VG retention curve |
///
/// Brent's method combines bisection's reliability with inverse-quadratic
/// interpolation speed. For VG inversion, convergence is typically < 10
/// iterations since the retention curve is smooth and monotone on (−∞, 0).
///
/// Returns `None` if `θ_target` is outside \[θr, θs\] or Brent fails.
#[must_use]
pub fn inverse_van_genuchten_h(
    theta_target: f64,
    theta_r: f64,
    theta_s: f64,
    alpha: f64,
    n_vg: f64,
) -> Option<f64> {
    if theta_target >= theta_s {
        return Some(0.0);
    }
    if theta_target <= theta_r {
        return None;
    }

    let f = |h: f64| van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg) - theta_target;

    brent(f, H_CLIP_MIN, -1e-6, 1e-8, 100).ok().map(|r| r.root)
}
