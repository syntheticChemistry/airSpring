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
/// (`barracuda::optimize::brent`).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Carsel & Parrish (1988) WRR 24:755-769 Table 1 — sand.
    const SAND: VanGenuchtenParams = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    /// Carsel & Parrish (1988) WRR 24:755-769 Table 1 — loam.
    const LOAM: VanGenuchtenParams = VanGenuchtenParams {
        theta_r: 0.078,
        theta_s: 0.43,
        alpha: 0.036,
        n_vg: 1.56,
        ks: 24.96,
    };

    /// Carsel & Parrish (1988) WRR 24:755-769 Table 1 — clay.
    const CLAY: VanGenuchtenParams = VanGenuchtenParams {
        theta_r: 0.068,
        theta_s: 0.38,
        alpha: 0.008,
        n_vg: 1.09,
        ks: 4.8,
    };

    const TOL: f64 = 1e-10;

    #[test]
    fn theta_at_saturation_returns_theta_s() {
        let p = SAND;
        let theta = van_genuchten_theta(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(
            (theta - p.theta_s).abs() < TOL,
            "θ(h=0) = {theta} should equal θ_s = {}",
            p.theta_s
        );
    }

    #[test]
    fn theta_at_very_negative_h_returns_near_theta_r() {
        let p = SAND;
        let h = -1e4;
        let theta = van_genuchten_theta(h, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(
            (theta - p.theta_r).abs() < 0.01,
            "θ(h={h}) = {theta} should be ≈ θ_r = {}",
            p.theta_r
        );
    }

    #[test]
    fn k_at_saturation_returns_k_sat() {
        let p = SAND;
        let k = van_genuchten_k(0.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(
            (k - p.ks).abs() < TOL,
            "K(h=0) = {k} should equal Ks = {}",
            p.ks
        );
    }

    #[test]
    fn k_at_very_dry_returns_near_zero() {
        let p = SAND;
        let h = -1e4;
        let k = van_genuchten_k(h, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(k < 1e-6, "K(h={h}) = {k} should be near zero");
    }

    #[test]
    fn inverse_h_round_trip() {
        let p = SAND;
        let theta_mid = f64::midpoint(p.theta_r, p.theta_s);
        let h = inverse_van_genuchten_h(theta_mid, p.theta_r, p.theta_s, p.alpha, p.n_vg)
            .expect("inverse should succeed for θ in (θr, θs)");
        let theta_back = van_genuchten_theta(h, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(
            (theta_back - theta_mid).abs() < 1e-6,
            "round-trip: θ={theta_mid} → h={h} → θ={theta_back}"
        );
    }

    #[test]
    fn carsel_parrish_sand_retention() {
        let p = SAND;
        assert!(
            (van_genuchten_theta(-10.0, p.theta_r, p.theta_s, p.alpha, p.n_vg) - 0.2143).abs()
                < 0.001,
            "sand θ(-10) ≈ 0.2143"
        );
        assert!(
            (van_genuchten_theta(-100.0, p.theta_r, p.theta_s, p.alpha, p.n_vg) - 0.0493).abs()
                < 0.001,
            "sand θ(-100) ≈ 0.0493"
        );
    }

    #[test]
    fn carsel_parrish_loam_retention() {
        let p = LOAM;
        let theta_0 = van_genuchten_theta(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((theta_0 - p.theta_s).abs() < TOL, "loam θ(0) = θ_s");
        let theta_dry = van_genuchten_theta(-500.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(
            theta_dry > p.theta_r && theta_dry < p.theta_s,
            "loam θ(-500) in (θr, θs)"
        );
    }

    #[test]
    fn carsel_parrish_clay_retention() {
        let p = CLAY;
        assert!(
            (van_genuchten_theta(-100.0, p.theta_r, p.theta_s, p.alpha, p.n_vg) - 0.3654).abs()
                < 0.005,
            "clay θ(-100) ≈ 0.3654"
        );
    }

    #[test]
    fn monotonicity_increasing_h_gives_increasing_theta() {
        let p = SAND;
        let heads: [f64; 8] = [-5000.0, -1000.0, -500.0, -100.0, -50.0, -10.0, -1.0, 0.0];
        let thetas: Vec<f64> = heads
            .iter()
            .map(|&h| van_genuchten_theta(h, p.theta_r, p.theta_s, p.alpha, p.n_vg))
            .collect();
        for i in 1..thetas.len() {
            assert!(
                thetas[i] >= thetas[i - 1],
                "monotonicity: θ(h={}) = {} should be ≥ θ(h={}) = {}",
                heads[i],
                thetas[i],
                heads[i - 1],
                thetas[i - 1]
            );
        }
    }

    #[test]
    fn boundary_se_theta_s_equals_one() {
        let p = SAND;
        let theta_s = p.theta_s;
        let se = (theta_s - p.theta_r) / (p.theta_s - p.theta_r);
        assert!((se - 1.0).abs() < TOL, "Se(θ_s) = {se} should equal 1.0");
    }

    #[test]
    fn boundary_se_theta_r_equals_zero() {
        let p = SAND;
        let theta_r = p.theta_r;
        let se = (theta_r - p.theta_r) / (p.theta_s - p.theta_r);
        assert!(se.abs() < TOL, "Se(θ_r) = {se} should equal 0.0");
    }
}
