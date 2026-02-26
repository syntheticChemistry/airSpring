// SPDX-License-Identifier: AGPL-3.0-or-later
//! Van Genuchten-Mualem soil hydraulic functions.
//!
//! Pure Rust, zero dependencies. Absorption target: `barracuda::pde::richards`
//! (already absorbed upstream — this module validates parity).
//!
//! # References
//!
//! - van Genuchten (1980) SSSA J 44:892-898
//! - Carsel & Parrish (1988) WRR 24:755-769

/// Van Genuchten water retention: θ(h).
///
/// θ = θr + (θs − θr) / [1 + (α|h|)^n]^m  where m = 1 − 1/n
///
/// # Examples
///
/// ```
/// use airspring_forge::van_genuchten::theta;
///
/// let t = theta(0.0, 0.045, 0.43, 0.145, 2.68);
/// assert!((t - 0.43).abs() < 1e-10);
/// ```
#[must_use]
pub fn theta(h: f64, theta_r: f64, theta_s: f64, alpha: f64, n: f64) -> f64 {
    if h >= 0.0 {
        return theta_s;
    }
    let m = 1.0 - 1.0 / n;
    let x = (alpha * h.abs()).powf(n);
    let se = 1.0 / (1.0 + x).powf(m);
    (theta_s - theta_r)
        .mul_add(se, theta_r)
        .clamp(theta_r, theta_s)
}

/// Mualem-van Genuchten hydraulic conductivity: K(h).
///
/// K = Ks × Se^0.5 × [1 − (1 − Se^(1/m))^m]²
#[must_use]
pub fn conductivity(h: f64, ks: f64, theta_r: f64, theta_s: f64, alpha: f64, n: f64) -> f64 {
    if h >= 0.0 {
        return ks;
    }
    let t = theta(h, theta_r, theta_s, alpha, n);
    let se = (t - theta_r) / (theta_s - theta_r);
    if se <= 0.0 {
        return 0.0;
    }
    if se >= 1.0 {
        return ks;
    }
    let m = 1.0 - 1.0 / n;
    let term = 1.0 - se.powf(1.0 / m);
    if term <= 0.0 {
        return ks;
    }
    ks * se.sqrt() * (1.0 - term.powf(m)).powi(2)
}

/// Specific moisture capacity C(h) = dθ/dh.
#[must_use]
pub fn capacity(h: f64, theta_r: f64, theta_s: f64, alpha: f64, n: f64) -> f64 {
    if h >= 0.0 {
        return 0.0;
    }
    let m = 1.0 - 1.0 / n;
    let ah = alpha * h.abs();
    let ah_n = ah.powf(n);
    let denom = (1.0 + ah_n).powf(m + 1.0);
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (theta_s - theta_r) * alpha * n * m * ah.powf(n - 1.0) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theta_saturation() {
        let t = theta(0.0, 0.045, 0.43, 0.145, 2.68);
        assert!((t - 0.43).abs() < 1e-10);
    }

    #[test]
    fn test_theta_dry() {
        let t = theta(-100.0, 0.045, 0.43, 0.145, 2.68);
        assert!((t - 0.0493).abs() < 0.001);
    }

    #[test]
    fn test_conductivity_saturated() {
        let k = conductivity(0.0, 712.8, 0.045, 0.43, 0.145, 2.68);
        assert!((k - 712.8).abs() < 1e-10);
    }

    #[test]
    fn test_conductivity_unsaturated() {
        let k = conductivity(-10.0, 712.8, 0.045, 0.43, 0.145, 2.68);
        assert!(k > 0.0 && k < 712.8);
    }

    #[test]
    fn test_capacity_positive() {
        let c = capacity(-10.0, 0.045, 0.43, 0.145, 2.68);
        assert!(c > 0.0);
    }
}
