// SPDX-License-Identifier: AGPL-3.0-or-later
//! Topp equation: dielectric permittivity ↔ volumetric water content.
//!
//! Topp GC, Davis JL, Annan AP (1980) "Electromagnetic determination of soil
//! water content" Water Resources Research 16(3), 574–582.

/// Topp (1980) polynomial coefficients: θv = A₀ + A₁·ε + A₂·ε² + A₃·ε³.
/// Source: Topp GC et al. (1980), Water Resources Research 16(3), Table 1.
const TOPP_A0: f64 = -5.3e-2;
const TOPP_A1: f64 = 2.92e-2;
const TOPP_A2: f64 = -5.5e-4;
const TOPP_A3: f64 = 4.3e-6;

/// Valid dielectric range for the Topp equation (air to saturated).
const TOPP_EPSILON_MIN: f64 = 1.0;
const TOPP_EPSILON_MAX: f64 = 80.0;

/// Newton-Raphson initial guess for inverse Topp (mid-range ε ≈ 10).
const INVERSE_TOPP_INITIAL_GUESS: f64 = 10.0;

/// Maximum Newton-Raphson iterations for inverse Topp.
const INVERSE_TOPP_MAX_ITER: usize = 50;

/// Newton-Raphson convergence tolerance for inverse Topp (ε change < this).
const INVERSE_TOPP_CONVERGENCE: f64 = 1e-8;

/// Derivative guard: stop if |f'(ε)| drops below this to avoid division by zero.
const INVERSE_TOPP_DERIV_GUARD: f64 = 1e-15;

/// Topp equation: dielectric permittivity → volumetric water content.
///
/// θv = −5.3 × 10⁻² + 2.92 × 10⁻² ε − 5.5 × 10⁻⁴ ε² + 4.3 × 10⁻⁶ ε³
///
/// Valid for mineral soils with ε ∈ \[1, 80\].
#[must_use]
pub fn topp_equation(dielectric: f64) -> f64 {
    let e = dielectric;
    // Horner's method: ((A₃·e + A₂)·e + A₁)·e + A₀
    TOPP_A3
        .mul_add(e, TOPP_A2)
        .mul_add(e, TOPP_A1)
        .mul_add(e, TOPP_A0)
}

/// Inverse Topp: volumetric water content → approximate dielectric.
///
/// Uses Newton–Raphson iteration with guaranteed convergence
/// for θv ∈ \[0, 0.5\] (valid range of Topp equation).
#[must_use]
pub fn inverse_topp(theta_v: f64) -> f64 {
    let mut e = INVERSE_TOPP_INITIAL_GUESS;
    for _ in 0..INVERSE_TOPP_MAX_ITER {
        let f = topp_equation(e) - theta_v;
        // Derivative: A₁ + 2·A₂·e + 3·A₃·e²
        let df = (3.0 * TOPP_A3).mul_add(e.powi(2), (2.0 * TOPP_A2).mul_add(e, TOPP_A1));
        if df.abs() < INVERSE_TOPP_DERIV_GUARD {
            break;
        }
        let e_new = e - f / df;
        if (e_new - e).abs() < INVERSE_TOPP_CONVERGENCE {
            break;
        }
        e = e_new.clamp(TOPP_EPSILON_MIN, TOPP_EPSILON_MAX);
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_values() {
        let cases = [
            (3.0, 0.031),
            (5.0, 0.083),
            (10.0, 0.187),
            (15.0, 0.271),
            (20.0, 0.347),
            (25.0, 0.405),
            (30.0, 0.440),
        ];
        for (eps, expected) in cases {
            let theta = topp_equation(eps);
            assert!(
                (theta - expected).abs() < 0.02,
                "θv at ε={eps}: {theta}, expected {expected}"
            );
        }
    }

    #[test]
    fn air_boundary() {
        let theta_air = topp_equation(1.0);
        assert!(theta_air < 0.01, "θv at ε=1: {theta_air}");
    }

    #[test]
    fn round_trip() {
        for &theta in &[0.10, 0.20, 0.30, 0.40] {
            let eps = inverse_topp(theta);
            let recovered = topp_equation(eps);
            assert!(
                (recovered - theta).abs() < 0.001,
                "Round-trip θ={theta}: recovered={recovered}"
            );
        }
    }

    #[test]
    fn monotonic_increasing() {
        let eps_values = [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let thetas: Vec<f64> = eps_values.iter().map(|&e| topp_equation(e)).collect();
        for w in thetas.windows(2) {
            assert!(w[1] > w[0], "Topp should be monotonically increasing");
        }
    }

    #[test]
    fn inverse_boundary() {
        let eps_dry = inverse_topp(0.05);
        let eps_wet = inverse_topp(0.45);
        assert!(
            eps_dry < eps_wet,
            "Drier soil → lower ε: dry={eps_dry}, wet={eps_wet}"
        );
        assert!(eps_dry >= 1.0, "ε must be ≥ 1 (air): {eps_dry}");
    }
}
