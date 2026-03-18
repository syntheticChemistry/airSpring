// SPDX-License-Identifier: AGPL-3.0-or-later
//! Saxton & Rawls (2006) pedotransfer functions.
//!
//! Estimates wilting point, field capacity, saturation, and Ksat from
//! sand fraction, clay fraction, and organic matter percentage.
//!
//! Saxton KE, Rawls WJ (2006) Soil Sci. Soc. Am. J. 70(5):1569-1578.

/// Continuous soil hydraulic properties from Saxton & Rawls (2006) regressions.
#[derive(Debug, Clone, Copy)]
pub struct SaxtonRawlsInput {
    /// Sand fraction (0–1).
    pub sand: f64,
    /// Clay fraction (0–1).
    pub clay: f64,
    /// Organic matter percentage (e.g. 2.5 for 2.5%).
    pub om_pct: f64,
}

/// Result from Saxton-Rawls pedotransfer regression.
#[derive(Debug, Clone, Copy)]
pub struct SaxtonRawlsResult {
    /// Wilting point θ at −1500 kPa (m³/m³).
    pub theta_wp: f64,
    /// Field capacity θ at −33 kPa (m³/m³).
    pub theta_fc: f64,
    /// Saturation moisture content / porosity (m³/m³).
    pub theta_s: f64,
    /// Saturated hydraulic conductivity (mm/hr).
    pub ksat_mm_hr: f64,
    /// Slope parameter λ of the moisture-tension curve.
    pub lambda: f64,
}

// Saxton-Rawls regressions use plain arithmetic to match Python baseline bit-for-bit.
// mul_add reordering changes float rounding enough to break validation tolerances.
#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_1500_first(s: f64, c: f64, om: f64) -> f64 {
    -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_1500(s: f64, c: f64, om: f64) -> f64 {
    let first = sr_theta_1500_first(s, c, om);
    first + 0.14 * first - 0.02
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_33_first(s: f64, c: f64, om: f64) -> f64 {
    -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_33(s: f64, c: f64, om: f64) -> f64 {
    let first = sr_theta_33_first(s, c, om);
    first + 1.283 * first * first - 0.374 * first - 0.015
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_s_33_first(s: f64, c: f64, om: f64) -> f64 {
    0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om - 0.027 * c * om - 0.584 * s * c + 0.078
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_s_33(s: f64, c: f64, om: f64) -> f64 {
    let first = sr_theta_s_33_first(s, c, om);
    first + 0.636 * first - 0.107
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
fn sr_theta_s(s: f64, c: f64, om: f64) -> f64 {
    sr_theta_33(s, c, om) + sr_theta_s_33(s, c, om) - 0.097 * s + 0.043
}

const SR_PRESSURE_1500_KPA: f64 = 1500.0;
const SR_PRESSURE_33_KPA: f64 = 33.0;
const SR_KSAT_COEFF: f64 = 1930.0;
const SR_KSAT_EXPONENT: f64 = 3.0;

fn sr_lambda(s: f64, c: f64, om: f64) -> f64 {
    let t33 = sr_theta_33(s, c, om);
    let t1500 = sr_theta_1500(s, c, om);
    let b = (SR_PRESSURE_1500_KPA.ln() - SR_PRESSURE_33_KPA.ln()) / (t33.ln() - t1500.ln());
    1.0 / b
}

fn sr_ksat(s: f64, c: f64, om: f64) -> f64 {
    let ts = sr_theta_s(s, c, om);
    let t33 = sr_theta_33(s, c, om);
    let lam = sr_lambda(s, c, om);
    SR_KSAT_COEFF * (ts - t33).powf(SR_KSAT_EXPONENT - lam)
}

/// Compute all Saxton-Rawls hydraulic properties from soil texture and OM.
///
/// # Arguments
/// * `input` — Sand/clay fractions (0–1) and organic matter percentage.
///
/// # Returns
/// Full set of hydraulic properties including `θ_wp`, `θ_fc`, `θ_s`, Ksat, and λ.
///
/// # Reference
/// Saxton KE, Rawls WJ (2006) Soil Sci. Soc. Am. J. 70(5):1569-1578.
#[must_use]
pub fn saxton_rawls(input: &SaxtonRawlsInput) -> SaxtonRawlsResult {
    let (s, c, om) = (input.sand, input.clay, input.om_pct);
    SaxtonRawlsResult {
        theta_wp: sr_theta_1500(s, c, om),
        theta_fc: sr_theta_33(s, c, om),
        theta_s: sr_theta_s(s, c, om),
        ksat_mm_hr: sr_ksat(s, c, om),
        lambda: sr_lambda(s, c, om),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
#[allow(clippy::expect_used, reason = "test code may use expect")]
mod tests {
    use super::*;

    #[test]
    fn loam() {
        let input = SaxtonRawlsInput {
            sand: 0.40,
            clay: 0.20,
            om_pct: 2.5,
        };
        let r = saxton_rawls(&input);
        assert!(r.theta_wp > 0.08 && r.theta_wp < 0.20, "wp={}", r.theta_wp);
        assert!(r.theta_fc > 0.20 && r.theta_fc < 0.40, "fc={}", r.theta_fc);
        assert!(r.theta_s > 0.35 && r.theta_s < 0.55, "θs={}", r.theta_s);
        assert!(
            r.ksat_mm_hr > 1.0 && r.ksat_mm_hr < 100.0,
            "Ksat={}",
            r.ksat_mm_hr
        );
        assert!(r.theta_wp < r.theta_fc, "wp < fc");
        assert!(r.theta_fc < r.theta_s, "fc < θs");
    }

    #[test]
    fn sand() {
        let input = SaxtonRawlsInput {
            sand: 0.92,
            clay: 0.03,
            om_pct: 1.0,
        };
        let r = saxton_rawls(&input);
        assert!(
            r.ksat_mm_hr > 50.0,
            "Sand Ksat should be high: {}",
            r.ksat_mm_hr
        );
        assert!(r.theta_wp < 0.10, "Sand WP should be low: {}", r.theta_wp);
    }

    #[test]
    fn clay() {
        let input = SaxtonRawlsInput {
            sand: 0.20,
            clay: 0.55,
            om_pct: 2.0,
        };
        let r = saxton_rawls(&input);
        assert!(
            r.ksat_mm_hr < 10.0,
            "Clay Ksat should be low: {}",
            r.ksat_mm_hr
        );
        assert!(r.theta_wp > 0.15, "Clay WP should be high: {}", r.theta_wp);
    }

    #[test]
    fn ordering() {
        let sand = saxton_rawls(&SaxtonRawlsInput {
            sand: 0.92,
            clay: 0.03,
            om_pct: 1.0,
        });
        let clay = saxton_rawls(&SaxtonRawlsInput {
            sand: 0.20,
            clay: 0.55,
            om_pct: 2.0,
        });
        assert!(sand.ksat_mm_hr > clay.ksat_mm_hr);
        assert!(clay.theta_wp > sand.theta_wp);
        assert!(clay.theta_fc > sand.theta_fc);
    }

    #[test]
    fn om_sensitivity() {
        let lo = saxton_rawls(&SaxtonRawlsInput {
            sand: 0.40,
            clay: 0.20,
            om_pct: 0.5,
        });
        let hi = saxton_rawls(&SaxtonRawlsInput {
            sand: 0.40,
            clay: 0.20,
            om_pct: 5.0,
        });
        assert!(hi.theta_wp > lo.theta_wp, "Higher OM → higher WP");
        assert!(hi.theta_fc > lo.theta_fc, "Higher OM → higher FC");
    }
}
