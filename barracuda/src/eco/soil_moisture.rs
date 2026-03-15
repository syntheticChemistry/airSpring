// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soil moisture sensor calibration — dielectric permittivity to volumetric water content.
//!
//! Implements the Topp equation (Topp et al., 1980) and soil-specific
//! calibration curves for converting dielectric sensor readings to
//! volumetric water content (θv).
//!
//! # Reference
//!
//! Topp GC, Davis JL, Annan AP (1980) "Electromagnetic
//! determination of soil water content" Water Resources Research 16(3), 574–582.
//!
//! Also implements:
//! - Field capacity (FC) and wilting point (WP) from soil texture
//! - Plant available water (PAW = FC − WP)
//! - Soil water deficit (SWD)

// ── Topp equation ────────────────────────────────────────────────────

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

// ── Soil texture classification ──────────────────────────────────────

/// USDA soil texture classes and their typical hydraulic properties.
///
/// Values sourced from Saxton & Rawls (2006) and USDA NRCS Soil Survey Manual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoilTexture {
    Sand,
    LoamySand,
    SandyLoam,
    Loam,
    SiltLoam,
    Silt,
    SandyClayLoam,
    ClayLoam,
    SiltyClayLoam,
    SandyClay,
    SiltyClay,
    Clay,
}

/// Hydraulic properties for a soil texture class.
#[derive(Debug, Clone, Copy)]
pub struct SoilHydraulicProps {
    /// Field capacity (m³/m³) at −33 kPa
    pub field_capacity: f64,
    /// Wilting point (m³/m³) at −1500 kPa
    pub wilting_point: f64,
    /// Saturated hydraulic conductivity Ksat (mm/hr)
    pub ksat_mm_hr: f64,
    /// Porosity (m³/m³)
    pub porosity: f64,
}

impl SoilTexture {
    /// Typical hydraulic properties from USDA soil texture triangle.
    ///
    /// Values from Saxton & Rawls (2006) and USDA NRCS.
    #[must_use]
    pub const fn hydraulic_properties(&self) -> SoilHydraulicProps {
        match self {
            Self::Sand => SoilHydraulicProps {
                field_capacity: 0.10,
                wilting_point: 0.05,
                ksat_mm_hr: 210.0,
                porosity: 0.43,
            },
            Self::LoamySand => SoilHydraulicProps {
                field_capacity: 0.12,
                wilting_point: 0.06,
                ksat_mm_hr: 61.0,
                porosity: 0.44,
            },
            Self::SandyLoam => SoilHydraulicProps {
                field_capacity: 0.18,
                wilting_point: 0.08,
                ksat_mm_hr: 26.0,
                porosity: 0.45,
            },
            Self::Loam => SoilHydraulicProps {
                field_capacity: 0.27,
                wilting_point: 0.12,
                ksat_mm_hr: 13.0,
                porosity: 0.46,
            },
            Self::SiltLoam => SoilHydraulicProps {
                field_capacity: 0.33,
                wilting_point: 0.13,
                ksat_mm_hr: 6.8,
                porosity: 0.47,
            },
            Self::Silt => SoilHydraulicProps {
                field_capacity: 0.33,
                wilting_point: 0.09,
                ksat_mm_hr: 6.8,
                porosity: 0.46,
            },
            Self::SandyClayLoam => SoilHydraulicProps {
                field_capacity: 0.26,
                wilting_point: 0.15,
                ksat_mm_hr: 4.3,
                porosity: 0.40,
            },
            Self::ClayLoam => SoilHydraulicProps {
                field_capacity: 0.32,
                wilting_point: 0.20,
                ksat_mm_hr: 2.3,
                porosity: 0.42,
            },
            Self::SiltyClayLoam => SoilHydraulicProps {
                field_capacity: 0.37,
                wilting_point: 0.22,
                ksat_mm_hr: 1.5,
                porosity: 0.43,
            },
            Self::SandyClay => SoilHydraulicProps {
                field_capacity: 0.30,
                wilting_point: 0.21,
                ksat_mm_hr: 1.2,
                porosity: 0.38,
            },
            Self::SiltyClay => SoilHydraulicProps {
                field_capacity: 0.37,
                wilting_point: 0.25,
                ksat_mm_hr: 0.9,
                porosity: 0.41,
            },
            Self::Clay => SoilHydraulicProps {
                field_capacity: 0.36,
                wilting_point: 0.25,
                ksat_mm_hr: 0.6,
                porosity: 0.38,
            },
        }
    }
}

// ── Saxton & Rawls (2006) pedotransfer functions ─────────────────────

/// Continuous soil hydraulic properties from Saxton & Rawls (2006) regressions.
///
/// Estimates wilting point, field capacity, saturation, and Ksat from
/// sand fraction, clay fraction, and organic matter percentage.
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
/// First estimate of wilting point moisture (Saxton & Rawls 2006).
#[must_use]
fn sr_theta_1500_first(s: f64, c: f64, om: f64) -> f64 {
    -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
/// Wilting point θ at −1500 kPa.
#[must_use]
fn sr_theta_1500(s: f64, c: f64, om: f64) -> f64 {
    let first = sr_theta_1500_first(s, c, om);
    first + 0.14 * first - 0.02
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
/// First estimate of field capacity moisture.
#[must_use]
fn sr_theta_33_first(s: f64, c: f64, om: f64) -> f64 {
    -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
/// Field capacity θ at −33 kPa.
#[must_use]
fn sr_theta_33(s: f64, c: f64, om: f64) -> f64 {
    let first = sr_theta_33_first(s, c, om);
    first + 1.283 * first * first - 0.374 * first - 0.015
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
/// First estimate of moisture between saturation and field capacity.
#[must_use]
fn sr_theta_s_33_first(s: f64, c: f64, om: f64) -> f64 {
    0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om - 0.027 * c * om - 0.584 * s * c + 0.078
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
/// Moisture between saturation and field capacity.
#[must_use]
fn sr_theta_s_33(s: f64, c: f64, om: f64) -> f64 {
    let first = sr_theta_s_33_first(s, c, om);
    first + 0.636 * first - 0.107
}

#[expect(
    clippy::suboptimal_flops,
    reason = "match reference implementation bit-for-bit"
)]
/// Saturation moisture content (porosity).
#[must_use]
fn sr_theta_s(s: f64, c: f64, om: f64) -> f64 {
    sr_theta_33(s, c, om) + sr_theta_s_33(s, c, om) - 0.097 * s + 0.043
}

/// Lambda parameter (slope of moisture-tension curve in log-log space).
#[must_use]
fn sr_lambda(s: f64, c: f64, om: f64) -> f64 {
    let t33 = sr_theta_33(s, c, om);
    let t1500 = sr_theta_1500(s, c, om);
    let b = (1500.0_f64.ln() - 33.0_f64.ln()) / (t33.ln() - t1500.ln());
    1.0 / b
}

/// Saturated hydraulic conductivity (mm/hr).
#[must_use]
fn sr_ksat(s: f64, c: f64, om: f64) -> f64 {
    let ts = sr_theta_s(s, c, om);
    let t33 = sr_theta_33(s, c, om);
    let lam = sr_lambda(s, c, om);
    1930.0 * (ts - t33).powf(3.0 - lam)
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

// ── Water availability calculations ──────────────────────────────────

/// Plant available water (PAW) in mm for a given soil depth.
///
/// PAW = (FC − WP) × `depth_mm`
#[must_use]
pub fn plant_available_water(fc: f64, wp: f64, depth_mm: f64) -> f64 {
    (fc - wp) * depth_mm
}

/// Soil water deficit: how much water is needed to reach field capacity.
///
/// SWD = (FC − `θv_current`) × `depth_mm`, clamped to ≥ 0.
#[must_use]
pub fn soil_water_deficit(fc: f64, current_theta: f64, depth_mm: f64) -> f64 {
    (fc - current_theta).max(0.0) * depth_mm
}

/// Management allowable depletion (MAD) for irrigation triggering.
///
/// Returns `true` when soil water depletion exceeds the MAD fraction of PAW,
/// indicating irrigation should be applied.
///
/// Typical MAD: 0.50 for most crops, 0.30 for sensitive crops.
#[must_use]
pub fn irrigation_trigger(fc: f64, wp: f64, current_theta: f64, mad_fraction: f64) -> bool {
    let paw = fc - wp;
    let depletion = fc - current_theta;
    depletion > mad_fraction * paw
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_topp_equation_published_values() {
        // Topp (1980) Table 1: dielectric → θv for mineral soils
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
    fn test_topp_air_boundary() {
        // At ε = 1 (air), θv should be near 0 or slightly negative
        let theta_air = topp_equation(1.0);
        assert!(theta_air < 0.01, "θv at ε=1: {theta_air}");
    }

    #[test]
    fn test_inverse_topp_round_trip() {
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
    fn test_hydraulic_properties_sandy_loam() {
        let props = SoilTexture::SandyLoam.hydraulic_properties();
        assert!((props.field_capacity - 0.18).abs() < f64::EPSILON);
        assert!((props.wilting_point - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plant_available_water() {
        // Sandy loam: FC=0.18, WP=0.08, 300 mm root zone → PAW = 30 mm
        let paw = plant_available_water(0.18, 0.08, 300.0);
        assert!((paw - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_soil_water_deficit() {
        // FC=0.33, θ=0.25, depth=600 mm → SWD = 48 mm
        let swd = soil_water_deficit(0.33, 0.25, 600.0);
        assert!((swd - 48.0).abs() < 0.01);
    }

    #[test]
    fn test_soil_water_deficit_above_fc() {
        // When θ > FC, deficit is zero (clamped)
        let swd = soil_water_deficit(0.33, 0.40, 600.0);
        assert!((swd - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_irrigation_trigger() {
        // Silt loam: FC=0.33, WP=0.13, MAD=0.50
        // At θ=0.22: depletion=0.11, MAD×PAW=0.10 → trigger
        assert!(irrigation_trigger(0.33, 0.13, 0.22, 0.50));
        // At θ=0.30: depletion=0.03 < 0.10 → no trigger
        assert!(!irrigation_trigger(0.33, 0.13, 0.30, 0.50));
        // At FC: depletion=0 → no trigger
        assert!(!irrigation_trigger(0.33, 0.13, 0.33, 0.50));
    }

    #[test]
    fn test_sandy_clay_not_sandy_cite() {
        // Regression: verify the typo SandyCite was fixed to SandyClay
        let props = SoilTexture::SandyClay.hydraulic_properties();
        assert!((props.field_capacity - 0.30).abs() < f64::EPSILON);
        assert!((props.wilting_point - 0.21).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_textures_have_valid_properties() {
        let textures = [
            SoilTexture::Sand,
            SoilTexture::LoamySand,
            SoilTexture::SandyLoam,
            SoilTexture::Loam,
            SoilTexture::SiltLoam,
            SoilTexture::Silt,
            SoilTexture::SandyClayLoam,
            SoilTexture::ClayLoam,
            SoilTexture::SiltyClayLoam,
            SoilTexture::SandyClay,
            SoilTexture::SiltyClay,
            SoilTexture::Clay,
        ];
        for texture in &textures {
            let p = texture.hydraulic_properties();
            assert!(p.field_capacity > p.wilting_point, "{texture:?}: FC > WP");
            assert!(p.porosity > p.field_capacity, "{texture:?}: porosity > FC");
            assert!(p.ksat_mm_hr > 0.0, "{texture:?}: Ksat > 0");
            assert!(p.wilting_point >= 0.0, "{texture:?}: WP >= 0");
        }
    }

    #[test]
    fn test_ksat_ordering_sand_to_clay() {
        // Sand should have highest Ksat, clay lowest
        let sand_ksat = SoilTexture::Sand.hydraulic_properties().ksat_mm_hr;
        let clay_ksat = SoilTexture::Clay.hydraulic_properties().ksat_mm_hr;
        assert!(
            sand_ksat > clay_ksat,
            "Sand Ksat {sand_ksat} > Clay Ksat {clay_ksat}"
        );
    }

    #[test]
    fn test_topp_monotonic_increasing() {
        let eps_values = [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let thetas: Vec<f64> = eps_values.iter().map(|&e| topp_equation(e)).collect();
        for w in thetas.windows(2) {
            assert!(w[1] > w[0], "Topp should be monotonically increasing");
        }
    }

    #[test]
    fn test_inverse_topp_boundary() {
        // Very dry (low θv) and very wet (high θv)
        let eps_dry = inverse_topp(0.05);
        let eps_wet = inverse_topp(0.45);
        assert!(
            eps_dry < eps_wet,
            "Drier soil → lower ε: dry={eps_dry}, wet={eps_wet}"
        );
        assert!(eps_dry >= 1.0, "ε must be ≥ 1 (air): {eps_dry}");
    }

    #[test]
    fn test_irrigation_trigger_at_boundaries() {
        // Exactly at MAD boundary
        let fc = 0.30;
        let wp = 0.10;
        let paw = fc - wp; // 0.20
        let mad = 0.5;
        let mad_depletion = mad * paw; // 0.10
        let theta_at_mad = fc - mad_depletion; // 0.20
        // At MAD boundary, depletion == MAD×PAW → not triggered (<=)
        assert!(!irrigation_trigger(fc, wp, theta_at_mad, mad));
        // Slightly below triggers
        assert!(irrigation_trigger(fc, wp, theta_at_mad - 0.001, mad));
    }

    #[test]
    fn test_soil_water_deficit_zero_depth() {
        assert!((soil_water_deficit(0.33, 0.25, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_saxton_rawls_loam() {
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
    fn test_saxton_rawls_sand() {
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
    fn test_saxton_rawls_clay() {
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
    fn test_saxton_rawls_ordering() {
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
    fn test_saxton_rawls_om_sensitivity() {
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

    #[test]
    fn test_plant_available_water_clay() {
        let p = SoilTexture::Clay.hydraulic_properties();
        let paw = plant_available_water(p.field_capacity, p.wilting_point, 500.0);
        // Clay: FC=0.36, WP=0.25 → PAW = 0.11 × 500 = 55 mm
        assert!((paw - 55.0).abs() < 0.1, "PAW={paw}");
    }
}
