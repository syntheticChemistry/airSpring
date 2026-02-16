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

/// Topp equation: dielectric permittivity → volumetric water content.
///
/// θv = −5.3 × 10⁻² + 2.92 × 10⁻² ε − 5.5 × 10⁻⁴ ε² + 4.3 × 10⁻⁶ ε³
///
/// Valid for mineral soils with ε ∈ \[1, 80\].
#[must_use]
pub fn topp_equation(dielectric: f64) -> f64 {
    let e = dielectric;
    // Horner's method: ((4.3e-6 × e − 5.5e-4) × e + 2.92e-2) × e − 5.3e-2
    4.3e-6f64
        .mul_add(e, -5.5e-4)
        .mul_add(e, 2.92e-2)
        .mul_add(e, -5.3e-2)
}

/// Inverse Topp: volumetric water content → approximate dielectric.
///
/// Uses Newton–Raphson iteration with guaranteed convergence
/// for θv ∈ \[0, 0.5\] (valid range of Topp equation).
#[must_use]
pub fn inverse_topp(theta_v: f64) -> f64 {
    let mut e = 10.0; // initial guess
    for _ in 0..50 {
        let f = topp_equation(e) - theta_v;
        // Derivative: 2.92e-2 − 2 × 5.5e-4 × e + 3 × 4.3e-6 × e²
        let df = (3.0 * 4.3e-6_f64).mul_add(e.powi(2), (-(2.0 * 5.5e-4_f64)).mul_add(e, 2.92e-2));
        if df.abs() < 1e-15 {
            break;
        }
        let e_new = e - f / df;
        if (e_new - e).abs() < 1e-8 {
            break;
        }
        e = e_new.clamp(1.0, 80.0);
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
#[derive(Debug, Clone)]
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
}
