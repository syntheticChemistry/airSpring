//! Soil moisture sensor calibration — dielectric permittivity to volumetric water content.
//!
//! Implements the Topp equation (Topp et al., 1980) and soil-specific
//! calibration curves for converting dielectric sensor readings to
//! volumetric water content (θv).
//!
//! Reference: Topp GC, Davis JL, Annan AP (1980) "Electromagnetic
//! determination of soil water content" Water Resources Research 16(3), 574-582.
//!
//! Also implements:
//! - Field capacity (FC) and wilting point (WP) from soil texture
//! - Plant available water (PAW = FC - WP)
//! - Soil water deficit (SWD)

/// Topp equation: dielectric permittivity → volumetric water content.
///
/// θv = -5.3e-2 + 2.92e-2·ε - 5.5e-4·ε² + 4.3e-6·ε³
///
/// Valid for mineral soils with ε in [1, 80].
pub fn topp_equation(dielectric: f64) -> f64 {
    let e = dielectric;
    -5.3e-2 + 2.92e-2 * e - 5.5e-4 * e.powi(2) + 4.3e-6 * e.powi(3)
}

/// Inverse Topp: volumetric water content → approximate dielectric.
/// Uses Newton-Raphson iteration.
pub fn inverse_topp(theta_v: f64) -> f64 {
    let mut e = 10.0; // initial guess
    for _ in 0..50 {
        let f = topp_equation(e) - theta_v;
        let df = 2.92e-2 - 2.0 * 5.5e-4 * e + 3.0 * 4.3e-6 * e.powi(2);
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

/// Soil texture classes and their typical hydraulic properties.
#[derive(Debug, Clone, Copy)]
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
    SandyCite,
    SiltyClay,
    Clay,
}

/// Hydraulic properties for a soil texture class.
#[derive(Debug, Clone)]
pub struct SoilHydraulicProps {
    /// Field capacity (m³/m³) at -33 kPa
    pub field_capacity: f64,
    /// Wilting point (m³/m³) at -1500 kPa
    pub wilting_point: f64,
    /// Saturated hydraulic conductivity Ksat (mm/hr)
    pub ksat_mm_hr: f64,
    /// Porosity (m³/m³)
    pub porosity: f64,
}

impl SoilTexture {
    /// Typical hydraulic properties from USDA soil texture triangle.
    /// Values from Saxton & Rawls (2006) and USDA NRCS.
    pub fn hydraulic_properties(&self) -> SoilHydraulicProps {
        match self {
            SoilTexture::Sand => SoilHydraulicProps {
                field_capacity: 0.10,
                wilting_point: 0.05,
                ksat_mm_hr: 210.0,
                porosity: 0.43,
            },
            SoilTexture::LoamySand => SoilHydraulicProps {
                field_capacity: 0.12,
                wilting_point: 0.06,
                ksat_mm_hr: 61.0,
                porosity: 0.44,
            },
            SoilTexture::SandyLoam => SoilHydraulicProps {
                field_capacity: 0.18,
                wilting_point: 0.08,
                ksat_mm_hr: 26.0,
                porosity: 0.45,
            },
            SoilTexture::Loam => SoilHydraulicProps {
                field_capacity: 0.27,
                wilting_point: 0.12,
                ksat_mm_hr: 13.0,
                porosity: 0.46,
            },
            SoilTexture::SiltLoam => SoilHydraulicProps {
                field_capacity: 0.33,
                wilting_point: 0.13,
                ksat_mm_hr: 6.8,
                porosity: 0.47,
            },
            SoilTexture::Silt => SoilHydraulicProps {
                field_capacity: 0.33,
                wilting_point: 0.09,
                ksat_mm_hr: 6.8,
                porosity: 0.46,
            },
            SoilTexture::SandyClayLoam => SoilHydraulicProps {
                field_capacity: 0.26,
                wilting_point: 0.15,
                ksat_mm_hr: 4.3,
                porosity: 0.40,
            },
            SoilTexture::ClayLoam => SoilHydraulicProps {
                field_capacity: 0.32,
                wilting_point: 0.20,
                ksat_mm_hr: 2.3,
                porosity: 0.42,
            },
            SoilTexture::SiltyClayLoam => SoilHydraulicProps {
                field_capacity: 0.37,
                wilting_point: 0.22,
                ksat_mm_hr: 1.5,
                porosity: 0.43,
            },
            SoilTexture::SandyCite => SoilHydraulicProps {
                field_capacity: 0.30,
                wilting_point: 0.21,
                ksat_mm_hr: 1.2,
                porosity: 0.38,
            },
            SoilTexture::SiltyClay => SoilHydraulicProps {
                field_capacity: 0.37,
                wilting_point: 0.25,
                ksat_mm_hr: 0.9,
                porosity: 0.41,
            },
            SoilTexture::Clay => SoilHydraulicProps {
                field_capacity: 0.36,
                wilting_point: 0.25,
                ksat_mm_hr: 0.6,
                porosity: 0.38,
            },
        }
    }
}

/// Plant available water (PAW) in mm for a given soil depth.
/// PAW = (FC - WP) * depth_mm
pub fn plant_available_water(fc: f64, wp: f64, depth_mm: f64) -> f64 {
    (fc - wp) * depth_mm
}

/// Soil water deficit: how much water is needed to reach field capacity.
/// SWD = (FC - θv_current) * depth_mm
pub fn soil_water_deficit(fc: f64, current_theta: f64, depth_mm: f64) -> f64 {
    ((fc - current_theta).max(0.0)) * depth_mm
}

/// Management allowable depletion (MAD) for irrigation triggering.
/// When soil water drops below MAD fraction of PAW, irrigate.
/// Typical: MAD = 0.50 for most crops, 0.30 for sensitive crops.
pub fn irrigation_trigger(fc: f64, wp: f64, current_theta: f64, mad_fraction: f64) -> bool {
    let paw = fc - wp;
    let depletion = fc - current_theta;
    depletion > mad_fraction * paw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topp_equation() {
        // Topp (1980) Table 1: at ε=3 (dry soil), θv ≈ 0.031
        let theta = topp_equation(3.0);
        assert!((theta - 0.031).abs() < 0.01, "θv at ε=3: {}", theta);

        // At ε=20 (wet soil), θv ≈ 0.347
        let theta_wet = topp_equation(20.0);
        assert!((theta_wet - 0.347).abs() < 0.02, "θv at ε=20: {}", theta_wet);

        // At ε=1 (air), θv should be near 0 or slightly negative
        let theta_air = topp_equation(1.0);
        assert!(theta_air < 0.01, "θv at ε=1: {}", theta_air);
    }

    #[test]
    fn test_inverse_topp() {
        let theta = 0.30;
        let eps = inverse_topp(theta);
        let recovered = topp_equation(eps);
        assert!(
            (recovered - theta).abs() < 0.001,
            "Round-trip: {} vs {}",
            recovered,
            theta
        );
    }

    #[test]
    fn test_plant_available_water() {
        // Sandy loam: FC=0.18, WP=0.08, 300mm root zone
        let paw = plant_available_water(0.18, 0.08, 300.0);
        assert!((paw - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_irrigation_trigger() {
        // Silt loam: FC=0.33, WP=0.13, MAD=0.50
        // At θv=0.22 → depletion=0.11, PAW=0.20, 0.11 > 0.10 → trigger
        assert!(irrigation_trigger(0.33, 0.13, 0.22, 0.50));
        // At θv=0.30 → depletion=0.03 < 0.10 → no trigger
        assert!(!irrigation_trigger(0.33, 0.13, 0.30, 0.50));
    }
}
