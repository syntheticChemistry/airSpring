// SPDX-License-Identifier: AGPL-3.0-or-later
//! USDA soil texture classes and their typical hydraulic properties.
//!
//! Values sourced from Saxton & Rawls (2006) and USDA NRCS Soil Survey Manual.

/// USDA soil texture classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoilTexture {
    /// Coarse-textured; high drainage, low water retention.
    Sand,
    /// Sandy with minor silt/clay; moderate drainage.
    LoamySand,
    /// Sandy with appreciable silt/clay; good aeration and moisture.
    SandyLoam,
    /// Balanced sand–silt–clay; widely used for agriculture.
    Loam,
    /// Silt-dominated; high water retention, moderate drainage.
    SiltLoam,
    /// Fine silt; high water retention, prone to compaction.
    Silt,
    /// Sandy with significant clay; moderate drainage and retention.
    SandyClayLoam,
    /// Balanced clay–silt–sand; moderate drainage, good fertility.
    ClayLoam,
    /// Silt and clay dominated; high water retention, slow drainage.
    SiltyClayLoam,
    /// Sandy with high clay; sticky when wet, hard when dry.
    SandyClay,
    /// Silt and clay; very high water retention, poor drainage.
    SiltyClay,
    /// Fine-textured; very high water retention, low permeability.
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

const PROPS_SAND: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.10,
    wilting_point: 0.05,
    ksat_mm_hr: 210.0,
    porosity: 0.43,
};
const PROPS_LOAMY_SAND: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.12,
    wilting_point: 0.06,
    ksat_mm_hr: 61.0,
    porosity: 0.44,
};
const PROPS_SANDY_LOAM: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.18,
    wilting_point: 0.08,
    ksat_mm_hr: 26.0,
    porosity: 0.45,
};
const PROPS_LOAM: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.27,
    wilting_point: 0.12,
    ksat_mm_hr: 13.0,
    porosity: 0.46,
};
const PROPS_SILT_LOAM: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.33,
    wilting_point: 0.13,
    ksat_mm_hr: 6.8,
    porosity: 0.47,
};
const PROPS_SILT: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.33,
    wilting_point: 0.09,
    ksat_mm_hr: 6.8,
    porosity: 0.46,
};
const PROPS_SANDY_CLAY_LOAM: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.26,
    wilting_point: 0.15,
    ksat_mm_hr: 4.3,
    porosity: 0.40,
};
const PROPS_CLAY_LOAM: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.32,
    wilting_point: 0.20,
    ksat_mm_hr: 2.3,
    porosity: 0.42,
};
const PROPS_SILTY_CLAY_LOAM: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.37,
    wilting_point: 0.22,
    ksat_mm_hr: 1.5,
    porosity: 0.43,
};
const PROPS_SANDY_CLAY: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.30,
    wilting_point: 0.21,
    ksat_mm_hr: 1.2,
    porosity: 0.38,
};
const PROPS_SILTY_CLAY: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.37,
    wilting_point: 0.25,
    ksat_mm_hr: 0.9,
    porosity: 0.41,
};
const PROPS_CLAY: SoilHydraulicProps = SoilHydraulicProps {
    field_capacity: 0.36,
    wilting_point: 0.25,
    ksat_mm_hr: 0.6,
    porosity: 0.38,
};

impl SoilTexture {
    /// Typical hydraulic properties from USDA soil texture triangle.
    ///
    /// Values from Saxton & Rawls (2006) and USDA NRCS.
    #[must_use]
    pub const fn hydraulic_properties(&self) -> SoilHydraulicProps {
        match self {
            Self::Sand => PROPS_SAND,
            Self::LoamySand => PROPS_LOAMY_SAND,
            Self::SandyLoam => PROPS_SANDY_LOAM,
            Self::Loam => PROPS_LOAM,
            Self::SiltLoam => PROPS_SILT_LOAM,
            Self::Silt => PROPS_SILT,
            Self::SandyClayLoam => PROPS_SANDY_CLAY_LOAM,
            Self::ClayLoam => PROPS_CLAY_LOAM,
            Self::SiltyClayLoam => PROPS_SILTY_CLAY_LOAM,
            Self::SandyClay => PROPS_SANDY_CLAY,
            Self::SiltyClay => PROPS_SILTY_CLAY,
            Self::Clay => PROPS_CLAY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandy_loam_properties() {
        let props = SoilTexture::SandyLoam.hydraulic_properties();
        assert!((props.field_capacity - 0.18).abs() < f64::EPSILON);
        assert!((props.wilting_point - 0.08).abs() < f64::EPSILON);
    }

    #[test]
    fn sandy_clay_not_sandy_cite() {
        let props = SoilTexture::SandyClay.hydraulic_properties();
        assert!((props.field_capacity - 0.30).abs() < f64::EPSILON);
        assert!((props.wilting_point - 0.21).abs() < f64::EPSILON);
    }

    #[test]
    fn all_textures_valid_properties() {
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
    fn ksat_ordering_sand_to_clay() {
        let sand_ksat = SoilTexture::Sand.hydraulic_properties().ksat_mm_hr;
        let clay_ksat = SoilTexture::Clay.hydraulic_properties().ksat_mm_hr;
        assert!(
            sand_ksat > clay_ksat,
            "Sand Ksat {sand_ksat} > Clay Ksat {clay_ksat}"
        );
    }
}
