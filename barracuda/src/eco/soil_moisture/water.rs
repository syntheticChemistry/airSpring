// SPDX-License-Identifier: AGPL-3.0-or-later
//! Water availability calculations: PAW, SWD, and irrigation triggering.

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
    use super::super::texture::SoilTexture;
    use super::*;

    #[test]
    fn paw_sandy_loam() {
        let paw = plant_available_water(0.18, 0.08, 300.0);
        assert!((paw - 30.0).abs() < 0.01);
    }

    #[test]
    fn paw_clay() {
        let p = SoilTexture::Clay.hydraulic_properties();
        let paw = plant_available_water(p.field_capacity, p.wilting_point, 500.0);
        assert!((paw - 55.0).abs() < 0.1, "PAW={paw}");
    }

    #[test]
    fn swd_basic() {
        let swd = soil_water_deficit(0.33, 0.25, 600.0);
        assert!((swd - 48.0).abs() < 0.01);
    }

    #[test]
    fn swd_above_fc() {
        let swd = soil_water_deficit(0.33, 0.40, 600.0);
        assert!((swd - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn swd_zero_depth() {
        assert!((soil_water_deficit(0.33, 0.25, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn trigger_basic() {
        assert!(irrigation_trigger(0.33, 0.13, 0.22, 0.50));
        assert!(!irrigation_trigger(0.33, 0.13, 0.30, 0.50));
        assert!(!irrigation_trigger(0.33, 0.13, 0.33, 0.50));
    }

    #[test]
    fn trigger_at_boundaries() {
        let fc = 0.30;
        let wp = 0.10;
        let paw = fc - wp;
        let mad = 0.5;
        let mad_depletion = mad * paw;
        let theta_at_mad = fc - mad_depletion;
        assert!(!irrigation_trigger(fc, wp, theta_at_mad, mad));
        assert!(irrigation_trigger(fc, wp, theta_at_mad - 0.001, mad));
    }
}
