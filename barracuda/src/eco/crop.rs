//! Crop coefficient (Kc) database — FAO-56 Table 12.
//!
//! Standard crop coefficients for use in water balance scheduling.
//! Values from FAO-56 Table 12 (Allen et al. 1998, pp. 110–114).
//!
//! Each crop has three Kc values corresponding to growth stages:
//! - `kc_ini`: initial stage (planting to ~10% ground cover)
//! - `kc_mid`: mid-season (full cover to start of maturity)
//! - `kc_end`: late season (maturity to harvest)
//!
//! # Reference
//!
//! Allen RG, Pereira LS, Raes D, Smith M (1998)
//! FAO Irrigation and Drainage Paper 56, Table 12.

/// A crop's FAO-56 Kc values for three growth stages.
#[derive(Debug, Clone, Copy)]
pub struct CropCoefficients {
    /// Crop name (for display).
    pub name: &'static str,
    /// Kc during initial growth stage.
    pub kc_ini: f64,
    /// Kc during mid-season (peak water use).
    pub kc_mid: f64,
    /// Kc during late season.
    pub kc_end: f64,
    /// Typical root depth at maturity (m).
    pub root_depth_m: f64,
    /// Depletion fraction p (fraction of TAW before stress).
    pub depletion_fraction: f64,
}

/// Crop types with FAO-56 Table 12 coefficients.
///
/// Values are for sub-humid climate with `RH_min` ~45% and u₂ ~2 m/s.
/// Adjust `Kc_mid` and `Kc_end` for different climatic conditions per FAO-56 Eq. 62.
#[derive(Debug, Clone, Copy)]
pub enum CropType {
    /// Field corn (grain, dent).
    Corn,
    /// Soybean.
    Soybean,
    /// Winter wheat.
    WinterWheat,
    /// Alfalfa (hay, cuttings).
    Alfalfa,
    /// Tomato (fresh market).
    Tomato,
    /// Potato.
    Potato,
    /// Sugar beet.
    SugarBeet,
    /// Dry bean.
    DryBean,
    /// Blueberry (Dong et al. 2024 field site).
    Blueberry,
    /// Turfgrass (cool season).
    Turfgrass,
}

impl CropType {
    /// Return FAO-56 Table 12 crop coefficients for this crop.
    #[must_use]
    pub const fn coefficients(self) -> CropCoefficients {
        match self {
            Self::Corn => CropCoefficients {
                name: "Corn (grain)",
                kc_ini: 0.30,
                kc_mid: 1.20,
                kc_end: 0.60,
                root_depth_m: 0.90,
                depletion_fraction: 0.55,
            },
            Self::Soybean => CropCoefficients {
                name: "Soybean",
                kc_ini: 0.40,
                kc_mid: 1.15,
                kc_end: 0.50,
                root_depth_m: 0.60,
                depletion_fraction: 0.50,
            },
            Self::WinterWheat => CropCoefficients {
                name: "Winter wheat",
                kc_ini: 0.70,
                kc_mid: 1.15,
                kc_end: 0.25,
                root_depth_m: 1.50,
                depletion_fraction: 0.55,
            },
            Self::Alfalfa => CropCoefficients {
                name: "Alfalfa (hay)",
                kc_ini: 0.40,
                kc_mid: 1.20,
                kc_end: 1.15,
                root_depth_m: 1.00,
                depletion_fraction: 0.55,
            },
            Self::Tomato => CropCoefficients {
                name: "Tomato",
                kc_ini: 0.60,
                kc_mid: 1.15,
                kc_end: 0.80,
                root_depth_m: 0.60,
                depletion_fraction: 0.40,
            },
            Self::Potato => CropCoefficients {
                name: "Potato",
                kc_ini: 0.50,
                kc_mid: 1.15,
                kc_end: 0.75,
                root_depth_m: 0.40,
                depletion_fraction: 0.35,
            },
            Self::SugarBeet => CropCoefficients {
                name: "Sugar beet",
                kc_ini: 0.35,
                kc_mid: 1.20,
                kc_end: 0.70,
                root_depth_m: 0.70,
                depletion_fraction: 0.55,
            },
            Self::DryBean => CropCoefficients {
                name: "Dry bean",
                kc_ini: 0.40,
                kc_mid: 1.15,
                kc_end: 0.35,
                root_depth_m: 0.60,
                depletion_fraction: 0.45,
            },
            Self::Blueberry => CropCoefficients {
                name: "Blueberry",
                kc_ini: 0.30,
                kc_mid: 1.05,
                kc_end: 0.65,
                root_depth_m: 0.40,
                depletion_fraction: 0.50,
            },
            Self::Turfgrass => CropCoefficients {
                name: "Turfgrass (cool season)",
                kc_ini: 0.90,
                kc_mid: 0.95,
                kc_end: 0.95,
                root_depth_m: 0.30,
                depletion_fraction: 0.40,
            },
        }
    }
}

/// Adjust `Kc_mid` or `Kc_end` for climate (FAO-56 Eq. 62).
///
/// `Kc_adj` = `Kc_table` + \[0.04(u₂ − 2) − 0.004(`RHmin` − 45)\] × (h/3)^0.3
///
/// # Arguments
///
/// * `kc_table` — Tabulated Kc from FAO-56 Table 12.
/// * `u2` — Mean wind speed at 2 m during the stage (m/s).
/// * `rh_min` — Mean minimum relative humidity during the stage (%).
/// * `crop_height_m` — Crop height (m) during the stage.
#[must_use]
pub fn adjust_kc_for_climate(kc_table: f64, u2: f64, rh_min: f64, crop_height_m: f64) -> f64 {
    let adjustment =
        (0.04f64.mul_add(u2 - 2.0, -0.004 * (rh_min - 45.0))) * (crop_height_m / 3.0).powf(0.3);
    (kc_table + adjustment).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corn_coefficients() {
        let kc = CropType::Corn.coefficients();
        assert_eq!(kc.name, "Corn (grain)");
        assert!((kc.kc_ini - 0.30).abs() < f64::EPSILON);
        assert!((kc.kc_mid - 1.20).abs() < f64::EPSILON);
        assert!((kc.kc_end - 0.60).abs() < f64::EPSILON);
        assert!((kc.root_depth_m - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blueberry_matches_dong_2024() {
        // Blueberry Kc should reflect values used in Dong et al. (2024) field trial
        let kc = CropType::Blueberry.coefficients();
        assert!(kc.kc_mid > 0.9 && kc.kc_mid < 1.2, "kc_mid: {}", kc.kc_mid);
        assert!(kc.root_depth_m < 0.6, "Shallow roots: {}", kc.root_depth_m);
    }

    #[test]
    fn test_kc_mid_always_highest() {
        // For most crops, kc_mid should be the peak water use stage
        let crops = [
            CropType::Corn,
            CropType::Soybean,
            CropType::WinterWheat,
            CropType::Tomato,
            CropType::Potato,
            CropType::DryBean,
        ];
        for crop in crops {
            let kc = crop.coefficients();
            assert!(
                kc.kc_mid >= kc.kc_ini && kc.kc_mid >= kc.kc_end,
                "{}: kc_mid ({}) should be >= kc_ini ({}) and kc_end ({})",
                kc.name,
                kc.kc_mid,
                kc.kc_ini,
                kc.kc_end
            );
        }
    }

    #[test]
    fn test_adjust_kc_standard_conditions() {
        // At u₂=2, RHmin=45: adjustment should be zero
        let kc = adjust_kc_for_climate(1.20, 2.0, 45.0, 2.0);
        assert!((kc - 1.20).abs() < 0.001, "No adjustment expected: {kc}");
    }

    #[test]
    fn test_adjust_kc_windy_dry() {
        // Higher wind and lower humidity → higher Kc
        let kc = adjust_kc_for_climate(1.20, 4.0, 30.0, 2.0);
        assert!(kc > 1.20, "Windy+dry should increase Kc: {kc}");
    }

    #[test]
    fn test_adjust_kc_calm_humid() {
        // Lower wind and higher humidity → lower Kc
        let kc = adjust_kc_for_climate(1.20, 1.0, 70.0, 2.0);
        assert!(kc < 1.20, "Calm+humid should decrease Kc: {kc}");
    }

    #[test]
    fn test_all_crop_types_return_valid_values() {
        let all_crops = [
            CropType::Corn,
            CropType::Soybean,
            CropType::WinterWheat,
            CropType::Alfalfa,
            CropType::Tomato,
            CropType::Potato,
            CropType::SugarBeet,
            CropType::DryBean,
            CropType::Blueberry,
            CropType::Turfgrass,
        ];
        for crop in all_crops {
            let kc = crop.coefficients();
            assert!(kc.kc_ini > 0.0 && kc.kc_ini <= 1.5, "{}: kc_ini", kc.name);
            assert!(kc.kc_mid > 0.0 && kc.kc_mid <= 1.5, "{}: kc_mid", kc.name);
            assert!(kc.kc_end > 0.0 && kc.kc_end <= 1.5, "{}: kc_end", kc.name);
            assert!(
                kc.root_depth_m > 0.0 && kc.root_depth_m <= 2.0,
                "{}: root",
                kc.name
            );
            assert!(
                kc.depletion_fraction > 0.0 && kc.depletion_fraction <= 1.0,
                "{}: p",
                kc.name
            );
        }
    }
}
