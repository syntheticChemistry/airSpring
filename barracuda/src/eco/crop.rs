// SPDX-License-Identifier: AGPL-3.0-or-later
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

/// Linear Kc interpolation within a growth stage (calendar-day based).
///
/// Wraps [`barracuda::stats::crop_coefficient`] (absorbed from airSpring
/// metalForge, S66 R-S66-002). Use for day-in-stage interpolation when
/// stage boundaries are defined by calendar days rather than GDD.
///
/// For GDD-based interpolation, use [`kc_from_gdd`] instead.
///
/// # Arguments
///
/// * `kc_prev` — Kc at the start of the stage
/// * `kc_next` — Kc at the end of the stage
/// * `day_in_stage` — Current day within the stage (0-indexed)
/// * `stage_length` — Total days in the stage
///
/// # Cross-Spring Provenance
///
/// Originally written in airSpring `metalForge`, absorbed into
/// `barracuda::stats::hydrology` (`BarraCuda` S66).
#[must_use]
pub fn crop_coefficient_stage(
    kc_prev: f64,
    kc_next: f64,
    day_in_stage: u32,
    stage_length: u32,
) -> f64 {
    barracuda::stats::crop_coefficient(kc_prev, kc_next, day_in_stage, stage_length)
}

// ── Growing Degree Days (GDD) ────────────────────────────────────────

/// Growing degree-day parameters for a crop.
#[derive(Debug, Clone)]
pub struct GddCropParams {
    /// Base temperature below which no development occurs (°C).
    pub tbase: f64,
    /// Ceiling temperature above which Tmax is clamped (°C).
    pub tceil: f64,
    /// Cumulative GDD at maturity.
    pub maturity_gdd: f64,
    /// GDD thresholds for stage boundaries (must start at 0).
    pub kc_stages_gdd: Vec<f64>,
    /// Kc value at each stage boundary (same length as `kc_stages_gdd`).
    pub kc_values: Vec<f64>,
}

impl CropType {
    /// Return GDD parameters for this crop.
    ///
    /// Base temperatures from `McMaster` & Wilhelm (1997); maturity GDD from
    /// Midwest US averages; Kc stages mapped to thermal time.
    #[must_use]
    pub fn gdd_params(self) -> GddCropParams {
        let kc = self.coefficients();
        match self {
            Self::Corn => GddCropParams {
                tbase: 10.0,
                tceil: 30.0,
                maturity_gdd: 2700.0,
                kc_stages_gdd: vec![0.0, 200.0, 800.0, 2200.0, 2700.0],
                kc_values: vec![kc.kc_ini, kc.kc_ini, kc.kc_mid, kc.kc_mid, kc.kc_end],
            },
            Self::Soybean => GddCropParams {
                tbase: 10.0,
                tceil: 30.0,
                maturity_gdd: 2600.0,
                kc_stages_gdd: vec![0.0, 200.0, 900.0, 2100.0, 2600.0],
                kc_values: vec![kc.kc_ini, kc.kc_ini, kc.kc_mid, kc.kc_mid, kc.kc_end],
            },
            Self::WinterWheat => GddCropParams {
                tbase: 0.0,
                tceil: 30.0,
                maturity_gdd: 2100.0,
                kc_stages_gdd: vec![0.0, 160.0, 700.0, 1700.0, 2100.0],
                kc_values: vec![kc.kc_ini, kc.kc_ini, kc.kc_mid, kc.kc_mid, kc.kc_end],
            },
            Self::Alfalfa => GddCropParams {
                tbase: 5.0,
                tceil: 30.0,
                maturity_gdd: 800.0,
                kc_stages_gdd: vec![0.0, 100.0, 300.0, 650.0, 800.0],
                kc_values: vec![kc.kc_ini, kc.kc_ini, kc.kc_mid, kc.kc_mid, kc.kc_end],
            },
            _ => GddCropParams {
                tbase: 10.0,
                tceil: 30.0,
                maturity_gdd: 2500.0,
                kc_stages_gdd: vec![0.0, 200.0, 800.0, 2000.0, 2500.0],
                kc_values: vec![kc.kc_ini, kc.kc_ini, kc.kc_mid, kc.kc_mid, kc.kc_end],
            },
        }
    }
}

/// Daily growing degree-days via the simple average method.
///
/// `GDD = max(0, (Tmax + Tmin)/2 − Tbase)`
///
/// `McMaster` & Wilhelm (1997) Method 1.
#[must_use]
pub fn gdd_avg(tmax: f64, tmin: f64, tbase: f64) -> f64 {
    (f64::midpoint(tmax, tmin) - tbase).max(0.0)
}

/// Daily growing degree-days via the clamped (modified) method.
///
/// `GDD = max(0, (min(Tmax, Tceil) + max(Tmin, Tbase))/2 − Tbase)`
///
/// Handles temperature extremes by clamping before averaging.
#[must_use]
pub fn gdd_clamp(tmax: f64, tmin: f64, tbase: f64, tceil: f64) -> f64 {
    let tmax_c = tmax.min(tceil);
    let tmin_c = tmin.max(tbase).min(tmax_c);
    (f64::midpoint(tmax_c, tmin_c) - tbase).max(0.0)
}

/// Accumulate growing degree-days over a season using the simple average method.
///
/// Returns a vector of cumulative GDD values, one per day.
///
/// # Errors
///
/// Returns `InvalidInput` if `daily_tmax` and `daily_tmin` have different lengths.
pub fn accumulated_gdd_avg(
    daily_tmax: &[f64],
    daily_tmin: &[f64],
    tbase: f64,
) -> crate::error::Result<Vec<f64>> {
    if daily_tmax.len() != daily_tmin.len() {
        return Err(crate::error::AirSpringError::InvalidInput(format!(
            "tmax length {} != tmin length {}",
            daily_tmax.len(),
            daily_tmin.len()
        )));
    }
    let mut cum = Vec::with_capacity(daily_tmax.len());
    let mut total = 0.0;
    for (&tx, &tn) in daily_tmax.iter().zip(daily_tmin) {
        total += gdd_avg(tx, tn, tbase);
        cum.push(total);
    }
    Ok(cum)
}

/// Accumulate growing degree-days over a season using the clamped method.
///
/// # Errors
///
/// Returns `InvalidInput` if `daily_tmax` and `daily_tmin` have different lengths.
pub fn accumulated_gdd_clamp(
    daily_tmax: &[f64],
    daily_tmin: &[f64],
    tbase: f64,
    tceil: f64,
) -> crate::error::Result<Vec<f64>> {
    if daily_tmax.len() != daily_tmin.len() {
        return Err(crate::error::AirSpringError::InvalidInput(format!(
            "tmax length {} != tmin length {}",
            daily_tmax.len(),
            daily_tmin.len()
        )));
    }
    let mut cum = Vec::with_capacity(daily_tmax.len());
    let mut total = 0.0;
    for (&tx, &tn) in daily_tmax.iter().zip(daily_tmin) {
        total += gdd_clamp(tx, tn, tbase, tceil);
        cum.push(total);
    }
    Ok(cum)
}

/// Interpolate crop coefficient from cumulative GDD using stage thresholds.
///
/// Linearly interpolates between Kc values defined at GDD breakpoints.
///
/// # Errors
///
/// Returns `InvalidInput` if `stages_gdd` and `kc_values` have different lengths,
/// or if `kc_values` is empty.
pub fn kc_from_gdd(
    cum_gdd: f64,
    stages_gdd: &[f64],
    kc_values: &[f64],
) -> crate::error::Result<f64> {
    if kc_values.is_empty() {
        return Err(crate::error::AirSpringError::InvalidInput(
            "kc_values must not be empty".to_string(),
        ));
    }
    if stages_gdd.len() != kc_values.len() {
        return Err(crate::error::AirSpringError::InvalidInput(format!(
            "stages_gdd length {} != kc_values length {}",
            stages_gdd.len(),
            kc_values.len()
        )));
    }
    for i in 0..stages_gdd.len().saturating_sub(1) {
        if cum_gdd <= stages_gdd[i + 1] {
            let span = stages_gdd[i + 1] - stages_gdd[i];
            if span <= 0.0 {
                return Ok(kc_values[i]);
            }
            let frac = (cum_gdd - stages_gdd[i]) / span;
            return Ok(frac.mul_add(kc_values[i + 1] - kc_values[i], kc_values[i]));
        }
    }
    // SAFETY: `kc_values.is_empty()` checked above — `last()` is always `Some`.
    Ok(kc_values[kc_values.len() - 1])
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code may use unwrap for clarity")]
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
    fn test_crop_coefficient_stage_midpoint() {
        let kc = crop_coefficient_stage(0.30, 1.20, 50, 100);
        assert!((kc - 0.75).abs() < 1e-10, "Midpoint should be 0.75: {kc}");
    }

    #[test]
    fn test_crop_coefficient_stage_boundaries() {
        let start = crop_coefficient_stage(0.30, 1.20, 0, 100);
        assert!((start - 0.30).abs() < 1e-10, "Start: {start}");
        let end = crop_coefficient_stage(0.30, 1.20, 100, 100);
        assert!((end - 1.20).abs() < 1e-10, "End: {end}");
    }

    #[test]
    fn test_crop_coefficient_stage_zero_length() {
        let kc = crop_coefficient_stage(0.30, 1.20, 5, 0);
        assert!(
            (kc - 0.30).abs() < 1e-10,
            "Zero-length returns kc_prev: {kc}"
        );
    }

    #[test]
    fn test_gdd_avg_basic() {
        assert!((gdd_avg(30.0, 20.0, 10.0) - 15.0).abs() < 1e-10);
        assert!(gdd_avg(8.0, 2.0, 10.0).abs() < f64::EPSILON);
        assert!((gdd_avg(30.0, 10.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_gdd_clamp_extremes() {
        assert!((gdd_clamp(40.0, 20.0, 10.0, 30.0) - 15.0).abs() < 1e-10);
        assert!((gdd_clamp(20.0, 5.0, 10.0, 30.0) - 5.0).abs() < 1e-10);
        assert!((gdd_clamp(40.0, 0.0, 10.0, 30.0) - 10.0).abs() < 1e-10);
        assert!(gdd_clamp(8.0, 2.0, 10.0, 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_accumulated_gdd_constant() {
        let tmax = vec![30.0; 100];
        let tmin = vec![20.0; 100];
        let cum = accumulated_gdd_avg(&tmax, &tmin, 10.0).unwrap();
        assert!((cum[99] - 1500.0).abs() < 1e-8);
    }

    #[test]
    fn test_accumulated_gdd_length_mismatch() {
        let result = accumulated_gdd_avg(&[30.0; 5], &[20.0; 3], 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_kc_from_gdd_corn() {
        let stages = vec![0.0, 200.0, 800.0, 2200.0, 2700.0];
        let kc_vals = vec![0.30, 0.30, 1.20, 1.20, 0.60];
        assert!((kc_from_gdd(0.0, &stages, &kc_vals).unwrap() - 0.30).abs() < 0.01);
        assert!((kc_from_gdd(1500.0, &stages, &kc_vals).unwrap() - 1.20).abs() < 0.01);
        assert!((kc_from_gdd(2700.0, &stages, &kc_vals).unwrap() - 0.60).abs() < 0.01);
    }

    #[test]
    fn test_kc_from_gdd_length_mismatch() {
        let result = kc_from_gdd(100.0, &[0.0, 200.0], &[0.3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gdd_monotonic() {
        let cum = accumulated_gdd_avg(&[25.0; 50], &[15.0; 50], 10.0).unwrap();
        for i in 0..cum.len() - 1 {
            assert!(cum[i] <= cum[i + 1]);
        }
    }

    #[test]
    fn test_corn_gdd_params() {
        let params = CropType::Corn.gdd_params();
        assert!((params.tbase - 10.0).abs() < f64::EPSILON);
        assert!((params.maturity_gdd - 2700.0).abs() < f64::EPSILON);
        assert_eq!(params.kc_stages_gdd.len(), params.kc_values.len());
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
