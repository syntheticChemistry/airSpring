// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dual crop coefficient (Kcb + Ke) — FAO-56 Chapters 7 + 11.
//!
//! Separates crop evapotranspiration into transpiration (Kcb) and soil
//! evaporation (Ke) for precision irrigation scheduling:
//!
//! ```text
//! ETc = (Kcb × Ks + Ke) × ET₀     (Eq. 69)
//! ```
//!
//! This module provides:
//! - [`BasalCropCoefficients`] — Table 17 Kcb values per crop
//! - [`CoverCropType`] — Cover crop Kcb values for no-till systems
//! - [`EvaporationParams`] — Table 19 REW/TEW soil parameters
//! - Pure functions matching every FAO-56 equation (69, 71–73, 77)
//! - [`mulched_ke`] — FAO-56 Ch 11 mulch reduction on soil evaporation
//! - [`EvaporationLayerState`] — stateful daily simulation (with/without mulch)
//!
//! # References
//!
//! Allen RG, Pereira LS, Raes D, Smith M (1998)
//! FAO Irrigation and Drainage Paper 56, Chapters 7 + 11.
//!
//! Islam R, Reeder R (2014) No-till and conservation agriculture.
//! ISWCR 2(3): 176-186.

use crate::eco::crop::CropType;
use crate::eco::soil_moisture::SoilTexture;

// ── Table 17: Basal crop coefficients ────────────────────────────────

/// FAO-56 Table 17 basal crop coefficients for three growth stages.
#[derive(Debug, Clone, Copy)]
pub struct BasalCropCoefficients {
    /// Kcb during initial growth stage (bare soil dominates).
    pub kcb_ini: f64,
    /// Kcb during mid-season (full cover, transpiration dominates).
    pub kcb_mid: f64,
    /// Kcb during late season (senescence).
    pub kcb_end: f64,
    /// Maximum crop height (m) — needed for `Kc_max` calculation.
    pub max_height_m: f64,
}

impl CropType {
    /// FAO-56 Table 17 basal crop coefficients.
    ///
    /// These are lower than [`CropType::coefficients`] (Table 12) because
    /// they exclude soil evaporation — the Ke component accounts for it.
    #[must_use]
    pub const fn basal_coefficients(self) -> BasalCropCoefficients {
        match self {
            Self::Corn => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.15,
                kcb_end: 0.50,
                max_height_m: 2.0,
            },
            Self::Soybean => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.30,
                max_height_m: 0.75,
            },
            Self::WinterWheat => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 1.0,
            },
            Self::Alfalfa => BasalCropCoefficients {
                kcb_ini: 0.30,
                kcb_mid: 0.90,
                kcb_end: 0.85,
                max_height_m: 0.7,
            },
            Self::Tomato => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.70,
                max_height_m: 0.6,
            },
            Self::Potato => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.65,
                max_height_m: 0.6,
            },
            Self::SugarBeet => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.15,
                kcb_end: 0.90,
                max_height_m: 0.5,
            },
            Self::DryBean => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 0.4,
            },
            Self::Blueberry => BasalCropCoefficients {
                kcb_ini: 0.20,
                kcb_mid: 0.95,
                kcb_end: 0.55,
                max_height_m: 1.5,
            },
            Self::Turfgrass => BasalCropCoefficients {
                kcb_ini: 0.80,
                kcb_mid: 0.85,
                kcb_end: 0.85,
                max_height_m: 0.10,
            },
        }
    }
}

// ── Table 19: Soil evaporation parameters ────────────────────────────

/// Readily evaporable water (REW) and parameters for the evaporation layer.
#[derive(Debug, Clone, Copy)]
pub struct EvaporationParams {
    /// Field capacity of evaporation layer (m³/m³).
    pub theta_fc: f64,
    /// Wilting point of evaporation layer (m³/m³).
    pub theta_wp: f64,
    /// Readily evaporable water (mm) — stage 1 limit.
    pub rew_mm: f64,
}

impl SoilTexture {
    /// FAO-56 Table 19 evaporation parameters.
    ///
    /// REW values are typical midpoints for each USDA texture class.
    /// `θFC` and `θWP` are from Table 19 (may differ slightly from
    /// [`SoilTexture::hydraulic_properties`] which uses Saxton & Rawls).
    #[must_use]
    pub const fn evaporation_params(&self) -> EvaporationParams {
        match self {
            Self::Sand => EvaporationParams {
                theta_fc: 0.12,
                theta_wp: 0.04,
                rew_mm: 6.0,
            },
            Self::LoamySand => EvaporationParams {
                theta_fc: 0.16,
                theta_wp: 0.06,
                rew_mm: 6.0,
            },
            Self::SandyLoam => EvaporationParams {
                theta_fc: 0.23,
                theta_wp: 0.10,
                rew_mm: 8.0,
            },
            Self::Loam => EvaporationParams {
                theta_fc: 0.30,
                theta_wp: 0.15,
                rew_mm: 9.0,
            },
            Self::SiltLoam => EvaporationParams {
                theta_fc: 0.33,
                theta_wp: 0.13,
                rew_mm: 10.0,
            },
            Self::Silt => EvaporationParams {
                theta_fc: 0.36,
                theta_wp: 0.15,
                rew_mm: 10.0,
            },
            Self::SandyClayLoam => EvaporationParams {
                theta_fc: 0.33,
                theta_wp: 0.19,
                rew_mm: 8.0,
            },
            Self::ClayLoam => EvaporationParams {
                theta_fc: 0.36,
                theta_wp: 0.21,
                rew_mm: 9.0,
            },
            Self::SiltyClayLoam => EvaporationParams {
                theta_fc: 0.37,
                theta_wp: 0.21,
                rew_mm: 9.0,
            },
            Self::SandyClay => EvaporationParams {
                theta_fc: 0.36,
                theta_wp: 0.21,
                rew_mm: 8.0,
            },
            Self::SiltyClay => EvaporationParams {
                theta_fc: 0.40,
                theta_wp: 0.23,
                rew_mm: 10.0,
            },
            Self::Clay => EvaporationParams {
                theta_fc: 0.42,
                theta_wp: 0.25,
                rew_mm: 10.0,
            },
        }
    }
}

// ── Pure equation functions ──────────────────────────────────────────

/// FAO-56 Eq. 69: dual crop evapotranspiration.
///
/// `ETc` = (Kcb × Ks + Ke) × ET₀
#[must_use]
pub fn etc_dual(kcb: f64, ks: f64, ke: f64, et0: f64) -> f64 {
    kcb.mul_add(ks, ke) * et0
}

/// FAO-56 Eq. 72: upper limit on evapotranspiration coefficient.
///
/// `Kc_max` = max(1.2 + \[0.04(u₂ − 2) − 0.004(RHmin − 45)\] × (h/3)^0.3,
///              Kcb + 0.05)
#[must_use]
pub fn kc_max(u2: f64, rh_min: f64, h: f64, kcb: f64) -> f64 {
    let h_clamp = h.max(0.001);
    let climate_term =
        (0.04f64.mul_add(u2 - 2.0, -0.004 * (rh_min - 45.0))) * (h_clamp / 3.0).powf(0.3);
    (1.2 + climate_term).max(kcb + 0.05)
}

/// FAO-56 Eq. 73: total evaporable water from the surface soil layer.
///
/// TEW = 1000 × (`θFC` − 0.5 × `θWP`) × Ze (mm)
#[must_use]
pub fn total_evaporable_water(theta_fc: f64, theta_wp: f64, ze: f64) -> f64 {
    1000.0 * theta_wp.mul_add(-0.5, theta_fc) * ze
}

/// FAO-56 Eq. 72: evaporation reduction coefficient.
///
/// Kr = 1.0 when De ≤ REW (stage 1 drying), otherwise
/// Kr = (TEW − De) / (TEW − REW) clamped to \[0, 1\].
#[must_use]
pub fn evaporation_reduction(tew: f64, rew: f64, de: f64) -> f64 {
    if de <= rew {
        return 1.0;
    }
    if tew <= rew {
        return 0.0;
    }
    ((tew - de) / (tew - rew)).clamp(0.0, 1.0)
}

/// FAO-56 Eq. 71: soil evaporation coefficient.
///
/// `Ke` = min(Kr × (`Kc_max` − `Kcb`), `few` × `Kc_max`), bounded ≥ 0.
#[must_use]
pub fn soil_evaporation_ke(kr: f64, kcb: f64, kc_max_val: f64, few: f64) -> f64 {
    let ke = kr * (kc_max_val - kcb);
    ke.min(few * kc_max_val).max(0.0)
}

/// FAO-56 Eq. 77 (simplified): daily evaporation layer water balance.
///
/// De,i = De,i−1 − P − I + (Ke × ET₀)/few, clamped to \[0, TEW\].
#[must_use]
pub fn evaporation_layer_balance(
    de_prev: f64,
    precip: f64,
    irrig: f64,
    ke: f64,
    et0: f64,
    few: f64,
    tew: f64,
) -> f64 {
    let evap = if few > 0.001 { ke * et0 / few } else { 0.0 };
    (de_prev - precip - irrig + evap).clamp(0.0, tew)
}

// ── Stateful simulation ──────────────────────────────────────────────

/// State of the evaporation layer for multi-day simulation.
#[derive(Debug, Clone, Copy)]
pub struct EvaporationLayerState {
    pub de: f64,
    pub tew: f64,
    pub rew: f64,
}

/// Output of a single dual Kc simulation step.
#[derive(Debug, Clone, Copy)]
pub struct DualKcOutput {
    pub de: f64,
    pub kr: f64,
    pub ke: f64,
    pub etc: f64,
}

/// Daily input for dual Kc simulation.
#[derive(Debug, Clone, Copy)]
pub struct DualKcInput {
    pub et0: f64,
    pub precipitation: f64,
    pub irrigation: f64,
}

/// Run a multi-day dual Kc simulation.
///
/// Returns per-day outputs and the final evaporation layer state.
#[must_use]
pub fn simulate_dual_kc(
    inputs: &[DualKcInput],
    kcb: f64,
    kc_max_val: f64,
    few: f64,
    state: &EvaporationLayerState,
) -> (Vec<DualKcOutput>, EvaporationLayerState) {
    let mut de = state.de;
    let tew = state.tew;
    let rew = state.rew;
    let mut outputs = Vec::with_capacity(inputs.len());

    for inp in inputs {
        de = (de - inp.precipitation - inp.irrigation).clamp(0.0, tew);

        let kr = evaporation_reduction(tew, rew, de);
        let ke = soil_evaporation_ke(kr, kcb, kc_max_val, few);
        let etc = etc_dual(kcb, 1.0, ke, inp.et0);

        outputs.push(DualKcOutput { de, kr, ke, etc });

        de = evaporation_layer_balance(de, 0.0, 0.0, ke, inp.et0, few, tew);
    }

    (outputs, EvaporationLayerState { de, tew, rew })
}

// ── Cover crops (FAO-56 Ch 11 + literature) ─────────────────────────

/// Cover crop types for no-till systems.
///
/// Kcb values adapted from FAO-56 Table 17 and cover crop literature.
#[derive(Debug, Clone, Copy)]
pub enum CoverCropType {
    /// Winter cereal rye — dominant Midwest cover crop.
    CerealRye,
    /// Crimson clover — legume cover with moderate transpiration.
    CrimsonClover,
    /// Winter wheat used as cover crop (terminated early).
    WinterWheatCover,
    /// Hairy vetch — vining legume, good ground cover.
    HairyVetch,
    /// Daikon/tillage radish — winterkills, acts as green mulch.
    TillageRadish,
}

impl CoverCropType {
    /// Basal crop coefficients for cover crops.
    #[must_use]
    pub const fn basal_coefficients(self) -> BasalCropCoefficients {
        match self {
            Self::CerealRye => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 1.2,
            },
            Self::CrimsonClover => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 0.95,
                kcb_end: 0.30,
                max_height_m: 0.5,
            },
            Self::WinterWheatCover => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 1.0,
            },
            Self::HairyVetch => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 0.90,
                kcb_end: 0.25,
                max_height_m: 0.4,
            },
            Self::TillageRadish => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 0.85,
                kcb_end: 0.20,
                max_height_m: 0.3,
            },
        }
    }
}

// ── Mulch reduction (FAO-56 Ch 11) ──────────────────────────────────

/// No-till residue coverage levels and their mulch reduction factors.
///
/// The mulch factor reduces Ke: `Ke_mulch = Ke × mulch_factor`.
/// This accounts for surface residue blocking radiation from reaching
/// the soil, reducing stage 1 and stage 2 evaporation.
#[derive(Debug, Clone, Copy)]
pub enum ResidueLevel {
    /// Conventional tillage, bare soil.
    NoResidue,
    /// Light residue (<30% ground cover).
    Light,
    /// Moderate residue (30–60% ground cover).
    Moderate,
    /// Heavy residue (>60%, typical no-till).
    Heavy,
    /// Nearly complete cover (thick mulch).
    FullMulch,
}

impl ResidueLevel {
    /// Mulch reduction factor for soil evaporation.
    ///
    /// FAO-56 Chapter 11: surface residue reduces energy reaching the soil
    /// surface, reducing both stage 1 and stage 2 evaporation rates.
    #[must_use]
    pub const fn mulch_factor(self) -> f64 {
        match self {
            Self::NoResidue => 1.00,
            Self::Light => 0.80,
            Self::Moderate => 0.60,
            Self::Heavy => 0.40,
            Self::FullMulch => 0.25,
        }
    }
}

/// Soil evaporation with mulch reduction (FAO-56 Ch 11).
///
/// `Ke_mulch` = Ke × `mulch_factor`
#[must_use]
pub fn mulched_ke(kr: f64, kcb: f64, kc_max_val: f64, few: f64, mulch_factor: f64) -> f64 {
    soil_evaporation_ke(kr, kcb, kc_max_val, few) * mulch_factor
}

/// Run a multi-day dual Kc simulation with mulch reduction on Ke.
///
/// Identical to [`simulate_dual_kc`] but applies `mulch_factor` to reduce
/// soil evaporation, modeling no-till residue effects.
#[must_use]
pub fn simulate_dual_kc_mulched(
    inputs: &[DualKcInput],
    kcb: f64,
    kc_max_val: f64,
    few: f64,
    mulch_factor: f64,
    state: &EvaporationLayerState,
) -> (Vec<DualKcOutput>, EvaporationLayerState) {
    let mut de = state.de;
    let tew = state.tew;
    let rew = state.rew;
    let mut outputs = Vec::with_capacity(inputs.len());

    for inp in inputs {
        de = (de - inp.precipitation - inp.irrigation).clamp(0.0, tew);

        let kr = evaporation_reduction(tew, rew, de);
        let ke = mulched_ke(kr, kcb, kc_max_val, few, mulch_factor);
        let etc = etc_dual(kcb, 1.0, ke, inp.et0);

        outputs.push(DualKcOutput { de, kr, ke, etc });

        de = evaporation_layer_balance(de, 0.0, 0.0, ke, inp.et0, few, tew);
    }

    (outputs, EvaporationLayerState { de, tew, rew })
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::eco::crop::CropType;
    use crate::eco::soil_moisture::SoilTexture;

    #[test]
    fn test_etc_dual_basic() {
        // ETc = (Kcb*Ks + Ke)*ET0
        let etc = etc_dual(1.0, 1.0, 0.1, 5.0);
        assert!((etc - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_etc_dual_zero_ks_zero_transpiration() {
        let etc = etc_dual(1.0, 0.0, 0.1, 5.0);
        assert!((etc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_etc_dual_zero_et0_zero_etc() {
        let etc = etc_dual(1.0, 1.0, 0.1, 0.0);
        assert_eq!(etc, 0.0);
    }

    #[test]
    fn test_kc_max_standard_conditions() {
        let kc = kc_max(2.0, 45.0, 1.0, 1.0);
        assert!((kc - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_kc_max_extreme_wind_short_crop() {
        let kc_high_wind = kc_max(5.0, 45.0, 0.1, 0.5);
        let kc_low = kc_max(1.0, 45.0, 0.1, 0.5);
        assert!(kc_high_wind > kc_low);
    }

    #[test]
    fn test_total_evaporable_water_analytical() {
        // TEW = 1000 * (theta_fc - 0.5*theta_wp) * ze
        let tew = total_evaporable_water(0.30, 0.15, 0.1);
        let expected = 1000.0 * 0.5f64.mul_add(-0.15, 0.30) * 0.1;
        assert!((tew - expected).abs() < 1e-10);
    }

    #[test]
    fn test_evaporation_reduction_kr_one_when_de_le_rew() {
        let kr = evaporation_reduction(20.0, 10.0, 5.0);
        assert_eq!(kr, 1.0);
    }

    #[test]
    fn test_evaporation_reduction_kr_zero_when_tew_le_rew() {
        // When TEW <= REW, formula is degenerate; function returns 0.
        // Need de > rew to bypass the "de <= rew => 1.0" branch.
        let kr = evaporation_reduction(8.0, 10.0, 11.0);
        assert_eq!(kr, 0.0);
    }

    #[test]
    fn test_evaporation_reduction_intermediate() {
        let kr = evaporation_reduction(20.0, 10.0, 15.0);
        assert!(kr > 0.0 && kr < 1.0);
        assert!((kr - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_soil_evaporation_ke_clamped_to_zero() {
        let ke = soil_evaporation_ke(-0.1, 1.0, 1.2, 0.5);
        assert_eq!(ke, 0.0);
    }

    #[test]
    fn test_soil_evaporation_ke_min_of_two_terms() {
        let ke = soil_evaporation_ke(1.0, 0.5, 1.2, 0.3);
        let kr_term: f64 = 1.0 * (1.2 - 0.5);
        let few_term: f64 = 0.3 * 1.2;
        assert_eq!(ke, kr_term.min(few_term));
    }

    #[test]
    fn test_evaporation_layer_balance_clamp_to_tew() {
        let de = evaporation_layer_balance(25.0, 0.0, 0.0, 0.1, 5.0, 0.5, 20.0);
        assert!(de <= 20.0);
    }

    #[test]
    fn test_evaporation_layer_balance_clamp_to_zero() {
        let de = evaporation_layer_balance(5.0, 10.0, 0.0, 0.0, 5.0, 0.5, 20.0);
        assert!(de >= 0.0);
    }

    #[test]
    fn test_simulate_dual_kc_de_increases_dry_down() {
        let state = EvaporationLayerState {
            de: 0.0,
            tew: 20.0,
            rew: 10.0,
        };
        let inputs = vec![
            DualKcInput {
                et0: 5.0,
                precipitation: 0.0,
                irrigation: 0.0,
            },
            DualKcInput {
                et0: 5.0,
                precipitation: 0.0,
                irrigation: 0.0,
            },
        ];
        let (outputs, _) = simulate_dual_kc(&inputs, 1.0, 1.2, 0.5, &state);
        assert!(outputs[1].de > outputs[0].de);
    }

    #[test]
    fn test_simulate_dual_kc_mulched_reduces_etc() {
        let state = EvaporationLayerState {
            de: 0.0,
            tew: 20.0,
            rew: 10.0,
        };
        let inputs = vec![DualKcInput {
            et0: 5.0,
            precipitation: 0.0,
            irrigation: 0.0,
        }];
        let (bare, _) = simulate_dual_kc(&inputs, 1.0, 1.2, 0.5, &state);
        let (mulched, _) = simulate_dual_kc_mulched(&inputs, 1.0, 1.2, 0.5, 0.4, &state);
        assert!(mulched[0].etc < bare[0].etc);
    }

    #[test]
    fn test_mulched_ke_zero_when_mulch_factor_zero() {
        let ke = mulched_ke(1.0, 0.5, 1.2, 0.5, 0.0);
        assert_eq!(ke, 0.0);
    }

    #[test]
    fn test_mulched_ke_same_as_bare_when_mulch_factor_one() {
        let ke_bare = soil_evaporation_ke(1.0, 0.5, 1.2, 0.5);
        let ke_mulched = mulched_ke(1.0, 0.5, 1.2, 0.5, 1.0);
        assert!((ke_bare - ke_mulched).abs() < 1e-10);
    }

    #[test]
    fn test_basal_crop_coefficients_kcb_ini_lt_kcb_mid() {
        for crop in [
            CropType::Corn,
            CropType::Soybean,
            CropType::WinterWheat,
            CropType::Alfalfa,
            CropType::Tomato,
            CropType::Potato,
            CropType::SugarBeet,
            CropType::DryBean,
            CropType::Blueberry,
        ] {
            let bc = crop.basal_coefficients();
            assert!(
                bc.kcb_ini < bc.kcb_mid,
                "{:?}: kcb_ini {} should be < kcb_mid {}",
                crop,
                bc.kcb_ini,
                bc.kcb_mid
            );
        }
    }

    #[test]
    fn test_turfgrass_basal_coefficients() {
        let bc = CropType::Turfgrass.basal_coefficients();
        assert!(bc.kcb_ini >= 0.7 && bc.kcb_ini <= 0.9);
        assert!(bc.kcb_mid >= 0.8 && bc.kcb_mid <= 0.9);
    }

    #[test]
    fn test_evaporation_params_rew_positive() {
        for texture in [
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
        ] {
            let ep = texture.evaporation_params();
            assert!(ep.rew_mm > 0.0, "{texture:?} has rew > 0");
        }
    }

    #[test]
    fn test_cover_crop_basal_coefficients_reasonable_ranges() {
        use super::CoverCropType;
        for cover in [
            CoverCropType::CerealRye,
            CoverCropType::CrimsonClover,
            CoverCropType::WinterWheatCover,
            CoverCropType::HairyVetch,
            CoverCropType::TillageRadish,
        ] {
            let bc = cover.basal_coefficients();
            assert!(bc.kcb_ini >= 0.1 && bc.kcb_ini <= 0.2);
            assert!(bc.kcb_mid >= 0.8 && bc.kcb_mid <= 1.2);
        }
    }

    #[test]
    fn test_residue_level_mulch_factors_monotonically_decreasing() {
        use super::ResidueLevel;
        let factors = [
            ResidueLevel::NoResidue.mulch_factor(),
            ResidueLevel::Light.mulch_factor(),
            ResidueLevel::Moderate.mulch_factor(),
            ResidueLevel::Heavy.mulch_factor(),
            ResidueLevel::FullMulch.mulch_factor(),
        ];
        for i in 1..factors.len() {
            assert!(
                factors[i] <= factors[i - 1],
                "mulch factors should decrease: {factors:?}"
            );
        }
    }
}
