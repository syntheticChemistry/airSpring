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
    /// Maximum crop height (m) — needed for Kc_max calculation.
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
    /// θFC and θWP are from Table 19 (may differ slightly from
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
/// ETc = (Kcb × Ks + Ke) × ET₀
#[must_use]
pub fn etc_dual(kcb: f64, ks: f64, ke: f64, et0: f64) -> f64 {
    kcb.mul_add(ks, ke) * et0
}

/// FAO-56 Eq. 72: upper limit on evapotranspiration coefficient.
///
/// Kc_max = max(1.2 + \[0.04(u₂ − 2) − 0.004(RHmin − 45)\] × (h/3)^0.3,
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
/// TEW = 1000 × (θFC − 0.5 × θWP) × Ze (mm)
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
/// Ke = min(Kr × (Kc_max − Kcb), few × Kc_max), bounded ≥ 0.
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
/// Ke_mulch = Ke × mulch_factor
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
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_etc_dual_eq69() {
        assert!((etc_dual(1.15, 1.0, 0.0, 5.0) - 5.75).abs() < TOL);
        assert!((etc_dual(1.10, 1.0, 0.10, 4.5) - 5.40).abs() < TOL);
        assert!((etc_dual(1.15, 0.6, 0.15, 6.0) - 5.04).abs() < TOL);
        assert!((etc_dual(0.15, 1.0, 0.85, 3.0) - 3.0).abs() < TOL);
    }

    #[test]
    fn test_kc_max_standard_climate() {
        let val = kc_max(2.0, 45.0, 2.0, 1.15);
        assert!((val - 1.20).abs() < 0.01, "standard: {val}");
    }

    #[test]
    fn test_kc_max_windy_dry() {
        let val = kc_max(4.0, 25.0, 2.0, 1.15);
        assert!(val > 1.2, "windy+dry should increase: {val}");
        assert!((val - 1.3417).abs() < 0.01, "expected ~1.34: {val}");
    }

    #[test]
    fn test_kc_max_calm_humid() {
        let val = kc_max(1.0, 70.0, 0.4, 1.10);
        assert!((val - 1.15).abs() < 0.01, "Kcb+0.05 floor: {val}");
    }

    #[test]
    fn test_tew_sandy_loam() {
        let tew = total_evaporable_water(0.23, 0.10, 0.10);
        assert!((tew - 18.0).abs() < TOL);
    }

    #[test]
    fn test_tew_loam() {
        let tew = total_evaporable_water(0.30, 0.15, 0.10);
        assert!((tew - 22.5).abs() < TOL);
    }

    #[test]
    fn test_tew_clay() {
        let tew = total_evaporable_water(0.42, 0.25, 0.10);
        assert!((tew - 29.5).abs() < TOL);
    }

    #[test]
    fn test_kr_stage1() {
        assert!((evaporation_reduction(22.5, 9.0, 0.0) - 1.0).abs() < TOL);
        assert!((evaporation_reduction(22.5, 9.0, 9.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_kr_stage2() {
        assert!((evaporation_reduction(22.5, 9.0, 15.75) - 0.5).abs() < TOL);
        assert!((evaporation_reduction(22.5, 9.0, 21.15) - 0.1).abs() < TOL);
        assert!(evaporation_reduction(22.5, 9.0, 22.5).abs() < TOL);
    }

    #[test]
    fn test_ke_boundaries() {
        assert!(soil_evaporation_ke(0.0, 1.15, 1.20, 1.0).abs() < TOL);
        assert!((soil_evaporation_ke(1.0, 0.15, 1.20, 1.0) - 1.05).abs() < TOL);
        assert!((soil_evaporation_ke(1.0, 0.15, 1.20, 0.3) - 0.36).abs() < TOL);
        assert!((soil_evaporation_ke(1.0, 1.15, 1.20, 0.05) - 0.05).abs() < TOL);
    }

    #[test]
    fn test_kcb_always_less_than_kc() {
        let crops = [
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
        for crop in crops {
            let kc = crop.coefficients();
            let kcb = crop.basal_coefficients();
            let diff = kc.kc_mid - kcb.kcb_mid;
            // Gap is typically 0.05–0.10 for full-cover crops, but alfalfa
            // (frequent cutting, exposed soil) can reach 0.30.
            assert!(
                (0.0..=0.35).contains(&diff),
                "{}: Kc_mid({}) - Kcb_mid({}) = {diff}",
                kc.name,
                kc.kc_mid,
                kcb.kcb_mid
            );
        }
    }

    #[test]
    fn test_kcb_ini_less_than_mid() {
        let crops = [
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
        for crop in crops {
            let kcb = crop.basal_coefficients();
            assert!(
                kcb.kcb_ini < kcb.kcb_mid,
                "{crop:?}: kcb_ini ({}) < kcb_mid ({})",
                kcb.kcb_ini,
                kcb.kcb_mid
            );
        }
    }

    #[test]
    fn test_tew_exceeds_rew_all_soils() {
        let soils = [
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
        for soil in soils {
            let ep = soil.evaporation_params();
            let tew = total_evaporable_water(ep.theta_fc, ep.theta_wp, 0.10);
            assert!(
                tew > ep.rew_mm,
                "{soil:?}: TEW ({tew:.1}) > REW ({:.1})",
                ep.rew_mm
            );
        }
    }

    #[test]
    fn test_bare_soil_drydown_simulation() {
        let state = EvaporationLayerState {
            de: 0.0,
            tew: 18.0,
            rew: 8.0,
        };
        let inputs: Vec<DualKcInput> = [5.0, 5.5, 5.0, 4.8, 5.2, 4.5, 5.0]
            .iter()
            .enumerate()
            .map(|(i, &et0)| DualKcInput {
                et0,
                precipitation: if i == 0 { 25.0 } else { 0.0 },
                irrigation: 0.0,
            })
            .collect();

        let (outputs, final_state) = simulate_dual_kc(&inputs, 0.15, 1.20, 1.0, &state);

        assert!((outputs[0].kr - 1.0).abs() < TOL, "day1 Kr=1");
        assert!(outputs[0].kr >= outputs[6].kr, "Kr declines");
        assert!(outputs[0].de <= outputs[6].de, "De increases");
        assert!(outputs[0].ke >= outputs[6].ke, "Ke declines");
        assert!(final_state.de <= state.tew, "De <= TEW");

        let total_etc: f64 = outputs.iter().map(|o| o.etc).sum();
        assert!(total_etc > 0.0, "total ETc > 0");
    }

    #[test]
    fn test_corn_mid_season_simulation() {
        let state = EvaporationLayerState {
            de: 0.0,
            tew: 22.5,
            rew: 9.0,
        };
        let et0_vals = [5.5, 6.0, 5.8, 5.2, 5.0];
        let precip_vals = [0.0, 0.0, 12.0, 0.0, 0.0];
        let inputs: Vec<DualKcInput> = et0_vals
            .iter()
            .zip(precip_vals.iter())
            .map(|(&et0, &precip)| DualKcInput {
                et0,
                precipitation: precip,
                irrigation: 0.0,
            })
            .collect();

        let (outputs, _) = simulate_dual_kc(&inputs, 1.15, 1.20, 0.05, &state);

        for (i, (out, &et0)) in outputs.iter().zip(et0_vals.iter()).enumerate() {
            let ratio = out.etc / et0;
            assert!(
                (ratio - 1.15).abs() < 0.10,
                "day {}: ETc/ET₀ ({ratio:.3}) ≈ Kcb (1.15)",
                i + 1
            );
        }
    }

    // ── Cover crop tests ────────────────────────────────────────

    #[test]
    fn test_cover_crop_kcb_reasonable() {
        let crops = [
            CoverCropType::CerealRye,
            CoverCropType::CrimsonClover,
            CoverCropType::WinterWheatCover,
            CoverCropType::HairyVetch,
            CoverCropType::TillageRadish,
        ];
        for crop in crops {
            let kcb = crop.basal_coefficients();
            assert!(kcb.kcb_ini < kcb.kcb_mid, "{crop:?}: ini < mid");
            assert!(kcb.kcb_mid >= 0.5 && kcb.kcb_mid <= 1.3, "{crop:?}: mid range");
            assert!(kcb.max_height_m > 0.0, "{crop:?}: height > 0");
        }
    }

    // ── Mulch tests ─────────────────────────────────────────────

    #[test]
    fn test_mulch_factor_ordering() {
        let levels = [
            ResidueLevel::NoResidue,
            ResidueLevel::Light,
            ResidueLevel::Moderate,
            ResidueLevel::Heavy,
            ResidueLevel::FullMulch,
        ];
        for pair in levels.windows(2) {
            assert!(
                pair[0].mulch_factor() > pair[1].mulch_factor(),
                "{:?} ({}) > {:?} ({})",
                pair[0],
                pair[0].mulch_factor(),
                pair[1],
                pair[1].mulch_factor()
            );
        }
    }

    #[test]
    fn test_mulched_ke_bare_soil() {
        let ke = mulched_ke(1.0, 0.15, 1.20, 1.0, 1.0);
        assert!((ke - 1.05).abs() < TOL, "no mulch = bare soil: {ke}");
    }

    #[test]
    fn test_mulched_ke_heavy_residue() {
        let ke = mulched_ke(1.0, 0.15, 1.20, 1.0, 0.40);
        assert!((ke - 0.42).abs() < TOL, "heavy residue: {ke}");
    }

    #[test]
    fn test_mulched_ke_full_mulch() {
        let ke = mulched_ke(1.0, 0.15, 1.20, 1.0, 0.25);
        assert!((ke - 0.2625).abs() < TOL, "full mulch: {ke}");
    }

    #[test]
    fn test_notill_saves_water_vs_conventional() {
        let state = EvaporationLayerState {
            de: 0.0,
            tew: 22.5,
            rew: 9.0,
        };
        let inputs: Vec<DualKcInput> = [4.0, 4.5, 4.2, 5.0, 5.5, 5.0, 4.8]
            .iter()
            .enumerate()
            .map(|(i, &et0)| DualKcInput {
                et0,
                precipitation: if i == 0 { 10.0 } else if i == 5 { 8.0 } else { 0.0 },
                irrigation: 0.0,
            })
            .collect();

        let (conv, _) = simulate_dual_kc(&inputs, 0.15, 1.20, 1.0, &state);
        let (notill, _) =
            simulate_dual_kc_mulched(&inputs, 0.15, 1.20, 1.0, 0.40, &state);

        let conv_et: f64 = conv.iter().map(|o| o.etc).sum();
        let notill_et: f64 = notill.iter().map(|o| o.etc).sum();

        assert!(
            notill_et < conv_et,
            "no-till ({notill_et:.2}) < conventional ({conv_et:.2})"
        );

        let savings_pct = 100.0 * (1.0 - notill_et / conv_et);
        assert!(
            (5.0..=50.0).contains(&savings_pct),
            "ET savings {savings_pct:.1}% in [5, 50]"
        );
    }
}
