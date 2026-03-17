// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unit tests for dual Kc module.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(clippy::float_cmp, reason = "test assertions on ET values")]

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
