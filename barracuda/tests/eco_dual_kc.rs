// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `eco::dual_kc` — FAO-56 dual crop coefficient.

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::eco::dual_kc::{
    etc_dual, evaporation_reduction, kc_max, mulched_ke, simulate_dual_kc,
    simulate_dual_kc_mulched, soil_evaporation_ke, total_evaporable_water, CoverCropType,
    DualKcInput, EvaporationLayerState, ResidueLevel,
};
use airspring_barracuda::eco::soil_moisture::SoilTexture;

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
        assert!(
            kcb.kcb_mid >= 0.5 && kcb.kcb_mid <= 1.3,
            "{crop:?}: mid range"
        );
        assert!(kcb.max_height_m > 0.0, "{crop:?}: height > 0");
    }
}

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
            precipitation: if i == 0 {
                10.0
            } else if i == 5 {
                8.0
            } else {
                0.0
            },
            irrigation: 0.0,
        })
        .collect();

    let (conv, _) = simulate_dual_kc(&inputs, 0.15, 1.20, 1.0, &state);
    let (notill, _) = simulate_dual_kc_mulched(&inputs, 0.15, 1.20, 1.0, 0.40, &state);

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
