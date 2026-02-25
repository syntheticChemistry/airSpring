//! Validate FAO-56 Chapter 7 dual crop coefficient against Python control.
//!
//! Benchmark source: `control/dual_kc/benchmark_dual_kc.json`
//! Python baseline: `control/dual_kc/dual_crop_coefficient.py` (63/63 PASS)
//!
//! Validates:
//! 1. Eq. 69 — ETc = (Kcb × Ks + Ke) × ET₀
//! 2. Eq. 72 — Kc_max upper limit
//! 3. Eq. 73 — TEW total evaporable water
//! 4. Eq. 72 — Kr evaporation reduction coefficient
//! 5. Ke boundary conditions
//! 6. Table 17 Kcb vs Table 12 Kc consistency
//! 7. Table 19 TEW > REW for all USDA soils
//! 8. Multi-day simulations (bare soil drydown, corn mid-season)

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::eco::dual_kc::{self, DualKcInput, EvaporationLayerState};
use airspring_barracuda::eco::soil_moisture::SoilTexture;
use airspring_barracuda::validation::{self, json_f64, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/dual_kc/benchmark_dual_kc.json");

fn validate_eq69(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Eq. 69: ETc = (Kcb × Ks + Ke) × ET₀");

    let cases = bench["equations"]["eq_69"]["test_cases"]
        .as_array()
        .expect("eq_69 test_cases");

    for tc in cases {
        let label = tc["label"].as_str().unwrap();
        let kcb = tc["kcb"].as_f64().unwrap();
        let ks = tc["ks"].as_f64().unwrap();
        let ke = tc["ke"].as_f64().unwrap();
        let et0 = tc["et0"].as_f64().unwrap();
        let expected = tc["expected_etc"].as_f64().unwrap();

        let result = dual_kc::etc_dual(kcb, ks, ke, et0);
        v.check_abs(label, result, expected, 1e-6);
    }
}

fn validate_kc_max(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Eq. 72: Kc_max");

    let cases = bench["equations"]["eq_71_kc_max"]["test_cases"]
        .as_array()
        .expect("kc_max test_cases");

    for tc in cases {
        let label = tc["label"].as_str().unwrap();
        let u2 = tc["u2"].as_f64().unwrap();
        let rh_min = tc["rh_min"].as_f64().unwrap();
        let h = tc["h"].as_f64().unwrap();
        let kcb = tc["kcb"].as_f64().unwrap();
        let expected = tc["expected_kc_max"].as_f64().unwrap();

        let result = dual_kc::kc_max(u2, rh_min, h, kcb);
        v.check_abs(label, result, expected, 0.01);
    }
}

fn validate_tew(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Eq. 73: TEW = 1000 × (θFC − 0.5×θWP) × Ze");

    let cases = bench["equations"]["eq_73_tew"]["test_cases"]
        .as_array()
        .expect("tew test_cases");

    for tc in cases {
        let label = tc["label"].as_str().unwrap();
        let theta_fc = tc["theta_fc"].as_f64().unwrap();
        let theta_wp = tc["theta_wp"].as_f64().unwrap();
        let ze = tc["ze_m"].as_f64().unwrap();
        let expected = tc["expected_tew"].as_f64().unwrap();

        let result = dual_kc::total_evaporable_water(theta_fc, theta_wp, ze);
        v.check_abs(label, result, expected, 1e-6);
    }
}

fn validate_kr(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Eq. 72: Kr evaporation reduction");

    let cases = bench["equations"]["eq_72_kr"]["test_cases"]
        .as_array()
        .expect("kr test_cases");

    for tc in cases {
        let label = tc["label"].as_str().unwrap();
        let tew = tc["tew"].as_f64().unwrap();
        let rew = tc["rew"].as_f64().unwrap();
        let de = tc["de"].as_f64().unwrap();
        let expected = tc["expected_kr"].as_f64().unwrap();

        let result = dual_kc::evaporation_reduction(tew, rew, de);
        v.check_abs(label, result, expected, 1e-6);
    }
}

fn validate_ke_boundaries(v: &mut ValidationHarness) {
    println!();
    validation::section("Ke boundary conditions");

    v.check_abs(
        "Ke=0 when Kr=0 (dry)",
        dual_kc::soil_evaporation_ke(0.0, 1.15, 1.20, 1.0),
        0.0,
        1e-10,
    );
    v.check_abs(
        "Ke=1.05 bare wet soil",
        dual_kc::soil_evaporation_ke(1.0, 0.15, 1.20, 1.0),
        1.05,
        1e-6,
    );
    v.check_abs(
        "Ke limited by few×Kc_max",
        dual_kc::soil_evaporation_ke(1.0, 0.15, 1.20, 0.3),
        0.36,
        1e-6,
    );
    v.check_abs(
        "Ke small under full cover",
        dual_kc::soil_evaporation_ke(1.0, 1.15, 1.20, 0.05),
        0.05,
        1e-6,
    );
}

fn validate_kcb_vs_kc(v: &mut ValidationHarness) {
    println!();
    validation::section("Table 17 vs Table 12: Kcb + evaporation ≈ Kc");

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
        v.check_bool(
            &format!(
                "{}: Kc_mid({:.2}) - Kcb_mid({:.2}) = {diff:.2} in [0, 0.35]",
                kc.name, kc.kc_mid, kcb.kcb_mid
            ),
            (0.0..=0.35).contains(&diff),
        );
    }
}

fn validate_tew_vs_rew(v: &mut ValidationHarness) {
    println!();
    validation::section("Table 19: TEW > REW for all soil types");

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
        let tew = dual_kc::total_evaporable_water(ep.theta_fc, ep.theta_wp, 0.10);
        v.check_bool(
            &format!("{soil:?}: TEW ({tew:.1}) > REW ({:.1})", ep.rew_mm),
            tew > ep.rew_mm,
        );
    }
}

fn validate_bare_soil_drydown(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Scenario: Bare soil drydown (7 days)");

    let scenario = &bench["validation_scenarios"]["bare_soil_drydown"];
    let et0_daily: Vec<f64> = scenario["et0_daily"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let precip_daily: Vec<f64> = scenario["precip_daily"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let kcb = json_f64(scenario, &["kcb"]).unwrap();
    let kc_max_val = json_f64(scenario, &["kc_max"]).unwrap();
    let few = json_f64(scenario, &["few"]).unwrap();
    let tew = json_f64(scenario, &["tew"]).unwrap();
    let rew = json_f64(scenario, &["rew"]).unwrap();

    let inputs: Vec<DualKcInput> = et0_daily
        .iter()
        .zip(precip_daily.iter())
        .map(|(&et0, &precip)| DualKcInput {
            et0,
            precipitation: precip,
            irrigation: 0.0,
        })
        .collect();

    let state = EvaporationLayerState { de: 0.0, tew, rew };
    let (outputs, final_state) = dual_kc::simulate_dual_kc(&inputs, kcb, kc_max_val, few, &state);

    v.check_bool("Day 1 Kr=1.0 (stage 1)", (outputs[0].kr - 1.0).abs() < 1e-10);
    v.check_bool("Kr declines", outputs[0].kr >= outputs[6].kr);
    v.check_bool("De increases", outputs[0].de <= outputs[6].de);
    v.check_bool("Ke declines", outputs[0].ke >= outputs[6].ke);

    let total_etc: f64 = outputs.iter().map(|o| o.etc).sum();
    v.check_bool(&format!("Total ETc > 0: {total_etc:.2} mm"), total_etc > 0.0);
    v.check_bool(
        &format!("Final De <= TEW: {:.2} <= {tew}", final_state.de),
        final_state.de <= tew,
    );

    // Cross-validate against Python daily values (from Python simulation output)
    let py_kr = [1.0, 1.0, 0.6975, 0.3313, 0.1643, 0.0746, 0.0394];
    for (i, (&py, out)) in py_kr.iter().zip(outputs.iter()).enumerate() {
        v.check_abs(&format!("Day {} Kr vs Python", i + 1), out.kr, py, 0.001);
    }
}

fn validate_corn_mid_season(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Scenario: Corn mid-season (5 days, full cover)");

    let scenario = &bench["validation_scenarios"]["corn_mid_season"];
    let et0_daily: Vec<f64> = scenario["et0_daily"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let precip_daily: Vec<f64> = scenario["precip_daily"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let kcb = json_f64(scenario, &["kcb"]).unwrap();
    let kc_max_val = json_f64(scenario, &["kc_max"]).unwrap();
    let few = json_f64(scenario, &["few"]).unwrap();
    let tew = json_f64(scenario, &["tew"]).unwrap();
    let rew = json_f64(scenario, &["rew"]).unwrap();

    let inputs: Vec<DualKcInput> = et0_daily
        .iter()
        .zip(precip_daily.iter())
        .map(|(&et0, &precip)| DualKcInput {
            et0,
            precipitation: precip,
            irrigation: 0.0,
        })
        .collect();

    let state = EvaporationLayerState { de: 0.0, tew, rew };
    let (outputs, _) = dual_kc::simulate_dual_kc(&inputs, kcb, kc_max_val, few, &state);

    for (i, (out, &et0)) in outputs.iter().zip(et0_daily.iter()).enumerate() {
        let ratio = out.etc / et0;
        v.check_bool(
            &format!("Day {}: ETc/ET₀ ({ratio:.3}) ≈ Kcb ({kcb})", i + 1),
            (ratio - kcb).abs() < 0.10,
        );
    }
}

fn main() {
    validation::banner("Dual Crop Coefficient Validation (FAO-56 Ch 7)");
    let mut v = ValidationHarness::new("Dual Kc Validation");
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark must parse");

    validate_eq69(&mut v, &bench);
    validate_kc_max(&mut v, &bench);
    validate_tew(&mut v, &bench);
    validate_kr(&mut v, &bench);
    validate_ke_boundaries(&mut v);
    validate_kcb_vs_kc(&mut v);
    validate_tew_vs_rew(&mut v);
    validate_bare_soil_drydown(&mut v, &bench);
    validate_corn_mid_season(&mut v, &bench);

    v.finish();
}
