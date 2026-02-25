// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate cover crop dual Kc + no-till mulch effects against Python control.
//!
//! Benchmark: `control/dual_kc/benchmark_cover_crop_kc.json`
//! Python:    `control/dual_kc/cover_crop_dual_kc.py` (40/40 PASS)
//!
//! Validates:
//! 1. Cover crop Kcb values (5 crops)
//! 2. Mulch factor reduces Ke proportionally
//! 3. No-till vs conventional ET comparison
//! 4. Cross-validation of mulch Ke against Python at 1e-6

use airspring_barracuda::eco::dual_kc::{
    self, CoverCropType, DualKcInput, EvaporationLayerState, ResidueLevel,
};
use airspring_barracuda::validation::{
    self, json_array, json_f64, json_field, json_str, parse_benchmark_json, ValidationHarness,
};

const BENCHMARK_JSON: &str = include_str!("../../../control/dual_kc/benchmark_cover_crop_kc.json");

fn validate_cover_crop_kcb(v: &mut ValidationHarness) {
    validation::section("Cover crop Kcb physical reasonableness");

    let crops = [
        ("cereal_rye", CoverCropType::CerealRye),
        ("crimson_clover", CoverCropType::CrimsonClover),
        ("winter_wheat_cover", CoverCropType::WinterWheatCover),
        ("hairy_vetch", CoverCropType::HairyVetch),
        ("radish_tillage", CoverCropType::TillageRadish),
    ];

    for (name, crop) in crops {
        let kcb = crop.basal_coefficients();
        v.check_bool(
            &format!(
                "{name}: Kcb_ini ({:.2}) < Kcb_mid ({:.2})",
                kcb.kcb_ini, kcb.kcb_mid
            ),
            kcb.kcb_ini < kcb.kcb_mid,
        );
        v.check_bool(
            &format!("{name}: Kcb_mid ({:.2}) in [0.5, 1.3]", kcb.kcb_mid),
            (0.5..=1.3).contains(&kcb.kcb_mid),
        );
        v.check_bool(
            &format!("{name}: max_height ({:.2}) > 0", kcb.max_height_m),
            kcb.max_height_m > 0.0,
        );
    }
}

fn validate_mulch_ke(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Mulch reduces Ke proportionally");

    for tc in json_array(
        bench,
        &[
            "validation_checks",
            "mulch_reduces_evaporation",
            "test_cases",
        ],
    ) {
        let result = dual_kc::mulched_ke(
            json_field(tc, "kr"),
            json_field(tc, "kcb"),
            json_field(tc, "kc_max"),
            json_field(tc, "few"),
            json_field(tc, "mulch_factor"),
        );
        v.check_abs(
            json_str(tc, "label"),
            result,
            json_field(tc, "expected_ke"),
            1e-6,
        );
    }
}

fn validate_mulch_factor_ordering(v: &mut ValidationHarness) {
    println!();
    validation::section("Mulch factor ordering");

    let levels = [
        ResidueLevel::NoResidue,
        ResidueLevel::Light,
        ResidueLevel::Moderate,
        ResidueLevel::Heavy,
        ResidueLevel::FullMulch,
    ];

    let mut prev = 1.1;
    for level in levels {
        let mf = level.mulch_factor();
        v.check_bool(
            &format!("{level:?}: mulch_factor ({mf:.2}) <= prev ({prev:.2})"),
            mf <= prev,
        );
        prev = mf;
    }
}

fn validate_notill_vs_conventional(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("No-till vs conventional: rye→corn transition");

    let state = EvaporationLayerState {
        de: 0.0,
        tew: 22.5,
        rew: 9.0,
    };

    let et0_daily = [4.0, 4.5, 4.2, 5.0, 5.5, 5.0, 4.8];
    let precip_daily = [10.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0];

    let inputs: Vec<DualKcInput> = et0_daily
        .iter()
        .zip(precip_daily.iter())
        .map(|(&et0, &precip)| DualKcInput {
            et0,
            precipitation: precip,
            irrigation: 0.0,
        })
        .collect();

    let (conv, conv_final) = dual_kc::simulate_dual_kc(&inputs, 0.15, 1.20, 1.0, &state);
    let (notill, notill_final) =
        dual_kc::simulate_dual_kc_mulched(&inputs, 0.15, 1.20, 1.0, 0.40, &state);

    let conv_et: f64 = conv.iter().map(|o| o.etc).sum();
    let notill_et: f64 = notill.iter().map(|o| o.etc).sum();

    v.check_bool(
        &format!("No-till ETc ({notill_et:.2}) < conventional ({conv_et:.2})"),
        notill_et < conv_et,
    );

    let savings_pct = 100.0 * (1.0 - notill_et / conv_et);
    let expected_range =
        &bench["validation_checks"]["no_till_conserves_water"]["expected_et_reduction_pct"];
    let min_pct = json_f64(expected_range, &["min"]).expect("expected_et_reduction_pct.min");
    let max_pct = json_f64(expected_range, &["max"]).expect("expected_et_reduction_pct.max");

    v.check_bool(
        &format!("ET savings {savings_pct:.1}% in [{min_pct}, {max_pct}]"),
        (min_pct..=max_pct).contains(&savings_pct),
    );

    v.check_bool(
        &format!(
            "No-till De ({:.2}) < conventional De ({:.2})",
            notill_final.de, conv_final.de
        ),
        notill_final.de < conv_final.de,
    );

    println!(
        "  Conventional: {conv_et:.2} mm, No-till: {notill_et:.2} mm, Savings: {savings_pct:.1}%"
    );
}

fn validate_transition_phases(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Rye→corn transition phases");

    for phase in json_array(bench, &["transition_scenarios", "rye_to_corn", "phases"]) {
        let period = json_str(phase, "period");
        let kcb = json_field(phase, "kcb");
        let mf = json_field(phase, "mulch_factor");

        v.check_bool(
            &format!("{period}: Kcb={kcb} in [0, 1.5]"),
            (0.0..=1.5).contains(&kcb),
        );
        v.check_bool(
            &format!("{period}: mf={mf} in [0, 1]"),
            (0.0..=1.0).contains(&mf),
        );
    }
}

fn validate_islam_observations(v: &mut ValidationHarness, bench: &serde_json::Value) {
    println!();
    validation::section("Islam et al. (2014) no-till observations");

    let obs = &bench["no_till_soil_moisture"]["observations"];

    let f = |metric: &str, variant: &str| -> f64 {
        json_f64(obs, &[metric, variant]).expect("Islam et al. benchmark value")
    };

    let nt_soc = f("soil_organic_carbon_pct", "no_till");
    let cv_soc = f("soil_organic_carbon_pct", "conventional");
    v.check_bool(
        &format!("SOC: no-till ({nt_soc}%) > conventional ({cv_soc}%)"),
        nt_soc > cv_soc,
    );

    let nt_bd = f("bulk_density_g_cm3", "no_till");
    let cv_bd = f("bulk_density_g_cm3", "conventional");
    v.check_bool(
        &format!("BD: no-till ({nt_bd}) < conventional ({cv_bd})"),
        nt_bd < cv_bd,
    );

    let nt_inf = f("infiltration_rate_mm_hr", "no_till");
    let cv_inf = f("infiltration_rate_mm_hr", "conventional");
    v.check_bool(
        &format!("Infiltration: no-till ({nt_inf}) > conventional ({cv_inf})"),
        nt_inf > cv_inf,
    );

    let nt_awc = f("available_water_capacity_mm", "no_till");
    let cv_awc = f("available_water_capacity_mm", "conventional");
    v.check_bool(
        &format!("AWC: no-till ({nt_awc}) > conventional ({cv_awc})"),
        nt_awc > cv_awc,
    );
}

fn main() {
    validation::banner("Cover Crop + No-Till Validation (FAO-56 Ch 11)");
    let mut v = ValidationHarness::new("Cover Crop Validation");
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark must parse");

    validate_cover_crop_kcb(&mut v);
    validate_mulch_ke(&mut v, &bench);
    validate_mulch_factor_ordering(&mut v);
    validate_islam_observations(&mut v, &bench);
    validate_notill_vs_conventional(&mut v, &bench);
    validate_transition_phases(&mut v, &bench);

    v.finish();
}
