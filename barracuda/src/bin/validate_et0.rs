// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate FAO-56 Penman-Monteith ET₀ against published examples.
//!
//! Benchmark source: `control/fao56/benchmark_fao56.json`
//! Provenance: Allen et al. (1998) FAO Paper 56, Tables 2.3–2.4, Examples 17–20.
//! script=`control/fao56/penman_monteith.py`, commit=94cc51d, date=2026-02-16
//! Run: `python3 control/fao56/penman_monteith.py`

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness, json_f64, parse_benchmark_json};

/// Benchmark JSON embedded at compile time for reproducibility.
const BENCHMARK_JSON: &str = include_str!("../../../control/fao56/benchmark_fao56.json");

/// Validate SVP Table 2.3 against benchmark data.
fn validate_svp_table(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Table 2.3: Saturation vapour pressure (benchmark JSON)");

    let svp_table = benchmark
        .get("saturation_vapour_pressure_table")
        .and_then(|t| t.get("data"))
        .and_then(|d| d.as_array())
        .expect("SVP table must exist in benchmark JSON");
    let svp_tol = json_f64(
        benchmark,
        &["saturation_vapour_pressure_table", "tolerance_kpa"],
    )
    .unwrap_or(tolerances::ET0_SAT_VAPOUR_PRESSURE.abs_tol);

    for entry in svp_table {
        let temp = entry
            .get("temp_c")
            .and_then(serde_json::Value::as_f64)
            .expect("SVP table entry must have 'temp_c'");
        let expected = entry
            .get("es_kpa")
            .and_then(serde_json::Value::as_f64)
            .expect("SVP table entry must have 'es_kpa'");
        let es = et::saturation_vapour_pressure(temp);
        v.check_abs(&format!("es({temp:.0}°C)"), es, expected, svp_tol);
    }
}

/// Validate Δ Table 2.4 against benchmark data.
fn validate_delta_table(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    println!();
    validation::section("Table 2.4: Slope Δ (benchmark JSON)");

    let delta_table = benchmark
        .get("slope_vapour_pressure_table")
        .and_then(|t| t.get("data"))
        .and_then(|d| d.as_array())
        .expect("Δ table must exist in benchmark JSON");
    let delta_tol = json_f64(
        benchmark,
        &["slope_vapour_pressure_table", "tolerance_kpa_per_c"],
    )
    .unwrap_or(tolerances::ET0_SLOPE_VAPOUR.abs_tol);

    for entry in delta_table {
        let temp = entry
            .get("temp_c")
            .and_then(serde_json::Value::as_f64)
            .expect("Delta table entry must have 'temp_c'");
        let expected = entry
            .get("delta_kpa_per_c")
            .and_then(serde_json::Value::as_f64)
            .expect("Delta table entry must have 'delta_kpa_per_c'");
        let delta = et::vapour_pressure_slope(temp);
        v.check_abs(&format!("Δ({temp:.0}°C)"), delta, expected, delta_tol);
    }
}

/// Validate Example 18 (Uccle, daily) — primary validation target.
/// Returns Uccle ET₀ for reuse in boundary checks.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "DOY from JSON f64 is a non-negative integer"
)]
fn validate_uccle(v: &mut ValidationHarness, benchmark: &serde_json::Value) -> f64 {
    println!();
    validation::section("FAO-56 Example 18: Uccle, Belgium (daily, benchmark JSON)");
    println!("  Location: 50°48'N, 100 m elevation, 6 July");
    println!("  Source: benchmark_fao56.json → example_18_uccle_daily");

    let uccle = &benchmark["example_18_uccle_daily"];

    let tmin_uc = json_f64(uccle, &["inputs", "tmin_c"]).expect("Uccle: inputs.tmin_c");
    let tmax_uc = json_f64(uccle, &["inputs", "tmax_c"]).expect("Uccle: inputs.tmax_c");
    let tmean_uc =
        json_f64(uccle, &["intermediates", "tmean_c"]).expect("Uccle: intermediates.tmean_c");
    let u2_uc = json_f64(uccle, &["intermediates", "u2_m_s"]).expect("Uccle: intermediates.u2_m_s");
    let ea_uc = json_f64(uccle, &["intermediates", "ea_kpa"]).expect("Uccle: intermediates.ea_kpa");
    let rs_uc = json_f64(uccle, &["intermediates", "rs_mj_m2_day"])
        .expect("Uccle: intermediates.rs_mj_m2_day");
    let lat_uc =
        json_f64(uccle, &["inputs", "latitude_deg_n"]).expect("Uccle: inputs.latitude_deg_n");
    let elev_uc = json_f64(uccle, &["inputs", "altitude_m"]).expect("Uccle: inputs.altitude_m");
    let doy_uc = json_f64(uccle, &["inputs", "day_of_year"]).expect("Uccle: inputs.day_of_year");
    let expected_et0_uc =
        json_f64(uccle, &["expected_et0_mm_day"]).expect("Uccle: expected_et0_mm_day");
    let tol_et0_uc = json_f64(uccle, &["tolerance_mm_day"]).expect("Uccle: tolerance_mm_day");

    let input_uc = DailyEt0Input {
        tmin: tmin_uc,
        tmax: tmax_uc,
        tmean: Some(tmean_uc),
        solar_radiation: rs_uc,
        wind_speed_2m: u2_uc,
        actual_vapour_pressure: ea_uc,
        elevation_m: elev_uc,
        latitude_deg: lat_uc,
        day_of_year: doy_uc.round() as u32,
    };

    let result_uc = et::daily_et0(&input_uc);
    println!("  ET₀ = {:.3} mm/day", result_uc.et0);
    println!("  Rn  = {:.3} MJ/m²/day", result_uc.rn);
    println!("  VPD = {:.3} kPa", result_uc.vpd);

    let pub_es =
        json_f64(uccle, &["intermediates", "es_kpa"]).expect("Uccle: intermediates.es_kpa");
    let pub_vpd =
        json_f64(uccle, &["intermediates", "vpd_kpa"]).expect("Uccle: intermediates.vpd_kpa");
    let pub_rn = json_f64(uccle, &["intermediates", "rn_mj_m2_day"])
        .expect("Uccle: intermediates.rn_mj_m2_day");

    v.check_abs(
        "Uccle es",
        result_uc.es,
        pub_es,
        tolerances::ET0_SAT_VAPOUR_PRESSURE.abs_tol,
    );
    v.check_abs(
        "Uccle VPD",
        result_uc.vpd,
        pub_vpd,
        tolerances::ET0_VPD.abs_tol,
    );
    v.check_abs(
        "Uccle Rn",
        result_uc.rn,
        pub_rn,
        tolerances::ET0_NET_RADIATION.abs_tol,
    );
    v.check_abs("Uccle ET₀", result_uc.et0, expected_et0_uc, tol_et0_uc);

    result_uc.et0
}

/// Validate Example 17 (Bangkok, monthly) component functions.
fn validate_bangkok(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    println!();
    validation::section("FAO-56 Example 17: Bangkok (monthly, G≠0 — components only)");
    println!("  Note: our daily_et0() uses G=0; Example 17 uses G=0.14.");
    println!("  We validate component functions, not final ET₀ (monthly bias).");

    let bangkok = &benchmark["example_17_bangkok_monthly"];
    let tmin_bk = json_f64(bangkok, &["inputs", "tmin_c"]).expect("Bangkok: inputs.tmin_c");
    let tmax_bk = json_f64(bangkok, &["inputs", "tmax_c"]).expect("Bangkok: inputs.tmax_c");

    let pub_es_bk =
        json_f64(bangkok, &["intermediates", "es_kpa"]).expect("Bangkok: intermediates.es_kpa");
    let calc_es_bk = et::mean_saturation_vapour_pressure(tmin_bk, tmax_bk);
    v.check_abs(
        "Bangkok es",
        calc_es_bk,
        pub_es_bk,
        tolerances::ET0_SAT_VAPOUR_PRESSURE_WIDE.abs_tol,
    );

    let pub_delta_bk = json_f64(bangkok, &["intermediates", "delta_kpa_per_c"])
        .expect("Bangkok: intermediates.delta_kpa_per_c");
    let tmean_bk =
        json_f64(bangkok, &["intermediates", "tmean_c"]).expect("Bangkok: intermediates.tmean_c");
    let calc_delta_bk = et::vapour_pressure_slope(tmean_bk);
    v.check_abs(
        "Bangkok Δ",
        calc_delta_bk,
        pub_delta_bk,
        tolerances::ET0_SLOPE_VAPOUR.abs_tol,
    );

    let pub_gamma_bk = json_f64(bangkok, &["intermediates", "gamma_kpa_per_c"])
        .expect("Bangkok: intermediates.gamma_kpa_per_c");
    let pub_p_bk = json_f64(bangkok, &["intermediates", "pressure_kpa"])
        .expect("Bangkok: intermediates.pressure_kpa");
    let calc_gamma_bk = et::psychrometric_constant(pub_p_bk);
    v.check_abs(
        "Bangkok γ",
        calc_gamma_bk,
        pub_gamma_bk,
        tolerances::PSYCHROMETRIC_CONSTANT.abs_tol,
    );
}

/// Validate boundary conditions (cold, high altitude, positivity).
fn validate_boundaries(v: &mut ValidationHarness, uccle_et0: f64) {
    println!();
    validation::section("Boundary conditions");

    let cold = DailyEt0Input {
        tmin: -5.0,
        tmax: 2.0,
        tmean: None,
        solar_radiation: 5.0,
        wind_speed_2m: 1.0,
        actual_vapour_pressure: 0.4,
        elevation_m: 200.0,
        latitude_deg: 60.0,
        day_of_year: 355,
    };
    let result_cold = et::daily_et0(&cold);
    v.check_abs(
        "Cold ET₀ ≥ 0",
        result_cold.et0,
        0.0,
        tolerances::ET0_COLD_CLIMATE.abs_tol,
    );

    let p_high = et::atmospheric_pressure(3000.0);
    let p_sea = et::atmospheric_pressure(0.0);
    v.check_bool("High altitude → lower pressure", p_high < p_sea);

    v.check_bool("Uccle ET₀ > 0", uccle_et0 > 0.0);
}

fn main() {
    validation::init_tracing();
    validation::banner("FAO-56 Penman-Monteith Validation");
    let mut v = ValidationHarness::new("FAO-56 Penman-Monteith Validation");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_fao56.json must parse");

    validate_svp_table(&mut v, &benchmark);
    validate_delta_table(&mut v, &benchmark);
    let uccle_et0 = validate_uccle(&mut v, &benchmark);
    validate_bangkok(&mut v, &benchmark);
    validate_boundaries(&mut v, uccle_et0);

    v.finish();
}
