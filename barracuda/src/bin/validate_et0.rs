//! Validate FAO-56 Penman-Monteith ET₀ against published examples.
//!
//! Benchmark source: `control/fao56/benchmark_fao56.json`
//! Provenance: Allen et al. (1998) FAO Paper 56, Tables 2.3–2.4, Examples 17–20.
//! Digitized: 2026-02-16, commit: initial airSpring.

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::validation::{json_f64, parse_benchmark_json, ValidationRunner};

/// Benchmark JSON embedded at compile time for reproducibility.
const BENCHMARK_JSON: &str = include_str!("../../../control/fao56/benchmark_fao56.json");

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn main() {
    let mut v = ValidationRunner::new("FAO-56 Penman-Monteith Validation");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_fao56.json must parse");

    // ── Table 2.3: Saturation vapour pressure ────────────────────
    v.section("Table 2.3: Saturation vapour pressure (benchmark JSON)");

    let svp_table = benchmark
        .get("saturation_vapour_pressure_table")
        .and_then(|t| t.get("data"))
        .and_then(|d| d.as_array())
        .expect("SVP table must exist in benchmark JSON");
    let svp_tol = json_f64(
        &benchmark,
        &["saturation_vapour_pressure_table", "tolerance_kpa"],
    )
    .unwrap_or(0.01);

    for entry in svp_table {
        let temp = entry
            .get("temp_c")
            .and_then(serde_json::Value::as_f64)
            .unwrap();
        let expected = entry
            .get("es_kpa")
            .and_then(serde_json::Value::as_f64)
            .unwrap();
        let es = et::saturation_vapour_pressure(temp);
        v.check(&format!("es({temp:.0}°C)"), es, expected, svp_tol);
    }

    // ── Table 2.4: Slope of vapour pressure curve ────────────────
    println!();
    v.section("Table 2.4: Slope Δ (benchmark JSON)");

    let delta_table = benchmark
        .get("slope_vapour_pressure_table")
        .and_then(|t| t.get("data"))
        .and_then(|d| d.as_array())
        .expect("Δ table must exist in benchmark JSON");
    let delta_tol = json_f64(
        &benchmark,
        &["slope_vapour_pressure_table", "tolerance_kpa_per_c"],
    )
    .unwrap_or(0.005);

    for entry in delta_table {
        let temp = entry
            .get("temp_c")
            .and_then(serde_json::Value::as_f64)
            .unwrap();
        let expected = entry
            .get("delta_kpa_per_c")
            .and_then(serde_json::Value::as_f64)
            .unwrap();
        let delta = et::vapour_pressure_slope(temp);
        v.check(&format!("Δ({temp:.0}°C)"), delta, expected, delta_tol);
    }

    // ── Example 18: Uccle, Belgium, 6 July (daily) ──────────────
    // This is our primary validation target because our daily_et0()
    // function matches Example 18's daily timestep (G = 0).
    println!();
    v.section("FAO-56 Example 18: Uccle, Belgium (daily, benchmark JSON)");
    println!("  Location: 50°48'N, 100 m elevation, 6 July");
    println!("  Source: benchmark_fao56.json → example_18_uccle_daily");

    let uccle = &benchmark["example_18_uccle_daily"];

    // Use exact published intermediates as inputs to our function.
    // Our function takes Rs directly (does not derive from sunshine hours).
    let tmin_uc = json_f64(uccle, &["inputs", "tmin_c"]).unwrap();
    let tmax_uc = json_f64(uccle, &["inputs", "tmax_c"]).unwrap();
    let tmean_uc = json_f64(uccle, &["intermediates", "tmean_c"]).unwrap();
    let u2_uc = json_f64(uccle, &["intermediates", "u2_m_s"]).unwrap();
    let ea_uc = json_f64(uccle, &["intermediates", "ea_kpa"]).unwrap();
    let rs_uc = json_f64(uccle, &["intermediates", "rs_mj_m2_day"]).unwrap();
    let lat_uc = json_f64(uccle, &["inputs", "latitude_deg_n"]).unwrap();
    let elev_uc = json_f64(uccle, &["inputs", "altitude_m"]).unwrap();
    let doy_uc = json_f64(uccle, &["inputs", "day_of_year"]).unwrap();
    let expected_et0_uc = json_f64(uccle, &["expected_et0_mm_day"]).unwrap();
    let tol_et0_uc = json_f64(uccle, &["tolerance_mm_day"]).unwrap();

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

    // Validate intermediates against published values
    let pub_es = json_f64(uccle, &["intermediates", "es_kpa"]).unwrap();
    let pub_vpd = json_f64(uccle, &["intermediates", "vpd_kpa"]).unwrap();
    let pub_rn = json_f64(uccle, &["intermediates", "rn_mj_m2_day"]).unwrap();

    v.check("Uccle es", result_uc.es, pub_es, 0.01);
    v.check("Uccle VPD", result_uc.vpd, pub_vpd, 0.02);
    v.check("Uccle Rn", result_uc.rn, pub_rn, 0.5);
    v.check("Uccle ET₀", result_uc.et0, expected_et0_uc, tol_et0_uc);

    // ── Example 17: Bangkok (monthly — G ≠ 0) ───────────────────
    // NOTE: Our daily_et0() assumes G = 0 (correct for daily timestep).
    // Example 17 is monthly (G = 0.14), so we expect a ~0.06 mm/day
    // systematic offset. We validate the components, not the final ET₀.
    println!();
    v.section("FAO-56 Example 17: Bangkok (monthly, G≠0 — components only)");
    println!("  Note: our daily_et0() uses G=0; Example 17 uses G=0.14.");
    println!("  We validate component functions, not final ET₀ (monthly bias).");

    let bangkok = &benchmark["example_17_bangkok_monthly"];
    let tmin_bk = json_f64(bangkok, &["inputs", "tmin_c"]).unwrap();
    let tmax_bk = json_f64(bangkok, &["inputs", "tmax_c"]).unwrap();

    let pub_es_bk = json_f64(bangkok, &["intermediates", "es_kpa"]).unwrap();
    let calc_es_bk = et::mean_saturation_vapour_pressure(tmin_bk, tmax_bk);
    v.check("Bangkok es", calc_es_bk, pub_es_bk, 0.02);

    let pub_delta_bk = json_f64(bangkok, &["intermediates", "delta_kpa_per_c"]).unwrap();
    let tmean_bk = json_f64(bangkok, &["intermediates", "tmean_c"]).unwrap();
    let calc_delta_bk = et::vapour_pressure_slope(tmean_bk);
    v.check("Bangkok Δ", calc_delta_bk, pub_delta_bk, 0.005);

    let pub_gamma_bk = json_f64(bangkok, &["intermediates", "gamma_kpa_per_c"]).unwrap();
    let pub_p_bk = json_f64(bangkok, &["intermediates", "pressure_kpa"]).unwrap();
    let calc_gamma_bk = et::psychrometric_constant(pub_p_bk);
    v.check("Bangkok γ", calc_gamma_bk, pub_gamma_bk, 0.001);

    // ── Boundary conditions ──────────────────────────────────────
    println!();
    v.section("Boundary conditions");

    // Cold conditions → low or zero ET₀
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
    v.check("Cold ET₀ ≥ 0", result_cold.et0, 0.0, 0.5);

    // High altitude → lower pressure
    let p_high = et::atmospheric_pressure(3000.0);
    let p_sea = et::atmospheric_pressure(0.0);
    v.check_bool("High altitude → lower pressure", p_high < p_sea, true);

    // ET₀ positive for typical conditions
    v.check_bool("Uccle ET₀ > 0", result_uc.et0 > 0.0, true);

    v.finish();
}
