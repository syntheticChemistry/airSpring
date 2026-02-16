//! Phase 2 cross-validation: Rust side.
//!
//! Computes the same values as `scripts/cross_validate.py` using identical
//! inputs. Output is JSON so the two can be diff'd directly.
//!
//! Usage:
//! ```bash
//! python scripts/cross_validate.py > /tmp/airspring_python.json
//! cargo run --release --bin cross_validate > /tmp/airspring_rust.json
//! diff /tmp/airspring_python.json /tmp/airspring_rust.json
//! ```

use airspring_barracuda::eco::{
    correction, evapotranspiration as et, sensor_calibration as sc, soil_moisture as sm,
    water_balance,
};
use airspring_barracuda::testutil;
use serde_json::json;

/// Round to 6 decimal places (matches Python's `round(x, 6)`).
fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn main() {
    // ── FAO-56 Example 18: Uccle, Belgium, 6 July (DOY 187) ─────────
    let tmin = 12.3;
    let tmax = 21.5;
    let tmean = 16.9;
    let rs = 22.07;
    let wind_10m = 2.78;
    let rh_min = 63.0;
    let rh_max = 84.0;
    let elevation_m = 100.0;
    let latitude_deg: f64 = 50.80;
    let doy: u32 = 187;

    // ── Atmospheric ──────────────────────────────────────────────────
    let pressure = et::atmospheric_pressure(elevation_m);
    let gamma = et::psychrometric_constant(pressure);
    let delta = et::vapour_pressure_slope(tmean);
    let es = et::mean_saturation_vapour_pressure(tmin, tmax);
    let ea = et::actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max);
    let u2 = et::wind_speed_at_2m(wind_10m, 10.0);

    // ── Solar geometry ───────────────────────────────────────────────
    let lat_rad = latitude_deg.to_radians();
    let dr = et::inverse_rel_distance(doy);
    let decl = et::solar_declination(doy);
    let ws = et::sunset_hour_angle(lat_rad, decl);
    let ra = et::extraterrestrial_radiation(lat_rad, doy);
    let n_hours = et::daylight_hours(lat_rad, doy);

    // ── Radiation ────────────────────────────────────────────────────
    let rso = et::clear_sky_radiation(elevation_m, ra);
    let rns = et::net_shortwave_radiation(rs, 0.23);
    let rnl = et::net_longwave_radiation(tmin, tmax, ea, rs, rso);
    let rn = et::net_radiation(rns, rnl);

    // ── ET₀ ──────────────────────────────────────────────────────────
    let vpd = es - ea;
    let input = et::DailyEt0Input {
        tmin,
        tmax,
        tmean: Some(tmean),
        solar_radiation: rs,
        wind_speed_2m: u2,
        actual_vapour_pressure: ea,
        elevation_m,
        latitude_deg,
        day_of_year: doy,
    };
    let et0_result = et::daily_et0(&input);

    // ── Topp equation ────────────────────────────────────────────────
    let eps_values = [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    let topp_results: serde_json::Map<String, serde_json::Value> = eps_values
        .iter()
        .map(|&e| {
            (
                format!("theta_eps_{}", e as u32),
                json!(round6(sm::topp_equation(e))),
            )
        })
        .collect();

    // ── SoilWatch 10 ─────────────────────────────────────────────────
    let raw_counts = [5000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0];
    let sw10_results: serde_json::Map<String, serde_json::Value> = raw_counts
        .iter()
        .map(|&rc| {
            (
                format!("vwc_rc_{}", rc as u32),
                json!(round6(sc::soilwatch10_vwc(rc))),
            )
        })
        .collect();

    // ── Irrigation recommendation ─────────────────────────────────────
    let ir_single = sc::irrigation_recommendation(0.12, 0.08, 30.0);
    let layers = [
        sc::SoilLayer {
            field_capacity: 0.12,
            current_vwc: 0.08,
            depth_cm: 30.0,
        },
        sc::SoilLayer {
            field_capacity: 0.15,
            current_vwc: 0.10,
            depth_cm: 30.0,
        },
        sc::SoilLayer {
            field_capacity: 0.18,
            current_vwc: 0.12,
            depth_cm: 30.0,
        },
    ];
    let ir_multi = sc::multi_layer_irrigation(&layers);

    // ── Statistical measures ──────────────────────────────────────────
    let obs = [0.10, 0.15, 0.20, 0.25, 0.30];
    let sim = [0.11, 0.14, 0.21, 0.24, 0.31];
    let rmse_val = testutil::rmse(&obs, &sim);
    let mbe_val = testutil::mbe(&obs, &sim);
    let ia_val = testutil::index_of_agreement(&obs, &sim);
    let nse_val = testutil::nash_sutcliffe(&obs, &sim);

    // ── SVP table ────────────────────────────────────────────────────
    let temps = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0];
    let svp_results: serde_json::Map<String, serde_json::Value> = temps
        .iter()
        .map(|&t| {
            (
                format!("es_{}c", t as u32),
                json!(round6(et::saturation_vapour_pressure(t))),
            )
        })
        .collect();

    // ── Sunshine-based radiation (FAO-56 Eq. 35) ─────────────────────
    let n_sunshine = 9.25; // Uccle, July
    let rs_sunshine = et::solar_radiation_from_sunshine(n_sunshine, n_hours, ra);

    // ── Temperature-based radiation (FAO-56 Eq. 50, Hargreaves) ─────
    let rs_temp_interior = et::solar_radiation_from_temperature(tmax, tmin, ra, 0.16);
    let rs_temp_coastal = et::solar_radiation_from_temperature(tmax, tmin, ra, 0.19);

    // ── Monthly soil heat flux (FAO-56 Eq. 43) ──────────────────────
    let g_warming = et::soil_heat_flux_monthly(25.0, 22.0);
    let g_cooling = et::soil_heat_flux_monthly(18.0, 25.0);

    // ── Hargreaves ET₀ (FAO-56 Eq. 52) ──────────────────────────────
    let ra_mm = ra / 2.45; // MJ → mm/day equivalent
    let harg_et0 = et::hargreaves_et0(tmin, tmax, ra_mm);

    // ── Low-level PM (same result, different API path) ───────────────
    let pm_lowlevel = et::fao56_penman_monteith(rn, 0.0, tmean, u2, vpd, delta, gamma);

    // ── Standalone water balance functions ──────────────────────────
    let taw = water_balance::total_available_water(0.30, 0.10, 500.0);
    let raw = water_balance::readily_available_water(taw, 0.5);
    let ks_at_raw = water_balance::stress_coefficient(raw, taw, raw);
    let ks_at_midpoint = water_balance::stress_coefficient(f64::midpoint(taw, raw), taw, raw);
    let (dr_new, actual_et, dp) =
        water_balance::daily_water_balance_step(20.0, 5.0, 0.0, 4.0, 1.0, 1.0, taw);

    // ── Correction model evaluation ────────────────────────────────
    let lin_val = correction::linear_model(0.15, 1.2, 0.01);
    let quad_val = correction::quadratic_model(0.15, 2.0, 1.0, 0.05);
    let exp_val = correction::exponential_model(0.15, 0.1, 3.0);
    let log_val = correction::logarithmic_model(0.15, 0.2, 0.1);

    // ── Assemble JSON ────────────────────────────────────────────────
    let output = json!({
        "atmospheric": {
            "pressure_kpa": round6(pressure),
            "gamma_kpa_c": round6(gamma),
            "delta_kpa_c": round6(delta),
            "es_kpa": round6(es),
            "ea_kpa": round6(ea),
            "u2_ms": round6(u2),
        },
        "solar": {
            "dr": round6(dr),
            "declination_rad": round6(decl),
            "sunset_hour_angle_rad": round6(ws),
            "ra_mj_m2_day": round6(ra),
            "daylight_hours": round6(n_hours),
        },
        "radiation": {
            "rso_mj_m2_day": round6(rso),
            "rns_mj_m2_day": round6(rns),
            "rnl_mj_m2_day": round6(rnl),
            "rn_mj_m2_day": round6(rn),
        },
        "et0": {
            "vpd_kpa": round6(vpd),
            "et0_mm_day": round6(et0_result.et0),
        },
        "topp": topp_results,
        "soilwatch10": sw10_results,
        "irrigation": {
            "ir_single_cm": round6(ir_single),
            "ir_multi_cm": round6(ir_multi),
        },
        "statistics": {
            "rmse": round6(rmse_val),
            "mbe": round6(mbe_val),
            "ia": round6(ia_val),
            "r2": round6(nse_val),
        },
        "sunshine_radiation": {
            "rs_sunshine_mj": round6(rs_sunshine),
        },
        "temp_radiation": {
            "rs_temp_interior_mj": round6(rs_temp_interior),
            "rs_temp_coastal_mj": round6(rs_temp_coastal),
        },
        "soil_heat_flux": {
            "g_warming_mj": round6(g_warming),
            "g_cooling_mj": round6(g_cooling),
        },
        "hargreaves": {
            "ra_mm_day": round6(ra_mm),
            "et0_hargreaves_mm": round6(harg_et0),
        },
        "svp_table": svp_results,
        "lowlevel_pm": {
            "et0_lowlevel_mm": round6(pm_lowlevel),
        },
        "water_balance_standalone": {
            "taw_mm": round6(taw),
            "raw_mm": round6(raw),
            "ks_at_raw": round6(ks_at_raw),
            "ks_at_midpoint": round6(ks_at_midpoint),
            "dr_new_mm": round6(dr_new),
            "actual_et_mm": round6(actual_et),
            "deep_percolation_mm": round6(dp),
        },
        "correction_models": {
            "linear_val": round6(lin_val),
            "quadratic_val": round6(quad_val),
            "exponential_val": round6(exp_val),
            "logarithmic_val": round6(log_val),
        },
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization")
    );
}
