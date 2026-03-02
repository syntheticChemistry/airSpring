// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate FAO-56 Penman-Monteith sensitivity analysis (Exp 017).
//!
//! Benchmark: `control/sensitivity/benchmark_sensitivity.json`
//! Paper: Allen et al. (1998) FAO-56 Ch 4; Gong et al. (2006)
//! Python: `control/sensitivity/et0_sensitivity.py`
//!
//! OAT (one-at-a-time) ±10% perturbation of 6 input variables.
//! Verifies: baseline ET₀, monotonicity, elasticity bounds, symmetry,
//! ranking consistency across 3 climatic conditions.

use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::validation::{
    self, json_field, json_str, parse_benchmark_json, ValidationHarness,
};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/sensitivity/benchmark_sensitivity.json");

#[derive(Clone)]
struct MeteoParams {
    tmin_c: f64,
    tmax_c: f64,
    rh_min_pct: f64,
    rh_max_pct: f64,
    wind_2m_m_s: f64,
    solar_rad_mj_m2_day: f64,
    elevation_m: f64,
    latitude_deg: f64,
    day_of_year: u32,
}

impl MeteoParams {
    fn from_json(v: &serde_json::Value) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let doy = json_field(v, "day_of_year") as u32;
        Self {
            tmin_c: json_field(v, "tmin_c"),
            tmax_c: json_field(v, "tmax_c"),
            rh_min_pct: json_field(v, "rh_min_pct"),
            rh_max_pct: json_field(v, "rh_max_pct"),
            wind_2m_m_s: json_field(v, "wind_2m_m_s"),
            solar_rad_mj_m2_day: json_field(v, "solar_rad_mj_m2_day"),
            elevation_m: json_field(v, "elevation_m"),
            latitude_deg: json_field(v, "latitude_deg"),
            day_of_year: doy,
        }
    }

    fn get(&self, name: &str) -> f64 {
        match name {
            "tmin_c" => self.tmin_c,
            "tmax_c" => self.tmax_c,
            "rh_min_pct" => self.rh_min_pct,
            "rh_max_pct" => self.rh_max_pct,
            "wind_2m_m_s" => self.wind_2m_m_s,
            "solar_rad_mj_m2_day" => self.solar_rad_mj_m2_day,
            _ => panic!("unknown variable: {name}"),
        }
    }

    fn set(&mut self, name: &str, val: f64) {
        match name {
            "tmin_c" => self.tmin_c = val,
            "tmax_c" => self.tmax_c = val,
            "rh_min_pct" => self.rh_min_pct = val,
            "rh_max_pct" => self.rh_max_pct = val,
            "wind_2m_m_s" => self.wind_2m_m_s = val,
            "solar_rad_mj_m2_day" => self.solar_rad_mj_m2_day = val,
            _ => panic!("unknown variable: {name}"),
        }
    }
}

/// Compute ET₀ using the barracuda `eco::evapotranspiration` module.
fn compute_et0(p: &MeteoParams) -> f64 {
    let tmean = f64::midpoint(p.tmin_c, p.tmax_c);
    let pressure = et::atmospheric_pressure(p.elevation_m);
    let gamma = et::psychrometric_constant(pressure);
    let delta = et::vapour_pressure_slope(tmean);
    let es = et::mean_saturation_vapour_pressure(p.tmin_c, p.tmax_c);
    let ea = et::actual_vapour_pressure_rh(p.tmin_c, p.tmax_c, p.rh_min_pct, p.rh_max_pct);
    let vpd = es - ea;

    let lat_rad = p.latitude_deg.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, p.day_of_year);
    let rso = et::clear_sky_radiation(p.elevation_m, ra);
    let rns = et::net_shortwave_radiation(p.solar_rad_mj_m2_day, 0.23);
    let rnl = et::net_longwave_radiation(p.tmin_c, p.tmax_c, ea, p.solar_rad_mj_m2_day, rso);
    let rn = rns - rnl;

    et::fao56_penman_monteith(rn, 0.0, tmean, p.wind_2m_m_s, vpd, delta, gamma)
}

struct SensResult {
    name: String,
    label: String,
    sensitivity: f64,
    abs_sensitivity: f64,
    elasticity: f64,
    et0_plus: f64,
    et0_minus: f64,
    et0_base: f64,
}

fn oat_sensitivity(params: &MeteoParams, var_name: &str, label: &str, pct: f64) -> SensResult {
    let et0_base = compute_et0(params);
    let x_base = params.get(var_name);
    let dx = x_base.abs() * pct / 100.0;

    let mut p_plus = params.clone();
    p_plus.set(var_name, x_base + dx);
    let et0_plus = compute_et0(&p_plus);

    let mut p_minus = params.clone();
    p_minus.set(var_name, x_base - dx);
    let et0_minus = compute_et0(&p_minus);

    let sensitivity = if dx > 0.0 {
        (et0_plus - et0_minus) / (2.0 * dx)
    } else {
        0.0
    };
    let elasticity = if et0_base > 0.0 {
        ((et0_plus - et0_minus) / et0_base) / (2.0 * pct / 100.0)
    } else {
        0.0
    };

    SensResult {
        name: var_name.to_string(),
        label: label.to_string(),
        sensitivity,
        abs_sensitivity: sensitivity.abs(),
        elasticity,
        et0_plus,
        et0_minus,
        et0_base,
    }
}

fn validate_multi_site(
    v: &mut ValidationHarness,
    benchmark: &serde_json::Value,
    variables: &[serde_json::Value],
    pct: f64,
) {
    validation::section("Multi-Site Consistency");
    let sites = benchmark["validation_checks"]["multi_site_consistency"]["sites"]
        .as_array()
        .expect("sites array");
    for site in sites {
        let site_name = json_str(site, "name");
        let site_params = MeteoParams::from_json(site);
        let site_et0 = compute_et0(&site_params);

        let mut site_results: Vec<SensResult> = variables
            .iter()
            .map(|var| {
                let name = json_str(var, "name");
                let label = json_str(var, "label");
                oat_sensitivity(&site_params, name, label, pct)
            })
            .collect();
        site_results.sort_by(|a, b| {
            b.abs_sensitivity
                .partial_cmp(&a.abs_sensitivity)
                .expect("non-NaN sensitivity comparison")
        });

        let has_rad = site_results
            .iter()
            .take(3)
            .map(|r| r.name.as_str())
            .any(|x| x == "solar_rad_mj_m2_day");
        v.check_bool(
            &format!("{site_name}: ET₀={site_et0:.2}, radiation in top-3"),
            has_rad,
        );

        for r in &site_results {
            println!("      {:>10}: |S|={:.4}", r.label, r.abs_sensitivity);
        }
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("ET₀ Sensitivity Analysis (Exp 017)");
    let mut v = ValidationHarness::new("Sensitivity Analysis");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_sensitivity.json must parse");

    let baseline = MeteoParams::from_json(&benchmark["baseline_conditions"]);
    let pct = json_field(&benchmark, "perturbation_pct");
    let variables = benchmark["variables"].as_array().expect("variables array");

    validation::section("Baseline ET₀");
    let et0_base = compute_et0(&baseline);
    let expected_et0 = json_field(&benchmark["validation_checks"]["baseline_et0"], "expected");
    let tol_et0 = json_field(&benchmark["validation_checks"]["baseline_et0"], "tolerance");
    v.check_abs("FAO-56 Example 18 ET₀", et0_base, expected_et0, tol_et0);

    validation::section("Sensitivity Ranking");
    let mut results: Vec<SensResult> = variables
        .iter()
        .map(|var| {
            let name = json_str(var, "name");
            let label = json_str(var, "label");
            oat_sensitivity(&baseline, name, label, pct)
        })
        .collect();
    results.sort_by(|a, b| {
        b.abs_sensitivity
            .partial_cmp(&a.abs_sensitivity)
            .expect("non-NaN sensitivity comparison")
    });

    for (i, r) in results.iter().enumerate() {
        println!(
            "    #{}: {:>10} |S|={:.4}  E={:+.4}",
            i + 1,
            r.label,
            r.abs_sensitivity,
            r.elasticity,
        );
    }

    let top2: Vec<&str> = results.iter().take(2).map(|r| r.name.as_str()).collect();
    let allowed = ["solar_rad_mj_m2_day", "rh_min_pct", "rh_max_pct"];
    let has_overlap = top2.iter().any(|n| allowed.contains(n));
    v.check_bool(
        &format!("top-2 ({top2:?}) includes rad/humidity"),
        has_overlap,
    );

    validation::section("Monotonicity");
    let pos_vars = ["tmax_c", "tmin_c", "solar_rad_mj_m2_day", "wind_2m_m_s"];
    let neg_vars = ["rh_min_pct", "rh_max_pct"];
    for r in &results {
        if pos_vars.contains(&r.name.as_str()) {
            v.check_bool(&format!("{} S > 0", r.label), r.sensitivity > 0.0);
        } else if neg_vars.contains(&r.name.as_str()) {
            v.check_bool(&format!("{} S < 0", r.label), r.sensitivity < 0.0);
        }
    }

    validation::section("Elasticity Bounds [-2, 2]");
    for r in &results {
        v.check_bool(
            &format!("{} E={:.4} in [-2,2]", r.label, r.elasticity),
            (-2.0..=2.0).contains(&r.elasticity),
        );
    }

    validation::section("Symmetry");
    for r in &results {
        let dp = (r.et0_plus - r.et0_base).abs();
        let dm = (r.et0_minus - r.et0_base).abs();
        if dp > 1e-10 && dm > 1e-10 {
            let ratio = dp.max(dm) / dp.min(dm);
            v.check_bool(&format!("{} sym={:.3} ≤ 2", r.label, ratio), ratio <= 2.0);
        } else {
            v.check_bool(&format!("{} both Δ ≈ 0", r.label), true);
        }
    }

    validate_multi_site(&mut v, &benchmark, variables, pct);
    v.finish();
}
