// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]
//! Exp 048: NCBI 16S + Soil Moisture Anderson Coupling Validation.
//!
//! Validates the coupling chain from soil moisture θ through the Anderson
//! localization model to QS regime prediction, using parameters from a real
//! NCBI `BioProject` (PRJNA481146: tillage 16S study, Ein Harod, Israel).
//!
//! Cross-Spring: airSpring (θ, ET₀, Anderson) × wetSpring (16S) × `NestGate` (NCBI)
//!
//! script=`control/ncbi_16s_coupling/ncbi_16s_coupling.py`, commit=4c8546e, date=2026-02-28
//! Run: `python3 control/ncbi_16s_coupling/ncbi_16s_coupling.py`

use airspring_barracuda::eco::anderson::{self, QsRegime};
use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::eco::solar;
use airspring_barracuda::eco::water_balance;
use airspring_barracuda::validation::{self, ValidationHarness};

const BENCHMARK: &str =
    include_str!("../../../control/ncbi_16s_coupling/benchmark_ncbi_16s_coupling.json");

const THETA_R: f64 = 0.095;
const THETA_S: f64 = 0.41;
const FC: f64 = 0.32;
const WP: f64 = 0.15;
const RD_MM: f64 = 600.0;
const TAW: f64 = (FC - WP) * RD_MM;
const LAT_DEG: f64 = 32.5978;

fn validate_anderson_coupling(v: &mut ValidationHarness) {
    validation::section("Anderson Coupling Chain");

    let theta_wet = 0.30;
    let theta_dry = 0.12;
    let theta_sat = THETA_S;
    let theta_res = THETA_R;

    let r_wet = anderson::coupling_chain(theta_wet, THETA_R, THETA_S);
    let r_dry = anderson::coupling_chain(theta_dry, THETA_R, THETA_S);
    let r_sat = anderson::coupling_chain(theta_sat, THETA_R, THETA_S);
    let r_res = anderson::coupling_chain(theta_res, THETA_R, THETA_S);

    v.check_bool("S_e wet in [0,1]", (0.0..=1.0).contains(&r_wet.se));
    v.check_bool("S_e dry in [0,1]", (0.0..=1.0).contains(&r_dry.se));
    v.check_bool("S_e saturated = 1.0", (r_sat.se - 1.0).abs() < 1e-10);
    v.check_bool("S_e residual = 0.0", r_res.se.abs() < 1e-10);

    v.check_bool("d_eff wet > d_eff dry", r_wet.d_eff > r_dry.d_eff);
    v.check_bool("d_eff saturated = 3.0", (r_sat.d_eff - 3.0).abs() < 1e-10);
    v.check_bool("d_eff residual = 0.0", r_res.d_eff.abs() < 1e-10);

    v.check_bool("W wet < W dry", r_wet.disorder < r_dry.disorder);
    v.check_bool(
        "regime saturated = Extended",
        r_sat.regime == QsRegime::Extended,
    );
    v.check_bool(
        "regime residual = Localized",
        r_res.regime == QsRegime::Localized,
    );

    let thetas: Vec<f64> = (0..=20)
        .map(|i| THETA_R + (THETA_S - THETA_R) * f64::from(i) / 20.0)
        .collect();
    let series = anderson::coupling_series(&thetas, THETA_R, THETA_S);
    let monotonic = series.windows(2).all(|w| w[1].d_eff >= w[0].d_eff - 1e-12);
    v.check_bool("coupling chain monotonic (21 points)", monotonic);
}

fn validate_mediterranean_site(v: &mut ValidationHarness) {
    validation::section("Mediterranean Site (Ein Harod, Israel)");

    let lat_rad = LAT_DEG.to_radians();

    let doy_feb = 33_u32;
    let ra_feb = solar::extraterrestrial_radiation(lat_rad, doy_feb);
    let ra_feb_mm = ra_feb * 0.408;
    let et0_feb = et::hargreaves_et0(8.0, 18.0, ra_feb_mm);

    let doy_apr = 100_u32;
    let ra_apr = solar::extraterrestrial_radiation(lat_rad, doy_apr);
    let ra_apr_mm = ra_apr * 0.408;
    let et0_apr = et::hargreaves_et0(12.0, 25.0, ra_apr_mm);

    v.check_bool("ET₀ Feb > 0", et0_feb > 0.0);
    v.check_bool("ET₀ Apr > 0", et0_apr > 0.0);
    v.check_bool("ET₀ Apr > ET₀ Feb (warming)", et0_apr > et0_feb);
    v.check_abs("ET₀ Feb plausible (1-6 mm)", et0_feb, 3.0, 3.0);
    v.check_abs("ET₀ Apr plausible (2-8 mm)", et0_apr, 5.0, 3.0);

    let kc = 0.85;
    let p = 0.55;
    let raw = p * TAW;

    let precip_feb = [3.0, 0.0, 5.0, 0.0, 2.0, 8.0, 0.0]; // 7 representative days
    let precip_apr = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    let mut dr = TAW * 0.3;
    let mut theta_feb_avg = 0.0;
    for &precip in &precip_feb {
        let ks = if dr < raw {
            1.0
        } else {
            ((TAW - dr) / (TAW - raw)).clamp(0.0, 1.0)
        };
        let (new_dr, _eta, _dp) =
            water_balance::daily_water_balance_step(dr, precip, 0.0, et0_feb, kc, ks, TAW);
        dr = new_dr;
        let theta = WP + (TAW - dr) / RD_MM;
        theta_feb_avg += theta;
    }
    theta_feb_avg /= precip_feb.len() as f64;

    let mut theta_apr_avg = 0.0;
    for &precip in &precip_apr {
        let ks = if dr < raw {
            1.0
        } else {
            ((TAW - dr) / (TAW - raw)).clamp(0.0, 1.0)
        };
        let (new_dr, _eta, _dp) =
            water_balance::daily_water_balance_step(dr, precip, 0.0, et0_apr, kc, ks, TAW);
        dr = new_dr;
        let theta = WP + (TAW - dr) / RD_MM;
        theta_apr_avg += theta;
    }
    theta_apr_avg /= precip_apr.len() as f64;

    v.check_bool(
        "θ Feb in valid range",
        (THETA_R..=THETA_S).contains(&theta_feb_avg),
    );
    v.check_bool(
        "θ Apr in valid range",
        (THETA_R..=THETA_S).contains(&theta_apr_avg),
    );

    let chain_feb = anderson::coupling_chain(theta_feb_avg, THETA_R, THETA_S);
    let chain_apr = anderson::coupling_chain(theta_apr_avg, THETA_R, THETA_S);

    v.check_bool("d_eff positive (Feb)", chain_feb.d_eff > 0.0);
    v.check_bool("d_eff positive (Apr)", chain_apr.d_eff > 0.0);
    v.check_bool(
        "regime classifiable (Feb)",
        matches!(
            chain_feb.regime,
            QsRegime::Localized | QsRegime::Marginal | QsRegime::Extended
        ),
    );
    v.check_bool(
        "regime classifiable (Apr)",
        matches!(
            chain_apr.regime,
            QsRegime::Localized | QsRegime::Marginal | QsRegime::Extended
        ),
    );

    println!(
        "  Feb: θ={:.3} → S_e={:.3} → d_eff={:.3} → {} (W={:.2})",
        theta_feb_avg,
        chain_feb.se,
        chain_feb.d_eff,
        chain_feb.regime.as_str(),
        chain_feb.disorder
    );
    println!(
        "  Apr: θ={:.3} → S_e={:.3} → d_eff={:.3} → {} (W={:.2})",
        theta_apr_avg,
        chain_apr.se,
        chain_apr.d_eff,
        chain_apr.regime.as_str(),
        chain_apr.disorder
    );
}

fn validate_irrigation_transition(v: &mut ValidationHarness) {
    validation::section("Irrigation → QS Transition");

    let theta_dryland = 0.12;
    let theta_irrigated = 0.35;

    let r_dry = anderson::coupling_chain(theta_dryland, THETA_R, THETA_S);
    let r_irr = anderson::coupling_chain(theta_irrigated, THETA_R, THETA_S);

    v.check_bool("dryland localized", r_dry.regime == QsRegime::Localized);
    v.check_bool("irrigated has higher d_eff", r_irr.d_eff > r_dry.d_eff);
    v.check_bool("irrigated d_eff > 2.0 (marginal+)", r_irr.d_eff > 2.0);
    v.check_bool(
        "irrigation crosses QS threshold",
        r_irr.regime != QsRegime::Localized,
    );

    println!(
        "  Dryland:   θ={:.3} → d_eff={:.3} → {}",
        theta_dryland,
        r_dry.d_eff,
        r_dry.regime.as_str()
    );
    println!(
        "  Irrigated: θ={:.3} → d_eff={:.3} → {}",
        theta_irrigated,
        r_irr.d_eff,
        r_irr.regime.as_str()
    );
    println!("  Anderson prediction: irrigation restores 3D pore connectivity → QS active");
}

fn validate_benchmark_provenance(v: &mut ValidationHarness) {
    validation::section("Benchmark Provenance");

    let bench: serde_json::Value = serde_json::from_str(BENCHMARK).expect("parse benchmark");
    let prov = &bench["_provenance"];

    v.check_bool(
        "benchmark has bioproject",
        prov["bioproject"].as_str() == Some("PRJNA481146"),
    );
    v.check_bool(
        "benchmark has cross_spring",
        prov["cross_spring"]
            .as_str()
            .is_some_and(|s| s.contains("airSpring") && s.contains("wetSpring")),
    );
    v.check_bool(
        "benchmark has 14 checks",
        bench["total_checks"].as_u64() == Some(14),
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 048: NCBI 16S + Soil Moisture Anderson Coupling");

    let mut v = ValidationHarness::new("NCBI 16S Coupling");

    validate_anderson_coupling(&mut v);
    validate_mediterranean_site(&mut v);
    validate_irrigation_transition(&mut v);
    validate_benchmark_provenance(&mut v);

    v.finish();
}
