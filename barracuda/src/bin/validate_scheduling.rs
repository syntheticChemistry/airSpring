// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate irrigation scheduling strategy comparison (Exp 014).
//!
//! Benchmark: `control/scheduling/benchmark_scheduling.json`
//! Paper: Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148
//! Python: `control/scheduling/irrigation_scheduling.py`
//!
//! Demonstrates the complete "Penny Irrigation" pipeline:
//!   ET₀ → Kc schedule → water balance → Stewart yield → WUE comparison
//!
//! Uses deterministic weather (sinusoidal ET₀ + periodic rain) to avoid
//! RNG mismatch between Python (numpy MT19937) and Rust.

use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response::{
    clamp_yield_ratio, water_use_efficiency, yield_ratio_single,
};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/scheduling/benchmark_scheduling.json");

fn f64_field(v: &serde_json::Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing f64 key '{key}'"))
}

/// Deterministic Michigan-like weather: sinusoidal ET₀ + periodic rain.
fn generate_deterministic_weather(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut et0 = Vec::with_capacity(n);
    let mut precip = Vec::with_capacity(n);

    for day in 0..n {
        let t = day as f64 / n as f64;
        let seasonal = (std::f64::consts::PI * t).sin();
        et0.push((3.5 + 2.5 * seasonal).max(0.5));
        precip.push(if day % 4 == 0 { 6.0 } else { 0.0 });
    }
    (et0, precip)
}

/// Deterministic Kc schedule matching FAO-56 corn growth stages.
fn kc_schedule(n: usize, kc_ini: f64, kc_mid: f64, kc_end: f64) -> Vec<f64> {
    let l_ini = 30_usize;
    let l_dev = 40_usize;
    let l_mid = 50_usize;
    let l_late = 30_usize;
    let total = l_ini + l_dev + l_mid + l_late;

    (0..n)
        .map(|d| {
            if d < l_ini {
                kc_ini
            } else if d < l_ini + l_dev {
                let frac = (d - l_ini) as f64 / l_dev as f64;
                kc_ini + (kc_mid - kc_ini) * frac
            } else if d < l_ini + l_dev + l_mid {
                kc_mid
            } else if d < total {
                let frac = (d - l_ini - l_dev - l_mid) as f64 / l_late as f64;
                kc_mid + (kc_end - kc_mid) * frac
            } else {
                kc_end
            }
        })
        .collect()
}

struct StrategyResult {
    total_irrigation: f64,
    total_et: f64,
    total_precip: f64,
    stress_days: usize,
    yield_ratio: f64,
    mass_balance_error: f64,
}

struct SimConfig<'a> {
    et0: &'a [f64],
    precip: &'a [f64],
    kc: &'a [f64],
    taw: f64,
    raw: f64,
    ky: f64,
    irrig_depth: f64,
}

fn simulate_strategy(
    cfg: &SimConfig<'_>,
    trigger: &dyn Fn(usize, f64, f64) -> bool,
) -> StrategyResult {
    let n = cfg.et0.len();
    let mut dr: f64 = cfg.raw * 0.5;
    let dr_initial = dr;
    let mut total_irrigation = 0.0_f64;
    let mut total_et = 0.0_f64;
    let mut total_precip = 0.0_f64;
    let mut total_dp = 0.0_f64;
    let mut total_etc = 0.0_f64;
    let mut stress_days = 0_usize;

    for d in 0..n {
        let ks = water_balance::stress_coefficient(dr, cfg.taw, cfg.raw);
        if ks < 0.99 {
            stress_days += 1;
        }

        let etc = cfg.kc[d] * cfg.et0[d];
        let eta = ks * etc;
        total_etc += etc;

        let irr = if trigger(d, dr, cfg.taw) {
            cfg.irrig_depth
        } else {
            0.0
        };

        total_irrigation += irr;
        total_et += eta;
        total_precip += cfg.precip[d];

        let dr_new = dr - cfg.precip[d] - irr + eta;
        if dr_new < 0.0 {
            total_dp += -dr_new;
            dr = 0.0;
        } else {
            dr = dr_new.min(cfg.taw);
        }
    }

    let mb_error = (total_precip + total_irrigation + dr - dr_initial - total_et - total_dp).abs();
    let season_eta_etc = if total_etc > 0.0 {
        total_et / total_etc
    } else {
        1.0
    };
    let yr = clamp_yield_ratio(yield_ratio_single(cfg.ky, season_eta_etc));

    StrategyResult {
        total_irrigation,
        total_et,
        total_precip,
        stress_days,
        yield_ratio: yr,
        mass_balance_error: mb_error,
    }
}

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Stewart Checks");
    let checks = &benchmark["validation_checks"]["analytical_checks"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap();
        let ky = f64_field(tc, "ky");
        let eta_etc = f64_field(tc, "eta_etc");
        let expected = f64_field(tc, "expected_yield_ratio");
        let tol = f64_field(tc, "tolerance");
        let computed = clamp_yield_ratio(yield_ratio_single(ky, eta_etc));
        v.check_abs(label, computed, expected, tol);
    }
}

fn validate_results(
    v: &mut ValidationHarness,
    benchmark: &serde_json::Value,
    strategies: &[(&str, &StrategyResult)],
) {
    validation::section("Mass Balance");
    for (name, r) in strategies {
        v.check_abs(
            &format!("{name} mass_balance"),
            r.mass_balance_error,
            0.0,
            0.01,
        );
    }

    validation::section("Yield Physical Bounds");
    for (name, r) in strategies {
        v.check_bool(
            &format!("{name} yield in [0,1]"),
            (0.0..=1.0).contains(&r.yield_ratio),
        );
    }

    validation::section("Yield Ordering");
    let rainfed = strategies[0].1;
    let mad50 = strategies[1].1;
    let mad60 = strategies[2].1;
    let mad70 = strategies[3].1;
    let growth = strategies[4].1;
    v.check_bool(
        "mad_70 >= rainfed",
        mad70.yield_ratio >= rainfed.yield_ratio - 0.001,
    );
    v.check_bool(
        "growth >= mad_70",
        growth.yield_ratio >= mad70.yield_ratio - 0.001,
    );
    v.check_bool(
        "mad_60 >= growth",
        mad60.yield_ratio >= growth.yield_ratio - 0.001,
    );
    v.check_bool(
        "mad_50 >= mad_60",
        mad50.yield_ratio >= mad60.yield_ratio - 0.001,
    );

    validation::section("Stress Days");
    v.check_bool("rainfed >= 5 stress days", rainfed.stress_days >= 5);
    v.check_bool(
        "mad_50 fewer stress than rainfed",
        mad50.stress_days < rainfed.stress_days,
    );

    validation::section("Irrigation Totals");
    v.check_abs("rainfed irrigation", rainfed.total_irrigation, 0.0, 1e-10);
    v.check_bool(
        "mad_50 in [50,500]",
        (50.0..=500.0).contains(&mad50.total_irrigation),
    );
    v.check_bool(
        "mad_60 in [25,400]",
        (25.0..=400.0).contains(&mad60.total_irrigation),
    );
    v.check_bool(
        "mad_70 in [0,350]",
        (0.0..=350.0).contains(&mad70.total_irrigation),
    );

    validation::section("Water Use Efficiency");
    let target_yield = f64_field(&benchmark["season_parameters"], "target_yield_kg_ha");
    for (name, r) in strategies {
        let ya = r.yield_ratio * target_yield;
        let total_water = r.total_precip + r.total_irrigation;
        if total_water > 0.0 {
            let wue = water_use_efficiency(ya, total_water).unwrap_or(0.0);
            v.check_bool(&format!("{name} WUE > 0"), wue > 0.0);
        }
    }
}

fn main() {
    validation::banner("Irrigation Scheduling Optimization (Exp 014)");
    let mut v = ValidationHarness::new("Scheduling Optimization");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_scheduling.json must parse");

    let crop = &benchmark["crop_parameters"];
    let soil = &benchmark["soil_parameters"];
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let season_len = f64_field(&benchmark["season_parameters"], "length_days") as usize;

    let taw = f64_field(soil, "taw_mm");
    let raw = f64_field(soil, "raw_mm");
    let (et0, precip) = generate_deterministic_weather(season_len);
    let kc = kc_schedule(
        season_len,
        f64_field(crop, "kc_ini"),
        f64_field(crop, "kc_mid"),
        f64_field(crop, "kc_end"),
    );

    println!(
        "  Season: {} days, mean ET₀={:.2} mm/d, total P={:.0} mm",
        season_len,
        et0.iter().sum::<f64>() / season_len as f64,
        precip.iter().sum::<f64>(),
    );

    validate_analytical(&mut v, &benchmark);

    validation::section("Strategy Simulation (deterministic weather)");
    let cfg = SimConfig {
        et0: &et0,
        precip: &precip,
        kc: &kc,
        taw,
        raw,
        ky: f64_field(crop, "ky_total"),
        irrig_depth: f64_field(&benchmark["season_parameters"], "irrigation_depth_mm"),
    };

    let rainfed = simulate_strategy(&cfg, &|_, _, _| false);
    let mad50 = simulate_strategy(&cfg, &|_, dr, tw| dr > 0.50 * tw);
    let mad60 = simulate_strategy(&cfg, &|_, dr, tw| dr > 0.60 * tw);
    let mad70 = simulate_strategy(&cfg, &|_, dr, tw| dr > 0.70 * tw);
    let growth = simulate_strategy(&cfg, &|d, dr, tw| (70..=120).contains(&d) && dr > 0.55 * tw);

    let strategies: [(&str, &StrategyResult); 5] = [
        ("rainfed", &rainfed),
        ("mad_50", &mad50),
        ("mad_60", &mad60),
        ("mad_70", &mad70),
        ("growth_stage", &growth),
    ];

    for (name, r) in &strategies {
        println!(
            "  {:>15}: I={:6.0} mm, ET={:6.1} mm, stress={:3} d, Ya/Ym={:.3}",
            name, r.total_irrigation, r.total_et, r.stress_days, r.yield_ratio,
        );
    }

    validate_results(&mut v, &benchmark, &strategies);
    v.finish();
}
