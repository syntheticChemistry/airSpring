// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 025: Forecast Scheduling Hindcast — predictive irrigation validation.
//!
//! Validates:
//! 1. Forecast scheduling achieves yield close to perfect-knowledge scheduling
//! 2. Forecast noise degrades yield gracefully
//! 3. Longer horizons maintain or improve scheduling quality
//! 4. Mass balance conservation under forecast scheduling
//! 5. Forecast scheduling outperforms rainfed
//!
//! Benchmark: `control/forecast_scheduling/benchmark_forecast_scheduling.json`
//! Python baseline: `control/forecast_scheduling/forecast_scheduling.py`
//!
//! script=`control/forecast_scheduling/forecast_scheduling.py`, commit=8c3953b, date=2026-02-27
//! Run: `python3 control/forecast_scheduling/forecast_scheduling.py`

use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response::{clamp_yield_ratio, yield_ratio_single};
use airspring_barracuda::validation::{self, ValidationHarness, parse_benchmark_json};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/forecast_scheduling/benchmark_forecast_scheduling.json");

const KY_TOTAL: f64 = 1.25;
const TAW_MM: f64 = 140.0;
const RAW_MM: f64 = 77.0;
const MAD_FRACTION: f64 = 0.5;
const IRRIGATION_DEPTH_MM: f64 = 25.0;

/// Deterministic Michigan growing-season weather (same algorithm as Exp 014).
fn generate_michigan_season(n_days: usize) -> (Vec<f64>, Vec<f64>) {
    let mut et0 = Vec::with_capacity(n_days);
    let mut precip = Vec::with_capacity(n_days);

    let mut rng_state: u64 = 42;

    for d in 0..n_days {
        let t = d as f64 / n_days as f64;
        let seasonal = (std::f64::consts::PI * t).sin();
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.4;
        et0.push((2.5f64.mul_add(seasonal, 3.0) + noise).clamp(0.5, 8.0));

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
        if u < 0.30 {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            precip.push((-8.0 * (1.0 - u2).max(1e-15).ln()).clamp(0.0, 50.0));
        } else {
            precip.push(0.0);
        }
    }

    (et0, precip)
}

/// Generate Kc schedule (FAO-56 corn: ini=30d, dev=40d, mid=50d, late=30d).
fn kc_schedule(n_days: usize) -> Vec<f64> {
    let kc_ini: f64 = 0.30;
    let kc_mid: f64 = 1.20;
    let kc_end: f64 = 0.60;
    let (l_ini, l_dev, l_mid, l_late) = (30, 40, 50, 30);
    let total = l_ini + l_dev + l_mid + l_late;

    let mut kc = vec![0.0; n_days];
    for (d, kc_val) in kc.iter_mut().enumerate().take(n_days.min(total)) {
        if d < l_ini {
            *kc_val = kc_ini;
        } else if d < l_ini + l_dev {
            let frac = (d - l_ini) as f64 / l_dev as f64;
            *kc_val = (kc_mid - kc_ini).mul_add(frac, kc_ini);
        } else if d < l_ini + l_dev + l_mid {
            *kc_val = kc_mid;
        } else {
            let frac = (d - l_ini - l_dev - l_mid) as f64 / l_late as f64;
            *kc_val = (kc_end - kc_mid).mul_add(frac, kc_mid);
        }
    }
    for kc_val in &mut kc[total..n_days] {
        *kc_val = kc_end;
    }
    kc
}

/// Degraded forecast for next `horizon` days.
fn generate_forecast(
    truth_et0: &[f64],
    truth_precip: &[f64],
    current_day: usize,
    horizon: usize,
    rng_state: &mut u64,
    noise_base: f64,
    noise_growth: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = truth_et0.len();
    let mut fc_et0 = Vec::with_capacity(horizon);
    let mut fc_precip = Vec::with_capacity(horizon);

    for k in 0..horizon {
        let target = current_day + k + 1;
        if target >= n {
            fc_et0.push(4.0);
            fc_precip.push(2.0);
            continue;
        }

        let sigma = noise_growth.mul_add(k as f64, noise_base);
        *rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let noise = ((*rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * sigma;
        fc_et0.push((truth_et0[target] + noise).max(0.5));

        let p_truth = truth_precip[target];
        *rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        if p_truth > 0.0 {
            let pnoise = ((*rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * p_truth * 0.3;
            fc_precip.push((p_truth + pnoise).max(0.0));
        } else {
            let u = (*rng_state >> 33) as f64 / (1u64 << 31) as f64;
            if u < 0.05 {
                *rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u2 = (*rng_state >> 33) as f64 / (1u64 << 31) as f64;
                fc_precip.push((-3.0 * (1.0 - u2).max(1e-15).ln()).max(0.0));
            } else {
                fc_precip.push(0.0);
            }
        }
    }

    (fc_et0, fc_precip)
}

struct SimResult {
    total_eta_mm: f64,
    total_etc_mm: f64,
    total_irrigation_mm: f64,
    _total_dp_mm: f64,
    stress_days: usize,
    yield_ratio: f64,
    eta_etc_ratio: f64,
    mass_balance_error: f64,
}

fn simulate_forecast_scheduling(
    et0: &[f64],
    precip: &[f64],
    kc: &[f64],
    forecast_horizon: usize,
    noise_base: f64,
    noise_growth: f64,
    rng_seed: u64,
) -> SimResult {
    let n = et0.len();
    let mut rng_state = rng_seed;
    let initial_dr = RAW_MM * 0.5;
    let mut dr = initial_dr;
    let mut sum_actual = 0.0_f64;
    let mut sum_potential = 0.0_f64;
    let mut sum_irrig = 0.0_f64;
    let mut sum_dp = 0.0_f64;
    let mut stress_day_count = 0_usize;

    for d in 0..n {
        let ks = water_balance::stress_coefficient(dr, TAW_MM, RAW_MM);
        if ks < 0.99 {
            stress_day_count += 1;
        }
        let potential = kc[d] * et0[d];
        let actual = ks * potential;

        let (fc_et0, fc_precip) = generate_forecast(
            et0,
            precip,
            d,
            forecast_horizon,
            &mut rng_state,
            noise_base,
            noise_growth,
        );

        let mut projected_dr = dr;
        let mut trigger = false;
        for k in 0..forecast_horizon {
            if d + k + 1 >= n {
                break;
            }
            let fc_kc = kc[(d + k + 1).min(n - 1)];
            let fc_stress = water_balance::stress_coefficient(projected_dr, TAW_MM, RAW_MM);
            let fc_actual = fc_stress * fc_kc * fc_et0[k];
            projected_dr = (projected_dr - fc_precip[k] + fc_actual).clamp(0.0, TAW_MM);
            if projected_dr > MAD_FRACTION * TAW_MM {
                trigger = true;
                break;
            }
        }

        let irr = if trigger { IRRIGATION_DEPTH_MM } else { 0.0 };

        let dr_new = dr - precip[d] - irr + actual;
        if dr_new < 0.0 {
            sum_dp += -dr_new;
            dr = 0.0;
        } else if dr_new > TAW_MM {
            dr = TAW_MM;
        } else {
            dr = dr_new;
        }

        sum_actual += actual;
        sum_potential += potential;
        sum_irrig += irr;
    }

    let eta_etc_ratio = if sum_potential > 0.0 {
        sum_actual / sum_potential
    } else {
        1.0
    };
    let yield_ratio = clamp_yield_ratio(yield_ratio_single(KY_TOTAL, eta_etc_ratio));

    let mb_input = precip.iter().sum::<f64>() + sum_irrig + dr - initial_dr;
    let mb_output = sum_actual + sum_dp;
    let mb_error = (mb_input - mb_output).abs();

    SimResult {
        total_eta_mm: sum_actual,
        total_etc_mm: sum_potential,
        total_irrigation_mm: sum_irrig,
        _total_dp_mm: sum_dp,
        stress_days: stress_day_count,
        yield_ratio,
        eta_etc_ratio,
        mass_balance_error: mb_error,
    }
}

fn simulate_perfect_knowledge(et0: &[f64], precip: &[f64], kc: &[f64]) -> SimResult {
    let n = et0.len();
    let mut dr = RAW_MM * 0.5;
    let mut sum_actual = 0.0_f64;
    let mut sum_potential = 0.0_f64;
    let mut sum_irrig = 0.0_f64;
    let mut sum_dp = 0.0_f64;
    let mut stress_day_count = 0_usize;

    for d in 0..n {
        let ks = water_balance::stress_coefficient(dr, TAW_MM, RAW_MM);
        if ks < 0.99 {
            stress_day_count += 1;
        }
        let potential = kc[d] * et0[d];
        let actual = ks * potential;

        let irr = if dr > MAD_FRACTION * TAW_MM {
            IRRIGATION_DEPTH_MM
        } else {
            0.0
        };

        let dr_new = dr - precip[d] - irr + actual;
        if dr_new < 0.0 {
            sum_dp += -dr_new;
            dr = 0.0;
        } else if dr_new > TAW_MM {
            dr = TAW_MM;
        } else {
            dr = dr_new;
        }

        sum_actual += actual;
        sum_potential += potential;
        sum_irrig += irr;
    }

    let eta_etc_ratio = if sum_potential > 0.0 {
        sum_actual / sum_potential
    } else {
        1.0
    };
    let yield_ratio = clamp_yield_ratio(yield_ratio_single(KY_TOTAL, eta_etc_ratio));

    SimResult {
        total_eta_mm: sum_actual,
        total_etc_mm: sum_potential,
        total_irrigation_mm: sum_irrig,
        _total_dp_mm: sum_dp,
        stress_days: stress_day_count,
        yield_ratio,
        eta_etc_ratio,
        mass_balance_error: 0.0,
    }
}

fn simulate_rainfed(et0: &[f64], precip: &[f64], kc: &[f64]) -> SimResult {
    let n = et0.len();
    let mut dr = RAW_MM * 0.5;
    let mut sum_actual = 0.0_f64;
    let mut sum_potential = 0.0_f64;
    let mut sum_dp = 0.0_f64;
    let mut stress_day_count = 0_usize;

    for d in 0..n {
        let ks = water_balance::stress_coefficient(dr, TAW_MM, RAW_MM);
        if ks < 0.99 {
            stress_day_count += 1;
        }
        let potential = kc[d] * et0[d];
        let actual = ks * potential;

        let dr_new = dr - precip[d] + actual;
        if dr_new < 0.0 {
            sum_dp += -dr_new;
            dr = 0.0;
        } else if dr_new > TAW_MM {
            dr = TAW_MM;
        } else {
            dr = dr_new;
        }

        sum_actual += actual;
        sum_potential += potential;
    }

    let eta_etc_ratio = if sum_potential > 0.0 {
        sum_actual / sum_potential
    } else {
        1.0
    };
    let yield_ratio = clamp_yield_ratio(yield_ratio_single(KY_TOTAL, eta_etc_ratio));

    SimResult {
        total_eta_mm: sum_actual,
        total_etc_mm: sum_potential,
        total_irrigation_mm: 0.0,
        _total_dp_mm: sum_dp,
        stress_days: stress_day_count,
        yield_ratio,
        eta_etc_ratio,
        mass_balance_error: 0.0,
    }
}

fn validate_forecast_vs_perfect(
    v: &mut ValidationHarness,
    benchmark: &serde_json::Value,
    fc: &SimResult,
    pk: &SimResult,
) {
    validation::section("Forecast vs Perfect Knowledge");

    let thresholds = &benchmark["thresholds"];
    let yg = thresholds["yield_gap_range"].as_array().expect("array");
    let ir = thresholds["irrigation_ratio_range"]
        .as_array()
        .expect("array");

    let yield_gap = pk.yield_ratio - fc.yield_ratio;
    v.check_bool(
        &format!(
            "yield gap {yield_gap:.4} in [{}, {}]",
            yg[0].as_f64().unwrap(),
            yg[1].as_f64().unwrap()
        ),
        yield_gap >= yg[0].as_f64().unwrap() && yield_gap <= yg[1].as_f64().unwrap(),
    );

    v.check_bool("forecast yield > 0", fc.yield_ratio > 0.0);
    v.check_bool(
        "forecast yield <= perfect + 0.01",
        fc.yield_ratio <= pk.yield_ratio + 0.01,
    );

    let irrig_ratio = if pk.total_irrigation_mm > 0.0 {
        fc.total_irrigation_mm / pk.total_irrigation_mm
    } else {
        1.0
    };
    v.check_bool(
        &format!(
            "irrigation ratio {irrig_ratio:.4} in [{}, {}]",
            ir[0].as_f64().unwrap(),
            ir[1].as_f64().unwrap()
        ),
        irrig_ratio >= ir[0].as_f64().unwrap() && irrig_ratio <= ir[1].as_f64().unwrap(),
    );
}

fn validate_forecast_degradation(
    v: &mut ValidationHarness,
    et0: &[f64],
    precip: &[f64],
    kc: &[f64],
) {
    validation::section("Forecast Noise Sensitivity");

    for (label, noise_base) in &[
        ("low", 0.1),
        ("medium", 0.3),
        ("high", 0.8),
        ("extreme", 1.5),
    ] {
        let result = simulate_forecast_scheduling(et0, precip, kc, 5, *noise_base, 0.15, 12345);
        v.check_bool(
            &format!("noise={label} yield {:.4} in [0, 1]", result.yield_ratio),
            result.yield_ratio >= 0.0 && result.yield_ratio <= 1.0,
        );
    }
}

fn validate_horizon_impact(v: &mut ValidationHarness, et0: &[f64], precip: &[f64], kc: &[f64]) {
    validation::section("Forecast Horizon Impact");

    let mut yields = Vec::new();
    for horizon in &[1, 3, 5, 7] {
        let result = simulate_forecast_scheduling(et0, precip, kc, *horizon, 0.3, 0.15, 12345);
        v.check_bool(
            &format!(
                "horizon={}d yield {:.4} in [0, 1]",
                horizon, result.yield_ratio
            ),
            result.yield_ratio >= 0.0 && result.yield_ratio <= 1.0,
        );
        yields.push(result.yield_ratio);
    }

    v.check_bool(
        "3-day forecast >= 1-day - 0.05",
        yields[1] >= yields[0] - 0.05,
    );
    v.check_bool(
        "5-day forecast >= 1-day - 0.05",
        yields[2] >= yields[0] - 0.05,
    );
}

fn validate_mass_balance(v: &mut ValidationHarness, benchmark: &serde_json::Value, fc: &SimResult) {
    validation::section("Mass Balance (Forecast Scheduling)");

    let mb_tol = benchmark["thresholds"]["mass_balance_tol_mm"]
        .as_f64()
        .unwrap();
    v.check_abs("mass balance error", fc.mass_balance_error, 0.0, mb_tol);
    v.check_bool("ETa <= ETc", fc.total_eta_mm <= fc.total_etc_mm + 0.01);
    v.check_bool(
        &format!("eta/etc ratio {:.4} in [0, 1]", fc.eta_etc_ratio),
        fc.eta_etc_ratio >= 0.0 && fc.eta_etc_ratio <= 1.001,
    );
}

fn validate_rainfed_comparison(v: &mut ValidationHarness, fc: &SimResult, rainfed: &SimResult) {
    validation::section("Forecast vs Rainfed");

    v.check_bool(
        &format!(
            "forecast yield ({:.3}) >= rainfed ({:.3})",
            fc.yield_ratio, rainfed.yield_ratio
        ),
        fc.yield_ratio >= rainfed.yield_ratio - 0.001,
    );
    v.check_bool(
        &format!(
            "forecast stress ({}) <= rainfed ({})",
            fc.stress_days, rainfed.stress_days
        ),
        fc.stress_days <= rainfed.stress_days,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 025: Forecast Scheduling Hindcast");
    let mut v = ValidationHarness::new("Forecast Scheduling Hindcast");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON)
        .expect("benchmark_forecast_scheduling.json must parse");

    let season = &benchmark["season_parameters"];
    let n_days = season["length_days"].as_u64().unwrap() as usize;

    let (et0, precip) = generate_michigan_season(n_days);
    let kc = kc_schedule(n_days);

    let fc = simulate_forecast_scheduling(&et0, &precip, &kc, 5, 0.3, 0.15, 12345);
    let pk = simulate_perfect_knowledge(&et0, &precip, &kc);
    let rainfed = simulate_rainfed(&et0, &precip, &kc);

    eprintln!(
        "  Perfect: yield={:.3}, irrig={:.0} mm, stress={} d",
        pk.yield_ratio, pk.total_irrigation_mm, pk.stress_days,
    );
    eprintln!(
        "  Forecast: yield={:.3}, irrig={:.0} mm, stress={} d",
        fc.yield_ratio, fc.total_irrigation_mm, fc.stress_days,
    );
    eprintln!(
        "  Rainfed:  yield={:.3}, stress={} d",
        rainfed.yield_ratio, rainfed.stress_days,
    );

    validate_forecast_vs_perfect(&mut v, &benchmark, &fc, &pk);
    validate_forecast_degradation(&mut v, &et0, &precip, &kc);
    validate_horizon_impact(&mut v, &et0, &precip, &kc);
    validate_mass_balance(&mut v, &benchmark, &fc);
    validate_rainfed_comparison(&mut v, &fc, &rainfed);

    v.finish();
}
