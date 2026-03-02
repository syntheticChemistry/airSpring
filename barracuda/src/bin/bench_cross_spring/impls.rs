// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark implementations for each domain.

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use airspring_barracuda::eco::anderson;
use airspring_barracuda::eco::crop;
use airspring_barracuda::eco::diversity;
use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::eco::infiltration;
use airspring_barracuda::eco::runoff;
use airspring_barracuda::gpu::device_info;
use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};
use airspring_barracuda::gpu::hargreaves::{BatchedHargreaves, HargreavesDay};
use airspring_barracuda::gpu::isotherm as gpu_iso;
use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
use airspring_barracuda::gpu::reduce::SeasonalReducer;
use airspring_barracuda::gpu::richards::BatchedRichards;
use airspring_barracuda::gpu::stream::{self, StreamSmoother};
use airspring_barracuda::gpu::water_balance::{BatchedWaterBalance, FieldDayInput};

use super::data::{sample_et0_input, sample_station_day, sand_richards_request};

pub fn bench_et0_cpu(n: usize) -> bool {
    let engine = BatchedEt0::cpu();
    let inputs: Vec<airspring_barracuda::eco::evapotranspiration::DailyEt0Input> = (0..n)
        .map(|i| sample_et0_input(1 + (u32::try_from(i).unwrap_or(0) % 365)))
        .collect();
    let result = engine.compute(&inputs);
    result.et0_values.len() == n && result.et0_values.iter().all(|v| v.is_finite() && *v > 0.0)
}

pub fn bench_et0_gpu(device: &Arc<WgpuDevice>, n: usize) -> bool {
    let engine = match BatchedEt0::gpu(Arc::clone(device)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("    GPU init failed: {e}");
            return false;
        }
    };
    let inputs: Vec<StationDay> = (0..n)
        .map(|i| sample_station_day(1 + (u32::try_from(i).unwrap_or(0) % 365)))
        .collect();
    match engine.compute_gpu(&inputs) {
        Ok(result) => {
            let all_finite = result.et0_values.iter().all(|v| v.is_finite());
            let n_f64 = f64::from(u32::try_from(n).unwrap_or(0));
            let mean_positive = result.et0_values.iter().sum::<f64>() / n_f64 > 0.0;
            result.et0_values.len() == n && all_finite && mean_positive
        }
        Err(e) => {
            eprintln!("    GPU dispatch failed: {e}");
            false
        }
    }
}

pub fn bench_et0_parity(device: &Arc<WgpuDevice>, n: usize) -> bool {
    let Ok(gpu_engine) = BatchedEt0::gpu(Arc::clone(device)) else {
        return false;
    };
    let cpu_engine = BatchedEt0::cpu();
    let inputs: Vec<StationDay> = (0..n)
        .map(|i| sample_station_day(100 + u32::try_from(i).unwrap_or(0)))
        .collect();
    let Ok(gpu_result) = gpu_engine.compute_gpu(&inputs) else {
        return false;
    };
    let Ok(cpu_result) = cpu_engine.compute_gpu(&inputs) else {
        return false;
    };
    if gpu_result.et0_values.len() != cpu_result.et0_values.len() {
        return false;
    }
    let max_diff: f64 = gpu_result
        .et0_values
        .iter()
        .zip(&cpu_result.et0_values)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);
    let report = device_info::probe_device(device);
    let tolerance = if report.builtins.exp && report.builtins.log {
        0.05
    } else {
        4.0
    };
    let parity = max_diff < tolerance;
    eprintln!(
        "    CPU↔GPU max diff: {max_diff:.4} mm/day (tol={tolerance}, {:?}, exp={} log={})",
        report.fp64_strategy, report.builtins.exp, report.builtins.log
    );
    parity
}

pub fn bench_wb_cpu_season(days: usize) -> bool {
    let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
    let inputs: Vec<airspring_barracuda::eco::water_balance::DailyInput> = (0..days)
        .map(|day| airspring_barracuda::eco::water_balance::DailyInput {
            precipitation: if day % 5 == 0 { 10.0 } else { 0.0 },
            irrigation: 0.0,
            et0: 2.0f64.mul_add((2.0 * std::f64::consts::PI * day as f64 / 365.0).sin(), 4.0),
            kc: 1.0,
        })
        .collect();
    let summary = engine.simulate_season(&inputs);
    summary.mass_balance_error < 0.01 && summary.daily_outputs.len() == days
}

pub fn bench_wb_gpu_step(device: &Arc<WgpuDevice>, n: usize) -> bool {
    let engine = match BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, Arc::clone(device)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("    GPU init failed: {e}");
            return false;
        }
    };
    let fields: Vec<FieldDayInput> = (0..n)
        .map(|i| FieldDayInput {
            dr_prev: (i as f64 % 80.0),
            precipitation: 2.0,
            irrigation: 0.0,
            etc: 4.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        })
        .collect();
    match engine.gpu_step(&fields) {
        Ok(results) => {
            results.len() == n && results.iter().all(|&dr| (0.0..=100.001).contains(&dr))
        }
        Err(e) => {
            eprintln!("    GPU step failed: {e}");
            false
        }
    }
}

pub fn bench_reduce_gpu(device: &Arc<WgpuDevice>, n: usize) -> bool {
    let reducer = match SeasonalReducer::new(Arc::clone(device)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    GPU reducer init failed: {e}");
            return false;
        }
    };
    let data: Vec<f64> = (0..n)
        .map(|i| 2.0f64.mul_add((i as f64 * 0.01).sin(), 3.0))
        .collect();
    match reducer.compute_stats(&data) {
        Ok(stats) => stats.total > 0.0 && stats.count == n,
        Err(e) => {
            eprintln!("    GPU reduce failed: {e}");
            false
        }
    }
}

pub fn bench_stream_cpu(n: usize, window: usize) -> bool {
    let data: Vec<f64> = (0..n)
        .map(|i| 3.0f64.mul_add((i as f64 * 0.1).sin(), 25.0))
        .collect();
    stream::smooth_cpu(&data, window).is_some()
}

pub fn bench_stream_gpu(device: &Arc<WgpuDevice>, n: usize, window: usize) -> bool {
    let smoother = StreamSmoother::new(Arc::clone(device));
    let data: Vec<f64> = (0..n)
        .map(|i| 3.0f64.mul_add((i as f64 * 0.1).sin(), 25.0))
        .collect();
    match smoother.smooth(&data, window) {
        Ok(result) => !result.mean.is_empty(),
        Err(e) => {
            eprintln!("    Stream smooth failed: {e}");
            false
        }
    }
}

pub fn bench_richards_cpu() -> bool {
    let req = sand_richards_request();
    let results = airspring_barracuda::gpu::richards::solve_batch_cpu(&[req]);
    results.len() == 1 && results[0].is_ok()
}

pub fn bench_richards_upstream() -> bool {
    let req = sand_richards_request();
    BatchedRichards::solve_upstream(&req).is_ok()
}

pub fn bench_richards_cn_diffusion() -> bool {
    let req = sand_richards_request();
    match BatchedRichards::solve_cn_diffusion(&req) {
        Ok(theta) => theta.len() == 20 && theta.iter().all(|&t| (0.04..=0.44).contains(&t)),
        Err(e) => {
            eprintln!("    CN diffusion failed: {e}");
            false
        }
    }
}

pub fn bench_isotherm_nm() -> bool {
    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];
    gpu_iso::fit_langmuir_nm(&ce, &qe).is_some_and(|f| f.r_squared > 0.95)
}

pub fn bench_isotherm_global() -> bool {
    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];
    gpu_iso::fit_langmuir_global(&ce, &qe, 8).is_some_and(|f| f.r_squared > 0.95)
}

pub fn bench_mc_et0() -> bool {
    let input = sample_et0_input(187);
    let result = mc_et0_cpu(&input, &Et0Uncertainties::default(), 5000, 42);
    let (lo, hi) = result.parametric_ci(0.90);
    result.n_samples > 4900 && lo < result.et0_mean && hi > result.et0_mean
}

pub fn bench_hargreaves_batch(n: usize) -> bool {
    let engine = BatchedHargreaves::cpu();
    let inputs: Vec<HargreavesDay> = (0..n)
        .map(|i| HargreavesDay {
            tmax: 0.01f64.mul_add(i as f64, 21.5),
            tmin: 0.005f64.mul_add(i as f64, 12.3),
            latitude_deg: 42.7,
            day_of_year: 1 + (u32::try_from(i).unwrap_or(0) % 365),
        })
        .collect();
    let result = engine.compute(&inputs);
    result.et0_values.len() == n && result.et0_values.iter().all(|v| v.is_finite() && *v > 0.0)
}

pub fn bench_diversity_alpha() -> bool {
    let counts = vec![120.0, 85.0, 45.0, 30.0, 20.0];
    let ad = diversity::alpha_diversity(&counts);
    ad.shannon > 1.0 && ad.simpson > 0.5 && (ad.observed - 5.0).abs() < 1e-10
}

pub fn bench_bray_curtis_matrix() -> bool {
    let samples: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            (0..50)
                .map(|j| (f64::from(i * 50 + j) * 0.1).sin().abs() * 100.0)
                .collect()
        })
        .collect();
    let mat = diversity::bray_curtis_matrix(&samples);
    mat.len() == 400
        && (0..20).all(|i| mat[i * 20 + i].abs() < 1e-12)
        && mat.iter().all(|v| *v >= 0.0 && *v <= 1.0)
}

pub fn bench_shannon_frequencies() -> bool {
    let counts = vec![120.0, 85.0, 45.0, 30.0, 20.0];
    let total: f64 = counts.iter().sum();
    let freqs: Vec<f64> = counts.iter().map(|&c| c / total).collect();
    let h1 = diversity::shannon(&counts);
    let h2 = diversity::shannon_from_frequencies(&freqs);
    (h1 - h2).abs() < 1e-10
}

pub fn bench_crop_kc_stage() -> bool {
    let kc_ini = 0.30;
    let kc_mid = 1.20;
    let all_reasonable = (0..180u32).all(|d| {
        let kc = crop::crop_coefficient_stage(kc_ini, kc_mid, d, 180);
        kc >= kc_ini - 1e-10 && kc <= kc_mid + 1e-10
    });
    let mid = crop::crop_coefficient_stage(kc_ini, kc_mid, 90, 180);
    all_reasonable && (mid - 0.75).abs() < 1e-10
}

pub fn bench_kc_from_gdd() -> bool {
    let params = crop::CropType::Corn.gdd_params();
    let cum_gdd: Vec<f64> = (0..=2700).step_by(100).map(f64::from).collect();
    let kc_vals: Vec<f64> = cum_gdd
        .iter()
        .map(|&g| crop::kc_from_gdd(g, &params.kc_stages_gdd, &params.kc_values).unwrap_or(0.0))
        .collect();
    kc_vals.iter().all(|k| (0.0..=1.5).contains(k))
        && kc_vals.first().is_some_and(|&k| (k - 0.30).abs() < 0.01)
}

pub fn bench_anderson_chain() -> bool {
    let theta_r = 0.045;
    let theta_s = 0.43;
    let theta_series: Vec<f64> = (0..10_000)
        .map(|i| 0.15_f64.mul_add((f64::from(i) * 0.01).sin(), 0.25))
        .collect();
    let results = anderson::coupling_series(&theta_series, theta_r, theta_s);
    let n_extended = results
        .iter()
        .filter(|r| r.regime == anderson::QsRegime::Extended)
        .count();
    let n_localized = results
        .iter()
        .filter(|r| r.regime == anderson::QsRegime::Localized)
        .count();
    n_extended > 0 && n_localized > 0 && results.len() == 10_000
}

pub fn bench_anderson_regimes() -> bool {
    let theta_r = 0.045;
    let theta_s = 0.43;
    let saturated = anderson::coupling_chain(theta_s, theta_r, theta_s);
    let residual = anderson::coupling_chain(theta_r, theta_r, theta_s);
    let mid_theta = 0.5_f64.mul_add(theta_s - theta_r, theta_r);
    let mid = anderson::coupling_chain(mid_theta, theta_r, theta_s);
    saturated.regime == anderson::QsRegime::Extended
        && residual.regime == anderson::QsRegime::Localized
        && mid.d_eff > anderson::D_EFF_CRITICAL
}

pub fn bench_blaney_criddle() -> bool {
    let mut sum = 0.0;
    for i in 0..10_000_i32 {
        let tmean = 5.0 + f64::from(i % 30);
        let lat_rad = (40.0 + f64::from(i % 20) * 0.1).to_radians();
        #[allow(clippy::cast_sign_loss)]
        let doy = (i % 365) as u32 + 1;
        sum += et::blaney_criddle_from_location(tmean, lat_rad, doy);
    }
    let mean_et0 = sum / 10_000.0;
    println!("    mean_ET₀={mean_et0:.3} mm/day");
    mean_et0 > 0.0 && mean_et0 < 15.0
}

pub fn bench_scs_cn() -> bool {
    let mut sum = 0.0;
    for i in 0..10_000 {
        let cn = 30.0 + f64::from(i % 70);
        let precip = f64::from(i % 200);
        sum += runoff::scs_cn_runoff_standard(precip, cn);
    }
    let mean_q = sum / 10_000.0;
    println!("    mean_Q={mean_q:.3} mm");
    mean_q >= 0.0
}

pub fn bench_scs_cn_amc() -> bool {
    let mut all_ordered = true;
    for cn_ii in (30..=95).map(f64::from) {
        let cn_i = runoff::amc_cn_dry(cn_ii);
        let cn_iii = runoff::amc_cn_wet(cn_ii);
        if cn_i >= cn_ii || cn_iii <= cn_ii {
            all_ordered = false;
        }
    }
    println!("    AMC-I < AMC-II < AMC-III: {all_ordered}");
    all_ordered
}

pub fn bench_green_ampt_soils() -> bool {
    let soils = [
        infiltration::GreenAmptParams::SAND,
        infiltration::GreenAmptParams::LOAMY_SAND,
        infiltration::GreenAmptParams::SANDY_LOAM,
        infiltration::GreenAmptParams::LOAM,
        infiltration::GreenAmptParams::SILT_LOAM,
        infiltration::GreenAmptParams::CLAY_LOAM,
        infiltration::GreenAmptParams::CLAY,
    ];
    let mut all_ok = true;
    for p in &soils {
        let f1 = infiltration::cumulative_infiltration(p, 1.0);
        let f10 = infiltration::cumulative_infiltration(p, 10.0);
        let rate = infiltration::infiltration_rate(p, f1);
        if f10 <= f1 || rate < p.ks_cm_hr {
            all_ok = false;
        }
    }
    println!("    7 soils: monotonic + rate≥Ks: {all_ok}");
    all_ok
}

pub fn bench_green_ampt_ponding() -> bool {
    let loam = infiltration::GreenAmptParams {
        delta_theta: 0.405,
        ..infiltration::GreenAmptParams::LOAM
    };
    let tp = infiltration::ponding_time(&loam, 2.0);
    let sand = infiltration::GreenAmptParams::SAND;
    let tp_sand = infiltration::ponding_time(&sand, 5.0);
    println!(
        "    loam tp={tp:.3} hr, sand no-pond={}",
        tp_sand.is_infinite()
    );
    (tp - 0.37).abs() < 0.1 && tp_sand.is_infinite()
}
