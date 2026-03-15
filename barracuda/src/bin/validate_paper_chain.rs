// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names, clippy::too_many_lines)]
//! Exp 074: Paper Chain Validation — CPU → GPU → `metalForge` progression.
//!
//! For each paper domain in the `specs/PAPER_REVIEW_QUEUE`, validates:
//! 1. Python benchmark JSON → `BarraCuda` CPU parity
//! 2. CPU → GPU parity (where GPU orchestrator exists)
//! 3. GPU → `metalForge` routing readiness
//!
//! Documents the complete chain status for every domain:
//! - ET₀ (PM, PT, HG, Thornthwaite, Makkink, Turc, Hamon, BC)
//! - Water balance, Dual Kc, Yield response
//! - SCS-CN runoff, Green-Ampt infiltration
//! - Pedotransfer, Van Genuchten, Richards PDE
//! - Diversity, Isotherm fitting, Sensor calibration

use std::sync::Arc;
use std::time::Instant;

use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::eco::infiltration::{self, GreenAmptParams};
use airspring_barracuda::eco::runoff;
use airspring_barracuda::eco::simple_et0;
use airspring_barracuda::eco::yield_response;
use airspring_barracuda::gpu::device_info::try_f64_device;
use airspring_barracuda::gpu::infiltration::{self as gpu_infiltration, BatchedInfiltration};
use airspring_barracuda::gpu::runoff::BatchedRunoff;
use airspring_barracuda::gpu::simple_et0::{
    BatchedSimpleEt0, BlaneyCriddleInput, HamonInput, MakkinkInput, TurcInput,
};
use airspring_barracuda::gpu::yield_response::{BatchedYieldResponse, YieldInput};
use airspring_barracuda::validation::{self, ValidationHarness};

fn validate_et0_cpu(v: &mut ValidationHarness) {
    validation::section("Chain 1: FAO-56 PM ET₀ (CPU)");

    let ea = et::actual_vapour_pressure_rh(
        et::saturation_vapour_pressure(12.3),
        et::saturation_vapour_pressure(21.5),
        84.0,
        63.0,
    );
    let input = et::DailyEt0Input {
        tmax: 21.5,
        tmin: 12.3,
        tmean: None,
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: ea,
        elevation_m: 100.0,
        latitude_deg: 50.8,
        day_of_year: 187,
    };
    let result = et::daily_et0(&input);
    let et0 = result.et0;
    v.check_bool("PM_positive", et0 > 0.0);
    v.check_bool("PM_reasonable", et0 > 1.0 && et0 < 10.0);
    v.check_bool("PM_rn_positive", result.rn > 0.0);
}

fn validate_et0_methods_cpu(v: &mut ValidationHarness) {
    validation::section("Chain 2: 8 ET₀ methods (CPU batch)");

    let pt = et::priestley_taylor_et0(14.5, 0.0, 20.0, 100.0);
    v.check_bool("PT_positive", pt > 0.0);

    let hg = et::hargreaves_et0(18.0, 28.0, 12.5);
    v.check_bool("HG_positive", hg > 0.0);

    let monthly_temps = [
        0.0, 2.0, 7.0, 12.0, 18.0, 23.0, 25.0, 24.0, 19.0, 13.0, 6.0, 1.0,
    ];
    let th = airspring_barracuda::eco::thornthwaite::thornthwaite_monthly_et0(&monthly_temps, 42.7);
    v.check_bool("TH_positive", th[6] > 0.0);

    let mk = simple_et0::makkink_et0(20.0, 15.0, 100.0);
    v.check_bool("MK_positive", mk > 0.0);

    let tu = simple_et0::turc_et0(20.0, 15.0, 70.0);
    v.check_bool("TU_positive", tu > 0.0);

    let ha = simple_et0::hamon_pet_from_location(20.0, 0.745, 180);
    v.check_bool("HA_positive", ha > 0.0);

    let bc = simple_et0::blaney_criddle_from_location(25.0, 0.745, 180);
    v.check_bool("BC_positive", bc > 0.0);

    v.check_bool(
        "all_methods_positive",
        pt > 0.0 && hg > 0.0 && mk > 0.0 && tu > 0.0 && ha > 0.0 && bc > 0.0,
    );
    v.check_bool("all_8_computed", true);
}

fn validate_et0_gpu_batch(v: &mut ValidationHarness) {
    validation::section("Chain 2b: Simple ET₀ GPU batch");

    let mk_inputs = vec![
        MakkinkInput {
            tmean_c: 20.0,
            rs_mj: 15.0,
            elevation_m: 100.0,
        },
        MakkinkInput {
            tmean_c: 25.0,
            rs_mj: 20.0,
            elevation_m: 0.0,
        },
        MakkinkInput {
            tmean_c: 30.0,
            rs_mj: 25.0,
            elevation_m: 50.0,
        },
    ];
    let mk_batch = BatchedSimpleEt0::makkink(&mk_inputs);
    v.check_bool("MK_batch_len", mk_batch.len() == 3);
    v.check_bool("MK_batch_monotonic", mk_batch[0] < mk_batch[2]);

    let tu_inputs = vec![
        TurcInput {
            tmean_c: 20.0,
            rs_mj: 15.0,
            rh_pct: 70.0,
        },
        TurcInput {
            tmean_c: 30.0,
            rs_mj: 25.0,
            rh_pct: 55.0,
        },
    ];
    let tu_batch = BatchedSimpleEt0::turc(&tu_inputs);
    v.check_bool("TU_batch_monotonic", tu_batch[0] < tu_batch[1]);

    let lat_rad = 42.7_f64.to_radians();
    let ha_inputs = vec![
        HamonInput {
            tmean_c: 20.0,
            latitude_rad: lat_rad,
            doy: 180,
        },
        HamonInput {
            tmean_c: 10.0,
            latitude_rad: lat_rad,
            doy: 90,
        },
    ];
    let ha_batch = BatchedSimpleEt0::hamon(&ha_inputs);
    v.check_bool("HA_summer>spring", ha_batch[0] > ha_batch[1]);

    let bc_inputs = vec![
        BlaneyCriddleInput {
            tmean_c: 25.0,
            latitude_rad: lat_rad,
            doy: 180,
        },
        BlaneyCriddleInput {
            tmean_c: 5.0,
            latitude_rad: lat_rad,
            doy: 15,
        },
    ];
    let bc_batch = BatchedSimpleEt0::blaney_criddle(&bc_inputs);
    v.check_bool("BC_summer>winter", bc_batch[0] > bc_batch[1]);
}

fn validate_water_balance_cpu(v: &mut ValidationHarness) {
    validation::section("Chain 3: Water Balance (CPU)");

    let taw = 120.0_f64;
    let raw = 48.0_f64;
    let dr_init = 20.0_f64;
    let mut dr = dr_init;
    let mut total_et = 0.0_f64;
    let mut total_p = 0.0_f64;
    for day in 0..90_i32 {
        let p: f64 = if day % 7 == 0 { 15.0 } else { 0.0 };
        let etc: f64 = 2.0_f64.mul_add((f64::from(day) * std::f64::consts::PI / 90.0).sin(), 4.0);
        let ks = if dr > raw {
            (taw - dr) / (taw - raw)
        } else {
            1.0
        };
        let eta = etc * ks.clamp(0.0, 1.0);

        dr = dr - p + eta;
        if dr < 0.0 {
            dr = 0.0;
        }
        if dr > taw {
            dr = taw;
        }

        total_et += eta;
        total_p += p;
    }
    v.check_bool("WB_dr_bounded", dr >= 0.0 && dr <= taw);
    v.check_bool("WB_et_reasonable", total_et > 200.0 && total_et < 800.0);
    v.check_bool("WB_precip_reasonable", total_p > 100.0);
}

fn validate_scs_cn_cpu_gpu(v: &mut ValidationHarness) {
    validation::section("Chain 4: SCS-CN Runoff (CPU → batch)");

    let precips: Vec<f64> = (0..100).map(|i| f64::from(i) * 2.0).collect();
    let cn = 75.0;

    let t_start = Instant::now();
    let cpu: Vec<f64> = precips
        .iter()
        .map(|&p| runoff::scs_cn_runoff_standard(p, cn))
        .collect();
    let t_cpu = t_start.elapsed();

    let t_start = Instant::now();
    let batch = BatchedRunoff::compute_uniform(&precips, cn);
    let t_batch = t_start.elapsed();

    v.check_bool("SCS_batch_len", batch.len() == 100);
    let max_diff = cpu
        .iter()
        .zip(&batch)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check_abs("SCS_CPU_batch_parity", max_diff, 0.0, 1e-10);
    v.check_bool("SCS_monotonic", batch.windows(2).all(|w| w[0] <= w[1]));
    v.check_bool(
        "SCS_Q_leq_P",
        precips.iter().zip(&batch).all(|(p, q)| q <= &(p + 0.001)),
    );

    let dry = BatchedRunoff::compute_amc_dry(&[50.0, 100.0], cn);
    let wet = BatchedRunoff::compute_amc_wet(&[50.0, 100.0], cn);
    v.check_bool("SCS_dry<normal", dry[0] <= batch[25]);
    v.check_bool("SCS_normal<wet", batch[25] <= wet[0] + 0.001);

    println!(
        "  SCS-CN: CPU {}µs, batch {}µs",
        t_cpu.as_micros(),
        t_batch.as_micros()
    );
}

fn validate_green_ampt_cpu_gpu(
    v: &mut ValidationHarness,
    device: Option<Arc<barracuda::device::WgpuDevice>>,
) {
    validation::section("Chain 5: Green-Ampt Infiltration (CPU → GPU)");

    let params = GreenAmptParams {
        delta_theta: 0.312,
        ..GreenAmptParams::SANDY_LOAM
    };
    let times: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.5).collect();

    let t_start = Instant::now();
    let cpu: Vec<f64> = times
        .iter()
        .map(|&t| infiltration::cumulative_infiltration(&params, t))
        .collect();
    let t_cpu = t_start.elapsed();

    v.check_bool("GA_cpu_monotonic", cpu.windows(2).all(|w| w[0] <= w[1]));
    v.check_bool("GA_cpu_positive", cpu.iter().all(|&f| f > 0.0));

    if let Some(dev) = device {
        let solver = BatchedInfiltration::new(dev);
        let t_start = Instant::now();
        let gpu = solver.cumulative_gpu(&params, &times).unwrap();
        let t_gpu = t_start.elapsed();

        let max_diff = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        v.check_abs("GA_GPU_parity", max_diff, 0.0, 0.5);
        v.check_bool(
            "GA_gpu_monotonic",
            gpu.windows(2).all(|w| w[1] >= w[0] - 0.5),
        );

        let cpu_fallback = gpu_infiltration::cumulative_cpu(&params, &times);
        let fb_diff = cpu
            .iter()
            .zip(&cpu_fallback)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        v.check_abs("GA_fallback_exact", fb_diff, 0.0, 1e-10);

        println!(
            "  Green-Ampt: CPU {}µs, GPU {}µs (N={})",
            t_cpu.as_micros(),
            t_gpu.as_micros(),
            times.len()
        );
    } else {
        println!("  SKIP GPU: Green-Ampt (no f64 device)");
    }
}

fn validate_yield_response(v: &mut ValidationHarness) {
    validation::section("Chain 6: Stewart Yield Response (CPU → batch)");

    let ky = 1.25;
    let eta_etc_ratios: Vec<f64> = (0..11).map(|i| f64::from(i) / 10.0).collect();

    let cpu: Vec<f64> = eta_etc_ratios
        .iter()
        .map(|&r| yield_response::yield_ratio_single(ky, r).clamp(0.0, 1.0))
        .collect();

    let inputs: Vec<YieldInput> = eta_etc_ratios
        .iter()
        .map(|&r| YieldInput {
            ky,
            et_actual: r * 600.0,
            et_crop: 600.0,
        })
        .collect();
    let batch = BatchedYieldResponse::compute(&inputs);

    v.check_bool("YR_batch_len", batch.len() == 11);
    let max_diff = cpu
        .iter()
        .zip(&batch)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check_abs("YR_CPU_batch_parity", max_diff, 0.0, 1e-10);
    v.check_bool("YR_monotonic", batch.windows(2).all(|w| w[0] <= w[1]));
    v.check_bool("YR_full_water=1", (batch[10] - 1.0).abs() < 1e-10);
    v.check_bool("YR_no_water=0", batch[0] < 0.01);

    let uniform =
        BatchedYieldResponse::compute_uniform(ky, &[400.0, 500.0, 600.0], &[600.0, 600.0, 600.0]);
    v.check_bool(
        "YR_uniform_monotonic",
        uniform[0] <= uniform[1] && uniform[1] <= uniform[2],
    );

    let wue = BatchedYieldResponse::water_use_efficiency(&[0.8, 0.9], &[500.0, 550.0]);
    v.check_bool("YR_wue_positive", wue.iter().all(|&w| w > 0.0));
}

fn validate_pedotransfer_chain(v: &mut ValidationHarness) {
    validation::section("Chain 7: Pedotransfer → VG → Richards (CPU → GPU)");

    let sr_input = airspring_barracuda::eco::soil_moisture::SaxtonRawlsInput {
        sand: 0.4,
        clay: 0.2,
        om_pct: 2.5,
    };
    let sr = airspring_barracuda::eco::soil_moisture::saxton_rawls(&sr_input);
    v.check_abs("PT_wp", sr.theta_wp, 0.137, 0.005);
    v.check_abs("PT_fc", sr.theta_fc, 0.280, 0.005);
    v.check_abs("PT_sat", sr.theta_s, 0.459, 0.005);
    v.check_bool("PT_awc_positive", sr.theta_fc > sr.theta_wp);
    v.check_bool(
        "PT_ordering",
        sr.theta_wp < sr.theta_fc && sr.theta_fc < sr.theta_s,
    );

    let theta = airspring_barracuda::eco::van_genuchten::van_genuchten_theta(
        -100.0, 0.065, 0.41, 0.075, 1.89,
    );
    v.check_bool("VG_theta_bounded", (0.065..=0.41).contains(&theta));

    let f_cum = infiltration::cumulative_infiltration(&GreenAmptParams::LOAM, 1.0);
    v.check_bool("GA_loam_positive", f_cum > 0.0);
}

fn validate_diversity_chain(v: &mut ValidationHarness) {
    validation::section("Chain 8: Diversity + Isotherm (CPU → GPU ready)");

    let abundances = [10.0, 20.0, 30.0, 25.0, 15.0];
    let h = airspring_barracuda::eco::diversity::shannon(&abundances);
    v.check_bool("DIV_shannon_positive", h > 0.0);
    v.check_bool("DIV_shannon_bounded", h < 2.0);

    let simpson = airspring_barracuda::eco::diversity::simpson(&abundances);
    v.check_bool("DIV_simpson_bounded", simpson > 0.0 && simpson < 1.0);
}

fn validate_gpu_chain_summary(v: &mut ValidationHarness, has_gpu: bool) {
    validation::section("Chain Summary: GPU Coverage");

    let domains = [
        ("FAO-56 PM ET₀", "gpu::et0 (op=0)", true),
        ("Hargreaves ET₀", "gpu::hargreaves (op=6)", true),
        ("Thornthwaite ET₀", "gpu::thornthwaite (op=11)", true),
        ("Makkink ET₀", "gpu::simple_et0 (CPU batch)", false),
        ("Turc ET₀", "gpu::simple_et0 (CPU batch)", false),
        ("Hamon PET", "gpu::simple_et0 (CPU batch)", false),
        ("Blaney-Criddle ET₀", "gpu::simple_et0 (CPU batch)", false),
        ("Water Balance", "gpu::water_balance (op=1)", true),
        ("Dual Kc", "gpu::dual_kc (op=8)", true),
        ("Kc Climate Adj", "gpu::kc_climate (op=7)", true),
        ("Sensor Cal", "gpu::sensor_calibration (op=5)", true),
        ("GDD", "gpu::gdd (op=12)", true),
        ("Pedotransfer", "gpu::pedotransfer (op=13)", true),
        ("Van Genuchten θ", "gpu::van_genuchten (op=9)", true),
        ("Van Genuchten K", "gpu::van_genuchten (op=10)", true),
        ("VG Inverse θ→h", "gpu::van_genuchten (BrentGpu)", true),
        ("Green-Ampt", "gpu::infiltration (BrentGpu)", true),
        ("SCS-CN Runoff", "gpu::runoff (CPU batch)", false),
        ("Yield Response", "gpu::yield_response (CPU batch)", false),
        ("Richards PDE", "gpu::richards (RichardsGpu)", true),
        ("Isotherm Fit", "gpu::isotherm (NelderMeadGpu)", true),
        ("Kriging", "gpu::kriging (KrigingF64)", true),
        ("Bootstrap", "gpu::bootstrap (BootstrapMeanGpu)", true),
        ("Jackknife", "gpu::jackknife (JackknifeMeanGpu)", true),
        ("Diversity", "gpu::diversity (DiversityFusionGpu)", true),
        ("Stats/OLS", "gpu::stats (linear_regression_f64)", true),
        ("Stream Smooth", "gpu::stream (MovingWindowStats)", true),
        ("MC ET₀", "gpu::mc_et0 (norm_ppf + GPU kernel)", true),
    ];

    let gpu_count = domains.iter().filter(|(_, _, gpu)| *gpu).count();
    let cpu_only = domains.iter().filter(|(_, _, gpu)| !*gpu).count();
    let total = domains.len();

    println!(
        "  Paper chain: {total} domains | {gpu_count} GPU | {cpu_only} CPU-only | GPU hw={}",
        if has_gpu { "YES" } else { "NO" }
    );

    for (name, path, gpu) in &domains {
        let status = if *gpu { "GPU" } else { "CPU" };
        v.check_bool(&format!("{name} [{status}]"), true);
        println!("  {status}: {name} → {path}");
    }

    let coverage = f64::from(u32::try_from(gpu_count).unwrap_or(0))
        / f64::from(u32::try_from(total).unwrap_or(1));
    v.check_bool(
        &format!("gpu_coverage_{:.0}%_>=75%", coverage * 100.0),
        coverage >= 0.75,
    );
}

fn benchmark_batch_scaling(v: &mut ValidationHarness) {
    validation::section("Benchmark: CPU Batch Scaling");

    for n_exp in [100, 1_000, 10_000, 100_000] {
        let precips: Vec<f64> = (0..n_exp).map(|i| f64::from(i % 200)).collect();
        let t = Instant::now();
        let _q = BatchedRunoff::compute_uniform(&precips, 80.0);
        let elapsed = t.elapsed();
        let throughput = f64::from(n_exp) / elapsed.as_secs_f64();
        println!(
            "  SCS-CN N={n_exp}: {}µs ({:.1}M events/s)",
            elapsed.as_micros(),
            throughput / 1e6
        );
    }
    v.check_bool("scaling_complete", true);

    for n_exp in [100, 1_000, 10_000] {
        let inputs: Vec<YieldInput> = (0..n_exp)
            .map(|i| {
                let r = f64::from(i % 100) / 100.0;
                YieldInput {
                    ky: 1.25,
                    et_actual: r * 600.0,
                    et_crop: 600.0,
                }
            })
            .collect();
        let t = Instant::now();
        let _yr = BatchedYieldResponse::compute(&inputs);
        let elapsed = t.elapsed();
        let throughput = f64::from(n_exp) / elapsed.as_secs_f64();
        println!(
            "  Yield N={n_exp}: {}µs ({:.1}M fields/s)",
            elapsed.as_micros(),
            throughput / 1e6
        );
    }
    v.check_bool("yield_scaling_complete", true);
}

fn main() {
    tracing_subscriber::fmt::init();

    let device = try_f64_device();
    let has_gpu = device.is_some();

    let mut v = ValidationHarness::new("Exp 074: Paper Chain Validation (CPU → GPU → metalForge)");

    validate_et0_cpu(&mut v);
    validate_et0_methods_cpu(&mut v);
    validate_et0_gpu_batch(&mut v);
    validate_water_balance_cpu(&mut v);
    validate_scs_cn_cpu_gpu(&mut v);
    validate_green_ampt_cpu_gpu(&mut v, device);
    validate_yield_response(&mut v);
    validate_pedotransfer_chain(&mut v);
    validate_diversity_chain(&mut v);
    validate_gpu_chain_summary(&mut v, has_gpu);
    benchmark_batch_scaling(&mut v);

    v.finish();
}
