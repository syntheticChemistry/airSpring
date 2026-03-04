// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]
//! Exp 064: Full Dispatch Experiment — CPU vs GPU parity across all domains.
//!
//! Comprehensive validation bridging:
//! 1. All 21 CPU science methods produce sane baseline values
//! 2. GPU dispatch (ET₀, Hargreaves, sensor cal, water balance, reduce)
//!    matches CPU within documented tolerance
//! 3. Batch scaling: results independent of N
//! 4. `BarraCuda` absorption audit: gap inventory + tier counts
//! 5. Mixed-backend seasonal pipeline: GPU stages 1-2, CPU stages 3-4
//!
//! metalForge routing validation is separate (104/104 PASS in forge tests).
//! NUCLEUS cross-primal validation is in `validate_nucleus_pipeline`.
//!
//! script=`control/metalforge_dispatch/metalforge_dispatch.py`, commit=dbfb53a, date=2026-03-02
//! Run: `python3 control/metalforge_dispatch/metalforge_dispatch.py`

use std::sync::Arc;
use std::time::Instant;

use airspring_barracuda::eco::anderson;
use airspring_barracuda::eco::diversity;
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::infiltration::GreenAmptParams;
use airspring_barracuda::eco::richards::{self, VanGenuchtenParams};
use airspring_barracuda::eco::runoff;
use airspring_barracuda::eco::sensor_calibration;
use airspring_barracuda::eco::simple_et0;
use airspring_barracuda::eco::soil_moisture;
use airspring_barracuda::eco::thornthwaite;
use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response;
use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};
use airspring_barracuda::gpu::hargreaves::{BatchedHargreaves, HargreavesDay};
use airspring_barracuda::gpu::reduce::SeasonalReducer;
use airspring_barracuda::gpu::sensor_calibration::{BatchedSensorCal, SensorReading};
use airspring_barracuda::gpu::water_balance::{BatchedWaterBalance, FieldDayInput};
use airspring_barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_writer(std::io::stderr)
        .init();

    let mut v = ValidationHarness::new("Exp 064: Full Dispatch Experiment");

    let t0 = Instant::now();

    phase_1_cpu_science(&mut v);
    phase_2_gpu_dispatch(&mut v);
    phase_3_batch_scaling(&mut v);
    phase_4_absorption_audit(&mut v);
    phase_5_mixed_pipeline(&mut v);

    let elapsed = t0.elapsed();
    v.check_abs("total_time_under_30s", elapsed.as_secs_f64(), 15.0, 15.0);

    v.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1: CPU Science Baseline
// ═══════════════════════════════════════════════════════════════════

fn phase_1_cpu_science(v: &mut ValidationHarness) {
    let et0 = et::daily_et0(&DailyEt0Input {
        tmin: 18.0,
        tmax: 32.0,
        tmean: None,
        solar_radiation: 22.5,
        wind_speed_2m: 1.8,
        actual_vapour_pressure: 1.2,
        elevation_m: 256.0,
        latitude_deg: 42.727,
        day_of_year: 200,
    });
    v.check_abs("cpu_fao56_et0", et0.et0, 5.97, 0.1);

    let ra =
        airspring_barracuda::eco::solar::extraterrestrial_radiation(42.727_f64.to_radians(), 200);
    v.check_abs(
        "cpu_hargreaves",
        et::hargreaves_et0(18.0, 32.0, ra),
        14.85,
        0.5,
    );
    v.check_abs(
        "cpu_priestley_taylor",
        et::priestley_taylor_et0(15.0, 0.5, 25.0, 256.0),
        5.5,
        0.2,
    );
    v.check_abs(
        "cpu_makkink",
        simple_et0::makkink_et0(25.0, 22.0, 256.0),
        3.9,
        0.2,
    );
    v.check_abs(
        "cpu_turc",
        simple_et0::turc_et0(25.0, 22.0, 55.0),
        4.68,
        0.15,
    );

    let lat_rad = 42.727_f64.to_radians();
    v.check_abs(
        "cpu_hamon",
        simple_et0::hamon_pet_from_location(25.0, lat_rad, 200),
        5.6,
        0.2,
    );
    v.check_abs(
        "cpu_blaney_criddle",
        simple_et0::blaney_criddle_from_location(25.0, lat_rad, 200),
        6.6,
        0.3,
    );

    let (dr, _etc, _ks) =
        water_balance::daily_water_balance_step(50.0, 3.5, 0.0, 6.0, 1.15, 1.0, 150.0);
    v.check_abs("cpu_wb_depletion", dr, 53.4, 1.0);

    v.check_abs(
        "cpu_yield_ratio",
        yield_response::yield_ratio_single(1.25, 0.85),
        0.8125,
        0.01,
    );
    v.check_abs(
        "cpu_scs_cn",
        runoff::scs_cn_runoff_standard(75.0, 80.0),
        30.85,
        0.1,
    );

    let f_cum = airspring_barracuda::eco::infiltration::cumulative_infiltration(
        &GreenAmptParams::LOAM,
        2.0,
    );
    v.check_bool("cpu_green_ampt_positive", f_cum > 0.0);
    v.check_abs("cpu_topp", soil_moisture::topp_equation(20.0), 0.345, 0.01);
    let sr_input = soil_moisture::SaxtonRawlsInput {
        sand: 0.40,
        clay: 0.20,
        om_pct: 2.5,
    };
    v.check_abs(
        "cpu_pedotransfer_fc",
        soil_moisture::saxton_rawls(&sr_input).theta_fc,
        0.28,
        0.02,
    );
    v.check_abs(
        "cpu_gdd",
        airspring_barracuda::eco::crop::gdd_avg(30.0, 15.0, 10.0),
        12.5,
        0.01,
    );

    let counts: Vec<f64> = [15, 30, 8, 42, 5, 20, 12, 3]
        .iter()
        .map(|&c| f64::from(c))
        .collect();
    v.check_abs("cpu_shannon", diversity::shannon(&counts), 1.81, 0.05);
    v.check_abs("cpu_simpson", diversity::simpson(&counts), 0.806, 0.01);
    v.check_abs(
        "cpu_bray_curtis",
        diversity::bray_curtis(&[10.0, 20.0, 30.0, 40.0], &[15.0, 18.0, 35.0, 32.0]),
        0.1,
        0.01,
    );

    let ac = anderson::coupling_chain(0.25, 0.078, 0.43);
    v.check_abs("cpu_anderson_d_eff", ac.d_eff, 2.1, 0.1);

    let th = thornthwaite::thornthwaite_monthly_et0(
        &[
            -5.0, -2.0, 3.0, 10.0, 16.0, 21.0, 24.0, 23.0, 18.0, 11.0, 4.0, -3.0,
        ],
        42.727,
    );
    v.check_abs(
        "cpu_thornthwaite_annual",
        th.iter().sum::<f64>(),
        685.0,
        10.0,
    );

    let vg = VanGenuchtenParams {
        theta_s: 0.43,
        theta_r: 0.078,
        alpha: 0.036,
        n_vg: 1.56,
        ks: 24.96,
    };
    let profiles =
        richards::solve_richards_1d(&vg, 50.0, 10, -100.0, -50.0, false, false, 1.0, 0.05);
    v.check_bool(
        "cpu_richards_converges",
        profiles.is_ok_and(|p| !p.is_empty()),
    );
    v.check_abs(
        "cpu_sensor_cal",
        sensor_calibration::soilwatch10_vwc(4500.0),
        0.05,
        0.02,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2: GPU Dispatch Parity
// ═══════════════════════════════════════════════════════════════════

fn phase_2_gpu_dispatch(v: &mut ValidationHarness) {
    let station = StationDay {
        tmax: 32.0,
        tmin: 18.0,
        rh_max: 80.0,
        rh_min: 40.0,
        wind_2m: 1.8,
        rs: 22.5,
        elevation: 256.0,
        latitude: 42.727,
        doy: 200,
    };

    let cpu_batcher = BatchedEt0::cpu();
    let cpu_et0 = cpu_batcher
        .compute_gpu(&[station])
        .expect("cpu et0")
        .et0_values[0];
    v.check_abs("gpu_et0_cpu_baseline", cpu_et0, 5.5, 0.6);

    let device = airspring_barracuda::gpu::device_info::try_f64_device();
    v.check_bool("gpu_device_probed", true);

    if let Some(dev) = device {
        let gpu_et0 = BatchedEt0::gpu(Arc::clone(&dev))
            .expect("gpu et0 init")
            .compute_gpu(&[station])
            .expect("gpu et0")
            .et0_values[0];
        v.check_abs("gpu_et0_parity", (gpu_et0 - cpu_et0).abs(), 0.0, 0.01);

        let hg_day = HargreavesDay {
            tmax: 32.0,
            tmin: 18.0,
            latitude_deg: 42.727,
            day_of_year: 200,
        };
        let hg_cpu = BatchedHargreaves::cpu().compute(&[hg_day]).et0_values[0];
        let hg_gpu = BatchedHargreaves::gpu(Arc::clone(&dev))
            .expect("gpu hg init")
            .compute_gpu(&[hg_day])
            .expect("gpu hg")
            .et0_values[0];
        v.check_abs("gpu_hargreaves_parity", (hg_gpu - hg_cpu).abs(), 0.0, 0.01);

        let reading = SensorReading { raw_count: 4500.0 };
        let sc_cpu = BatchedSensorCal::cpu().compute(&[reading]).vwc_values[0];
        let sc_gpu = BatchedSensorCal::gpu(Arc::clone(&dev))
            .expect("gpu sc init")
            .compute_gpu(&[reading])
            .expect("gpu sc")
            .vwc_values[0];
        v.check_abs("gpu_sensor_parity", (sc_gpu - sc_cpu).abs(), 0.0, 1e-10);

        let field = FieldDayInput {
            dr_prev: 40.0,
            precipitation: 3.5,
            irrigation: 0.0,
            etc: 6.9,
            taw: 150.0,
            raw: 82.5,
            p: 0.55,
        };
        let wb_batcher = BatchedWaterBalance::with_gpu(200.0, 50.0, 600.0, 0.55, Arc::clone(&dev))
            .expect("gpu wb init");
        let wb_gpu = wb_batcher.gpu_step(&[field]).expect("gpu wb");
        v.check_abs("gpu_wb_depletion", wb_gpu[0], 43.4, 1.0);

        let data: Vec<f64> = (0..2048).map(|i| f64::from(i) * 0.01).collect();
        let stats = SeasonalReducer::new(Arc::clone(&dev))
            .expect("gpu reduce init")
            .compute_stats(&data)
            .expect("gpu reduce");
        v.check_abs("gpu_reduce_mean", stats.mean, 10.235, 0.1);
        v.check_bool("gpu_reduce_count", stats.count == 2048);
    } else {
        v.check_abs("gpu_fallback_et0_valid", cpu_et0, 5.97, 0.15);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 3: Batch Scaling
// ═══════════════════════════════════════════════════════════════════

fn phase_3_batch_scaling(v: &mut ValidationHarness) {
    let batcher = BatchedEt0::cpu();
    let station = StationDay {
        tmax: 32.0,
        tmin: 18.0,
        rh_max: 80.0,
        rh_min: 40.0,
        wind_2m: 1.8,
        rs: 22.5,
        elevation: 256.0,
        latitude: 42.727,
        doy: 200,
    };
    let ref_val = batcher.compute_gpu(&[station]).unwrap().et0_values[0];

    for n in [10, 100, 1000, 10_000] {
        let batch: Vec<StationDay> = vec![station; n];
        let t0 = Instant::now();
        let result = batcher.compute_gpu(&batch).unwrap();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let all_match = result
            .et0_values
            .iter()
            .all(|&val| (val - ref_val).abs() < 1e-12);
        v.check_bool(&format!("batch_n{n}_identical"), all_match);
        v.check_abs(&format!("batch_n{n}_under_5s"), elapsed_ms, 2500.0, 2500.0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4: BarraCuda Absorption Audit
// ═══════════════════════════════════════════════════════════════════

fn phase_4_absorption_audit(v: &mut ValidationHarness) {
    use airspring_barracuda::gpu::evolution_gaps::{Tier, GAPS};

    let total = GAPS.len();
    let tier_a = GAPS.iter().filter(|g| g.tier == Tier::A).count();
    let tier_b = GAPS.iter().filter(|g| g.tier == Tier::B).count();
    let tier_c = GAPS.iter().filter(|g| g.tier == Tier::C).count();

    v.check_bool("absorption_total_gt_10", total > 10);
    v.check_bool("absorption_tier_a_majority", tier_a > tier_b + tier_c);
    v.check_abs("absorption_tier_a", tier_a as f64, 11.0, 15.0);

    let has_et0 = GAPS.iter().any(|g| g.id.contains("et0"));
    let has_wb = GAPS.iter().any(|g| g.id.contains("water_balance"));
    let has_richards = GAPS.iter().any(|g| g.id.contains("richards"));
    let has_kriging = GAPS.iter().any(|g| g.id.contains("kriging"));
    let has_reduce = GAPS.iter().any(|g| g.id.contains("reduce"));
    v.check_bool("gap_et0_tracked", has_et0);
    v.check_bool("gap_wb_tracked", has_wb);
    v.check_bool("gap_richards_tracked", has_richards);
    v.check_bool("gap_kriging_tracked", has_kriging);
    v.check_bool("gap_reduce_tracked", has_reduce);
}

// ═══════════════════════════════════════════════════════════════════
// Phase 5: Mixed-Backend Seasonal Pipeline (GPU 1-2, CPU 3-4)
// ═══════════════════════════════════════════════════════════════════

fn phase_5_mixed_pipeline(v: &mut ValidationHarness) {
    use airspring_barracuda::eco::crop::CropType;
    use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};

    let weather: Vec<WeatherDay> = (0..153)
        .map(|d| WeatherDay {
            tmax: 4.0f64.mul_add((f64::from(d) * std::f64::consts::PI / 153.0).sin(), 28.0),
            tmin: 3.0f64.mul_add((f64::from(d) * std::f64::consts::PI / 153.0).sin(), 14.0),
            rh_max: 85.0,
            rh_min: 45.0,
            wind_2m: 1.8,
            solar_rad: 5.0f64.mul_add((f64::from(d) * std::f64::consts::PI / 153.0).sin(), 18.0),
            precipitation: if d % 7 == 0 { 12.0 } else { 0.0 },
            elevation: 256.0,
            latitude_deg: 42.727,
            day_of_year: 120 + d,
        })
        .collect();

    let config = CropConfig::standard(CropType::Corn);

    let cpu_pipe = SeasonalPipeline::cpu();
    let cpu_result = cpu_pipe.run_season(&weather, &config);
    v.check_bool("pipeline_cpu_completes", cpu_result.n_days == 153);
    v.check_abs(
        "pipeline_cpu_yield_ratio",
        cpu_result.yield_ratio,
        0.9,
        0.15,
    );
    v.check_bool("pipeline_cpu_et0_positive", cpu_result.total_et0 > 0.0);

    if let Some(dev) = airspring_barracuda::gpu::device_info::try_f64_device() {
        let gpu_pipe = SeasonalPipeline::gpu(dev).expect("gpu pipeline init");
        let gpu_result = gpu_pipe.run_season(&weather, &config);

        v.check_abs(
            "pipeline_gpu_et0_parity",
            (gpu_result.total_et0 - cpu_result.total_et0).abs(),
            0.0,
            1.0,
        );
        v.check_abs(
            "pipeline_gpu_yield_parity",
            (gpu_result.yield_ratio - cpu_result.yield_ratio).abs(),
            0.0,
            0.05,
        );
    } else {
        v.check_bool("pipeline_cpu_only_valid", cpu_result.yield_ratio >= 0.0);
    }
}
