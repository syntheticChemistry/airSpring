// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU orchestrator functional integration tests for airSpring `BarraCuda`.
//!
//! Tests GPU-accelerated paths (batched ET₀, water balance, kriging,
//! seasonal reducer, Richards PDE, isotherm fitting) with CPU cross-validation.
//! GPU tests gracefully skip when no `f64`-capable device is available.
//!
//! Related test modules:
//! - [`gpu_evolution`]: Evolution gap catalog and `BarraCuda` issue tracking
//! - [`gpu_determinism`]: Bit-identical rerun validation

mod common;

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::water_balance::DailyInput;
use common::device_or_skip;

// ── GPU orchestrator: Batched ET₀ ───────────────────────────────────

#[test]
fn test_gpu_batched_et0_cpu_path() {
    use airspring_barracuda::gpu::et0::BatchedEt0;

    let inputs: Vec<DailyEt0Input> = (0..50)
        .map(|i| DailyEt0Input {
            tmin: f64::from(i).mul_add(0.1, 12.0),
            tmax: f64::from(i).mul_add(0.2, 25.0),
            tmean: None,
            solar_radiation: f64::from(i).mul_add(0.1, 18.0),
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 1.5,
            elevation_m: 200.0,
            latitude_deg: 42.0,
            day_of_year: 150 + i,
        })
        .collect();

    let engine = BatchedEt0::cpu();
    let batched = engine.compute(&inputs);

    for (i, input) in inputs.iter().enumerate() {
        let scalar = et::daily_et0(input).et0;
        assert!(
            (batched.et0_values[i] - scalar).abs() < f64::EPSILON,
            "Day {i}: batched={} vs scalar={scalar}",
            batched.et0_values[i]
        );
    }
}

#[test]
fn test_gpu_batched_et0_station_day_gpu_dispatch() {
    use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};

    let device = device_or_skip!();

    let engine = BatchedEt0::gpu(device).unwrap();

    let station_days: Vec<StationDay> = (0_u32..50)
        .map(|i| StationDay {
            tmax: f64::from(i).mul_add(0.2, 21.0),
            tmin: f64::from(i).mul_add(0.1, 12.0),
            rh_max: 84.0,
            rh_min: 63.0,
            wind_2m: 2.078,
            rs: f64::from(i).mul_add(0.05, 20.0),
            elevation: 100.0,
            latitude: 50.80,
            doy: 150 + i,
        })
        .collect();

    let Some(result) = common::try_gpu_dispatch(|| engine.compute_gpu(&station_days)) else {
        return;
    };
    let result = result.unwrap();
    assert_eq!(result.et0_values.len(), 50);

    for (i, &val) in result.et0_values.iter().enumerate() {
        assert!(val > 0.5 && val < 10.0, "Day {i}: ET₀={val} out of range");
    }

    let cpu_engine = BatchedEt0::cpu();
    let cpu_result = cpu_engine.compute_gpu(&station_days).unwrap();

    for (i, (gpu, cpu)) in result
        .et0_values
        .iter()
        .zip(&cpu_result.et0_values)
        .enumerate()
    {
        assert!(
            (gpu - cpu).abs() < 0.1,
            "Day {i}: GPU={gpu} vs CPU={cpu}, diff={}",
            (gpu - cpu).abs()
        );
    }
}

// ── GPU orchestrator: Water Balance ─────────────────────────────────

#[test]
fn test_gpu_water_balance_mass_conservation() {
    use airspring_barracuda::gpu::water_balance::BatchedWaterBalance;

    let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
    let inputs: Vec<DailyInput> = (0..90)
        .map(|day| DailyInput {
            precipitation: if day % 7 == 3 { 15.0 } else { 0.0 },
            irrigation: if day % 14 == 0 { 25.0 } else { 0.0 },
            et0: 4.5,
            kc: 1.0,
        })
        .collect();

    let summary = engine.simulate_season(&inputs);
    assert!(
        summary.mass_balance_error < 0.01,
        "Mass balance: {}",
        summary.mass_balance_error
    );
    assert_eq!(summary.daily_outputs.len(), 90);
    assert!(summary.total_actual_et > 0.0);
}

#[test]
fn test_gpu_water_balance_gpu_step_dispatch() {
    use airspring_barracuda::gpu::water_balance::{BatchedWaterBalance, FieldDayInput};

    let device = device_or_skip!();

    let engine = BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, device).unwrap();

    let fields = vec![
        FieldDayInput {
            dr_prev: 20.0,
            precipitation: 5.0,
            irrigation: 0.0,
            etc: 4.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        },
        FieldDayInput {
            dr_prev: 50.0,
            precipitation: 0.0,
            irrigation: 25.0,
            etc: 6.0,
            taw: 120.0,
            raw: 48.0,
            p: 0.4,
        },
        FieldDayInput {
            dr_prev: 0.0,
            precipitation: 30.0,
            irrigation: 0.0,
            etc: 3.0,
            taw: 80.0,
            raw: 40.0,
            p: 0.5,
        },
        FieldDayInput {
            dr_prev: 90.0,
            precipitation: 0.0,
            irrigation: 0.0,
            etc: 5.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        },
    ];

    let Some(gpu_results) = common::try_gpu_dispatch(|| engine.gpu_step(&fields)) else {
        return;
    };
    let gpu_results = gpu_results.unwrap();
    assert_eq!(gpu_results.len(), 4);

    let cpu_engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
    let cpu_results = cpu_engine.gpu_step(&fields).unwrap();

    for (i, (gpu, cpu)) in gpu_results.iter().zip(&cpu_results).enumerate() {
        assert!((gpu - cpu).abs() < 0.5, "Field {i}: GPU={gpu} vs CPU={cpu}");
    }

    for (i, (&val, field)) in gpu_results.iter().zip(&fields).enumerate() {
        assert!(
            val >= 0.0 && val <= field.taw,
            "Field {i}: Dr={val} out of [0, {}]",
            field.taw
        );
    }
}

// ── GPU orchestrator: Kriging ───────────────────────────────────────

#[test]
fn test_gpu_kriging_interpolation() {
    use airspring_barracuda::gpu::kriging::*;

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
        SensorReading {
            x: 0.0,
            y: 100.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 100.0,
            y: 100.0,
            vwc: 0.35,
        },
    ];

    let targets = vec![
        TargetPoint { x: 50.0, y: 50.0 },
        TargetPoint { x: 25.0, y: 25.0 },
    ];

    let result = interpolate_soil_moisture(
        &sensors,
        &targets,
        SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 150.0,
        },
    );

    assert_eq!(result.vwc_values.len(), 2);
    assert!(
        result.vwc_values[0] > 0.20 && result.vwc_values[0] < 0.35,
        "Center VWC: {}",
        result.vwc_values[0]
    );
    for &v in &result.variances {
        assert!(v > 0.0 && v.is_finite(), "Variance: {v}");
    }
}

#[test]
fn test_kriging_interpolator_matches_idw_at_sensor() {
    use airspring_barracuda::gpu::kriging::{
        KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint,
    };

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
        SensorReading {
            x: 0.0,
            y: 100.0,
            vwc: 0.25,
        },
    ];
    let targets = vec![TargetPoint { x: 0.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 150.0,
    };

    let result = interp.interpolate(&sensors, &targets, variogram).unwrap();

    assert!(
        (result.vwc_values[0] - 0.20).abs() < 0.01,
        "At-sensor VWC should be ~0.20, got {}",
        result.vwc_values[0]
    );
    assert!(
        result.variances[0] < 0.005,
        "At-sensor variance should be small, got {}",
        result.variances[0]
    );
}

#[test]
fn test_kriging_interpolator_midpoint() {
    use airspring_barracuda::gpu::kriging::{
        KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint,
    };

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![TargetPoint { x: 50.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 150.0,
    };

    let result = interp.interpolate(&sensors, &targets, variogram).unwrap();

    assert!(
        (result.vwc_values[0] - 0.25).abs() < 0.02,
        "Midpoint VWC should be ~0.25, got {}",
        result.vwc_values[0]
    );
    assert!(
        result.variances[0] > 0.0 && result.variances[0].is_finite(),
        "Midpoint variance: {}",
        result.variances[0]
    );
}

#[test]
fn test_kriging_interpolator_empty_inputs() {
    use airspring_barracuda::gpu::kriging::{KrigingInterpolator, SoilVariogram, TargetPoint};

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();
    let targets = vec![TargetPoint { x: 0.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };

    let result = interp.interpolate(&[], &targets, variogram).unwrap();
    assert_eq!(result.vwc_values.len(), 1);
    assert!(result.variances[0].is_infinite());
}

#[test]
fn test_kriging_fit_variogram() {
    use airspring_barracuda::gpu::kriging::{KrigingInterpolator, SensorReading};

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.22,
        },
        SensorReading {
            x: 20.0,
            y: 0.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 30.0,
            y: 0.0,
            vwc: 0.28,
        },
        SensorReading {
            x: 40.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];

    let result = KrigingInterpolator::fit_variogram(&sensors, 5, 50.0);
    assert!(result.is_ok(), "Variogram fitting should succeed");
    let (lags, gammas) = result.unwrap();
    assert_eq!(lags.len(), gammas.len());
    assert!(!lags.is_empty(), "Should have at least one lag bin");
}

#[test]
fn test_kriging_interpolator_exponential_variogram() {
    use airspring_barracuda::gpu::kriging::{
        KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint,
    };

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
        SensorReading {
            x: 0.0,
            y: 100.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 100.0,
            y: 100.0,
            vwc: 0.35,
        },
    ];
    let targets = vec![
        TargetPoint { x: 50.0, y: 50.0 },
        TargetPoint { x: 25.0, y: 25.0 },
    ];
    let variogram = SoilVariogram::Exponential {
        nugget: 0.001,
        sill: 0.01,
        range: 150.0,
    };

    let result = interp.interpolate(&sensors, &targets, variogram).unwrap();

    assert_eq!(result.vwc_values.len(), 2);
    assert!(
        result.vwc_values[0] > 0.20 && result.vwc_values[0] < 0.35,
        "Center VWC with exponential variogram: {}",
        result.vwc_values[0]
    );
    for &v in &result.variances {
        assert!(
            v > 0.0 && v.is_finite(),
            "Exponential variogram variance: {v}"
        );
    }
}

// ── GPU orchestrator: Seasonal Reducer ──────────────────────────────

#[test]
fn test_gpu_seasonal_stats() {
    use airspring_barracuda::gpu::reduce;

    let et0_values = [4.2, 5.1, 3.8, 6.0, 4.5, 5.5, 3.2, 4.8, 5.0, 4.0];
    let stats = reduce::compute_seasonal_stats(&et0_values);

    assert_eq!(stats.count, 10);
    assert!((stats.total - 46.1).abs() < 1e-10);
    assert!((stats.mean - 4.61).abs() < 1e-10);
    assert!((stats.max - 6.0).abs() < 1e-10);
    assert!((stats.min - 3.2).abs() < 1e-10);
    assert!(stats.std_dev > 0.5 && stats.std_dev < 1.5);
}

#[test]
fn test_seasonal_reducer_sum_matches_cpu() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let gpu_sum = reducer.sum(&values).unwrap();
    let cpu_sum = reduce::seasonal_sum(&values);

    assert!(
        (gpu_sum - cpu_sum).abs() < 1e-6,
        "GPU sum ({gpu_sum}) should match CPU sum ({cpu_sum})"
    );
}

#[test]
fn test_seasonal_reducer_max_min_matches_cpu() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values = [4.2, 5.1, 3.8, 6.0, 4.5, 5.5, 3.2, 4.8, 5.0, 4.0];
    let gpu_max = reducer.max(&values).unwrap();
    let gpu_min = reducer.min(&values).unwrap();
    let cpu_max = reduce::seasonal_max(&values);
    let cpu_min = reduce::seasonal_min(&values);

    assert!(
        (gpu_max - cpu_max).abs() < 1e-10,
        "GPU max ({gpu_max}) should match CPU max ({cpu_max})"
    );
    assert!(
        (gpu_min - cpu_min).abs() < 1e-10,
        "GPU min ({gpu_min}) should match CPU min ({cpu_min})"
    );
}

#[test]
fn test_seasonal_reducer_stats_matches_cpu() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values = [4.2, 5.1, 3.8, 6.0, 4.5, 5.5, 3.2, 4.8, 5.0, 4.0];
    let gpu_stats = reducer.compute_stats(&values).unwrap();
    let cpu_stats = reduce::compute_seasonal_stats(&values);

    assert_eq!(gpu_stats.count, cpu_stats.count);
    assert!(
        (gpu_stats.total - cpu_stats.total).abs() < 1e-6,
        "total: GPU={} CPU={}",
        gpu_stats.total,
        cpu_stats.total
    );
    assert!(
        (gpu_stats.mean - cpu_stats.mean).abs() < 1e-6,
        "mean: GPU={} CPU={}",
        gpu_stats.mean,
        cpu_stats.mean
    );
    assert!(
        (gpu_stats.max - cpu_stats.max).abs() < 1e-10,
        "max: GPU={} CPU={}",
        gpu_stats.max,
        cpu_stats.max
    );
    assert!(
        (gpu_stats.min - cpu_stats.min).abs() < 1e-10,
        "min: GPU={} CPU={}",
        gpu_stats.min,
        cpu_stats.min
    );
    assert!(
        (gpu_stats.std_dev - cpu_stats.std_dev).abs() < 0.01,
        "std_dev: GPU={} CPU={}",
        gpu_stats.std_dev,
        cpu_stats.std_dev
    );
}

#[test]
fn test_seasonal_reducer_large_array_gpu_dispatch() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values: Vec<f64> = (0..2048).map(|i| f64::from(i) * 0.01).collect();
    let cpu_sum = reduce::seasonal_sum(&values);

    let gpu_sum = reducer.sum(&values).unwrap();
    assert!(
        (gpu_sum - cpu_sum).abs() < 1e-4,
        "Large array sum: GPU={gpu_sum} CPU={cpu_sum}"
    );

    let gpu_max = reducer.max(&values).unwrap();
    let gpu_min = reducer.min(&values).unwrap();
    let cpu_max = reduce::seasonal_max(&values);
    let cpu_min = reduce::seasonal_min(&values);

    assert!(
        (gpu_max - cpu_max).abs() < 1e-10,
        "Large array max: GPU={gpu_max} CPU={cpu_max}"
    );
    assert!(
        (gpu_min - cpu_min).abs() < 1e-10,
        "Large array min: GPU={gpu_min} CPU={cpu_min}"
    );
}

#[test]
fn test_seasonal_reducer_empty() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();
    let stats = reducer.compute_stats(&[]).unwrap();
    assert_eq!(stats.count, 0);
    assert!((stats.total).abs() < 1e-10);
}

// ── GPU orchestrator: Richards PDE ──────────────────────────────────

#[test]
fn test_gpu_richards_drainage_physical_bounds() {
    use airspring_barracuda::eco::richards::VanGenuchtenParams;
    use airspring_barracuda::gpu::richards::{solve_batch_cpu, RichardsRequest};

    let silt_loam = VanGenuchtenParams {
        theta_r: 0.067,
        theta_s: 0.45,
        alpha: 0.02,
        n_vg: 1.41,
        ks: 10.8,
    };

    let request = RichardsRequest {
        params: silt_loam,
        depth_cm: 50.0,
        n_nodes: 20,
        h_initial: -100.0,
        h_top: -30.0,
        zero_flux_top: false,
        bottom_free_drain: true,
        duration_days: 1.0,
        dt_days: 0.01,
    };

    let cpu_results = solve_batch_cpu(&[request]);
    assert_eq!(cpu_results.len(), 1);
    let profiles = cpu_results[0]
        .as_ref()
        .expect("Richards solve should succeed");
    assert!(!profiles.is_empty(), "Should produce at least one profile");

    let final_profile = profiles.last().unwrap();
    for &theta in &final_profile.theta {
        assert!(
            theta >= silt_loam.theta_r && theta <= silt_loam.theta_s,
            "theta={theta} outside [{}, {}]",
            silt_loam.theta_r,
            silt_loam.theta_s
        );
    }
}

#[test]
fn test_gpu_richards_cross_validate_cpu_upstream() {
    use airspring_barracuda::eco::richards::VanGenuchtenParams;
    use airspring_barracuda::gpu::richards::{BatchedRichards, RichardsRequest};

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    let req = RichardsRequest {
        params: sand,
        depth_cm: 30.0,
        n_nodes: 10,
        h_initial: -20.0,
        h_top: 0.0,
        zero_flux_top: false,
        bottom_free_drain: false,
        duration_days: 0.5,
        dt_days: 0.05,
    };

    let result = BatchedRichards::cross_validate(&req);
    assert!(
        result.is_ok(),
        "Cross-validation should succeed: {:?}",
        result.err()
    );
}

// ── GPU orchestrator: Isotherm fitting ──────────────────────────────

#[test]
fn test_gpu_isotherm_nm_matches_linearized() {
    use airspring_barracuda::eco::isotherm;
    use airspring_barracuda::gpu::isotherm as gpu_iso;

    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    let lin = isotherm::fit_langmuir(&ce, &qe).expect("linearized fit");
    let nm = gpu_iso::fit_langmuir_nm(&ce, &qe).expect("NM fit");

    assert!(
        nm.r_squared >= lin.r_squared - 0.01,
        "NM R²={} should be ≥ linearized R²={} (within tolerance)",
        nm.r_squared,
        lin.r_squared
    );
}

#[test]
fn test_gpu_isotherm_global_beats_single_start() {
    use airspring_barracuda::gpu::isotherm as gpu_iso;

    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    let nm = gpu_iso::fit_langmuir_nm(&ce, &qe).expect("NM fit");
    let global = gpu_iso::fit_langmuir_global(&ce, &qe, 8).expect("global fit");

    assert!(
        global.r_squared >= nm.r_squared - 0.02,
        "Global R²={} should be competitive with NM R²={}",
        global.r_squared,
        nm.r_squared
    );
    assert!(global.r_squared > 0.95, "Global R²={}", global.r_squared);
}

#[test]
fn test_gpu_isotherm_batch_global_field_scale() {
    use airspring_barracuda::eco::isotherm;
    use airspring_barracuda::gpu::isotherm as gpu_iso;

    let ce = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0];
    let qe_site_a: Vec<f64> = ce
        .iter()
        .map(|&c| isotherm::langmuir(c, 15.0, 0.05))
        .collect();
    let qe_site_b: Vec<f64> = ce
        .iter()
        .map(|&c| isotherm::langmuir(c, 25.0, 0.03))
        .collect();
    let qe_site_c: Vec<f64> = ce
        .iter()
        .map(|&c| isotherm::freundlich(c, 2.5, 1.0 / 2.8))
        .collect();

    let datasets: Vec<(&[f64], &[f64])> =
        vec![(&ce, &qe_site_a), (&ce, &qe_site_b), (&ce, &qe_site_c)];

    let results = gpu_iso::fit_batch_global(&datasets, 4);
    assert_eq!(results.len(), 3);

    for (i, (lang, freund)) in results.iter().enumerate() {
        assert!(
            lang.is_some(),
            "Site {i}: Langmuir global fit should succeed"
        );
        assert!(
            freund.is_some(),
            "Site {i}: Freundlich global fit should succeed"
        );
        let r2 = lang.as_ref().unwrap().r_squared;
        assert!(r2 > 0.90, "Site {i}: Langmuir R²={r2}");
    }
}
