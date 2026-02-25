//! GPU orchestrator and evolution gap integration tests for airSpring `BarraCuda`.
//!
//! Tests GPU-accelerated paths (batched ET₀, water balance, kriging,
//! seasonal reducer), evolution gap tracking, and `ToadStool` issue resolution.
//! GPU tests gracefully skip when no `f64`-capable device is available.

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::water_balance::DailyInput;

// ── GPU device helpers ──────────────────────────────────────────────

/// Try to create an `f64`-capable `WgpuDevice`. Returns `None` on CI/headless
/// or if the GPU doesn't support `SHADER_F64`.
fn try_create_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
    pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
        .ok()
        .map(std::sync::Arc::new)
}

/// Get a device or skip the test.
macro_rules! device_or_skip {
    () => {
        match try_create_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: No GPU device available");
                return;
            }
        }
    };
}

// ── Evolution gap infrastructure ────────────────────────────────────

#[test]
fn test_evolution_gaps_catalogued() {
    use airspring_barracuda::gpu::evolution_gaps::{Tier, GAPS};

    assert!(GAPS.len() >= 8, "Expected 8+ gaps, got {}", GAPS.len());

    let tier_a = GAPS.iter().filter(|g| g.tier == Tier::A).count();
    let tier_b = GAPS.iter().filter(|g| g.tier == Tier::B).count();
    let tier_c = GAPS.iter().filter(|g| g.tier == Tier::C).count();

    assert!(tier_a >= 4, "Expected 4+ Tier A gaps, got {tier_a}");
    assert!(tier_b >= 2, "Expected 2+ Tier B gaps, got {tier_b}");
    assert!(tier_c >= 1, "Expected 1+ Tier C gaps, got {tier_c}");

    for gap in GAPS.iter().filter(|g| g.tier == Tier::A) {
        assert!(
            gap.toadstool_primitive.is_some(),
            "Tier A gap '{}' should reference a ToadStool primitive",
            gap.id
        );
    }

    for gap in GAPS {
        assert!(!gap.id.is_empty(), "Gap id must not be empty");
        assert!(
            !gap.description.is_empty(),
            "Gap description must not be empty"
        );
        assert!(!gap.action.is_empty(), "Gap action must not be empty");
    }
}

#[test]
fn test_evolution_gaps_unique_ids() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for gap in GAPS {
        assert!(
            seen.insert(gap.id),
            "Duplicate evolution gap id: '{}'",
            gap.id
        );
    }
}

#[test]
fn test_batched_et0_gap_documented() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;

    let et0_gap = GAPS.iter().find(|g| g.id == "batched_et0_gpu");
    assert!(et0_gap.is_some(), "Batched ET₀ GPU gap must be documented");

    let gap = et0_gap.unwrap();
    assert!(
        gap.toadstool_primitive
            .unwrap()
            .contains("batched_elementwise"),
        "Should reference the batched elementwise shader"
    );
}

#[test]
fn test_kriging_gap_documented() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;

    let kriging_gap = GAPS.iter().find(|g| g.id == "kriging_soil_moisture");
    assert!(
        kriging_gap.is_some(),
        "Kriging soil moisture gap must be documented"
    );

    let gap = kriging_gap.unwrap();
    assert!(
        gap.toadstool_primitive.unwrap().contains("kriging"),
        "Should reference kriging_f64"
    );
}

// ── ToadStool issue tracking tests ──────────────────────────────────

#[test]
fn test_toadstool_issues_all_resolved() {
    use airspring_barracuda::gpu::evolution_gaps::{IssueStatus, TOADSTOOL_ISSUES};

    assert_eq!(
        TOADSTOOL_ISSUES.len(),
        4,
        "Expected 4 ToadStool issues, got {}",
        TOADSTOOL_ISSUES.len()
    );

    for issue in TOADSTOOL_ISSUES {
        assert_eq!(
            issue.status,
            IssueStatus::Resolved,
            "{} should be Resolved, got {:?}",
            issue.id,
            issue.status
        );
        assert!(!issue.id.is_empty());
        assert!(!issue.file.is_empty());
        assert!(!issue.summary.is_empty());
        assert!(!issue.fix.is_empty());
        assert!(!issue.blocks.is_empty());
    }
}

#[test]
fn test_toadstool_issues_by_id() {
    use airspring_barracuda::gpu::evolution_gaps::TOADSTOOL_ISSUES;

    let ts001 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-001").unwrap();
    assert_eq!(ts001.severity, "CRITICAL");
    assert!(ts001.file.contains("batched_elementwise"));
    assert!(ts001.fix.contains("RESOLVED"));

    let ts002 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-002").unwrap();
    assert_eq!(ts002.severity, "MEDIUM");
    assert!(ts002.fix.contains("RESOLVED"));

    let ts003 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-003").unwrap();
    assert_eq!(ts003.severity, "LOW");
    assert!(ts003.fix.contains("RESOLVED"));

    let ts004 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-004").unwrap();
    assert_eq!(ts004.severity, "HIGH");
    assert!(ts004.fix.contains("RESOLVED"));
}

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

    let result = engine.compute_gpu(&station_days).unwrap();
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

    let gpu_results = engine.gpu_step(&fields).unwrap();
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

    // N=2048 — triggers GPU dispatch in `FusedMapReduceF64` (threshold: 1024).
    // TS-004 resolved: buffer conflict is fixed, GPU dispatch works.
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

// ── GPU determinism: same input → bit-identical output ──────────────

#[test]
fn test_gpu_batched_et0_deterministic() {
    use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};

    let device = device_or_skip!();

    let engine = BatchedEt0::gpu(device).unwrap();

    let station_days: Vec<StationDay> = (0_u32..30)
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

    let run1 = engine.compute_gpu(&station_days).unwrap();
    let run2 = engine.compute_gpu(&station_days).unwrap();

    for (i, (a, b)) in run1.et0_values.iter().zip(&run2.et0_values).enumerate() {
        assert!(
            (a - b).abs() < f64::EPSILON,
            "Day {i}: run1={a} vs run2={b} — GPU not deterministic"
        );
    }
}

#[test]
fn test_gpu_water_balance_deterministic() {
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
    ];

    let run1 = engine.gpu_step(&fields).unwrap();
    let run2 = engine.gpu_step(&fields).unwrap();

    for (i, (a, b)) in run1.iter().zip(&run2).enumerate() {
        assert!(
            (a - b).abs() < f64::EPSILON,
            "Field {i}: run1={a} vs run2={b} — GPU not deterministic"
        );
    }
}

#[test]
fn test_gpu_reducer_deterministic() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values: Vec<f64> = (0..2048).map(|i| f64::from(i) * 0.01).collect();

    let sum1 = reducer.sum(&values).unwrap();
    let sum2 = reducer.sum(&values).unwrap();
    assert!(
        (sum1 - sum2).abs() < f64::EPSILON,
        "sum: run1={sum1} vs run2={sum2}"
    );

    let max1 = reducer.max(&values).unwrap();
    let max2 = reducer.max(&values).unwrap();
    assert!(
        (max1 - max2).abs() < f64::EPSILON,
        "max: run1={max1} vs run2={max2}"
    );

    let stats1 = reducer.compute_stats(&values).unwrap();
    let stats2 = reducer.compute_stats(&values).unwrap();
    assert!(
        (stats1.total - stats2.total).abs() < f64::EPSILON,
        "stats.total: run1={} vs run2={}",
        stats1.total,
        stats2.total
    );
    assert!(
        (stats1.std_dev - stats2.std_dev).abs() < f64::EPSILON,
        "stats.std_dev: run1={} vs run2={}",
        stats1.std_dev,
        stats2.std_dev
    );
}

#[test]
fn test_gpu_kriging_deterministic() {
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
    let targets = vec![
        TargetPoint { x: 50.0, y: 50.0 },
        TargetPoint { x: 25.0, y: 75.0 },
    ];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 150.0,
    };

    let run1 = interp.interpolate(&sensors, &targets, variogram).unwrap();
    let run2 = interp.interpolate(&sensors, &targets, variogram).unwrap();

    for (i, (a, b)) in run1.vwc_values.iter().zip(&run2.vwc_values).enumerate() {
        assert!(
            (a - b).abs() < f64::EPSILON,
            "Point {i}: VWC run1={a} vs run2={b}"
        );
    }
    for (i, (a, b)) in run1.variances.iter().zip(&run2.variances).enumerate() {
        assert!(
            (a - b).abs() < f64::EPSILON,
            "Point {i}: variance run1={a} vs run2={b}"
        );
    }
}
