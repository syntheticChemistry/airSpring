// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU determinism tests: same input must produce bit-identical output.
//!
//! Validates that all GPU orchestrators are deterministic across reruns.
//! This is a cross-cutting concern for scientific reproducibility.

mod common;

use airspring_barracuda::gpu;
use common::device_or_skip;

#[test]
fn test_gpu_batched_et0_deterministic() {
    use gpu::et0::{BatchedEt0, StationDay};

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

    let Some(run1) = common::try_gpu_dispatch(|| engine.compute_gpu(&station_days)) else {
        return;
    };
    let run1 = run1.unwrap();
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
    use gpu::water_balance::{BatchedWaterBalance, FieldDayInput};

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

    let Some(run1) = common::try_gpu_dispatch(|| engine.gpu_step(&fields)) else {
        return;
    };
    let run1 = run1.unwrap();
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
    use gpu::reduce;

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
    use gpu::kriging::{KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint};

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
