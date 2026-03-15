// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `gpu::kriging` — soil moisture spatial interpolation.

mod common;

use airspring_barracuda::gpu::kriging::{
    KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint, interpolate_soil_moisture,
};

use common::try_create_device;

#[test]
fn test_interpolate_at_sensor() {
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![TargetPoint { x: 0.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };

    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert!(
        (result.vwc_values[0] - 0.20).abs() < 1e-10,
        "At sensor, VWC should match: {}",
        result.vwc_values[0]
    );
}

#[test]
fn test_interpolate_midpoint() {
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![TargetPoint { x: 5.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };

    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert!(
        (result.vwc_values[0] - 0.25).abs() < 0.01,
        "Midpoint VWC: {}",
        result.vwc_values[0]
    );
}

#[test]
fn test_interpolate_empty() {
    let result = interpolate_soil_moisture(
        &[],
        &[TargetPoint { x: 0.0, y: 0.0 }],
        SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        },
    );
    assert_eq!(result.vwc_values.len(), 1);
    assert!(result.variances[0].is_infinite());
}

#[test]
fn test_interpolate_empty_targets() {
    let sensors = vec![SensorReading {
        x: 0.0,
        y: 0.0,
        vwc: 0.20,
    }];
    let result = interpolate_soil_moisture(
        &sensors,
        &[],
        SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        },
    );
    assert!(result.vwc_values.is_empty());
    assert!(result.variances.is_empty());
}

#[test]
fn test_interpolate_exponential_variogram() {
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.15,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 0.0,
            y: 10.0,
            vwc: 0.20,
        },
    ];
    let targets = vec![TargetPoint { x: 5.0, y: 5.0 }];
    let variogram = SoilVariogram::Exponential {
        nugget: 0.001,
        sill: 0.01,
        range: 20.0,
    };
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert_eq!(result.vwc_values.len(), 1);
    assert!(
        result.vwc_values[0] > 0.10 && result.vwc_values[0] < 0.30,
        "VWC={} out of range",
        result.vwc_values[0]
    );
    assert!(result.variances[0] > 0.0, "Variance should be positive");
}

#[test]
fn test_interpolate_closer_sensor_dominates() {
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.10,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.40,
        },
    ];
    let targets = vec![TargetPoint { x: 1.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 50.0,
    };
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert!(
        result.vwc_values[0] < 0.15,
        "Closer sensor (0.10) should dominate: {}",
        result.vwc_values[0]
    );
}

#[test]
fn test_interpolate_multiple_targets() {
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![
        TargetPoint { x: 2.0, y: 0.0 },
        TargetPoint { x: 5.0, y: 0.0 },
        TargetPoint { x: 8.0, y: 0.0 },
    ];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert_eq!(result.vwc_values.len(), 3);
    assert!(result.vwc_values[0] < result.vwc_values[1]);
    assert!(result.vwc_values[1] < result.vwc_values[2]);
}

#[test]
fn test_variance_at_sensor_is_nugget() {
    let sensors = vec![SensorReading {
        x: 0.0,
        y: 0.0,
        vwc: 0.20,
    }];
    let targets = vec![TargetPoint { x: 0.0, y: 0.0 }];
    let nugget = 0.005;
    let variogram = SoilVariogram::Spherical {
        nugget,
        sill: 0.01,
        range: 15.0,
    };
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert!(
        (result.variances[0] - nugget).abs() < 1e-10,
        "Variance at sensor should be nugget: {}",
        result.variances[0]
    );
}

#[test]
fn test_interpolate_single_sensor() {
    let sensors = vec![SensorReading {
        x: 50.0,
        y: 50.0,
        vwc: 0.25,
    }];
    let targets = vec![
        TargetPoint { x: 50.0, y: 50.0 },
        TargetPoint { x: 60.0, y: 50.0 },
        TargetPoint { x: 70.0, y: 50.0 },
    ];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 20.0,
    };
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert_eq!(result.vwc_values.len(), 3);
    assert!(
        (result.vwc_values[0] - 0.25).abs() < 1e-10,
        "At sensor, VWC should match: {}",
        result.vwc_values[0]
    );
    for i in 1..3 {
        assert!(
            (result.vwc_values[i] - 0.25).abs() < 1e-6,
            "Single sensor: all targets get same VWC via IDW: {}",
            result.vwc_values[i]
        );
    }
}

#[test]
fn test_interpolate_many_target_points() {
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
            x: 50.0,
            y: 100.0,
            vwc: 0.25,
        },
    ];
    let targets: Vec<TargetPoint> = (0..50)
        .flat_map(|i| {
            (0..20).map(move |j| TargetPoint {
                x: f64::from(i) * 2.0,
                y: f64::from(j) * 5.0,
            })
        })
        .collect();
    assert_eq!(targets.len(), 1000);
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 80.0,
    };
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert_eq!(result.vwc_values.len(), 1000);
    assert_eq!(result.variances.len(), 1000);
    for (&vwc, &var) in result.vwc_values.iter().zip(&result.variances) {
        assert!((0.0..=1.0).contains(&vwc), "VWC out of range: {vwc}");
        assert!(var >= 0.0 && var.is_finite(), "Variance invalid: {var}");
    }
}

#[test]
fn test_variance_increases_with_distance() {
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 1.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.02,
        range: 15.0,
    };
    let targets = vec![
        TargetPoint { x: 2.0, y: 0.0 },
        TargetPoint { x: 5.0, y: 0.0 },
        TargetPoint { x: 10.0, y: 0.0 },
        TargetPoint { x: 20.0, y: 0.0 },
    ];
    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    for i in 1..targets.len() {
        assert!(
            result.variances[i] >= result.variances[i - 1] - 1e-10,
            "Variance should increase with distance: var[{}]={} < var[{}]={}",
            i,
            result.variances[i],
            i - 1,
            result.variances[i - 1]
        );
    }
}

#[test]
fn test_kriging_interpolator_new() {
    let Some(device) = try_create_device() else {
        eprintln!("SKIP: No GPU device for KrigingInterpolator");
        return;
    };
    let interp = KrigingInterpolator::new(device);
    assert!(interp.is_ok(), "KrigingInterpolator::new should succeed");
}

#[test]
fn test_kriging_interpolator_interpolate() {
    let Some(device) = try_create_device() else {
        eprintln!("SKIP: No GPU device for KrigingInterpolator");
        return;
    };
    let interp = KrigingInterpolator::new(device).unwrap();
    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![TargetPoint { x: 5.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };
    let result = interp.interpolate(&sensors, &targets, variogram).unwrap();
    assert_eq!(result.vwc_values.len(), 1);
    assert!(
        result.vwc_values[0] > 0.18 && result.vwc_values[0] < 0.32,
        "Midpoint VWC: {}",
        result.vwc_values[0]
    );
    assert!(result.variances[0] > 0.0 && result.variances[0].is_finite());
}

#[test]
fn test_kriging_interpolator_interpolate_empty_inputs() {
    let Some(device) = try_create_device() else {
        eprintln!("SKIP: No GPU device for KrigingInterpolator");
        return;
    };
    let interp = KrigingInterpolator::new(device).unwrap();
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };

    let result = interp
        .interpolate(&[], &[TargetPoint { x: 0.0, y: 0.0 }], variogram)
        .unwrap();
    assert_eq!(result.vwc_values.len(), 1);
    assert!(result.vwc_values[0].abs() < f64::EPSILON);
    assert!(result.variances[0].is_infinite());

    let sensors = vec![SensorReading {
        x: 0.0,
        y: 0.0,
        vwc: 0.20,
    }];
    let result = interp.interpolate(&sensors, &[], variogram).unwrap();
    assert!(result.vwc_values.is_empty());
    assert!(result.variances.is_empty());
}

#[test]
fn test_kriging_fit_variogram() {
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
    assert!(!lags.is_empty());
}

#[test]
fn test_kriging_fit_variogram_insufficient_data() {
    let empty: Vec<SensorReading> = vec![];
    let result = KrigingInterpolator::fit_variogram(&empty, 5, 50.0);
    if let Ok((lags, gammas)) = &result {
        assert_eq!(lags.len(), gammas.len());
    }

    let single = vec![SensorReading {
        x: 0.0,
        y: 0.0,
        vwc: 0.20,
    }];
    let result = KrigingInterpolator::fit_variogram(&single, 5, 50.0);
    if let Ok((lags, gammas)) = &result {
        assert_eq!(lags.len(), gammas.len());
    }
}
