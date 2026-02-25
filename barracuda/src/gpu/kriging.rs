// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soil moisture spatial interpolation via `ToadStool` Kriging.
//!
//! Wraps [`barracuda::ops::kriging_f64::KrigingF64`] with domain-specific types
//! for precision agriculture soil moisture mapping.
//!
//! # Two API Levels
//!
//! | API | GPU? | Dependency |
//! |-----|:----:|------------|
//! | [`interpolate_soil_moisture`] | No | None (IDW fallback) |
//! | [`KrigingInterpolator`] | Yes¹ | `Arc<WgpuDevice>` |
//!
//! ¹ `KrigingF64` currently solves on CPU (LU); GPU dispatch planned when
//!   `ToadStool` ships a kriging compute shader.
//!
//! # Usage
//!
//! Given a sparse sensor network (3–20 sensors), interpolate volumetric water
//! content (VWC) across a field grid. This feeds the water balance model for
//! variable-rate irrigation scheduling.
//!
//! # `ToadStool` Primitive
//!
//! Uses [`barracuda::ops::kriging_f64::KrigingF64`] for ordinary kriging with
//! proper variogram-based covariance and LU solve. When `ToadStool` adds GPU
//! dispatch for large kriging systems, the [`KrigingInterpolator`] wrapper will
//! automatically benefit.

/// Squared distance below which a prediction point is considered
/// co-located with a known observation. Set to 0.01 mm² to avoid
/// division by zero in IDW weights while remaining far below any
/// physical sensor spacing.
const COLLOCATED_DIST_SQ: f64 = 1e-10;

/// A soil moisture sensor reading with spatial coordinates.
#[derive(Debug, Clone, Copy)]
pub struct SensorReading {
    /// Easting or longitude (m or degrees).
    pub x: f64,
    /// Northing or latitude (m or degrees).
    pub y: f64,
    /// Volumetric water content (m³/m³).
    pub vwc: f64,
}

/// A target point for interpolation.
#[derive(Debug, Clone, Copy)]
pub struct TargetPoint {
    /// Easting or longitude.
    pub x: f64,
    /// Northing or latitude.
    pub y: f64,
}

/// Variogram model for soil moisture spatial correlation.
#[derive(Debug, Clone, Copy)]
pub enum SoilVariogram {
    /// Spherical model — most common for soil properties.
    Spherical {
        /// Nugget (measurement noise).
        nugget: f64,
        /// Sill (total variance).
        sill: f64,
        /// Range (distance at which correlation vanishes).
        range: f64,
    },
    /// Exponential model — for smooth spatial fields.
    Exponential {
        /// Nugget.
        nugget: f64,
        /// Sill.
        sill: f64,
        /// Range.
        range: f64,
    },
}

/// Result of soil moisture interpolation.
#[derive(Debug, Clone)]
pub struct InterpolationResult {
    /// Interpolated VWC at each target point (m³/m³).
    pub vwc_values: Vec<f64>,
    /// Kriging variance at each target point — measures interpolation uncertainty.
    pub variances: Vec<f64>,
}

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::kriging_f64;

// ── Device-backed Kriging (wraps barracuda::ops::kriging_f64) ────────

/// Device-backed ordinary kriging interpolator.
///
/// Wraps [`barracuda::ops::kriging_f64::KrigingF64`] with precision-agriculture
/// domain types. Solves the full kriging system (covariance matrix + LU) instead
/// of the IDW approximation used by [`interpolate_soil_moisture`].
///
/// # Construction
///
/// Requires an `Arc<WgpuDevice>`. Create one via `WgpuDevice::new()` (async)
/// or `WgpuDevice::new_cpu()` for headless environments.
pub struct KrigingInterpolator {
    engine: kriging_f64::KrigingF64,
}

impl KrigingInterpolator {
    /// Create a new device-backed kriging interpolator.
    ///
    /// # Errors
    ///
    /// Returns an error if the `KrigingF64` engine cannot be initialised.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = kriging_f64::KrigingF64::new(device)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
        Ok(Self { engine })
    }

    /// Ordinary kriging interpolation using the full variogram-based covariance system.
    ///
    /// Unlike [`interpolate_soil_moisture`] (IDW), this solves the N×N kriging
    /// system via LU decomposition for statistically optimal weights and produces
    /// proper kriging variances (prediction uncertainty).
    ///
    /// # Errors
    ///
    /// Returns an error if the kriging system is singular (e.g., collocated points).
    pub fn interpolate(
        &self,
        sensors: &[SensorReading],
        targets: &[TargetPoint],
        variogram: SoilVariogram,
    ) -> crate::error::Result<InterpolationResult> {
        if sensors.is_empty() || targets.is_empty() {
            return Ok(InterpolationResult {
                vwc_values: vec![0.0; targets.len()],
                variances: vec![f64::INFINITY; targets.len()],
            });
        }

        // Convert domain types → barracuda types
        let known: Vec<(f64, f64, f64)> = sensors.iter().map(|s| (s.x, s.y, s.vwc)).collect();
        let target_pts: Vec<(f64, f64)> = targets.iter().map(|t| (t.x, t.y)).collect();
        let bv = to_barracuda_variogram(variogram);

        let result = self
            .engine
            .interpolate(&known, &target_pts, bv)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;

        Ok(InterpolationResult {
            vwc_values: result.values,
            variances: result.variances,
        })
    }

    /// Fit an empirical variogram from the sensor data.
    ///
    /// Returns `(lag_distances, semi_variances)` for visualisation and model selection.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (e.g., insufficient data).
    pub fn fit_variogram(
        sensors: &[SensorReading],
        n_lags: usize,
        max_distance: f64,
    ) -> crate::error::Result<(Vec<f64>, Vec<f64>)> {
        let known: Vec<(f64, f64, f64)> = sensors.iter().map(|s| (s.x, s.y, s.vwc)).collect();
        kriging_f64::KrigingF64::fit_variogram(&known, n_lags, max_distance)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
    }
}

/// Convert domain variogram to barracuda variogram model.
const fn to_barracuda_variogram(v: SoilVariogram) -> kriging_f64::VariogramModel {
    match v {
        SoilVariogram::Spherical {
            nugget,
            sill,
            range,
        } => kriging_f64::VariogramModel::Spherical {
            nugget,
            sill,
            range,
        },
        SoilVariogram::Exponential {
            nugget,
            sill,
            range,
        } => kriging_f64::VariogramModel::Exponential {
            nugget,
            sill,
            range,
        },
    }
}

// ── Free function: CPU-only IDW fallback (no device needed) ─────────

/// Interpolate soil moisture from a sparse sensor network (CPU-only IDW).
///
/// This is a lightweight inverse-distance weighting approximation.
/// For proper ordinary kriging with variogram-based weights and prediction
/// variance, use [`KrigingInterpolator`] instead.
///
/// # Variogram usage
///
/// Only `nugget` and `sill` are used. The variance approximation uses
/// γ(h) ≈ nugget + sill·(1 - exp(-h)) (exponential with implicit range=1).
/// The variogram `range` is ignored; for range-aware variance, use
/// [`KrigingInterpolator::interpolate`].
///
/// # Current Status
///
/// CPU-only. For the device-backed path (proper kriging via LU solve),
/// see [`KrigingInterpolator::interpolate`].
#[must_use]
pub fn interpolate_soil_moisture(
    sensors: &[SensorReading],
    targets: &[TargetPoint],
    variogram: SoilVariogram,
) -> InterpolationResult {
    if sensors.is_empty() || targets.is_empty() {
        return InterpolationResult {
            vwc_values: vec![0.0; targets.len()],
            variances: vec![f64::INFINITY; targets.len()],
        };
    }

    // Simple inverse-distance weighting as CPU fallback
    // (Full kriging system solve will come via ToadStool KrigingF64 wiring)
    let mut vwc_values = Vec::with_capacity(targets.len());
    let mut variances = Vec::with_capacity(targets.len());

    let (nugget, sill) = match variogram {
        SoilVariogram::Spherical { nugget, sill, .. }
        | SoilVariogram::Exponential { nugget, sill, .. } => (nugget, sill),
    };

    for target in targets {
        let mut weight_sum = 0.0f64;
        let mut weighted_vwc = 0.0f64;
        let mut min_dist_sq = f64::INFINITY;

        for sensor in sensors {
            let dx = sensor.x - target.x;
            let dy = sensor.y - target.y;
            let dist_sq = dy.mul_add(dy, dx * dx);

            if dist_sq < COLLOCATED_DIST_SQ {
                // Exact sensor location
                weight_sum = 1.0;
                weighted_vwc = sensor.vwc;
                min_dist_sq = 0.0;
                break;
            }

            let weight = 1.0 / dist_sq;
            weight_sum += weight;
            weighted_vwc += weight * sensor.vwc;
            min_dist_sq = min_dist_sq.min(dist_sq);
        }

        let vwc = if weight_sum > 0.0 {
            weighted_vwc / weight_sum
        } else {
            0.0
        };

        // Approximate variance from distance and variogram.
        // Uses γ(h) ≈ nugget + sill·(1 - exp(-h)) — a simplified exponential
        // variogram with implicit range=1. The `range` field of the variogram
        // is ignored in this IDW path. For proper variogram-based variance
        // (including range), use [`KrigingInterpolator::interpolate`].
        let variance = if min_dist_sq < COLLOCATED_DIST_SQ {
            nugget
        } else {
            sill.mul_add(1.0 - (-min_dist_sq.sqrt()).exp(), nugget)
        };

        vwc_values.push(vwc);
        variances.push(variance);
    }

    InterpolationResult {
        vwc_values,
        variances,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Midpoint should be close to 0.25 (equidistant → equal weights)
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
        // Values should increase from sensor 1 (0.20) toward sensor 2 (0.30)
        assert!(result.vwc_values[0] < result.vwc_values[1]);
        assert!(result.vwc_values[1] < result.vwc_values[2]);
    }

    #[test]
    fn test_variogram_to_barracuda_conversion() {
        let sph = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let exp = SoilVariogram::Exponential {
            nugget: 0.002,
            sill: 0.02,
            range: 30.0,
        };
        // Just ensure conversion doesn't panic
        let _b1 = to_barracuda_variogram(sph);
        let _b2 = to_barracuda_variogram(exp);
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
}
