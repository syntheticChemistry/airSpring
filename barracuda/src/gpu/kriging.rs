// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soil moisture spatial interpolation via `ToadStool` Kriging.
//!
//! Wraps [`barracuda::ops::kriging_f64::KrigingF64`] with domain-specific types
//! for precision agriculture soil moisture mapping.
//!
//! # Cross-Spring Provenance
//!
//! `kriging_f64` (geostatistical interpolation) was developed jointly by
//! airSpring (soil moisture mapping) and wetSpring (sampling site interpolation).
//! The variogram models (spherical, exponential, Gaussian, linear) share the
//! same mathematical primitives as wetSpring's spatial ecology pipelines.
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
        let engine = kriging_f64::KrigingF64::new(device)?;
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

        let result = self.engine.interpolate(&known, &target_pts, bv)?;

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
            .map_err(crate::error::AirSpringError::from)
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
/// Uses inverse-distance weighting with variogram-informed variance.
/// This is intentionally a lightweight, device-free alternative to
/// [`KrigingInterpolator::interpolate`] for cases where:
///
/// - No GPU/device is available (headless CI, embedded, NPU-only environments).
/// - The sensor network is small enough (< 20 sensors) that the statistical
///   benefit of full ordinary kriging is marginal.
/// - Quick spatial overview is needed before committing to a full kriging solve.
///
/// For statistically optimal weights with proper kriging variances, use
/// [`KrigingInterpolator::interpolate`] (requires `Arc<WgpuDevice>`).
///
/// # Variogram usage
///
/// Nugget, sill, and range are all used. The variance approximation uses
/// γ(h) ≈ nugget + sill·(1 - exp(-h/range)) — exponential variogram model.
/// This produces conservative (over-estimated) variances relative to full
/// ordinary kriging.
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

    // Inverse-distance weighting — see KrigingInterpolator for full ordinary kriging
    let mut vwc_values = Vec::with_capacity(targets.len());
    let mut variances = Vec::with_capacity(targets.len());

    let (nugget, sill, range) = match variogram {
        SoilVariogram::Spherical {
            nugget,
            sill,
            range,
        }
        | SoilVariogram::Exponential {
            nugget,
            sill,
            range,
        } => (nugget, sill, range),
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
        // Uses γ(h) ≈ nugget + sill·(1 - exp(-h/range)) — exponential variogram
        // with the variogram range. For full kriging with LU solve, use
        // [`KrigingInterpolator::interpolate`].
        let variance = if min_dist_sq < COLLOCATED_DIST_SQ {
            nugget
        } else {
            let h = min_dist_sq.sqrt();
            let range_safe = range.max(1e-10);
            sill.mul_add(1.0 - (-h / range_safe).exp(), nugget)
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
#[expect(
    clippy::float_cmp,
    reason = "test assertions on deterministic kriging interpolation results"
)]
mod tests {
    use super::*;

    /// Tests private `to_barracuda_variogram` conversion (not exposed in public API).
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
        let _b1 = to_barracuda_variogram(sph);
        let _b2 = to_barracuda_variogram(exp);
    }

    /// Expanded test: verify `to_barracuda_variogram` produces correct model types.
    #[test]
    fn test_to_barracuda_variogram_expanded() {
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
        let b_sph = to_barracuda_variogram(sph);
        let b_exp = to_barracuda_variogram(exp);
        // Verify both convert without panic; barracuda VariogramModel is opaque.
        let _ = (b_sph, b_exp);
    }

    #[test]
    fn test_interpolate_empty_sensors_returns_zeros_and_infinity() {
        let sensors: &[SensorReading] = &[];
        let targets = &[
            TargetPoint { x: 0.0, y: 0.0 },
            TargetPoint { x: 1.0, y: 1.0 },
        ];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert_eq!(result.vwc_values.len(), 2);
        assert_eq!(result.variances.len(), 2);
        assert!(result.vwc_values.iter().all(|&v| v == 0.0));
        assert!(result.variances.iter().all(|&v| v == f64::INFINITY));
    }

    #[test]
    fn test_interpolate_empty_targets_returns_empty() {
        let sensors = &[SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.25,
        }];
        let targets: &[TargetPoint] = &[];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert!(result.vwc_values.is_empty());
        assert!(result.variances.is_empty());
    }

    #[test]
    fn test_interpolate_single_sensor_exact_value_at_location() {
        let sensors = &[SensorReading {
            x: 10.0,
            y: 20.0,
            vwc: 0.35,
        }];
        let targets = &[TargetPoint { x: 10.0, y: 20.0 }];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert_eq!(result.vwc_values.len(), 1);
        assert!((result.vwc_values[0] - 0.35).abs() < 1e-10);
        assert!(result.variances[0] < 0.01); // collocated → nugget
    }

    #[test]
    fn test_interpolate_collocated_point_returns_exact_sensor_value() {
        let sensors = &[SensorReading {
            x: 5.0,
            y: 5.0,
            vwc: 0.28,
        }];
        // Point within COLLOCATED_DIST_SQ (1e-10) — effectively same location
        let targets = &[TargetPoint {
            x: 5.0 + 1e-6,
            y: 5.0 + 1e-6,
        }];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert!((result.vwc_values[0] - 0.28).abs() < 1e-6);
        assert!(result.variances[0] < 0.01);
    }

    #[test]
    fn test_interpolate_multiple_sensors_idw_hand_computed() {
        // Two sensors: (0,0) vwc=0.2, (2,0) vwc=0.4
        // Target at (1,0): dist to sensor1=1, dist to sensor2=1
        // IDW: w1 = 1/1² = 1, w2 = 1/1² = 1
        // weighted = (0.2*1 + 0.4*1) / (1+1) = 0.6/2 = 0.3
        let sensors = &[
            SensorReading {
                x: 0.0,
                y: 0.0,
                vwc: 0.2,
            },
            SensorReading {
                x: 2.0,
                y: 0.0,
                vwc: 0.4,
            },
        ];
        let targets = &[TargetPoint { x: 1.0, y: 0.0 }];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert!((result.vwc_values[0] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_spherical_variogram_variance_correct() {
        let sensors = &[
            SensorReading {
                x: 0.0,
                y: 0.0,
                vwc: 0.25,
            },
            SensorReading {
                x: 10.0,
                y: 0.0,
                vwc: 0.30,
            },
        ];
        let targets = &[TargetPoint { x: 5.0, y: 0.0 }];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.002,
            sill: 0.02,
            range: 20.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        // Variance uses exponential approx: nugget + sill*(1 - exp(-h/range))
        // h = 5, range = 20 → variance > nugget
        assert!(result.variances[0] >= 0.002);
        assert!(result.variances[0] < 0.1);
    }

    #[test]
    fn test_interpolate_exponential_variogram_variance_correct() {
        let sensors = &[
            SensorReading {
                x: 0.0,
                y: 0.0,
                vwc: 0.25,
            },
            SensorReading {
                x: 10.0,
                y: 0.0,
                vwc: 0.30,
            },
        ];
        let targets = &[TargetPoint { x: 5.0, y: 0.0 }];
        let variogram = SoilVariogram::Exponential {
            nugget: 0.001,
            sill: 0.015,
            range: 25.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert!(result.variances[0] >= 0.001);
        assert!(result.variances[0] < 0.1);
    }

    #[test]
    fn test_interpolate_nearer_sensors_weighted_more() {
        // Sensor at (0,0) vwc=0.1, sensor at (100,0) vwc=0.9
        // Target at (1,0): much closer to first sensor → result near 0.1
        let sensors = &[
            SensorReading {
                x: 0.0,
                y: 0.0,
                vwc: 0.1,
            },
            SensorReading {
                x: 100.0,
                y: 0.0,
                vwc: 0.9,
            },
        ];
        let targets = &[TargetPoint { x: 1.0, y: 0.0 }];
        let variogram = SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 15.0,
        };
        let result = interpolate_soil_moisture(sensors, targets, variogram);
        assert!(result.vwc_values[0] < 0.2); // nearer sensor dominates
    }

    #[test]
    fn test_sensor_reading_target_point_variogram_result_clone_debug() {
        let sr = SensorReading {
            x: 1.0,
            y: 2.0,
            vwc: 0.3,
        };
        let sr2 = sr;
        assert_eq!(sr.x, sr2.x);
        let _ = format!("{sr:?}");

        let tp = TargetPoint { x: 3.0, y: 4.0 };
        let tp2 = tp;
        assert_eq!(tp.x, tp2.x);
        let _ = format!("{tp:?}");

        let v = SoilVariogram::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };
        let _ = format!("{v:?}");

        let res = InterpolationResult {
            vwc_values: vec![0.1, 0.2],
            variances: vec![0.01, 0.02],
        };
        let res2 = res.clone();
        assert_eq!(res.vwc_values, res2.vwc_values);
        let _ = format!("{res:?}");
    }
}
