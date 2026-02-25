// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched ET₀ GPU orchestrator — GPU-first via `ToadStool` `BatchedElementwiseF64`.
//!
//! Dispatches N station-day ET₀ computations to the GPU via
//! [`barracuda::ops::batched_elementwise_f64::BatchedElementwiseF64`].
//! All four `ToadStool` issues (TS-001 through TS-004) are **resolved** as of
//! commit `0c477306`.
//!
//! # Two API Levels
//!
//! | API | Device? | Backend |
//! |-----|:-------:|---------|
//! | [`BatchedEt0::compute`] | No | CPU via `eco::evapotranspiration` |
//! | [`BatchedEt0::compute_gpu`] | Yes | GPU via `BatchedElementwiseF64` |
//!
//! # GPU Input
//!
//! The GPU shader accepts raw humidity data (`rh_max`, `rh_min`) and computes
//! actual vapour pressure internally. Use [`StationDay`] for the GPU path.
//! The CPU path still accepts [`DailyEt0Input`] with pre-computed `ea`.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{self as bef64, BatchedElementwiseF64};

use crate::eco::evapotranspiration::{self as et, DailyEt0Input};

/// Station-day input matching `ToadStool` shader layout.
///
/// `(tmax, tmin, rh_max, rh_min, wind_2m, rs, elevation, latitude, doy)`
#[derive(Debug, Clone, Copy)]
pub struct StationDay {
    /// Maximum temperature (°C).
    pub tmax: f64,
    /// Minimum temperature (°C).
    pub tmin: f64,
    /// Maximum relative humidity (%).
    pub rh_max: f64,
    /// Minimum relative humidity (%).
    pub rh_min: f64,
    /// Wind speed at 2 m (m/s).
    pub wind_2m: f64,
    /// Solar radiation Rs (MJ/m²/day).
    pub rs: f64,
    /// Elevation (m).
    pub elevation: f64,
    /// Latitude (decimal degrees).
    pub latitude: f64,
    /// Day of year (1–366).
    pub doy: u32,
}

impl StationDay {
    /// Convert to `ToadStool` `StationDayInput` tuple.
    #[must_use]
    pub const fn to_toadstool(self) -> bef64::StationDayInput {
        (
            self.tmax,
            self.tmin,
            self.rh_max,
            self.rh_min,
            self.wind_2m,
            self.rs,
            self.elevation,
            self.latitude,
            self.doy,
        )
    }
}

/// Backend selection for batched ET₀.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Validated CPU path.
    Cpu,
    /// GPU path via `ToadStool` `BatchedElementwiseF64` — **default** (all TS issues resolved).
    #[default]
    Gpu,
}

/// Result from a batched ET₀ computation.
#[derive(Debug, Clone)]
pub struct BatchedEt0Result {
    /// ET₀ values in mm/day, one per input row.
    pub et0_values: Vec<f64>,
    /// Which backend was actually used.
    pub backend_used: Backend,
}

/// Batched ET₀ orchestrator — GPU-first.
///
/// Computes FAO-56 Penman-Monteith ET₀ for N station-days in a single call.
/// With a `WgpuDevice`, dispatches to the `BatchedElementwiseF64` GPU shader.
/// Without a device, falls back to the validated CPU path.
pub struct BatchedEt0 {
    backend: Backend,
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedEt0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedEt0")
            .field("backend", &self.backend)
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedEt0 {
    /// Create a GPU-backed batched ET₀ orchestrator.
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
        Ok(Self {
            backend: Backend::Gpu,
            gpu_engine: Some(engine),
        })
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            gpu_engine: None,
        }
    }

    /// Compute ET₀ for a batch of station-days on the GPU.
    ///
    /// This is the primary GPU path. Accepts [`StationDay`] inputs which
    /// include raw humidity data (`rh_max`, `rh_min`) as required by the shader.
    ///
    /// Falls back to CPU if no GPU engine is available.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    pub fn compute_gpu(&self, inputs: &[StationDay]) -> crate::error::Result<BatchedEt0Result> {
        if let Some(engine) = &self.gpu_engine {
            let station_days: Vec<bef64::StationDayInput> =
                inputs.iter().map(|s| s.to_toadstool()).collect();
            let et0_values = engine
                .fao56_et0_batch(&station_days)
                .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
            Ok(BatchedEt0Result {
                et0_values,
                backend_used: Backend::Gpu,
            })
        } else {
            // CPU fallback: compute ea from rh_max/rh_min, then use validated path
            let et0_values: Vec<f64> = inputs
                .iter()
                .map(|s| {
                    let ea = et::actual_vapour_pressure_rh(s.tmin, s.tmax, s.rh_min, s.rh_max);
                    let input = DailyEt0Input {
                        tmin: s.tmin,
                        tmax: s.tmax,
                        tmean: None,
                        solar_radiation: s.rs,
                        wind_speed_2m: s.wind_2m,
                        actual_vapour_pressure: ea,
                        elevation_m: s.elevation,
                        latitude_deg: s.latitude,
                        day_of_year: s.doy,
                    };
                    et::daily_et0(&input).et0
                })
                .collect();
            Ok(BatchedEt0Result {
                et0_values,
                backend_used: Backend::Cpu,
            })
        }
    }

    /// Compute ET₀ using the validated CPU path (pre-computed `ea`).
    ///
    /// Each input is a [`DailyEt0Input`] from the validated CPU module.
    /// Returns one ET₀ value per input.
    #[must_use]
    pub fn compute(&self, inputs: &[DailyEt0Input]) -> BatchedEt0Result {
        let et0_values: Vec<f64> = inputs
            .iter()
            .map(|input| et::daily_et0(input).et0)
            .collect();
        BatchedEt0Result {
            et0_values,
            backend_used: Backend::Cpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> DailyEt0Input {
        DailyEt0Input {
            tmin: 12.3,
            tmax: 21.5,
            tmean: Some(16.9),
            solar_radiation: 22.07,
            wind_speed_2m: 2.078,
            actual_vapour_pressure: 1.409,
            elevation_m: 100.0,
            latitude_deg: 50.80,
            day_of_year: 187,
        }
    }

    fn sample_station_day() -> StationDay {
        StationDay {
            tmax: 21.5,
            tmin: 12.3,
            rh_max: 84.0,
            rh_min: 63.0,
            wind_2m: 2.078,
            rs: 22.07,
            elevation: 100.0,
            latitude: 50.80,
            doy: 187,
        }
    }

    #[test]
    fn test_batched_et0_single() {
        let engine = BatchedEt0::cpu();
        let result = engine.compute(&[sample_input()]);
        assert_eq!(result.et0_values.len(), 1);
        assert!(result.et0_values[0] > 2.0 && result.et0_values[0] < 6.0);
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    #[test]
    fn test_batched_et0_matches_scalar() {
        let engine = BatchedEt0::cpu();
        let input = sample_input();
        let scalar = et::daily_et0(&input).et0;
        let batched = engine.compute(std::slice::from_ref(&input));
        assert!(
            (batched.et0_values[0] - scalar).abs() < f64::EPSILON,
            "Batched {} != scalar {scalar}",
            batched.et0_values[0]
        );
    }

    #[test]
    fn test_batched_et0_multiple() {
        let engine = BatchedEt0::cpu();
        let inputs: Vec<DailyEt0Input> = (0..100)
            .map(|i| DailyEt0Input {
                day_of_year: 150 + i,
                ..sample_input()
            })
            .collect();
        let result = engine.compute(&inputs);
        assert_eq!(result.et0_values.len(), 100);
        for &val in &result.et0_values {
            assert!(val > 0.0, "ET₀ should be positive: {val}");
        }
    }

    #[test]
    fn test_batched_et0_empty() {
        let engine = BatchedEt0::cpu();
        let result = engine.compute(&[]);
        assert!(result.et0_values.is_empty());
    }

    #[test]
    fn test_batched_et0_deterministic() {
        let engine = BatchedEt0::cpu();
        let inputs = vec![sample_input(); 50];
        let r1 = engine.compute(&inputs);
        let r2 = engine.compute(&inputs);
        for (a, b) in r1.et0_values.iter().zip(&r2.et0_values) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_station_day_cpu_fallback() {
        let engine = BatchedEt0::cpu();
        let result = engine.compute_gpu(&[sample_station_day()]).unwrap();
        assert_eq!(result.et0_values.len(), 1);
        assert!(result.et0_values[0] > 2.0 && result.et0_values[0] < 6.0);
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    #[test]
    fn test_station_day_multiple() {
        let engine = BatchedEt0::cpu();
        let inputs: Vec<StationDay> = (0..100)
            .map(|i| StationDay {
                doy: 150 + i,
                ..sample_station_day()
            })
            .collect();
        let result = engine.compute_gpu(&inputs).unwrap();
        assert_eq!(result.et0_values.len(), 100);
        for &val in &result.et0_values {
            assert!(val > 0.0, "ET₀ should be positive: {val}");
        }
    }

    #[test]
    fn test_station_day_to_toadstool() {
        let sd = sample_station_day();
        let tt = sd.to_toadstool();
        assert!((tt.0 - sd.tmax).abs() < f64::EPSILON);
        assert!((tt.1 - sd.tmin).abs() < f64::EPSILON);
        assert_eq!(tt.8, sd.doy);
    }

    #[test]
    fn test_backend_default_is_gpu() {
        assert_eq!(Backend::default(), Backend::Gpu);
    }

    #[test]
    fn test_cpu_debug_format() {
        let engine = BatchedEt0::cpu();
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("Cpu"));
        assert!(dbg.contains("false"));
    }

    #[test]
    fn test_compute_gpu_empty() {
        let engine = BatchedEt0::cpu();
        let result = engine.compute_gpu(&[]).unwrap();
        assert!(result.et0_values.is_empty());
    }

    #[test]
    fn test_et0_seasonal_variation() {
        let engine = BatchedEt0::cpu();
        let winter = DailyEt0Input {
            tmin: -5.0,
            tmax: 5.0,
            tmean: Some(0.0),
            solar_radiation: 6.0,
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 0.4,
            elevation_m: 100.0,
            latitude_deg: 50.80,
            day_of_year: 15,
        };
        let summer = sample_input();
        let r = engine.compute(&[winter, summer]);
        assert!(
            r.et0_values[0] < r.et0_values[1],
            "Winter ET₀ ({}) should be less than summer ({})",
            r.et0_values[0],
            r.et0_values[1]
        );
    }
}
