// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched Hargreaves-Samani ET₀ GPU orchestrator (Tier B — pending op=6 absorption).
//!
//! The Hargreaves-Samani (1985) equation estimates ET₀ from temperature range and
//! extraterrestrial radiation alone — no humidity or wind data required:
//!
//! ```text
//! ET₀_HG = 0.0023 × (Tmean + 17.8) × √(Tmax − Tmin) × Ra
//! ```
//!
//! # Two API Levels
//!
//! | API | Device? | Backend |
//! |-----|:-------:|---------|
//! | [`BatchedHargreaves::compute`] | No | CPU via `eco::evapotranspiration::hargreaves_et0` |
//! | [`BatchedHargreaves::compute_gpu`] | Yes | GPU via `BatchedElementwiseF64` (op=6, pending) |
//!
//! # GPU Readiness
//!
//! The CPU path is fully validated against FAO-56 Eq. 52. The GPU interface
//! is wired and ready for `ToadStool` absorption. Once `ToadStool` adds
//! `hargreaves_batch` (op=6, stride=4: `[tmax, tmin, lat_rad, doy]`),
//! the GPU path activates automatically.
//!
//! # Reference
//!
//! Hargreaves GH, Samani ZA (1985) "Reference crop evapotranspiration from
//! temperature." *Applied Engineering in Agriculture* 1(2): 96-99.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::BatchedElementwiseF64;

use crate::eco::solar;

/// Per-day input for batched Hargreaves ET₀.
///
/// Stride-4 layout for future GPU shader: `[tmax, tmin, lat_rad, doy]`.
#[derive(Debug, Clone, Copy)]
pub struct HargreavesDay {
    /// Maximum temperature (°C).
    pub tmax: f64,
    /// Minimum temperature (°C).
    pub tmin: f64,
    /// Latitude (decimal degrees).
    pub latitude_deg: f64,
    /// Day of year (1–366).
    pub day_of_year: u32,
}

/// Backend selection for batched Hargreaves ET₀.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Validated CPU path (always available).
    #[default]
    Cpu,
    /// GPU path via `BatchedElementwiseF64` op=6 (pending `ToadStool` absorption).
    Gpu,
}

/// Result from a batched Hargreaves ET₀ computation.
#[derive(Debug, Clone)]
pub struct BatchedHargreavesResult {
    /// ET₀ values in mm/day, one per input row.
    pub et0_values: Vec<f64>,
    /// Which backend was actually used.
    pub backend_used: Backend,
}

/// Batched Hargreaves-Samani ET₀ orchestrator.
///
/// Computes Hargreaves ET₀ for N station-days in a single call.
/// Currently CPU-only (Tier B). When `ToadStool` absorbs op=6,
/// the GPU engine activates automatically.
pub struct BatchedHargreaves {
    backend: Backend,
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedHargreaves {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedHargreaves")
            .field("backend", &self.backend)
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedHargreaves {
    /// Create with GPU engine (Tier B — currently falls back to CPU).
    ///
    /// Once `ToadStool` absorbs `hargreaves_batch` (op=6), this path
    /// will dispatch to the GPU shader automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device)?;
        Ok(Self {
            backend: Backend::Gpu,
            gpu_engine: Some(engine),
        })
    }

    /// Returns a reference to the GPU engine, if available.
    /// Used for `ToadStool` GPU dispatch when the shader is wired.
    #[must_use]
    pub const fn gpu_engine(&self) -> Option<&BatchedElementwiseF64> {
        self.gpu_engine.as_ref()
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            gpu_engine: None,
        }
    }

    /// Compute Hargreaves ET₀ for a batch of station-days.
    ///
    /// When `ToadStool` absorbs op=6 (stride=4: `[tmax, tmin, lat_rad, doy]`),
    /// this method dispatches to the GPU engine automatically. Until then,
    /// the validated CPU path is authoritative.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails irrecoverably.
    pub fn compute_gpu(
        &self,
        inputs: &[HargreavesDay],
    ) -> crate::error::Result<BatchedHargreavesResult> {
        // TODO(toadstool): When op=6 is absorbed, replace with:
        //   let packed = Self::pack_gpu_input(inputs);
        //   engine.execute(&packed, inputs.len(), Op::Hargreaves)?
        let et0_values = Self::compute_cpu_batch(inputs);
        Ok(BatchedHargreavesResult {
            et0_values,
            backend_used: Backend::Cpu,
        })
    }

    /// Pack inputs into stride-4 GPU layout: `[tmax, tmin, lat_rad, doy]`.
    ///
    /// Ready for `ToadStool` op=6 absorption — produces the flat `f64` array
    /// that `BatchedElementwiseF64::execute` expects.
    #[must_use]
    pub fn pack_gpu_input(inputs: &[HargreavesDay]) -> Vec<f64> {
        let mut data = Vec::with_capacity(inputs.len() * 4);
        for d in inputs {
            data.push(d.tmax);
            data.push(d.tmin);
            data.push(d.latitude_deg.to_radians());
            data.push(f64::from(d.day_of_year));
        }
        data
    }

    /// Compute Hargreaves ET₀ using the validated CPU path.
    #[must_use]
    pub fn compute(&self, inputs: &[HargreavesDay]) -> BatchedHargreavesResult {
        BatchedHargreavesResult {
            et0_values: Self::compute_cpu_batch(inputs),
            backend_used: Backend::Cpu,
        }
    }

    fn compute_cpu_batch(inputs: &[HargreavesDay]) -> Vec<f64> {
        // Pre-compute Ra (mm/day) for each day from latitude and DOY.
        let lambda = 2.45_f64;
        let ra: Vec<f64> = inputs
            .iter()
            .map(|d| {
                solar::extraterrestrial_radiation(d.latitude_deg.to_radians(), d.day_of_year)
                    / lambda
            })
            .collect();
        let tmax: Vec<f64> = inputs.iter().map(|d| d.tmax).collect();
        let tmin: Vec<f64> = inputs.iter().map(|d| d.tmin).collect();

        // Delegate to ToadStool batch (absorbed from airSpring metalForge, S66 R-S66-002).
        // Upstream uses (ra, tmax, tmin) parameter order; returns None only on length mismatch.
        barracuda::stats::hargreaves_et0_batch(&ra, &tmax, &tmin).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eco::evapotranspiration as et;

    fn sample_day() -> HargreavesDay {
        HargreavesDay {
            tmax: 21.5,
            tmin: 12.3,
            latitude_deg: 50.80,
            day_of_year: 187,
        }
    }

    #[test]
    fn single_day_positive() {
        let engine = BatchedHargreaves::cpu();
        let result = engine.compute(&[sample_day()]);
        assert_eq!(result.et0_values.len(), 1);
        assert!(
            result.et0_values[0] > 0.5 && result.et0_values[0] < 10.0,
            "ET₀ = {:.3}",
            result.et0_values[0]
        );
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    #[test]
    fn matches_scalar() {
        let day = sample_day();
        let lat_rad = day.latitude_deg.to_radians();
        let extra_rad = solar::extraterrestrial_radiation(lat_rad, day.day_of_year);
        let ra_mm = extra_rad / 2.45;
        let scalar = et::hargreaves_et0(day.tmin, day.tmax, ra_mm);

        let engine = BatchedHargreaves::cpu();
        let batched = engine.compute(&[day]);
        assert!(
            (batched.et0_values[0] - scalar).abs() < f64::EPSILON,
            "Batched {:.6} != scalar {scalar:.6}",
            batched.et0_values[0]
        );
    }

    #[test]
    fn multiple_days() {
        let engine = BatchedHargreaves::cpu();
        let inputs: Vec<HargreavesDay> = (0..100)
            .map(|i| HargreavesDay {
                day_of_year: 100 + i,
                ..sample_day()
            })
            .collect();
        let result = engine.compute(&inputs);
        assert_eq!(result.et0_values.len(), 100);
        for &val in &result.et0_values {
            assert!(val > 0.0, "ET₀ should be positive: {val}");
        }
    }

    #[test]
    fn empty_batch() {
        let engine = BatchedHargreaves::cpu();
        let result = engine.compute(&[]);
        assert!(result.et0_values.is_empty());
    }

    #[test]
    fn deterministic() {
        let engine = BatchedHargreaves::cpu();
        let inputs = vec![sample_day(); 50];
        let r1 = engine.compute(&inputs);
        let r2 = engine.compute(&inputs);
        for (a, b) in r1.et0_values.iter().zip(&r2.et0_values) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn seasonal_variation() {
        let engine = BatchedHargreaves::cpu();
        let winter = HargreavesDay {
            tmax: 2.0,
            tmin: -5.0,
            latitude_deg: 50.80,
            day_of_year: 15,
        };
        let summer = sample_day();
        let r = engine.compute(&[winter, summer]);
        assert!(
            r.et0_values[0] < r.et0_values[1],
            "Winter ET₀ ({:.3}) should be less than summer ({:.3})",
            r.et0_values[0],
            r.et0_values[1]
        );
    }

    #[test]
    fn compute_gpu_fallback() {
        let engine = BatchedHargreaves::cpu();
        let result = engine.compute_gpu(&[sample_day()]).unwrap();
        assert_eq!(result.et0_values.len(), 1);
        assert!(result.et0_values[0] > 0.5 && result.et0_values[0] < 10.0);
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    #[test]
    fn debug_format() {
        let engine = BatchedHargreaves::cpu();
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("BatchedHargreaves"));
        assert!(dbg.contains("Cpu"));
    }

    #[test]
    fn default_backend_is_cpu() {
        assert_eq!(Backend::default(), Backend::Cpu);
    }

    #[test]
    fn high_latitude_lower_et0() {
        let engine = BatchedHargreaves::cpu();
        let low_lat = HargreavesDay {
            latitude_deg: 30.0,
            ..sample_day()
        };
        let high_lat = HargreavesDay {
            latitude_deg: 60.0,
            ..sample_day()
        };
        let r = engine.compute(&[low_lat, high_lat]);
        assert!(
            r.et0_values[0] > r.et0_values[1],
            "Lower latitude ET₀ ({:.3}) should exceed higher ({:.3}) in summer",
            r.et0_values[0],
            r.et0_values[1]
        );
    }
}
