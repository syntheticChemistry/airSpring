// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated diversity metrics (Shannon, Simpson, Pielou evenness).
//!
//! Wraps `barracuda::ops::bio::diversity_fusion::DiversityFusionGpu` from `BarraCuda` S70.
//! Computes alpha diversity for multiple samples in a single GPU dispatch.
//!
//! # Cross-Spring Provenance
//!
//! - **wetSpring S28+**: Shannon, Simpson, Bray-Curtis diversity indices
//! - **neuralSpring**: GPU fusion pattern
//! - **`BarraCuda` S70**: `diversity_fusion_f64.wgsl` shader
//! - **airSpring**: Agroecology — cover crop biodiversity, soil 16S microbiome

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::diversity_fusion::{
    diversity_fusion_cpu, DiversityFusionGpu, DiversityResult,
};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Alpha diversity metrics for a single sample.
#[derive(Debug, Clone, Copy)]
pub struct DiversityMetrics {
    /// Shannon entropy H'.
    pub shannon: f64,
    /// Simpson index D.
    pub simpson: f64,
    /// Pielou evenness J'.
    pub evenness: f64,
}

impl From<DiversityResult> for DiversityMetrics {
    fn from(r: DiversityResult) -> Self {
        Self {
            shannon: r.shannon,
            simpson: r.simpson,
            evenness: r.evenness,
        }
    }
}

/// GPU-accelerated diversity fusion orchestrator.
///
/// Dispatches to `DiversityFusionGpu` when a GPU engine is configured;
/// falls back to CPU otherwise.
pub struct GpuDiversity {
    gpu_engine: Option<DiversityFusionGpu>,
}

impl std::fmt::Debug for GpuDiversity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDiversity")
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl GpuDiversity {
    /// Create with GPU engine.
    ///
    /// # Errors
    ///
    /// Returns an error if `DiversityFusionGpu` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = DiversityFusionGpu::new(device)?;
        Ok(Self {
            gpu_engine: Some(engine),
        })
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self { gpu_engine: None }
    }

    /// Compute alpha diversity for multiple samples.
    ///
    /// `abundances` is row-major `[n_samples × n_species]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `abundances.len() != n_samples * n_species` or GPU dispatch fails.
    pub fn compute_alpha(
        &self,
        abundances: &[f64],
        n_samples: usize,
        n_species: usize,
    ) -> crate::error::Result<Vec<DiversityMetrics>> {
        if abundances.len() != n_samples * n_species {
            return Err(crate::error::AirSpringError::InvalidInput(format!(
                "abundances length {} != n_samples ({}) * n_species ({})",
                abundances.len(),
                n_samples,
                n_species
            )));
        }
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        if let Some(engine) = &self.gpu_engine {
            let results = engine.compute(abundances, n_samples, n_species)?;
            Ok(results.into_iter().map(DiversityMetrics::from).collect())
        } else {
            Ok(compute_diversity_cpu(abundances, n_species))
        }
    }
}

/// CPU fallback: fused diversity via `diversity_fusion_cpu`.
#[must_use]
fn compute_diversity_cpu(abundances: &[f64], n_species: usize) -> Vec<DiversityMetrics> {
    diversity_fusion_cpu(abundances, n_species)
        .into_iter()
        .map(DiversityMetrics::from)
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        try_f64_device()
    }

    #[test]
    fn test_uniform_distribution() {
        let engine = GpuDiversity::cpu();
        let abundances = [25.0, 25.0, 25.0, 25.0];
        let results = engine.compute_alpha(&abundances, 1, 4).unwrap();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert!((r.shannon - 4.0_f64.ln()).abs() < 0.001);
        assert!((r.simpson - 0.75).abs() < 0.001);
        assert!((r.evenness - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for GpuDiversity");
            return;
        };
        let gpu_engine = GpuDiversity::gpu(device).unwrap();
        let cpu_engine = GpuDiversity::cpu();

        let abundances = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let gpu_result = gpu_engine.compute_alpha(&abundances, 1, 6).unwrap();
        let cpu_result = cpu_engine.compute_alpha(&abundances, 1, 6).unwrap();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (g, c) in gpu_result.iter().zip(&cpu_result) {
            assert!((g.shannon - c.shannon).abs() < 0.001);
            assert!((g.simpson - c.simpson).abs() < 0.001);
            assert!((g.evenness - c.evenness).abs() < 0.001);
        }
    }

    #[test]
    fn test_empty() {
        let engine = GpuDiversity::cpu();
        let result = engine.compute_alpha(&[], 0, 4).unwrap();
        assert!(result.is_empty());
    }
}
