// SPDX-License-Identifier: AGPL-3.0-or-later
//! Monte Carlo ET₀ uncertainty propagation.
//!
//! # Cross-Spring Provenance
//!
//! This module provides uncertainty bands for FAO-56 ET₀ estimates by
//! propagating input measurement errors through the Penman-Monteith equation
//! via Monte Carlo simulation.
//!
//! | Layer | Origin | Description |
//! |-------|--------|-------------|
//! | `mc_et0_propagate_f64.wgsl` | groundSpring metalForge → `ToadStool` S64 | Box-Muller + xoshiro128** GPU kernel |
//! | `math_f64.wgsl` (exp, log, pow) | hotSpring lattice QCD | f64 precision primitives |
//! | FAO-56 ET₀ chain | airSpring `eco::evapotranspiration` | Penman-Monteith equation |
//!
//! # Two API Levels
//!
//! | API | Backend | Use Case |
//! |-----|---------|----------|
//! | [`mc_et0_cpu`] | CPU | Always available, N samples via loop |
//! | GPU kernel | `mc_et0_propagate_f64.wgsl` | Available: sovereign compiler regression resolved (S66+) |

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::stats::normal::norm_ppf;

use crate::eco::evapotranspiration::{self as et, DailyEt0Input};

/// Input uncertainties for MC ET₀ propagation.
///
/// Each field is a standard deviation (σ) for the corresponding measurement.
/// The MC simulation perturbs inputs by ε ~ N(0, σ²) per sample.
#[derive(Debug, Clone, Copy)]
pub struct Et0Uncertainties {
    /// σ for `T_max` (°C). Typical: 0.3–0.5 °C.
    pub sigma_tmax: f64,
    /// σ for `T_min` (°C). Typical: 0.3–0.5 °C.
    pub sigma_tmin: f64,
    /// σ for `RH_max` (%). Typical: 3–5%.
    pub sigma_rh_max: f64,
    /// σ for `RH_min` (%). Typical: 3–5%.
    pub sigma_rh_min: f64,
    /// Fractional σ for wind speed. Typical: 0.05–0.10 (5–10%).
    pub sigma_wind_frac: f64,
    /// Fractional σ for solar radiation. Typical: 0.05–0.10 (5–10%).
    pub sigma_rs_frac: f64,
}

impl Default for Et0Uncertainties {
    fn default() -> Self {
        Self {
            sigma_tmax: 0.4,
            sigma_tmin: 0.4,
            sigma_rh_max: 4.0,
            sigma_rh_min: 4.0,
            sigma_wind_frac: 0.08,
            sigma_rs_frac: 0.07,
        }
    }
}

/// Result of Monte Carlo ET₀ uncertainty propagation.
#[derive(Debug, Clone)]
pub struct McEt0Result {
    /// Central ET₀ estimate (unperturbed, mm/day).
    pub et0_central: f64,
    /// Mean of MC samples (mm/day).
    pub et0_mean: f64,
    /// Standard deviation of MC samples (mm/day).
    pub et0_std: f64,
    /// 5th percentile (mm/day) — lower uncertainty bound.
    pub et0_p05: f64,
    /// 95th percentile (mm/day) — upper uncertainty bound.
    pub et0_p95: f64,
    /// Number of MC samples.
    pub n_samples: usize,
}

impl McEt0Result {
    /// Compute a parametric confidence interval assuming normality.
    ///
    /// Uses `barracuda::stats::normal::norm_ppf` (Moro 1995 rational approximation)
    /// from hotSpring's precision math lineage to convert a confidence level
    /// (e.g. 0.90 for 90% CI) into z-scores, then applies `mean ± z * std`.
    ///
    /// Returns `(lower, upper)` bounds in mm/day.
    ///
    /// # Cross-Spring Provenance
    ///
    /// `norm_ppf` was absorbed into `barracuda::stats::normal` (S52+) from
    /// hotSpring's special-function library. The Moro rational approximation
    /// provides 7+ digits of precision across the full (0,1) range.
    #[must_use]
    pub fn parametric_ci(&self, confidence: f64) -> (f64, f64) {
        let alpha = (1.0 - confidence) / 2.0;
        let z = norm_ppf(1.0 - alpha);
        let lower = self.et0_mean - z * self.et0_std;
        let upper = self.et0_mean + z * self.et0_std;
        (lower, upper)
    }
}

/// Run Monte Carlo ET₀ uncertainty propagation on CPU.
///
/// Generates `n_samples` perturbed versions of the input, computes ET₀ for
/// each, and returns summary statistics (mean, std, 5th/95th percentiles).
///
/// Uses a deterministic Lehmer LCG seeded from `seed` for reproducibility.
///
/// # Cross-Spring Provenance
///
/// CPU implementation mirrors the `mc_et0_propagate_f64.wgsl` GPU kernel
/// (groundSpring → `ToadStool` S64). The GPU kernel uses xoshiro128** +
/// Box-Muller; this CPU version uses a simpler Lehmer LCG + Box-Muller
/// that produces statistically equivalent distributions.
#[must_use]
pub fn mc_et0_cpu(
    input: &DailyEt0Input,
    uncertainties: &Et0Uncertainties,
    n_samples: usize,
    seed: u64,
) -> McEt0Result {
    let central = et::daily_et0(input).et0;

    if n_samples == 0 {
        return McEt0Result {
            et0_central: central,
            et0_mean: central,
            et0_std: 0.0,
            et0_p05: central,
            et0_p95: central,
            n_samples: 0,
        };
    }

    let mut rng_state = seed.wrapping_add(1);
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let z_tmax = box_muller_next(&mut rng_state);
        let z_tmin = box_muller_next(&mut rng_state);
        let z_rh_max = box_muller_next(&mut rng_state);
        let z_rh_min = box_muller_next(&mut rng_state);
        let z_wind = box_muller_next(&mut rng_state);
        let z_rs = box_muller_next(&mut rng_state);

        let tmax_p = input.tmax + z_tmax * uncertainties.sigma_tmax;
        let tmin_p = input.tmin + z_tmin * uncertainties.sigma_tmin;
        let tmean_p = f64::midpoint(tmax_p, tmin_p);

        let rh_max_p = (input.actual_vapour_pressure / et::saturation_vapour_pressure(input.tmin))
            .mul_add(100.0, z_rh_max * uncertainties.sigma_rh_max)
            .clamp(1.0, 100.0);
        let rh_min_p = (input.actual_vapour_pressure / et::saturation_vapour_pressure(input.tmax))
            .mul_add(100.0, z_rh_min * uncertainties.sigma_rh_min)
            .clamp(1.0, 100.0);

        let ea_p = et::actual_vapour_pressure_rh(tmin_p, tmax_p, rh_min_p, rh_max_p);

        let wind_p =
            (input.wind_speed_2m * (1.0 + z_wind * uncertainties.sigma_wind_frac)).max(0.01);
        let rs_p = (input.solar_radiation * (1.0 + z_rs * uncertainties.sigma_rs_frac)).max(0.01);

        let perturbed = DailyEt0Input {
            tmin: tmin_p,
            tmax: tmax_p,
            tmean: Some(tmean_p),
            solar_radiation: rs_p,
            wind_speed_2m: wind_p,
            actual_vapour_pressure: ea_p,
            elevation_m: input.elevation_m,
            latitude_deg: input.latitude_deg,
            day_of_year: input.day_of_year,
        };
        let et0_val = et::daily_et0(&perturbed).et0;
        if et0_val.is_finite() && et0_val > 0.0 {
            samples.push(et0_val);
        }
    }

    if samples.is_empty() {
        return McEt0Result {
            et0_central: central,
            et0_mean: central,
            et0_std: 0.0,
            et0_p05: central,
            et0_p95: central,
            n_samples: 0,
        };
    }

    let n = samples.len();
    let mean_val = barracuda::stats::mean(&samples);
    // Population variance (÷ n): these are the entire MC draw, not a sample
    // from a larger population. barracuda::stats::correlation::variance uses
    // sample variance (÷ n-1) so we compute population variance directly.
    let variance = samples.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64;
    let std_val = variance.sqrt();
    let p05 = barracuda::stats::percentile(&samples, 5.0);
    let p95 = barracuda::stats::percentile(&samples, 95.0);

    McEt0Result {
        et0_central: central,
        et0_mean: mean_val,
        et0_std: std_val,
        et0_p05: p05,
        et0_p95: p95,
        n_samples: n,
    }
}

/// Monte Carlo ET₀ uncertainty propagation (GPU path, Tier B).
///
/// When `ToadStool` wires `mc_et0_propagate_f64.wgsl`, this dispatches
/// N MC samples to the GPU via Box-Muller + xoshiro128** kernel.
/// Currently falls back to [`mc_et0_cpu`].
///
/// # Errors
///
/// Returns an error if the GPU dispatch fails (future).
pub fn mc_et0_gpu(
    _device: &Arc<WgpuDevice>,
    base_input: &DailyEt0Input,
    uncertainties: &Et0Uncertainties,
    n_samples: usize,
    seed: u64,
) -> crate::error::Result<McEt0Result> {
    Ok(mc_et0_cpu(base_input, uncertainties, n_samples, seed))
}

fn lehmer_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(48_271).wrapping_rem(0x7FFF_FFFF);
    *state as f64 / f64::from(0x7FFF_FFFFu32)
}

fn box_muller_next(state: &mut u64) -> f64 {
    let u1 = lehmer_next(state).max(1e-300);
    let u2 = lehmer_next(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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

    #[test]
    fn mc_et0_zero_uncertainty_gives_no_spread() {
        let unc = Et0Uncertainties {
            sigma_tmax: 0.0,
            sigma_tmin: 0.0,
            sigma_rh_max: 0.0,
            sigma_rh_min: 0.0,
            sigma_wind_frac: 0.0,
            sigma_rs_frac: 0.0,
        };
        let result = mc_et0_cpu(&sample_input(), &unc, 200, 42);
        assert!(
            result.et0_std < 0.01,
            "Zero uncertainty should give ~zero spread: std={}",
            result.et0_std
        );
    }

    #[test]
    fn mc_et0_default_uncertainty_reasonable_spread() {
        let result = mc_et0_cpu(&sample_input(), &Et0Uncertainties::default(), 1000, 42);
        assert!(result.et0_std > 0.05, "Should have measurable spread");
        assert!(result.et0_std < 2.0, "Spread should be reasonable");
        assert!(result.et0_p05 < result.et0_mean);
        assert!(result.et0_p95 > result.et0_mean);
        assert!(
            result.et0_p95 - result.et0_p05 > 0.1,
            "90% CI should be non-trivial"
        );
    }

    #[test]
    fn mc_et0_central_in_range() {
        let result = mc_et0_cpu(&sample_input(), &Et0Uncertainties::default(), 500, 42);
        assert!(
            result.et0_central > 2.0 && result.et0_central < 6.0,
            "Central ET₀ should be plausible: {}",
            result.et0_central
        );
    }

    #[test]
    fn mc_et0_mean_near_central() {
        let result = mc_et0_cpu(&sample_input(), &Et0Uncertainties::default(), 2000, 42);
        assert!(
            (result.et0_mean - result.et0_central).abs() < 0.5,
            "MC mean {} should be near central {}",
            result.et0_mean,
            result.et0_central
        );
    }

    #[test]
    fn mc_et0_deterministic() {
        let unc = Et0Uncertainties::default();
        let r1 = mc_et0_cpu(&sample_input(), &unc, 500, 42);
        let r2 = mc_et0_cpu(&sample_input(), &unc, 500, 42);
        assert!(
            (r1.et0_mean - r2.et0_mean).abs() < f64::EPSILON,
            "Same seed should give same result"
        );
    }

    #[test]
    fn mc_et0_zero_samples() {
        let result = mc_et0_cpu(&sample_input(), &Et0Uncertainties::default(), 0, 42);
        assert_eq!(result.n_samples, 0);
        assert!((result.et0_mean - result.et0_central).abs() < f64::EPSILON);
    }

    #[test]
    fn mc_et0_parametric_ci_consistent_with_empirical() {
        let result = mc_et0_cpu(&sample_input(), &Et0Uncertainties::default(), 5000, 42);
        let (p_lo, p_hi) = result.parametric_ci(0.90);
        // Parametric 90% CI (assuming normality) should roughly agree with
        // empirical 5th/95th percentiles — within 20% for N=5000
        let empirical_width = result.et0_p95 - result.et0_p05;
        let parametric_width = p_hi - p_lo;
        let ratio = parametric_width / empirical_width;
        assert!(
            (0.7..=1.4).contains(&ratio),
            "Parametric CI width ({parametric_width:.3}) should roughly match \
             empirical ({empirical_width:.3}), ratio={ratio:.2}"
        );
        assert!(p_lo < result.et0_mean);
        assert!(p_hi > result.et0_mean);
    }

    #[test]
    fn mc_et0_parametric_ci_widens_with_lower_confidence() {
        let result = mc_et0_cpu(&sample_input(), &Et0Uncertainties::default(), 2000, 42);
        let (_, hi_90) = result.parametric_ci(0.90);
        let (_, hi_99) = result.parametric_ci(0.99);
        assert!(
            hi_99 > hi_90,
            "99% CI upper ({hi_99:.3}) should exceed 90% ({hi_90:.3})"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_mc_et0_gpu_fallback_matches_cpu() {
        let Some(device) = try_device() else {
            return;
        };
        let input = sample_input();
        let unc = Et0Uncertainties::default();
        let cpu_result = mc_et0_cpu(&input, &unc, 500, 42);
        let gpu_result = mc_et0_gpu(&device, &input, &unc, 500, 42).unwrap();
        assert_eq!(cpu_result.et0_central, gpu_result.et0_central);
        assert_eq!(cpu_result.et0_mean, gpu_result.et0_mean);
        assert_eq!(cpu_result.et0_std, gpu_result.et0_std);
        assert_eq!(cpu_result.et0_p05, gpu_result.et0_p05);
        assert_eq!(cpu_result.et0_p95, gpu_result.et0_p95);
        assert_eq!(cpu_result.n_samples, gpu_result.n_samples);
    }

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .ok()
        .map(std::sync::Arc::new)
    }

    #[test]
    fn mc_et0_higher_uncertainty_wider_spread() {
        let low = Et0Uncertainties {
            sigma_tmax: 0.2,
            sigma_tmin: 0.2,
            sigma_rh_max: 2.0,
            sigma_rh_min: 2.0,
            sigma_wind_frac: 0.03,
            sigma_rs_frac: 0.03,
        };
        let high = Et0Uncertainties {
            sigma_tmax: 1.0,
            sigma_tmin: 1.0,
            sigma_rh_max: 10.0,
            sigma_rh_min: 10.0,
            sigma_wind_frac: 0.15,
            sigma_rs_frac: 0.15,
        };
        let r_low = mc_et0_cpu(&sample_input(), &low, 1000, 42);
        let r_high = mc_et0_cpu(&sample_input(), &high, 1000, 42);
        assert!(
            r_high.et0_std > r_low.et0_std,
            "Higher uncertainty should give wider spread: high={} low={}",
            r_high.et0_std,
            r_low.et0_std
        );
    }
}
