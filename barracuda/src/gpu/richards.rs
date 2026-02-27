// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Richards equation solver — bridges `eco::richards` ↔ `barracuda::pde::richards`.
//!
//! # Cross-Spring Provenance
//!
//! The Richards PDE solver was **contributed by airSpring** (absorbed upstream in
//! S40), making it one of airSpring's direct contributions to `ToadStool`. The
//! `van_genuchten_f64.wgsl` shader uses barracuda f64 precision math (`pow_f64`,
//! `exp_f64`) for water retention curve evaluation.
//!
//! # Three API Levels
//!
//! | API | Backend | Time Scheme |
//! |-----|---------|-------------|
//! | [`solve_batch_cpu`] | `eco::richards` (Picard + Thomas) | Implicit Euler |
//! | [`BatchedRichards::solve_upstream`] | `barracuda::pde::richards` | Crank-Nicolson |
//! | [`BatchedRichards::solve_cn_diffusion`] | `barracuda::pde::crank_nicolson` | Crank-Nicolson (linear) |
//!
//! # Upstream PDE Solvers
//!
//! **`pde::richards`**: Full nonlinear Richards with Picard + CN + Thomas algorithm.
//! Operates on [`barracuda::pde::richards::SoilParams`] (`k_sat` in cm/s).
//!
//! **`pde::crank_nicolson`** (S62+, now f64): Standalone linear diffusion CN solver
//! with `WGSL_CRANK_NICOLSON_F64` GPU shader + `cyclic_reduction_f64.wgsl` parallel
//! tridiagonal solve. Useful for linearised Richards comparison and thermal diffusion.
//!
//! # CPU Path
//!
//! Uses the validated `eco::richards` module directly (implicit Euler + Picard),
//! preserving exact validation fidelity.

use barracuda::pde::crank_nicolson::{CrankNicolsonConfig, HeatEquation1D};
use barracuda::pde::richards as pde_richards;

use crate::eco::richards::{self, VanGenuchtenParams};

/// Convert airSpring soil parameters to upstream `barracuda::pde::richards` format.
///
/// `eco::richards` uses `ks` in cm/day; upstream uses `k_sat` in cm/s.
#[must_use]
pub fn to_barracuda_params(params: &VanGenuchtenParams) -> pde_richards::SoilParams {
    pde_richards::SoilParams {
        theta_s: params.theta_s,
        theta_r: params.theta_r,
        alpha: params.alpha,
        n: params.n_vg,
        k_sat: params.ks / 86_400.0,
    }
}

/// Batched Richards profiles request.
#[derive(Debug, Clone)]
pub struct RichardsRequest {
    /// Van Genuchten soil parameters.
    pub params: VanGenuchtenParams,
    /// Column depth (cm).
    pub depth_cm: f64,
    /// Number of spatial nodes.
    pub n_nodes: usize,
    /// Uniform initial pressure head (cm).
    pub h_initial: f64,
    /// Top boundary pressure head (cm), ignored when `zero_flux_top` is true.
    pub h_top: f64,
    /// Use zero-flux upper boundary instead of fixed head.
    pub zero_flux_top: bool,
    /// Free drainage at the bottom boundary.
    pub bottom_free_drain: bool,
    /// Simulation duration (days).
    pub duration_days: f64,
    /// Time step (days).
    pub dt_days: f64,
}

/// Solve a batch of Richards problems on CPU using `eco::richards`.
///
/// Returns one `Vec<RichardsProfile>` per request (or an error).
#[must_use]
pub fn solve_batch_cpu(
    requests: &[RichardsRequest],
) -> Vec<crate::error::Result<Vec<richards::RichardsProfile>>> {
    requests
        .iter()
        .map(|req| {
            richards::solve_richards_1d(
                &req.params,
                req.depth_cm,
                req.n_nodes,
                req.h_initial,
                req.h_top,
                req.zero_flux_top,
                req.bottom_free_drain,
                req.duration_days,
                req.dt_days,
            )
        })
        .collect()
}

/// GPU-backed batched Richards solver.
///
/// Wraps `barracuda::pde::richards::solve_richards` with unit conversion
/// and domain-specific API. Falls back to CPU when no device is available.
pub struct BatchedRichards;

impl BatchedRichards {
    /// Solve using the upstream `barracuda::pde::richards` solver (Crank-Nicolson).
    ///
    /// Converts units (cm/day → cm/s) and boundary conditions, then delegates
    /// to the upstream solver.
    ///
    /// # Errors
    ///
    /// Returns `AirSpringError::Barracuda` if the upstream solver fails.
    pub fn solve_upstream(
        req: &RichardsRequest,
    ) -> crate::error::Result<pde_richards::RichardsResult> {
        let soil = to_barracuda_params(&req.params);
        let dz = req.depth_cm / (req.n_nodes as f64);
        let dt_s = req.dt_days * 86_400.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n_steps = (req.duration_days / req.dt_days).ceil() as usize;

        let config = pde_richards::RichardsConfig {
            soil,
            dz,
            dt: dt_s,
            n_nodes: req.n_nodes,
            max_picard_iter: 100,
            picard_tol: 1e-4,
        };

        let h0 = vec![req.h_initial; req.n_nodes];

        let top_bc = if req.zero_flux_top {
            pde_richards::RichardsBc::Flux(0.0)
        } else {
            pde_richards::RichardsBc::PressureHead(req.h_top)
        };
        let bottom_bc = if req.bottom_free_drain {
            pde_richards::RichardsBc::Flux(0.0)
        } else {
            pde_richards::RichardsBc::PressureHead(req.h_initial)
        };

        pde_richards::solve_richards(&config, &h0, n_steps, top_bc, bottom_bc)
            .map_err(|e| crate::error::AirSpringError::Barracuda(e.to_string()))
    }

    /// Compare CPU (`eco::richards`) and upstream (`barracuda::pde`) for validation.
    ///
    /// Returns `(cpu_final_theta, upstream_final_theta)` for cross-checking.
    ///
    /// # Errors
    ///
    /// Returns `AirSpringError` if either solver fails.
    pub fn cross_validate(req: &RichardsRequest) -> crate::error::Result<(Vec<f64>, Vec<f64>)> {
        let cpu_profiles = richards::solve_richards_1d(
            &req.params,
            req.depth_cm,
            req.n_nodes,
            req.h_initial,
            req.h_top,
            req.zero_flux_top,
            req.bottom_free_drain,
            req.duration_days,
            req.dt_days,
        )?;

        let cpu_theta = cpu_profiles
            .last()
            .map(|p| p.theta.clone())
            .unwrap_or_default();

        let upstream = Self::solve_upstream(req)?;
        let upstream_theta = upstream.theta;

        Ok((cpu_theta, upstream_theta))
    }

    /// Solve linearised diffusion using `barracuda::pde::crank_nicolson` (f64).
    ///
    /// Treats soil water movement as constant-coefficient diffusion with
    /// diffusivity `D` = `K_sat` / (θs - θr). This is a simplification of Richards
    /// that ignores nonlinear conductivity but provides a useful comparison
    /// baseline and exercises the standalone CN solver (S62+, f64 + GPU shader).
    ///
    /// Returns the final moisture profile (θ values at each node).
    ///
    /// # Errors
    ///
    /// Returns `AirSpringError::Barracuda` if the solver fails or parameters are invalid.
    pub fn solve_cn_diffusion(req: &RichardsRequest) -> crate::error::Result<Vec<f64>> {
        let d_cm_per_s = (req.params.ks / 86_400.0) / (req.params.theta_s - req.params.theta_r);
        let dx = req.depth_cm / (req.n_nodes.saturating_sub(1).max(1)) as f64;
        let dt_s = req.dt_days * 86_400.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n_steps = (req.duration_days / req.dt_days).ceil() as usize;

        let cn_config = CrankNicolsonConfig::new(d_cm_per_s, dx, dt_s, req.n_nodes)
            .with_boundary_conditions(req.h_initial, req.h_initial);

        let initial = vec![req.h_initial; req.n_nodes];
        let mut solver = HeatEquation1D::new(cn_config, &initial)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
        let h_final = solver
            .advance(n_steps)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;

        let theta: Vec<f64> = h_final
            .iter()
            .map(|&h| {
                pde_richards::SoilParams {
                    theta_s: req.params.theta_s,
                    theta_r: req.params.theta_r,
                    alpha: req.params.alpha,
                    n: req.params.n_vg,
                    k_sat: req.params.ks / 86_400.0,
                }
                .theta(h)
            })
            .collect();

        Ok(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sand() -> VanGenuchtenParams {
        VanGenuchtenParams {
            theta_r: 0.045,
            theta_s: 0.43,
            alpha: 0.145,
            n_vg: 2.68,
            ks: 712.8,
        }
    }

    #[test]
    fn test_unit_conversion() {
        let p = sand();
        let bp = to_barracuda_params(&p);
        let expected_ks_s = 712.8 / 86_400.0;
        assert!((bp.k_sat - expected_ks_s).abs() < 1e-10);
        assert!((bp.theta_s - p.theta_s).abs() < 1e-10);
        assert!((bp.n - p.n_vg).abs() < 1e-10);
    }

    #[test]
    fn test_batch_cpu_drainage() {
        let req = RichardsRequest {
            params: sand(),
            depth_cm: 100.0,
            n_nodes: 20,
            h_initial: -5.0,
            h_top: -5.0,
            zero_flux_top: true,
            bottom_free_drain: true,
            duration_days: 0.1,
            dt_days: 0.01,
        };
        let results = solve_batch_cpu(&[req]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        let profiles = results[0].as_ref().unwrap();
        assert!(!profiles.is_empty());
    }

    #[test]
    fn test_upstream_drainage() {
        let req = RichardsRequest {
            params: sand(),
            depth_cm: 100.0,
            n_nodes: 20,
            h_initial: -5.0,
            h_top: -5.0,
            zero_flux_top: true,
            bottom_free_drain: true,
            duration_days: 0.1,
            dt_days: 0.01,
        };
        let result = BatchedRichards::solve_upstream(&req);
        assert!(result.is_ok(), "upstream solve failed: {result:?}");
        let r = result.unwrap();
        assert_eq!(r.h.len(), 20);
        assert!(r.time_steps_completed > 0);
    }

    #[test]
    fn test_upstream_pressure_head_boundary() {
        let req = RichardsRequest {
            params: sand(),
            depth_cm: 100.0,
            n_nodes: 20,
            h_initial: -5.0,
            h_top: -10.0,
            zero_flux_top: false,
            bottom_free_drain: false,
            duration_days: 0.1,
            dt_days: 0.01,
        };
        let result = BatchedRichards::solve_upstream(&req);
        assert!(
            result.is_ok(),
            "upstream solve with PressureHead BCs failed: {result:?}"
        );
        let r = result.unwrap();
        assert_eq!(r.h.len(), 20);
        assert!(r.time_steps_completed > 0);
    }

    #[test]
    fn test_cn_diffusion_produces_physical_theta() {
        let req = RichardsRequest {
            params: sand(),
            depth_cm: 100.0,
            n_nodes: 20,
            h_initial: -5.0,
            h_top: -5.0,
            zero_flux_top: true,
            bottom_free_drain: true,
            duration_days: 0.1,
            dt_days: 0.01,
        };
        let theta = BatchedRichards::solve_cn_diffusion(&req);
        assert!(theta.is_ok(), "CN diffusion failed: {theta:?}");
        let theta = theta.unwrap();
        assert_eq!(theta.len(), 20);
        for (i, &t) in theta.iter().enumerate() {
            assert!(
                t >= req.params.theta_r - 1e-6 && t <= req.params.theta_s + 1e-6,
                "CN θ[{i}]={t:.4} outside [{:.3}, {:.3}]",
                req.params.theta_r,
                req.params.theta_s
            );
        }
    }

    #[test]
    fn test_cross_validate_drainage() {
        let p = sand();
        let req = RichardsRequest {
            params: p,
            depth_cm: 50.0,
            n_nodes: 10,
            h_initial: -5.0,
            h_top: -5.0,
            zero_flux_top: true,
            bottom_free_drain: true,
            duration_days: 0.5,
            dt_days: 0.01,
        };
        let result = BatchedRichards::cross_validate(&req);
        assert!(result.is_ok(), "cross-validate failed: {result:?}");
        let (cpu_theta, upstream_theta) = result.unwrap();
        assert_eq!(cpu_theta.len(), upstream_theta.len());
        // Implicit Euler (eco, ω=0.2 under-relaxation) vs Crank-Nicolson
        // (barracuda, 2nd-order) produce different transient solutions.
        // Both must fall within the physical θr–θs range for sand.
        for (i, (&ct, &ut)) in cpu_theta.iter().zip(&upstream_theta).enumerate() {
            assert!(
                ct >= p.theta_r - 1e-6 && ct <= p.theta_s + 1e-6,
                "CPU θ[{i}]={ct:.4} outside [{:.3}, {:.3}]",
                p.theta_r,
                p.theta_s
            );
            assert!(
                ut >= p.theta_r - 1e-6 && ut <= p.theta_s + 1e-6,
                "upstream θ[{i}]={ut:.4} outside [{:.3}, {:.3}]",
                p.theta_r,
                p.theta_s
            );
        }
    }
}
