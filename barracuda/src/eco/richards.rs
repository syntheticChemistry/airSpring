// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D Richards equation solver with van Genuchten-Mualem hydraulics.
//!
//! Implements vadose zone flow: ∂θ/∂t = ∂/∂z [K(h)(∂h/∂z + 1)]
//! using finite differences in space and implicit Euler with Picard iteration in time.
//!
//! References:
//! - van Genuchten (1980) SSSA J 44:892-898
//! - Richards (1931) Physics 1:318-333

use crate::error::{AirSpringError, Result};

/// Default pressure head clipping for `richards_rhs`.
const H_CLIP_MIN: f64 = -10_000.0;
const H_CLIP_MAX: f64 = 100.0;

/// Near-zero capacity guard for RHS computation.
const CAPACITY_EPSILON: f64 = 1e-12;

/// Mass balance denominator guard — below this, balance is trivially zero.
const MASS_BALANCE_EPSILON: f64 = 1e-10;

/// Solver configuration for the implicit Picard iteration.
///
/// Exposes the numerical parameters that were previously hardcoded.
/// Use [`RichardsConfig::default()`] to get the validated defaults.
#[derive(Debug, Clone, Copy)]
pub struct RichardsConfig {
    /// Minimum pressure head clipping (cm). Default: -10 000.
    pub h_clip_min: f64,
    /// Maximum pressure head clipping (cm). Default: 100.
    pub h_clip_max: f64,
    /// Picard convergence tolerance (cm). Default: 1e-4.
    pub picard_tol: f64,
    /// Maximum Picard iterations per time step. Default: 100.
    pub picard_max_iter: usize,
    /// Under-relaxation factor (0, 1]. Default: 0.2.
    pub relaxation: f64,
}

impl Default for RichardsConfig {
    fn default() -> Self {
        Self {
            h_clip_min: -10_000.0,
            h_clip_max: 100.0,
            picard_tol: 1e-4,
            picard_max_iter: 100,
            relaxation: 0.2,
        }
    }
}

// Re-export VG types and functions for backward compatibility.
pub use super::van_genuchten::{
    inverse_van_genuchten_h, van_genuchten_capacity, van_genuchten_k, van_genuchten_theta,
    VanGenuchtenParams,
};

/// 1D Richards equation solution at a single time step.
#[derive(Debug, Clone)]
pub struct RichardsProfile {
    /// Node depths (cm), positive downward.
    pub z: Vec<f64>,
    /// Pressure head (cm).
    pub h: Vec<f64>,
    /// Water content (m³/m³).
    pub theta: Vec<f64>,
}

/// Solve tridiagonal system Ax = d via `barracuda::linalg::tridiagonal_solve`.
///
/// # Cross-Spring Provenance
///
/// | Primitive | Origin | Upstream |
/// |-----------|--------|----------|
/// | Thomas algorithm | airSpring (pre-v0.5.8) | Local implementation |
/// | `tridiagonal_solve` | `barracuda::linalg` (S52+) | Shared `ToadStool` primitive |
/// | `CyclicReductionF64` | `barracuda::ops` (S62+) | GPU variant for batch PDE |
///
/// The local Thomas solver was replaced by the upstream `barracuda::linalg::tridiagonal_solve`
/// to eliminate duplicate code. The upstream version uses the same Thomas algorithm with
/// identical numerical properties but exposes `Result` for singularity detection.
///
/// Accepts n-length padded arrays (a\[0\]=0, c\[n-1\]=0) matching the Picard assembly
/// loop, and extracts n-1 sub/super-diagonal slices for the upstream API.
#[allow(clippy::many_single_char_names)]
fn tridiag_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64], x: &mut [f64]) -> bool {
    let n = d.len();
    if n == 0 {
        return true;
    }
    let sub = &a[1..];
    let sup = &c[..n - 1];
    barracuda::linalg::tridiagonal_solve(sub, b, sup, d).is_ok_and(|sol| {
        x[..n].copy_from_slice(&sol);
        true
    })
}

/// Compute flux divergence dh/dt from Richards RHS (for explicit step).
#[allow(clippy::too_many_arguments)]
fn richards_rhs(
    h: &[f64],
    params: &VanGenuchtenParams,
    dz: f64,
    h_top: f64,
    zero_flux_top: bool,
    bottom_free_drain: bool,
    dh_dt: &mut [f64],
    q_buf: &mut [f64],
) {
    let n = h.len();
    let theta_r = params.theta_r;
    let theta_s = params.theta_s;
    let alpha = params.alpha;
    let n_vg = params.n_vg;
    let ks = params.ks;

    let q = q_buf;
    if zero_flux_top {
        q[0] = 0.0;
    } else {
        let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
        q[0] = k_top * ((h_top - h[0]) / (0.5 * dz) + 1.0);
    }
    for i in 0..n - 1 {
        let k_mid = 0.5
            * (van_genuchten_k(h[i], ks, theta_r, theta_s, alpha, n_vg)
                + van_genuchten_k(h[i + 1], ks, theta_r, theta_s, alpha, n_vg));
        q[i + 1] = k_mid * ((h[i + 1] - h[i]) / dz + 1.0);
    }
    if bottom_free_drain {
        q[n] = van_genuchten_k(h[n - 1], ks, theta_r, theta_s, alpha, n_vg);
    } else {
        q[n] = 0.0;
    }

    for i in 0..n {
        let dtheta_dt = (q[i] - q[i + 1]) / dz;
        let c = van_genuchten_capacity(
            h[i].clamp(H_CLIP_MIN, H_CLIP_MAX),
            theta_r,
            theta_s,
            alpha,
            n_vg,
        );
        dh_dt[i] = if c > CAPACITY_EPSILON {
            dtheta_dt / c
        } else {
            0.0
        };
    }
}

/// Solve 1D Richards equation using implicit Euler with Picard iteration.
/// Falls back to explicit Euler when Picard fails (stiff problems).
///
/// Method of lines: finite differences in space, implicit Euler in time
/// with Picard linearization for the nonlinear terms.
///
/// # Errors
///
/// Returns `AirSpringError::InvalidInput` if both methods fail.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn solve_richards_1d(
    params: &VanGenuchtenParams,
    depth_cm: f64,
    n_nodes: usize,
    h_initial: f64,
    h_top: f64,
    zero_flux_top: bool,
    bottom_free_drain: bool,
    duration_days: f64,
    dt_days: f64,
) -> Result<Vec<RichardsProfile>> {
    solve_richards_1d_with_config(
        params,
        depth_cm,
        n_nodes,
        h_initial,
        h_top,
        zero_flux_top,
        bottom_free_drain,
        duration_days,
        dt_days,
        &RichardsConfig::default(),
    )
}

/// Like [`solve_richards_1d`] but accepts a [`RichardsConfig`] for solver tuning.
///
/// # Errors
///
/// Returns `AirSpringError::InvalidInput` if inputs are invalid.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn solve_richards_1d_with_config(
    params: &VanGenuchtenParams,
    depth_cm: f64,
    n_nodes: usize,
    h_initial: f64,
    h_top: f64,
    zero_flux_top: bool,
    bottom_free_drain: bool,
    duration_days: f64,
    dt_days: f64,
    config: &RichardsConfig,
) -> Result<Vec<RichardsProfile>> {
    if n_nodes < 2 {
        return Err(AirSpringError::InvalidInput(
            "n_nodes must be >= 2".to_string(),
        ));
    }
    if dt_days <= 0.0 || duration_days <= 0.0 {
        return Err(AirSpringError::InvalidInput(
            "dt_days and duration_days must be positive".to_string(),
        ));
    }

    let dz = depth_cm / (n_nodes as f64);
    let theta_r = params.theta_r;
    let theta_s = params.theta_s;
    let alpha = params.alpha;
    let n_vg = params.n_vg;
    let ks = params.ks;

    let h_clip_min = config.h_clip_min;
    let h_clip_max = config.h_clip_max;
    let mut h: Vec<f64> = vec![h_initial.clamp(h_clip_min, h_clip_max); n_nodes];
    let mut profiles = Vec::new();

    let n_steps = usize::try_from((duration_days / dt_days).ceil() as u64).unwrap_or(1);
    let mut t = 0.0_f64;

    let mut a = vec![0.0_f64; n_nodes];
    let mut b = vec![0.0_f64; n_nodes];
    let mut c = vec![0.0_f64; n_nodes];
    let mut d = vec![0.0_f64; n_nodes];
    let mut h_prev = vec![0.0_f64; n_nodes];
    let mut h_old = vec![0.0_f64; n_nodes];
    let mut q_buf = vec![0.0_f64; n_nodes + 1];

    for _step in 0..n_steps {
        let t_next = (t + dt_days).min(duration_days);
        let dt = t_next - t;
        t = t_next;

        h_prev.copy_from_slice(&h);
        let mut converged = false;
        for _picard in 0..config.picard_max_iter {
            h_old.copy_from_slice(&h);

            a.fill(0.0);
            b.fill(0.0);
            c.fill(0.0);
            d.fill(0.0);

            for i in 0..n_nodes {
                let hi = h[i].clamp(h_clip_min, h_clip_max);
                let ci = van_genuchten_capacity(hi, theta_r, theta_s, alpha, n_vg);
                let ki = van_genuchten_k(hi, ks, theta_r, theta_s, alpha, n_vg);

                let k_im12 = if i == 0 {
                    if zero_flux_top {
                        0.0
                    } else {
                        let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
                        0.5 * (k_top + ki)
                    }
                } else {
                    let ki_m1 = van_genuchten_k(
                        h[i - 1].clamp(h_clip_min, h_clip_max),
                        ks,
                        theta_r,
                        theta_s,
                        alpha,
                        n_vg,
                    );
                    0.5 * (ki_m1 + ki)
                };

                let k_ip12 = if i == n_nodes - 1 {
                    0.0
                } else {
                    let ki_p1 = van_genuchten_k(
                        h[i + 1].clamp(h_clip_min, h_clip_max),
                        ks,
                        theta_r,
                        theta_s,
                        alpha,
                        n_vg,
                    );
                    0.5 * (ki + ki_p1)
                };

                let dz2 = dz * dz;
                if i > 0 {
                    a[i] = -k_im12 / dz2;
                }
                b[i] = ci / dt + k_im12 / dz2 + k_ip12 / dz2;
                if i < n_nodes - 1 {
                    c[i] = -k_ip12 / dz2;
                }

                if i == 0 {
                    if zero_flux_top {
                        d[i] = (ci / dt).mul_add(h_old[i], -k_ip12 / dz);
                    } else {
                        let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
                        let q_top = k_top * ((h_top - h_old[0]) / (0.5 * dz) + 1.0);
                        d[i] = (ci / dt).mul_add(h_old[i], q_top / dz - k_ip12 / dz);
                    }
                } else if i == n_nodes - 1 && bottom_free_drain {
                    d[i] = (ci / dt).mul_add(h_old[i], (ki - k_im12) / dz);
                } else {
                    d[i] = (ci / dt).mul_add(h_old[i], (k_ip12 - k_im12) / dz);
                }
            }

            if !tridiag_solve(&a, &b, &c, &d, &mut h) {
                break;
            }

            let omega = config.relaxation;
            for (hi, h_old_i) in h.iter_mut().zip(h_old.iter()) {
                *hi = omega
                    .mul_add(*hi, (1.0 - omega) * *h_old_i)
                    .clamp(h_clip_min, h_clip_max);
            }

            let max_diff = h
                .iter()
                .zip(h_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_diff < config.picard_tol {
                converged = true;
                break;
            }
        }

        if !converged {
            let mut dh_dt = vec![0.0_f64; n_nodes];
            richards_rhs(
                &h_prev,
                params,
                dz,
                h_top,
                zero_flux_top,
                bottom_free_drain,
                &mut dh_dt,
                &mut q_buf,
            );
            for (hi, (h_prev_i, dhdti)) in h.iter_mut().zip(h_prev.iter().zip(dh_dt.iter())) {
                *hi = (*h_prev_i + dt * dhdti).clamp(h_clip_min, h_clip_max);
            }
        }

        let theta: Vec<f64> = h
            .iter()
            .map(|&hi| van_genuchten_theta(hi, theta_r, theta_s, alpha, n_vg))
            .collect();
        let z: Vec<f64> = (0..n_nodes).map(|i| dz * (i as f64 + 0.5)).collect();

        profiles.push(RichardsProfile {
            z,
            h: h.clone(),
            theta,
        });
    }

    Ok(profiles)
}

/// Cumulative drainage at bottom (cm) for a sequence of profiles.
/// Assumes free drainage: q = K at bottom.
#[must_use]
pub fn cumulative_drainage(
    params: &VanGenuchtenParams,
    profiles: &[RichardsProfile],
    dt_days: f64,
) -> Vec<f64> {
    let mut cum = 0.0_f64;
    profiles
        .iter()
        .map(|p| {
            let h_bot = *p.h.last().unwrap_or(&0.0);
            let k_bot = van_genuchten_k(
                h_bot,
                params.ks,
                params.theta_r,
                params.theta_s,
                params.alpha,
                params.n_vg,
            );
            cum += k_bot * dt_days;
            cum
        })
        .collect()
}

/// Mass balance check: inflow - outflow ≈ storage change.
/// Returns error as percentage of total water flux.
/// `h_initial` is the uniform initial pressure head (cm) used to compute initial θ.
///
/// Returns `0.0` when `profiles` is empty (no water movement to check).
#[must_use]
pub fn mass_balance_check(
    params: &VanGenuchtenParams,
    profiles: &[RichardsProfile],
    h_initial: f64,
    h_top: f64,
    zero_flux_top: bool,
    dt_days: f64,
    dz: f64,
) -> f64 {
    if profiles.is_empty() {
        return 0.0;
    }
    let n = profiles[0].h.len();
    let theta_r = params.theta_r;
    let theta_s = params.theta_s;
    let alpha = params.alpha;
    let n_vg = params.n_vg;
    let ks = params.ks;

    let mut total_inflow = 0.0;
    let mut total_outflow = 0.0;

    for (j, p) in profiles.iter().enumerate() {
        let q_top = if zero_flux_top {
            0.0
        } else {
            let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
            k_top * ((h_top - p.h[0]) / (0.5 * dz) + 1.0)
        };
        let q_top_prev = if j == 0 {
            if zero_flux_top {
                0.0
            } else {
                let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
                k_top * ((h_top - h_initial) / (0.5 * dz) + 1.0)
            }
        } else if zero_flux_top {
            0.0
        } else {
            let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
            k_top * ((h_top - profiles[j - 1].h[0]) / (0.5 * dz) + 1.0)
        };
        let k_bot = van_genuchten_k(p.h[n - 1], ks, theta_r, theta_s, alpha, n_vg);
        let k_bot_prev = if j == 0 {
            van_genuchten_k(h_initial, ks, theta_r, theta_s, alpha, n_vg)
        } else {
            van_genuchten_k(profiles[j - 1].h[n - 1], ks, theta_r, theta_s, alpha, n_vg)
        };
        total_inflow += 0.5 * (q_top + q_top_prev) * dt_days;
        total_outflow += 0.5 * (k_bot + k_bot_prev) * dt_days;
    }

    let theta_init_val = van_genuchten_theta(h_initial, theta_r, theta_s, alpha, n_vg);
    let theta_init: Vec<f64> = (0..n).map(|_| theta_init_val).collect();
    let Some(last_profile) = profiles.last() else {
        return 0.0;
    };
    let theta_final: Vec<f64> = last_profile
        .h
        .iter()
        .map(|&hi| van_genuchten_theta(hi, theta_r, theta_s, alpha, n_vg))
        .collect();
    let storage_change: f64 = theta_init
        .iter()
        .zip(theta_final.iter())
        .map(|(a, b)| (b - a) * dz)
        .sum();

    let imbalance = total_inflow - total_outflow - storage_change;
    let total_water = total_inflow.abs() + total_outflow.abs() + storage_change.abs();
    if total_water < MASS_BALANCE_EPSILON {
        return 0.0;
    }
    100.0 * imbalance.abs() / total_water
}
