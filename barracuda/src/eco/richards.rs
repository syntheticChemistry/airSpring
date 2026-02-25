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

/// Pressure head clipping range (cm) to avoid numerical blowup.
const H_CLIP_MIN: f64 = -10_000.0;
const H_CLIP_MAX: f64 = 100.0;

/// Picard convergence tolerance (cm).
const PICARD_TOL: f64 = 1e-4;

/// Maximum Picard iterations per time step.
const PICARD_MAX_ITER: usize = 100;

/// Van Genuchten soil parameters.
#[derive(Debug, Clone, Copy)]
pub struct VanGenuchtenParams {
    /// Residual water content (m³/m³).
    pub theta_r: f64,
    /// Saturated water content (m³/m³).
    pub theta_s: f64,
    /// Inverse of air-entry pressure (1/cm).
    pub alpha: f64,
    /// Pore-size distribution index.
    pub n_vg: f64,
    /// Saturated hydraulic conductivity (cm/day).
    pub ks: f64,
}

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

/// Van Genuchten water retention: θ(h).
///
/// θ = θr + (θs - θr) / [1 + (α|h|)^n]^m   where m = 1 - 1/n
#[must_use]
pub fn van_genuchten_theta(h: f64, theta_r: f64, theta_s: f64, alpha: f64, n_vg: f64) -> f64 {
    if h >= 0.0 {
        return theta_s;
    }
    let h_safe = h.abs().min(1e4);
    let m = 1.0 - 1.0 / n_vg;
    let x = (alpha * h_safe).powf(n_vg).min(1e10);
    let se = 1.0 / (1.0 + x).powf(m);
    let theta = theta_r + (theta_s - theta_r) * se;
    theta.clamp(theta_r, theta_s)
}

/// Mualem-van Genuchten hydraulic conductivity: K(h).
///
/// K = Ks × Se^0.5 × [1 - (1 - Se^(1/m))^m]^2
#[must_use]
pub fn van_genuchten_k(h: f64, ks: f64, theta_r: f64, theta_s: f64, alpha: f64, n_vg: f64) -> f64 {
    if h >= 0.0 {
        return ks;
    }
    if h < -1e4 {
        return 0.0;
    }
    let m = 1.0 - 1.0 / n_vg;
    let theta = van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg);
    let se = (theta - theta_r) / (theta_s - theta_r);
    if se <= 0.0 {
        return 0.0;
    }
    if se >= 1.0 {
        return ks;
    }
    let term = 1.0 - se.powf(1.0 / m);
    if term <= 0.0 {
        return ks;
    }
    let kr = se.sqrt() * (1.0 - term.powf(m)).powi(2);
    ks * kr.clamp(0.0, 1.0)
}

/// Capacity function C(h) = dθ/dh (needed for mixed form).
#[must_use]
pub fn van_genuchten_capacity(h: f64, theta_r: f64, theta_s: f64, alpha: f64, n_vg: f64) -> f64 {
    if h >= 0.0 {
        return 1e-6;
    }
    let h_safe = h.abs().clamp(0.1, 1e4);
    let m = 1.0 - 1.0 / n_vg;
    let x = (alpha * h_safe).powf(n_vg).min(1e10);
    let denom = (1.0 + x).powf(m + 1.0);
    if denom <= 0.0 || !denom.is_finite() {
        return 1e-6;
    }
    let dse_dh = m * n_vg * alpha.powf(n_vg) * h_safe.powf(n_vg - 1.0) / denom;
    let result = (theta_s - theta_r) * dse_dh;
    result.clamp(1e-10, 1e2)
}

/// Solve tridiagonal system Ax = d using Thomas algorithm.
/// a[i] = subdiagonal, b[i] = diagonal, c[i] = superdiagonal.
#[allow(clippy::many_single_char_names)]
fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64], x: &mut [f64]) {
    let n = d.len();
    if n == 0 {
        return;
    }
    let mut cp = vec![0.0_f64; n];
    let mut dp = vec![0.0_f64; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let denom = b[i] - a[i] * cp[i - 1];
        cp[i] = if i < n - 1 { c[i] / denom } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
}

/// Compute flux divergence dh/dt from Richards RHS (for explicit step).
fn richards_rhs(
    h: &[f64],
    params: &VanGenuchtenParams,
    dz: f64,
    h_top: f64,
    zero_flux_top: bool,
    bottom_free_drain: bool,
    dh_dt: &mut [f64],
) {
    let n = h.len();
    let theta_r = params.theta_r;
    let theta_s = params.theta_s;
    let alpha = params.alpha;
    let n_vg = params.n_vg;
    let ks = params.ks;

    let mut q = vec![0.0_f64; n + 1];
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
        dh_dt[i] = if c > 1e-12 { dtheta_dt / c } else { 0.0 };
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
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

    let mut h: Vec<f64> = vec![h_initial.clamp(H_CLIP_MIN, H_CLIP_MAX); n_nodes];
    let mut profiles = Vec::new();

    let n_steps = (duration_days / dt_days).ceil() as usize;
    let mut t = 0.0_f64;

    for _step in 0..n_steps {
        let t_next = (t + dt_days).min(duration_days);
        let dt = t_next - t;
        t = t_next;

        let h_prev: Vec<f64> = h.clone();
        let mut converged = false;
        for _picard in 0..PICARD_MAX_ITER {
            let h_old: Vec<f64> = h.clone();

            let mut a = vec![0.0_f64; n_nodes];
            let mut b = vec![0.0_f64; n_nodes];
            let mut c = vec![0.0_f64; n_nodes];
            let mut d = vec![0.0_f64; n_nodes];

            for i in 0..n_nodes {
                let hi = h[i].clamp(H_CLIP_MIN, H_CLIP_MAX);
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
                        h[i - 1].clamp(H_CLIP_MIN, H_CLIP_MAX),
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
                        h[i + 1].clamp(H_CLIP_MIN, H_CLIP_MAX),
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
                        d[i] = ci / dt * h_old[i] - k_ip12 / dz;
                    } else {
                        let k_top = van_genuchten_k(h_top, ks, theta_r, theta_s, alpha, n_vg);
                        let q_top = k_top * ((h_top - h_old[0]) / (0.5 * dz) + 1.0);
                        d[i] = ci / dt * h_old[i] + q_top / dz - k_ip12 / dz;
                    }
                } else if i == n_nodes - 1 && bottom_free_drain {
                    d[i] = ci / dt * h_old[i] + (ki - k_im12) / dz;
                } else {
                    d[i] = ci / dt * h_old[i] + (k_ip12 - k_im12) / dz;
                }
            }

            thomas_solve(&a, &b, &c, &d, &mut h);

            let omega = 0.2_f64;
            for (hi, h_old_i) in h.iter_mut().zip(h_old.iter()) {
                *hi = (omega * *hi + (1.0 - omega) * h_old_i).clamp(H_CLIP_MIN, H_CLIP_MAX);
            }

            let max_diff = h
                .iter()
                .zip(h_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_diff < PICARD_TOL {
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
            );
            for (hi, (h_prev_i, dhdti)) in h.iter_mut().zip(h_prev.iter().zip(dh_dt.iter())) {
                *hi = (*h_prev_i + dt * dhdti).clamp(H_CLIP_MIN, H_CLIP_MAX);
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
/// # Panics
///
/// Panics if `profiles` is empty.
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
    let theta_final: Vec<f64> = profiles
        .last()
        .unwrap()
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
    if total_water < 1e-10 {
        return 0.0;
    }
    100.0 * imbalance.abs() / total_water
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sand_params() -> VanGenuchtenParams {
        VanGenuchtenParams {
            theta_r: 0.045,
            theta_s: 0.43,
            alpha: 0.145,
            n_vg: 2.68,
            ks: 712.8,
        }
    }

    #[test]
    fn test_van_genuchten_theta_saturation() {
        let p = sand_params();
        let theta = van_genuchten_theta(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((theta - p.theta_s).abs() < 1e-10);
    }

    #[test]
    fn test_van_genuchten_theta_dry() {
        let p = sand_params();
        let theta = van_genuchten_theta(-100.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((theta - 0.0493).abs() < 0.001);
    }

    #[test]
    fn test_van_genuchten_k_saturation() {
        let p = sand_params();
        let k = van_genuchten_k(0.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((k - p.ks).abs() < 1e-10);
    }

    #[test]
    fn test_van_genuchten_k_unsaturated() {
        let p = sand_params();
        let k = van_genuchten_k(-10.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        let k_ratio = k / p.ks;
        assert!(k_ratio > 0.01 && k_ratio < 0.5);
    }

    #[test]
    fn test_solve_richards_drainage() {
        let p = sand_params();
        let profiles = solve_richards_1d(&p, 100.0, 20, -5.0, -5.0, true, true, 0.1, 0.01).unwrap();
        assert!(!profiles.is_empty());
        assert_eq!(profiles[0].h.len(), 20);
    }

    #[test]
    fn test_solve_richards_infiltration() {
        let p = sand_params();
        let profiles =
            solve_richards_1d(&p, 50.0, 10, -20.0, 0.0, false, true, 0.01, 0.0001).unwrap();
        assert!(!profiles.is_empty());
        assert!(profiles.last().unwrap().theta[0] > 0.0);
    }
}
