// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D Richards equation solver with van Genuchten-Mualem hydraulics.
//!
//! Implements vadose zone flow: ∂θ/∂t = ∂/∂z [K(h)(∂h/∂z + 1)]
//! using finite differences in space and implicit Euler with Picard iteration in time.
//!
//! References:
//! - van Genuchten (1980) SSSA J 44:892-898
//! - Richards (1931) Physics 1:318-333

use barracuda::optimize::brent;

use crate::error::{AirSpringError, Result};

/// Pressure head clipping range (cm) to avoid numerical blowup.
const H_CLIP_MIN: f64 = -10_000.0;
const H_CLIP_MAX: f64 = 100.0;

/// Picard convergence tolerance (cm).
const PICARD_TOL: f64 = 1e-4;

/// Maximum Picard iterations per time step.
const PICARD_MAX_ITER: usize = 100;

/// Maximum absolute head for VG θ(h) input guard (cm).
const VG_H_ABS_MAX: f64 = 1e4;

/// Overflow guard for (α|h|)^n computation.
const VG_POWF_MAX: f64 = 1e10;

/// Capacity C(h) returned at saturation (h ≥ 0) — small positive value
/// ensures the mixed-form C·∂h/∂t denominator never vanishes.
const SATURATED_CAPACITY: f64 = 1e-6;

/// Minimum absolute head for capacity calculation (cm) — avoids
/// numerical instability as h → 0⁻ in the VG capacity derivative.
const CAPACITY_H_MIN: f64 = 0.1;

/// Minimum capacity allowed from VG derivative.
const CAPACITY_FLOOR: f64 = 1e-10;

/// Maximum capacity (cm⁻¹) — physical upper bound.
const CAPACITY_CEIL: f64 = 1e2;

/// Near-zero capacity guard for RHS computation.
const CAPACITY_EPSILON: f64 = 1e-12;

/// Mass balance denominator guard — below this, balance is trivially zero.
const MASS_BALANCE_EPSILON: f64 = 1e-10;

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
    let h_safe = h.abs().min(VG_H_ABS_MAX);
    let m = 1.0 - 1.0 / n_vg;
    let x = (alpha * h_safe).powf(n_vg).min(VG_POWF_MAX);
    let se = 1.0 / (1.0 + x).powf(m);
    let theta = (theta_s - theta_r).mul_add(se, theta_r);
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
    if h < H_CLIP_MIN {
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
        return SATURATED_CAPACITY;
    }
    let h_safe = h.abs().clamp(CAPACITY_H_MIN, VG_H_ABS_MAX);
    let m = 1.0 - 1.0 / n_vg;
    let x = (alpha * h_safe).powf(n_vg).min(VG_POWF_MAX);
    let denom = (1.0 + x).powf(m + 1.0);
    if denom <= 0.0 || !denom.is_finite() {
        return SATURATED_CAPACITY;
    }
    let dse_dh = m * n_vg * alpha.powf(n_vg) * h_safe.powf(n_vg - 1.0) / denom;
    let result = (theta_s - theta_r) * dse_dh;
    result.clamp(CAPACITY_FLOOR, CAPACITY_CEIL)
}

/// Inverse Van Genuchten: find pressure head `h` (cm) for a target moisture θ.
///
/// Solves θ(h) − `θ_target` = 0 using Brent's root-finding method
/// (`barracuda::optimize::brent`, neuralSpring optimizer lineage).
///
/// # Cross-Spring Provenance
///
/// | Primitive | Origin | Purpose |
/// |-----------|--------|---------|
/// | `brent` | Brent (1973) via `barracuda::optimize` S52+ | Guaranteed-convergence root-finder |
/// | `van_genuchten_theta` | van Genuchten (1980), airSpring | Forward VG retention curve |
///
/// Brent's method combines bisection's reliability with inverse-quadratic
/// interpolation speed. For VG inversion, convergence is typically < 10
/// iterations since the retention curve is smooth and monotone on (−∞, 0).
///
/// Returns `None` if `θ_target` is outside \[θr, θs\] or Brent fails.
#[must_use]
pub fn inverse_van_genuchten_h(
    theta_target: f64,
    theta_r: f64,
    theta_s: f64,
    alpha: f64,
    n_vg: f64,
) -> Option<f64> {
    if theta_target >= theta_s {
        return Some(0.0);
    }
    if theta_target <= theta_r {
        return None;
    }

    let f = |h: f64| van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg) - theta_target;

    brent(f, H_CLIP_MIN, -1e-6, 1e-8, 100).ok().map(|r| r.root)
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
        let denom = a[i].mul_add(-cp[i - 1], b[i]);
        cp[i] = if i < n - 1 { c[i] / denom } else { 0.0 };
        dp[i] = a[i].mul_add(-dp[i - 1], d[i]) / denom;
    }
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = cp[i].mul_add(-x[i + 1], dp[i]);
    }
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
        for _picard in 0..PICARD_MAX_ITER {
            h_old.copy_from_slice(&h);

            a.fill(0.0);
            b.fill(0.0);
            c.fill(0.0);
            d.fill(0.0);

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

            thomas_solve(&a, &b, &c, &d, &mut h);

            let omega = 0.2_f64;
            for (hi, h_old_i) in h.iter_mut().zip(h_old.iter()) {
                *hi = omega
                    .mul_add(*hi, (1.0 - omega) * *h_old_i)
                    .clamp(H_CLIP_MIN, H_CLIP_MAX);
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
                &mut q_buf,
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

    // --- van_genuchten_theta edge cases ---
    #[test]
    fn test_van_genuchten_theta_h_clip_min() {
        let p = sand_params();
        let theta = van_genuchten_theta(-10_000.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(theta >= p.theta_r && theta <= p.theta_s);
        assert!(theta < p.theta_s); // very dry
    }

    #[test]
    fn test_van_genuchten_theta_very_negative_h() {
        let p = sand_params();
        let theta = van_genuchten_theta(-50_000.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(theta >= p.theta_r && theta <= p.theta_s);
    }

    #[test]
    fn test_van_genuchten_theta_slightly_below_zero() {
        let p = sand_params();
        let theta = van_genuchten_theta(-0.1, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(theta > p.theta_r && theta <= p.theta_s);
    }

    #[test]
    fn test_van_genuchten_theta_positive_h() {
        let p = sand_params();
        let theta = van_genuchten_theta(5.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((theta - p.theta_s).abs() < 1e-10);
    }

    // --- van_genuchten_k edge cases ---
    #[test]
    fn test_van_genuchten_k_below_clip_min() {
        let p = sand_params();
        let k = van_genuchten_k(-15_000.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(k.abs() < f64::EPSILON);
    }

    #[test]
    fn test_van_genuchten_k_at_clip_min() {
        let p = sand_params();
        let k = van_genuchten_k(-10_000.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(k >= 0.0 && k <= p.ks);
    }

    // --- van_genuchten_capacity edge cases ---
    #[test]
    fn test_van_genuchten_capacity_saturated() {
        let p = sand_params();
        let c = van_genuchten_capacity(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((c - super::SATURATED_CAPACITY).abs() < f64::EPSILON);
    }

    #[test]
    fn test_van_genuchten_capacity_positive_h() {
        let p = sand_params();
        let c = van_genuchten_capacity(10.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!((c - super::SATURATED_CAPACITY).abs() < f64::EPSILON);
    }

    #[test]
    fn test_van_genuchten_capacity_extreme_negative() {
        let p = sand_params();
        let c = van_genuchten_capacity(-100.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(c > 0.0 && c < 1e2);
    }

    // --- Error paths ---
    #[test]
    fn test_solve_richards_n_nodes_one() {
        let p = sand_params();
        let result = solve_richards_1d(&p, 100.0, 1, -5.0, -5.0, true, true, 0.1, 0.01);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(crate::error::AirSpringError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_solve_richards_dt_zero() {
        let p = sand_params();
        let result = solve_richards_1d(&p, 100.0, 20, -5.0, -5.0, true, true, 0.1, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_richards_duration_zero() {
        let p = sand_params();
        let result = solve_richards_1d(&p, 100.0, 20, -5.0, -5.0, true, true, 0.0, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_richards_dt_negative() {
        let p = sand_params();
        let result = solve_richards_1d(&p, 100.0, 20, -5.0, -5.0, true, true, 0.1, -0.01);
        assert!(result.is_err());
    }

    // --- cumulative_drainage ---
    #[test]
    fn test_cumulative_drainage() {
        let p = sand_params();
        let profiles =
            solve_richards_1d(&p, 100.0, 20, -50.0, -50.0, true, true, 0.5, 0.05).unwrap();
        let drainage = cumulative_drainage(&p, &profiles, 0.05);
        assert_eq!(drainage.len(), profiles.len());
        for (i, &d) in drainage.iter().enumerate() {
            assert!(d >= 0.0, "drainage[{i}] = {d} should be non-negative");
        }
        for i in 1..drainage.len() {
            assert!(
                drainage[i] >= drainage[i - 1],
                "drainage should be accumulating: {} >= {}",
                drainage[i],
                drainage[i - 1]
            );
        }
    }

    #[test]
    fn test_cumulative_drainage_empty_profiles() {
        let p = sand_params();
        let profiles: Vec<RichardsProfile> = vec![];
        let drainage = cumulative_drainage(&p, &profiles, 0.01);
        assert!(drainage.is_empty());
    }

    // --- mass_balance_check ---
    #[test]
    fn test_mass_balance_check() {
        let p = sand_params();
        // Zero flux top + free drainage: closed system, mass balance should be small
        let profiles =
            solve_richards_1d(&p, 100.0, 20, -50.0, -50.0, true, true, 0.2, 0.02).unwrap();
        let dz = 100.0 / 20.0;
        let err_pct = mass_balance_check(&p, &profiles, -50.0, -50.0, true, 0.02, dz);
        assert!(
            err_pct < 15.0,
            "mass balance error {err_pct}% should be small"
        );
    }

    #[test]
    fn test_mass_balance_check_empty_profiles() {
        let p = sand_params();
        let profiles: Vec<RichardsProfile> = vec![];
        let err = mass_balance_check(&p, &profiles, -20.0, -10.0, false, 0.01, 5.0);
        assert!(err.abs() < f64::EPSILON);
    }

    // --- Zero flux top ---
    #[test]
    fn test_zero_flux_top() {
        let p = sand_params();
        let profiles =
            solve_richards_1d(&p, 50.0, 15, -30.0, 0.0, true, true, 0.05, 0.005).unwrap();
        let dz = 50.0 / 15.0;
        let err_pct = mass_balance_check(&p, &profiles, -30.0, 0.0, true, 0.005, dz);
        assert!(err_pct < 10.0);
        // With zero flux top and free drainage, storage should decrease or stay similar
        let theta_init = van_genuchten_theta(-30.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        let theta_final: f64 = profiles.last().unwrap().theta.iter().sum::<f64>()
            / profiles.last().unwrap().theta.len() as f64;
        assert!(theta_final <= theta_init + 0.05); // no significant inflow
    }

    // --- Free drainage bottom: true and false ---
    #[test]
    fn test_free_drainage_bottom_true() {
        let p = sand_params();
        let profiles =
            solve_richards_1d(&p, 100.0, 20, -100.0, -100.0, true, true, 0.2, 0.02).unwrap();
        let drainage = cumulative_drainage(&p, &profiles, 0.02);
        assert!(drainage.last().unwrap() > &0.0);
    }

    #[test]
    fn test_free_drainage_bottom_false() {
        let p = sand_params();
        let profiles =
            solve_richards_1d(&p, 100.0, 20, -50.0, -50.0, true, false, 0.1, 0.01).unwrap();
        assert!(!profiles.is_empty());
        let drainage = cumulative_drainage(&p, &profiles, 0.01);
        // With bottom_free_drain=false, cumulative_drainage still computes K at bottom
        // but conceptually no free drainage; profiles should still be valid
        assert_eq!(drainage.len(), profiles.len());
    }

    // --- Loam soil ---
    fn loam_params() -> VanGenuchtenParams {
        VanGenuchtenParams {
            theta_r: 0.078,
            theta_s: 0.43,
            alpha: 0.036,
            n_vg: 1.56,
            ks: 24.96,
        }
    }

    #[test]
    fn test_loam_soil_solve() {
        let p = loam_params();
        let profiles =
            solve_richards_1d(&p, 80.0, 16, -40.0, -20.0, false, true, 0.05, 0.005).unwrap();
        assert!(!profiles.is_empty());
        assert_eq!(profiles[0].h.len(), 16);
        assert!(profiles.last().unwrap().theta[0] > loam_params().theta_r);
    }

    #[test]
    fn test_loam_van_genuchten_functions() {
        let p = loam_params();
        let theta = van_genuchten_theta(-20.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(theta > p.theta_r && theta < p.theta_s);
        let k = van_genuchten_k(-20.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(k > 0.0 && k < p.ks);
        let c = van_genuchten_capacity(-20.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        assert!(c > 0.0);
    }

    // --- Multiple time steps ---
    #[test]
    fn test_multiple_time_steps_profile_count() {
        let p = sand_params();
        let duration = 0.1_f64;
        let dt = 0.01_f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n_steps_expected = (duration / dt).ceil() as usize;
        let profiles =
            solve_richards_1d(&p, 100.0, 20, -10.0, -10.0, true, true, duration, dt).unwrap();
        assert_eq!(profiles.len(), n_steps_expected);
    }

    #[test]
    fn test_multiple_time_steps_fractional() {
        let p = sand_params();
        let profiles =
            solve_richards_1d(&p, 100.0, 20, -10.0, -10.0, true, true, 0.07, 0.02).unwrap();
        // 0.07/0.02 = 3.5 -> ceil = 4 steps
        assert_eq!(profiles.len(), 4);
    }

    // --- Picard failure / explicit fallback (stiff clay) ---
    fn clay_params() -> VanGenuchtenParams {
        VanGenuchtenParams {
            theta_r: 0.068,
            theta_s: 0.38,
            alpha: 0.008,
            n_vg: 1.09,
            ks: 4.8,
        }
    }

    #[test]
    fn test_clay_stiff_solver() {
        let p = clay_params();
        // Stiff clay with large head gradient may trigger Picard fallback
        let profiles = solve_richards_1d(
            &p, 50.0, 25, -500.0, // very dry initial
            0.0,    // saturated top - large gradient
            false, true, 0.01, 0.0005, // fine time step
        )
        .unwrap();
        assert!(!profiles.is_empty());
        assert_eq!(profiles[0].h.len(), 25);
        for (i, pr) in profiles.iter().enumerate() {
            for (j, &h) in pr.h.iter().enumerate() {
                assert!(
                    (-10_100.0..=110.0).contains(&h),
                    "profile {i} node {j} h={h}"
                );
            }
        }
    }

    #[test]
    fn test_inverse_vg_round_trip_silt_loam() {
        let (theta_r, theta_s, alpha, n_vg) = (0.067, 0.45, 0.02, 1.41);
        for &h_orig in &[-1.0, -10.0, -50.0, -100.0, -500.0, -1000.0, -5000.0] {
            let theta = van_genuchten_theta(h_orig, theta_r, theta_s, alpha, n_vg);
            let h_inv = inverse_van_genuchten_h(theta, theta_r, theta_s, alpha, n_vg)
                .expect("should invert");
            let theta_check = van_genuchten_theta(h_inv, theta_r, theta_s, alpha, n_vg);
            assert!(
                (theta_check - theta).abs() < 1e-6,
                "Round-trip θ at h={h_orig}: expected {theta:.6}, got {theta_check:.6}"
            );
        }
    }

    #[test]
    fn test_inverse_vg_saturated_returns_zero() {
        let h = inverse_van_genuchten_h(0.45, 0.067, 0.45, 0.02, 1.41);
        assert_eq!(h, Some(0.0), "θ=θs should map to h=0");
    }

    #[test]
    fn test_inverse_vg_below_residual_returns_none() {
        assert!(inverse_van_genuchten_h(0.01, 0.067, 0.45, 0.02, 1.41).is_none());
    }

    #[test]
    fn test_inverse_vg_multiple_soil_types() {
        let soils = [
            ("sand", 0.045, 0.43, 0.145, 2.68),
            ("clay", 0.068, 0.38, 0.008, 1.09),
            ("loam", 0.078, 0.43, 0.036, 1.56),
        ];
        for (name, theta_r, theta_s, alpha, n_vg) in soils {
            let h_test = -100.0;
            let theta = van_genuchten_theta(h_test, theta_r, theta_s, alpha, n_vg);
            if let Some(h_inv) = inverse_van_genuchten_h(theta, theta_r, theta_s, alpha, n_vg) {
                let theta_rt = van_genuchten_theta(h_inv, theta_r, theta_s, alpha, n_vg);
                assert!(
                    (theta_rt - theta).abs() < 1e-5,
                    "{name}: round-trip fail θ={theta:.6} → h={h_inv:.2} → θ={theta_rt:.6}"
                );
            }
        }
    }
}
