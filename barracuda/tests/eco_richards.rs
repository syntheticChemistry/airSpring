// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `eco::richards` — 1D Richards equation solver.

use airspring_barracuda::eco::richards::{
    RichardsProfile, VanGenuchtenParams, cumulative_drainage, inverse_van_genuchten_h,
    mass_balance_check, solve_richards_1d, van_genuchten_capacity, van_genuchten_k,
    van_genuchten_theta,
};
use airspring_barracuda::error::AirSpringError;

const fn sand_params() -> VanGenuchtenParams {
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
    let profiles = solve_richards_1d(&p, 50.0, 10, -20.0, 0.0, false, true, 0.01, 0.0001).unwrap();
    assert!(!profiles.is_empty());
    assert!(profiles.last().unwrap().theta[0] > 0.0);
}

#[test]
fn test_van_genuchten_theta_h_clip_min() {
    let p = sand_params();
    let theta = van_genuchten_theta(-10_000.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
    assert!(theta >= p.theta_r && theta <= p.theta_s);
    assert!(theta < p.theta_s);
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

#[test]
fn test_van_genuchten_capacity_saturated() {
    let p = sand_params();
    let c = van_genuchten_capacity(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
    assert!((c - 1e-6).abs() < f64::EPSILON);
}

#[test]
fn test_van_genuchten_capacity_positive_h() {
    let p = sand_params();
    let c = van_genuchten_capacity(10.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
    assert!((c - 1e-6).abs() < f64::EPSILON);
}

#[test]
fn test_van_genuchten_capacity_extreme_negative() {
    let p = sand_params();
    let c = van_genuchten_capacity(-100.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
    assert!(c > 0.0 && c < 1e2);
}

#[test]
fn test_solve_richards_n_nodes_one() {
    let p = sand_params();
    let result = solve_richards_1d(&p, 100.0, 1, -5.0, -5.0, true, true, 0.1, 0.01);
    assert!(result.is_err());
    assert!(matches!(result, Err(AirSpringError::InvalidInput(_))));
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

#[test]
fn test_cumulative_drainage() {
    let p = sand_params();
    let profiles = solve_richards_1d(&p, 100.0, 20, -50.0, -50.0, true, true, 0.5, 0.05).unwrap();
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

#[test]
fn test_mass_balance_check() {
    let p = sand_params();
    let profiles = solve_richards_1d(&p, 100.0, 20, -50.0, -50.0, true, true, 0.2, 0.02).unwrap();
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

#[test]
fn test_zero_flux_top() {
    let p = sand_params();
    let profiles = solve_richards_1d(&p, 50.0, 15, -30.0, 0.0, true, true, 0.05, 0.005).unwrap();
    let dz = 50.0 / 15.0;
    let err_pct = mass_balance_check(&p, &profiles, -30.0, 0.0, true, 0.005, dz);
    assert!(err_pct < 10.0);
    let theta_init = van_genuchten_theta(-30.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
    #[allow(clippy::cast_precision_loss)]
    let theta_final: f64 = profiles.last().unwrap().theta.iter().sum::<f64>()
        / profiles.last().unwrap().theta.len() as f64;
    assert!(theta_final <= theta_init + 0.05);
}

#[test]
fn test_free_drainage_bottom_true() {
    let p = sand_params();
    let profiles = solve_richards_1d(&p, 100.0, 20, -100.0, -100.0, true, true, 0.2, 0.02).unwrap();
    let drainage = cumulative_drainage(&p, &profiles, 0.02);
    assert!(drainage.last().unwrap() > &0.0);
}

#[test]
fn test_free_drainage_bottom_false() {
    let p = sand_params();
    let profiles = solve_richards_1d(&p, 100.0, 20, -50.0, -50.0, true, false, 0.1, 0.01).unwrap();
    assert!(!profiles.is_empty());
    let drainage = cumulative_drainage(&p, &profiles, 0.01);
    assert_eq!(drainage.len(), profiles.len());
}

const fn loam_params() -> VanGenuchtenParams {
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
    let profiles = solve_richards_1d(&p, 80.0, 16, -40.0, -20.0, false, true, 0.05, 0.005).unwrap();
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
    let profiles = solve_richards_1d(&p, 100.0, 20, -10.0, -10.0, true, true, 0.07, 0.02).unwrap();
    assert_eq!(profiles.len(), 4);
}

const fn clay_params() -> VanGenuchtenParams {
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
    let profiles = solve_richards_1d(&p, 50.0, 25, -500.0, 0.0, false, true, 0.01, 0.0005).unwrap();
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
        let h_inv =
            inverse_van_genuchten_h(theta, theta_r, theta_s, alpha, n_vg).expect("should invert");
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
