// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::float_cmp)]
//! Bitwise determinism tests — same inputs must produce identical outputs.
//!
//! These tests run each computation twice and assert exact equality (not
//! approximate tolerance). Used to detect non-determinism from threading,
//! floating-point order dependence, or RNG.

use airspring_barracuda::eco::evapotranspiration::{
    daily_et0, saturation_vapour_pressure, DailyEt0Input, Et0Result,
};
use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};
use airspring_barracuda::eco::van_genuchten::{
    inverse_van_genuchten_h, van_genuchten_capacity, van_genuchten_k, van_genuchten_theta,
};
use airspring_barracuda::eco::water_balance::{DailyInput, WaterBalanceState};
use airspring_barracuda::gpu::isotherm::fit_langmuir_nm;

fn assert_et0_result_eq(a: &Et0Result, b: &Et0Result) {
    assert_eq!(a.et0, b.et0, "et0");
    assert_eq!(a.rn, b.rn, "rn");
    assert_eq!(a.g, b.g, "g");
    assert_eq!(a.delta, b.delta, "delta");
    assert_eq!(a.gamma, b.gamma, "gamma");
    assert_eq!(a.es, b.es, "es");
    assert_eq!(a.vpd, b.vpd, "vpd");
    assert_eq!(a.ra, b.ra, "ra");
}

#[test]
fn test_et0_pm_deterministic() {
    let input = DailyEt0Input {
        tmin: 15.0,
        tmax: 28.0,
        tmean: None,
        solar_radiation: 18.0,
        wind_speed_2m: 2.0,
        actual_vapour_pressure: 1.5,
        elevation_m: 50.0,
        latitude_deg: 45.0,
        day_of_year: 200,
    };
    let result1 = daily_et0(&input);
    let result2 = daily_et0(&input);
    assert_et0_result_eq(&result1, &result2);
}

#[test]
fn test_water_balance_deterministic() {
    let state1 = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let state2 = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let input = DailyInput {
        precipitation: 5.0,
        irrigation: 0.0,
        et0: 4.0,
        kc: 1.0,
    };
    let mut s1 = state1;
    let mut s2 = state2;
    let out1 = s1.step(&input);
    let out2 = s2.step(&input);
    assert_eq!(out1.depletion, out2.depletion);
    assert_eq!(out1.etc, out2.etc);
    assert_eq!(out1.deep_percolation, out2.deep_percolation);
    assert_eq!(out1.runoff, out2.runoff);
    assert_eq!(out1.ks, out2.ks);
    assert_eq!(out1.actual_et, out2.actual_et);
    assert_eq!(out1.needs_irrigation, out2.needs_irrigation);
}

#[test]
fn test_saturation_vapour_pressure_deterministic() {
    let temp = 25.0;
    let result1 = saturation_vapour_pressure(temp);
    let result2 = saturation_vapour_pressure(temp);
    assert_eq!(result1, result2);
}

#[test]
fn test_van_genuchten_deterministic() {
    let (theta_r, theta_s, alpha, n_vg, ks) = (0.045, 0.43, 0.145, 2.68, 712.8);
    let h = -50.0;

    let theta1 = van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg);
    let theta2 = van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg);
    assert_eq!(theta1, theta2);

    let k1 = van_genuchten_k(h, ks, theta_r, theta_s, alpha, n_vg);
    let k2 = van_genuchten_k(h, ks, theta_r, theta_s, alpha, n_vg);
    assert_eq!(k1, k2);

    let c1 = van_genuchten_capacity(h, theta_r, theta_s, alpha, n_vg);
    let c2 = van_genuchten_capacity(h, theta_r, theta_s, alpha, n_vg);
    assert_eq!(c1, c2);

    let theta_mid = theta_r.midpoint(theta_s);
    let inv1 = inverse_van_genuchten_h(theta_mid, theta_r, theta_s, alpha, n_vg);
    let inv2 = inverse_van_genuchten_h(theta_mid, theta_r, theta_s, alpha, n_vg);
    assert_eq!(inv1, inv2);
}

#[test]
fn test_isotherm_nelder_mead_deterministic() {
    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    let fit1 = fit_langmuir_nm(&ce, &qe).expect("fit should succeed");
    let fit2 = fit_langmuir_nm(&ce, &qe).expect("fit should succeed");

    assert_eq!(fit1.model, fit2.model);
    assert_eq!(fit1.params.len(), fit2.params.len());
    for (i, (p1, p2)) in fit1.params.iter().zip(fit2.params.iter()).enumerate() {
        assert_eq!(p1, p2, "params[{i}]");
    }
    assert_eq!(fit1.r_squared, fit2.r_squared);
    assert_eq!(fit1.rmse, fit2.rmse);
}

#[test]
fn test_richards_deterministic() {
    let params = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    let profiles1 = solve_richards_1d(&params, 100.0, 20, -50.0, -50.0, true, true, 0.1, 0.01)
        .expect("solve should succeed");
    let profiles2 = solve_richards_1d(&params, 100.0, 20, -50.0, -50.0, true, true, 0.1, 0.01)
        .expect("solve should succeed");

    assert_eq!(profiles1.len(), profiles2.len());
    for (p1, p2) in profiles1.iter().zip(profiles2.iter()) {
        assert_eq!(p1.z.len(), p2.z.len());
        for (z1, z2) in p1.z.iter().zip(p2.z.iter()) {
            assert_eq!(z1, z2);
        }
        for (h1, h2) in p1.h.iter().zip(p2.h.iter()) {
            assert_eq!(h1, h2);
        }
        for (t1, t2) in p1.theta.iter().zip(p2.theta.iter()) {
            assert_eq!(t1, t2);
        }
    }
}
