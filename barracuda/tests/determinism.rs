// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::unwrap_used)]
#![allow(clippy::float_cmp)]
//! Determinism tests: rerun-identical with fixed seeds.
//!
//! These tests run each computation twice with identical inputs and assert
//! bit-exact equality. Used to detect non-determinism from threading,
//! floating-point order dependence, or RNG.

use airspring_barracuda::eco::diversity::shannon;
use airspring_barracuda::eco::evapotranspiration::{daily_et0, DailyEt0Input, Et0Result};
use airspring_barracuda::eco::runoff::scs_cn_runoff_standard;
use airspring_barracuda::eco::water_balance::{simulate_season, DailyInput, WaterBalanceState};
use airspring_barracuda::nautilus::{AirSpringBrain, AirSpringBrainConfig, WeatherObservation};

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

/// ET₀ determinism: FAO-56 Penman-Monteith ET₀ computed twice with identical
/// inputs must produce bit-exact results.
#[test]
fn test_et0_determinism() {
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

/// Water balance determinism: A full season simulation run twice with
/// identical parameters must produce bit-exact outputs.
#[test]
fn test_water_balance_determinism() {
    let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let inputs: Vec<DailyInput> = (0..60)
        .map(|day| DailyInput {
            precipitation: if day % 5 == 0 { 20.0 } else { 0.0 },
            irrigation: if day % 12 == 0 { 30.0 } else { 0.0 },
            et0: 4.5,
            kc: 1.0,
        })
        .collect();

    let (final1, out1) = simulate_season(&state, &inputs);
    let (final2, out2) = simulate_season(&state, &inputs);

    assert_eq!(final1.depletion, final2.depletion);
    assert_eq!(out1.len(), out2.len());
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert_eq!(a.depletion, b.depletion);
        assert_eq!(a.etc, b.etc);
        assert_eq!(a.deep_percolation, b.deep_percolation);
        assert_eq!(a.runoff, b.runoff);
        assert_eq!(a.ks, b.ks);
        assert_eq!(a.actual_et, b.actual_et);
        assert_eq!(a.needs_irrigation, b.needs_irrigation);
    }
}

/// Diversity index determinism: Shannon H' computed twice from the same
/// OTU table must produce bit-exact results.
#[test]
fn test_diversity_determinism() {
    // OTU table: species abundance counts (e.g. cover crop mix or microbiome)
    let counts: Vec<f64> = vec![120.0, 85.0, 45.0, 30.0, 20.0];

    let h1 = shannon(&counts);
    let h2 = shannon(&counts);

    assert_eq!(h1, h2);
}

/// SCS-CN runoff determinism: Runoff computed twice with identical
/// precipitation and curve number must produce bit-exact results.
#[test]
fn test_scs_cn_runoff_determinism() {
    let precip_mm = 50.0;
    let cn = 75.0;

    let q1 = scs_cn_runoff_standard(precip_mm, cn);
    let q2 = scs_cn_runoff_standard(precip_mm, cn);

    assert_eq!(q1, q2);
}

/// Nautilus brain determinism: Two brains created with the same config
/// (and thus same seed) trained on identical data must produce bit-exact
/// predictions.
#[test]
fn test_nautilus_brain_determinism() {
    let config = AirSpringBrainConfig::default();

    let mut brain1 = AirSpringBrain::new(config.clone(), "determinism-test");
    let mut brain2 = AirSpringBrain::new(config, "determinism-test");

    // Same training data for both
    let observations: Vec<WeatherObservation> = (1_u16..=10)
        .map(|doy| {
            let fd = f64::from(doy);
            WeatherObservation {
                doy,
                tmax: fd.mul_add(0.5, 25.0),
                tmin: fd.mul_add(0.3, 12.0),
                rh_mean: 65.0,
                wind_2m: 2.0,
                solar_rad: 20.0,
                precip: if doy % 3 == 0 { 5.0 } else { 0.0 },
                et0_observed: fd.mul_add(0.1, 4.0),
                soil_deficit: 0.2,
                crop_stress: 0.9,
            }
        })
        .collect();

    for obs in &observations {
        brain1.observe(obs.clone());
        brain2.observe(obs.clone());
    }

    let mse1 = brain1.train();
    let mse2 = brain2.train();
    assert_eq!(mse1, mse2, "Training MSE must match");

    let test_obs = WeatherObservation {
        doy: 15,
        tmax: 28.0,
        tmin: 14.0,
        rh_mean: 60.0,
        wind_2m: 2.5,
        solar_rad: 22.0,
        precip: 0.0,
        et0_observed: 5.0,
        soil_deficit: 0.25,
        crop_stress: 0.85,
    };

    let pred1 = brain1.predict(&test_obs).expect("brain1 should be trained");
    let pred2 = brain2.predict(&test_obs).expect("brain2 should be trained");

    assert_eq!(pred1.et0, pred2.et0);
    assert_eq!(pred1.soil_deficit, pred2.soil_deficit);
    assert_eq!(pred1.crop_stress, pred2.crop_stress);
}
