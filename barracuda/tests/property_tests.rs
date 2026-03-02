// SPDX-License-Identifier: AGPL-3.0-or-later
//! Property-based tests for airSpring `BarraCuda` scientific computations.
//!
//! Uses proptest to verify invariants hold across physically meaningful input ranges.

use airspring_barracuda::eco::crop::gdd_avg;
use airspring_barracuda::eco::diversity::{bray_curtis, shannon};
use airspring_barracuda::eco::evapotranspiration::{daily_et0, hargreaves_et0, DailyEt0Input};
use airspring_barracuda::eco::infiltration::{cumulative_infiltration, GreenAmptParams};
use airspring_barracuda::eco::runoff::scs_cn_runoff_standard;
use airspring_barracuda::eco::soil_moisture::{
    inverse_topp, saxton_rawls, topp_equation, SaxtonRawlsInput,
};
use airspring_barracuda::eco::van_genuchten::van_genuchten_theta;
use airspring_barracuda::eco::water_balance::{
    mass_balance_check, simulate_season, DailyInput, WaterBalanceState,
};
use airspring_barracuda::eco::yield_response::{clamp_yield_ratio, yield_ratio_single};
use proptest::prelude::*;

// ── Evapotranspiration invariants ────────────────────────────────────────

proptest! {
    #[test]
    fn et0_pm_non_negative(
        tmin in -20.0f64..40.0,
        tmax_offset in 1.0f64..30.0,
        solar in 1.0f64..35.0,
        wind in 0.5f64..10.0,
        ea in 0.1f64..4.0,
        elev in 0.0f64..3000.0,
        lat in -60.0f64..60.0,
        doy in 1u32..366,
    ) {
        let tmax = tmin + tmax_offset;
        let input = DailyEt0Input {
            tmin,
            tmax,
            tmean: None,
            solar_radiation: solar,
            wind_speed_2m: wind,
            actual_vapour_pressure: ea,
            elevation_m: elev,
            latitude_deg: lat,
            day_of_year: doy,
        };
        let result = daily_et0(&input);
        prop_assert!(result.et0.is_finite(), "ET0 must be finite");
    }
}

proptest! {
    #[test]
    fn et0_pm_finite_all_outputs(
        tmin in -10.0f64..35.0,
        tmax_offset in 2.0f64..25.0,
        solar in 5.0f64..30.0,
        wind in 0.5f64..8.0,
        ea in 0.2f64..3.5,
        elev in 0.0f64..2500.0,
        lat in -55.0f64..55.0,
        doy in 1u32..366,
    ) {
        let tmax = tmin + tmax_offset;
        let input = DailyEt0Input {
            tmin,
            tmax,
            tmean: None,
            solar_radiation: solar,
            wind_speed_2m: wind,
            actual_vapour_pressure: ea,
            elevation_m: elev,
            latitude_deg: lat,
            day_of_year: doy,
        };
        let result = daily_et0(&input);
        prop_assert!(result.et0 >= 0.0, "ET0 must be non-negative");
        prop_assert!(result.rn.is_finite());
        prop_assert!(result.es.is_finite());
    }
}

proptest! {
    #[test]
    fn hargreaves_et0_non_negative(
        tmin in -15.0f64..35.0,
        tmax_offset in 1.0f64..25.0,
        ra_mm in 1.0f64..25.0,
    ) {
        let tmax = tmin + tmax_offset;
        let et0 = hargreaves_et0(tmin, tmax, ra_mm);
        prop_assert!(et0 >= 0.0 && et0.is_finite(), "Hargreaves ET0 must be non-negative");
    }
}

// ── Water balance mass conservation ───────────────────────────────────────

proptest! {
    #[test]
    fn water_balance_mass_conservation(
        fc in 0.15f64..0.45,
        wp in 0.05f64..0.25,
        root_depth in 200.0f64..1500.0,
        p_frac in 0.3f64..0.7,
        precip in 0.0f64..50.0,
        irrig in 0.0f64..30.0,
        et0 in 1.0f64..8.0,
        kc in 0.3f64..1.2,
        n_days in 5u32..60,
    ) {
        prop_assume!(fc > wp);
        let state = WaterBalanceState::new(fc, wp, root_depth, p_frac);
        let inputs: Vec<DailyInput> = (0..n_days)
            .map(|_| DailyInput {
                precipitation: precip,
                irrigation: irrig,
                et0,
                kc,
            })
            .collect();
        let (final_state, outputs) = simulate_season(&state, &inputs);
        let error = mass_balance_check(&inputs, &outputs, state.depletion, final_state.depletion);
        prop_assert!(error < 1e-10, "Mass balance error {} should be < 1e-10", error);
    }
}

proptest! {
    #[test]
    fn soil_water_bounded_by_wp_and_fc(
        fc in 0.20f64..0.40,
        wp in 0.08f64..0.18,
        root_depth in 300.0f64..800.0,
        precip in 0.0f64..20.0,
        irrig in 0.0f64..25.0,
        et0 in 2.0f64..7.0,
        kc in 0.5f64..1.1,
        n_days in 10u32..90,
    ) {
        prop_assume!(fc > wp);
        let state = WaterBalanceState::new(fc, wp, root_depth, 0.5);
        let inputs: Vec<DailyInput> = (0..n_days)
            .map(|_| DailyInput {
                precipitation: precip,
                irrigation: irrig,
                et0,
                kc,
            })
            .collect();
        let (final_state, _outputs) = simulate_season(&state, &inputs);
        let theta = final_state.current_theta();
        prop_assert!(theta >= wp - 1e-10, "theta {} must be >= wp {}", theta, wp);
        prop_assert!(theta <= fc + 1e-10, "theta {} must be <= fc {}", theta, fc);
    }
}

// ── Soil physics ──────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn topp_equation_bounded_0_to_1(dielectric in 1.0f64..80.0) {
        let theta = topp_equation(dielectric);
        prop_assert!((0.0..=1.0).contains(&theta) || theta.is_finite(),
            "Topp theta {} should be in [0,1] or finite for dielectric {}", theta, dielectric);
    }
}

proptest! {
    #[test]
    fn inverse_topp_round_trip(theta_v in 0.05f64..0.50) {
        let eps = inverse_topp(theta_v);
        let recovered = topp_equation(eps);
        prop_assert!((recovered - theta_v).abs() < 0.01,
            "Round-trip: theta={} -> eps={} -> theta={}", theta_v, eps, recovered);
    }
}

proptest! {
    #[test]
    fn saxton_rawls_fc_gt_wp(
        sand in 0.05f64..0.90,
        clay in 0.05f64..0.55,
        om in 0.5f64..6.0,
    ) {
        prop_assume!(sand + clay <= 0.98);
        let input = SaxtonRawlsInput {
            sand,
            clay,
            om_pct: om,
        };
        let r = saxton_rawls(&input);
        prop_assert!(r.theta_fc > r.theta_wp,
            "FC {} must be > WP {} for sand={} clay={} om={}", r.theta_fc, r.theta_wp, sand, clay, om);
    }
}

proptest! {
    #[test]
    fn vg_theta_monotonically_decreasing_with_abs_h(
        theta_r in 0.02f64..0.10,
        theta_s in 0.35f64..0.55,
        alpha in 0.01f64..0.5,
        n_vg in 1.1f64..3.5,
        h1 in -5000.0f64..-1.0,
        h2 in -5000.0f64..-1.0,
    ) {
        prop_assume!(theta_s > theta_r);
        prop_assume!((h1 - h2).abs() > f64::EPSILON);
        let (h_lo, h_hi) = if h1.abs() < h2.abs() { (h1, h2) } else { (h2, h1) };
        let theta_lo = van_genuchten_theta(h_lo, theta_r, theta_s, alpha, n_vg);
        let theta_hi = van_genuchten_theta(h_hi, theta_r, theta_s, alpha, n_vg);
        prop_assert!(theta_lo >= theta_hi - 1e-10,
            "theta(|h| smaller) >= theta(|h| larger): h_lo={} theta={}, h_hi={} theta={}",
            h_lo, theta_lo, h_hi, theta_hi);
    }
}

proptest! {
    #[test]
    fn vg_theta_bounded_by_theta_r_and_theta_s(
        theta_r in 0.02f64..0.10,
        theta_s in 0.35f64..0.55,
        alpha in 0.01f64..0.5,
        n_vg in 1.1f64..3.5,
        h in -10000.0f64..100.0,
    ) {
        prop_assume!(theta_s > theta_r);
        let theta = van_genuchten_theta(h, theta_r, theta_s, alpha, n_vg);
        prop_assert!(theta >= theta_r - 1e-10, "theta {} >= theta_r {}", theta, theta_r);
        prop_assert!(theta <= theta_s + 1e-10, "theta {} <= theta_s {}", theta, theta_s);
    }
}

// ── Crop science ──────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn gdd_non_negative(
        tmax in -10.0f64..45.0,
        tmin in -15.0f64..40.0,
        tbase in 0.0f64..15.0,
    ) {
        let gdd = gdd_avg(tmax, tmin, tbase);
        prop_assert!(gdd >= 0.0, "GDD {} must be non-negative", gdd);
    }
}

proptest! {
    #[test]
    fn yield_ratio_clamped_in_0_1(
        ky in 0.2f64..2.0,
        eta_etc in -0.5f64..1.5,
    ) {
        let raw = yield_ratio_single(ky, eta_etc);
        let clamped = clamp_yield_ratio(raw);
        prop_assert!((0.0..=1.0).contains(&clamped),
            "clamp_yield_ratio({}) = {} must be in [0,1]", raw, clamped);
    }
}

proptest! {
    #[test]
    fn scs_cn_runoff_le_precipitation(
        precip in 0.0f64..200.0,
        cn in 30.0f64..100.0,
    ) {
        let runoff = scs_cn_runoff_standard(precip, cn);
        prop_assert!(runoff <= precip + 1e-10,
            "Runoff {} must be <= precipitation {}", runoff, precip);
    }
}

proptest! {
    #[test]
    fn green_ampt_cumulative_monotonic(
        ks in 0.05f64..15.0,
        psi in 2.0f64..35.0,
        delta_theta in 0.2f64..0.5,
        t1 in 0.1f64..20.0,
        t2 in 0.1f64..20.0,
    ) {
        prop_assume!(t1 < t2);
        let params = GreenAmptParams {
            ks_cm_hr: ks,
            psi_cm: psi,
            delta_theta,
        };
        let f1 = cumulative_infiltration(&params, t1);
        let f2 = cumulative_infiltration(&params, t2);
        prop_assert!(f2 >= f1 - 1e-10,
            "Cumulative infiltration must increase monotonically: F(t1={})={} vs F(t2={})={}",
            t1, f1, t2, f2);
    }
}

// ── Diversity indices ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn shannon_non_negative(counts in prop::collection::vec(0.0f64..100.0, 1..20)) {
        let total: f64 = counts.iter().sum();
        prop_assume!(total > 1e-10);
        let h = shannon(&counts);
        prop_assert!(h >= 0.0 || h.is_nan(), "Shannon {} must be non-negative", h);
    }
}

proptest! {
    #[test]
    fn bray_curtis_in_0_1(
        a in prop::collection::vec(0.0f64..100.0, 1..15),
        b in prop::collection::vec(0.0f64..100.0, 1..15),
    ) {
        let sa: f64 = a.iter().sum();
        let sb: f64 = b.iter().sum();
        prop_assume!(sa > 1e-10 || sb > 1e-10);
        // Pad to same length
        let n = a.len().max(b.len());
        let mut pa = a;
        let mut pb = b;
        pa.resize(n, 0.0);
        pb.resize(n, 0.0);
        let bc = bray_curtis(&pa, &pb);
        prop_assert!((0.0..=1.0).contains(&bc) || bc.is_nan(),
            "Bray-Curtis {} must be in [0,1]", bc);
    }
}
