// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring absorption validation — proves S64/S66/S68 upstream absorptions
//! produce identical results to local implementations and remain stable across
//! the universal precision refactor.
//!
//! S64: Stats absorption (metrics, diversity, bootstrap from Springs)
//! S66: Cross-spring absorption (regression, hydrology, `moving_window_f64`)
//! S68: Universal precision (334+ shaders evolved to f64-canonical)

// ── §7 — ToadStool S64: Cross-Spring Stats Absorption ─────────────────

#[test]
fn s64_stats_rmse_delegates_to_upstream() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sim = [1.1, 2.1, 2.9, 4.2, 4.8];
    let local = airspring_barracuda::testutil::rmse(&obs, &sim);
    let upstream = barracuda::stats::rmse(&obs, &sim);
    assert!(
        (local - upstream).abs() < f64::EPSILON,
        "airSpring testutil::rmse should delegate to upstream barracuda::stats::rmse; \
         local={local} upstream={upstream} — stats absorbed in S64"
    );
}

#[test]
fn s64_stats_mbe_delegates_to_upstream() {
    let obs = [5.0, 6.0, 7.0];
    let sim = [4.0, 5.5, 7.5];
    let local = airspring_barracuda::testutil::mbe(&obs, &sim);
    let upstream = barracuda::stats::mbe(&obs, &sim);
    assert!(
        (local - upstream).abs() < f64::EPSILON,
        "airSpring testutil::mbe should delegate to upstream; \
         local={local} upstream={upstream} — stats absorbed in S64"
    );
}

#[test]
fn s64_stats_new_reexports_from_upstream() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let m = airspring_barracuda::testutil::mean(&data);
    assert!(
        (m - 3.0).abs() < 1e-12,
        "mean re-export from barracuda::stats::mean (S64 absorption)"
    );

    let d = airspring_barracuda::testutil::dot(&data, &data);
    assert!(
        (d - 55.0).abs() < 1e-12,
        "dot re-export from barracuda::stats::dot (S64 absorption)"
    );

    let l2 = airspring_barracuda::testutil::l2_norm(&data);
    assert!(
        (l2 - 55.0_f64.sqrt()).abs() < 1e-12,
        "l2_norm re-export from barracuda::stats::l2_norm (S64 absorption)"
    );
}

// ── §8 — wetSpring S64: Diversity Metrics for Agroecology ─────────────

#[test]
fn s64_wetspring_diversity_shannon_for_cover_crops() {
    use airspring_barracuda::eco::diversity;

    let cover_mix = [120.0, 85.0, 45.0, 30.0, 20.0];
    let monoculture = [300.0, 0.0, 0.0, 0.0, 0.0];

    let h_mix = diversity::shannon(&cover_mix);
    let h_mono = diversity::shannon(&monoculture);

    assert!(
        h_mix > 1.0 && h_mono < 0.01,
        "wetSpring diversity::shannon wired for agroecology: \
         5-species cover crop mix H'={h_mix} > monoculture H'={h_mono}"
    );
}

#[test]
fn s64_wetspring_diversity_bray_curtis_field_comparison() {
    use airspring_barracuda::eco::diversity;

    let field_a = [120.0, 85.0, 45.0, 30.0, 20.0];
    let field_b = [90.0, 100.0, 55.0, 25.0, 30.0];
    let field_c = [0.0, 0.0, 0.0, 0.0, 300.0];

    let bc_similar = diversity::bray_curtis(&field_a, &field_b);
    let bc_different = diversity::bray_curtis(&field_a, &field_c);

    assert!(
        bc_similar < bc_different,
        "wetSpring Bray-Curtis: similar fields BC={bc_similar} < different fields BC={bc_different}"
    );
}

#[test]
fn s64_wetspring_alpha_diversity_comprehensive() {
    use airspring_barracuda::eco::diversity;

    let counts = [120.0, 85.0, 45.0, 30.0, 20.0];
    let ad = diversity::alpha_diversity(&counts);

    assert!((ad.observed - 5.0).abs() < 1e-10, "observed species = 5");
    assert!(ad.shannon > 1.0, "Shannon H' > 1.0 for 5-species mix");
    assert!(ad.simpson > 0.5, "Simpson D > 0.5 for multi-species");
    assert!(ad.chao1 >= 5.0, "Chao1 >= observed");
    assert!(
        (0.0..=1.0).contains(&ad.evenness),
        "Pielou J' in [0,1]: wetSpring bio diversity absorbed in S64"
    );
}

// ── §9 — groundSpring S64: MC ET₀ Uncertainty Propagation ────────────

#[test]
fn s64_groundspring_mc_et0_uncertainty_bands() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
    use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};

    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    };

    let result = mc_et0_cpu(&input, &Et0Uncertainties::default(), 2000, 42);

    assert!(
        result.et0_std > 0.05 && result.et0_std < 2.0,
        "MC ET₀ should show measurable uncertainty: σ={} — \
         groundSpring mc_et0_propagate_f64.wgsl absorbed in S64",
        result.et0_std
    );
    assert!(
        result.et0_p05 < result.et0_central && result.et0_p95 > result.et0_central,
        "90% CI [{}, {}] should bracket central ET₀={}",
        result.et0_p05,
        result.et0_p95,
        result.et0_central
    );
}

#[test]
fn s64_groundspring_mc_et0_deterministic_seed() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
    use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};

    let input = DailyEt0Input {
        tmin: 15.0,
        tmax: 28.0,
        tmean: None,
        solar_radiation: 18.5,
        wind_speed_2m: 1.5,
        actual_vapour_pressure: 1.2,
        elevation_m: 200.0,
        latitude_deg: 35.0,
        day_of_year: 200,
    };

    let r1 = mc_et0_cpu(&input, &Et0Uncertainties::default(), 500, 99);
    let r2 = mc_et0_cpu(&input, &Et0Uncertainties::default(), 500, 99);
    assert!(
        (r1.et0_mean - r2.et0_mean).abs() < f64::EPSILON,
        "MC ET₀ must be deterministic for same seed — \
         mirrors GPU kernel's xoshiro128** reproducibility"
    );
}

// ── §11 — ToadStool S66: Cross-Spring Absorption Wave ─────────────────
//
// S66 absorbed all pending airSpring metalForge modules upstream:
// regression (R-S66-001), hydrology (R-S66-002), moving_window_f64 (R-S66-003),
// spearman re-export (R-S66-005), 8 named SoilParams (R-S66-006),
// mae (R-S66-036), shannon_from_frequencies (R-S66-037).

#[test]
fn s66_regression_absorbed_upstream() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.1, 3.9, 6.1, 7.9, 10.1];

    let upstream = barracuda::stats::regression::fit_linear(&x, &y)
        .expect("barracuda::stats::regression::fit_linear should succeed (R-S66-001)");
    let local = airspring_barracuda::eco::correction::fit_linear(&x, &y)
        .expect("eco::correction::fit_linear should succeed");

    assert!(
        (upstream.r_squared - local.r_squared).abs() < 1e-6,
        "S66 upstream regression R²={} should match local R²={} — \
         airSpring metalForge regression absorbed upstream (R-S66-001)",
        upstream.r_squared,
        local.r_squared
    );
    assert!(
        upstream.r_squared > 0.99,
        "Near-perfect linear data should yield R²>0.99: got {}",
        upstream.r_squared
    );
}

#[test]
fn s66_hydrology_hargreaves_absorbed_upstream() {
    let tmin = 19.1_f64;
    let tmax = 32.6;
    let ra_mm = 40.55;

    let upstream = barracuda::stats::hydrology::hargreaves_et0(ra_mm, tmax, tmin)
        .expect("barracuda::stats::hydrology::hargreaves_et0 should succeed (R-S66-002)");
    let local = airspring_barracuda::eco::evapotranspiration::hargreaves_et0(tmin, tmax, ra_mm);

    assert!(
        (upstream - local).abs() < 0.01,
        "S66 upstream Hargreaves ET₀={upstream:.4} should match local={local:.4} — \
         airSpring metalForge hydrology absorbed upstream (R-S66-002), \
         param order differs: upstream(ra,tmax,tmin) vs local(tmin,tmax,ra)"
    );
    assert!(
        upstream > 0.0 && upstream < 15.0,
        "Hargreaves ET₀ should be physically reasonable: got {upstream}"
    );
}

#[test]
fn s66_moving_window_f64_absorbed_upstream() {
    let data: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
    let result = barracuda::stats::moving_window_f64::moving_window_stats_f64(&data, 10)
        .expect("barracuda::stats::moving_window_f64 should succeed (R-S66-003)");

    assert_eq!(
        result.mean.len(),
        91,
        "100 values with window=10 → 91 output windows"
    );
    assert!(
        result.variance.iter().all(|&v| v >= 0.0),
        "All variances must be non-negative — \
         airSpring metalForge moving_window_f64 absorbed upstream (R-S66-003)"
    );
}

#[test]
fn s66_spearman_reexport_available() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];

    let rho = barracuda::stats::spearman_correlation(&x, &y)
        .expect("spearman_correlation should be re-exported from stats (R-S66-005)");
    assert!(
        (rho - 1.0).abs() < 1e-10,
        "Perfect monotonic → Spearman ρ=1.0; got {rho} — \
         R-S66-005 added re-export from stats::correlation"
    );
}

#[test]
fn s66_soil_params_named_constants() {
    use barracuda::pde::richards::SoilParams;

    let sandy_loam = SoilParams::SANDY_LOAM;
    assert!(
        sandy_loam.theta_s > sandy_loam.theta_r,
        "θs > θr for sandy loam (Carsel & Parrish 1988, R-S66-006)"
    );
    assert!(
        sandy_loam.alpha > 0.0 && sandy_loam.n > 1.0,
        "VG parameters physical: α={}, n={}",
        sandy_loam.alpha,
        sandy_loam.n
    );

    let clay = SoilParams::CLAY;
    assert!(
        clay.k_sat < sandy_loam.k_sat,
        "Clay K_sat={} < sandy loam K_sat={} (R-S66-006)",
        clay.k_sat,
        sandy_loam.k_sat
    );

    let theta_at_saturation = sandy_loam.theta(0.0);
    assert!(
        (theta_at_saturation - sandy_loam.theta_s).abs() < 1e-6,
        "θ(h=0) should equal θs for sandy loam: got {theta_at_saturation}"
    );
}

#[test]
fn s66_metrics_mae_available() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sim = [1.5, 2.5, 2.5, 4.5, 4.5];

    let mae = barracuda::stats::mae(&obs, &sim);
    assert!(
        (mae - 0.5).abs() < 1e-12,
        "MAE of ±0.5 deviations should be 0.5; got {mae} — R-S66-036"
    );
}

#[test]
fn s66_diversity_shannon_from_frequencies() {
    let freqs = [0.5, 0.3, 0.2];
    let h = barracuda::stats::diversity::shannon_from_frequencies(&freqs);
    assert!(
        h > 0.0 && h < 2.0,
        "Shannon from frequencies should be positive and bounded; got {h} — R-S66-037"
    );

    let uniform = [0.25, 0.25, 0.25, 0.25];
    let h_max = barracuda::stats::diversity::shannon_from_frequencies(&uniform);
    assert!(
        h_max > h,
        "Uniform distribution should have higher entropy than skewed: \
         {h_max} > {h} — R-S66-037"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// §13 — S68 universal precision validation
//
// ToadStool S68 evolved ALL WGSL shaders to f64 canonical and introduced
// `downcast_f64_to_f32()` for backward compatibility. The precision chain:
//
//   hotSpring → df64_core.wgsl + math_f64.wgsl (S54: nuclear physics needs f64)
//   wetSpring → kriging_f64.wgsl (S28+: spatial stats needs f64 for conditioning)
//   neuralSpring → simplex_ops_f64.wgsl (S62: NM convergence needs f64)
//   airSpring → batched_elementwise_f64.wgsl (S40: FAO-56 ET₀ needs f64)
//   groundSpring → mc_et0_propagate_f64.wgsl (S64: Monte Carlo CI needs f64)
//
// S67 codified: "math is universal, precision is silicon"
// S68 executed: 334+ shaders evolved, ZERO f32-only remain
//
// These tests verify that airSpring's upstream barracuda functions still
// produce identical results after S68's precision refactor.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn s68_validation_harness_tracing_migration() {
    use barracuda::validation::ValidationHarness;

    let mut v = ValidationHarness::new("S68 tracing test");
    v.check_abs(
        "precision preserved",
        std::f64::consts::PI,
        std::f64::consts::PI,
        1e-15,
    );
    v.check_bool("universal precision", true);
    assert_eq!(v.passed_count(), 2, "S68 ValidationHarness API unchanged");
    assert_eq!(v.total_count(), 2);
}

#[test]
fn s68_regression_f64_precision_stable() {
    use barracuda::stats::regression::{fit_linear, fit_quadratic};

    let x: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let y_lin: Vec<f64> = x.iter().map(|&xi| 2.5f64.mul_add(xi, 1.3)).collect();

    let lin = fit_linear(&x, &y_lin).expect("linear fit");
    assert!(
        (lin.params[0] - 2.5).abs() < 1e-10,
        "S68: linear slope precision stable (airSpring→S66 absorption); got {}",
        lin.params[0]
    );
    assert!(
        (lin.params[1] - 1.3).abs() < 1e-10,
        "S68: linear intercept precision stable; got {}",
        lin.params[1]
    );

    let y_quad: Vec<f64> = x
        .iter()
        .map(|&xi| (0.5 * xi).mul_add(xi, 3.0 - xi))
        .collect();
    let quad = fit_quadratic(&x, &y_quad).expect("quadratic fit");
    assert!(
        (quad.params[0] - 0.5).abs() < 1e-8,
        "S68: quadratic fit precision stable; got a={}",
        quad.params[0]
    );
}

#[test]
fn s68_hydrology_hargreaves_stable() {
    use barracuda::stats::hydrology::hargreaves_et0;

    let et0 = hargreaves_et0(42.0, 25.0, 15.0).expect("valid hargreaves");
    assert!(
        et0 > 5.0 && et0 < 20.0,
        "S68: Hargreaves ET₀ in expected range; got {et0}"
    );

    let et0_b = hargreaves_et0(42.0, 25.0, 15.0).expect("valid");
    assert!(
        (et0 - et0_b).abs() < f64::EPSILON,
        "S68: Hargreaves deterministic after precision refactor"
    );
}

#[test]
fn s68_diversity_cross_spring_lineage() {
    use barracuda::stats::diversity::{shannon, simpson};

    let counts = [10.0_f64, 20.0, 30.0, 40.0];
    let h = shannon(&counts);
    let d = simpson(&counts);

    assert!(
        (1.2..1.4).contains(&h),
        "S68: Shannon index stable (wetSpring S64 lineage); got {h}"
    );
    assert!(
        (0.7..0.8).contains(&d),
        "S68: Simpson index stable; got {d}"
    );
}

#[test]
fn s68_moving_window_precision() {
    use barracuda::stats::moving_window_f64::moving_window_stats_f64;

    let data: Vec<f64> = (0..50).map(|i| f64::from(i).mul_add(0.5, 1.0)).collect();
    let result = moving_window_stats_f64(&data, 5).expect("valid window");

    let expected_len = data.len() - 5 + 1;
    assert_eq!(result.mean.len(), expected_len, "output length");
    for i in 0..result.mean.len() {
        assert!(
            result.mean[i].is_finite(),
            "S68: moving window mean is finite"
        );
        assert!(
            result.variance[i] >= 0.0,
            "S68: moving window variance non-negative"
        );
    }
}

#[test]
fn s68_brent_optimizer_cross_spring() {
    use barracuda::optimize::brent::brent_minimize;

    let (x_min, _f_min, _iters) = brent_minimize(
        |x| (x - 2.5).powi(2),
        0.0,  // a
        1.25, // b interior
        5.0,  // c
        1e-10,
        100,
    )
    .expect("convergence");
    assert!(
        (x_min - 2.5).abs() < 1e-8,
        "S68: Brent optimizer precision stable (neuralSpring lineage); got {x_min}"
    );
}

#[test]
fn s68_spearman_correlation_stable() {
    use barracuda::stats::correlation::spearman_correlation;

    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

    let rho = spearman_correlation(&x, &y).expect("spearman");
    assert!(
        (rho - 1.0).abs() < 1e-10,
        "S68: Spearman rho=1.0 for monotonic data (cross-spring metric); got {rho}"
    );
}

#[test]
fn s68_bootstrap_ci_stable() {
    use barracuda::stats::bootstrap::bootstrap_mean;

    let data: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let ci = bootstrap_mean(&data, 1000, 0.95, 42).expect("bootstrap");

    assert!(
        ci.lower < ci.upper,
        "S68: Bootstrap CI lower < upper (hotSpring+groundSpring lineage)"
    );
    assert!(
        ci.lower > 0.0 && ci.upper < 10.0,
        "S68: Bootstrap CI in reasonable range; got [{}, {}]",
        ci.lower,
        ci.upper
    );
}
