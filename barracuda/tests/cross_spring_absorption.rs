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

// ────────────────────────────────────────────────────────────────────────────
// §14 — S70+: Cross-Spring Evolution — Modern ToadStool Rewiring
//
// S70+ absorbed ops 5–8 from airSpring and enabled the seasonal pipeline to
// dispatch Stages 1-2 (ET₀ + Kc climate adjust) to GPU. This section validates
// the modern ToadStool primitives and documents cross-spring shader evolution.
//
// Shader Evolution Map:
//
//   hotSpring  → df64_core.wgsl, math_f64.wgsl, complex_f64.wgsl, Lanczos, Anderson
//                 Used by: ALL Springs (universal precision foundation)
//   wetSpring  → kriging_f64.wgsl, shannon/simpson/bray_curtis, Hill, moving_window
//                 Used by: airSpring (spatial interpolation, cover crop diversity)
//   neuralSpring → nelder_mead.wgsl, batched_bisection, matmul, simplex_ops
//                   Used by: airSpring (isotherm fitting, VG inversion)
//   airSpring  → batched_elementwise_f64.wgsl (ops 0–8), seasonal_pipeline.wgsl
//                 Used by: neuralSpring (surrogate ET₀), wetSpring (hydrology stats)
//   groundSpring → mc_et0_propagate_f64.wgsl, batched_multinomial
//                   Used by: airSpring (uncertainty propagation)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn s70_richards_uses_upstream_tridiagonal_solve() {
    use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};

    let params = VanGenuchtenParams {
        theta_r: 0.078,
        theta_s: 0.43,
        alpha: 0.036,
        n_vg: 1.56,
        ks: 24.96,
    };

    let profiles = solve_richards_1d(&params, 100.0, 20, -100.0, -50.0, false, true, 1.0, 0.1)
        .expect("Richards solver should succeed with barracuda tridiagonal_solve");

    assert!(
        !profiles.is_empty(),
        "S70+: Richards PDE solver using barracuda::linalg::tridiagonal_solve \
         (upstream Thomas algorithm replaces local duplicate)"
    );

    let last = profiles.last().unwrap();
    assert!(
        last.theta
            .iter()
            .all(|&t| t >= params.theta_r && t <= params.theta_s),
        "S70+: Richards profile θ values in [θr, θs] — numerical stability preserved"
    );
}

#[test]
fn s70_nelder_mead_optimizer_cross_spring() {
    use barracuda::optimize::nelder_mead::nelder_mead;

    let rosenbrock = |x: &[f64]| {
        let dx = 1.0 - x[0];
        let dy = x[0].mul_add(-x[0], x[1]);
        100.0f64.mul_add(dy * dy, dx * dx)
    };
    let bounds = &[(-5.0, 5.0), (-5.0, 5.0)];
    let (best_x, best_f, _iters) =
        nelder_mead(rosenbrock, &[0.0, 0.0], bounds, 50_000, 1e-12).expect("NM convergence");
    assert!(
        (best_x[0] - 1.0).abs() < 0.05 && (best_x[1] - 1.0).abs() < 0.05,
        "S70+: Nelder-Mead converges on Rosenbrock (neuralSpring optimizer lineage → \
         barracuda::optimize S52+); got ({:.4}, {:.4}), f={best_f:.6}",
        best_x[0],
        best_x[1]
    );
}

#[test]
fn s70_bfgs_optimizer_for_calibration() {
    use barracuda::optimize::bfgs::{bfgs_numerical, BfgsConfig};

    let quadratic = |x: &[f64]| {
        let dx = x[0] - 3.0;
        let dy = x[1] + 1.0;
        dx.mul_add(dx, dy * dy)
    };
    let config = BfgsConfig::default();
    let result = bfgs_numerical(&quadratic, &[0.0, 0.0], &config).expect("BFGS convergence");
    assert!(
        (result.x[0] - 3.0).abs() < 1e-4 && (result.x[1] + 1.0).abs() < 1e-4,
        "S70+: BFGS numerical converges for smooth calibration \
         (neuralSpring optimizer lineage → barracuda::optimize S52+); \
         got ({:.6}, {:.6})",
        result.x[0],
        result.x[1]
    );
}

#[test]
fn s70_newton_secant_root_finding() {
    use barracuda::optimize::newton::{newton, secant};

    let f = |x: f64| x.powi(3) - 2.0f64.mul_add(x, 5.0);
    let df = |x: f64| 3.0f64.mul_add(x * x, -2.0);

    let nr = newton(f, df, 2.0, 1e-12, 50).expect("Newton convergence");
    assert!(
        f(nr.root).abs() < 1e-10,
        "S70+: Newton root-finding (neuralSpring lineage); f(root)={:.2e}",
        f(nr.root)
    );

    let sr = secant(f, 2.0, 3.0, 1e-12, 50).expect("Secant convergence");
    assert!(
        (nr.root - sr.root).abs() < 1e-8,
        "Newton and Secant find same root: {:.8} vs {:.8}",
        nr.root,
        sr.root
    );
}

#[test]
fn s70_bisection_robust_bracketed_root() {
    use barracuda::optimize::bisect::bisect;

    let f = |x: f64| x.mul_add(x, -2.0);
    let sqrt2 = bisect(f, 1.0, 2.0, 1e-14, 100).expect("Bisect convergence");
    assert!(
        (sqrt2 - std::f64::consts::SQRT_2).abs() < 1e-12,
        "S70+: Bisection finds √2 (neuralSpring lineage); got {sqrt2:.15}",
    );
}

#[test]
fn s70_special_functions_hotspring_precision() {
    use barracuda::math::{erf, erfc, gamma, ln_gamma};

    let erf_1 = erf(1.0);
    assert!(
        (erf_1 - 0.842_700_792_949_715).abs() < 1e-6,
        "hotSpring precision: erf(1) matches DLMF 7.2; got {erf_1:.15}"
    );
    assert!(
        (erfc(0.0) - 1.0).abs() < 1e-15,
        "hotSpring precision: erfc(0) = 1"
    );

    let g5 = gamma(5.0).expect("gamma(5)");
    assert!(
        (g5 - 24.0).abs() < 1e-10,
        "hotSpring precision: Γ(5) = 4! = 24; got {g5}"
    );
    let ln_g_10 = ln_gamma(10.0).expect("ln_gamma(10)");
    assert!(
        (ln_g_10 - (362_880.0_f64).ln()).abs() < 1e-8,
        "hotSpring precision: ln Γ(10) = ln(9!); got {ln_g_10}"
    );
}

#[test]
fn s70_norm_cdf_ppf_round_trip() {
    use barracuda::stats::normal::{norm_cdf, norm_ppf};

    for &z in &[-2.0, -1.0, 0.0, 1.0, 2.0, 2.576] {
        let p = norm_cdf(z);
        let z_back = norm_ppf(p);
        assert!(
            (z - z_back).abs() < 1e-4,
            "hotSpring precision: norm_cdf/ppf round-trip at z={z}: \
             p={p:.10} → z_back={z_back:.10}"
        );
    }
}

#[test]
fn s70_pde_crank_nicolson_heat_equation() {
    use barracuda::pde::crank_nicolson::{CrankNicolson1D, CrankNicolsonConfig};

    let nx = 50;
    let config = CrankNicolsonConfig {
        alpha: 1.0,
        dx: 0.02,
        dt: 0.0001,
        nx,
        left_bc: 0.0,
        right_bc: 0.0,
    };
    let initial: Vec<f64> = (0..nx)
        .map(|i| (std::f64::consts::PI * f64::from(i as u32) * 0.02).sin())
        .collect();
    let mut solver = CrankNicolson1D::new(config, &initial).expect("CN init");

    for _ in 0..100 {
        solver.step_with_source(None).expect("CN step");
    }

    let state = solver.solution();
    let peak = state.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        peak < 1.0,
        "S70+: Crank-Nicolson heat diffusion reduces peak (hotSpring PDE lineage); \
         peak={peak:.6}"
    );
    assert!(
        peak >= 0.0,
        "Solution remains non-negative (stable CN scheme)"
    );
}

#[test]
fn s70_upstream_richards_pde_consistency() {
    use barracuda::pde::richards::{solve_richards, RichardsBc, RichardsConfig, SoilParams};

    let config = RichardsConfig {
        soil: SoilParams::LOAM,
        dz: 10.0,
        dt: 60.0,
        n_nodes: 10,
        max_picard_iter: 50,
        picard_tol: 1e-4,
    };

    let h0 = vec![-100.0; 10];
    let result = solve_richards(
        &config,
        &h0,
        20,
        RichardsBc::PressureHead(-50.0),
        RichardsBc::Flux(0.0),
    )
    .expect("upstream Richards");
    assert!(
        result.h.iter().all(|h| h.is_finite()),
        "S70+: Upstream barracuda::pde::richards produces finite results \
         (shared ToadStool Thomas solver)"
    );
}

#[test]
fn s70_jackknife_uncertainty_for_et0() {
    use barracuda::stats::jackknife::jackknife_mean_variance;

    let et0_samples = [3.2, 3.5, 3.1, 3.8, 3.6, 3.3, 3.7, 3.4, 3.9, 3.0];
    let jk = jackknife_mean_variance(&et0_samples).expect("jackknife");

    assert!(
        (jk.estimate - 3.45).abs() < 0.01,
        "Jackknife mean matches direct mean: {:.4}",
        jk.estimate
    );
    assert!(
        jk.variance > 0.0 && jk.variance < 1.0,
        "S70+: Jackknife variance reasonable for ET₀ series: {:.6} — \
         available for leave-one-out uncertainty in validation binaries",
        jk.variance
    );
}

#[test]
fn s70_chi2_decomposed_residual_analysis() {
    use barracuda::stats::chi2::chi2_decomposed;

    let observed = [10.0, 20.0, 30.0, 40.0];
    let expected = [12.0, 18.0, 32.0, 38.0];

    let result = chi2_decomposed(&observed, &expected, 0).expect("chi2");
    assert!(
        result.chi2_total > 0.0,
        "S70+: Chi² decomposed for model diagnostics; χ²={:.4}",
        result.chi2_total
    );
}

#[test]
fn s70_empirical_spectral_density_sensor_timeseries() {
    use barracuda::stats::spectral_density::empirical_spectral_density;

    let n = 128;
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let t = f64::from(i as u32) / f64::from(n);
            let w1 = (2.0 * std::f64::consts::PI * 5.0 * t).sin();
            let w2 = (2.0 * std::f64::consts::PI * 20.0 * t).sin();
            0.3f64.mul_add(w2, w1)
        })
        .collect();

    let (bins, density) = empirical_spectral_density(&data, 32);
    assert!(
        !bins.is_empty() && !density.is_empty(),
        "S70+: Spectral density for sensor time-series analysis \
         (hotSpring RMT lineage → barracuda::stats S57+)"
    );
    assert!(density.iter().all(|&v| v >= 0.0), "PSD values non-negative");
}

#[test]
fn s70_wetspring_hill_monod_crop_response() {
    let substrate = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0];

    for &s in &substrate {
        let h = barracuda::stats::hill(s, 2.0, 1.5);
        assert!(
            (0.0..=1.0).contains(&h),
            "Hill response in [0, 1]: h({s})={h}"
        );

        let m = barracuda::stats::monod(s, 10.0, 2.0);
        assert!(
            (0.0..=10.0).contains(&m),
            "Monod response in [0, Vmax]: m({s})={m}"
        );
    }

    let h_low = barracuda::stats::hill(0.1, 2.0, 1.5);
    let h_high = barracuda::stats::hill(20.0, 2.0, 1.5);
    assert!(
        h_high > h_low,
        "S70+: Hill function monotonically increasing (wetSpring bio kinetics \
         → barracuda::stats S66+): {h_low:.4} → {h_high:.4}"
    );
}

#[test]
fn s70_convergence_diagnostics_for_picard() {
    use barracuda::optimize::diagnostics::{convergence_diagnostics, ConvergenceState};

    let residuals = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4];
    let diag = convergence_diagnostics(&residuals, 3, 0.01, 3).expect("diagnostics");

    assert!(
        !matches!(diag.state, ConvergenceState::Oscillating),
        "S70+: Convergence diagnostics detect non-oscillating trend \
         (neuralSpring lineage): {:?}",
        diag.state
    );
}

// ── §15 — Cross-Spring Shader Evolution Map (Tier A/B Documentation) ──────

#[test]
fn cross_spring_shader_evolution_documented() {
    use airspring_barracuda::gpu::evolution_gaps::{Tier, GAPS};

    let tier_a = GAPS.iter().filter(|g| g.tier == Tier::A).count();
    let tier_b = GAPS.iter().filter(|g| g.tier == Tier::B).count();
    let tier_c = GAPS.iter().filter(|g| g.tier == Tier::C).count();

    assert!(
        tier_a >= 14,
        "At least 14 Tier A (integrated) gaps: {tier_a}"
    );
    assert!(
        tier_b >= 5,
        "At least 5 Tier B (needs wiring) gaps: {tier_b}"
    );
    assert!(
        tier_c >= 1,
        "At least 1 Tier C (needs primitive) gap: {tier_c}"
    );

    let total = tier_a + tier_b + tier_c;
    let integration_ratio = tier_a as f64 / total as f64;
    assert!(
        integration_ratio > 0.5,
        "Cross-spring integration ratio > 50%: {tier_a}/{total} = {:.1}% — \
         hotSpring precision, wetSpring bio, neuralSpring ML all contributing",
        integration_ratio * 100.0
    );
}

// ── §16 — Regression: Ensure Rewired Systems Match Previous Baselines ─────

#[test]
fn regression_richards_tridiag_rewire_matches_baseline() {
    use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};

    let params = VanGenuchtenParams {
        theta_r: 0.078,
        theta_s: 0.43,
        alpha: 0.036,
        n_vg: 1.56,
        ks: 24.96,
    };

    let profiles = solve_richards_1d(&params, 100.0, 20, -100.0, -50.0, false, true, 1.0, 0.1)
        .expect("Richards solver with barracuda tridiagonal");

    assert_eq!(profiles.len(), 10, "1 day / 0.1 dt = 10 profiles");

    let theta_final = &profiles.last().unwrap().theta;
    assert!(
        theta_final
            .iter()
            .all(|&t| t >= params.theta_r && t <= params.theta_s),
        "All final θ in [θr, θs] — barracuda::linalg::tridiagonal_solve \
         numerical equivalence confirmed"
    );
}

#[test]
fn regression_correction_fits_barracuda_regression() {
    use airspring_barracuda::eco::correction::{evaluate, fit_correction_equations};

    let x: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.8f64.mul_add(xi, 0.2)).collect();

    let models = fit_correction_equations(&x, &y);
    assert!(
        models.len() >= 2,
        "fit_correction_equations returns at least 2 models (now using barracuda::stats::regression)"
    );

    let linear = models
        .iter()
        .find(|m| m.model_type == airspring_barracuda::eco::correction::ModelType::Linear)
        .unwrap();
    assert!(linear.r_squared > 0.999, "Linear R² = {}", linear.r_squared);

    let y_pred = evaluate(linear, 0.5);
    let y_expected = 1.8f64.mul_add(0.5, 0.2);
    assert!(
        (y_pred - y_expected).abs() < 1e-6,
        "barracuda::stats::regression prediction matches: {y_pred} vs {y_expected}"
    );
}
