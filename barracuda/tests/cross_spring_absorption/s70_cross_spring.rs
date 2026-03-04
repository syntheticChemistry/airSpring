// SPDX-License-Identifier: AGPL-3.0-or-later
//! S70+: Cross-Spring Evolution — Modern ToadStool Rewiring
//!
//! Validates modern ToadStool primitives and documents cross-spring shader
//! evolution (Richards PDE, optimizers, special functions, Hill/Monod).

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
