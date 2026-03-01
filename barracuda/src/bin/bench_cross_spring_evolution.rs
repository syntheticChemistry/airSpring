// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]

//! Cross-Spring Evolution Benchmark — Modern `ToadStool` + `BarraCUDA` Validation
//!
//! Validates the complete cross-spring shader evolution by exercising primitives
//! from each contributing Spring and documenting when and where each capability
//! evolved into the shared ecosystem.
//!
//! # Cross-Spring Shader Provenance
//!
//! | Spring | Domain | Key Contributions |
//! |--------|--------|-------------------|
//! | hotSpring | Nuclear/precision physics | `df64`, `math_f64`, Lanczos, Anderson, erf/gamma |
//! | wetSpring | Bio/environmental | Shannon/Simpson/Bray-Curtis, kriging, Hill, `moving_window` |
//! | neuralSpring | ML/optimization | Nelder-Mead, BFGS, `ValidationHarness`, batch IPR |
//! | airSpring | Precision agriculture | regression, hydrology, Richards PDE, ET₀ ops 0-8 |
//! | groundSpring | Uncertainty/stats | MC ET₀ propagation, `batched_multinomial`, `rawr_mean` |
//!
//! # Evolution Timeline
//!
//! - S40: airSpring Richards PDE → `barracuda::pde::richards`
//! - S52: neuralSpring Nelder-Mead, BFGS, bisect, chi² → `barracuda::optimize`
//! - S54: hotSpring `df64_core`, `math_f64` → universal precision foundation
//! - S58: hotSpring ridge regression, `Fp64Strategy` → `barracuda::linalg`
//! - S62: hotSpring `CrankNicolson1D` (f64 + GPU) → `barracuda::pde`
//! - S64: wetSpring Shannon/Simpson/BC, groundSpring MC ET₀ → `barracuda::stats`
//! - S66: airSpring regression, hydrology, `moving_window` → `barracuda::stats`
//! - S68: Universal precision (334+ shaders → f64 canonical) → ALL Springs
//! - S70+: airSpring ops 5-8, `seasonal_pipeline` → `barracuda::ops`

use std::time::Instant;

use airspring_barracuda::eco::correction;
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};
use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Cross-Spring Evolution Benchmark (v0.5.8)");
    println!("  ToadStool S70+++ — Modern Rewiring Validation");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut v = ValidationHarness::new("Cross-Spring Evolution");

    bench_hotspring_precision(&mut v);
    bench_wetspring_bio(&mut v);
    bench_neuralspring_optimizers(&mut v);
    bench_airspring_rewired(&mut v);
    bench_groundspring_uncertainty(&mut v);
    bench_tridiagonal_rewire(&mut v);

    println!();
    v.finish();
}

fn bench_hotspring_precision(v: &mut ValidationHarness) {
    println!("\n── hotSpring Precision Lineage ──────────────────────────────");

    let t0 = Instant::now();
    let erf_1 = barracuda::math::erf(1.0);
    v.check_abs(
        "erf(1) [hotSpring → math_f64 S54]",
        erf_1,
        0.842_700_792_949_715,
        1e-6,
    );

    let gamma_5 = barracuda::math::gamma(5.0).expect("gamma(5)");
    v.check_abs("Γ(5) = 4! [hotSpring → special S54]", gamma_5, 24.0, 1e-10);

    let norm_cdf_0 = barracuda::stats::normal::norm_cdf(0.0);
    v.check_abs(
        "Φ(0) = 0.5 [hotSpring → norm_cdf S60]",
        norm_cdf_0,
        0.5,
        1e-14,
    );

    for &z in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
        let p = barracuda::stats::normal::norm_cdf(z);
        let z_back = barracuda::stats::normal::norm_ppf(p);
        v.check_abs(
            &format!("norm_cdf/ppf round-trip z={z:.0}"),
            z_back,
            z,
            1e-4,
        );
    }
    println!("  hotSpring precision: {:.1?}", t0.elapsed());
}

fn bench_wetspring_bio(v: &mut ValidationHarness) {
    println!("\n── wetSpring Bio Lineage ────────────────────────────────────");

    let t0 = Instant::now();
    let counts = [120.0, 85.0, 45.0, 30.0, 20.0];
    let h = barracuda::stats::diversity::shannon(&counts);
    v.check_lower("Shannon H' > 1.0 [wetSpring → S64 absorption]", h, 1.0);

    let d = barracuda::stats::diversity::simpson(&counts);
    v.check_lower("Simpson D > 0.5 [wetSpring → S64]", d, 0.5);

    let field_a = [120.0, 85.0, 45.0, 30.0, 20.0];
    let field_b = [90.0, 100.0, 55.0, 25.0, 30.0];
    let bc = barracuda::stats::diversity::bray_curtis(&field_a, &field_b);
    v.check_lower("Bray-Curtis dissimilarity > 0 [wetSpring → S64]", bc, 0.0);
    v.check_upper("Bray-Curtis dissimilarity < 1", bc, 1.0);

    for &s in &[0.1, 1.0, 5.0, 20.0_f64] {
        let h_val = barracuda::stats::hill(s, 2.0, 1.5);
        v.check_lower(&format!("Hill({s}) > 0 [wetSpring → S66]"), h_val, 0.0);
        v.check_upper(&format!("Hill({s}) ≤ 1"), h_val, 1.0);
    }

    let data: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
    let mw = barracuda::stats::moving_window_f64::moving_window_stats_f64(&data, 10)
        .expect("moving window");
    v.check_abs(
        "moving_window 100pts/w=10 → 91 [wetSpring → S66]",
        mw.mean.len() as f64,
        91.0,
        0.5,
    );
    println!("  wetSpring bio: {:.1?}", t0.elapsed());
}

fn bench_neuralspring_optimizers(v: &mut ValidationHarness) {
    println!("\n── neuralSpring Optimizer Lineage ───────────────────────────");

    let t0 = Instant::now();

    let rosenbrock = |x: &[f64]| {
        let dx = 1.0 - x[0];
        let dy = x[0].mul_add(-x[0], x[1]);
        100.0f64.mul_add(dy * dy, dx * dx)
    };
    let bounds = &[(-5.0, 5.0), (-5.0, 5.0)];
    let (best_x, best_f, _iters) = barracuda::optimize::nelder_mead::nelder_mead(
        rosenbrock,
        &[0.0, 0.0],
        bounds,
        50_000,
        1e-12,
    )
    .expect("NM convergence");
    v.check_abs(
        "NM Rosenbrock x[0] [neuralSpring → S52]",
        best_x[0],
        1.0,
        0.05,
    );
    v.check_abs("NM Rosenbrock x[1]", best_x[1], 1.0, 0.05);
    v.check_lower("NM Rosenbrock f < 0.01", 0.01 - best_f, 0.0);

    let config = barracuda::optimize::bfgs::BfgsConfig::default();
    let quad = |x: &[f64]| {
        let dx = x[0] - 3.0;
        let dy = x[1] + 1.0;
        (2.0 * dy).mul_add(dy, dx * dx)
    };
    let bfgs = barracuda::optimize::bfgs::bfgs_numerical(&quad, &[0.0, 0.0], &config)
        .expect("BFGS convergence");
    v.check_abs(
        "BFGS quadratic x[0] [neuralSpring → S52]",
        bfgs.x[0],
        3.0,
        1e-4,
    );
    v.check_abs("BFGS quadratic x[1]", bfgs.x[1], -1.0, 1e-4);

    let f = |x: f64| x.powi(3) - 2.0f64.mul_add(x, 5.0);
    let df = |x: f64| 3.0f64.mul_add(x * x, -2.0);
    let nr = barracuda::optimize::newton::newton(f, df, 2.0, 1e-12, 50).expect("Newton");
    v.check_abs(
        "Newton x³-2x-5 residual [neuralSpring → S52]",
        f(nr.root),
        0.0,
        1e-10,
    );

    let sqrt2 =
        barracuda::optimize::bisect::bisect(|x: f64| x.mul_add(x, -2.0), 1.0, 2.0, 1e-14, 100)
            .expect("Bisect");
    v.check_abs(
        "Bisect √2 [neuralSpring → S52]",
        sqrt2,
        std::f64::consts::SQRT_2,
        1e-12,
    );

    let brent =
        barracuda::optimize::brent::brent(|x: f64| x.mul_add(x, -2.0), 1.0, 2.0, 1e-14, 100)
            .expect("Brent");
    v.check_abs(
        "Brent √2 [neuralSpring → S52]",
        brent.root,
        std::f64::consts::SQRT_2,
        1e-10,
    );

    println!("  neuralSpring optimizers: {:.1?}", t0.elapsed());
}

fn bench_airspring_rewired(v: &mut ValidationHarness) {
    println!("\n── airSpring Rewired Systems ────────────────────────────────");

    let t0 = Instant::now();

    let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.2).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.5f64.mul_add(xi, 1.0)).collect();
    let fit = barracuda::stats::regression::fit_linear(&x, &y).expect("linear fit");
    v.check_abs(
        "regression slope [airSpring → S66 absorption]",
        fit.params[0],
        2.5,
        1e-10,
    );
    v.check_abs("regression intercept", fit.params[1], 1.0, 1e-10);

    let tmin = 18.0_f64;
    let tmax = 32.0;
    let ra_mm = 40.0 / 2.45;
    let hg_upstream = barracuda::stats::hydrology::hargreaves_et0(ra_mm, tmax, tmin)
        .expect("Hargreaves upstream");
    let hg_local = et::hargreaves_et0(tmin, tmax, ra_mm);
    v.check_abs(
        "Hargreaves upstream==local [airSpring → S66]",
        hg_upstream,
        hg_local,
        0.01,
    );

    let ridge_x: Vec<f64> = (0..30).map(f64::from).collect();
    let ridge_y: Vec<f64> = ridge_x.iter().map(|&xi| 1.5f64.mul_add(xi, 2.0)).collect();
    let ridge = correction::fit_ridge(&ridge_x, &ridge_y, 1e-10).expect("ridge fit");
    v.check_abs(
        "ridge regression slope [wetSpring ESN → S58]",
        ridge.params[0],
        1.5,
        0.05,
    );

    println!("  airSpring rewired: {:.1?}", t0.elapsed());
}

fn bench_groundspring_uncertainty(v: &mut ValidationHarness) {
    println!("\n── groundSpring Uncertainty Lineage ─────────────────────────");

    let t0 = Instant::now();

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

    let mc = mc_et0_cpu(&input, &Et0Uncertainties::default(), 5000, 42);
    v.check_lower("MC ET₀ σ > 0.05 [groundSpring → S64]", mc.et0_std, 0.05);
    v.check_upper("MC ET₀ σ < 2.0", mc.et0_std, 2.0);
    v.check_lower("MC P05 < central", mc.et0_central - mc.et0_p05, 0.0);
    v.check_lower("MC P95 > central", mc.et0_p95 - mc.et0_central, 0.0);

    let (ci_lo, ci_hi) = mc.parametric_ci(0.90);
    v.check_lower("parametric CI lower < mean", mc.et0_mean - ci_lo, 0.0);
    v.check_lower("parametric CI upper > mean", ci_hi - mc.et0_mean, 0.0);

    let mc2 = mc_et0_cpu(&input, &Et0Uncertainties::default(), 5000, 42);
    v.check_abs(
        "MC ET₀ deterministic (same seed)",
        mc.et0_mean,
        mc2.et0_mean,
        1e-14,
    );

    let ci = barracuda::stats::bootstrap::bootstrap_mean(
        &[3.2, 3.5, 3.1, 3.8, 3.6, 3.3, 3.7, 3.4, 3.9, 3.0],
        1000,
        0.95,
        42,
    )
    .expect("bootstrap");
    v.check_lower(
        "Bootstrap CI lower < upper [hotSpring+groundSpring]",
        ci.upper - ci.lower,
        0.0,
    );

    println!("  groundSpring uncertainty: {:.1?}", t0.elapsed());
}

fn bench_tridiagonal_rewire(v: &mut ValidationHarness) {
    println!("\n── Tridiagonal Solver Rewire ────────────────────────────────");

    let t0 = Instant::now();

    let sub = vec![1.0, 1.0, 1.0];
    let diag = vec![4.0, 4.0, 4.0, 4.0];
    let sup = vec![1.0, 1.0, 1.0];
    let rhs = vec![5.0, 6.0, 6.0, 5.0];

    let x =
        barracuda::linalg::tridiagonal_solve(&sub, &diag, &sup, &rhs).expect("tridiagonal solve");

    let mut residual = 0.0_f64;
    let n = x.len();
    for i in 0..n {
        let mut ax_i = diag[i] * x[i];
        if i > 0 {
            ax_i += sub[i - 1] * x[i - 1];
        }
        if i < n - 1 {
            ax_i += sup[i] * x[i + 1];
        }
        residual += (ax_i - rhs[i]).powi(2);
    }
    residual = residual.sqrt();

    v.check_abs(
        "barracuda tridiagonal ‖Ax-d‖ [replaces local thomas_solve]",
        residual,
        0.0,
        1e-12,
    );

    let params = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };
    let profiles = solve_richards_1d(&params, 100.0, 20, -200.0, -50.0, false, true, 2.0, 0.1)
        .expect("Richards with barracuda tridiag");

    v.check_lower("Richards profiles non-empty", profiles.len() as f64, 1.0);
    let last = profiles.last().unwrap();
    v.check_bool(
        "Richards θ in [θr, θs] after rewire",
        last.theta
            .iter()
            .all(|&t| t >= params.theta_r && t <= params.theta_s),
    );

    println!("  Tridiagonal rewire: {:.1?}", t0.elapsed());
}
