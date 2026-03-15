// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision and math lineage benchmarks — hotSpring, wetSpring, neuralSpring, tridiagonal.

use std::time::Instant;

use airspring_barracuda::eco::richards::{VanGenuchtenParams, solve_richards_1d};
use barracuda::validation::ValidationHarness;

pub fn bench_hotspring_precision(v: &mut ValidationHarness) {
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

pub fn bench_wetspring_bio(v: &mut ValidationHarness) {
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

pub fn bench_neuralspring_optimizers(v: &mut ValidationHarness) {
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

pub fn bench_tridiagonal_rewire(v: &mut ValidationHarness) {
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
    let last = profiles.last().expect("non-empty result");
    v.check_bool(
        "Richards θ in [θr, θs] after rewire",
        last.theta
            .iter()
            .all(|&t| t >= params.theta_r && t <= params.theta_s),
    );

    println!("  Tridiagonal rewire: {:.1?}", t0.elapsed());
}
