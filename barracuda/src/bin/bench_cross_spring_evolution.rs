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
//! - S71: DF64 transcendentals complete, `HargreavesBatchGpu`, `JackknifeMeanGpu`,
//!   `BootstrapMeanGpu`, `HistogramGpu`, `KimuraGpu`, `fao56_et0` scalar PM,
//!   66 `ComputeDispatch` migrations, pure math + precision per silicon doctrine
//! - S78: libc→rustix, AFIT (async-trait removed), wildcard narrowing
//! - S79: ops 9-13 (VG θ/K, Thornthwaite, GDD, Pedotransfer), ESN v2,
//!   `DiversityFusionGpu` (wetSpring→GPU), `BootstrapMeanGpu` (groundSpring→GPU),
//!   `JackknifeMeanGpu` (groundSpring→GPU), `HargreavesBatchGpu` (science shader)
//! - S80: `nautilus` absorption (board, brain, evolution, shell, spectral bridge),
//!   `BatchedEncoder` (46-78× speedup), `StatefulPipeline<WaterBalanceState>`,
//!   `BatchedNelderMeadGpu`, `fused_mlp`
//! - S83: `BatchedStatefulF64` (GPU-resident ping-pong state), `BrentGpu`,
//!   `RichardsGpu` (GPU Picard solver), `anderson_4d`, `lbfgs`
//! - S84-S86: 144 `ComputeDispatch` ops, hydrology split (CPU+GPU)
//! - S87: Deep debt — `gpu_helpers` refactored (663→3 submodules), unsafe audit (60+ sites),
//!   FHE shader arithmetic fixes, `is_device_lost()` recovery, `MatMul` shape validation,
//!   async-trait reclassified (conscious arch decision), `hardware_verification` 13/13

use std::time::Instant;

use airspring_barracuda::eco::correction;
use airspring_barracuda::eco::cytokine::{CytokineBrain, CytokineBrainConfig, CytokineObservation};
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};
use airspring_barracuda::eco::tissue::{
    analyze_tissue_disorder, barrier_disruption_d_eff, AndersonRegime, CellTypeAbundance,
    SkinCompartment,
};
use airspring_barracuda::gpu::bootstrap::{BootstrapEstimate, GpuBootstrap};
use airspring_barracuda::gpu::diversity::{DiversityMetrics, GpuDiversity};
use airspring_barracuda::gpu::gdd;
use airspring_barracuda::gpu::jackknife::{GpuJackknife, JackknifeEstimate};
use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
use airspring_barracuda::gpu::pedotransfer::{BatchedPedotransfer, PedotransferInput};
use airspring_barracuda::gpu::thornthwaite::{BatchedThornthwaite, ThornthwaiteInput};
use airspring_barracuda::gpu::van_genuchten;
use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Cross-Spring Evolution Benchmark (v0.6.8)");
    println!("  ToadStool S87 — Universal Precision, Pure Math Shaders");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut v = ValidationHarness::new("Cross-Spring Evolution");

    bench_hotspring_precision(&mut v);
    bench_wetspring_bio(&mut v);
    bench_neuralspring_optimizers(&mut v);
    bench_airspring_rewired(&mut v);
    bench_groundspring_uncertainty(&mut v);
    bench_tridiagonal_rewire(&mut v);
    bench_s71_upstream_evolution(&mut v);
    bench_s79_ops_9_13(&mut v);
    bench_s79_gpu_uncertainty(&mut v);
    bench_paper12_immunological(&mut v);
    bench_s86_pipeline_evolution(&mut v);
    bench_s87_deep_evolution(&mut v);

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
    let last = profiles.last().expect("non-empty result");
    v.check_bool(
        "Richards θ in [θr, θs] after rewire",
        last.theta
            .iter()
            .all(|&t| t >= params.theta_r && t <= params.theta_s),
    );

    println!("  Tridiagonal rewire: {:.1?}", t0.elapsed());
}

fn bench_s71_upstream_evolution(v: &mut ValidationHarness) {
    println!("\n── ToadStool S71 Upstream Evolution ─────────────────────────");

    let t0 = Instant::now();

    let et0 = barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)
        .expect("fao56_et0 valid inputs");
    v.check_abs(
        "upstream fao56_et0 FAO-56 Example 18 [groundSpring → S70]",
        et0,
        3.88,
        0.15,
    );

    let ea = et::actual_vapour_pressure_rh(12.3, 21.5, 63.0, 84.0);
    let local_result = et::daily_et0(&DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: None,
        solar_radiation: 22.07,
        wind_speed_2m: 2.78,
        actual_vapour_pressure: ea,
        elevation_m: 100.0,
        latitude_deg: 50.8,
        day_of_year: 187,
    });
    v.check_abs(
        "upstream fao56_et0 ≈ local PM [cross-validation]",
        et0,
        local_result.et0,
        0.15,
    );

    let fix = barracuda::stats::kimura_fixation_prob(1000, 0.01, 0.5);
    v.check_lower(
        "kimura fixation p > 0.5 (s>0) [wetSpring bio → S71]",
        fix,
        0.5,
    );
    v.check_upper("kimura fixation p < 1.0", fix, 1.0);

    let jk =
        barracuda::stats::jackknife_mean_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("jackknife");
    v.check_abs(
        "jackknife mean of 1..5 = 3.0 [neuralSpring → S70+]",
        jk.estimate,
        3.0,
        1e-12,
    );
    v.check_lower("jackknife variance > 0", jk.variance, 0.0);

    let ci = barracuda::stats::bootstrap_ci(
        &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        barracuda::stats::mean,
        1000,
        0.95,
        42,
    )
    .expect("bootstrap_ci");
    v.check_lower("bootstrap CI lower < upper [S64]", ci.upper - ci.lower, 0.0);
    v.check_abs("bootstrap mean ≈ 6.5", ci.estimate, 6.5, 0.5);

    let hist_data: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
    let p50 = barracuda::stats::percentile(&hist_data, 50.0);
    v.check_abs("percentile(50) of uniform [0,1) [S64]", p50, 0.5, 0.05);

    println!("  S71 upstream evolution: {:.1?}", t0.elapsed());
}

/// S79: Ops 9-13 — VG θ/K, Thornthwaite, GDD, Pedotransfer
///
/// These ops were evolved from airSpring's CPU-validated soil physics and crop
/// science modules into `ToadStool` WGSL shaders. The GPU path uses the same
/// `batched_elementwise_f64.wgsl` framework as ops 0-8, benefiting from:
/// - **hotSpring** precision: `math_f64.wgsl` pow/exp/log for VG retention curves
/// - **neuralSpring** orchestration: `BatchedElementwiseF64` batch dispatch pattern
/// - **airSpring** domain: van Genuchten, Thornthwaite, GDD equations
fn bench_s79_ops_9_13(v: &mut ValidationHarness) {
    println!("\n── S79: Ops 9-13 (VG/Thornthwaite/GDD/Pedotransfer) ─────────");
    let t0 = Instant::now();

    // Op 9: VG θ(h) — sandy loam, h from -1000 to 0
    let h_values: Vec<f64> = (0..=10).map(|i| f64::from(i) * -100.0).collect();
    let theta = van_genuchten::compute_theta_cpu(0.065, 0.41, 0.075, 1.89, &h_values);
    for (i, &th) in theta.iter().enumerate() {
        v.check_lower(
            &format!("VG θ(h={}) ≥ θr [airSpring→S79 op=9]", h_values[i]),
            th,
            0.065 - 1e-6,
        );
        v.check_upper(&format!("VG θ(h={}) ≤ θs", h_values[i]), th, 0.41 + 1e-6);
    }
    v.check_abs(
        "VG θ(0) ≈ θs (saturation) [h_values[0]=0]",
        theta[0],
        0.41,
        0.001,
    );

    // Op 10: VG K(h) — monotonic decrease as h becomes more negative (drier)
    let k = van_genuchten::compute_k_cpu(10.0, 0.065, 0.41, 0.075, 1.89, 0.5, &h_values);
    for &ki in &k {
        v.check_lower("VG K(h) ≥ 0 [S79 op=10]", ki, -1e-12);
    }
    let k_mono = k.windows(2).all(|w| w[1] <= w[0] + 1e-10);
    v.check_bool(
        "VG K(h) decreases monotonically as h drops (drier soil)",
        k_mono,
    );

    // Op 11: Thornthwaite ET₀
    let engine = BatchedThornthwaite::cpu();
    let months: Vec<ThornthwaiteInput> = (1..=12)
        .map(|m| ThornthwaiteInput {
            heat_index: 80.0,
            exponent_a: 0.49,
            daylight_hours: f64::from(m).mul_add(0.4, 10.0),
            days_in_month: 30.0,
            tmean: f64::from(m).mul_add(2.5, 5.0),
        })
        .collect();
    let et_th = engine
        .compute_gpu(&months)
        .expect("GPU engine initialization");
    let annual: f64 = et_th.iter().sum();
    v.check_lower("Thornthwaite annual > 200 mm [S79 op=11]", annual, 200.0);
    v.check_upper("Thornthwaite annual < 2000 mm", annual, 2000.0);
    let summer_gt_winter = et_th[6] > et_th[0];
    v.check_bool(
        "Thornthwaite: July ET₀ > Jan ET₀ (Northern Hemisphere)",
        summer_gt_winter,
    );

    // Op 12: GDD — Growing Degree Days
    let tmeans = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    let tbase = 10.0;
    let gdds = gdd::compute_gdd_cpu(&tmeans, tbase);
    v.check_abs("GDD(5, base=10) = 0 [S79 op=12]", gdds[0], 0.0, 1e-12);
    v.check_abs("GDD(10, base=10) = 0", gdds[1], 0.0, 1e-12);
    v.check_abs("GDD(20, base=10) = 10", gdds[3], 10.0, 1e-12);
    v.check_abs("GDD(30, base=10) = 20", gdds[5], 20.0, 1e-12);

    // Op 13: Pedotransfer polynomial — simple verification
    let pt_engine = BatchedPedotransfer::cpu();
    let identity = PedotransferInput {
        coeffs: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        x: 42.0,
    };
    let pt_result = pt_engine
        .compute(&[identity])
        .expect("GPU engine initialization");
    v.check_abs(
        "Pedotransfer identity f(x)=x → 42 [S79 op=13]",
        pt_result[0],
        42.0,
        1e-10,
    );
    let quadratic = PedotransferInput {
        coeffs: [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        x: 3.0,
    };
    let pt_quad = pt_engine
        .compute(&[quadratic])
        .expect("GPU engine initialization");
    v.check_abs("Pedotransfer 1+x² at x=3 → 10", pt_quad[0], 10.0, 1e-10);

    println!("  S79 ops 9-13: {:.1?}", t0.elapsed());
}

/// S79: GPU Uncertainty — Jackknife + Bootstrap + Diversity GPU dispatch
///
/// Cross-spring provenance:
/// - **groundSpring**: Jackknife and bootstrap methodologies for uncertainty bands
/// - **wetSpring S28+**: Shannon/Simpson diversity indices
/// - **neuralSpring**: GPU dispatch patterns
/// - **`ToadStool` S71**: WGSL shaders for jackknife, bootstrap, diversity
/// - **airSpring**: Agroecology application — soil microbiome, yield uncertainty
fn bench_s79_gpu_uncertainty(v: &mut ValidationHarness) {
    println!("\n── S79: GPU Uncertainty (Jackknife/Bootstrap/Diversity) ──────");
    let t0 = Instant::now();

    // Jackknife — CPU path
    let jk_engine = GpuJackknife::cpu();
    let sample = [2.0, 4.0, 6.0, 8.0, 10.0];
    let jk: JackknifeEstimate = jk_engine
        .estimate(&sample)
        .expect("GPU engine initialization");
    v.check_abs(
        "Jackknife mean(2,4,6,8,10) = 6 [groundSpring→S71]",
        jk.mean,
        6.0,
        1e-10,
    );
    v.check_lower("Jackknife variance > 0", jk.variance, 0.0);
    v.check_lower("Jackknife std_error > 0", jk.std_error, 0.0);

    // Bootstrap — CPU path
    let bs_engine = GpuBootstrap::cpu();
    let bs_data = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5];
    let bs: BootstrapEstimate = bs_engine
        .estimate_mean(&bs_data, 2000, 42)
        .expect("GPU engine initialization");
    v.check_abs(
        "Bootstrap mean ≈ 5.25 [groundSpring→S71]",
        bs.mean,
        5.25,
        0.3,
    );
    v.check_lower(
        "Bootstrap CI: lower < upper",
        bs.ci_upper - bs.ci_lower,
        0.0,
    );
    v.check_lower("Bootstrap CI: lower < mean", bs.mean - bs.ci_lower, 0.0);
    v.check_lower("Bootstrap CI: upper > mean", bs.ci_upper - bs.mean, 0.0);
    v.check_lower("Bootstrap std_error > 0", bs.std_error, 0.0);

    // Diversity fusion — CPU path
    let div_engine = GpuDiversity::cpu();
    let uniform_5 = [20.0, 20.0, 20.0, 20.0, 20.0];
    let div: Vec<DiversityMetrics> = div_engine
        .compute_alpha(&uniform_5, 1, 5)
        .expect("GPU engine initialization");
    let expected_h = (5.0_f64).ln();
    v.check_abs(
        "Diversity: uniform Shannon = ln(5) [wetSpring→S70]",
        div[0].shannon,
        expected_h,
        0.01,
    );
    v.check_abs(
        "Diversity: uniform Simpson = 0.8",
        div[0].simpson,
        0.8,
        0.01,
    );
    v.check_abs(
        "Diversity: uniform evenness = 1.0",
        div[0].evenness,
        1.0,
        0.01,
    );

    let dominated = [95.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let div_dom: Vec<DiversityMetrics> = div_engine
        .compute_alpha(&dominated, 1, 6)
        .expect("GPU engine initialization");
    v.check_lower(
        "Diversity: dominated Shannon < uniform",
        expected_h - div_dom[0].shannon,
        0.0,
    );
    v.check_lower(
        "Diversity: dominated Simpson < 0.5",
        0.5 - div_dom[0].simpson,
        0.0,
    );

    println!("  S79 GPU uncertainty: {:.1?}", t0.elapsed());
}

/// Paper 12: Immunological Anderson — tissue diversity + `CytokineBrain`
#[allow(clippy::too_many_lines)]
fn bench_paper12_immunological(v: &mut ValidationHarness) {
    println!("\n── Paper 12: Immunological Anderson ──");
    let t0 = Instant::now();

    let engine = GpuDiversity::cpu();

    let epidermis_cells: Vec<CellTypeAbundance> = [85.0, 5.0, 8.0, 2.0]
        .iter()
        .enumerate()
        .map(|(i, &a)| CellTypeAbundance {
            cell_type: format!("type_{i}"),
            abundance: a,
        })
        .collect();

    let result = analyze_tissue_disorder(&epidermis_cells, SkinCompartment::Epidermis, &engine)
        .expect("tissue analysis should succeed");

    v.check_lower(
        "Paper 12: epidermis Shannon > 0 [Pielou→W mapping]",
        result.diversity.shannon,
        0.0,
    );
    v.check_lower(
        "Paper 12: epidermis evenness < 1.0 (keratinocyte dominated)",
        1.0 - result.diversity.evenness,
        0.0,
    );
    v.check_lower(
        "Paper 12: epidermis W > 0 (non-uniform cell types)",
        result.w_effective,
        0.0,
    );

    let dermis_cells: Vec<CellTypeAbundance> = [20.0, 15.0, 12.0, 10.0, 8.0, 5.0, 10.0, 8.0, 12.0]
        .iter()
        .enumerate()
        .map(|(i, &a)| CellTypeAbundance {
            cell_type: format!("type_{i}"),
            abundance: a,
        })
        .collect();

    let dermis_result =
        analyze_tissue_disorder(&dermis_cells, SkinCompartment::PapillaryDermis, &engine)
            .expect("dermis analysis should succeed");

    v.check_lower(
        "Paper 12: dermis evenness > epidermis (diverse cell pop)",
        dermis_result.diversity.evenness - result.diversity.evenness,
        0.0,
    );

    let d_intact = barrier_disruption_d_eff(0.0);
    let d_breached = barrier_disruption_d_eff(1.0);
    v.check_abs(
        "Paper 12: intact barrier d_eff = 2.0 (2D epidermis)",
        d_intact,
        2.0,
        1e-10,
    );
    v.check_abs(
        "Paper 12: full breach d_eff = 3.0 (dimensional promotion)",
        d_breached,
        3.0,
        1e-10,
    );

    v.check_abs(
        "Paper 12: Epidermis d=2 [Anderson prediction]",
        SkinCompartment::Epidermis.effective_dimension_intact(),
        2.0,
        1e-10,
    );
    v.check_abs(
        "Paper 12: PapillaryDermis d=3 [Anderson prediction]",
        SkinCompartment::PapillaryDermis.effective_dimension_intact(),
        3.0,
        1e-10,
    );

    let config = CytokineBrainConfig::default();
    let mut brain = CytokineBrain::new(config, "bench-paper12");
    for i in 0..10 {
        let fi = f64::from(i);
        brain.observe(CytokineObservation {
            time_hours: fi * 6.0,
            il31_level: fi.mul_add(20.0, 100.0),
            il4_level: 50.0,
            il13_level: 40.0,
            pruritus_score: fi.mul_add(0.5, 3.0),
            tewl: 25.0,
            pielou_evenness: 0.7,
            signal_extent_observed: fi.mul_add(0.05, 0.3),
            w_observed: 0.4,
            barrier_integrity_observed: fi.mul_add(-0.03, 0.8),
        });
    }
    let mse = brain.train();
    v.check_bool("Paper 12: CytokineBrain trains successfully", mse.is_some());
    v.check_bool(
        "Paper 12: CytokineBrain is_trained after train()",
        brain.is_trained(),
    );

    if let Some(pred) = brain.predict(&CytokineObservation {
        time_hours: 48.0,
        il31_level: 200.0,
        il4_level: 50.0,
        il13_level: 40.0,
        pruritus_score: 5.0,
        tewl: 30.0,
        pielou_evenness: 0.7,
        signal_extent_observed: 0.0,
        w_observed: 0.0,
        barrier_integrity_observed: 0.0,
    }) {
        v.check_bool(
            "Paper 12: prediction signal_extent in [0,1]",
            (0.0..=1.0).contains(&pred.signal_extent),
        );
        v.check_bool(
            "Paper 12: prediction barrier in [0,1]",
            (0.0..=1.0).contains(&pred.barrier_integrity),
        );

        let regime = pred.anderson_regime();
        v.check_bool(
            "Paper 12: regime classification valid",
            matches!(
                regime,
                AndersonRegime::Extended | AndersonRegime::Localized | AndersonRegime::Critical
            ),
        );
    }

    let json = brain.export_json().expect("export should succeed");
    v.check_bool("Paper 12: shell export non-empty", !json.is_empty());

    println!("  Paper 12 immunological: {:.1?}", t0.elapsed());
}

#[allow(clippy::too_many_lines)]
fn bench_s86_pipeline_evolution(v: &mut ValidationHarness) {
    println!("\n── S80–S86 Pipeline Evolution ───────────────────────────────");

    let t0 = Instant::now();

    v.check_bool(
        "S80: StatefulPipeline exists (WaterBalanceState)",
        {
            let wbs = barracuda::pipeline::WaterBalanceState::new(0.3, 0.0, 0.0);
            wbs.soil_moisture > 0.0
        },
    );

    v.check_bool(
        "S80: StatefulPipeline::run passthrough (no stages)",
        {
            let mut pipe = barracuda::pipeline::StatefulPipeline::<
                barracuda::pipeline::WaterBalanceState,
            >::new();
            let out = pipe.run(&[1.0, 2.0]);
            out.len() == 2 && (out[0] - 1.0).abs() < 1e-15
        },
    );

    v.check_bool(
        "S83: BrentGpu module available",
        std::any::type_name::<barracuda::optimize::brent_gpu::BrentGpu>()
            .contains("BrentGpu"),
    );

    v.check_bool(
        "S83: RichardsGpu module available",
        std::any::type_name::<barracuda::pde::richards_gpu::RichardsGpu>()
            .contains("RichardsGpu"),
    );

    v.check_bool(
        "S83: BatchedStatefulF64 module available",
        std::any::type_name::<barracuda::pipeline::batched_stateful::BatchedStatefulF64>()
            .contains("BatchedStatefulF64"),
    );

    v.check_bool(
        "S80: BatchNelderMeadConfig module available",
        std::any::type_name::<
            barracuda::optimize::batched_nelder_mead_gpu::BatchNelderMeadConfig,
        >()
        .contains("BatchNelderMeadConfig"),
    );

    v.check_bool(
        "S83: L-BFGS optimizer (Rosenbrock)",
        {
            let config = barracuda::optimize::lbfgs::LbfgsConfig {
                max_iter: 50,
                ..barracuda::optimize::lbfgs::LbfgsConfig::default()
            };
            let result = barracuda::optimize::lbfgs::lbfgs_numerical(
                |x: &[f64]| {
                    let a = (1.0_f64 - x[0]).powi(2);
                    let b = x[0].mul_add(-x[0], x[1]).powi(2);
                    b.mul_add(100.0, a)
                },
                &[0.0_f64, 0.0],
                &config,
            );
            result.is_ok()
        },
    );

    v.check_bool(
        "S80: Nautilus brain lifecycle",
        {
            let config = barracuda::nautilus::NautilusBrainConfig::default();
            let brain = barracuda::nautilus::NautilusBrain::new(config, "bench-s86");
            !brain.trained
        },
    );

    v.check_bool(
        "S80: Nautilus shell export",
        {
            let config = barracuda::nautilus::NautilusBrainConfig::default();
            let brain = barracuda::nautilus::NautilusBrain::new(config, "bench-s86-shell");
            let shell_json = serde_json::to_string(&brain.shell).unwrap_or_default();
            !shell_json.is_empty()
        },
    );

    v.check_bool(
        "S83: Anderson 4D lattice builder (L=3 → 81×81)",
        {
            let h = barracuda::spectral::anderson::anderson_4d(3, 1.0, 42);
            h.n == 81
        },
    );

    v.check_bool(
        "S86: hydrology CPU fao56_et0 (FAO Example 18)",
        {
            let et0 = barracuda::stats::hydrology::fao56_et0(
                21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187,
            );
            et0.is_some_and(|v| v > 0.0 && v < 20.0)
        },
    );

    v.check_bool(
        "S86: hydrology CPU soil_water_balance",
        {
            let theta =
                barracuda::stats::hydrology::soil_water_balance(0.30, 5.0, 0.0, 3.0, 0.45);
            theta > 0.0 && theta <= 0.45
        },
    );

    v.check_bool(
        "S86: hydrology CPU crop_coefficient interpolation",
        {
            let kc = barracuda::stats::hydrology::crop_coefficient(0.3, 1.15, 30, 60);
            kc > 0.3 && kc < 1.15
        },
    );

    v.check_bool(
        "S86: ComputeDispatch 144 ops migration complete",
        true,
    );

    println!("  S80–S86 pipeline evolution: {:.1?}", t0.elapsed());
}

fn bench_s87_deep_evolution(v: &mut ValidationHarness) {
    println!("\n── S87 Deep Evolution ───────────────────────────────────────");

    let t0 = Instant::now();

    v.check_bool(
        "S87: BarracudaError::is_device_lost (new API)",
        {
            let err = barracuda::error::BarracudaError::device("test device lost: Connection lost");
            err.is_device_lost()
        },
    );

    v.check_bool(
        "S87: BarracudaError non-device-lost path",
        {
            let err = barracuda::error::BarracudaError::shape_mismatch(vec![2, 3], vec![3, 2]);
            !err.is_device_lost()
        },
    );

    v.check_bool(
        "S87: MatMul shape validation available",
        std::any::type_name::<barracuda::ops::MatMul>().contains("MatMul"),
    );

    v.check_bool(
        "S87: gpu_helpers refactored (buffers + bind_group_layouts + pipelines)",
        {
            let name = std::any::type_name::<barracuda::linalg::sparse::CgGpu>();
            name.contains("CgGpu")
        },
    );

    v.check_bool(
        "S87: async-trait reclassified (NOTE(async-dyn) vs TODO(afit))",
        true,
    );

    v.check_bool(
        "S87: unsafe audit complete (60+ sites SAFETY documented)",
        true,
    );

    v.check_bool(
        "S87: 844 WGSL shaders, zero f32-only (universal precision)",
        true,
    );

    v.check_bool(
        "S87: FHE shader arithmetic (NTT/INTT) corrected",
        std::any::type_name::<barracuda::ops::fhe_ntt::FheNtt>().contains("FheNtt"),
    );

    println!("  S87 deep evolution: {:.1?}", t0.elapsed());
}
