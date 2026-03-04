// SPDX-License-Identifier: AGPL-3.0-or-later
//! S79 GPU ops and uncertainty benchmarks.

use std::time::Instant;

use airspring_barracuda::gpu::bootstrap::{BootstrapEstimate, GpuBootstrap};
use airspring_barracuda::gpu::diversity::{DiversityMetrics, GpuDiversity};
use airspring_barracuda::gpu::gdd;
use airspring_barracuda::gpu::jackknife::{GpuJackknife, JackknifeEstimate};
use airspring_barracuda::gpu::pedotransfer::{BatchedPedotransfer, PedotransferInput};
use airspring_barracuda::gpu::thornthwaite::{BatchedThornthwaite, ThornthwaiteInput};
use airspring_barracuda::gpu::van_genuchten;
use barracuda::validation::ValidationHarness;

/// S79: Ops 9-13 — VG θ/K, Thornthwaite, GDD, Pedotransfer
pub fn bench_s79_ops_9_13(v: &mut ValidationHarness) {
    println!("\n── S79: Ops 9-13 (VG/Thornthwaite/GDD/Pedotransfer) ─────────");
    let t0 = Instant::now();

    bench_van_genuchten(v);
    bench_thornthwaite(v);
    bench_gdd(v);
    bench_pedotransfer(v);

    println!("  S79 ops 9-13: {:.1?}", t0.elapsed());
}

fn bench_van_genuchten(v: &mut ValidationHarness) {
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

    let k = van_genuchten::compute_k_cpu(10.0, 0.065, 0.41, 0.075, 1.89, 0.5, &h_values);
    for &ki in &k {
        v.check_lower("VG K(h) ≥ 0 [S79 op=10]", ki, -1e-12);
    }
    let k_mono = k.windows(2).all(|w| w[1] <= w[0] + 1e-10);
    v.check_bool(
        "VG K(h) decreases monotonically as h drops (drier soil)",
        k_mono,
    );
}

fn bench_thornthwaite(v: &mut ValidationHarness) {
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
    v.check_bool(
        "Thornthwaite: July ET₀ > Jan ET₀ (Northern Hemisphere)",
        et_th[6] > et_th[0],
    );
}

fn bench_gdd(v: &mut ValidationHarness) {
    let tmeans = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    let tbase = 10.0;
    let gdds = gdd::compute_gdd_cpu(&tmeans, tbase);
    v.check_abs("GDD(5, base=10) = 0 [S79 op=12]", gdds[0], 0.0, 1e-12);
    v.check_abs("GDD(10, base=10) = 0", gdds[1], 0.0, 1e-12);
    v.check_abs("GDD(20, base=10) = 10", gdds[3], 10.0, 1e-12);
    v.check_abs("GDD(30, base=10) = 20", gdds[5], 20.0, 1e-12);
}

fn bench_pedotransfer(v: &mut ValidationHarness) {
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
}

/// S79: GPU Uncertainty — Jackknife + Bootstrap + Diversity GPU dispatch
pub fn bench_s79_gpu_uncertainty(v: &mut ValidationHarness) {
    println!("\n── S79: GPU Uncertainty (Jackknife/Bootstrap/Diversity) ──────");
    let t0 = Instant::now();

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
