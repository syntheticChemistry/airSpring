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

mod domain;
mod gpu_ops;
mod paper12;
mod pipeline;
mod precision;

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

    precision::bench_hotspring_precision(&mut v);
    precision::bench_wetspring_bio(&mut v);
    precision::bench_neuralspring_optimizers(&mut v);
    domain::bench_airspring_rewired(&mut v);
    domain::bench_groundspring_uncertainty(&mut v);
    precision::bench_tridiagonal_rewire(&mut v);
    domain::bench_s71_upstream_evolution(&mut v);
    gpu_ops::bench_s79_ops_9_13(&mut v);
    gpu_ops::bench_s79_gpu_uncertainty(&mut v);
    paper12::bench_paper12_immunological(&mut v);
    pipeline::bench_s86_pipeline_evolution(&mut v);
    pipeline::bench_s87_deep_evolution(&mut v);

    println!();
    v.finish();
}
