// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: JSON fixture counts and indices; bounded for parity checks"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: JSON and harness counts; bounded for test fixtures"
)]

//! Exp 057: GPU Ops 5-8 Rewire Validation + Cross-Spring Benchmark
//!
//! Validates `BarraCuda` S70+ absorption: batched elementwise ops GPU vs CPU, timing
//! benchmarks, and cross-spring shader provenance. Core logic lives in the
//! `validate_gpu_rewire_support` module.

mod validate_gpu_rewire_support;

use std::sync::Arc;

use airspring_barracuda::validation::{self, ValidationHarness};
use barracuda::device::WgpuDevice;

use validate_gpu_rewire_support::{run_cpu_only_validation, run_gpu_validation};

fn main() {
    validation::init_tracing();
    validation::banner("Exp 057: GPU Ops 5-8 Rewire Validation + Cross-Spring Benchmark");
    println!(
        "Validates BarraCuda S70+ absorption: all 6 batched ops GPU vs CPU,\n\
         timing benchmarks, and cross-spring evolution provenance.\n"
    );

    let mut v = ValidationHarness::new("GPU Rewire + Benchmark");

    if let Ok(d) = barracuda::device::test_pool::tokio_block_on(WgpuDevice::new_f64_capable()) {
        let device = Arc::new(d);
        println!("GPU device: {:?}", device.adapter_info());
        run_gpu_validation(&mut v, &device);
    } else {
        println!("No f64-capable GPU — running CPU-only validation");
        run_cpu_only_validation(&mut v);
        v.finish();
    }
}
