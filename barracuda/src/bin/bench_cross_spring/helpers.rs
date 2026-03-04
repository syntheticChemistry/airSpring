// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark runner infrastructure and report printing.

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

use airspring_barracuda::gpu::device_info::{self, PROVENANCE};

pub fn run_bench(
    pass: &mut u32,
    fail: &mut u32,
    name: &str,
    origin: &str,
    body: impl FnOnce() -> bool,
) {
    let t0 = Instant::now();
    let ok = body();
    let elapsed = t0.elapsed();
    let status = if ok { "PASS" } else { "FAIL" };
    if ok {
        *pass += 1;
    } else {
        *fail += 1;
    }
    println!(
        "  [{status}] {:<40} {:>8.2}ms  ({})",
        name,
        elapsed.as_secs_f64() * 1000.0,
        origin
    );
}

#[macro_export]
macro_rules! bench_suite {
    ($pass:expr, $fail:expr, $(($name:expr, $origin:expr, $body:expr)),+ $(,)?) => {
        $($crate::helpers::run_bench($pass, $fail, $name, $origin, $body);)+
    };
}

pub fn print_device_report(device: Option<&Arc<WgpuDevice>>) {
    if let Some(dev) = device {
        let report = device_info::probe_device(dev);
        println!("\n── Device Precision Report ──────────────────────────────────");
        println!("{report}");
        println!();
    } else {
        println!("  [No f64-capable GPU found — CPU-only benchmarks]\n");
    }
}

pub fn print_summary(pass: u32, fail: u32) {
    println!("\n── Summary ─────────────────────────────────────────────────\n");
    println!("  Total:  {} benchmarks", pass + fail);
    println!("  PASS:   {pass}");
    println!("  FAIL:   {fail}");
    if fail == 0 {
        println!("\n  All cross-spring GPU paths validated.");
    }
    println!();
}

pub fn print_provenance_report() {
    println!("── Cross-Spring Shader Provenance ───────────────────────────\n");
    println!(
        "  {:30} {:22} {:>5}  airSpring Use",
        "Shader", "Origin", "Prims"
    );
    println!("  {}", "─".repeat(90));
    for p in PROVENANCE {
        println!(
            "  {:30} {:22} {:>5}  {}",
            p.shader,
            p.origin,
            p.primitives.len(),
            truncate(p.domain_use, 35)
        );
    }
    let total_prims: usize = PROVENANCE.iter().map(|p| p.primitives.len()).sum();
    let origins: std::collections::HashSet<&str> = PROVENANCE.iter().map(|p| p.origin).collect();
    println!("  {}", "─".repeat(90));
    println!(
        "  {} shaders, {} primitives, {} origin Springs",
        PROVENANCE.len(),
        total_prims,
        origins.len()
    );
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
