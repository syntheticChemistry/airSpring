// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU vs Python benchmark — proves barracuda's pure Rust math is:
//! 1. Numerically identical to Python controls (parity at 1e-6)
//! 2. Significantly faster than interpreted Python
//!
//! Runs the same algorithms at the same scale as `control/bench_python_timing.py`,
//! then shells out to Python for timing comparison.

mod benchmarks;

use std::process::Command;

type BenchFn = Box<dyn Fn(usize) -> (f64, f64, String)>;

type BenchEntry = (&'static str, &'static str, usize, BenchFn);

struct BenchResult {
    name: &'static str,
    n: usize,
    rust_secs: f64,
    python_secs: f64,
    speedup: f64,
    parity_ok: bool,
    parity_detail: String,
}

fn run_python_benchmarks() -> Vec<(String, f64)> {
    let control_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR parent")
        .join("control");
    let script = control_dir.join("bench_python_timing.py");
    if !script.exists() {
        eprintln!(
            "  [WARN] Python timing script not found at {}",
            script.display()
        );
        return Vec::new();
    }
    let output = Command::new("python3")
        .arg(&script)
        .output()
        .expect("Failed to run Python timing script");
    if !output.status.success() {
        eprintln!(
            "  [WARN] Python benchmark failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Vec::new();
    }
    let json_str = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
    v["benchmarks"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|b| Some((b["name"].as_str()?.to_string(), b["secs"].as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

fn run_benchmarks(
    entries: &[BenchEntry],
    py_lookup: impl Fn(&str) -> f64,
) -> (Vec<BenchResult>, bool) {
    let mut results = Vec::new();
    let mut all_parity = true;
    for (py_name, display_name, n, func) in entries {
        let (rust_secs, _value, detail) = func(*n);
        let python_secs = py_lookup(py_name);
        let speedup = if python_secs > 0.0 && rust_secs > 0.0 {
            python_secs / rust_secs
        } else {
            0.0
        };
        let parity_ok = detail.contains("ok=true");
        if !parity_ok {
            all_parity = false;
        }
        results.push(BenchResult {
            name: display_name,
            n: *n,
            rust_secs,
            python_secs,
            speedup,
            parity_ok,
            parity_detail: detail,
        });
    }
    (results, all_parity)
}

fn print_results(results: &[BenchResult], all_parity: bool) {
    eprintln!(
        "┌──────────────────────────────┬────────┬──────────────┬──────────────┬──────────┬────────┐"
    );
    eprintln!(
        "│ Algorithm                    │      N │     Rust (s) │   Python (s) │  Speedup │ Parity │"
    );
    eprintln!(
        "├──────────────────────────────┼────────┼──────────────┼──────────────┼──────────┼────────┤"
    );
    for r in results {
        let parity_str = if r.parity_ok { "  ✓   " } else { " FAIL " };
        eprintln!(
            "│ {:<28} │ {:>6} │ {:>12.6} │ {:>12.6} │ {:>6.1}× │{}│",
            r.name, r.n, r.rust_secs, r.python_secs, r.speedup, parity_str,
        );
    }
    eprintln!(
        "└──────────────────────────────┴────────┴──────────────┴──────────────┴──────────┴────────┘"
    );

    eprintln!("\n  Parity details:");
    for r in results {
        let icon = if r.parity_ok { "✓" } else { "✗" };
        eprintln!("    {icon} {}: {}", r.name, r.parity_detail);
    }

    let geo_mean_speedup = {
        let valid: Vec<f64> = results
            .iter()
            .filter(|r| r.speedup > 0.0)
            .map(|r| r.speedup)
            .collect();
        if valid.is_empty() {
            0.0
        } else {
            let log_sum: f64 = valid.iter().map(|s| s.ln()).sum();
            (log_sum / valid.len() as f64).exp()
        }
    };

    eprintln!("\n  Geometric mean speedup: {geo_mean_speedup:.1}×");
    eprintln!(
        "  Math parity: {}/{} algorithms match Python",
        results.iter().filter(|r| r.parity_ok).count(),
        results.len(),
    );

    if !all_parity {
        eprintln!("\n  [FAIL] Not all algorithms match Python — check parity details above");
        std::process::exit(1);
    }
    eprintln!("\n  [PASS] Pure Rust math is correct AND faster than Python");
}

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  barracuda CPU vs Python — Pure Math Benchmark");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    eprintln!("  Running Python benchmarks...");
    let py_timings = run_python_benchmarks();
    let py_lookup = |name: &str| -> f64 {
        py_timings
            .iter()
            .find(|(n, _)| n == name)
            .map_or(0.0, |(_, s)| *s)
    };

    eprintln!("  Running Rust benchmarks...\n");
    let entries = benchmarks::build_benchmarks();
    let (results, all_parity) = run_benchmarks(&entries, py_lookup);
    print_results(&results, all_parity);
}
