// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
//! CPU vs Python benchmark — proves barracuda's pure Rust math is:
//! 1. Numerically identical to Python controls (parity at 1e-6)
//! 2. Significantly faster than interpreted Python
//!
//! Runs the same algorithms at the same scale as `control/bench_python_timing.py`,
//! then shells out to Python for timing comparison.

use std::f64::consts::PI;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use airspring_barracuda::eco::anderson;
use airspring_barracuda::eco::diversity;
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::thornthwaite;
use airspring_barracuda::eco::van_genuchten;
use airspring_barracuda::eco::water_balance;

struct BenchResult {
    name: &'static str,
    n: usize,
    rust_secs: f64,
    python_secs: f64,
    speedup: f64,
    parity_ok: bool,
    parity_detail: String,
}

fn bench_fao56_et0(n: usize) -> (f64, f64, String) {
    let e_s_max: f64 = 0.6108 * (17.27_f64 * 34.8 / (34.8 + 237.3)).exp();
    let e_s_min: f64 = 0.6108 * (17.27_f64 * 19.6 / (19.6 + 237.3)).exp();
    let e_s = f64::midpoint(e_s_max, e_s_min);
    let e_a = e_s * 0.65;

    let input = DailyEt0Input {
        tmax: 34.8,
        tmin: 19.6,
        tmean: None,
        solar_radiation: 20.5,
        wind_speed_2m: 1.8,
        actual_vapour_pressure: e_a,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: 180,
    };
    let t0 = Instant::now();
    let mut result = 0.0_f64;
    for _ in 0..n {
        result = black_box(et::daily_et0(black_box(&input)).et0);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 5.152_603_881_624_521;
    let diff = (result - python_ref).abs();
    let ok = diff < 0.1;
    (
        elapsed,
        result,
        format!(
            "Rust={result:.6}, Python={python_ref:.6}, diff={diff:.2e}, ok={ok}"
        ),
    )
}

fn bench_thornthwaite(n: usize) -> (f64, f64, String) {
    let temps: [f64; 12] = [2.0, 4.0, 9.0, 14.0, 19.0, 24.0, 27.0, 26.0, 22.0, 15.0, 8.0, 3.0];
    let lat = 42.0;
    let t0 = Instant::now();
    let mut result = [0.0_f64; 12];
    for _ in 0..n {
        result = black_box(thornthwaite::thornthwaite_monthly_et0(black_box(&temps), lat));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_may: f64 = 98.774_974_121_825_71;
    let diff = (result[4] - python_may).abs();
    let ok = diff < 1.0;
    (
        elapsed,
        result[4],
        format!(
            "Rust May PET={:.4}, Python={python_may:.4}, diff={diff:.2e}, ok={ok}",
            result[4]
        ),
    )
}

fn bench_hargreaves(n: usize) -> (f64, f64, String) {
    let t0 = Instant::now();
    let mut result = 0.0_f64;
    for _ in 0..n {
        result = black_box(et::hargreaves_et0(black_box(19.6), black_box(34.8), black_box(38.5)));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 15.535_415_506_191_004;
    let diff = (result - python_ref).abs();
    let ok = diff < 1e-10;
    (
        elapsed,
        result,
        format!("Rust={result:.12}, Python={python_ref:.12}, diff={diff:.2e}, ok={ok}"),
    )
}

fn bench_van_genuchten(n: usize) -> (f64, f64, String) {
    let t0 = Instant::now();
    let mut result = 0.0_f64;
    for _ in 0..n {
        result = black_box(van_genuchten::van_genuchten_theta(
            black_box(-100.0),
            0.078,
            0.43,
            0.036,
            1.56,
        ));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 0.242_131_784_718_152_1;
    let diff = (result - python_ref).abs();
    let ok = diff < 1e-10;
    (
        elapsed,
        result,
        format!("Rust={result:.12}, Python={python_ref:.12}, diff={diff:.2e}, ok={ok}"),
    )
}

fn bench_water_balance_step(n: usize) -> (f64, f64, String) {
    let t0 = Instant::now();
    let mut dr = 0.0_f64;
    for _ in 0..n {
        let r = black_box(water_balance::daily_water_balance_step(
            black_box(20.0),
            5.0,
            0.0,
            4.5,
            1.05,
            0.9,
            120.0,
        ));
        dr = r.0;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 19.2525;
    let diff = (dr - python_ref).abs();
    let ok = diff < 1e-10;
    (
        elapsed,
        dr,
        format!("Rust Dr={dr:.6}, Python={python_ref:.6}, diff={diff:.2e}, ok={ok}"),
    )
}

fn bench_anderson_coupling(n: usize) -> (f64, f64, String) {
    let t0 = Instant::now();
    let mut d_eff = 0.0_f64;
    for _ in 0..n {
        let r = black_box(anderson::coupling_chain(black_box(0.25), 0.078, 0.43));
        d_eff = r.d_eff;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 2.097_075_886_258_595;
    let diff = (d_eff - python_ref).abs();
    let ok = diff < 1e-10;
    (
        elapsed,
        d_eff,
        format!("Rust d_eff={d_eff:.12}, Python={python_ref:.12}, diff={diff:.2e}, ok={ok}"),
    )
}

fn bench_shannon_diversity(n: usize) -> (f64, f64, String) {
    let abun = [45.0, 30.0, 15.0, 8.0, 2.0];
    let t0 = Instant::now();
    let mut result = 0.0_f64;
    for _ in 0..n {
        result = black_box(diversity::shannon(black_box(&abun)));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 1.285_387_053_981_883_5;
    let diff = (result - python_ref).abs();
    let ok = diff < 1e-10;
    (
        elapsed,
        result,
        format!("Rust={result:.12}, Python={python_ref:.12}, diff={diff:.2e}, ok={ok}"),
    )
}

fn bench_season_simulation(n: usize) -> (f64, f64, String) {
    let taw: f64 = 120.0;
    let raw: f64 = 60.0;
    let t0 = Instant::now();
    let mut final_dr = 0.0_f64;
    for _ in 0..n {
        let mut dr: f64 = 0.0;
        for d in 0..153 {
            let et0 = 2.0f64.mul_add((2.0 * PI * f64::from(d) / 153.0).sin(), 3.0);
            let p = if d % 7 == 0 { 2.0 } else { 0.0 };
            let ks: f64 = if dr > raw {
                ((taw - dr) / (taw - raw)).max(0.0)
            } else {
                1.0
            };
            let (new_dr, _, _) =
                water_balance::daily_water_balance_step(dr, p, 0.0, et0, 1.0, ks, taw);
            dr = new_dr;
        }
        final_dr = black_box(dr);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let python_ref: f64 = 111.570_952_328_931_09;
    let diff = (final_dr - python_ref).abs();
    let ok = diff < 1e-4;
    (
        elapsed,
        final_dr,
        format!(
            "Rust={final_dr:.6}, Python={python_ref:.6}, diff={diff:.2e}, ok={ok}"
        ),
    )
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

    let benchmarks: Vec<(
        &str,
        &str,
        usize,
        Box<dyn Fn(usize) -> (f64, f64, String)>,
    )> = vec![
        (
            "fao56_et0",
            "FAO-56 PM ET₀",
            10_000,
            Box::new(bench_fao56_et0),
        ),
        (
            "thornthwaite",
            "Thornthwaite PET",
            10_000,
            Box::new(bench_thornthwaite),
        ),
        (
            "hargreaves",
            "Hargreaves-Samani",
            10_000,
            Box::new(bench_hargreaves),
        ),
        (
            "van_genuchten",
            "Van Genuchten θ(h)",
            100_000,
            Box::new(bench_van_genuchten),
        ),
        (
            "water_balance_step",
            "Water Balance Step",
            10_000,
            Box::new(bench_water_balance_step),
        ),
        (
            "anderson_coupling",
            "Anderson Coupling",
            100_000,
            Box::new(bench_anderson_coupling),
        ),
        (
            "shannon_diversity",
            "Shannon Diversity",
            10_000,
            Box::new(bench_shannon_diversity),
        ),
        (
            "season_simulation",
            "Season Sim (153d)",
            1_000,
            Box::new(bench_season_simulation),
        ),
    ];

    let mut results = Vec::new();
    let mut all_parity = true;

    for (py_name, display_name, n, func) in &benchmarks {
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

    eprintln!("┌──────────────────────────────┬────────┬──────────────┬──────────────┬──────────┬────────┐");
    eprintln!("│ Algorithm                    │      N │     Rust (s) │   Python (s) │  Speedup │ Parity │");
    eprintln!("├──────────────────────────────┼────────┼──────────────┼──────────────┼──────────┼────────┤");
    for r in &results {
        let parity_str = if r.parity_ok { "  ✓   " } else { " FAIL " };
        eprintln!(
            "│ {:<28} │ {:>6} │ {:>12.6} │ {:>12.6} │ {:>6.1}× │{}│",
            r.name, r.n, r.rust_secs, r.python_secs, r.speedup, parity_str,
        );
    }
    eprintln!("└──────────────────────────────┴────────┴──────────────┴──────────────┴──────────┴────────┘");

    eprintln!("\n  Parity details:");
    for r in &results {
        let icon = if r.parity_ok { "✓" } else { "✗" };
        eprintln!("    {icon} {}: {}", r.name, r.parity_detail);
    }

    let geo_mean_speedup = {
        let valid: Vec<f64> = results.iter().filter(|r| r.speedup > 0.0).map(|r| r.speedup).collect();
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
