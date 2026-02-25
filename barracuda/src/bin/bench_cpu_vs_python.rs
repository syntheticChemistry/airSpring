//! Benchmark Rust CPU vs Python baseline for airSpring core computations.
//!
//! Measures wall-clock time for:
//! 1. FAO-56 PM ET₀ computation (N station-days)
//! 2. Dual Kc simulation (N-day growing seasons)
//! 3. Cover crop + mulch simulation
//!
//! Outputs timing data in a format directly comparable to Python controls.
//! Run with `--release` for production-representative numbers:
//!
//! ```sh
//! cargo run --release --bin bench_cpu_vs_python
//! ```

use airspring_barracuda::eco::dual_kc::{
    self, DualKcInput, EvaporationLayerState,
};
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use std::time::Instant;

const WARMUP: usize = 5;
const MEASURE: usize = 20;

fn make_station_days(n: usize) -> Vec<DailyEt0Input> {
    (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let day = i as f64;
            let tmax = 30.0 + 5.0 * (day * 0.017).sin();
            let tmin = 15.0 + 3.0 * (day * 0.017).cos();
            DailyEt0Input {
                tmax,
                tmin,
                tmean: None,
                solar_radiation: 18.0 + 4.0 * (day * 0.017).sin(),
                wind_speed_2m: 2.0 + 0.5 * (day * 0.05).sin(),
                actual_vapour_pressure: et::saturation_vapour_pressure(tmin) * 0.6,
                elevation_m: 190.0,
                latitude_deg: 42.5,
                day_of_year: ((i % 365) + 1) as u32,
            }
        })
        .collect()
}

fn make_dual_kc_inputs(n: usize) -> Vec<DualKcInput> {
    (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let day = i as f64;
            DualKcInput {
                et0: 4.0 + 2.0 * (day * 0.017).sin(),
                precipitation: if i % 7 == 0 { 12.0 } else { 0.0 },
                irrigation: 0.0,
            }
        })
        .collect()
}

fn bench<F: Fn()>(label: &str, n: usize, f: F) {
    for _ in 0..WARMUP {
        f();
    }

    let start = Instant::now();
    for _ in 0..MEASURE {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / MEASURE as u32;
    let throughput = n as f64 / per_iter.as_secs_f64();

    println!(
        "  {label:<40} {n:>8} items  {per_iter:>10.2?}/iter  {throughput:>12.0} items/s",
    );
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring CPU Benchmark — Rust vs Python baseline");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("── FAO-56 PM ET₀ computation ──\n");
    for &n in &[100, 1_000, 10_000, 100_000] {
        let data = make_station_days(n);
        bench(&format!("ET₀ ({n} station-days)"), n, || {
            for d in &data {
                std::hint::black_box(et::daily_et0(d));
            }
        });
    }

    println!("\n── Dual Kc simulation ──\n");
    let state = EvaporationLayerState {
        de: 0.0,
        tew: 22.5,
        rew: 9.0,
    };
    for &n in &[30, 180, 365, 3650] {
        let inputs = make_dual_kc_inputs(n);
        bench(&format!("Dual Kc ({n}-day season)"), n, || {
            std::hint::black_box(dual_kc::simulate_dual_kc(
                &inputs, 1.15, 1.20, 0.05, &state,
            ));
        });
    }

    println!("\n── Dual Kc + mulch (no-till) ──\n");
    for &n in &[30, 180, 365, 3650] {
        let inputs = make_dual_kc_inputs(n);
        bench(&format!("Mulched Kc ({n}-day season)"), n, || {
            std::hint::black_box(dual_kc::simulate_dual_kc_mulched(
                &inputs, 0.15, 1.20, 1.0, 0.40, &state,
            ));
        });
    }

    println!("\n── Scaling: ET₀ batch (1M station-days) ──\n");
    let big_data = make_station_days(1_000_000);
    bench("ET₀ (1M station-days)", 1_000_000, || {
        for d in &big_data {
            std::hint::black_box(et::daily_et0(d));
        }
    });

    println!("\n── Scaling: Dual Kc (10-year season) ──\n");
    let big_inputs = make_dual_kc_inputs(3650);
    bench("Dual Kc (3650-day, 10yr)", 3650, || {
        std::hint::black_box(dual_kc::simulate_dual_kc(
            &big_inputs, 1.15, 1.20, 0.05, &state,
        ));
    });

    println!();
    println!("Python baseline reference (Exp 009 control):");
    println!("  63 checks in ~0.8s interpreted = ~80 checks/s");
    println!("  Rust target: 1M+ ET₀/s, 100K+ Kc days/s\n");
    println!("Done.");
}
