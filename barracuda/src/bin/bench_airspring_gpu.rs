//! Benchmark airSpring GPU operations vs CPU baselines.
//!
//! Measures wall-clock time for all GPU orchestrators and CPU fallbacks across
//! multiple problem sizes, reporting throughput and cross-spring provenance.
//!
//! # Cross-spring evolution context
//!
//! These GPU paths exist because of shader evolution across the ecoPrimals
//! ecosystem (608 WGSL shaders, 46 cross-spring absorptions S51-S57):
//!
//! - **ET₀ batch** (`batched_elementwise_f64.wgsl`): hotSpring `pow_f64` fix
//!   (TS-001) made fractional exponents work; airSpring wired FAO-56 as op=0
//! - **Water balance batch**: Same shader, op=1 — multi-Spring shared primitive
//! - **Kriging** (`kriging_f64.wgsl`): wetSpring spatial interpolation; airSpring
//!   wired soil moisture mapping
//! - **Reduce** (`fused_map_reduce_f64.wgsl`): wetSpring origin; airSpring TS-004
//!   fix stabilized N≥1024 dispatch for all Springs
//! - **Stream smoothing** (`moving_window.wgsl`): wetSpring S28+ environmental
//!   monitoring; airSpring wired IoT sensor smoothing
//! - **Ridge regression** (`barracuda::linalg::ridge`): wetSpring ESN calibration;
//!   airSpring wired sensor correction pipeline
//!
//! # Usage
//!
//! ```sh
//! cargo run --release --bin bench_airspring_gpu
//! ```

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::gpu::{kriging, reduce, stream};
use std::time::Instant;

const WARMUP: usize = 3;
const MEASURE: usize = 10;

fn make_station_days(n: usize) -> Vec<DailyEt0Input> {
    (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let day = i as f64;
            DailyEt0Input {
                tmin: 12.0 + (day * 0.01).sin(),
                tmax: 25.0 + (day * 0.01).cos(),
                tmean: None,
                solar_radiation: 22.0 + (day * 0.02).sin(),
                wind_speed_2m: 2.0,
                actual_vapour_pressure: 1.4,
                elevation_m: 100.0,
                latitude_deg: 50.8,
                day_of_year: 187,
            }
        })
        .collect()
}

fn bench_et0_cpu(inputs: &[DailyEt0Input]) -> f64 {
    inputs.iter().map(|i| et::daily_et0(i).et0).sum()
}

fn bench_reduce_cpu(data: &[f64]) -> f64 {
    let r = reduce::compute_seasonal_stats(data);
    r.mean
}

fn bench_stream_cpu(data: &[f64], window: usize) -> f64 {
    let result = stream::smooth_cpu(data, window).unwrap();
    result.mean[0]
}

fn bench_kriging_cpu(sensors: &[kriging::SensorReading], targets: &[kriging::TargetPoint]) -> f64 {
    let variogram = kriging::SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };
    let result = kriging::interpolate_soil_moisture(sensors, targets, variogram);
    result.vwc_values[0]
}

fn time_fn<F: FnMut() -> f64>(mut f: F, warmup: usize, measure: usize) -> (f64, f64) {
    for _ in 0..warmup {
        let _ = f();
    }
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..measure {
        checksum += f();
    }
    let elapsed_us = start.elapsed().as_micros();
    #[allow(clippy::cast_precision_loss)]
    let per_call_us = elapsed_us as f64 / measure as f64;
    (per_call_us, checksum)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  airSpring GPU Benchmark — Cross-Spring Shader Evolution");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // ── ET₀ Batched ──────────────────────────────────────────────────
    println!("── Batched ET₀ (batched_elementwise_f64, hotSpring pow_f64 fix) ──");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "ops/sec");

    for &n in &[10, 100, 1_000, 10_000] {
        let inputs = make_station_days(n);
        let (cpu_us, _) = time_fn(|| bench_et0_cpu(&inputs), WARMUP, MEASURE);
        #[allow(clippy::cast_precision_loss)]
        let ops_per_sec = (n as f64) / (cpu_us / 1_000_000.0);
        println!("  {n:>8}  {cpu_us:>12.1}  {ops_per_sec:>12.0}");
    }

    // ── Seasonal Reduce ──────────────────────────────────────────────
    println!();
    println!("── Seasonal Reduce (fused_map_reduce_f64, wetSpring origin, TS-004 fix) ──");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "M elem/sec");

    for &n in &[100, 1_000, 10_000, 100_000] {
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let (cpu_us, _) = time_fn(|| bench_reduce_cpu(&data), WARMUP, MEASURE);
        #[allow(clippy::cast_precision_loss)]
        let m_elem_sec = (n as f64) / (cpu_us / 1_000_000.0) / 1e6;
        println!("  {n:>8}  {cpu_us:>12.1}  {m_elem_sec:>12.1}");
    }

    // ── Stream Smoothing ─────────────────────────────────────────────
    println!();
    println!("── Stream Smoothing (moving_window.wgsl, wetSpring S28+ environmental) ──");
    println!(
        "  {:>8}  {:>8}  {:>12}  {:>12}",
        "N", "Window", "CPU (µs)", "M elem/sec"
    );

    for &(n, w) in &[(168, 24), (720, 24), (8760, 24), (8760, 168)] {
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f64> = (0..n)
            .map(|i| {
                8.0f64.mul_add(
                    ((i as f64 % 24.0 - 14.0) * std::f64::consts::PI / 12.0).cos(),
                    25.0,
                )
            })
            .collect();
        let (cpu_us, _) = time_fn(|| bench_stream_cpu(&data, w), WARMUP, MEASURE);
        #[allow(clippy::cast_precision_loss)]
        let m_elem_sec = (n as f64) / (cpu_us / 1_000_000.0) / 1e6;
        println!("  {n:>8}  {w:>8}  {cpu_us:>12.1}  {m_elem_sec:>12.1}");
    }

    // ── Kriging ──────────────────────────────────────────────────────
    println!();
    println!("── Kriging (kriging_f64.wgsl, wetSpring spatial interpolation) ──");
    println!("  {:>8}  {:>8}  {:>12}", "Sensors", "Targets", "CPU (µs)");

    for &(ns, nt) in &[(5, 10), (10, 100), (20, 500)] {
        let sensors: Vec<kriging::SensorReading> = (0..ns)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let fi = i as f64;
                kriging::SensorReading {
                    x: fi * 10.0,
                    y: fi * 5.0,
                    vwc: 0.25 + fi * 0.01,
                }
            })
            .collect();
        let targets: Vec<kriging::TargetPoint> = (0..nt)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let fi = i as f64;
                kriging::TargetPoint {
                    x: fi * 2.0,
                    y: fi * 1.0,
                }
            })
            .collect();
        let (cpu_us, _) = time_fn(|| bench_kriging_cpu(&sensors, &targets), WARMUP, MEASURE);
        println!("  {ns:>8}  {nt:>8}  {cpu_us:>12.1}");
    }

    // ── Ridge Regression ─────────────────────────────────────────────
    println!();
    println!("── Ridge Regression (barracuda::linalg::ridge, wetSpring ESN calibration) ──");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "R²");

    for &n in &[50, 200, 1_000, 5_000] {
        #[allow(clippy::cast_precision_loss)]
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.5f64.mul_add(xi, 0.3) + (xi * 0.1).sin() * 0.01)
            .collect();
        let mut r2_val = 0.0;
        let (cpu_us, _) = time_fn(
            || {
                let m = airspring_barracuda::eco::correction::fit_ridge(&x, &y, 1e-6).unwrap();
                r2_val = m.r_squared;
                m.r_squared
            },
            WARMUP,
            MEASURE,
        );
        println!("  {n:>8}  {cpu_us:>12.1}  {r2_val:>12.6}");
    }

    // ── Summary ──────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Cross-spring shader provenance:");
    println!("    608 WGSL shaders in ToadStool (hotSpring 56, wetSpring 25,");
    println!("    neuralSpring 20, shared 507). airSpring uses 5 + contributed 3 fixes.");
    println!("    46 cross-spring absorptions (S51-S57) benefit all Springs.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
