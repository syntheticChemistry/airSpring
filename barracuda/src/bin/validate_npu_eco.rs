// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validates NPU integration for ecological workloads (Exp 028).
//!
//! Benchmark: analytical (int8 quantization round-trip fidelity)
//! Hardware: `BrainChip` `AKD1000` `PCIe` (optional, skipped gracefully)
//!
//! Tests:
//! 1. int8 quantization round-trip fidelity
//! 2. Crop stress classifier feature quantization
//! 3. Irrigation decision feature quantization
//! 4. Sensor anomaly feature quantization
//! 5. Live NPU DMA inference (if hardware present)

use airspring_barracuda::validation::{self, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/npu_eco/benchmark_npu_eco.json");

fn validate_benchmark_provenance(v: &mut ValidationHarness) {
    validation::section("Benchmark Provenance");

    let bench =
        validation::parse_benchmark_json(BENCHMARK_JSON).expect("benchmark JSON must parse");

    let experiments = validation::json_object_opt(&bench, &["experiments"]);
    v.check_bool(
        "experiments has 3 entries",
        experiments.is_some_and(|e| e.len() == 3),
    );

    let binary = validation::json_str_opt(&bench, &["experiments", "028", "binary"]);
    v.check_bool(
        "028 binary == validate_npu_eco",
        binary == Some("validate_npu_eco"),
    );

    let name = validation::json_str_opt(&bench, &["experiments", "028", "name"]);
    v.check_bool(
        "028 name contains NPU Edge Inference",
        name.is_some_and(|n| n.contains("NPU Edge Inference")),
    );

    let method = validation::json_str_opt(&bench, &["_provenance", "method"]);
    v.check_bool(
        "provenance method non-empty",
        method.is_some_and(|s| !s.is_empty()),
    );
}

fn quantize_i8(val: f64, lo: f64, hi: f64) -> i8 {
    let normalized = ((val - lo) / (hi - lo)).clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation)]
    let result = (normalized * 127.0) as i8;
    result
}

fn dequantize_i8(val: i8, lo: f64, hi: f64) -> f64 {
    let normalized = f64::from(val) / 127.0;
    normalized.mul_add(hi - lo, lo)
}

fn validate_quantization(v: &mut ValidationHarness) {
    validation::section("int8 Quantization Round-Trip");

    let test_cases: &[(f64, f64, f64, f64)] = &[
        (0.5, 0.0, 1.0, 0.01),
        (7.5, 5.0, 10.0, 0.1),
        (0.0, 0.0, 1.0, 0.01),
        (1.0, 0.0, 1.0, 0.01),
        (150.0, 0.0, 300.0, 3.0),
        (4.2, 0.0, 15.0, 0.2),
    ];

    for &(val, lo, hi, tol) in test_cases {
        let q = quantize_i8(val, lo, hi);
        let deq = dequantize_i8(q, lo, hi);
        v.check_abs(&format!("roundtrip {val:.1} in [{lo},{hi}]"), deq, val, tol);
    }

    let q_lo = quantize_i8(-5.0, 0.0, 10.0);
    v.check_bool("clamp below → 0", q_lo == 0);

    let q_hi = quantize_i8(20.0, 0.0, 10.0);
    v.check_bool("clamp above → 127", q_hi == 127);

    v.check_bool(
        "monotonic: q(0.3) < q(0.7)",
        quantize_i8(0.3, 0.0, 1.0) < quantize_i8(0.7, 0.0, 1.0),
    );
}

fn validate_crop_stress_features(v: &mut ValidationHarness) {
    validation::section("Crop Stress Classifier Features");

    let healthy = [0.2_f64, 0.95, 0.35, 0.95];
    let stressed = [0.85_f64, 0.3, 0.10, 0.15];

    let h_q: Vec<i8> = healthy
        .iter()
        .zip([(0.0, 1.0), (0.0, 1.5), (0.0, 0.6), (0.0, 1.0)])
        .map(|(&val, (lo, hi))| quantize_i8(val, lo, hi))
        .collect();

    let s_q: Vec<i8> = stressed
        .iter()
        .zip([(0.0, 1.0), (0.0, 1.5), (0.0, 0.6), (0.0, 1.0)])
        .map(|(&val, (lo, hi))| quantize_i8(val, lo, hi))
        .collect();

    v.check_bool("healthy: 4 features", h_q.len() == 4);
    v.check_bool("stressed: 4 features", s_q.len() == 4);
    v.check_bool("stressed depletion > healthy depletion", s_q[0] > h_q[0]);
    v.check_bool("stressed ET ratio < healthy ET ratio", s_q[1] < h_q[1]);
    v.check_bool("stressed θ < healthy θ", s_q[2] < h_q[2]);
    v.check_bool("stressed Ks < healthy Ks", s_q[3] < h_q[3]);
}

fn validate_irrigation_features(v: &mut ValidationHarness) {
    validation::section("Irrigation Decision Features");

    let ranges: &[(f64, f64)] = &[
        (0.0, 15.0),
        (0.0, 0.6),
        (0.0, 300.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ];

    let irrigate_now = [8.0_f64, 0.15, 200.0, 0.5, 0.2, 0.05];
    let hold = [3.0_f64, 0.35, 200.0, 0.5, 0.85, 0.7];

    let irr_q: Vec<i8> = irrigate_now
        .iter()
        .zip(ranges)
        .map(|(&val, &(lo, hi))| quantize_i8(val, lo, hi))
        .collect();
    let hold_q: Vec<i8> = hold
        .iter()
        .zip(ranges)
        .map(|(&val, &(lo, hi))| quantize_i8(val, lo, hi))
        .collect();

    v.check_bool("irrigate: 6 features", irr_q.len() == 6);
    v.check_bool("hold: 6 features", hold_q.len() == 6);
    v.check_bool("irrigate has higher ET₀ quantized", irr_q[0] > hold_q[0]);
    v.check_bool("irrigate has lower θ quantized", irr_q[1] < hold_q[1]);
    v.check_bool("irrigate has lower Ks quantized", irr_q[4] < hold_q[4]);
    v.check_bool("hold has higher rain probability", hold_q[5] > irr_q[5]);
}

fn validate_sensor_anomaly_features(v: &mut ValidationHarness) {
    validation::section("Sensor Anomaly Features");

    let normal_reading = 4.0_f64;
    let normal_mean = 4.2;
    let normal_sigma = 0.5;

    let anomaly_reading = 9.5_f64;
    let anomaly_mean = 4.2;
    let anomaly_sigma = 0.5;

    let scale = (0.0, 15.0);

    let n_q = [
        quantize_i8(normal_reading, scale.0, scale.1),
        quantize_i8(normal_mean, scale.0, scale.1),
        quantize_i8(normal_sigma, 0.0, scale.1 - scale.0),
    ];
    let a_q = [
        quantize_i8(anomaly_reading, scale.0, scale.1),
        quantize_i8(anomaly_mean, scale.0, scale.1),
        quantize_i8(anomaly_sigma, 0.0, scale.1 - scale.0),
    ];

    v.check_bool("normal: 3 features", n_q.len() == 3);
    v.check_bool("anomaly: 3 features", a_q.len() == 3);
    v.check_bool("anomaly reading quantized > normal", a_q[0] > n_q[0]);
    v.check_bool("same mean quantizes identically", n_q[1] == a_q[1]);
    v.check_bool("same σ quantizes identically", n_q[2] == a_q[2]);

    let z_normal = (normal_reading - normal_mean) / normal_sigma;
    let z_anomaly = (anomaly_reading - anomaly_mean) / anomaly_sigma;
    v.check_bool(
        "z-score: normal < anomaly",
        z_normal.abs() < z_anomaly.abs(),
    );
    v.check_bool("z-score anomaly > 3σ", z_anomaly.abs() > 3.0);
}

#[cfg(feature = "npu")]
fn validate_npu_hardware(v: &mut ValidationHarness) {
    use airspring_barracuda::npu;

    validation::section("NPU Hardware Discovery");

    let available = npu::npu_available();
    println!("  NPU available: {available}");

    if !available {
        println!("  [SKIP] No AKD1000 detected — skipping hardware tests");
        return;
    }

    v.check_bool("NPU detected", available);

    match npu::npu_summary() {
        Ok(summary) => {
            println!(
                "  Chip: {}, NPUs: {}, SRAM: {} MB, PCIe: {:.1} GB/s",
                summary.chip, summary.npu_count, summary.memory_mb, summary.bandwidth_gbps
            );
            v.check_bool("NPU count > 0", summary.npu_count > 0);
            v.check_bool("SRAM > 0 MB", summary.memory_mb > 0);
            v.check_bool("bandwidth > 0", summary.bandwidth_gbps > 0.0);
        }
        Err(e) => {
            v.check_bool(&format!("npu_summary failed: {e}"), false);
        }
    }

    match npu::discover_npu() {
        Ok(mut handle) => {
            println!("  Device opened: {:?}", handle.chip_version());

            let stress_input = npu::CropStressInput {
                depletion: 0.7,
                et_ratio: 0.5,
                theta: 0.2,
                ks: 0.3,
            };
            let input_i8 = stress_input.to_i8();

            match npu::npu_infer_i8(&mut handle, &input_i8, 2) {
                Ok(result) => {
                    println!(
                        "  Crop stress DMA: write {}ns, read {}ns → class {}",
                        result.write_ns, result.read_ns, result.class
                    );
                    v.check_bool(
                        "DMA round-trip < 10ms",
                        result.write_ns + result.read_ns < 10_000_000,
                    );
                    v.check_bool("class in [0,1]", result.class <= 1);
                }
                Err(e) => {
                    v.check_bool(&format!("NPU crop stress inference: {e}"), false);
                }
            }

            let irr_input = npu::IrrigationInput {
                forecast_et0: 6.0,
                theta: 0.2,
                taw: 180.0,
                stage: 0.5,
                ks: 0.4,
                rain_prob: 0.1,
            };
            let irr_i8 = irr_input.to_i8();

            match npu::npu_infer_i8(&mut handle, &irr_i8, 3) {
                Ok(result) => {
                    println!(
                        "  Irrigation DMA: write {}ns, read {}ns → class {}",
                        result.write_ns, result.read_ns, result.class
                    );
                    v.check_bool("class in [0,1,2]", result.class <= 2);
                }
                Err(e) => {
                    v.check_bool(&format!("NPU irrigation inference: {e}"), false);
                }
            }
        }
        Err(e) => {
            v.check_bool(&format!("discover_npu failed: {e}"), false);
        }
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 028: NPU Edge Inference for Agriculture");

    let mut v = ValidationHarness::new("NPU Eco Validation");

    validate_quantization(&mut v);
    validate_crop_stress_features(&mut v);
    validate_irrigation_features(&mut v);
    validate_sensor_anomaly_features(&mut v);

    validate_benchmark_provenance(&mut v);

    #[cfg(feature = "npu")]
    validate_npu_hardware(&mut v);

    #[cfg(not(feature = "npu"))]
    {
        println!("\n── NPU Hardware (skipped — build without --features npu) ───");
        println!("  [SKIP] NPU discovery");
        println!("  [SKIP] NPU DMA inference");
    }

    v.finish();
}
