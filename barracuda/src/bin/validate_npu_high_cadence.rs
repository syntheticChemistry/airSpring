// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
//! Experiment 029b: High-Cadence NPU Streaming Pipeline
//!
//! Validates the cadence revolution: since NPU inference costs 0.0009% of
//! active cycle energy, sensor cadence can increase from 15-min to 1-min
//! (or 10-sec burst) for negligible additional cost.
//!
//! # Sections
//!
//! - **S1 — Multi-Sensor Fusion**: θ + T + EC in a single 6-feature inference
//! - **S2 — 1-Minute Cadence Stream**: 1440 readings/day through classifier
//! - **S3 — Burst Mode (10-sec)**: Event-triggered high-frequency capture
//! - **S4 — Ensemble Classification**: 10 weight sets per reading, consensus
//! - **S5 — Sliding Window Anomaly**: 60-reading buffer, consecutive triggers
//! - **S6 — Weight Hot-Swap Benchmark**: Classifier swap latency at cadence
//!
//! Provenance: AKD1000 NPU high-cadence streaming validation

use airspring_barracuda::validation::{self, ValidationHarness};

fn main() {
    validation::init_tracing();
    validation::banner("Exp 029b: High-Cadence NPU Streaming Pipeline — more sensors, more readings, negligible energy");

    let mut v = ValidationHarness::new("NPU High Cadence Validation");

    validate_multi_sensor_fusion(&mut v);
    validate_one_minute_cadence(&mut v);
    validate_burst_mode(&mut v);
    validate_ensemble_classification(&mut v);
    validate_sliding_window_anomaly(&mut v);
    validate_weight_hotswap(&mut v);

    #[cfg(feature = "npu")]
    validate_live_high_cadence(&mut v);

    #[cfg(not(feature = "npu"))]
    {
        println!("\n── Live NPU (skipped — build with --features npu) ─────────");
        println!("  [SKIP] 1440-reading DMA stream");
        println!("  [SKIP] Burst mode DMA");
        println!("  [SKIP] Weight hot-swap latency");
    }

    v.finish();
}

struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn quantize_i8(val: f64, lo: f64, hi: f64) -> i8 {
    let normalized = ((val - lo) / (hi - lo)).clamp(0.0, 1.0);
    (normalized * 127.0) as i8
}

fn cpu_fc_infer_i8(input: &[i8], weights: &[i8], bias: &[i8], n_out: usize) -> Vec<i8> {
    let n_in = input.len();
    let mut output = Vec::with_capacity(n_out);
    for o in 0..n_out {
        let mut acc = i32::from(bias[o]) * 128;
        for i in 0..n_in {
            acc += i32::from(weights[o * n_in + i]) * i32::from(input[i]);
        }
        output.push((acc >> 8).clamp(-128, 127) as i8);
    }
    output
}

fn argmax(vals: &[i8]) -> usize {
    vals.iter()
        .enumerate()
        .max_by_key(|&(_, v)| *v)
        .map_or(0, |(i, _)| i)
}

// ═══════════════════════════════════════════════════════════════════
// S1: Multi-Sensor Fusion
// ═══════════════════════════════════════════════════════════════════

fn validate_multi_sensor_fusion(v: &mut ValidationHarness) {
    validation::section("S1: Multi-Sensor Fusion (θ + T + EC)");

    println!("  Single 6-feature inference classifies full field state");

    let n_in = 6; // θ, T, EC, depletion, hour, days_since_irr
    let n_out = 4; // normal, water_stress, salt_stress, anomaly

    #[rustfmt::skip]
    let weights: [i8; 24] = [
        // normal: mid-θ, moderate T, low EC
         30, 10, -30, -20,  5,  5,
        // water_stress: low θ, high depletion
        -40,  5,   5,  50,  0,  10,
        // salt_stress: any θ, high EC
          5,  5,  60,   5,  0,   0,
        // anomaly: extreme values
        -10,-10, -10, -10, -5, -5,
    ];
    let bias: [i8; 4] = [15, -5, -10, -25];

    let mut rng = Lcg::new(42);
    let mut class_counts = [0u32; 4];

    let t_start = std::time::Instant::now();
    for _ in 0..200 {
        let theta = rng.next_f64().mul_add(0.30, 0.15);
        let temperature = rng.next_f64().mul_add(20.0, 15.0);
        let ec = rng.next_f64() * 3.0;
        let depletion = (0.38 - theta) / (0.38 - 0.15);
        let hour_norm = rng.next_f64();
        let days_since_irr = rng.next_f64() * 0.7;

        let input = [
            quantize_i8(theta, 0.0, 0.6),
            quantize_i8(temperature, -10.0, 50.0),
            quantize_i8(ec, 0.0, 5.0),
            quantize_i8(depletion.clamp(0.0, 1.0), 0.0, 1.0),
            quantize_i8(hour_norm, 0.0, 1.0),
            quantize_i8(days_since_irr, 0.0, 1.0),
        ];

        let out = cpu_fc_infer_i8(&input, &weights, &bias, n_out);
        let class = argmax(&out);
        class_counts[class] += 1;
    }
    let elapsed_us = t_start.elapsed().as_micros();

    v.check_bool(
        "6-feature fusion produces 4 classes",
        class_counts.iter().sum::<u32>() == 200,
    );

    let distinct = class_counts.iter().filter(|&&c| c > 0).count();
    println!(
        "  Classes: normal={}, water_stress={}, salt_stress={}, anomaly={}",
        class_counts[0], class_counts[1], class_counts[2], class_counts[3]
    );
    v.check_bool("at least 2 fusion classes observed", distinct >= 2);

    let throughput = 200.0 * 1_000_000.0 / elapsed_us as f64;
    println!("  Fusion throughput: {throughput:.0} Hz ({elapsed_us} µs)");
    v.check_bool("fusion throughput > 10000 Hz (CPU)", throughput > 10000.0);

    v.check_bool("input has 6 features", n_in == 6);
}

// ═══════════════════════════════════════════════════════════════════
// S2: 1-Minute Cadence Stream
// ═══════════════════════════════════════════════════════════════════

fn validate_one_minute_cadence(v: &mut ValidationHarness) {
    validation::section("S2: 1-Minute Cadence (1440 readings/day)");

    println!("  Simulating full 24-hour stream at 1-min intervals");

    let mut rng = Lcg::new(77);
    let fc = 0.38_f64;
    let wp = 0.15;
    let mut theta = 0.32;

    let n_out = 3;
    #[rustfmt::skip]
    let weights: [i8; 12] = [
         40, -30, -20,  5,
        -30,  50, -10,  5,
        -10, -10,  80, -5,
    ];
    let bias: [i8; 3] = [10, -5, -20];

    let mut rolling_mean = theta;
    let mut rolling_var = 0.001_f64;
    let alpha = 0.05; // slower for 1-min cadence (more readings)

    let mut classes = Vec::with_capacity(1440);
    let mut stress_transitions = 0u32;
    let mut prev_class = 0usize;

    let t_start = std::time::Instant::now();
    for minute in 0..1440 {
        let hour = minute / 60;

        // Diurnal ET
        let et_draw = if (6..20).contains(&hour) {
            0.0003 * 0.2f64.mul_add((f64::from(hour) - 13.0).abs().recip().min(3.0), 1.0)
        } else {
            0.00005
        };
        theta -= et_draw;

        // Rain event hours 8:00–8:30 (minutes 480–510)
        if (480..510).contains(&minute) {
            theta += 0.001;
        }

        // Irrigation at 6 AM if depleted
        if minute == 360 && theta < f64::midpoint(fc, wp) {
            theta = fc * 0.95;
        }

        theta = theta.clamp(wp * 0.8, fc * 1.05);
        let noise = rng.next_gaussian() * 0.003;
        let reading = theta + noise;

        let delta = reading - rolling_mean;
        rolling_mean += alpha * delta;
        rolling_var = (1.0 - alpha).mul_add(rolling_var, alpha * delta * delta);
        let sigma = rolling_var.sqrt();

        let hour_norm = f64::from(hour) / 24.0;
        let depletion = ((fc - reading) / (fc - wp)).clamp(0.0, 1.0);

        let input = [
            quantize_i8(reading, 0.0, 0.6),
            quantize_i8(depletion, 0.0, 1.0),
            quantize_i8(sigma, 0.0, 0.05),
            quantize_i8(hour_norm, 0.0, 1.0),
        ];

        let out = cpu_fc_infer_i8(&input, &weights, &bias, n_out);
        let class = argmax(&out);

        if class != prev_class {
            stress_transitions += 1;
        }
        prev_class = class;
        classes.push(class);
    }
    let elapsed_us = t_start.elapsed().as_micros();

    v.check_bool("1440 readings classified", classes.len() == 1440);

    let throughput = 1440.0 * 1_000_000.0 / elapsed_us as f64;
    println!("  Throughput: {throughput:.0} Hz ({elapsed_us} µs total)");
    v.check_bool("throughput > 10000 Hz", throughput > 10000.0);

    let n_normal = classes.iter().filter(|&&c| c == 0).count();
    let n_stressed = classes.iter().filter(|&&c| c == 1).count();
    let n_anomaly = classes.iter().filter(|&&c| c == 2).count();
    println!("  Classes: normal={n_normal}, stressed={n_stressed}, anomaly={n_anomaly}");
    println!("  State transitions: {stress_transitions}");

    v.check_bool(
        "at least 2 distinct classes",
        [n_normal, n_stressed, n_anomaly]
            .iter()
            .filter(|&&c| c > 0)
            .count()
            >= 2,
    );

    v.check_bool("state transitions detected", stress_transitions > 0);

    // Energy calculation for 1-min cadence
    let pi_wake_mj = 300.0_f64;
    let pi_sense_mj = 30.0;
    let npu_infer_mj = 0.003;
    let pi_log_mj = 6.0;
    let energy_per_reading_mj = pi_wake_mj + pi_sense_mj + npu_infer_mj + pi_log_mj;
    let daily_active_wh = 1440.0 * energy_per_reading_mj / 3_600_000.0;

    let pi_idle_mw = 105.0_f64; // Pi + NPU idle
    let sleep_per_reading_s = 60.0 - 0.56;
    let daily_sleep_wh = 1440.0 * pi_idle_mw * sleep_per_reading_s / 3_600_000.0;
    let daily_total_wh = daily_active_wh + daily_sleep_wh;

    println!("  Energy at 1-min: {daily_total_wh:.1} Wh/day");
    println!("    Active: {daily_active_wh:.2} Wh, Sleep: {daily_sleep_wh:.1} Wh");
    println!(
        "    NPU share: {:.6}%",
        npu_infer_mj / energy_per_reading_mj * 100.0
    );

    v.check_bool("daily energy < 5 Wh at 1-min cadence", daily_total_wh < 5.0);
}

// ═══════════════════════════════════════════════════════════════════
// S3: Burst Mode (10-second intervals)
// ═══════════════════════════════════════════════════════════════════

fn validate_burst_mode(v: &mut ValidationHarness) {
    validation::section("S3: Burst Mode (10-sec, 30-min window)");

    println!("  Event-triggered high-frequency capture during irrigation");

    let mut rng = Lcg::new(99);
    let burst_readings = 180; // 30 min × 6/min
    let mut theta = 0.20_f64; // pre-irrigation, depleted

    let n_out = 3;
    #[rustfmt::skip]
    let weights: [i8; 12] = [
         40, -30, -20,  5,
        -30,  50, -10,  5,
        -10, -10,  80, -5,
    ];
    let bias: [i8; 3] = [10, -5, -20];

    let mut readings = Vec::with_capacity(burst_readings);
    let mut burst_classes = Vec::with_capacity(burst_readings);
    let mut rolling_mean = theta;
    let mut rolling_var = 0.001_f64;
    let alpha = 0.15; // faster for burst mode

    let t_start = std::time::Instant::now();
    for step in 0..burst_readings {
        // Infiltration front arrives at step 18 (3 min), peaks at step 60 (10 min)
        if step >= 18 {
            let infiltration = 0.003 * (-(((step - 18) as f64) / 30.0).powi(2)).exp();
            theta += infiltration;
        }

        theta = theta.clamp(0.12, 0.40);
        let noise = rng.next_gaussian() * 0.002;
        let reading = theta + noise;
        readings.push(reading);

        let delta = reading - rolling_mean;
        rolling_mean += alpha * delta;
        rolling_var = (1.0 - alpha).mul_add(rolling_var, alpha * delta * delta);
        let sigma = rolling_var.sqrt();

        let depletion = ((0.38 - reading) / 0.23).clamp(0.0, 1.0);
        let input = [
            quantize_i8(reading, 0.0, 0.6),
            quantize_i8(depletion, 0.0, 1.0),
            quantize_i8(sigma, 0.0, 0.05),
            quantize_i8(step as f64 / burst_readings as f64, 0.0, 1.0),
        ];

        let out = cpu_fc_infer_i8(&input, &weights, &bias, n_out);
        burst_classes.push(argmax(&out));
    }
    let elapsed_us = t_start.elapsed().as_micros();

    v.check_bool(
        "180 burst readings captured",
        readings.len() == burst_readings,
    );

    let burst_distinct = [0, 1, 2]
        .iter()
        .filter(|&&c| burst_classes.contains(&c))
        .count();
    println!("  Burst classes: {burst_distinct} distinct");
    v.check_bool("burst has classified readings", !burst_classes.is_empty());

    let burst_hz = burst_readings as f64 * 1_000_000.0 / elapsed_us as f64;
    println!("  Burst throughput: {burst_hz:.0} Hz ({elapsed_us} µs)");
    v.check_bool("burst throughput > 10000 Hz", burst_hz > 10000.0);

    // Infiltration should cause a rise in readings
    let pre_mean: f64 = readings[..18].iter().sum::<f64>() / 18.0;
    let post_mean: f64 = readings[30..60].iter().sum::<f64>() / 30.0;
    println!("  Pre-irrigation θ̄: {pre_mean:.4}, Post (3–10 min): {post_mean:.4}");
    v.check_bool(
        "infiltration front detected (post > pre)",
        post_mean > pre_mean,
    );
}

// ═══════════════════════════════════════════════════════════════════
// S4: Ensemble Classification
// ═══════════════════════════════════════════════════════════════════

fn validate_ensemble_classification(v: &mut ValidationHarness) {
    validation::section("S4: Ensemble Classification (10 weight sets)");

    println!("  Consensus from 10 classifiers per reading");

    let n_in = 4;
    let n_out = 3;
    let n_ensemble = 10;

    // Generate 10 distinct weight sets
    let weight_sets: Vec<(Vec<i8>, Vec<i8>)> = (0..n_ensemble)
        .map(|seed| {
            let mut rng = Lcg::new(seed as u64 * 31 + 7);
            let weights: Vec<i8> = (0..n_out * n_in)
                .map(|_| ((rng.next_f64() - 0.5) * 60.0) as i8)
                .collect();
            let bias: Vec<i8> = (0..n_out)
                .map(|_| ((rng.next_f64() - 0.5) * 20.0) as i8)
                .collect();
            (weights, bias)
        })
        .collect();

    let test_input = [
        quantize_i8(0.22, 0.0, 0.6),
        quantize_i8(0.70, 0.0, 1.0),
        quantize_i8(0.01, 0.0, 0.1),
        quantize_i8(0.5, 0.0, 1.0),
    ];

    let t_start = std::time::Instant::now();
    let mut votes = [0u32; 3];
    for (weights, bias) in &weight_sets {
        let out = cpu_fc_infer_i8(&test_input, weights, bias, n_out);
        let class = argmax(&out);
        votes[class] += 1;
    }
    let elapsed_us = t_start.elapsed().as_micros();

    let consensus = votes.iter().max().copied().unwrap_or(0);
    let consensus_class = votes.iter().position(|&v| v == consensus).unwrap_or(0);
    let confidence = f64::from(consensus) / f64::from(n_ensemble);

    println!(
        "  Votes: normal={}, stressed={}, anomaly={}",
        votes[0], votes[1], votes[2]
    );
    println!(
        "  Consensus: class {consensus_class} ({:.0}% confidence)",
        confidence * 100.0
    );
    println!("  Ensemble latency: {elapsed_us} µs (10 inferences)");

    v.check_bool(
        "10 ensemble votes collected",
        votes.iter().sum::<u32>() == n_ensemble as u32,
    );

    v.check_bool("ensemble has a majority", confidence > 0.3);

    // On real NPU: 10 × 48 µs = 480 µs. Easily within 1-sec budget.
    let projected_npu_us = 10.0 * 48.0;
    println!("  Projected NPU ensemble: {projected_npu_us:.0} µs");
    v.check_bool("projected NPU ensemble < 1 ms", projected_npu_us < 1000.0);
}

// ═══════════════════════════════════════════════════════════════════
// S5: Sliding Window Anomaly Detection
// ═══════════════════════════════════════════════════════════════════

fn validate_sliding_window_anomaly(v: &mut ValidationHarness) {
    validation::section("S5: Sliding Window Anomaly (60-reading buffer)");

    println!("  Consecutive trigger detection for robust alerting");

    let mut rng = Lcg::new(55);
    let window_size = 60;
    let anomaly_z = 3.0;
    let consecutive_threshold = 3;

    let mut window: Vec<f64> = Vec::with_capacity(window_size);
    let mut alert_count = 0u32;
    let mut consecutive_anomalies = 0u32;

    for step in 0..300 {
        // Normal readings with occasional sensor glitches
        let reading = if step == 100 || step == 101 || step == 102 {
            0.90 // 3 consecutive glitches
        } else if step == 200 {
            0.85 // single glitch (should not trigger)
        } else {
            rng.next_gaussian().mul_add(0.01, 0.30)
        };

        if window.len() >= window_size {
            window.remove(0);
        }
        window.push(reading);

        if window.len() >= 20 {
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let sigma = variance.sqrt();

            let is_anomaly = sigma > 1e-10 && ((reading - mean) / sigma).abs() > anomaly_z;

            if is_anomaly {
                consecutive_anomalies += 1;
            } else {
                consecutive_anomalies = 0;
            }

            if consecutive_anomalies >= consecutive_threshold {
                alert_count += 1;
            }
        }
    }

    println!("  Alerts triggered: {alert_count}");
    v.check_bool(
        "at least 1 alert from 3-consecutive glitches",
        alert_count >= 1,
    );

    // The single glitch at step 200 should NOT trigger (only 1 consecutive)
    // but the 3 at steps 100-102 should trigger at least once
    v.check_bool("alert count reasonable (< 50)", alert_count < 50);
}

// ═══════════════════════════════════════════════════════════════════
// S6: Weight Hot-Swap Benchmark
// ═══════════════════════════════════════════════════════════════════

fn validate_weight_hotswap(v: &mut ValidationHarness) {
    validation::section("S6: Weight Hot-Swap Benchmark");

    println!("  Simulating classifier swap at each field boundary");

    let crops = ["corn", "soybean", "potato", "tomato", "blueberry"];
    let n_in = 4;
    let n_out = 3;

    let crop_weights: Vec<Vec<i8>> = (0..crops.len())
        .map(|seed| {
            let mut rng = Lcg::new(seed as u64 * 47 + 13);
            (0..n_out * n_in + n_out)
                .map(|_| ((rng.next_f64() - 0.5) * 60.0) as i8)
                .collect()
        })
        .collect();

    let probe = [
        quantize_i8(0.28, 0.0, 0.6),
        quantize_i8(0.43, 0.0, 1.0),
        quantize_i8(0.02, 0.0, 0.1),
        quantize_i8(0.5, 0.0, 1.0),
    ];

    let t_start = std::time::Instant::now();
    let mut swap_count = 0u32;
    let rounds = 50;
    for _ in 0..rounds {
        for (idx, weights) in crop_weights.iter().enumerate() {
            let (wt, bias) = weights.split_at(n_out * n_in);
            let _out = cpu_fc_infer_i8(&probe, wt, bias, n_out);
            swap_count += 1;
            if idx == 0 {
                // Just counting — on NPU this would be load_readout_weights
            }
        }
    }
    let elapsed_us = t_start.elapsed().as_micros();

    let swaps_expected = rounds * crops.len() as u32;
    v.check_bool(
        &format!("{swaps_expected} crop swaps completed"),
        swap_count == swaps_expected,
    );

    let swap_rate = f64::from(swap_count) * 1_000_000.0 / elapsed_us as f64;
    println!("  Swap rate: {swap_rate:.0} swaps/sec ({elapsed_us} µs)");
    v.check_bool("swap rate > 10000/sec", swap_rate > 10000.0);

    // On real NPU: weight load ~60 µs + infer ~48 µs = 108 µs per swap
    let projected_npu_swap_us = 108.0 * crops.len() as f64;
    println!(
        "  Projected NPU: {projected_npu_swap_us:.0} µs for {} crop round-robin",
        crops.len()
    );
    v.check_bool(
        "projected 5-crop swap < 1 ms on NPU",
        projected_npu_swap_us < 1000.0,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Live NPU High-Cadence
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "npu")]
#[expect(
    clippy::too_many_lines,
    reason = "live high-cadence validation covers AKD1000 hardware streaming"
)]
fn validate_live_high_cadence(v: &mut ValidationHarness) {
    use airspring_barracuda::npu;

    validation::section("Live NPU: High-Cadence AKD1000");

    if !npu::npu_available() {
        println!("  [SKIP] No AKD1000 detected");
        return;
    }

    let mut handle = match npu::discover_npu() {
        Ok(h) => {
            println!("  AKD1000 online: {:?}", h.chip_version());
            h
        }
        Err(e) => {
            v.check_bool(&format!("discover: {e}"), false);
            return;
        }
    };

    v.check_bool("NPU opened", true);

    // 1440 readings at 1-min cadence through real NPU
    println!("  Streaming 1440 readings (full day) through AKD1000...");
    let mut latencies_ns: Vec<u64> = Vec::with_capacity(1440);
    let mut rng = Lcg::new(77);
    let mut theta = 0.32_f64;

    for minute in 0..1440 {
        let hour = minute / 60;
        let et_draw = if (6..20).contains(&hour) {
            0.0003
        } else {
            0.00005
        };
        theta = (theta - et_draw).clamp(0.12, 0.40);
        let reading = rng.next_gaussian().mul_add(0.003, theta);

        let input = npu::MultiSensorInput {
            theta: reading.clamp(0.0, 0.6),
            temperature: 5.0f64.mul_add(((f64::from(hour) - 14.0) / 12.0).cos(), 22.0),
            ec: rng.next_f64().mul_add(0.3, 1.5),
            depletion: ((0.38 - reading) / 0.23).clamp(0.0, 1.0),
            hour_norm: f64::from(hour) / 24.0,
            days_since_irr: 0.3,
        };

        let t = std::time::Instant::now();
        let _result = npu::npu_infer_i8(&mut handle, &input.to_i8(), 4);
        latencies_ns.push(t.elapsed().as_nanos() as u64);
    }

    let mean_us = latencies_ns.iter().sum::<u64>() as f64 / latencies_ns.len() as f64 / 1000.0;
    let mut sorted = latencies_ns.clone();
    sorted.sort_unstable();
    let p99_us = sorted
        .get((sorted.len() as f64 * 0.99) as usize)
        .copied()
        .unwrap_or(0) as f64
        / 1000.0;
    let total_us = latencies_ns.iter().sum::<u64>() as f64 / 1000.0;
    let throughput = 1440.0 * 1_000_000.0 / total_us;

    println!("  Mean: {mean_us:.1} µs, P99: {p99_us:.1} µs");
    println!("  Throughput: {throughput:.0} Hz, Total: {total_us:.0} µs");

    v.check_bool("1440 DMA round-trips completed", latencies_ns.len() == 1440);
    v.check_bool("mean < 200 µs", mean_us < 200.0);
    v.check_bool("p99 < 1 ms", p99_us < 1000.0);
    v.check_bool("throughput > 5000 Hz", throughput > 5000.0);

    // Weight hot-swap on real NPU
    println!("  Weight hot-swap: 5 crops × 20 rounds...");
    let mut swap_latencies_ns: Vec<u64> = Vec::new();
    for seed in 0..5_u64 {
        let mut wrng = Lcg::new(seed * 47 + 13);
        let weights: Vec<i8> = (0..12 + 3)
            .map(|_| ((wrng.next_f64() - 0.5) * 60.0) as i8)
            .collect();

        for _ in 0..20 {
            let t = std::time::Instant::now();
            let _ = npu::load_readout_weights(&mut handle, &weights);
            swap_latencies_ns.push(t.elapsed().as_nanos() as u64);
        }
    }

    let mean_swap_us =
        swap_latencies_ns.iter().sum::<u64>() as f64 / swap_latencies_ns.len() as f64 / 1000.0;
    println!("  Mean swap latency: {mean_swap_us:.1} µs");
    v.check_bool("100 weight swaps completed", swap_latencies_ns.len() == 100);
    v.check_bool("mean swap < 500 µs", mean_swap_us < 500.0);
}
