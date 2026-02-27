# NPU-Accelerated Agricultural IoT (LOCOMOS → Edge Sovereign)

**Date:** February 26, 2026
**Status:** **Validated** — Exp 029: 32/32 checks (CPU + live AKD1000), Exp 028: 35/35 + 21/21
**Data Tier:** 0 (no external data — hardware validation + analytical models)
**Dependencies:** `barracuda::npu` (feature-gated), `airspring-forge`, `akida-driver`
**Cross-Spring:** wetSpring (NPU driver + ESN readout) × airSpring (sensor pipeline + water balance)

**Hardware:** BrainChip AKD1000 PCIe — 80 NPs, 10 MB SRAM, live on Eastgate.

---

## What This Is

Demonstrate that the BrainChip AKD1000 neuromorphic processor transforms
agricultural IoT from cloud-dependent monitoring into edge-sovereign intelligence.
Validated on real hardware, not simulated.

The core finding: **NPU inference adds 0.0009% of active cycle energy** — effectively
zero. This unlocks a radical design shift: instead of conserving energy by reducing
sensor cadence (Dong 2024: 15-min intervals), you can increase cadence to 1-minute
or even 10-second intervals for free. The NPU classifies every reading on-device.
No cloud. No connectivity. No latency.

## Why It Matters for LOCOMOS

Dong's LOCOMOS (Low-Cost Monitoring System) pattern:

```
Current:  sensor → Pi → WiFi → cloud → decision → WiFi → actuator
          Latency: seconds to minutes. Requires connectivity. Monthly cost.

With NPU: sensor → Pi + AKD1000 → decision → actuator
          Latency: 50 ms. No connectivity. One-time $99.
```

But the bigger impact is cadence. At 15-min intervals, LOCOMOS collects 96
readings/day. The power budget is dominated by Pi wake/sleep — not inference.
Since NPU inference costs effectively nothing:

| Cadence | Readings/day | NPU Energy/day | Pi Energy/day | Enables |
|---------|:------------:|:--------------:|:-------------:|---------|
| 15 min (current) | 96 | 0.3 µWh | 2.53 Wh | Basic monitoring |
| 5 min | 288 | 0.9 µWh | 7.59 Wh | Irrigation pulse detection |
| 1 min | 1,440 | 4.3 µWh | 37.9 Wh | Real-time stress tracking |
| 10 sec (burst) | 8,640 | 25.9 µWh | — (stay awake) | Event-triggered transients |

At 5-min cadence, a small 20Wh solar setup still covers it. At 1-min cadence,
you need a larger panel (50W) or grid power — but the NPU's share is still
negligible. The real cost is always the Pi, never the NPU.

The 10-second burst mode is the most interesting: keep the Pi awake during
critical periods (irrigation events, storm approach, frost risk) and stream
every reading through the NPU. At 48 µs per inference, the AKD1000 can
classify **20,000 readings per second** — the sensor is the bottleneck, not
the neural processor.

## Validated Results (Exp 029)

### S1: Streaming Soil Moisture (6/6 PASS)

500-step synthetic sensor stream at 15-min cadence simulating a 5-day irrigation
cycle with ET drawdown, rain event, irrigation trigger, and sensor glitch.

- **Live AKD1000**: 20,545 Hz throughput, mean 48.7 µs, P99 68.9 µs
- Semi-trained FC classifier distinguishes normal/stressed/anomaly states
- Sensor glitch at step 350 (reading = 0.95 VWC) correctly flagged

### S2: Seasonal Weight Evolution (4/4 PASS)

(1+1)-ES evolves int8 crop stress classifier weights across three seasonal
phases (early/mid/late). This is the key capability for LOCOMOS: the device
adapts its classifier as the growing season progresses without cloud retraining.

| Phase | θ̄ | σ | Start Fitness | Final Fitness |
|-------|-----|-----|:------------:|:------------:|
| Early (emergence) | 0.28 | 0.08 | 61% | 97% |
| Mid (peak growth) | 0.32 | 0.04 | 93% | 96% |
| Late (senescence) | 0.25 | 0.06 | 47% | 98% |

All fitness curves monotonically non-decreasing. AKD1000 supports weight mutation
via DMA — `load_readout_weights` takes ~60 µs.

### S3: Multi-Crop Crosstalk (6/6 PASS)

Rapidly switch between corn/soybean/potato classifiers (100 rounds × 3 crops).
All responses deterministically stable. No SRAM bleed between classifier contexts.

This means a single AKD1000 can serve multiple crop zones by hot-swapping
classifier weights per field sector — one NPU per farm, not per field.

### S4: LOCOMOS Power Budget (7/7 PASS)

| Metric | Value |
|--------|-------|
| Daily energy (96 readings, Pi + NPU) | 2.53 Wh (505 mAh @ 5V) |
| 5W solar panel output (4 peak sun hours) | 20 Wh/day — 8× surplus |
| NPU energy fraction of active cycle | 0.0009% |
| Per-reading active energy: NPU edge | 336 mJ |
| Per-reading active energy: cloud WiFi | 3,600 mJ |
| **NPU edge saves** | **10.7× active energy** |
| Cost breakeven (NPU $99 vs cloud $60/yr) | 20 months |

### S5: Noise Resilience (3/3 PASS)

Anderson-style sensor noise sweep (σ = 0.000 to 0.150 VWC). Classification
stays at 74.5–78.5% across all noise levels. The int8 quantization naturally
provides a noise floor — sensor degradation over seasons doesn't break inference.

## Validated Modules Used

| Module | Experiment | Checks |
|--------|-----------|--------|
| `barracuda::npu` (feature-gated) | Exp 028 | 35/35 |
| `airspring-forge` | Exp 028 | 21/21 |
| `validate_npu_funky_eco` | Exp 029 | 32/32 |
| `io::csv_ts` (sensor streaming) | Exp 003 | 11/11 |
| `gpu::stream::StreamSmoother` | Exp 003 | Wired |
| `eco::evapotranspiration` | Exp 001 | 64/64 |
| `eco::water_balance` | Exp 004 | 18/18 |

## Implementation Plan

### Phase 1: Validated ✓
- [x] AKD1000 driver (`akida-driver`, pure Rust) — from ToadStool
- [x] Three int8 classifiers: crop stress, irrigation decision, sensor anomaly
- [x] metalForge dispatch: GPU > NPU > CPU routing
- [x] Streaming inference: 500 readings, <100 µs mean
- [x] Seasonal weight evolution: (1+1)-ES, 3 phases
- [x] Multi-crop crosstalk: corn/soybean/potato, zero bleed
- [x] Power budget: 18650 feasible, solar surplus

### Phase 2: High-Cadence Pipeline (next)
- [ ] 1-minute cadence streaming through live AKD1000
- [ ] Event-triggered burst mode (10-sec during irrigation)
- [ ] Rolling anomaly detector with adaptive threshold
- [ ] Multi-sensor fusion (θ + T + EC in single inference)

### Phase 3: Field Integration
- [ ] SoilWatch 10 raw count → NPU classifier (bypass Python)
- [ ] Integration with scheduling optimizer (Exp 014)
- [ ] Real-time depletion tracking with NPU + water balance
- [ ] Dong lab field data when available (2026)

### Phase 4: Penny Irrigation + NPU
- [ ] $200 sensor + $35 Pi + $99 NPU = $334 sovereign field unit
- [ ] Solar powered, battery backed, no cloud dependency
- [ ] Classifier adapts per growing season
- [ ] Multi-crop per farm via weight hot-swap

## Connection to Penny Irrigation

The Penny Irrigation vision (Phase 4 of airSpring evolution) is:
"$200 sensor + free weather + $600 GPU = sovereign irrigation scheduling."

NPU changes the equation. The $600 GPU handles batch computation (atlas,
historical analysis, seasonal planning). The $99 NPU handles real-time field
inference. Together:

```
Farm office:  $600 GPU running BarraCuda
              → seasonal planning, weather forecast integration
              → train classifiers for this season's crops

Each field:   $334 sensor + Pi + NPU
              → real-time soil monitoring at 1-min cadence
              → on-device crop stress classification
              → irrigation decision without connectivity
              → weight updates when Pi syncs over WiFi (nightly)
```

Total cost for a 4-field farm: $600 + 4×$334 = **$1,936 one-time**.
No subscription. No cloud. No vendor lock-in. Sovereign.

---

## References

Dong, X., Werling, S., Cao, K., Li, Y. (2024). Implementation of an In-Field
IoT System for Precision Irrigation Management. Frontiers in Water 6, 1353597.

Ali, K., Dong, X., & Lavely, E. (2024). Irrigation scheduling optimization for
cotton. Agricultural Water Management 306.

BrainChip Inc. AKD1000 Hardware Reference Manual, 2024.
