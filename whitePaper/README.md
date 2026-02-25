# airSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the replication of precision agriculture computational methods on consumer hardware using BarraCuda
**Date**: February 2026

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [METHODOLOGY.md](METHODOLOGY.md) | Multi-phase validation protocol (Python control → Rust evolution → cross-validation) | Methodology review |
| [STUDY.md](STUDY.md) | Full results: paper benchmarks, real data pipeline, Rust validation | Reviewers, collaborators |

---

## What This Study Is

airSpring replicates published precision irrigation and soil science methods from Dr. Younsuk Dong (Michigan State University) and the FAO-56 standard, first in Python/R (the original tools), then in Rust via BarraCuda, with the goal of GPU-accelerated sovereign irrigation scheduling on consumer hardware.

The study answers three questions:

1. **Can published agricultural science be independently reproduced using open tools?**
   Answer: yes — 142/142 Python checks pass against digitized paper benchmarks (FAO-56 examples, Dong 2020 soil sensor data, Dong 2024 IoT irrigation system).

2. **Can open data replace institutional weather station access?**
   Answer: yes — Open-Meteo (free, no key, 80+ years) provides real historical Michigan weather at 10km resolution. Our FAO-56 ET₀ matches Open-Meteo's independent computation with R²=0.967 across 918 station-days. NOAA CDO and OpenWeatherMap supplement with GHCND daily records and real-time forecasts.

3. **Can Rust + WebGPU replace Python/Excel for precision agriculture?**
   Answer: yes (validation complete) — Rust BarraCuda passes 123/123 validation checks across 8 binaries with 293 tests (253 barracuda + 40 forge). A cross-validation harness confirms 65/65 Python-Rust value matches within 1e-5 tolerance. The Rust crate now includes Hargreaves ET₀, crop Kc database (10 crops), sensor calibration, and a full growing-season pipeline demonstration. GPU acceleration is LIVE — 6 orchestrators, 4/4 ToadStool issues resolved, GPU determinism verified. metalForge stages 4 absorption-ready modules for upstream barracuda integration following hotSpring's Write → Absorb → Lean pattern.

---

## Key Results

### Phase 0 (Python Control): 142/142 checks pass

| Experiment | Paper | Checks | Key Validation |
|------------|-------|:------:|----------------|
| FAO-56 Penman-Monteith | Allen et al. 1998 | 64/64 | Bangkok 5.72, Uccle 3.88, Lyon 4.56 mm/day |
| Soil Sensor Calibration | Dong et al. 2020 | 36/36 | Topp eq, RMSE/IA/MBE, correction fits |
| IoT Irrigation Pipeline | Dong et al. 2024 | 24/24 | SoilWatch 10, irrigation model, ANOVA |
| Water Balance | FAO-56 Chapter 8 | 18/18 | Mass balance 0.0000 mm, Ks bounds |

### Phase 0+ (Real Data): 918 station-days, R²=0.967

| Station | RMSE (mm/d) | R² | Total ET₀ (ours) | Total ET₀ (Open-Meteo) |
|---------|:-----------:|:--:|:-----------------:|:----------------------:|
| East Lansing | 0.295 | 0.965 | 660.1 mm | 642.0 mm |
| Grand Junction | 0.244 | 0.971 | 657.4 mm | 649.7 mm |
| Hart (tomato) | 0.220 | 0.974 | 662.9 mm | 655.7 mm |
| West Olive (blueberry) | 0.257 | 0.963 | 639.1 mm | 635.2 mm |
| **Overall** | **0.267** | **0.967** | — | — |

Water balance simulations on real data: 53-72% water savings with smart scheduling vs naive irrigation — consistent with Dong et al. (2024) published results.

### Phase 1 (Rust BarraCuda): 123/123 checks pass, 253 tests

| Binary | Checks | Key Validation |
|--------|:------:|----------------|
| validate_et0 | 31/31 | FAO-56 Tables 2.3/2.4, Example 18 within 0.0005 mm/day |
| validate_soil | 26/26 | Topp equation, 5 USDA textures, PAW |
| validate_iot | 11/11 | CSV time series, column statistics |
| validate_water_balance | 13/13 | Mass balance, Ks bounds, MI summer |
| validate_sensor_calibration | 21/21 | SoilWatch 10, irrigation model, Dong 2024 |
| validate_real_data | 21/21 | Real data pipeline, Open-Meteo ET₀ |
| cross_validate | 65 values | Python↔Rust JSON harness |
| simulate_season | — | Full growing-season pipeline |

### Phase 2 (Cross-validation): 65/65 MATCH

Python and Rust produce identical results (within 1e-5 tolerance) for 65 values
across atmospheric, solar, radiation, ET₀, Topp, SoilWatch 10, irrigation,
statistics, Hargreaves, sunshine Rs, and monthly soil heat flux.

---

## Reproduction

```bash
# Phase 0 + real data (~30 seconds)
pip install -r control/requirements.txt
bash scripts/run_all_baselines.sh

# Phase 1 — Rust (requires barracuda dependency)
cd barracuda && cargo run --release --bin validate_et0
```

No institutional access required. No proprietary software. AGPL-3.0 licensed.

---

## Next Phase: Paper Review Candidates

airSpring's current work centers on **Younsuk Dong** (BAE, MSU — new lab 2026) and the FAO-56 standard. The faculty network reveals several extensions through cross-spring connections and Dong's broader program.

### Extension of Dong's Work

| Priority | Paper / Direction | Why |
|----------|-------------------|-----|
| **Tier 1** | Dong et al. — IoT soil moisture sensor network calibration (MSU field data) | Direct extension of current Exp 002/003. Real multi-sensor field data for site-specific calibration curves. Bridges to groundSpring Exp 001 (sensor noise characterization) |
| **Tier 1** | Dong et al. (2024) — Full IoT irrigation scheduling system with weather forecast integration | Extends current Exp 004 from single-field to multi-site demonstration. Real growing-season decision pipeline |
| **Tier 2** | Allen et al. (1998) FAO-56 Chapter 7 — Extended crop coefficient (Kc) studies | Current work validates ET₀; next step is crop-specific ETc via dual Kc. BarraCuda Rust crate already has 10-crop Kc database |
| **Tier 2** | Regional ET₀ model intercomparison across Michigan microclimates | Quantify how much ET₀ varies across the state using Open-Meteo's 80-year archive. Statistical framework for "is my local model portable?" |

### Cross-Spring Connections

| Spring | Connection to airSpring | What It Means |
|--------|------------------------|---------------|
| **groundSpring** | Exp 003 (error propagation) quantifies that humidity dominates ET₀ uncertainty at 66% | airSpring knows *how* to compute ET₀; groundSpring tells it *which input to worry about most* |
| **neuralSpring** | Exp 004 (transfer learning) shows Michigan→NM gap is ΔR²=0.326 | neuralSpring tells airSpring *how* to port a model to a new location; fine-tuning with 200 local samples bridges most of the domain gap |
| **neuralSpring** | Exp 001 (surrogate) replaces the full FAO-56 chain with a 4,673-param MLP at R²=0.999 | The surrogate is the fast inner loop for BarraCuda real-time irrigation scheduling |
| **wetSpring** | Soil microbiome health affects plant water uptake efficiency | Long-term: soil biology data from wetSpring pipelines informs crop stress models |

### Faculty Who Extend airSpring

| Professor | Department | Relevance |
|-----------|-----------|-----------|
| **Younsuk Dong** (primary) | BAE, MSU | ET₀, soil sensors, IoT irrigation — direct paper reproduction target |
| **Emily Dolson** (indirect) | CSE, MSU | Evolutionary optimization of irrigation scheduling parameters — directed evolution of sensor placement |
| **Christopher Waters** (indirect) | MMG, MSU | Soil microbiome signaling — how bacterial health in the rhizosphere affects plant water dynamics |

### BarraCuda Status for airSpring Extension

| Feature | Status | Next |
|---------|--------|------|
| ET₀ (Hargreaves + PM) | Rust validated, 123/123 | GPU: batch multi-site ET₀ |
| Crop Kc database | 10 crops in Rust | Extend to dual Kc (FAO-56 Ch 7) |
| Water balance | Rust validated, 13/13 | Multi-field scheduling optimizer |
| Sensor calibration | Rust validated, 21/21 | Real-time IoT stream processing |
| Correction equation fitting | Complete (eco::correction, pure Rust) | — |
| Cross-validation | 65/65 Python↔Rust match | Foundation for GPU promotion |
