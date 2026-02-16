# airSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the replication of precision agriculture computational methods on consumer hardware using BarraCUDA
**Date**: February 2026

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [METHODOLOGY.md](METHODOLOGY.md) | Multi-phase validation protocol (Python control → Rust evolution → cross-validation) | Methodology review |
| [STUDY.md](STUDY.md) | Full results: paper benchmarks, real data pipeline, Rust validation | Reviewers, collaborators |

---

## What This Study Is

airSpring replicates published precision irrigation and soil science methods from Dr. Younsuk Dong (Michigan State University) and the FAO-56 standard, first in Python/R (the original tools), then in Rust via BarraCUDA, with the goal of GPU-accelerated sovereign irrigation scheduling on consumer hardware.

The study answers three questions:

1. **Can published agricultural science be independently reproduced using open tools?**
   Answer: yes — 142/142 Python checks pass against digitized paper benchmarks (FAO-56 examples, Dong 2020 soil sensor data, Dong 2024 IoT irrigation system).

2. **Can open data replace institutional weather station access?**
   Answer: yes — Open-Meteo (free, no key, 80+ years) provides real historical Michigan weather at 10km resolution. Our FAO-56 ET₀ matches Open-Meteo's independent computation with R²=0.967 across 918 station-days. NOAA CDO and OpenWeatherMap supplement with GHCND daily records and real-time forecasts.

3. **Can Rust + WebGPU replace Python/Excel for precision agriculture?**
   Answer: yes (validation complete) — Rust BarraCUDA passes 101/101 validation checks across 5 binaries with 106 tests. A cross-validation harness confirms 53/53 Python-Rust value matches within 1e-5 tolerance. The Rust crate now includes Hargreaves ET₀, crop Kc database (10 crops), sensor calibration, and a full growing-season pipeline demonstration. GPU acceleration is the next phase.

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

### Phase 1 (Rust BarraCUDA): 101/101 checks pass, 106 tests

| Binary | Checks | Key Validation |
|--------|:------:|----------------|
| validate_et0 | 31/31 | FAO-56 Tables 2.3/2.4, Example 18 within 0.0005 mm/day |
| validate_soil | 25/25 | Topp equation, 5 USDA textures, PAW |
| validate_iot | 11/11 | CSV time series, column statistics |
| validate_water_balance | 13/13 | Mass balance, Ks bounds, MI summer |
| validate_sensor_calibration | 21/21 | SoilWatch 10, irrigation model, Dong 2024 |

### Phase 2 (Cross-validation): 53/53 MATCH

Python and Rust produce identical results (within 1e-5 tolerance) for 53 values
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
