# airSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the replication of precision agriculture computational methods on consumer hardware using BarraCUDA
**Date**: February 2026

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [METHODOLOGY.md](METHODOLOGY.md) | Two-phase validation protocol (Python control → Rust evolution) | Methodology review |
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
   Answer: in progress — Rust BarraCUDA passes 70/70 validation checks (ET₀, soil moisture, water balance, IoT parsing). Cross-validation with Python baselines and GPU acceleration are the next phases.

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

### Phase 1 (Rust BarraCUDA): 70/70 checks pass

| Binary | Checks | Key Validation |
|--------|:------:|----------------|
| validate_et0 | 22/22 | FAO-56 tables, Bangkok/Uccle ET₀ |
| validate_soil | 25/25 | Topp equation, 5 USDA textures, PAW |
| validate_iot | 11/11 | CSV time series, column statistics |
| validate_water_balance | 12/12 | Mass balance, Ks bounds, MI summer |

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
