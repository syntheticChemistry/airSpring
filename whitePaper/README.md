# airSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the replication of precision agriculture computational methods on consumer hardware using BarraCuda
**Date**: February 2026 (v0.4.4)

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [METHODOLOGY.md](METHODOLOGY.md) | Multi-phase validation protocol (Python control → Rust evolution → cross-validation) | Methodology review |
| [STUDY.md](STUDY.md) | Full results: paper benchmarks, real data pipeline, Rust validation | Reviewers, collaborators |
| [baseCamp/README.md](baseCamp/README.md) | Per-faculty research briefings and next steps | Lab planning |

---

## What This Study Is

airSpring replicates published precision irrigation, soil science, and environmental systems methods from Dr. Younsuk Dong (Michigan State University), the FAO-56 standard, and classical soil physics, first in Python/R (the original tools), then in Rust via BarraCuda, with the goal of GPU-accelerated sovereign irrigation scheduling on consumer hardware.

The study answers four questions:

1. **Can published agricultural science be independently reproduced using open tools?**
   Answer: yes — 400/400 Python checks pass against digitized paper benchmarks (FAO-56 examples, Dong 2020 soil sensor data, Dong 2024 IoT irrigation, Richards equation, biochar isotherms, yield response, CW2D, 60-year water balance).

2. **Can open data replace institutional weather station access?**
   Answer: yes — Open-Meteo (free, no key, 80+ years) provides real historical Michigan weather at 10km resolution. Our FAO-56 ET₀ matches Open-Meteo's independent computation with R²=0.967 across 918 station-days. NOAA CDO and OpenWeatherMap supplement with GHCND daily records and real-time forecasts.

3. **Can Rust + WebGPU replace Python/Excel for precision agriculture?**
   Answer: yes (validation complete) — Rust BarraCuda passes 643 tests across 18 binaries (96.81% coverage). A cross-validation harness confirms 75/75 Python-Rust value matches within 1e-5 tolerance. 11 Tier A modules wired to ToadStool/BarraCuda primitives including Richards PDE, isotherm fitting, MC ET₀ uncertainty (parametric CI via `norm_ppf`), VG pressure head inversion (via `brent`), and agroecological diversity. CPU benchmarks: 12.5M ET₀/s, 38.9M VG θ/s, 175K NM fits/s.

4. **Can the math be truly portable across hardware?**
   In progress — metalForge stages 6 absorption-ready modules for upstream barracuda integration following hotSpring's Write → Absorb → Lean pattern. 2 modules already absorbed upstream (van_genuchten into pde::richards, isotherm into optimize). GPU wiring proves the compute is hardware-portable; metalForge will demonstrate mixed CPU/GPU/NPU dispatch.

---

## Key Results

### Phase 0 (Python Control): 400/400 checks pass (13 experiments)

| Experiment | Paper | Checks | Key Validation |
|------------|-------|:------:|----------------|
| FAO-56 Penman-Monteith | Allen et al. 1998 | 64/64 | Bangkok 5.72, Uccle 3.88, Lyon 4.56 mm/day |
| Soil Sensor Calibration | Dong et al. 2020 | 36/36 | Topp eq, RMSE/IA/MBE, correction fits |
| IoT Irrigation Pipeline | Dong et al. 2024 | 24/24 | SoilWatch 10, irrigation model, ANOVA |
| Water Balance | FAO-56 Chapter 8 | 18/18 | Mass balance 0.0000 mm, Ks bounds |
| Dual Kc | FAO-56 Chapter 7 | 63/63 | 10 crops, 11 soils, Kcb+Ke partitioning |
| Cover Crops | FAO-56 Ch 11 + Islam | 40/40 | 5 species, mulch reduction, no-till savings |
| Regional ET₀ | 6 Michigan stations | 61/61 | CV, pairwise r, geographic consistency |
| Richards Equation | van Genuchten 1980 | 14/14 | VG retention, conductivity, infiltration, drainage |
| Biochar Isotherms | Kumari et al. 2025 | 14/14 | Langmuir/Freundlich R², RL factor |
| 60-Year Water Balance | OSU Triplett, ERA5 | 10/10 | Decadal stability, mass balance, climate trends |
| Yield Response | Stewart 1977, FAO-56 Ch 10 | 32/32 | Ky table, single/multi-stage, WUE, scheduling |
| CW2D Richards | Dong et al. 2019, HYDRUS | 24/24 | Gravel/organic VG, infiltration, mass balance |

### Phase 0+ (Real Data): 918 station-days, R²=0.967

| Station | RMSE (mm/d) | R² | Total ET₀ (ours) | Total ET₀ (Open-Meteo) |
|---------|:-----------:|:--:|:-----------------:|:----------------------:|
| East Lansing | 0.295 | 0.965 | 660.1 mm | 642.0 mm |
| Grand Junction | 0.244 | 0.971 | 657.4 mm | 649.7 mm |
| Hart (tomato) | 0.220 | 0.974 | 662.9 mm | 655.7 mm |
| West Olive (blueberry) | 0.257 | 0.963 | 639.1 mm | 635.2 mm |
| **Overall** | **0.267** | **0.967** | — | — |

### Phase 1 (Rust BarraCuda): 643 tests, 18 binaries, 96.81% coverage

| Binary | Checks | Key Validation |
|--------|:------:|----------------|
| validate_et0 | 31/31 | FAO-56 Tables 2.3/2.4, Example 18 within 0.0005 mm/day |
| validate_soil | 26/26 | Topp equation, 5 USDA textures, PAW |
| validate_iot | 11/11 | CSV time series, column statistics |
| validate_water_balance | 13/13 | Mass balance, Ks bounds, MI summer |
| validate_sensor_calibration | 21/21 | SoilWatch 10, irrigation model |
| validate_real_data | 23/23 | Real data pipeline, Open-Meteo ET₀ |
| validate_dual_kc | 61/61 | FAO-56 Ch 7, 10 crops, Kcb+Ke |
| validate_cover_crop | 40/40 | 5 cover crops, mulch, no-till |
| validate_regional_et0 | 61/61 | 6-station intercomparison |
| validate_richards | 15/15 | VG retention, infiltration, drainage, mass balance |
| validate_biochar | 14/14 | Langmuir/Freundlich R², RL, residuals |
| validate_long_term_wb | 11/11 | 60-year ET₀, water balance, climate |
| validate_yield | 32/32 | Stewart 1977, Ky table, multi-stage, WUE |
| validate_cw2d | 24/24 | CW2D media, VG retention, mass balance |
| cross_validate | 75 values | Python↔Rust JSON harness |

### Phase 2 (Cross-validation): 75/75 MATCH

Python and Rust produce identical results (within 1e-5 tolerance) for 75 values
across atmospheric, solar, radiation, ET₀, Topp, SoilWatch 10, irrigation,
statistics, Hargreaves, sunshine Rs, monthly soil heat flux, van Genuchten
retention/conductivity, and Langmuir/Freundlich isotherm predictions.

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

## Next Phase: GPU Validation & metalForge

See `specs/PAPER_REVIEW_QUEUE.md` for the full paper queue and compute pipeline.
See `wateringHole/handoffs/` for the latest ToadStool/BarraCuda handoff (V012).
See `CHANGELOG.md` for the full evolution history.
