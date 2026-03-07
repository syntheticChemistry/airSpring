# airSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the replication of precision agriculture computational methods on consumer hardware using BarraCuda
**Date**: March 2026 (v0.7.3)

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
   Answer: yes — 1237/1237 Python checks pass against digitized paper benchmarks (FAO-56, soil sensors, IoT irrigation, Richards equation, biochar, yield, CW2D, 60-year WB, Priestley-Taylor, 3-method intercomparison, Thornthwaite, GDD, pedotransfer, AmeriFlux, Hargreaves, diversity, Anderson coupling, Blaney-Criddle, SCS-CN runoff, Green-Ampt infiltration, coupled runoff-infiltration, VG inverse fitting, full-season water budget).

2. **Can open data replace institutional weather station access?**
   Answer: yes — Open-Meteo (free, no key, 80+ years) provides real historical Michigan weather at 10km resolution. Our FAO-56 ET₀ matches Open-Meteo's independent computation with R²=0.967 across 15,300 station-days. NOAA CDO and OpenWeatherMap supplement with GHCND daily records and real-time forecasts.

3. **Can Rust + WebGPU replace Python/Excel for precision agriculture?**
   Answer: yes (validation complete) — Rust BarraCuda passes 846 lib + 62 forge tests across 86 binaries (pedantic + nursery 0 warnings). A cross-validation harness confirms 75/75 Python-Rust value matches within 1e-5 tolerance; 690 crop-station yield pairs within 0.01. 25 Tier A + 6 GPU-local + 5 Tier B GPU orchestrators wired to ToadStool/BarraCuda primitives including Richards PDE, isotherm fitting, MC ET₀ uncertainty, seasonal pipeline (ET₀→Kc→WB→Yield, 73/73 real data), atlas streaming (12 stations, 4800 crop-year results), Anderson soil-moisture coupling, coupled runoff-infiltration (292/292), VG inverse (84/84), and full-season water budget audit (34/34). S93 synced with universal f64 precision. CPU benchmarks: 14.5× geometric mean speedup vs Python (21/21 parity). NUCLEUS primal integration: 30 capabilities, ecology domain, 28/28 cross-primal pipeline. Paper 12 immunological Anderson (Exp 066-069).

4. **Can the math be truly portable across hardware?**
   Complete — all 6 metalForge modules absorbed upstream into barracuda (S64: metrics; S66: regression, hydrology, moving_window_f64; S40: van_genuchten; S64: isotherm). airSpring now leans on upstream primitives following the Write → Absorb → Lean cycle. GPU wiring proves the compute is hardware-portable; metalForge demonstrates the cross-system absorption pattern.

---

## Key Results

### Phase 0 (Python Control): 1237/1237 checks pass (54 experiments)

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
| Scheduling Optimization | Ali, Dong & Lavely 2024 | 25/25 | 5 strategies, mass balance, yield ordering, WUE |
| Lysimeter ET | Dong & Hansen 2023 | 26/26 | Mass-to-ET, temp compensation, calibration R² |
| ET₀ Sensitivity | Gong et al. 2006 methodology | 23/23 | OAT ±10%, 3 climatic zones, monotonicity |
| Priestley-Taylor ET₀ | Priestley & Taylor 1972 | 32/32 | PT α=1.26, analytical, cross-val vs PM, climate gradient |
| ET₀ 3-Method Intercomparison | PM/PT/HG on real data | 36/36 | 6 MI stations, R², bias, coastal effects |
| Thornthwaite ET₀ | Thornthwaite 1948 | 23/23 | Monthly heat index, temperature-based ET₀ |
| Growing Degree Days | Phenology standard | 33/33 | GDD accumulation, kc_from_gdd |
| Pedotransfer (Saxton-Rawls) | Saxton & Rawls 2006 | 70/70 | θs/θr/Ks from texture |
| NASS Yield Validation | Stewart 1977 pipeline | 41/41 | Drought ranking, 5 MI crops |
| Forecast Scheduling | Hindcast vs perfect knowledge | 19/19 | Noise sensitivity, mass balance |
| USDA SCAN Soil Moisture | Carsel & Parrish + SCAN | 34/34 | Richards vs in-situ θ profiles |
| Multi-Crop Water Budget | FAO-56 full pipeline | 47/47 | 5 crops, irrigated/rainfed/dual Kc |
| AmeriFlux ET (Baldocchi) | Baldocchi 2003 | 27/27 | Eddy covariance gold standard |
| Hargreaves-Samani ET₀ | Hargreaves & Samani 1985 | 24/24 | Temperature-only ET₀ |
| Ecological Diversity | Shannon/Simpson/Chao1 | 22/22 | Agroecosystem diversity indices |

### Phase 0+ (Real Data): 15,300 station-days, R²=0.967

| Station | RMSE (mm/d) | R² | Total ET₀ (ours) | Total ET₀ (Open-Meteo) |
|---------|:-----------:|:--:|:-----------------:|:----------------------:|
| East Lansing | 0.295 | 0.965 | 660.1 mm | 642.0 mm |
| Grand Junction | 0.244 | 0.971 | 657.4 mm | 649.7 mm |
| Hart (tomato) | 0.220 | 0.974 | 662.9 mm | 655.7 mm |
| West Olive (blueberry) | 0.257 | 0.963 | 639.1 mm | 635.2 mm |
| **Overall** | **0.267** | **0.967** | — | — |

### Phase 1 (Rust BarraCuda): 846 lib + 62 forge tests, 86 binaries

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
| validate_scheduling | 28/28 | 5 strategies, mass balance, yield ordering |
| validate_lysimeter | 25/25 | Mass-to-ET, calibration, diurnal pattern |
| validate_sensitivity | 23/23 | OAT ±10%, 3 climatic zones, ranking |
| validate_priestley_taylor | 32/32 | PT analytical, Uccle cross-val, climate gradient |
| validate_et0_intercomparison | 36/36 | PM/PT/HG, 6 stations, R², bias, RMSE |
| validate_thornthwaite | 50/50 | Thornthwaite monthly ET₀ |
| validate_gdd | 26/26 | GDD accumulation, kc_from_gdd |
| validate_pedotransfer | 58/58 | Saxton-Rawls 2006 θs/θr/Ks |
| validate_atlas | 1498/1498 | 100 Michigan stations, ValidationHarness checks |
| validate_nass_yield | 40/40 | Stewart pipeline, drought ranking, 5 MI crops |
| validate_forecast | 19/19 | Forecast vs perfect knowledge, noise sensitivity |
| validate_scan_moisture | 34/34 | Richards 1D vs SCAN in-situ profiles |
| validate_multicrop | 47/47 | 5-crop water budget, irrigated/rainfed/dual Kc |
| validate_npu_eco | 35/35 | AKD1000 crop stress/irrigation/anomaly |
| validate_npu_funky_eco | 32/32 | NPU streaming, evolution, LOCOMOS power |
| validate_npu_high_cadence | 28/28 | NPU 1-min cadence, burst, fusion, ensemble |
| validate_ameriflux | 27/27 | AmeriFlux eddy covariance ET |
| validate_hargreaves | 24/24 | Hargreaves-Samani temperature-based ET₀ |
| validate_diversity | 22/22 | Shannon, Simpson, Chao1, Bray-Curtis |
| validate_dispatch_routing | 21/21 | metalForge CPU+GPU+NPU dispatch (forge) |
| cross_validate | 75 values | Python↔Rust JSON harness |

### Phase 2 (Cross-validation): 75/75 MATCH + 690 crop-station yield pairs within 0.01

Python and Rust produce identical results (within 1e-5 tolerance) for 75 values
across atmospheric, solar, radiation, ET₀, Topp, SoilWatch 10, irrigation,
statistics, Hargreaves, sunshine Rs, monthly soil heat flux, van Genuchten
retention/conductivity, and Langmuir/Freundlich isotherm predictions.

---

## Reproduction

```bash
# Phase 0 + real data (~30 seconds)
pip install -r control/requirements.txt
bash run_all_baselines.sh

# Phase 1 — Rust (requires barracuda dependency)
cd barracuda && cargo run --release --bin validate_et0
```

No institutional access required. No proprietary software. AGPL-3.0 licensed.

---

## Next Phase: GPU Validation & metalForge

See `specs/PAPER_REVIEW_QUEUE.md` for the full paper queue and compute pipeline.
See `wateringHole/handoffs/` for the latest handoffs (V073 current).
See `CHANGELOG.md` for the full evolution history.
