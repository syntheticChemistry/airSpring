# baseCamp: Per-Faculty Research Briefings

**Updated**: February 25, 2026
**Project**: airSpring — Ecological & Agricultural Sciences (v0.4.2)
**Status**: 11 experiments, 344/344 Python + 328 Rust tests + 75/75 cross-validation + 8 GPU orchestrators

---

## Evolution Path

```
Phase 0   Python/R baselines    — reproduce paper results with original tools (344/344)
Phase 0+  Real open data        — compute on Open-Meteo, NOAA, USDA (no institutional access)
Phase 1   Rust BarraCuda CPU    — cross-validated to 1e-5 vs Python (328 tests, 16 binaries)
Phase 2   BarraCuda GPU         — 8 orchestrators wired (cross-spring shader evolution)
Phase 3   metalForge            — mixed CPU/GPU/NPU, 6 modules (2 absorbed, 4 pending)
Phase 4   Penny Irrigation      — sovereign scheduling on consumer hardware ($600 GPU)
```

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domain |
|---------|------------|-------|:------:|:-----------:|:------:|--------|
| Dong | MSU BAE | Irrigation & Soil | 8+ | 11 | 344+328 | ET₀, soil moisture, IoT, water balance, dual Kc, cover crops, Richards, biochar |

## Faculty: Younsuk Dong, PhD

**Position**: Assistant Professor & Irrigation Specialist
**Department**: Biosystems and Agricultural Engineering, Michigan State University
**Note**: Establishing new lab in 2026. Currently sole PI for precision irrigation at MSU.

### Papers Reproduced

| # | Paper | Phase | Checks | Open Data |
|---|-------|:-----:|:------:|:---------:|
| 1 | Allen et al. (1998) FAO-56 Ch 2/4 — Penman-Monteith ET₀ | 0→GPU | 64/64 | FAO-56 (open literature) |
| 2 | Dong et al. (2020) Soil sensor calibration — CS616/EC5 | 0→GPU | 36/36 | Published Tables 3-4 |
| 3 | Dong et al. (2024) IoT irrigation pipeline — SoilWatch 10 | 0→GPU | 24/24 | Published tables/equations |
| 4 | FAO-56 Chapter 8 — Water balance scheduling | 0→GPU | 18/18 | FAO-56 Ch 8 + USDA |
| 5 | Real data pipeline — 6 Michigan stations, 918 days | 0+ | R²=0.967 | Open-Meteo ERA5 |
| 6 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc (Kcb+Ke) | 0→CPU | 63/63 | FAO-56 Tables 17, 19 |
| 7 | Regional ET₀ intercomparison — 6 Michigan stations | 0→CPU | 61/61 | Open-Meteo ERA5 |
| 8 | Islam et al. (2014) No-till + FAO-56 Ch 11 cover crops | 0→CPU | 40/40 | ISWCR + FAO-56 |
| 9 | van Genuchten (1980) Richards equation | 0→GPU | 14+15 | Carsel & Parrish (1988) |
| 10 | Kumari et al. (2025) Biochar P adsorption | 0→GPU | 14+14 | Representative literature |
| 11 | 60-year water balance (OSU Triplett-Van Doren) | 0→CPU | 10+11 | Open-Meteo ERA5 60yr |

### Rust Validation (Phase 1) — 16 binaries

| Binary | Checks | Modules Exercised |
|--------|:------:|-------------------|
| `validate_et0` | 31 | evapotranspiration, atmospheric, solar, radiation |
| `validate_soil` | 26 | soil_moisture, Topp, hydraulic properties |
| `validate_iot` | 11 | csv_ts streaming parser, round-trip |
| `validate_water_balance` | 13 | water_balance, mass conservation, Michigan season |
| `validate_sensor_calibration` | 21 | sensor_calibration, SoilWatch 10, irrigation |
| `validate_real_data` | 23 | 4 crops × rainfed+irrigated, capability-based discovery |
| `validate_dual_kc` | 61 | dual_kc, Kcb+Ke partitioning, FAO-56 Ch 7 |
| `validate_cover_crop` | 40 | dual_kc cover crops, no-till mulch, Islam et al. |
| `validate_regional_et0` | 61 | regional intercomparison, cross-station Pearson r |
| `validate_richards` | 15 | van Genuchten retention/K, infiltration, drainage, mass balance |
| `validate_biochar` | 14 | Langmuir/Freundlich fitting, R², RL, residuals |
| `validate_long_term_wb` | 11 | 60-year ET₀, water balance, climate trends |
| `cross_validate` | 75 | Python↔Rust exact match (tol=1e-5) |

### GPU Orchestrators (Phase 2) — 8 wired

| Orchestrator | BarraCuda Primitive | Cross-Spring Provenance | Status |
|-------------|--------------------|----|---|
| `BatchedEt0` | `batched_elementwise_f64` (op=0) | hotSpring `pow_f64` fix (TS-001) | **GPU-FIRST** |
| `BatchedWaterBalance` | `batched_elementwise_f64` (op=1) | Multi-spring shared | **GPU-STEP** |
| `BatchedDualKc` | CPU path (Tier B → GPU pending) | airSpring v0.3.10 | CPU ready |
| `KrigingInterpolator` | `kriging_f64::KrigingF64` | wetSpring spatial interpolation | **Integrated** |
| `SeasonalReducer` | `fused_map_reduce_f64` | wetSpring + airSpring TS-004 fix | **GPU N≥1024** |
| `StreamSmoother` | `moving_window_stats` | wetSpring S28+ environmental | **Wired** |
| `BatchedRichards` | `pde::richards::solve_richards` | airSpring→ToadStool S40 absorption | **Wired** |
| `fit_*_nm/global` | `optimize::nelder_mead` + `multi_start` | neuralSpring optimizer | **Wired** |

### CPU Benchmarks (v0.4.2)

| Operation | Throughput | Cross-Spring Provenance |
|-----------|-----------|------------------------|
| ET₀ (FAO-56) | 12.5M station-days/s | hotSpring df64, multi-spring elementwise |
| VG θ(h) retention | 38.9M evals/s | hotSpring df64 precision |
| Dual Kc season | 59M days/s | airSpring `eco::dual_kc` |
| Reduce (seasonal) | 395M elem/s | wetSpring `fused_map_reduce` |
| Richards PDE | 72 sims/s (50 nodes) | airSpring→ToadStool, hotSpring df64 |
| Isotherm (NM 1-start) | 175K fits/s | neuralSpring `nelder_mead` |
| Isotherm (NM 8×LHS) | 42.5K fits/s | neuralSpring `multi_start_nelder_mead` |

### Next Steps (Dong Lab)

- **Paper 12+**: Multi-sensor calibration network (awaiting field data from new lab)
- **Paper 13+**: Full IoT irrigation with forecast integration (awaiting field data)
- **GPU validation**: Move Richards and isotherm to pure GPU via ToadStool shaders
- **metalForge**: Mixed hardware demonstration (GPU → NPU → CPU dispatch)
- **Weighing lysimeter**: Dong & Hansen (2023) load cell → direct ET (ready)

### What Good Science Looks Like

Every Dong lab paper we reproduced:
1. Uses published equations (FAO-56, Topp, van Genuchten) — anyone can implement
2. Uses measurable inputs (temperature, humidity, soil permittivity) — no proprietary sensors
3. Reports quantitative results (RMSE, R², IA) — we can check
4. The IoT system costs ~$200/node — consumer hardware, not lab equipment

This is *exactly* the kind of science that benefits from sovereign compute.
The farmer doesn't need a $5000 Campbell Scientific station. They need a
$200 sensor, Open-Meteo weather data, and a $600 GPU running BarraCuda.
