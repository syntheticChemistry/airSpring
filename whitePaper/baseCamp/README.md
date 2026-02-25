# baseCamp: Per-Faculty Research Briefings

**Updated**: February 25, 2026
**Project**: airSpring — Ecological & Agricultural Sciences
**Status**: 8 papers reproduced, 306/306 Python + 287/287 Rust + 65/65 cross-validation

---

## Evolution Path

```
Phase 0   Python/R baselines    — reproduce paper results with original tools (306/306)
Phase 0+  Real open data        — compute on Open-Meteo, NOAA, USDA (no institutional access)
Phase 1   Rust BarraCuda CPU    — cross-validated to 1e-5 vs Python (279 tests, 287 checks)
Phase 2   BarraCuda GPU         — 7 orchestrators, GPU-FIRST (ToadStool shaders)
Phase 3   metalForge            — mixed CPU/GPU/NPU, staged for upstream absorption
Phase 4   Penny Irrigation      — sovereign scheduling on consumer hardware ($600 GPU)
```

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domain |
|---------|------------|-------|:------:|:-----------:|:------:|--------|
| [Dong](dong.md) | MSU BAE | Irrigation & Soil | 8 | 11 | 306+287 | ET₀, soil moisture, IoT, water balance, dual Kc, cover crops |

## Faculty: Younsuk Dong, PhD

**Position**: Assistant Professor & Irrigation Specialist
**Department**: Biosystems and Agricultural Engineering, Michigan State University
**Note**: Establishing new lab in 2026. Currently sole PI for precision irrigation at MSU.

### Papers Reproduced

| # | Paper | Phase | Checks | Open Data |
|---|-------|:-----:|:------:|:---------:|
| 1 | Allen et al. (1998) FAO-56 Ch 2/4 — Penman-Monteith ET₀ | 0 | 64/64 | FAO-56 examples (open literature) |
| 2 | Dong et al. (2020) Soil sensor calibration — CS616/EC5 | 0 | 36/36 | Published Tables 3-4 |
| 3 | Dong et al. (2024) IoT irrigation pipeline — SoilWatch 10 | 0 | 24/24 | Published tables/equations |
| 4 | FAO-56 Chapter 8 — Water balance scheduling | 0 | 18/18 | FAO-56 Ch 8 + USDA |
| 5 | Real data pipeline — 6 Michigan stations, 918 days | 0+ | R²=0.967 | Open-Meteo ERA5 |
| 6 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc (Kcb+Ke) | 0 | 63/63 | FAO-56 Tables 17, 19 |
| 7 | Regional ET₀ intercomparison — 6 Michigan stations | 0 | 61/61 | Open-Meteo ERA5 |
| 8 | Islam et al. (2014) No-till + FAO-56 Ch 11 cover crops | 0 | 40/40 | ISWCR + FAO-56 |

### Rust Validation (Phase 1)

| Binary | Checks | Modules Exercised |
|--------|:------:|-------------------|
| `validate_et0` | 31 | evapotranspiration, atmospheric, solar, radiation |
| `validate_soil` | 26 | soil_moisture, Topp, hydraulic properties |
| `validate_iot` | 11 | csv_ts streaming parser, round-trip |
| `validate_water_balance` | 13 | water_balance, mass conservation, Michigan season |
| `validate_sensor_calibration` | 21 | sensor_calibration, SoilWatch 10, irrigation |
| `validate_real_data` | 23 | 4 crops × rainfed+irrigated on real weather, capability-based |
| `validate_dual_kc` | 61 | dual_kc, Kcb+Ke partitioning, FAO-56 Ch 7 |
| `validate_cover_crop` | 40 | dual_kc cover crops, no-till mulch, Islam et al. |
| `validate_regional_et0` | 61 | regional intercomparison, cross-station Pearson r |
| `cross_validate` | 65 | Python↔Rust exact match (tol=1e-5) |

### GPU Orchestrators (Phase 2)

| Orchestrator | BarraCuda Primitive | Provenance |
|-------------|--------------------|----|
| `BatchedEt0` | `batched_elementwise_f64` (op=0) | hotSpring pow_f64 fix |
| `BatchedWaterBalance` | `batched_elementwise_f64` (op=1) | Multi-spring |
| `BatchedDualKc` | CPU path (Tier B → GPU pending) | airSpring v0.3.10 |
| `KrigingInterpolator` | `kriging_f64::KrigingF64` | wetSpring |
| `SeasonalReducer` | `fused_map_reduce_f64` | wetSpring, airSpring TS-004 fix |
| `StreamSmoother` | `moving_window_stats` | wetSpring S28+ |
| `fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN |

### CPU Benchmarks (Phase 1+)

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| ET₀ (FAO-56) | 12.7M station-days/s | `bench_cpu_vs_python` |
| Dual Kc | 59M days/s | Kcb+Ke partitioning |
| Mulched Kc | 64M days/s | No-till mulch reduction |

### Next Steps (Dong Lab)

- **Paper 9+**: Multi-sensor calibration network (awaiting field data from new lab)
- **Paper 10+**: Full IoT irrigation with forecast integration (awaiting field data)
- **GPU validation**: Move dual Kc and ET₀ to pure GPU via ToadStool shaders
- **Weighing lysimeter**: Dong & Hansen (2023) load cell → direct ET (ready)

### What Good Science Looks Like

Every Dong lab paper we reproduced:
1. Uses published equations (FAO-56, Topp) — anyone can implement
2. Uses measurable inputs (temperature, humidity, soil permittivity) — no proprietary sensors
3. Reports quantitative results (RMSE, R², IA) — we can check
4. The IoT system costs ~$200/node — consumer hardware, not lab equipment

This is *exactly* the kind of science that benefits from sovereign compute.
The farmer doesn't need a $5000 Campbell Scientific station. They need a
$200 sensor, Open-Meteo weather data, and a $600 GPU running BarraCuda.
