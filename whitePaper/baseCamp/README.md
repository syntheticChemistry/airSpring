# baseCamp: Per-Faculty Research Briefings

**Updated**: February 25, 2026
**Project**: airSpring — Ecological & Agricultural Sciences
**Status**: 5 papers reproduced, 142/142 Python + 123/123 Rust + 65/65 cross-validation

---

## Evolution Path

```
Phase 0   Python/R baselines    — reproduce paper results with original tools
Phase 0+  Real open data        — compute on Open-Meteo, NOAA, USDA (no institutional access)
Phase 1   Rust BarraCuda CPU    — cross-validated to 1e-5 vs Python (253 tests)
Phase 2   BarraCuda GPU         — 6 orchestrators, GPU-FIRST (ToadStool shaders)
Phase 3   metalForge            — mixed CPU/GPU/NPU, staged for upstream absorption
Phase 4   Penny Irrigation      — sovereign scheduling on consumer hardware ($600 GPU)
```

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domain |
|---------|------------|-------|:------:|:-----------:|:------:|--------|
| [Dong](dong.md) | MSU BAE | Irrigation & Soil | 5 | 5 | 142+123 | ET₀, soil moisture, IoT, water balance |

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

### Rust Validation (Phase 1)

| Binary | Checks | Modules Exercised |
|--------|:------:|-------------------|
| `validate_et0` | 31 | evapotranspiration, atmospheric, solar, radiation |
| `validate_soil` | 26 | soil_moisture, Topp, hydraulic properties |
| `validate_iot` | 11 | csv_ts streaming parser, round-trip |
| `validate_water_balance` | 13 | water_balance, mass conservation, Michigan season |
| `validate_sensor_calibration` | 21 | sensor_calibration, SoilWatch 10, irrigation |
| `validate_real_data` | 21 | 4 crops × rainfed+irrigated on real weather |
| `cross_validate` | 65 | Python↔Rust exact match (tol=1e-5) |

### GPU Orchestrators (Phase 2)

| Orchestrator | BarraCuda Primitive | Provenance |
|-------------|--------------------|----|
| `BatchedEt0` | `batched_elementwise_f64` (op=0) | hotSpring pow_f64 fix |
| `BatchedWaterBalance` | `batched_elementwise_f64` (op=1) | Multi-spring |
| `KrigingInterpolator` | `kriging_f64::KrigingF64` | wetSpring |
| `SeasonalReducer` | `fused_map_reduce_f64` | wetSpring, airSpring TS-004 fix |
| `StreamSmoother` | `moving_window_stats` | wetSpring S28+ |
| `fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN |

### Next Steps (Dong Lab)

- **Paper 6**: Multi-sensor calibration network (awaiting field data from new lab)
- **Paper 7**: Full IoT irrigation with forecast integration (awaiting field data)
- **Paper 8**: FAO-56 Chapter 7 dual Kc (open literature, ready to implement)
- **Paper 9**: Regional ET₀ intercomparison (80-year Open-Meteo archive, ready)
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
