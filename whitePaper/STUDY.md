# airSpring Study: Precision Agriculture on Consumer Hardware

**Status**: Working draft
**Date**: February 2026
**See also**: [METHODOLOGY.md](METHODOLOGY.md) for validation protocol

---

## Abstract

We independently replicate four precision agriculture computational methods — FAO-56 Penman-Monteith evapotranspiration, dielectric soil moisture calibration, IoT-based irrigation scheduling, and daily soil water balance — using only open-source tools and publicly available data. The Python/R baselines (142/142 checks) validate against digitized paper benchmarks. A real data pipeline using Open-Meteo historical weather (918 station-days, 6 Michigan agricultural stations, 2023 growing season) produces ET₀ with R²=0.967 against an independent computation. Water balance simulations on real data show 53-72% water savings with smart scheduling — consistent with published results. A Rust implementation via BarraCUDA passes 70/70 validation checks, establishing the foundation for GPU-accelerated precision irrigation on consumer hardware.

---

## 1. Introduction

### 1.1 The Problem

Agricultural irrigation consumes approximately 70% of global freshwater withdrawals. Inefficient scheduling wastes 30-50% of applied water while degrading soil health. Current precision irrigation relies on proprietary sensor systems ($500-$5000 per field), vendor-locked scheduling software, and institutional weather station networks with limited spatial resolution.

### 1.2 This Study

We replicate the computational methods published by Dr. Younsuk Dong (Michigan State University) and the FAO Irrigation and Drainage Paper No. 56, establishing a fully open, reproducible precision irrigation pipeline that requires:
- No institutional access
- No proprietary software
- No synthetic data (in the primary pipeline)
- Only consumer hardware

### 1.3 Evolution Path

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCUDA) → GPU (ToadStool) → Penny Irrigation
```

---

## 2. Phase 0: Python/R Control (142/142 PASS)

### 2.1 FAO-56 Penman-Monteith ET₀ (64/64 PASS)

Reference: Allen et al. (1998) FAO Irrigation and Drainage Paper 56.

**Implementation**: Manual numpy implementation following the paper's calculation sheets exactly. Each equation (Eqs. 6-50) implemented as a separate function with FAO-56 equation numbers in docstrings.

**Benchmark data**: Digitized from FAO-56 Chapter 4:
- Example 17: Bangkok, April (monthly, measured ea) — expected ET₀ = 5.72 mm/day
- Example 18: Uccle, 6 July (daily, RH + wind conversion) — expected ET₀ = 3.88 mm/day
- Example 20: Lyon, July (missing data, Tmax/Tmin only) — expected ET₀ = 4.56 mm/day
- Tables 2.3 and 2.4: Saturation vapour pressure and slope (11 + 10 points)

**Results**: All 64 checks pass. ET₀ errors: Bangkok 0.004 mm/d, Uccle 0.000 mm/d, Lyon 0.000 mm/d. All intermediate values within specified tolerances.

### 2.2 Soil Sensor Calibration (36/36 PASS)

Reference: Dong et al. (2020) Agriculture 10(12), 598.

**Implementation**: Topp equation, RMSE/IA/MBE formulas (Paper Eqs. 1-3), correction equation fitting (linear, quadratic, exponential, logarithmic) using scipy curve_fit.

**Results**: All 36 checks pass. Topp equation matches computed values within 0.005 m³/m³. Statistical formulas analytically exact. Quadratic correction confirmed as best fit (paper's conclusion). Field RMSE improvements validated for all 6 sensor/soil combinations.

### 2.3 IoT Irrigation Pipeline (24/24 PASS)

Reference: Dong et al. (2024) Frontiers in Water 6, 1353597.

**Implementation**: SoilWatch 10 calibration equation (Paper Eq. 5), irrigation recommendation model (Paper Eq. 1). R ANOVA script for yield analysis.

**Results**: All 24 Python checks pass. SoilWatch 10 produces correct VWC from raw capacitance. Irrigation model correctly recommends water application based on field capacity and current VWC. Published sensor performance criteria and field demonstration results validated.

### 2.4 Water Balance (18/18 PASS)

Reference: FAO-56 Chapter 8 (Allen et al. 1998).

**Implementation**: Daily root zone depletion tracking (Eq. 85), stress coefficient (Eq. 84), total available water, readily available water.

**Results**: All 18 checks pass. Mass balance closes to 0.000000 mm for dry-down, irrigated, and heavy rain scenarios. Ks correctly transitions from 1.0 to stress at Dr > RAW. Michigan summer simulation produces realistic ET and irrigation schedule.

---

## 3. Phase 0+: Real Data Pipeline (918 Station-Days)

### 3.1 Data Sources

| Source | Records | Coverage | Key |
|--------|---------|----------|-----|
| Open-Meteo archive | 918 station-days | 6 MI stations, May-Sep 2023 | None needed |
| OpenWeatherMap | 42 station-days | 6 MI stations, current + forecast | testing-secrets/ |
| NOAA CDO (GHCND) | 153 station-days | Lansing, May-Sep 2023 | testing-secrets/ |

### 3.2 ET₀ Cross-Validation

Our FAO-56 implementation vs Open-Meteo's independent ET₀ computation:

| Station | Days | RMSE (mm/d) | MBE (mm/d) | R² |
|---------|:----:|:-----------:|:----------:|:--:|
| East Lansing (MSU) | 153 | 0.295 | +0.119 | 0.965 |
| Grand Junction | 153 | 0.244 | +0.051 | 0.971 |
| Sparta | 153 | 0.279 | +0.100 | 0.970 |
| Hart (tomato site) | 153 | 0.220 | +0.048 | 0.974 |
| West Olive (blueberry) | 153 | 0.257 | +0.025 | 0.963 |
| Manchester (corn) | 153 | 0.297 | +0.116 | 0.960 |
| **All stations** | **918** | **0.267** | **+0.076** | **0.967** |

The small positive bias (+0.076 mm/d) is consistent with differences between Open-Meteo's ERA5 reanalysis inputs and our strict FAO-56 implementation (which uses the reported daily min/max for vapour pressure deficit computation). R² > 0.96 at every station confirms our ET₀ is correct on real data.

### 3.3 Water Balance on Real Data

Four crop scenarios simulated at actual Dong (2024) demonstration sites:

| Crop | Station | Rain-fed ET | Irrigated ET | Smart Irrig | Savings vs Naive |
|------|---------|:-----------:|:------------:|:-----------:|:----------------:|
| Blueberry | West Olive | 378.7 mm | 538.8 mm | 210 mm (14 events) | **72%** |
| Tomato | Hart | 424.9 mm | 693.5 mm | 350 mm (14 events) | **53%** |
| Corn | Manchester | 467.7 mm | 759.3 mm | 330 mm (11 events) | **56%** |
| Ref. grass | East Lansing | 434.5 mm | 656.7 mm | 300 mm (12 events) | **60%** |

All mass balances close to 0.0000 mm. Water savings of 53-72% are consistent with Dong et al. (2024) reported savings of ~30% for tomato (our higher figure reflects comparison against a naive 25mm/5-day schedule, while Dong compared against traditional scheduling).

---

## 4. Phase 1: Rust BarraCUDA (70/70 PASS)

### 4.1 Module Structure

| Module | Functions | Tests |
|--------|----------|:-----:|
| `eco::evapotranspiration` | 17 FAO-56 functions + DailyEt0Input struct | 22 |
| `eco::soil_moisture` | Topp eq, inverse, PAW, SoilTexture enum | 25 |
| `eco::water_balance` | WaterBalanceState, simulate_season, mass_balance | 12 |
| `io::csv_ts` | TimeseriesData parser, ColumnStats | 11 |

### 4.2 Current Gaps (Python → Rust)

| Capability | Python | Rust | Gap |
|-----------|:------:|:----:|-----|
| FAO-56 ET₀ (full chain) | Yes | Yes | None — full implementation |
| Sunshine hours Rs estimate | Yes | No | Rust expects Rs directly |
| Hargreaves Rs from temp | Yes | No | Missing data pathway |
| RMSE/IA/MBE statistics | Yes | No | Rust has Topp only, not stats |
| Correction equation fitting | Yes | No | scipy curve_fit not ported |
| SoilWatch 10 calibration | Yes | No | Rust does CSV parsing only |
| Irrigation recommendation | Yes | No | Not yet in Rust |
| R ANOVA | R script | No | Statistical test |
| Real data download | Yes | No | Python scripts handle APIs |
| Runoff model | RO=0 (FAO) | Simple rule | Minor difference |

### 4.3 Rust Validation Results

| Binary | Checks | Key Results |
|--------|:------:|-------------|
| validate_et0 | 22/22 | FAO-56 tables (10 es, 5 Delta), Uccle/Bangkok ET₀ |
| validate_soil | 25/25 | Topp (7 points), inverse round-trip, 5 USDA textures, PAW |
| validate_iot | 11/11 | 168 records, 5 columns, CSV round-trip |
| validate_water_balance | 12/12 | Mass balance 0.0000 (3 scenarios), Ks bounds |

---

## 5. Lessons and Findings

### 5.1 Open Data Is Sufficient

The combination of Open-Meteo (historical), OpenWeatherMap (real-time), and NOAA CDO (station-level) provides all weather data needed for precision agriculture in Michigan — and anywhere globally. No institutional weather station access required.

### 5.2 Paper Benchmarks Are Essential

The digitized paper values caught implementation bugs that would have been invisible with only synthetic test data. Example 18 (Uccle) requires wind speed conversion from 10m to 2m — the FAO-56 equation for this is easy to get wrong.

### 5.3 The FAO-56 Is Remarkably Complete

FAO Paper 56 provides worked examples with all intermediate values — a rare and valuable practice. This made it possible to validate each equation independently, not just the final ET₀ output.

### 5.4 Real Data Reveals Model Behavior

On real Michigan data, blueberry at West Olive experiences 81/153 stress days without irrigation but only 14 with smart scheduling — a dramatic difference that synthetic data would not capture with the same specificity.

### 5.5 Cross-Spring Patterns

airSpring's validation methodology mirrors hotSpring's two-phase approach:
- **hotSpring**: Python plasma physics → Rust nuclear EOS → GPU HFB
- **airSpring**: Python agricultural science → Rust ET₀/soil/water → GPU irrigation

The same BarraCUDA/ToadStool infrastructure supports both domains. The key shared abstractions are: ODE/PDE solvers, nonlinear fitting, time series processing, and spatial interpolation.

---

## 6. Evolution Path

### Near Term (Phase 2)
- Cross-validate Python vs Rust: identical ET₀ outputs for same weather inputs
- Port RMSE/IA/MBE statistics to Rust
- Port SoilWatch 10 calibration to Rust
- Benchmark: Rust vs Python throughput on 918 station-days

### Medium Term (Phase 3)
- GPU acceleration via ToadStool for ET₀ computation across spatial grids
- Spatial interpolation (kriging) for soil moisture mapping
- Real-time IoT stream processing on GPU
- Richards equation solver (1D unsaturated flow)

### Long Term (Phase 4: Penny Irrigation)
- Sovereign irrigation scheduling on consumer hardware ($600 GPU)
- Sub-field spatial resolution from cheap sensor networks
- Model-predictive control with weather forecast integration
- Open alternative to $5000/field proprietary systems

---

*February 2026 — 212/212 checks pass, 918 station-days real data, zero synthetic*
