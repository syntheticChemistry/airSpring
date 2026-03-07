# airSpring Control Experiment — Status Report

**Date**: 2026-02-16 (Project initialized)
**Updated**: 2026-03-07 (v0.7.4 — 81 experiments, barraCuda 0.3.3 (wgpu 28), 1284/1284 Python + 854 lib + 186 forge + 381/381 validation + 146/146 cross-spring evolution + 33/33 cross-validation. **14.5× Rust-vs-Python speedup** (21/21 algorithms). All 20 ops upstream (`BatchedElementwiseF64`), `local_dispatch` retired (v0.7.2). `PrecisionRoutingAdvice` wired, upstream provenance registry (v0.7.3). metalForge 66/66 mixed pipeline. Write→Absorb→Lean cycle complete. New: Exp 079 MC ET₀ (26/26), Exp 080 Bootstrap/Jackknife (20/20), Exp 081 SPI Drought Index (20/20).)
**Gate**: Eastgate (i9-12900K, 64 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**License**: AGPL-3.0-or-later

---

## Data Strategy

**Paper data** (digitized in `benchmark_*.json`) = ground truth benchmarks
that validate our methods are mathematically correct.

**Open data** (real weather from public APIs) = the actual data we compute on.
No synthetic data except as explicit last-resort fallback.

The open data also enables large sweeps across years, stations, and seasons
that go far beyond what any single paper covers.

| Layer | Source | Purpose | Key |
|-------|--------|---------|-----|
| Benchmark | FAO-56 tables/examples | Validate our PM equation | None |
| Benchmark | Dong 2020 Tables 3-4 | Validate Topp/calibration | None |
| Benchmark | Dong 2024 Eq 5, Table 2 | Validate IoT pipeline | None |
| Open Data | Open-Meteo archive | 80+ yr Michigan weather | **None (free)** |
| Open Data | OpenWeatherMap | Current + 5-day forecast | `testing-secrets/` |
| Open Data | NOAA CDO | Historical daily (GHCND) | `testing-secrets/` |
| Open Data | USDA Web Soil Survey | Soil properties by county | None |
| Fallback | Synthetic generation | Only if API unreachable | N/A |

---

## Replication Protocol

Anyone can reproduce all results by:

```bash
git clone git@github.com:syntheticChemistry/airSpring.git
cd airSpring

# 1. Install Python baselines (FAO-56, scipy, pandas)
pip install -r control/requirements.txt

# 2. Download REAL weather data (Open-Meteo — free, no key required)
python scripts/download_open_meteo.py --all-stations --growing-season 2023

# 3. Download current weather (OpenWeatherMap — key in testing-secrets/)
python scripts/download_enviroweather.py --all-stations

# 4. Run ALL baselines: paper validation + real data pipeline
bash run_all_baselines.sh

# 5. Optionally run R ANOVA (requires R >= 4.0)
# Rscript control/iot_irrigation/anova_irrigation.R

# 6. Pre-cache ERA5 data for Exp 015 (60-year water balance)
#    Downloads ~23 MB from Open-Meteo ERA5 archive (one-time).
#    Cached to: control/long_term_wb/data/wooster_era5_1960_2023.json
python control/long_term_wb/long_term_water_balance.py

# 7. Run Rust validation binaries (854+1498 checks across 89 binaries)
cd barracuda
for bin in validate_et0 validate_soil validate_iot validate_water_balance \
  validate_sensor_calibration validate_real_data cross_validate \
  validate_dual_kc validate_cover_crop validate_regional_et0 \
  validate_richards validate_biochar validate_long_term_wb \
  validate_yield validate_cw2d validate_scheduling \
  validate_lysimeter validate_sensitivity validate_atlas \
  validate_priestley_taylor validate_et0_intercomparison \
  validate_thornthwaite validate_gdd validate_pedotransfer \
  validate_nass_yield validate_forecast validate_scan_moisture \
  validate_multicrop validate_ameriflux validate_hargreaves \
  validate_diversity \
  validate_coupled_runoff validate_vg_inverse validate_season_wb validate_climate_scenario; do
  cargo run --release --bin $bin
done

# 7b. Run NPU validation (requires --features npu and AKD1000 hardware)
# cargo run --release --features npu --bin validate_npu_eco
# cargo run --release --features npu --bin validate_npu_funky_eco
# cargo run --release --features npu --bin validate_npu_high_cadence

# 8. Run Phase 2 cross-validation (75 values, Python vs Rust)
cd .. && python3 scripts/cross_validate.py > /tmp/py.json
cd barracuda && cargo run --release --bin cross_validate > /tmp/rs.json

# 9. Run full season simulation demo
cargo run --release --bin simulate_season
```

No institutional access required. All data is from public open APIs
(Open-Meteo, OpenWeatherMap, NOAA CDO, USDA Web Soil Survey, FAO papers).
All tools are open source. Zero synthetic data in the default pipeline.

---

## Hardware Gate

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (16C/24T, 5.2 GHz) |
| RAM | 64 GB DDR5-4800 |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| Storage | 1 TB NVMe SSD |
| OS | Pop!_OS 22.04 (Ubuntu-based) |

---

## Research Context

### Track 1: Precision Agriculture — Principal Investigator

**Younsuk Dong, PhD** — Assistant Professor & Irrigation Specialist, Biosystems
and Agricultural Engineering, Michigan State University. PhD Biosystems
Engineering, MSU 2018. Research: precision irrigation, IoT field monitoring,
soil moisture sensing, evapotranspiration modeling, agrivoltaics.

#### Key Publications

- Dong et al. (2020) "Performance evaluation of soil moisture sensors in
  coarse- and fine-textured Michigan agricultural soils" Agriculture 10(12), 598
- Dong & Hansen (2023) "Development and design of an affordable field scale
  weighing lysimeter using a microcontroller system" Smart Ag Tech 4, 100147
- Dong et al. (2024) "Implementation of an In-Field IoT System for Precision
  Irrigation Management" Frontiers in Water 6, 1353597
- Ali, Dong & Lavely (2024) "Impact of irrigation scheduling on yield and water
  use efficiency of apples, peaches, and sweet cherries" Ag Water Mgmt 306, 109148
- Mane et al. (2024) "Advancements in dielectric soil moisture sensor calibration"
  Computers and Electronics in Agriculture 218, 108686

### Track 2: Environmental Systems

- Dong et al. (2019) "Land-based wastewater treatment system modeling using
  HYDRUS CW2D" J. Sustainable Water in the Built Environment 5(4), 04019005
- Kumari, Dong & Safferman (2025) "Phosphorus adsorption and recovery from
  waste streams using biochar" Applied Water Science 15(7), 162
- Lee et al. (2024) "Grid-connected PV inverter for driving induction machines"
  IEEE Access 12, 5177-5187 (agrivoltaics power systems)

---

## Datasets

### Benchmark Data (from papers — validates our methods)

| ID | Dataset | Source | Format | Status |
|----|---------|--------|--------|--------|
| B1 | FAO-56 Examples 17, 18, 20 + Tables 2.3-2.4 | Allen et al. 1998 | `benchmark_fao56.json` | Digitized |
| B2 | Soil sensor calibration Tables 3-4 | Dong et al. 2020 | `benchmark_dong2020.json` | Digitized |
| B3 | IoT irrigation Eq 5, Table 2, yield | Dong et al. 2024 | `benchmark_dong2024.json` | Digitized |
| B4 | Water balance Chapter 8 equations | FAO-56 | `benchmark_water_balance.json` | Digitized |

### Open Data (real observations — what we compute on)

| ID | Dataset | Source | API Key | Status |
|----|---------|--------|---------|--------|
| D1 | Historical weather (80+ yr, 10km) | Open-Meteo | **None needed** | **918 station-days downloaded** |
| D2 | Current + 5-day forecast | OpenWeatherMap | `testing-secrets/` | **6 stations live** |
| D3 | Historical daily (GHCND) | NOAA CDO | `testing-secrets/` | **153 days Lansing downloaded** |
| D4 | Soil properties by county | USDA Web Soil Survey | None | Public |
| D5 | MSU Enviro-weather stations | MSU AgWeather | JavaScript site | Needs scraper |
| D6 | Lysimeter ET data | Dong & Hansen 2023 | DOI: [10.1016/j.atech.2023.100147](https://doi.org/10.1016/j.atech.2023.100147) | Supplementary — contact author |

### Track 2: Environmental Systems

| ID | Dataset | Source | Size | Status |
|----|---------|--------|------|--------|
| D7 | HYDRUS CW2D examples | PC-Progress | [hydrus.pc-progress.com](https://www.pc-progress.com/en/Default.aspx?hydrus-3d) | Public download |
| D8 | Biochar adsorption data | Kumari et al. 2025 | ~1 MB | Paper suppl. |
| D9 | EPA PFAS soil data | EPA ORD | [epa.gov/pfas](https://www.epa.gov/pfas/pfas-analytical-methods) | Public — EPA CompTox Dashboard |
| D10 | Agrivoltaic PAR data | MSU Solar Farm | Contact: MSU BAE Dept | To identify — IEEE Access DOI: [10.1109/ACCESS.2024.3350866](https://doi.org/10.1109/ACCESS.2024.3350866) |

---

## Run Log

### 2026-02-16: Real Data Pipeline — 918 Station-Days, ET₀ R²=0.97

Wired real Open-Meteo historical data into the validated FAO-56 pipeline.
Zero synthetic data in the primary pipeline.

**Weather download (6 Michigan stations, 2023 growing season):**

| Station | Data Source | Days | Tmax Range | Total Precip | Total ET₀ (Open-Meteo) |
|---------|-----------|------|-----------|-------------|----------------------|
| East Lansing (MSU) | Open-Meteo archive | 153 | 4.3–32.8°C | 423.4 mm | 642.0 mm |
| Grand Junction | Open-Meteo archive | 153 | 3.6–33.4°C | 454.6 mm | 649.7 mm |
| Sparta | Open-Meteo archive | 153 | 4.7–34.3°C | 379.5 mm | 672.2 mm |
| Hart (tomato site) | Open-Meteo archive | 153 | 2.8–32.0°C | 388.8 mm | 655.7 mm |
| West Olive (blueberry) | Open-Meteo archive | 153 | 3.8–32.3°C | 477.9 mm | 635.2 mm |
| Manchester (corn) | Open-Meteo archive | 153 | 4.8–31.8°C | 365.2 mm | 644.8 mm |

**ET₀ cross-check (our FAO-56 vs Open-Meteo's ET₀, 918 station-days):**

| Station | RMSE (mm/d) | MBE (mm/d) | R² | Our Total | OM Total |
|---------|------------|-----------|------|-----------|----------|
| East Lansing | 0.295 | +0.119 | 0.965 | 660.1 mm | 642.0 mm |
| Grand Junction | 0.244 | +0.051 | 0.971 | 657.4 mm | 649.7 mm |
| Hart | 0.220 | +0.048 | 0.974 | 662.9 mm | 655.7 mm |
| Manchester | 0.297 | +0.116 | 0.960 | 662.6 mm | 644.8 mm |
| Sparta | 0.279 | +0.100 | 0.970 | 687.5 mm | 672.2 mm |
| West Olive | 0.257 | +0.025 | 0.963 | 639.1 mm | 635.2 mm |
| **Overall** | **0.267** | **+0.076** | **0.967** | — | — |

Small positive bias (0.076 mm/d) consistent with Open-Meteo using ERA5 reanalysis
vs our strict FAO-56 implementation. R² > 0.96 confirms our ET₀ is correct on real data.

**Water balance on real data (4 crop scenarios, 2023 growing season):**

| Crop | Station | Rain-fed ET | Irrigated ET | Smart Irrig | Naive Irrig | Savings |
|------|---------|------------|-------------|------------|------------|---------|
| Blueberry | West Olive | 378.7 mm | 538.8 mm | 210 mm (14 events) | 750 mm | 72% |
| Tomato | Hart | 424.9 mm | 693.5 mm | 350 mm (14 events) | 750 mm | 53% |
| Corn | Manchester | 467.7 mm | 759.3 mm | 330 mm (11 events) | 750 mm | 56% |
| Ref. grass | East Lansing | 434.5 mm | 656.7 mm | 300 mm (12 events) | 750 mm | 60% |

All mass balances close to 0.0000 mm. Water savings (53–72%) consistent with
Dong et al. (2024) reported savings of ~30% for tomato and significant yield
improvements for blueberry.

### 2026-02-16: Phase 0 — Python/R Science Baselines (142/142 PASS)

Replicated Dr. Dong's studies using the same open-source tools the original
papers used. Each experiment has digitized benchmark data from the published
papers and a Python (or R) validation script in `control/`.

**Principle**: Validate the science with open tools the way the papers did,
BEFORE evolving to Rust/BarraCuda.

| Baseline Script | Paper | Checks | Key Validations |
|----------------|-------|--------|-----------------|
| `control/fao56/penman_monteith.py` | FAO-56 (Allen et al. 1998) | 64/64 | 11 es, 10 Delta, Examples 17 (Bangkok 5.72), 18 (Uccle 3.88), 20 (Lyon 4.56) with all intermediates |
| `control/soil_sensors/calibration_dong2020.py` | Dong et al. 2020, Agriculture 10(12) | 36/36 | Topp eq (8 pts), RMSE/IA/MBE formulas (7 analytical), Table 3 criteria (6), curve fitting R² (5), field RMSE (6), soil classification (3) |
| `control/iot_irrigation/calibration_dong2024.py` | Dong et al. 2024, Frontiers in Water 6 | 24/24 | SoilWatch 10 Eq 5 (4), irrigation model Eq 1 (4), Table 2 performance (6), field demos (6), synthetic stats (4) |
| `control/water_balance/fao56_water_balance.py` | FAO-56 Chapter 8 | 18/18 | TAW/RAW (3), Ks bounds (5), dry-down mass balance (2), irrigated mass balance (3), MI summer 535mm ET (3), heavy rain DP (2) |
| `control/iot_irrigation/anova_irrigation.R` | Dong et al. 2024 (R v4.3.1) | — | Written, awaiting R install; one-way ANOVA on blueberry/tomato yield |

**Total Python: 1284/1284 checks PASS, 54/54 baseline experiments PASS**
**Exp 018 Atlas: 1393/1393 Rust checks PASS (100-station full Michigan, 10 crops, cross-validated vs Python)**
**R ANOVA: script written, 1 skip (R not installed)**

Tools used: numpy, scipy (curve_fit, solve_ivp), json (benchmarks), base Python math.
All benchmark data digitized directly from published papers (FAO-56 tables,
Dong 2020 Tables 3-4, Dong 2024 Eq 5 + Table 2, Stewart 1977, CW2D media params).

### 2026-02-16 → 2026-02-25: Project Initialization → v0.4.3 (Rust — 439/439 PASS, 456 lib + 126 integration tests)

- Created airSpring repository
- Scaffolded Track 1 (Precision Agriculture) and Track 2 (Environmental Systems)
- Identified Dr. Younsuk Dong (MSU BAE) as principal investigator
- Defined 8 experiments across both tracks
- Created airspring-barracuda Rust crate v0.2.0 (6 eco modules, 5 validation binaries, 7 binaries total)
- Dependencies: barracuda (phase1/toadstool), serde, serde_json
- Comprehensive audit and evolution to modern idiomatic Rust (zero clippy pedantic/nursery warnings)
- `AirSpringError` unified error type replaces ad-hoc `String` errors
- Phase 2 cross-validation harness: 75/75 values match Python within 1e-5
- **All validation binaries PASS:**

| Binary | Track | Checks | Key validations |
|--------|-------|--------|----------------|
| validate_et0 | T1 | 31/31 | FAO-56 Tables 2.3/2.4, Example 18 Uccle within 0.0005 mm/day |
| validate_soil | T1 | 26/26 | Topp equation (7 points), inverse round-trip, 5 USDA textures, PAW |
| validate_iot | T1 | 11/11 | 168 records, 5 columns, CSV round-trip, diurnal statistics (synthetic data via `testutil::generate_synthetic_iot_data`) |
| validate_water_balance | T1 | 13/13 | Mass balance 0.0000 (3 scenarios), Ks bounds, MI summer |
| validate_sensor_calibration | T1 | 21/21 | SoilWatch 10 VWC, irrigation model, Dong 2024 field results |
| validate_real_data | T1 | 23/23 | Open-Meteo ERA5, 6+ MI stations, R²>0.85, capability-based discovery |
| cross_validate | T1/T2 | — | 75/75 Python↔Rust parity at 1e-5 |
| validate_dual_kc | T1 | 61/61 | FAO-56 Ch 7 Eqs 69/71-73/77, Table 17+19, multi-day sims |
| validate_cover_crop | T1 | 40/40 | FAO-56 Ch 11 mulch, 5 cover crops, no-till vs conventional |
| validate_regional_et0 | T1 | 61/61 | 6 MI stations, spatial CV, cross-station r, geographic consistency |
| validate_richards | T2 | 15/15 | van Genuchten θ/K/C, implicit Euler + Picard, Thomas algorithm |
| validate_biochar | T2 | 14/14 | Langmuir/Freundlich isotherms, wood + sugar beet biochar |
| validate_long_term_wb | T1 | 11/11 | 64-year Wooster OH, Hargreaves ET₀, decade trends |
| validate_yield | T1 | 32/32 | Stewart 1977, FAO-56 Table 24, multi-stage, WUE, scheduling |
| validate_cw2d | T2 | 24/24 | CW2D media (gravel, organic), VG retention, mass balance |
| validate_scheduling | T1 | 28/28 | 5 strategies, mass balance, yield ordering, WUE |
| validate_lysimeter | T1 | 25/25 | Mass-to-ET, temp compensation, calibration, diurnal |
| validate_sensitivity | T1 | 23/23 | OAT ±10%, 3 climatic zones, monotonicity, ranking |
| validate_priestley_taylor | T1 | 32/32 | PT α=1.26, analytical, cross-val vs PM, climate gradient |
| validate_et0_intercomparison | T1 | 36/36 | PM/PT/HG, 6 MI stations, R², bias, RMSE, coastal effects |
| validate_thornthwaite | T1 | 50/50 | Thornthwaite monthly ET₀, heat index, day-length |
| validate_gdd | T1 | 26/26 | GDD accumulation, kc_from_gdd, phenology |
| validate_pedotransfer | T1 | 58/58 | Saxton-Rawls 2006, θs/θr/Ks from texture |

**Total Rust: 651 tests + 1393 atlas checks PASS, 527 lib tests + 20 integration PASS**
**Phase 2 cross-validation: 75/75 MATCH (Python↔Rust, tol=1e-5)**
**Phase 3 GPU-first: 11 orchestrators wired, 4/4 ToadStool issues RESOLVED**
**Phase 3.5 NPU edge: AKD1000 live, 3 experiments, 95/95 NPU checks, ~48µs inference**
**Phase 3.7 metalForge mixed: CPU+GPU+NPU substrate routing, 14 eco workloads, 26 forge tests**
**CPU benchmarks: ET₀ 12.7M station-days/s, dual Kc 59M days/s, mulched Kc 64M days/s**
**Quality: zero `.unwrap()`, zero `panic!()`, zero `unsafe`, zero clippy pedantic warnings, all tolerances named `const`**

---

## Experiment Log

### Experiment 001: FAO-56 Penman-Monteith — PHASE 0 COMPLETE

**Goal**: Validate reference evapotranspiration (ET₀) calculation against
FAO Paper 56 published example values.

The Penman-Monteith equation is the foundational calculation for all irrigation
scheduling — analogous to Galaxy Bootstrap in wetSpring.

```
ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
```

**Phase 0 (Python baseline — 64/64 PASS):**
- [x] Digitize FAO-56 Examples 17, 18, 20 with all intermediate values (`benchmark_fao56.json`)
- [x] Implement FAO-56 PM in Python with numpy (`control/fao56/penman_monteith.py`)
- [x] Validate 11 saturation vapour pressures against FAO Table 2.3
- [x] Validate 10 slope values against FAO Table 2.4
- [x] Validate Example 17 (Bangkok monthly): ET₀ = 5.72 mm/day
- [x] Validate Example 18 (Uccle daily): ET₀ = 3.88 mm/day
- [x] Validate Example 20 (Lyon missing data): ET₀ = 4.56 mm/day

**Real data (918 station-days, R²=0.967):**
- [x] Download real 2023 Michigan weather from Open-Meteo (6 stations, 153 days each)
- [x] Compute ET₀ on real data (`control/fao56/compute_et0_real_data.py`)
- [x] Cross-check vs Open-Meteo's own ET₀: RMSE=0.267 mm/d, R²=0.967
- [x] Confirmed positive bias (+0.076 mm/d) from ERA5 reanalysis differences

**Rust (Phase 1 — 31/31 PASS, Phase 2 — 65/65 MATCH):**
- [x] Implement in Rust (`eco::evapotranspiration`) — 22 FAO-56 functions + Hargreaves, sunshine Rs, temp Rs, monthly G
- [x] Validate against FAO Paper 56 tables (31 checks in `validate_et0`)
- [x] Cross-validate: Python vs Rust identical outputs — 75/75 values match within 1e-5
- [x] Benchmark: Rust vs Python throughput (12.7M ET₀/s, `bench_cpu_vs_python`)

### Experiment 002: Soil Sensor Calibration — PHASE 0 COMPLETE

**Goal**: Reproduce dielectric permittivity → volumetric water content
calibration curves against Dong et al. (2020) Michigan soil data.

**Phase 0 (Python baseline — 36/36 PASS):**
- [x] Digitize Dong 2020 Tables 3, 4 (`benchmark_dong2020.json`)
- [x] Implement Topp equation in Python (`control/soil_sensors/calibration_dong2020.py`)
- [x] Implement RMSE/IA/MBE formulas matching paper's Eqs 1-3
- [x] Validate Topp equation against 8 published points
- [x] Validate correction equation fitting (linear/quad/exp/log) with R² > 0.65
- [x] Verify paper's conclusion: quadratic best for all soils
- [x] Confirm field RMSE improvements from Table 3 → corrected

**Rust (Phase 1 — 26/26 PASS, Phase 2 — cross-validated):**
- [x] Implement Topp equation in Rust (`eco::soil_moisture`)
- [x] Validate against published values (26 checks in `validate_soil`)
- [x] Cross-validate: Python vs Rust identical outputs — Topp values match within 1e-5

### Experiment 003: IoT Irrigation Pipeline — PHASE 0 COMPLETE

**Goal**: Replicate IoT-based precision irrigation system from
Dong et al. (2024) Frontiers in Water 6, 1353597.

**Phase 0 (Python + R — 24/24 PASS + R pending):**
- [x] Digitize SoilWatch 10 Eq 5, Table 2, field results (`benchmark_dong2024.json`)
- [x] Implement SoilWatch 10 calibration in Python (`control/iot_irrigation/calibration_dong2024.py`)
- [x] Implement irrigation recommendation model (Eq. 1)
- [x] Validate published RMSE/IA/MBE criteria (sand, loamy sand)
- [x] Validate blueberry yield significance (p=0.025, p=0.013)
- [x] Validate tomato water savings (30%, p>0.05)
- [x] Write R ANOVA script matching paper's R v4.3.1 (`anova_irrigation.R`)
- [ ] Run R ANOVA when R installed

**Rust (Phase 1 — 11/11 PASS + 21/21 sensor calibration):**
- [x] CSV time series parser in Rust (`io::csv_ts`)
- [x] SoilWatch 10 calibration in Rust (`eco::sensor_calibration`)
- [x] Irrigation recommendation model in Rust (`eco::sensor_calibration`)
- [x] `validate_sensor_calibration` binary: 21 checks against `benchmark_dong2024.json`
- [x] Cross-validate: Python vs Rust SoilWatch 10 + irrigation values match within 1e-5

### Experiment 004: Water Balance — PHASE 0 COMPLETE

**Goal**: Implement field-scale water balance model for irrigation scheduling
following FAO-56 Chapter 8.

**Phase 0 (Python baseline — 18/18 PASS):**
- [x] Implement FAO-56 soil water balance (`control/water_balance/fao56_water_balance.py`)
- [x] Validate TAW/RAW calculations for sandy loam, loam
- [x] Validate Ks stress coefficient bounds (5 cases)
- [x] Validate mass balance closure: dry-down, irrigated, heavy rain (all 0.000000)
- [x] Michigan summer: 90 days, 535 mm ET, 11 irrigation events

**Rust (Phase 1 — 13/13 PASS, Phase 2 — cross-validated):**
- [x] Implement water balance in Rust (`eco::water_balance`)
- [x] Validate mass conservation (13 checks in `validate_water_balance`)
- [x] Crop coefficient database (`eco::crop`) — FAO-56 Table 12, 10 crops
- [x] Full pipeline demo: `simulate_season` binary (crop Kc → soil → ET₀ → water balance → scheduling)
- [x] Cross-validate: Python vs Rust identical outputs for water balance computations

### Experiment 005: Lysimeter Validation — NOT STARTED (DEFERRED)

**Goal**: Reproduce ET measurements from weighing lysimeter load cell data
(Dong & Hansen, 2023). Deferred until supplementary data located.

### Experiment 006: HYDRUS Richards Equation — PHASE 0+1 COMPLETE

**Goal**: Implement pure 1D Richards equation solver (van Genuchten-Mualem),
validate against published HYDRUS results and analytical solutions.

**Phase 0 (Python baseline — 14/14 PASS):**
- [x] van Genuchten retention curve (4 analytical checks)
- [x] Mualem-van Genuchten conductivity (K(0)=Ks for 3 soils)
- [x] Sand infiltration: wetting front, surface θ → θs
- [x] Silt loam drainage: cumulative drainage, mass balance
- [x] Steady-state flux: Darcy's law for all soils

**Rust (Phase 1 — 15/15 PASS):**
- [x] `eco::richards` — van Genuchten θ(h), K(h), C(h)
- [x] Implicit Euler + Picard iteration solver
- [x] Thomas algorithm for tridiagonal systems
- [x] `validate_richards` binary: 15/15 checks

### Experiment 007: Biochar Isotherms — PHASE 0+1 COMPLETE

**Goal**: Fit Langmuir/Freundlich adsorption isotherms to published biochar
phosphorus adsorption data (Kumari, Dong & Safferman, 2025).

**Phase 0 (Python baseline — 14/14 PASS):**
- [x] Langmuir fit: qmax, KL, R² for wood and sugar beet biochar
- [x] Freundlich fit: KF, n, R² for both biochars
- [x] Model comparison: Langmuir R² > Freundlich R² for plateau data
- [x] Separation factor RL in (0,1) — favorable adsorption

**Rust (Phase 1 — 14/14 PASS):**
- [x] `eco::isotherm` — Langmuir, Freundlich models + linearized fitting
- [x] `validate_biochar` binary: 14/14 checks
- [x] Results match Python: wood qmax ≈ 18 mg/g, sugar qmax ≈ 16 mg/g

### Experiment 015: 60-Year Water Balance — PHASE 0+1 COMPLETE

**Goal**: Reconstruct 64 years (1960-2023) of growing season water balance
for Wooster, OH (OSU Triplett-Van Doren tillage study site) using Open-Meteo
ERA5 80-year archive and validated FAO-56 water balance.

**Phase 0 (Python baseline — 10/10 PASS):**
- [x] Download ERA5 data (64 growing seasons, May-Sep)
- [x] FAO-56 water balance for corn on Wooster silt loam
- [x] Physical reasonableness: ET₀ 400-800 mm, mass balance closure
- [x] Climate trends: positive ET₀ trend, precip CV 22%, decade stability
- [x] Cross-validation: ET/Precip ratio 0.6-1.8, irrigation in 100% of seasons

**Rust (Phase 1 — 11/11 PASS):**
- [x] Hargreaves ET₀ + existing water balance at 64-year scale
- [x] `validate_long_term_wb` binary: 11/11 checks
- [x] Rust Hargreaves vs Open-Meteo ET₀ within 1.4%

### Experiment 011: Cover Crop Dual Kc + No-Till — PHASE 0 COMPLETE

**Goal**: Extend dual Kc with cover crop coefficients and no-till mulch reduction
from FAO-56 Chapter 11. Digitize Islam et al. (2014) Brandt farm observations.

**Phase 0 (Python baseline — 40/40 PASS):**
- [x] 5 cover crop Kcb values (cereal rye, crimson clover, winter wheat, hairy vetch, radish)
- [x] 5 no-till mulch reduction levels (0.25–1.0)
- [x] Islam et al. (2014) observations: SOC, bulk density, infiltration, AWC
- [x] Rye→corn transition: 5-phase seasonal Kcb + mulch factor progression
- [x] No-till ET savings: 39.6% during initial stage (evaporation-dominated)
- [x] Mulch factor ordering and proportionality validated

### Experiment 010: Regional ET₀ Intercomparison — PHASE 0 COMPLETE

**Goal**: Validate FAO-56 PM ET₀ across Michigan microclimates using Open-Meteo ERA5
data. Establishes spatial variability baseline for GPU-batched ET₀ at scale.

**Phase 0 (Python baseline — 61/61 PASS):**
- [x] Compute ET₀ for 6 Michigan stations (918 station-days, 2023 growing season)
- [x] Per-station R² > 0.96 vs Open-Meteo, RMSE < 0.33 mm/day
- [x] Season totals 633–677 mm (matches MSU Enviro-weather references)
- [x] Spatial CV = 2.0% (expected for Lower MI stations)
- [x] 15 station-pair temporal correlations: r = 0.80–0.96
- [x] Geographic consistency: latitude span 1.55°, all in MI range

### Experiment 008: Yield Response to Water Stress — PHASE 0+1 COMPLETE

**Goal**: Implement the Stewart (1977) yield response model from FAO-56 Chapter 10.
Single-stage Ya/Ymax = 1 - Ky*(1 - ETa/ETc), multi-stage product formula (FAO-56 Eq. 90),
water use efficiency, Ky table (9 crops from FAO-56 Table 24), scheduling comparison.

**Phase 0 (Python baseline — 32/32 PASS):**
- [x] Ky table values (7 crops)
- [x] Single-stage Stewart equation (8 analytical)
- [x] Multi-stage product formula (5 analytical)
- [x] WUE calculations (4 crops)
- [x] Scheduling comparison (3 strategies × metrics + ordering checks)

**Rust (Phase 1 — 32/32 PASS, 16 unit tests):**
- [x] `eco::yield_response` — single, multi, WUE, ky_table
- [x] `validate_yield` binary: 32/32 checks

### Experiment 012: CW2D Richards Extension — PHASE 0+1 COMPLETE

**Goal**: Validate existing Richards solver on constructed wetland media
(Dong et al. 2019, HYDRUS CW2D). Extreme VG parameters (gravel Ks=5000 cm/day,
organic θs=0.60).

**Phase 0 (Python baseline — 24/24 PASS):**
- [x] VG retention curves for 4 CW2D media
- [x] Mualem conductivity for gravel + organic
- [x] Gravel infiltration (60 cm, 1 hour)
- [x] Organic substrate drainage
- [x] Mass balance checks

**Rust (Phase 1 — 24/24 PASS):**
- [x] Reuses `eco::richards` with CW2D media parameters
- [x] `validate_cw2d` binary: 24/24 checks

### Experiment 013: (Unassigned)

Number 013 is reserved. Skipped in original numbering to avoid confusion
with superseded experiment draft.

### Experiment 014: Irrigation Scheduling Pipeline — PHASE 0+1 COMPLETE

**Goal**: Complete "Penny Irrigation" pipeline comparison — 5 strategies
(fixed, threshold, MAD50, MAD30, rainfed) with deterministic weather.

**Phase 0 (Python baseline — 28/28 PASS):**
- [x] Deterministic sinusoidal ET₀ + periodic rainfall
- [x] 5 scheduling strategies × mass balance + yield ordering
- [x] WUE comparison, irrigation efficiency metrics

**Rust (Phase 1 — 28/28 PASS):**
- [x] `validate_scheduling` binary: 28/28 checks vs benchmark JSON
- [x] Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148

### Deferred: Agrivoltaics PAR — NOT STARTED

**Goal**: Model photosynthetically active radiation interception under solar
panel arrays for dual-use agriculture. Deferred until MSU Solar Farm data identified.
(Originally Experiment 008; renumbered to avoid conflict with Yield Response.)

### Experiment 016: Lysimeter ET Direct Measurement — PHASE 0+1 COMPLETE

**Goal**: Validate lysimeter mass-to-ET conversion pipeline — load cell
calibration, thermal drift correction, rain rejection, diurnal patterns.

**Phase 0 (Python baseline — 22/22 PASS):**
- [x] Mass-to-ET conversion, temperature compensation
- [x] Data quality filtering, load cell calibration
- [x] Hourly diurnal ET pattern, daily comparison vs ET₀

**Rust (Phase 1 — 22/22 PASS):**
- [x] `validate_lysimeter` binary: 22/22 checks vs benchmark JSON
- [x] Dong & Hansen (2023) Smart Ag Tech 4:100147

### Experiment 017: ET₀ Sensitivity Analysis — PHASE 0+1 COMPLETE

**Goal**: OAT (one-at-a-time) ±10% perturbation of 6 input variables
across 3 climatic conditions (humid, arid, continental).

**Phase 0 (Python baseline — 30/30 PASS):**
- [x] Baseline ET₀ for 3 climates
- [x] Monotonicity, elasticity bounds, symmetry
- [x] Ranking consistency (Gong et al. 2006)

**Rust (Phase 1 — 30/30 PASS):**
- [x] `validate_sensitivity` binary: 30/30 checks vs benchmark JSON
- [x] Allen et al. (1998) FAO-56 Ch 4; Gong et al. (2006)

### Experiment 018: Michigan Crop Water Atlas — ACTIVE

**Goal**: Run the validated ET₀ + water balance + yield response pipeline
across 100 Michigan stations, 10 crops, and up to 80 years of Open-Meteo ERA5 data.

**Data**: Open-Meteo ERA5 archive (free, no API key). 100 stations defined in `specs/ATLAS_STATION_LIST.md`.

**Phase 0 (Python baseline):**
- [x] `control/atlas/atlas_water_budget.py` — 10-crop pipeline for cross-validation
- [x] FAO-56 ET₀ + water balance + Stewart yield (reuses validated Exp 001/004/008)

**Phase 1 (Rust):**
- [x] `validate_atlas` binary — 1354/1354 checks on 100-station full Michigan atlas
- [x] Station discovery from filesystem (capability-based)
- [x] Per-station ET₀ R² > 0.96, mass balance = 0.000 mm
- [x] 10-crop yield ratios in [0.3, 1.0] range
- [x] CSV output: `atlas_station_summary.csv`, `atlas_crop_summary.csv`

**Phase 1.5 (100-station pilot — COMPLETE):**
- [x] Downloaded 100 stations × 2023 growing season (15,300 station-days)
- [x] 1354/1354 Rust checks PASS (ValidationHarness checks)
- [x] Python cross-validation: 690 crop-station yield pairs within 0.01 (mean diff 0.0003)
- [x] Mean ET₀ diff Python vs Rust: 0.133% across 69 matched stations
- [x] Statewide mean ET₀ = 640 mm (growing season 2023)
- [x] Processing time: 141s in release mode

**Phase 2 (Scale-out):**
- [ ] Download 100 stations × 80yr data (`python scripts/download_atlas_80yr.py`)
- [ ] Run atlas pipeline at full 80yr scale (est. ~2hr CPU)
- [ ] Decade trend analysis (1945-2024 ET₀ and yield trends)
- [ ] Spatial interpolation via `gpu::kriging` (Phase 3)

### Experiment 019: Priestley-Taylor ET₀ — PHASE 0+1 COMPLETE

**Goal**: Implement radiation-based Priestley-Taylor ET₀ (α=1.26) and
cross-validate against Penman-Monteith. Fills the radiation-only method gap
in the ET₀ portfolio.

**Phase 0 (Python baseline — 32/32 PASS):**
- [x] PT analytical checks (zero Rn, typical summer, negative clamped)
- [x] Uccle cross-validation: PT/PM ratio in [0.85, 1.25] (Xu & Singh 2002)
- [x] Climate gradient: 5 sites (arid→humid), PT tracks PM
- [x] Monotonicity: increasing Rn → increasing ET₀
- [x] Temperature sensitivity: increasing T → increasing ET₀

**Rust (Phase 1 — 32/32 PASS):**
- [x] `eco::evapotranspiration::priestley_taylor_et0()` — pure Rust PT
- [x] `validate_priestley_taylor` binary: 32/32 checks
- [x] 8 new unit tests (zero Rn, clamp, range, mono, temp, altitude, G, cross-val)

### Experiment 020: ET₀ 3-Method Intercomparison — PHASE 0+1 COMPLETE

**Goal**: Compare Penman-Monteith, Priestley-Taylor, and Hargreaves-Samani
on real Open-Meteo ERA5 data for 6 Michigan stations (2023 growing season).

**Phase 0 (Python baseline — 36/36 PASS):**
- [x] Per-station R², bias, RMSE for PT vs PM and HG vs PM
- [x] PM vs Open-Meteo R² > 0.95
- [x] PT vs PM R² > 0.70 (coastal lake-effect variability per Droogers & Allen 2002)
- [x] HG vs PM R² > 0.55 (temperature-only limitation)
- [x] All methods produce physically reasonable totals (400-800 mm/season)

**Rust (Phase 1 — 36/36 PASS):**
- [x] `validate_et0_intercomparison` binary: 36/36 checks
- [x] Metrics match Python baseline within documented tolerances

### Experiment 021: Thornthwaite Monthly ET₀ — PHASE 0+1 COMPLETE

**Goal**: Implement Thornthwaite (1948) temperature-based monthly ET₀ for
data-sparse applications (heat index, day-length correction).

**Phase 0 (Python baseline — 23/23 PASS):**
- [x] Heat index, monthly correction factor, day-length
- [x] Analytical checks, climate gradient, monotonicity

**Rust (Phase 1 — 50/50 PASS):**
- [x] `eco::evapotranspiration::thornthwaite_monthly_et0()`
- [x] `validate_thornthwaite` binary: 50/50 checks

### Experiment 022: Growing Degree Days (GDD) — PHASE 0+1 COMPLETE

**Goal**: Implement GDD accumulation and Kc-from-GDD for phenology-driven
irrigation scheduling.

**Phase 0 (Python baseline — 33/33 PASS):**
- [x] gdd_avg, gdd_clamp, accumulated_gdd_avg, kc_from_gdd
- [x] Base temperature variants, accumulation, crop-specific curves

**Rust (Phase 1 — 26/26 PASS):**
- [x] `eco::crop::gdd_avg()`, `gdd_clamp()`, `accumulated_gdd_avg()`, `kc_from_gdd()`
- [x] `validate_gdd` binary: 26/26 checks

### Experiment 023: Pedotransfer Functions (Saxton-Rawls 2006) — PHASE 0+1 COMPLETE

**Goal**: Implement Saxton-Rawls (2006) pedotransfer for soil hydraulic
properties (θs, θr, Ks) from texture (sand, clay, organic matter).

**Phase 0 (Python baseline — 70/70 PASS):**
- [x] θs, θr, Ks from sand/clay/OM
- [x] 12 USDA textures, retention curve consistency, conductivity ordering

**Rust (Phase 1 — 58/58 PASS):**
- [x] `eco::soil_moisture::saxton_rawls()`
- [x] `validate_pedotransfer` binary: 58/58 checks

---

### Experiment 024: NASS Yield Validation — PHASE 0+1 COMPLETE

**Goal**: Validate the full airSpring pipeline (ET₀ → water balance → Stewart
yield response) against physically consistent targets using Michigan crops,
soils, and climate. Prepares infrastructure for scoring against real USDA NASS
county-level yields when API access is available.

**Phase 0 (Python baseline — 41/41 PASS):**
- [x] Ky table consistency for 5 Michigan crops (FAO-56 Table 24)
- [x] Drought response monotonicity (normal → mild → moderate → severe)
- [x] Soil sensitivity (sandy_loam < loam < clay_loam under drought)
- [x] Multi-year variability (20 years, corn on loam)
- [x] Crop ranking (soybean > corn under drought, all in [0,1])
- [x] Mass balance conservation (ETa ≤ ETc)

**Rust (Phase 1 — 40/40 PASS):**
- [x] `eco::yield_response` extended: `winter_wheat`, `dry_bean` added to `ky_table`
- [x] `validate_nass_yield` binary: 40/40 checks
- [x] NASS download script ready (`scripts/download_usda_nass.py`)

---

### Experiment 025: Forecast Scheduling Hindcast — PHASE 0+1 COMPLETE

**Goal**: Evaluate 5-day weather forecast-driven irrigation scheduling vs
perfect-knowledge scheduling (Exp 014). Tests forecast degradation, horizon
sensitivity, and mass balance under stochastic forecast noise.

**Phase 0 (Python baseline — 19/19 PASS):**
- [x] Forecast vs perfect knowledge (yield gap, irrigation ratio)
- [x] Noise sensitivity (low → extreme)
- [x] Horizon impact (1, 3, 5, 7 days)
- [x] Mass balance conservation
- [x] Forecast vs rainfed (yield improvement, stress reduction)

**Rust (Phase 1 — 19/19 PASS):**
- [x] `validate_forecast` binary: 19/19 checks
- [x] Deterministic RNG for reproducibility

---

### Experiment 026: USDA SCAN Soil Moisture Validation — PHASE 0+1 COMPLETE

**Goal**: Validate Richards 1D solver against published USDA SCAN soil moisture
profiles and Carsel & Parrish (1988) van Genuchten parameters for 3 representative
Michigan soil textures (sand, silt loam, clay).

**Phase 0 (Python baseline — 34/34 PASS):**
- [x] VG retention curves match analytical Eq. 1 (8 test cases)
- [x] Mualem-VG conductivity K/Ks ratios (6 test cases)
- [x] Richards solver bounded θ profiles (3 soils × 3 checks)
- [x] Ks and K(h) ordering: sand > silt_loam > clay
- [x] Seasonal θ within SCAN-published ranges (3 soils × 2 seasons)
- [x] Depth-dependent response to infiltration

**Rust (Phase 1 — 34/34 PASS):**
- [x] `validate_scan_moisture` binary: 34/34 checks
- [x] Uses `eco::richards` + `eco::van_genuchten` library modules

---

### Experiment 027: Multi-Crop Water Budget Validation — PHASE 0+1 COMPLETE

**Goal**: Exercise full FAO-56 pipeline (ET₀ → dual Kc → water balance → Stewart
yield response) across 5 major Michigan crops with deterministic synthetic weather.
Each crop×season is an independent GPU-parallelizable work unit.

**Phase 0 (Python baseline — 47/47 PASS):**
- [x] Single Kc irrigated water balance (5 crops × 2 checks)
- [x] Rainfed scenario: stress, yield, zero irrigation (5 crops × 3 checks)
- [x] Drought hierarchy: Potato drop > WinterWheat (shallow roots + high Ky)
- [x] Irrigated yield >= rainfed for all 5 crops
- [x] Dual Kc evaporation layer Ke > 0 (5 crops × 2 checks)
- [x] Crop-water productivity ETa/yield in [200, 1200] mm

**Rust (Phase 1 — 47/47 PASS):**
- [x] `validate_multicrop` binary: 47/47 checks
- [x] Uses `eco::water_balance` + `eco::yield_response` library modules

---

### Experiment 028: NPU Edge Inference for Agriculture — PHASE 1 COMPLETE

**Goal**: Integrate BrainChip AKD1000 NPU via ToadStool `akida-driver` for edge
inference workloads in agricultural monitoring. Validates int8 quantization fidelity,
feature encoding for 3 crop/irrigation/anomaly classifiers, and live DMA round-trips.

**metalForge forge crate** (`metalForge/forge/` — 26 tests, 21 validation checks):
- [x] Substrate abstraction: CPU, GPU, NPU runtime discovery
- [x] Capability-based dispatch: GPU > NPU > CPU priority routing
- [x] 14 eco workloads: 9 GPU-absorbed, 3 NPU-native, 2 CPU-only
- [x] Live inventory: i9-12900K, RTX 4070, TITAN V, AKD1000 all discovered
- [x] All NPU workloads route to AKD1000 (quant int8)
- [x] All GPU workloads route to RTX 4070 (f64 + shader)

**barracuda NPU module** (`npu.rs`, feature-gated — 35/35 checks):
- [x] `akida-driver` dependency (optional, `--features npu`)
- [x] `NpuHandle`: discover, load, infer, raw DMA
- [x] int8 quantization round-trip: <0.01 error on [0,1], <3mm on [0,300]
- [x] Crop stress classifier: 4 features (depletion, ETa/ETc, θ, Ks) → 2 classes
- [x] Irrigation decision: 6 features (ET₀, θ, TAW, stage, Ks, rain_prob) → 3 classes
- [x] Sensor anomaly: 3 features (reading, mean, σ) → 2 classes
- [x] Live AKD1000: 80 NPs, 10 MB SRAM, 0.5 GB/s PCIe
- [x] DMA inference: ~84µs crop stress, ~64µs irrigation decision

---

### Experiment 029: Funky NPU for Agricultural IoT — PHASE 1 COMPLETE

**Goal**: Demonstrate advanced AKD1000 capabilities for Dong's LOCOMOS-style
field-deployed IoT systems. Bridge from lab compute to edge-sovereign agriculture.

**S1 — Streaming Soil Moisture (6/6 PASS):**
- 500-step synthetic sensor stream at 15-min cadence (5-day irrigation cycle)
- Semi-trained FC classifier: normal/stressed/anomaly detection
- CPU throughput: 2.6M Hz; **live AKD1000: 20,545 Hz, mean 48.7 µs, P99 68.9 µs**

**S2 — Seasonal Weight Evolution (4/4 PASS):**
- (1+1)-ES adapts crop stress weights across 3 seasonal phases (early/mid/late)
- Fitness: 47–61% → 96–98% (monotonically non-decreasing)
- Demonstrates online readout adaptation without full retraining

**S3 — Multi-Crop Crosstalk (6/6 PASS):**
- Rapid switching between corn/soybean/potato classifiers (100 rounds × 3 crops)
- All responses deterministically stable (CPU); DMA path verified (live NPU)
- Crops produce distinct responses — no classifier confusion

**S4 — LOCOMOS Power Budget (7/7 PASS):**
- 96 readings/day at 15-min cadence, Pi Zero 2 W + AKD1000
- Daily energy: 2.53 Wh (505 mAh @ 5V) — 18650 battery feasible
- 5W solar panel provides 20 Wh/day — 8× surplus
- **NPU saves 10.7× active energy vs cloud round-trip**
- NPU energy: 0.0009% of active cycle — negligible
- Cost breakeven: 20 months ($99 NPU vs $60/yr cloud)

**S5 — Noise Resilience (3/3 PASS):**
- Anderson-style sweep: σ = 0.000 to 0.150 VWC
- Classification robust across all noise levels (74.5–78.5%)

**Live AKD1000 (6/6 PASS):** 500 streaming DMA round-trips, P99 < 1 ms

---

### Experiment 029b: High-Cadence NPU Streaming Pipeline — PHASE 1 COMPLETE

**Goal**: Build out the cadence revolution — since NPU inference costs 0.0009%
of active cycle energy, increase sensor cadence from 15-min to 1-min and burst.

**S1 — Multi-Sensor Fusion (4/4 PASS):**
- 6-feature input (θ + T + EC + depletion + hour + days_since_irr) → 4 classes
- Single inference classifies full field state (normal/water_stress/salt_stress/anomaly)

**S2 — 1-Minute Cadence (5/5 PASS):**
- 1,440 readings/day (full 24-hour simulation at 1-min intervals)
- **Daily energy at 1-min: 2.6 Wh** (NPU share: 0.0009%)
- 13 state transitions detected across diurnal cycle

**S3 — Burst Mode (4/4 PASS):**
- 180 readings at 10-sec intervals (30-min irrigation event)
- Infiltration front detected (θ̄ rises from 0.20 to 0.26)

**S4 — Ensemble Classification (3/3 PASS):**
- 10 weight sets per reading, consensus voting
- Projected NPU ensemble: 480 µs (10 × 48 µs)

**S5 — Sliding Window Anomaly (2/2 PASS):**
- 60-reading buffer, 3-consecutive trigger threshold
- Correctly fires on 3-glitch sequence, ignores single glitch

**S6 — Weight Hot-Swap (3/3 PASS):**
- 5 crops (corn/soybean/potato/tomato/blueberry) × 50 rounds
- Projected NPU: 540 µs for 5-crop round-robin

**Live AKD1000 (7/7 PASS):**
- 1,440 DMA round-trips: mean 47.6 µs, P99 64.2 µs, 21,023 Hz
- Weight hot-swap: 23.5 µs mean per crop load

---

### Experiment 030: AmeriFlux Eddy Covariance ET — PHASE 0+1 COMPLETE

**Goal**: Validate airSpring FAO-56 PM ET₀ against direct eddy covariance ET
measurements from the AmeriFlux network (Baldocchi 2003). Provides ground truth
from energy balance closure — the "gold standard" for ET validation.

**Phase 0 (Python baseline — 27/27 PASS):**
- [x] AmeriFlux flux tower data parsing and quality filtering
- [x] Energy balance closure checks (Rn - G ≈ H + LE)
- [x] ET₀ vs eddy covariance ET comparison (R², RMSE, bias)
- [x] Seasonal and diurnal pattern analysis

**Rust (Phase 1 — 27/27 PASS):**
- [x] `validate_ameriflux` binary: 27/27 checks
- [x] Uses `eco::evapotranspiration` FAO-56 PM + Hargreaves

---

### Experiment 031: Hargreaves-Samani Temperature ET₀ — PHASE 0+1 COMPLETE

**Goal**: Validate temperature-only ET₀ estimation (Hargreaves & Samani 1985,
FAO-56 Eq. 52) for data-sparse environments where full PM inputs are unavailable.

**Phase 0 (Python baseline — 24/24 PASS):**
- [x] Hargreaves ET₀ analytical checks (climate gradient)
- [x] Cross-validation vs PM (R², bias correction)
- [x] Temperature sensitivity and latitude effects
- [x] Monthly and seasonal totals

**Rust (Phase 1 — 24/24 PASS):**
- [x] `validate_hargreaves` binary: 24/24 checks
- [x] Uses `eco::evapotranspiration::hargreaves_et0()`

---

### Experiment 032: Ecological Diversity Indices — PHASE 0+1 COMPLETE

**Goal**: Validate ecological diversity metrics (Shannon, Simpson, Chao1,
Pielou evenness, Bray-Curtis dissimilarity, rarefaction curves) for agroecosystem
assessment — cover crop biodiversity, soil microbiome, field margin evaluation.

**Phase 0 (Python baseline — 22/22 PASS):**
- [x] Shannon, Simpson, richness index validation
- [x] Pielou evenness (normalized Shannon)
- [x] Bray-Curtis dissimilarity (pairwise)
- [x] Rarefaction curve generation

**Rust (Phase 1 — 22/22 PASS):**
- [x] `validate_diversity` binary: 22/22 checks
- [x] Uses `eco::diversity` (wired to `barracuda::stats::diversity` from wetSpring S64)

---

### Experiment 046: Atlas Stream Real Data Validation — PHASE 1 COMPLETE

**Goal**: Wire partially downloaded 80-year Open-Meteo ERA5 station data through
the new `AtlasStream` and `SeasonalPipeline` GPU orchestrators. End-to-end
integration test for the full airSpring agricultural pipeline on real data.

**Phase 1 (Rust — 73/73 PASS):**
- [x] Discover 80yr CSVs in `data/open_meteo/` (12 stations available)
- [x] Parse Open-Meteo CSVs by column name → `WeatherDay` structs
- [x] Filter to growing season (DOY 121-273), batch by station
- [x] Process through `AtlasStream::new().process_batch()` with 5 crop types
- [x] Validate: ≥1 station, ≥50 valid seasons, result count = stations × crops
- [x] Mass balance < 1 mm (observed ~2e-13 mm)
- [x] Mean yield ratio in [0.3, 1.0], mean daily ET₀ in [2, 8] mm
- **12 stations**, **4800 crop-year results**, **73/73 PASS**

**Binary**: `validate_atlas_stream`

---

### Experiment 009: FAO-56 Dual Crop Coefficient — PHASE 0 COMPLETE

**Goal**: Implement the dual crop coefficient approach (Kcb + Ke) from FAO-56
Chapter 7, separating transpiration from soil evaporation for precision scheduling.

**Phase 0 (Python baseline — 63/63 PASS):**
- [x] Digitize FAO-56 Table 17 Kcb values (10 crops: corn, soybean, wheat, alfalfa, tomato, potato, sugar beet, dry bean, blueberry, turfgrass)
- [x] Digitize FAO-56 Table 19 REW/TEW values (11 USDA soil types)
- [x] Implement Eq. 69 (ETc dual), Eq. 71/72 (Kc_max, Kr), Eq. 73 (TEW)
- [x] Implement evaporation layer water balance (Eq. 77)
- [x] Validate Kcb + evaporation ≈ Kc single (Table 17 vs Table 12 consistency)
- [x] Validate TEW > REW for all soil types
- [x] 7-day bare soil drydown simulation (stage 1 → stage 2 evaporation)
- [x] 5-day corn mid-season simulation (ETc/ET₀ ≈ Kcb under full cover)

---

### Experiment 057: GPU Ops 5-8 Rewire Validation + Benchmark — PHASE 1 COMPLETE

**Goal**: Validate the ToadStool S70+ absorption rewire — all 6 batched elementwise
GPU ops (0, 1, 5, 6, 7, 8) dispatched and cross-validated against CPU baselines,
with timing benchmarks and cross-spring evolution provenance tracking.

**Phase 1 (Rust — 26/26 PASS):**
- [x] Op 0 (FAO-56 ET₀): GPU dispatch, N=1000, range validation
- [x] Op 1 (Water Balance): GPU dispatch, N=1000, non-negative validation
- [x] Op 5 (SensorCal): GPU↔CPU max err < 0.01, VWC(10000) ≈ 0.1323
- [x] Op 6 (Hargreaves): GPU↔CPU max err < 0.10 (NVK polyfill acos drift)
- [x] Op 7 (Kc Climate): GPU↔CPU max err < 0.01, standard conditions ≈ 1.20
- [x] Op 8 (DualKc Ke): All Ke ∈ [0, 1.5), N=1000 valid
- [x] GPU throughput scaling: 64K→11.5M items/s (N=100→50K)
- [x] Seasonal pipeline: GPU ET₀ parity < 1%, yield parity < 5%, mass balance < 0.5mm
- [x] Cross-spring provenance documented (hotSpring/wetSpring/neuralSpring/airSpring/groundSpring)

**Binary**: `validate_gpu_rewire_benchmark`

---

### Experiment 058: Climate Scenario Analysis — PHASE 1 COMPLETE

**Goal**: Validate climate scenario pipeline — synthetic climate perturbations
across ET₀, water balance, and yield response for scenario-based planning.

**Phase 1 (Rust — 46/46 PASS):**
- [x] Climate scenario generation and perturbation
- [x] ET₀ → water balance → yield pipeline under scenarios
- [x] Mass balance, physical bounds, cross-scenario consistency

**Binary**: `validate_climate_scenario`

---

### Experiment 079: Monte Carlo ET₀ Uncertainty Propagation — PHASE 0+1 COMPLETE

**Goal**: Propagate measurement uncertainties through the FAO-56 Penman-Monteith
equation via Monte Carlo simulation. Validates the `gpu::mc_et0` CPU and GPU
paths against Python baselines.

**Phase 0 (Python baseline — 12/12 PASS):**
- [x] MC propagation through FAO-56 PM (N=2000, seed=42)
- [x] Zero uncertainty → zero spread
- [x] High uncertainty → wider CI
- [x] Arid vs humid climate gradient
- [x] Convergence analysis (std stabilises with N)

**Phase 1 (Rust — 26/26 PASS):**
- [x] `gpu::mc_et0::mc_et0_cpu` — deterministic MC propagation
- [x] `validate_mc_et0` binary: 26/26 checks vs benchmark JSON
- [x] Cross-climate: arid ET₀ > humid ET₀, plausible ranges
- [x] Parametric CI consistent with empirical percentiles
- [x] Determinism: same seed → identical output

**Binary**: `validate_mc_et0`

---

### Experiment 080: Bootstrap & Jackknife CI for Seasonal ET₀ — PHASE 0+1 COMPLETE

**Goal**: Validate GPU-accelerated bootstrap mean CI and leave-one-out
jackknife variance estimation on seasonal ET₀ data (153-day Michigan
growing season).

**Phase 0 (Python baseline — 18/18 PASS):**
- [x] Bootstrap 95% CI for mean ET₀ (B=1000)
- [x] Jackknife mean variance (leave-one-out)
- [x] Known analytical: [1..10] mean=5.5, jackknife var≈0.917
- [x] Small sample wider CI, constant data zero variance

**Phase 1 (Rust — 20/20 PASS):**
- [x] `gpu::bootstrap::GpuBootstrap` (CPU path) — bootstrap CI
- [x] `gpu::jackknife::GpuJackknife` (CPU path) — jackknife variance
- [x] `validate_bootstrap_jackknife` binary: 20/20 checks
- [x] Bootstrap/Jackknife SE agreement (ratio ≈ 1.0)

**Binary**: `validate_bootstrap_jackknife`

---

### Experiment 081: Standardized Precipitation Index (SPI) — PHASE 0+1 COMPLETE

**Goal**: Implement the McKee et al. (1993) SPI drought classification
algorithm. New `eco::drought_index` module with gamma MLE fitting and
standard normal transformation for multi-scale drought analysis.

**Phase 0 (Python baseline — 17/17 PASS):**
- [x] SPI-1, SPI-3, SPI-6, SPI-12 on 5-year synthetic Michigan precip
- [x] Gamma MLE fit (Thom 1958 approximation)
- [x] WMO drought classification (7 categories)
- [x] Scale ordering: SPI-3 smoother than SPI-1

**Phase 1 (Rust — 20/20 PASS):**
- [x] `eco::drought_index` — `compute_spi()`, `gamma_mle_fit()`, `DroughtClass`
- [x] `validate_drought_index` binary: 20/20 checks
- [x] SPI values ≥90% within 0.1 of Python baseline
- [x] Mean ≈ 0, std ≈ 1 (standard normal property)

**Binary**: `validate_drought_index`

---

## Evolution Roadmap

```
Track 1 (Precision Agriculture):
  Phase 0  [COMPLETE]: Python baselines — 1284/1284 PASS (57 experiments)
  Phase 0+ [COMPLETE]: Real data pipeline — 15,300 station-days, ET₀ R²=0.97
  Phase 1  [COMPLETE]: Rust validation — 854 lib + 62 forge tests, 89 binaries
  Phase 1.5[COMPLETE]: CPU benchmark — Rust 14.5× faster than Python (21/21 parity)
  Phase 2  [COMPLETE]: Cross-validation — 75/75 MATCH (Python↔Rust, tol=1e-5)
  Phase 2.5[COMPLETE]: Ops 5-8 GPU-first — 4 orchestrators rewired (ToadStool S70+ absorbed)
  Phase 2.6[COMPLETE]: Seasonal pipeline — GPU Stages 1-2 (ET₀ + Kc), 73/73 real data (12 stations)
  Phase 3  [COMPLETE]: GPU bridge — 15 Tier A modules wired to ToadStool primitives
  Phase 3.5[COMPLETE]: NPU edge — AKD1000 live, 3 experiments, ~48µs inference
  Phase 3.7[COMPLETE]: metalForge mixed — CPU+GPU+NPU substrate routing, 18 eco workloads
  Phase 3.8[COMPLETE]: Cross-system routing — 29/29 PASS, NUCLEUS atomic ready
  Phase 3.9[COMPLETE]: Mixed pipeline — NPU→GPU PCIe bypass, NUCLEUS atomics, biomeOS graphs
  Phase 3.9+[COMPLETE]:NUCLEUS primal — 16 capabilities, ecology domain, 29/29 parity (Exp 062)
  Phase 4.0[COMPLETE]: Cross-primal pipeline — capability.call routing, 28/28 PASS (Exp 063)
  Phase 4.1:           Penny irrigation (sovereign, consumer hardware)

Track 2 (Environmental Systems):
  Phase B0 [COMPLETE]: HYDRUS Richards + biochar isotherms + CW2D (Python + Rust)
  Phase B1:            Contaminant transport + adsorption (scipy validation)
  Phase B2:            GPU acceleration (field-scale PDE, Monte Carlo)
  Phase B3:            Sovereign remediation monitoring
```

### GPU Acceleration Targets — Track 1

| Pipeline Stage | Current Tool | GPU Potential | Why |
|---------------|-------------|:------------:|-----|
| ET₀ computation (hourly) | Python/Excel | **High** | Embarrassingly parallel across stations/fields |
| Sensor calibration | scipy curve_fit | **Medium** | Nonlinear least squares on GPU |
| Spatial interpolation (kriging) | R gstat / Python pykrige | **High** | Large matrix operations → ToadStool |
| Water balance (field grid) | Spreadsheet | **High** | Per-cell parallel on GPU |
| IoT stream processing | Python/Node-RED | **High** | Real-time edge inference |
| Irrigation scheduling | Custom | **High** | Model-predictive control on GPU |

### GPU Acceleration Targets — Track 2

| Pipeline Stage | Current Tool | GPU Potential | Why |
|---------------|-------------|:------------:|-----|
| Richards equation solver | HYDRUS (Fortran) | **High** | Finite element/volume on GPU mesh |
| Advection-dispersion | HYDRUS/MODFLOW | **High** | Stencil operations → GPU native |
| Adsorption isotherms | scipy curve_fit | **Medium** | Batch fitting across materials |
| Monte Carlo UQ | Custom Python | **High** | Massively parallel → GPU |
| Signal propagation (LPWAN) | MATLAB ray tracing | **High** | Ray tracing is GPU's raison d'etre |

### Shared GPU Kernels (Cross-Spring)

| Kernel | airSpring Use | wetSpring Use | hotSpring Use |
|--------|-------------|-------------|-------------|
| ODE/PDE solver | Richards equation | — | HFB eigensolve |
| Time series filter | Sensor smoothing | LC-MS chromatogram | — |
| Nonlinear fitting | Calibration curves, isotherms | Peak fitting | SEMF optimization |
| Spatial interpolation | Kriging (soil moisture) | — | — |
| Monte Carlo | Uncertainty quantification | Rarefaction | Nuclear EOS |
| Matrix operations | Covariance (kriging) | Distance matrices | Overlap matrices |
| Reduction | Temporal aggregation | Peak areas | Binding energy sums |

---

## Relationship to wetSpring

wetSpring and airSpring share the same agricultural/environmental ecosystem:

| Aspect | wetSpring | airSpring |
|--------|-----------|-----------|
| Focus | Points in a system | The system itself |
| Samples | Water samples, microbiome, analytes | Fields, soil profiles, sensor networks |
| Chemistry | PFAS detection, LC-MS metabolomics | Soil chemistry, nutrient transport |
| Biology | Microbial communities (16S) | Crop physiology, ET |
| Physics | — | Fluid dynamics (Richards eq), radiation (PAR) |
| IoT | — | Soil moisture, weather, leaf wetness sensors |
| Dr. Dong's water quality work | **In wetSpring** (analyte data) | — |
| Dr. Dong's irrigation/sensing | — | **In airSpring** (system data) |
| Dr. Jones's PFAS work | **In wetSpring** (LC-MS, screening) | Expanded by airSpring (soil remediation) |

---

*Initialized: February 16, 2026 — Updated: March 7, 2026 (v0.7.4)*
*81 experiments, 1284/1284 Python, 854 lib + 186 forge tests, 89 binaries, 381/381 validation, 146/146 evolution, 33/33 cross-validation, 20.6× CPU speedup (24/24 parity), barraCuda 0.3.3 (wgpu 28), fused Welford + Pearson wired.*
*8 ET₀ methods + SCS-CN runoff + Green-Ampt infiltration + coupled runoff-infiltration + VG inverse + full-season WB + Exp 058 Climate Scenario (46/46).*
*NUCLEUS primal (16 capabilities, 28/28 cross-primal pipeline). Atlas decade 80yr (102/102). NASS real (99/99). NCBI diversity (63/63).*
*25 Tier A + 6 GPU-local modules. Ops 5-8 GPU-first (ToadStool S87). GPU stats (neuralSpring S69).*
*Seasonal pipeline GPU Stages 1-3. 73/73 atlas PASS (12 stations, 4800 results). 146/146 + 32/32 cross-spring benchmarks (Exp 077). Exp 064-069 immunological Anderson (Paper 12).*
*metalForge 27 workloads, 29/29 cross-system. AKD1000 NPU live (3 experiments).*
*Quality: zero .unwrap(), zero unsafe, zero clippy pedantic + nursery warnings. AGPL-3.0-or-later.*
