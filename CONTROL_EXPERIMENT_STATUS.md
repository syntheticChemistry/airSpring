# airSpring Control Experiment — Status Report

**Date**: 2026-02-16 (Project initialized)
**Updated**: 2026-02-16 (BarraCUDA Rust validation — 70/70 PASS)
**Gate**: Eastgate (i9-12900K, 64 GB DDR5, RTX 4070 12GB, Pop!_OS 22.04)
**License**: AGPL-3.0-or-later

---

## Replication Protocol

Anyone can reproduce all results by:

```bash
git clone git@github.com:syntheticChemistry/airSpring.git
cd airSpring

# 1. Install Python baselines (FAO-56, scipy, pandas)
pip install -r control/requirements.txt

# 2. Download public datasets (NOAA, MSU AgWeather, USDA)
./scripts/download_data.sh --all

# 3. Run Python baselines
python control/fao56/penman_monteith.py
python control/iot_sensors/parse_agweather.py

# 4. Run Rust validation binaries
cd barracuda && cargo run --release --bin validate_et0
```

No institutional access required. All data is from public repositories
(NOAA CDO, USDA Web Soil Survey, MSU Enviro-weather, published papers).
All tools are open source.

---

## Hardware Gate

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (16C/24T, 5.2 GHz) |
| RAM | 64 GB DDR5-4800 |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| Storage | 1 TB NVMe SSD |
| OS | Pop!_OS 22.04 (Ubuntu-based) |
| Docker | 24.x with Compose v2 |

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

### Track 1: Precision Agriculture

| ID | Dataset | Source | Size | Status |
|----|---------|--------|------|--------|
| D1 | FAO-56 Paper 56 reference tables | FAO | ~1 MB | Available |
| D2 | MSU Enviro-weather stations | MSU AgWeather | ~50 MB | Public API |
| D3 | Michigan soil sensor benchmarks | Dong et al. 2020 | ~5 MB | Paper suppl. |
| D4 | USDA Web Soil Survey | NRCS/USDA | ~10 MB | Public |
| D5 | Lysimeter ET data | Dong & Hansen 2023 | ~2 MB | Paper suppl. |
| D6 | Irrigation meta-analysis data | Ali et al. 2024 | ~1 MB | Paper suppl. |

### Track 2: Environmental Systems

| ID | Dataset | Source | Size | Status |
|----|---------|--------|------|--------|
| D7 | HYDRUS CW2D examples | PC-Progress | ~5 MB | Available |
| D8 | Biochar adsorption data | Kumari et al. 2025 | ~1 MB | Paper suppl. |
| D9 | NOAA CDO Michigan weather | NOAA CDO | ~20 MB | Public |
| D10 | Agrivoltaic PAR data | MSU Solar Farm | ~5 MB | To identify |
| D11 | EPA PFAS soil data | EPA ORD | ~2 MB | Public |

---

## Run Log

### 2026-02-16: Project Initialization

- Created airSpring repository
- Scaffolded Track 1 (Precision Agriculture) and Track 2 (Environmental Systems)
- Identified Dr. Younsuk Dong (MSU BAE) as principal investigator
- Defined 8 experiments across both tracks
- Created airspring-barracuda Rust crate with module structure
- Created airspring-barracuda Rust crate (4 modules, 4 validation binaries, 15 unit tests)
- Dependencies: barracuda (phase1/toadstool), serde, rayon
- **All validation binaries PASS:**

| Binary | Track | Checks | Key validations |
|--------|-------|--------|----------------|
| validate_et0 | T1 | 22/22 | FAO-56 tables (10 es, 5 Δ), Uccle ET₀ 3.33 mm/day, Bangkok 3.53 mm/day |
| validate_soil | T1 | 25/25 | Topp equation (7 points), inverse round-trip, 5 USDA textures, PAW, triggers |
| validate_iot | T1 | 11/11 | 168 records, 5 columns, CSV round-trip, diurnal statistics |
| validate_water_balance | T1 | 12/12 | Mass balance 0.0000 (3 scenarios), Ks bounds, MI summer 485mm ET |

**Total: 70/70 checks PASS, 15/15 unit tests PASS**

---

## Experiment Log

### Experiment 001: FAO-56 Penman-Monteith Bootstrap — IN PROGRESS

**Goal**: Validate reference evapotranspiration (ET₀) calculation against
FAO Paper 56 published example values.

The Penman-Monteith equation is the foundational calculation for all irrigation
scheduling — analogous to Galaxy Bootstrap in wetSpring.

```
ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
```

- [x] Implement FAO-56 Penman-Monteith in Rust (`eco::evapotranspiration`)
- [x] Validate against FAO Paper 56 Example 18 (Bangkok, Thailand)
- [x] Validate against FAO Paper 56 Example 20 (Uccle, Belgium)
- [ ] Validate against MSU Enviro-weather station data
- [ ] Benchmark: Rust vs Python (numpy) throughput

### Experiment 002: Soil Sensor Calibration — IN PROGRESS

**Goal**: Reproduce dielectric permittivity → volumetric water content
calibration curves against Dong et al. (2020) Michigan soil data.

- [x] Implement Topp equation in Rust (`eco::soil_moisture`)
- [x] Implement soil-specific calibration curve fitting
- [x] Validate against published Topp equation values
- [ ] Validate against Michigan field calibration data
- [ ] Benchmark: Rust vs Python (scipy) curve fitting

### Experiment 003: IoT Data Pipeline — IN PROGRESS

**Goal**: Parse real IoT sensor time series data from agricultural field
monitoring systems.

- [x] Implement CSV time series parser (`io::csv_ts`)
- [ ] Parse MSU Enviro-weather station data
- [ ] Parse MQTT-format IoT sensor streams
- [ ] Validate: identical statistics to pandas on same data

### Experiment 004: Water Balance — IN PROGRESS

**Goal**: Implement field-scale water balance model for irrigation scheduling.

- [x] Implement water balance model (`eco::water_balance`)
- [x] Validate: mass conservation (P + I = ET + D + ΔS)
- [ ] Validate against published lysimeter data
- [ ] Implement threshold-based irrigation trigger

### Experiment 005: Lysimeter Validation — NOT STARTED

**Goal**: Reproduce ET measurements from weighing lysimeter load cell data
(Dong & Hansen, 2023).

### Experiment 006: HYDRUS Benchmark — NOT STARTED

**Goal**: Reproduce 1D Richards equation solution for unsaturated flow,
validate against HYDRUS CW2D published examples.

### Experiment 007: Biochar Isotherms — NOT STARTED

**Goal**: Fit Langmuir/Freundlich adsorption isotherms to published biochar
data (Kumari, Dong & Safferman, 2025).

### Experiment 008: Agrivoltaics PAR — NOT STARTED

**Goal**: Model photosynthetically active radiation interception under solar
panel arrays for dual-use agriculture.

---

## Evolution Roadmap

```
Track 1 (Precision Agriculture):
  Phase 0 [ACTIVE]: FAO-56 + sensor calibration validation (Rust baselines)
  Phase 1:          IoT pipeline + irrigation scheduling (real data)
  Phase 2:          GPU acceleration (real-time IoT, spatial kriging)
  Phase 3:          Penny irrigation (sovereign, consumer hardware)

Track 2 (Environmental Systems):
  Phase B0:         HYDRUS benchmarks + biochar fitting (Python baselines)
  Phase B1:         Contaminant transport + adsorption (scipy validation)
  Phase B2:         Rust ports (BarraCUDA: Richards solver, isotherm fitting)
  Phase B3:         GPU acceleration (field-scale PDE, Monte Carlo)
  Phase B4:         Sovereign remediation monitoring
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

*Initialized: February 16, 2026*
*BarraCUDA validation — 70/70 PASS (ET₀, soil, IoT, water balance): February 16, 2026*
