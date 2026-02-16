# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **hotSpring** validates nuclear physics (clean math, f64) and **wetSpring** validates *points in a system* (microbiome, mass spectra, PFAS), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCUDA) → GPU (ToadStool) → Penny Irrigation
```

## Current Status

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **142/142 PASS** | FAO-56, soil, IoT, water balance |
| Phase 0+: Real data pipeline | **918 station-days** | ET₀ R²=0.967 vs Open-Meteo |
| Phase 1: Rust validation | **70/70 PASS** | BarraCUDA: ET₀, soil, IoT, water balance |
| Phase 2: Real data → Rust | Planned | Cross-validate Python vs Rust |
| Phase 3: GPU acceleration | Planned | Spatial kriging, real-time IoT |
| Phase 4: Penny Irrigation | Vision | Sovereign, consumer hardware |

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r control/requirements.txt

# 2. Download REAL weather data (Open-Meteo — free, no key)
python scripts/download_open_meteo.py --all-stations --growing-season 2023

# 3. Run full validation suite (paper benchmarks + real data pipeline)
bash scripts/run_all_baselines.sh

# 4. Run Rust validation (requires barracuda dependency)
cd barracuda && cargo run --release --bin validate_et0
```

No institutional access required. Zero synthetic data in the default pipeline.

## Data Strategy

Paper data validates our methods. Open data is what we compute on.

| Layer | Source | Purpose | API Key |
|-------|--------|---------|---------|
| **Benchmark** | FAO-56 tables, Dong 2020/2024 | Ground truth (digitized) | None |
| **Open Data** | Open-Meteo archive | 80+ yr Michigan weather | **None (free)** |
| **Open Data** | OpenWeatherMap | Current + 5-day forecast | `testing-secrets/` |
| **Open Data** | NOAA CDO (GHCND) | Historical daily | `testing-secrets/` |
| **Open Data** | USDA Web Soil Survey | Soil properties | None |
| **Fallback** | Synthetic generation | Only if API unreachable | N/A |

## Research Context

### Track 1: Precision Irrigation & Soil Science

**Younsuk Dong, PhD** — Assistant Professor & Irrigation Specialist, Biosystems and Agricultural Engineering, Michigan State University.

**The Problem**: Agricultural irrigation consumes ~70% of global freshwater withdrawals. Inefficient scheduling wastes 30-50% of applied water. Current precision irrigation relies on proprietary sensor systems ($500-$5000/field) and vendor-locked software. Small-to-medium farms cannot afford the instrumentation or expertise.

**Computational Methods**:

1. **Evapotranspiration (FAO-56)** — Penman-Monteith reference ET₀. The foundational calculation for all irrigation scheduling.
2. **Soil moisture calibration** — Dielectric permittivity → VWC using Topp equation and soil-specific corrections.
3. **IoT field monitoring** — Real-time soil moisture, weather, PAR from distributed sensor networks.
4. **Water balance scheduling** — FAO-56 Chapter 8 daily root zone depletion tracking.
5. **Weighing lysimeter** — Direct ET measurement from load cell data (Dong & Hansen, 2023).

### Track 2: Environmental Systems & Land Treatment

1. **Richards equation** — Unsaturated flow in soils (open-source alternative to HYDRUS).
2. **Biochar adsorption** — Langmuir/Freundlich isotherms (Kumari, Dong & Safferman, 2025).
3. **Agrivoltaics** — PAR interception modeling under solar panel arrays.

## Key Publications

- Dong et al. (2020) "Soil moisture sensor evaluation in Michigan soils" *Agriculture* 10(12), 598
- Dong & Hansen (2023) "Affordable weighing lysimeter design" *Smart Ag Tech* 4, 100147
- Dong et al. (2024) "In-field IoT for precision irrigation" *Frontiers in Water* 6, 1353597
- Ali, Dong & Lavely (2024) "Irrigation scheduling vs yield" *Ag Water Mgmt* 306, 109148
- Dong et al. (2019) "Land-based wastewater modeling using HYDRUS CW2D" *J. SWBE* 5(4)
- Kumari, Dong & Safferman (2025) "Phosphorus adsorption using biochar" *Applied Water Sci* 15(7)

## Directory Structure

```
airSpring/
├── control/                     # Phase 0: Python/R baselines
│   ├── fao56/                   # FAO-56 Penman-Monteith ET₀
│   │   ├── penman_monteith.py   #   Paper benchmark validation (64/64)
│   │   ├── compute_et0_real_data.py  # ET₀ on real Michigan data
│   │   └── benchmark_fao56.json #   Digitized FAO-56 examples
│   ├── soil_sensors/            # Soil moisture calibration
│   │   ├── calibration_dong2020.py   # Topp eq + corrections (36/36)
│   │   └── benchmark_dong2020.json
│   ├── iot_irrigation/          # IoT irrigation pipeline
│   │   ├── calibration_dong2024.py   # SoilWatch 10 + scheduling (24/24)
│   │   ├── anova_irrigation.R        # R ANOVA (paper used R v4.3.1)
│   │   └── benchmark_dong2024.json
│   ├── water_balance/           # FAO-56 soil water balance
│   │   ├── fao56_water_balance.py    # Mass-conserving model (18/18)
│   │   ├── simulate_real_data.py     # 4 crops on real weather
│   │   └── benchmark_water_balance.json
│   └── requirements.txt
├── barracuda/                   # Phase 1: Rust validation (70/70)
│   ├── src/
│   │   ├── eco/                 # evapotranspiration, soil_moisture, water_balance
│   │   ├── io/                  # csv_ts (IoT time series parser)
│   │   └── bin/                 # validate_et0, validate_soil, validate_iot, validate_water_balance
│   └── Cargo.toml
├── scripts/                     # Data download + orchestration
│   ├── download_open_meteo.py   # Open-Meteo (free, no key, 80+ yr)
│   ├── download_enviroweather.py # OpenWeatherMap (current + forecast)
│   ├── download_noaa.py         # NOAA CDO (GHCND historical)
│   └── run_all_baselines.sh     # Full validation suite
├── data/                        # Downloaded real data (not committed)
│   ├── open_meteo/              # 918 station-days, 6 Michigan stations
│   ├── enviroweather/           # OpenWeatherMap current weather
│   ├── noaa/                    # NOAA CDO GHCND data
│   ├── et0_results/             # Computed ET₀ on real data
│   └── water_balance_results/   # Water balance simulations
├── whitePaper/                  # Methodology and study documentation
├── CONTROL_EXPERIMENT_STATUS.md # Detailed experiment log
└── LICENSE                      # AGPL-3.0-or-later
```

## Relationship to Other Springs

| | hotSpring | wetSpring | **airSpring** |
|--|-----------|-----------|---------------|
| Domain | Nuclear physics | Life/analytical science | Ecological systems |
| Scale | Nucleus (fm) | Cell/molecule | Field/watershed |
| Validation | Binding energies | Organism/PFAS ID | ET₀, yield, water balance |
| Baseline | Python/scipy | Galaxy/QIIME2/asari | **FAO-56/Python/R** |
| Data | AME2020 (IAEA) | GenBank, MassBank | **Open-Meteo, NOAA, FAO** |
| Evolution | Rust BarraCUDA | Rust BarraCUDA | **Rust BarraCUDA** |
| GPU layer | ToadStool (wgpu) | ToadStool | **ToadStool** |
| Success metric | chi² match | Same taxonomy/PFAS | **Same ET₀/yield** |
| Ultimate goal | Sovereign nuclear | Penny water monitoring | **Penny irrigation** |

### Cross-Spring GPU Kernels

| Kernel | hotSpring | wetSpring | airSpring |
|--------|-----------|-----------|-----------|
| ODE/PDE solver | HFB eigensolve | — | Richards equation |
| Time series filter | — | LC-MS chromatogram | IoT sensor streams |
| Nonlinear fitting | SEMF optimization | Peak fitting | Calibration curves |
| Spatial interpolation | — | — | Soil moisture kriging |
| Monte Carlo | Nuclear EOS | Rarefaction | Uncertainty quantification |
| Reduction | Binding energy sums | Peak areas | Temporal aggregation |

## Hardware Gate

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (16C/24T, 5.2 GHz) |
| RAM | 64 GB DDR5-4800 |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| Storage | 1 TB NVMe SSD |
| OS | Pop!_OS 22.04 (Ubuntu-based) |

## License

AGPL-3.0-or-later

---

*Initialized: February 16, 2026*
*Phase 0 Python baselines: 142/142 PASS*
*Real data pipeline: 918 station-days, ET₀ R²=0.97, 4 crop water balance*
*BarraCUDA Rust validation: 70/70 PASS*
*NOAA CDO: live with real token*
