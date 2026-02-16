# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **wetSpring** validates *points in a system* (microbiome samples, water analytes, PFAS data), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

The evolution path:

```
Python/MATLAB baselines → Rust validation (BarraCUDA) → GPU acceleration (ToadStool) → sovereign pipeline
```

## Research Context

### Track 1: Precision Irrigation & Soil Science

**Younsuk Dong, PhD** — Assistant Professor & Irrigation Specialist, Biosystems
and Agricultural Engineering, Michigan State University. PhD Biosystems
Engineering, MSU 2018. Research: precision irrigation IoT, soil moisture sensing,
evapotranspiration modeling, agricultural water management. Published in
Computers and Electronics in Agriculture, Frontiers in Water, Agricultural Water
Management, Smart Agricultural Technology.

#### The Problem

Agricultural irrigation consumes ~70% of global freshwater withdrawals.
Inefficient scheduling wastes 30-50% of applied water while simultaneously
degrading soil health through over- or under-watering. Current precision
irrigation relies on proprietary sensor systems ($500-$5000/field), commercial
scheduling software (vendor-locked to specific sensor brands), and institutional
weather station networks with limited spatial resolution. Small-to-medium farms
cannot afford the instrumentation or expertise.

#### Computational Methods

1. **Evapotranspiration modeling (FAO-56)** — Reference ET₀ and crop ET using
   the Penman-Monteith equation. The foundational calculation for all
   irrigation scheduling. Current: FAO paper tables + Excel/Python scripts.

2. **Soil moisture sensor calibration** — Dielectric permittivity → volumetric
   water content using Topp equation and soil-specific calibration curves.
   Current: manufacturer curves + field calibration (Python/R).

3. **IoT field monitoring** — Real-time soil moisture, PAR (photosynthetically
   active radiation), leaf wetness, and weather data from distributed sensor
   networks. Current: Arduino/ESP32 + MQTT + Python/Node-RED dashboards.

4. **Irrigation scheduling algorithms** — Water balance methods (checkbook),
   soil moisture threshold-based, and model-predictive approaches.
   Current: spreadsheet models + commercial software.

5. **Weighing lysimeter design** — Direct measurement of evapotranspiration
   from load cell data. Current: microcontroller-based (Dong & Hansen, 2023,
   Smart Agricultural Technology).

### Track 2: Environmental Systems & Land Treatment

#### The Problem

Wastewater land application, contaminant transport through soils, and PFAS
remediation via biochar require numerical modeling that currently depends on
proprietary software (HYDRUS, MODFLOW) or fragile Python/Fortran wrappers.
Agrivoltaics (dual-use solar + agriculture) is an emerging field requiring
microclimate modeling that current tools cannot perform at field scale in
real time.

#### Computational Methods

1. **Contaminant transport modeling** — Richards equation for unsaturated flow,
   advection-dispersion for solute transport. Current: HYDRUS CW2D
   (Fortran + Python wrapper). Dong et al., 2019, J. Environ. Sci. Health.

2. **Biochar adsorption modeling** — Heavy metal and phosphorus adsorption
   isotherms (Langmuir, Freundlich, BET). Current: scipy curve_fit +
   custom Python scripts. Kumari, Dong & Safferman, 2025, Applied Water Sci.

3. **Agrivoltaics microclimate** — Coupling solar panel shading geometry with
   crop growth models (PAR interception, soil temperature, ET modification).
   Current: empirical + MATLAB/Python models.

4. **Agricultural LPWAN/backscatter** — Signal propagation modeling for
   underground (cross-soil) and aerial agricultural IoT networks.
   Current: MATLAB + ray tracing. Ren, Dong, Cao et al., MobiCom 2024.

5. **Meta-analysis** — Systematic review and meta-regression of irrigation
   scheduling effects on crop yield. Ali, Dong & Lavely, 2024,
   Agricultural Water Management.

## Relationship to Other Springs

| | hotSpring | wetSpring T1 | wetSpring T2 | **airSpring T1** | **airSpring T2** |
|--|-----------|-------------|--------------|-----------------|-----------------|
| Domain | Nuclear physics | Life science | Analytical chem | Precision agriculture | Environmental systems |
| Scale | Nucleus (fm) | Cell/organism | Molecule | Field/watershed | Soil profile |
| Validation | Binding energies | Organism ID | PFAS ID | ET₀ / yield | Contaminant flux |
| Baseline | Python scipy | Galaxy/QIIME2 | asari/PFΔScreen | FAO-56/Python | HYDRUS/scipy |
| Evolution | Rust BarraCUDA | Rust BarraCUDA | Rust BarraCUDA | Rust BarraCUDA | Rust BarraCUDA |
| GPU layer | ToadStool (wgpu) | ToadStool | ToadStool | ToadStool | ToadStool |
| Success metric | chi² match | Same taxonomy | Same PFAS detected | Same ET₀/yield | Same transport |
| Ultimate goal | Sovereign nuclear | Sovereign metagenomics | Penny water monitoring | **Penny irrigation** | **Sovereign remediation** |

### The Three Springs

- **hotSpring** proved BarraCUDA/ToadStool can replicate institutional physics
  (clean math, f64, nuclear binding energies)
- **wetSpring** expands to messy biological data (sequences, mass spectra,
  chemical fingerprints) — *points in a system*
- **airSpring** expands to field-scale ecological systems (time series, spatial
  data, IoT streams, coupled PDE solvers) — *the system itself*

### Cross-Spring Kernels

| Kernel | hotSpring | wetSpring | airSpring |
|--------|-----------|-----------|-----------|
| ODE/PDE solvers | HFB solver | — | Richards eq, advection-dispersion |
| Time series processing | — | LC-MS chromatograms | IoT sensor streams |
| Statistical fitting | SEMF optimization | Peak fitting (Gaussian) | Calibration curves, meta-regression |
| Spatial interpolation | — | — | Soil moisture kriging, PAR mapping |
| Signal processing | — | Peak detection (asari) | Sensor filtering, leaf wetness |
| Monte Carlo | Nuclear matter EOS | Rarefaction (ecology) | Uncertainty quantification |
| Distance matrices | — | Bray-Curtis (diversity) | Spatial autocorrelation |

## Datasets

### Track 1: Precision Agriculture — Public Sources

| ID | Dataset | Source | Size | Notes |
|----|---------|--------|------|-------|
| D1 | FAO-56 reference tables | FAO Paper 56 | ~1 MB | ET₀ validation (Penman-Monteith) |
| D2 | Michigan AgWeather stations | MSU Enviro-weather | ~50 MB | Hourly weather (temp, humidity, wind, solar) |
| D3 | Soil moisture sensor benchmarks | Dong et al. 2020 | ~5 MB | Coarse/fine-textured Michigan soils |
| D4 | USDA Web Soil Survey | NRCS/USDA | ~10 MB | Soil hydraulic properties (Ksat, FC, WP) |
| D5 | Lysimeter ET measurements | Dong & Hansen 2023 | ~2 MB | Direct ET from weighing lysimeter |
| D6 | Irrigation meta-analysis | Ali, Dong & Lavely 2024 | ~1 MB | Global crop yield vs irrigation schedule |

### Track 2: Environmental Systems — Public Sources

| ID | Dataset | Source | Size | Notes |
|----|---------|--------|------|-------|
| D7 | HYDRUS CW2D examples | PC-Progress | ~5 MB | Constructed wetland test cases |
| D8 | Biochar adsorption isotherms | Kumari et al. 2025 | ~1 MB | Heavy metals + phosphorus |
| D9 | NOAA Climate Data Online | NOAA CDO | ~20 MB | Historical weather for Michigan |
| D10 | Agrivoltaic PAR measurements | MSU Solar Farm | ~5 MB | Sub-panel PAR and temperature |
| D11 | EPA PFAS soil remediation | EPA ORD | ~2 MB | Soil PFAS concentrations |

## Experiments

### Track 1: Precision Agriculture

| Exp | Name | Goal | Status |
|-----|------|------|--------|
| 001 | FAO-56 Bootstrap | Validate Penman-Monteith ET₀ against FAO paper tables | Planned |
| 002 | Soil Sensor Calibration | Reproduce Topp equation + Michigan soil calibration | Planned |
| 003 | IoT Data Pipeline | Parse real IoT sensor time series (soil moisture, PAR, weather) | Planned |
| 004 | Irrigation Scheduling | Implement water balance + threshold scheduling | Planned |
| 005 | Lysimeter Validation | Reproduce ET from lysimeter load cell data | Planned |

### Track 2: Environmental Systems

| Exp | Name | Goal | Status |
|-----|------|------|--------|
| 006 | HYDRUS Benchmark | Reproduce 1D Richards equation solution against HYDRUS | Planned |
| 007 | Biochar Isotherms | Fit Langmuir/Freundlich curves, validate against published data | Planned |
| 008 | Agrivoltaics PAR | Model PAR interception under solar panels | Planned |

## Evolution Roadmap

```
Track 1 (Precision Agriculture):
  Phase 0:      FAO-56 + sensor calibration validation (Python baselines)
  Phase 1:      IoT pipeline + irrigation scheduling (Python/R)
  Phase 2:      Rust ports (BarraCUDA: ET₀, calibration, water balance)
  Phase 3:      GPU acceleration (real-time IoT, spatial interpolation)
  Phase 4:      Penny irrigation (sovereign, low-cost sensor + GPU)

Track 2 (Environmental Systems):
  Phase B0:     HYDRUS benchmarks + biochar fitting (Python baselines)
  Phase B1:     Contaminant transport + adsorption (Python/scipy)
  Phase B2:     Rust ports (BarraCUDA: Richards solver, isotherm fitting)
  Phase B3:     GPU acceleration (field-scale PDE solver, Monte Carlo)
  Phase B4:     Sovereign remediation monitoring
```

## Hardware Gate

Same gate as wetSpring/hotSpring — all ecoPrimals springs validate on:

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
