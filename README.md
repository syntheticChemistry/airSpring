# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **hotSpring** validates nuclear physics (clean math, f64) and **wetSpring** validates *points in a system* (microbiome, mass spectra, PFAS), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCuda CPU)
     → GPU (ToadStool shaders) → metalForge (mixed hardware) → Penny Irrigation
```

## Current Status (v0.3.8, 2026-02-25)

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **142/142 PASS** | FAO-56, soil, IoT, water balance |
| Phase 0+: Real data pipeline | **918 station-days** | ET₀ R²=0.967 vs Open-Meteo |
| Phase 1: Rust validation | **123/123 PASS** | 8 binaries, 253 tests, 97% coverage |
| Phase 2: Cross-validation | **65/65 MATCH** | Python↔Rust identical (tol=1e-5) |
| Phase 3: GPU bridge | **GPU-FIRST** | 6 orchestrators, 4/4 ToadStool issues resolved |
| Phase 4: Penny Irrigation | Vision | Sovereign, consumer hardware |

### BarraCuda Integration

| airSpring Module | BarraCuda Primitive | Origin | Status |
|-----------------|--------------------|----|---|
| `gpu::et0` | `ops::batched_elementwise_f64` (op=0) | Multi-spring | **GPU-FIRST** |
| `gpu::water_balance` | `ops::batched_elementwise_f64` (op=1) | Multi-spring | **GPU-STEP** |
| `gpu::kriging` | `ops::kriging_f64::KrigingF64` | wetSpring | **Integrated** |
| `gpu::reduce` | `ops::fused_map_reduce_f64` | wetSpring | **GPU N≥1024** |
| `gpu::stream` | `ops::moving_window_stats` | wetSpring S28+ | **Wired** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN | **Wired** |
| `validation` | `validation::ValidationHarness` | neuralSpring | **Absorbed** |
| `testutil` | `stats::pearson`, `spearman`, `bootstrap_ci` | Shared | **Wired** |

Evolution gaps: 17 total (8 Tier A integrated, 8 Tier B, 1 Tier C).
Richards PDE promoted C→B — upstream `pde::richards::solve_richards` now available.
See `barracuda/src/gpu/evolution_gaps.rs` for the full roadmap.

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

# 5. Run benchmark (CPU baselines with cross-spring provenance)
cd barracuda && cargo run --release --bin bench_airspring_gpu
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
├── control/                     # Phase 0: Python/R baselines (142/142)
│   ├── fao56/                   # FAO-56 Penman-Monteith ET₀ (64/64)
│   ├── soil_sensors/            # Soil moisture calibration (36/36)
│   ├── iot_irrigation/          # IoT irrigation pipeline (24/24)
│   ├── water_balance/           # FAO-56 soil water balance (18/18)
│   └── requirements.txt
├── barracuda/                   # Phase 1: Rust validation (123/123, 253 tests)
│   ├── src/
│   │   ├── eco/                 # correction, crop, evapotranspiration, sensor_calibration,
│   │   │                        #   soil_moisture, water_balance
│   │   ├── io/                  # csv_ts (streaming columnar IoT parser)
│   │   ├── gpu/                 # ToadStool/BarraCuda GPU bridge (GPU-FIRST)
│   │   │   ├── et0.rs           #   BatchedEt0 GPU-first (BatchedElementwiseF64)
│   │   │   ├── water_balance.rs #   BatchedWaterBalance GPU-step + CPU season
│   │   │   ├── kriging.rs       #   KrigingInterpolator (barracuda::ops::kriging_f64)
│   │   │   ├── reduce.rs        #   SeasonalReducer (barracuda::ops::fused_map_reduce_f64)
│   │   │   ├── stream.rs        #   StreamSmoother (barracuda::ops::moving_window_stats)
│   │   │   └── evolution_gaps.rs#   15 gaps (8A+5B+2C), 4/4 ToadStool issues RESOLVED
│   │   ├── error.rs             # AirSpringError enum (proper error types)
│   │   ├── validation.rs        # Re-exports barracuda::validation::ValidationHarness
│   │   ├── testutil.rs          # IA, NSE, RMSE, MBE, R², Spearman, bootstrap CI
│   │   └── bin/                 # validate_*, cross_validate, simulate_season,
│   │                            #   bench_airspring_gpu
│   ├── tests/                   # 76 integration tests across 4 files
│   └── Cargo.toml
├── scripts/                     # Data download + orchestration
├── data/                        # Downloaded real data (not committed)
├── metalForge/                  # Upstream absorption staging (→ barracuda)
│   └── forge/                   # airspring-forge v0.2.0 (40 tests, 4 modules)
├── specs/                       # Specifications and requirements
│   ├── PAPER_REVIEW_QUEUE.md    #   Paper reproduction queue
│   ├── BARRACUDA_REQUIREMENTS.md#   GPU kernel requirements
│   └── CROSS_SPRING_EVOLUTION.md#   Cross-spring shader provenance (608 shaders)
├── whitePaper/                  # Methodology and study documentation
│   ├── baseCamp/                #   Per-faculty research briefings
│   ├── METHODOLOGY.md           #   Multi-phase validation protocol
│   └── STUDY.md                 #   Full results narrative
├── experiments/                 # Experiment protocols and results
├── wateringHole/                # Spring-local handoffs to ToadStool/BarraCuda
│   └── handoffs/                #   Versioned handoff documents
├── CHANGELOG.md                 # Keep-a-Changelog versioned history
├── CONTROL_EXPERIMENT_STATUS.md # Detailed experiment log
└── LICENSE                      # AGPL-3.0-or-later
```

## Evolution Architecture

```
Write → Absorb → Lean

  airSpring eco:: (CPU, validated against FAO-56 papers)
       │
       ▼
  airSpring gpu:: wrappers (domain-specific batched API)
       │
       ▼
  barracuda::ops/linalg/stats primitives (GPU dispatch + CPU fallback)
       │
       ▼
  ToadStool WGSL shaders (f64 precision on GPU)
       │
       ▼
  metalForge (mixed CPU + GPU + future NPU)
```

**Cross-spring evolution**: ToadStool contains 608 WGSL shaders across 41
categories. airSpring uses 5 shared shaders and contributed 3 upstream fixes
(TS-001/003/004). 46 cross-spring absorptions (S51-S57) unite hotSpring
precision physics, wetSpring bio/environmental, and neuralSpring ML shaders
into a shared compute foundation. See `specs/CROSS_SPRING_EVOLUTION.md`.

## Relationship to Other Springs

| | hotSpring | wetSpring | **airSpring** |
|--|-----------|-----------|---------------|
| Domain | Nuclear physics | Life/analytical science | Ecological systems |
| Scale | Nucleus (fm) | Cell/molecule | Field/watershed |
| Validation | Binding energies | Organism/PFAS ID | ET₀, yield, water balance |
| Baseline | Python/scipy | Galaxy/QIIME2/asari | **FAO-56/Python/R** |
| Data | AME2020 (IAEA) | GenBank, MassBank | **Open-Meteo, NOAA, FAO** |
| Evolution | Rust BarraCuda | Rust BarraCuda | **Rust BarraCuda** |
| GPU layer | ToadStool (wgpu) | ToadStool | **ToadStool** |
| Success metric | chi² match | Same taxonomy/PFAS | **Same ET₀/yield** |
| Ultimate goal | Sovereign nuclear | Penny water monitoring | **Penny irrigation** |

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

*February 25, 2026 — v0.3.8. 123 validation checks (8 binaries), 293 Rust tests
(253 barracuda + 40 forge), 918 real station-days, 65/65 Python-Rust cross-validation
match. 97.2% library test coverage. GPU-FIRST with 6 orchestrators (BatchedEt0,
BatchedWaterBalance, KrigingInterpolator, SeasonalReducer, StreamSmoother, fit_ridge).
metalForge v0.2.0: 4 absorption-ready modules (metrics, regression, moving_window_f64,
hydrology) following hotSpring's Write → Absorb → Lean pattern. Cross-spring shader
evolution: 608 WGSL shaders, 46 absorptions, 3 airSpring fixes contributed upstream.
17 evolution gaps (8A+8B+1C) — Richards PDE promoted C→B (upstream solver available).
Pure Rust + BarraCuda GPU pipeline. AGPL-3.0.*
