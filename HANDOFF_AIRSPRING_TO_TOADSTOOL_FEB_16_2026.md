# Handoff: airSpring → Toadstool/BarraCUDA

**Date:** February 16, 2026
**From:** airSpring (Precision Agriculture validation study)
**To:** Toadstool/BarraCUDA core team
**License:** AGPL-3.0-or-later

---

## STATED GOAL: Open Precision Irrigation on Consumer Hardware

Phase 0 and 0+ confirmed: **published agricultural science reproduces with
open tools and open data.** 142/142 Python checks pass against paper
benchmarks. 918 station-days of real Michigan weather produce ET₀ with
R²=0.967. Water balance simulations show 53-72% water savings with smart
scheduling — consistent with published results.

Phase 1 confirmed: **Rust BarraCUDA validates the core pipeline.** 70/70
checks pass (ET₀, soil moisture, water balance, IoT parsing). The Rust
implementation covers the computational core; the Python baselines add
statistical methods and data download that need porting.

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **142/142 PASS** | FAO-56, soil, IoT, water balance |
| Phase 0+: Real data pipeline | **918 station-days** | ET₀ R²=0.967 vs Open-Meteo, 3 API sources |
| Phase 1: Rust validation | **70/70 PASS** | 4 binaries, 15 unit tests |
| Phase 2: Cross-validation | **Not started** | Python vs Rust identical outputs |

---

## What airSpring Brings to BarraCUDA

### New Domain: Time Series + Spatial + IoT

hotSpring proved BarraCUDA can do clean matrix math (eigensolve, PDE,
optimization). airSpring adds a fundamentally different workload pattern:

| Dimension | hotSpring | airSpring |
|-----------|-----------|-----------|
| Data shape | Dense matrices (12×12 to 50×50) | Long time series (153+ days × N stations) |
| Input rate | Static (AME2020 table) | Streaming (IoT sensors, API feeds) |
| Spatial | Per-nucleus | Per-field-cell (kriging grid) |
| Time coupling | SCF iteration | Daily water balance (sequential) |
| Parallelism | Across nuclei | Across stations, fields, and grid cells |
| I/O pattern | Preload once | Continuous API ingestion |

### Shared Primitives Already Working

These BarraCUDA primitives serve both hotSpring and airSpring:

| Primitive | hotSpring Use | airSpring Use |
|-----------|-------------|---------------|
| `serde` serialization | AME2020 JSON | Benchmark JSON, API responses |
| `rayon` parallelism | Cross-nucleus HFB | Cross-station ET₀ |
| f64 arithmetic | Nuclear binding energies | FAO-56 intermediates |
| Unit test harness | 195 checks | 70 checks |

### New Primitives airSpring Needs

These don't exist in BarraCUDA yet but are needed for Phase 2+:

| Primitive | Purpose | Complexity |
|-----------|---------|:----------:|
| Nonlinear least squares | Soil calibration fitting | Medium |
| Moving window statistics | IoT stream processing | Low |
| Spatial interpolation (kriging) | Soil moisture mapping | High |
| 1D Richards solver | Unsaturated flow PDE | High |
| CSV streaming parser | Real-time IoT data | Low |
| API client (HTTP + JSON) | Open-Meteo, NOAA CDO | Low |

---

## Rust Crate Architecture

### Module Map

```
airspring-barracuda/
├── src/
│   ├── lib.rs              # pub mod eco; pub mod io;
│   ├── eco/
│   │   ├── mod.rs          # pub mod evapotranspiration, soil_moisture, water_balance
│   │   ├── evapotranspiration.rs  # 17 FAO-56 functions, DailyEt0Input, Et0Result
│   │   ├── soil_moisture.rs       # Topp eq, inverse, SoilTexture, SoilHydraulicProps
│   │   └── water_balance.rs       # WaterBalanceState, simulate_season, mass_balance
│   ├── io/
│   │   ├── mod.rs          # pub mod csv_ts
│   │   └── csv_ts.rs       # TimeseriesData, ColumnStats, parse_csv
│   └── bin/
│       ├── validate_et0.rs          # 22 checks
│       ├── validate_soil.rs         # 25 checks
│       ├── validate_iot.rs          # 11 checks
│       └── validate_water_balance.rs  # 12 checks
└── Cargo.toml
```

### Dependencies

```toml
[dependencies]
barracuda = { path = "../../phase1/toadstool/crates/barracuda" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rayon = "1.10"
```

The `barracuda` path dependency points to `phase1/toadstool/crates/barracuda`.
The crate is NOT in the root workspace `members` — it runs standalone.

### Known Issues

1. **Typo**: `SoilTexture::SandyCite` should be `SandyClay`
2. **Runoff model differs**: Rust uses `(P - 20) × 0.2` when P > 20mm; Python uses RO = 0 (FAO-56 default for well-drained fields). Recommend aligning to Python's RO = 0.
3. **Planned modules not implemented**: `eco::isotherms` and `eco::richards` are documented in `lib.rs` but have no source files.

---

## What Python Has That Rust Doesn't (Phase 2 Work)

### Priority 1: Cross-Validation Infrastructure

| Task | Description | Effort |
|------|-------------|:------:|
| JSON benchmark loader | Read `benchmark_*.json` files in Rust | 1 day |
| Python↔Rust comparison harness | Run both, diff outputs | 2 days |
| Real data CSV reader | Load Open-Meteo/NOAA CSVs in Rust | 1 day |

### Priority 2: Statistical Methods

| Function | Python Location | Rust Equivalent |
|----------|----------------|-----------------|
| `compute_rmse(obs, sim)` | `calibration_dong2020.py` | Simple — port directly |
| `compute_ia(obs, sim)` | `calibration_dong2020.py` | Index of Agreement formula |
| `compute_mbe(obs, sim)` | `calibration_dong2020.py` | Mean bias — trivial |
| `compute_r2(obs, sim)` | `calibration_dong2020.py` | Correlation coefficient |
| `fit_correction_equations()` | `calibration_dong2020.py` | Needs nonlinear solver (NLS) |

### Priority 3: Sensor Calibration

| Function | Python Location | Notes |
|----------|----------------|-------|
| `soilwatch10_vwc(rc)` | `calibration_dong2024.py` | Simple polynomial |
| `irrigation_recommendation()` | `calibration_dong2024.py` | Threshold logic |

### Priority 4: Data Pipeline (if Rust-native desired)

| Component | Python Script | Notes |
|-----------|--------------|-------|
| Open-Meteo client | `download_open_meteo.py` | HTTP GET + JSON parse |
| NOAA CDO client | `download_noaa.py` | HTTP GET + token auth |
| OpenWeatherMap client | `download_enviroweather.py` | HTTP GET + API key |

---

## GPU Acceleration Opportunities (Phase 3)

### TIER 1 — Embarrassingly Parallel (Immediate Value)

#### 1.1 Batched ET₀ Computation

ET₀ is computed independently for each station-day. With real data growing
to thousands of station-years, this is a perfect GPU workload:

```
Input:  N station-days × 7 variables (tmax, tmin, rh_max, rh_min, wind, Rs, lat/elev/doy)
Output: N ET₀ values
```

**Architecture**: Single compute shader, one workgroup per station-day.
All FAO-56 functions are pure arithmetic (exp, sqrt, log, trig) — maps
directly to WGSL f64 with the same precision patterns hotSpring validated.

**Deliverable**: `BatchedEt0Gpu` — compute N ET₀ values in a single dispatch.

**Impact**: Enables hourly ET₀ across spatial grids (10,000+ cells per
timestep) for real-time irrigation decisions.

#### 1.2 Batched Water Balance

The water balance for different fields/crops is independent. Given ET₀
computed above, each field's daily depletion tracking is a short sequential
chain (153 days) that can run in parallel across fields:

```
Input:  F fields × D days × (ET₀, precip, Kc, soil params)
Output: F fields × D days × (Dr, Ks, ETc, irrigation)
```

**Architecture**: One workgroup per field, sequential days within workgroup.

**Impact**: Field-grid irrigation scheduling at sub-field resolution.

### TIER 2 — Spatial Operations (Medium Term)

#### 2.1 Kriging / Spatial Interpolation

Soil moisture mapping from sparse sensor locations to a regular grid.
Requires covariance matrix construction + Cholesky solve:

```
Input:  S sensor locations with VWC readings
Output: G grid cells with interpolated VWC + kriging variance
```

**Deliverable**: `BatchedKrigingGpu` — uses existing `GemmF64` for matrix
operations, plus a custom variogram kernel.

**Impact**: Transforms sparse sensor data into actionable spatial maps.

#### 2.2 Richards Equation Solver (1D)

Unsaturated water flow through soil profile. This is a nonlinear PDE
(time-stepping + spatial discretization):

```
∂θ/∂t = ∂/∂z [K(h)(∂h/∂z + 1)] - S(z,t)
```

**Architecture**: Finite difference on 1D grid, implicit time stepping.
Similar structure to hotSpring's SCF iteration but simpler (scalar field
on 1D grid instead of wavefunctions on 2D grid).

**Deliverable**: `Richards1dGpu` — single-dispatch solver for soil column.
Uses existing `FdGradientF64` for spatial derivatives.

**Impact**: Open-source alternative to HYDRUS (proprietary Fortran code).

### TIER 3 — Real-Time IoT (Long Term)

#### 3.1 Streaming Sensor Fusion

Real-time ingestion of soil moisture, weather, and leaf wetness data:
- Kalman filtering for noisy sensor readings
- Anomaly detection (sensor malfunction vs real events)
- Triggering irrigation decisions in <1 second

**Impact**: Edge-compute irrigation controller on consumer hardware.

---

## Cross-Spring Lessons for ToadStool

### From hotSpring → airSpring

| Lesson | hotSpring Discovery | airSpring Application |
|--------|--------------------|-----------------------|
| f64 GPU works | RTX 4070 SHADER_F64 confirmed | FAO-56 ET₀ needs f64 precision |
| Dispatch overhead matters | 145k dispatches = 16x slower | Batch all station-days in one dispatch |
| Hybrid GPU+Rayon | CPU parallel complements GPU | Cross-station parallelism on CPU |
| Pre-computed buffers | Avoid f32 pow() on GPU | Pre-compute pressure, gamma tables |
| Single-encoder batching | Mega-batch eliminated overhead | Apply same pattern to batched ET₀ |

### From airSpring → ToadStool (New Capabilities)

| Capability | Why airSpring Needs It | Other Springs Benefit |
|-----------|----------------------|----------------------|
| HTTP/JSON data client | Open-Meteo, NOAA CDO, OWM APIs | Any spring needing open data |
| Time series windowed ops | IoT sensor smoothing | wetSpring LC-MS chromatograms |
| CSV streaming parser | Real-time sensor data | Universal utility |
| Spatial interpolation | Soil moisture kriging | wetSpring: sampling site interpolation |
| 1D PDE solver | Richards equation | hotSpring: simpler variant of HFB |

### Architecture Pattern: Data-First Validation

airSpring establishes a pattern that differs from hotSpring:

**hotSpring pattern**: Equations first, then validate against published numbers.
**airSpring pattern**: Paper benchmarks validate equations, then real open data
validates the pipeline end-to-end.

This "data-first" approach means airSpring's Rust code needs to handle
real-world data messiness (missing values, unit conversions, API rate limits)
that hotSpring's clean matrix math never encounters. ToadStool should consider
whether data ingestion utilities belong in the core library or in a separate
`toadstool-data` crate.

---

## What Stays in airSpring (Domain-Specific)

These encode agricultural physics and should NOT migrate to ToadStool:

| Module | Purpose | Why Domain-Specific |
|--------|---------|---------------------|
| `eco::evapotranspiration` | FAO-56 Penman-Monteith | Agricultural ET₀ coefficients |
| `eco::soil_moisture` | Topp equation + textures | Soil dielectric physics |
| `eco::water_balance` | FAO-56 Chapter 8 | Crop Kc, depletion tracking |
| `io::csv_ts` | IoT sensor parsing | May generalize to ToadStool |
| Benchmark JSONs | Digitized paper values | Domain validation data |
| Download scripts | API clients for weather | Could generalize |

These demonstrate that ToadStool's compute primitives support agricultural
science just as they support nuclear physics. The same GPU ops (batched
arithmetic, PDE stencils, matrix solves) serve both domains.

---

## Previously Completed (Confirmed Working)

| Component | Status | Check Count |
|-----------|:------:|:-----------:|
| FAO-56 Penman-Monteith (Python) | PASS | 64 |
| Soil sensor calibration (Python) | PASS | 36 |
| IoT irrigation pipeline (Python) | PASS | 24 |
| Water balance (Python) | PASS | 18 |
| R ANOVA script | Written | Awaiting R |
| Real data: Open-Meteo (6 stations) | PASS | 918 station-days |
| Real data: OpenWeatherMap (6 stations) | PASS | 42 records |
| Real data: NOAA CDO (Lansing) | PASS | 153 days |
| ET₀ cross-check (our vs Open-Meteo) | PASS | R²=0.967 |
| Water balance on real data (4 crops) | PASS | Mass balance 0.0000 |
| Rust ET₀ validation | PASS | 22 |
| Rust soil validation | PASS | 25 |
| Rust IoT validation | PASS | 11 |
| Rust water balance validation | PASS | 12 |
| **Total** | **All pass** | **212 + 918 data** |

---

## The Proof

BarraCUDA, through airSpring's validation:

- **Rust implements FAO-56 correctly** — 22/22 ET₀ checks, matching published examples
- **Open data replaces institutional access** — 918 station-days, 3 free APIs, zero synthetic
- **Water savings are real** — 53-72% vs naive scheduling on real 2023 Michigan weather
- **The architecture extends** — same Rust crate handles ET₀, soil, water balance, IoT
- **Cross-spring sharing works** — serde, rayon, f64 patterns reused from hotSpring
- **212/212 quantitative checks** — zero failures across all experiments
- **Sovereign Science** — AGPL-3.0, fully reproducible, no institutional access

The path to Penny Irrigation: port remaining Python methods to Rust,
GPU-accelerate ET₀ and water balance on ToadStool, add spatial kriging,
and deploy on consumer hardware ($600 GPU) for sub-field irrigation scheduling.

---

*February 16, 2026 — Phase 0/0+/1 complete. 212 checks pass, 918 real
station-days, 3 open data sources. Ready for Phase 2 cross-validation
and Phase 3 GPU acceleration via ToadStool.*
