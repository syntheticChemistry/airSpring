# baseCamp: Per-Faculty Research Briefings

**Updated**: February 26, 2026
**Project**: airSpring — Ecological & Agricultural Sciences (v0.4.8)
**Status**: 22 experiments, 594/594 Python + 491 Rust tests + 570 validation + 1393 atlas checks + 75/75 cross-validation + 11 Tier A modules + 69x CPU speedup

---

## Evolution Path

```
Phase 0   Python/R baselines    — reproduce paper results with original tools (594/594)
Phase 0+  Real open data        — compute on Open-Meteo, NOAA, USDA (no institutional access)
Phase 1   Rust BarraCuda CPU    — cross-validated to 1e-5 vs Python (491 tests, 570 validation + 1393 atlas, 27 binaries, 97.45% coverage)
Phase 2   BarraCuda GPU         — 11 Tier A modules wired (cross-spring S68 fully rewired)
Phase 3   metalForge            — Write→Absorb→Lean complete, 6/6 modules absorbed upstream
Phase 4   Penny Irrigation      — sovereign scheduling on consumer hardware ($600 GPU)
```

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domain |
|---------|------------|-------|:------:|:-----------:|:------:|--------|
| Dong | MSU BAE | Irrigation & Soil | 10+ | 22 | 594+570 | ET₀, soil moisture, IoT, water balance, dual Kc, cover crops, Richards, biochar, yield, CW2D, Thornthwaite, GDD, pedotransfer |

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
| 5 | Real data pipeline — 100 Michigan stations, 15,300 days | 0+ | R²=0.967 | Open-Meteo ERA5 |
| 6 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc (Kcb+Ke) | 0→CPU | 63/63 | FAO-56 Tables 17, 19 |
| 7 | Regional ET₀ intercomparison — 6 Michigan stations | 0→CPU | 61/61 | Open-Meteo ERA5 |
| 8 | Islam et al. (2014) No-till + FAO-56 Ch 11 cover crops | 0→CPU | 40/40 | ISWCR + FAO-56 |
| 9 | van Genuchten (1980) Richards equation | 0→GPU | 14+15 | Carsel & Parrish (1988) |
| 10 | Kumari et al. (2025) Biochar P adsorption | 0→GPU | 14+14 | Representative literature |
| 11 | 60-year water balance (OSU Triplett-Van Doren) | 0→CPU | 10+11 | Open-Meteo ERA5 60yr |
| 12 | Stewart (1977) yield response to water stress | 0→CPU | 32+32 | FAO-56 Table 24 |
| 13 | Dong et al. (2019) CW2D Richards extension | 0→CPU | 24+24 | HYDRUS CW2D params |
| 14 | Ali, Dong & Lavely (2024) Irrigation scheduling optimization | 0→CPU | 25+28 | Ag Water Mgmt 306 |
| 16 | Dong & Hansen (2023) Weighing lysimeter ET measurement | 0→CPU | 26+25 | Smart Ag Tech 4 |
| 17 | Gong et al. (2006) ET₀ sensitivity analysis (OAT) | 0→CPU | 23+23 | Ag Water Mgmt 86 |
| 18 | Priestley & Taylor (1972) radiation-based ET₀ | 0→CPU | 32+32 | FAO-56 intermediates |
| 19 | ET₀ 3-method intercomparison (PM/PT/HG) | 0→CPU | 36+36 | Open-Meteo ERA5 |
| 20 | Thornthwaite (1948) monthly ET₀ | 0→CPU | 23+50 | Temperature-based heat index |
| 21 | Growing Degree Days (GDD) | 0→CPU | 33+26 | Phenology, kc_from_gdd |
| 22 | Saxton & Rawls (2006) pedotransfer | 0→CPU | 70+58 | θs/θr/Ks from texture |

### Rust Validation (Phase 1) — 27 binaries

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
| `validate_yield` | 32 | Stewart 1977, Ky table, multi-stage, WUE, scheduling |
| `validate_cw2d` | 24 | CW2D media, VG retention, mass balance |
| `validate_atlas` | 1393/1393 | 100 Michigan stations × 13 checks each |
| `validate_thornthwaite` | 50 | Thornthwaite monthly ET₀ |
| `validate_gdd` | 26 | GDD accumulation, kc_from_gdd |
| `validate_pedotransfer` | 58 | Saxton-Rawls 2006 |
| `cross_validate` | 75 | Python↔Rust exact match (tol=1e-5) |

### GPU Orchestrators (Phase 2) — 11 Tier A wired

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

### CPU Benchmarks (v0.4.8) — Rust 69x Faster Than Python

| Operation | Rust Throughput | Speedup vs Python | Cross-Spring Provenance |
|-----------|----------------|:-----------------:|------------------------|
| ET₀ (FAO-56) | 12.7M/s | **20x** | hotSpring df64, multi-spring elementwise |
| VG θ(h) retention | 35.8M/s | **83x** | hotSpring df64 precision |
| Yield single-stage | 1.08B/s | **81x** | airSpring `eco::yield_response` |
| Dual Kc season | 59M/s | — | airSpring `eco::dual_kc` |
| Richards PDE (50 nodes) | 3,620/s | **502x** | airSpring→ToadStool, hotSpring df64 |
| Isotherm (NM 1-start) | 175K fits/s | neuralSpring `nelder_mead` |
| Isotherm (NM 8×LHS) | 42.5K fits/s | neuralSpring `multi_start_nelder_mead` |

### Evolution Documents

| Document | Purpose |
|----------|---------|
| `barracuda/EVOLUTION_READINESS.md` | Tier A/B/C breakdown, absorbed vs stays-local, quality gates |
| `metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66) |
| `wateringHole/handoffs/` | V022 active handoff — Thornthwaite, GDD, pedotransfer, S68 sync |
| `specs/CROSS_SPRING_EVOLUTION.md` | 774 WGSL shader provenance across all Springs |

### Next Steps (Dong Lab)

- **Paper 12+**: Multi-sensor calibration network (awaiting field data from new lab)
- **Paper 13+**: Full IoT irrigation with forecast integration (awaiting field data)
- **GPU validation**: Move Richards and isotherm to pure GPU via ToadStool shaders
- **metalForge absorption**: COMPLETE — 6/6 modules absorbed upstream (S64+S66)
- **metalForge mixed hardware**: CPU+GPU+NPU dispatch demonstration (future)
- **Weighing lysimeter**: Dong & Hansen (2023) load cell → direct ET (Exp 016 complete)
- **Coverage**: 97.45% → target 98%+ (remaining gaps: GPU-dependent code paths)

### What Good Science Looks Like

Every Dong lab paper we reproduced:
1. Uses published equations (FAO-56, Topp, van Genuchten) — anyone can implement
2. Uses measurable inputs (temperature, humidity, soil permittivity) — no proprietary sensors
3. Reports quantitative results (RMSE, R², IA) — we can check
4. The IoT system costs ~$200/node — consumer hardware, not lab equipment

This is *exactly* the kind of science that benefits from sovereign compute.
The farmer doesn't need a $5000 Campbell Scientific station. They need a
$200 sensor, Open-Meteo weather data, and a $600 GPU running BarraCuda.

---

## Extension Explorations

With 22 paper reproductions validated and the full Python → Rust CPU → GPU pipeline
operational, airSpring can now extend beyond reproduction into new science. These
explorations use the validated stack to answer questions the original papers did not.

| File | Title | Data Tier | Status |
|------|-------|:---------:|--------|
| [open_data_atlas.md](open_data_atlas.md) | Michigan Crop Water Atlas (80yr, 100 stations) | Tier 1 (free) | **Complete** — 1393/1393 atlas checks, 15,300 station-days |
| [yield_validation.md](yield_validation.md) | Stewart Yield Model vs USDA NASS Reality | Tier 1 (free) | Planning |
| [et_gold_standard.md](et_gold_standard.md) | Direct ET Validation via AmeriFlux Eddy Covariance | Tier 2 (registration) | Planning |
| [forecast_scheduling.md](forecast_scheduling.md) | Predictive Irrigation with Weather Forecasts | Tier 2 (API key) | Planning |
| [soil_moisture_validation.md](soil_moisture_validation.md) | Richards theta(t) vs USDA SCAN In-Situ | Tier 2 (free) | Planning |

Cross-spring explorations (no-till Anderson coupling, soil microbiome response)
are documented in `gen3/baseCamp/06_notill_anderson.md`.

### Data Extension Roadmap

**Tier 1 — Trivial (tools exist, just download)**

| Source | Records | Size | Cost | Notes |
|--------|---------|------|------|-------|
| Open-Meteo 80yr (100 stations) | ~2.9M | ~600MB | Free | 15,300 station-days, atlas complete |
| Open-Meteo 80yr (100 stations) | ~2.9M | ~600MB | Free | More lat/lon points |
| USDA NASS crop yields (Michigan) | ~66K | ~10MB | Free | County-level CSV |
| FAO-56 additional crop Kc | ~50 crops | ~5KB | Free | Published tables |

**Tier 2 — Moderate (new API integrations)**

| Source | Records | Size | Cost | Notes |
|--------|---------|------|------|-------|
| Open-Meteo full Michigan grid (10km) | ~292M | ~60GB | Free | ~1000 grid cells |
| USDA SCAN soil moisture stations | ~2M | ~400MB | Free | In-situ theta(t) |
| AmeriFlux/FLUXNET eddy covariance | ~365K | ~75MB | Free (registration) | Direct ET |
| PRISM 4km daily (Michigan) | ~730M | ~150GB | Free (academic) | Higher-res |

**Tier 3 — Substantial (satellite, reanalysis)**

| Source | Records | Size | Cost | Notes |
|--------|---------|------|------|-------|
| NASA SMAP soil moisture | ~1.5M | ~300MB | Free | 9km, 2015-present |
| ERA5-Land (Michigan) | ~15B | ~3TB | Free (CDS) | 9km, hourly, soil layers |
| Sentinel-2 NDVI (Michigan) | ~50K tiles | ~500GB | Free (Copernicus) | Crop health |

### Compute Requirements (Eastgate: i9-12900K + RTX 4070)

| Workload | Scale | CPU Time | GPU Time |
|----------|-------|----------|----------|
| ET₀ for 100 stations, 80yr | 2.9M calcs | 0.2 sec | ~0.01 sec |
| Water balance 100 stations, 80yr | 2.9M steps | ~3 sec | ~0.1 sec |
| Richards 1D, 1000 grid cells, 80yr | 29M sims | ~2.2 hrs | ~6 min |
| Kriging 100 stations per timestep | O(100^3) x 29K | ~8 hrs | ~20 min |
| Full Michigan grid 80yr (ET₀+WB+yield) | ~1B calcs | ~2 min | ~5 sec |

Tier 1-2 data fits on Eastgate (2TB NVMe). Tier 3 benefits from Westgate ZFS
(76TB cold). Compute is not the bottleneck — download time is.

### Primal Integration Path

```
Step 1: NestGate data providers (Open-Meteo, NOAA, USDA)
        Model after ncbi_live_provider.rs — download once, store with provenance
Step 2: Local NUCLEUS on Eastgate (Tower + Node + Nest)
        airSpring as first workload consumer via capability.call
Step 3: LAN HPC (Plasmodium across gates via 10G mesh)
        Westgate=Nest (76TB), Southgate=Node (RTX 3090), Eastgate=Node+NPU
Step 4: Full scale
        Strandgate (dual EPYC), Northgate (RTX 5090), cross-spring pipelines
```
