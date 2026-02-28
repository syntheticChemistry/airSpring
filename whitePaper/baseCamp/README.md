# baseCamp: Per-Faculty Research Briefings

**Updated**: February 28, 2026
**Project**: airSpring — Ecological & Agricultural Sciences (v0.5.4)
**Status**: 54 experiments, 1237/1237 Python + 618 Rust lib tests + 1498 atlas checks + 33/33 cross-validation + 11 Tier A + 4 Tier B GPU orchestrators + seasonal pipeline + atlas stream + MC GPU path + GPU math portability 46/46 + Titan V GPU live + AKD1000 NPU live + metalForge live (5 substrates, 18 workloads, 29/29 dispatch) + 25.9× CPU speedup + 8 ET₀ methods + coupled runoff-infiltration (292/292) + VG inverse (84/84) + full-season WB audit (34/34) + 42+ named constants + zero dead code + capability-based GPU + ToadStool S68 sync (universal precision, 700 WGSL, 6-Spring provenance, 30/30 benchmarks)

---

## Evolution Path

```
Phase 0   Python/R baselines    — reproduce paper results with original tools (1237/1237)
Phase 0+  Real open data        — compute on Open-Meteo, NOAA, USDA (no institutional access)
Phase 1   Rust BarraCuda CPU    — cross-validated to 1e-5 vs Python (618 lib + 1498 atlas, 59 binaries + 30 benchmarks)
Phase 1.5 CPU benchmark         — 25.9× Rust-vs-Python geometric mean (8/8 parity)
Phase 2   BarraCuda GPU bridge  — 11 Tier A modules wired (cross-spring S68 fully rewired)
Phase 2.5 Tier B orchestrators — Hargreaves (op=6), Kc climate (op=7), dual Kc (op=8), sensor cal (op=5)
Phase 2.6 Seasonal pipeline    — ET₀→Kc→WB→Yield chained, atlas stream, MC ET₀ GPU path
Phase 3   GPU live dispatch     — Titan V validated (24/24 PASS, 0.04% seasonal parity, 10K batch)
Phase 3.5 NPU edge             — AKD1000 live: 3 experiments, ~48µs inference, LOCOMOS power budget
Phase 3.7 metalForge live      — RTX 4070 + Titan V + AKD1000 + i9-12900K discovered, 18 workloads route
Phase 3.8 Cross-system routing — GPU+NPU+CPU dispatch proven (29/29 PASS), NUCLEUS atomic ready
Phase 4   Penny Irrigation      — sovereign scheduling on consumer hardware ($600 GPU + $99 NPU)
```

## Faculty Summary

| Faculty | Institution | Track | Papers | Experiments | Checks | Domain |
|---------|------------|-------|:------:|:-----------:|:------:|--------|
| Dong | MSU BAE | Irrigation & Soil | 10+ | 54 | 1237+618 | ET₀ (8 methods), soil, IoT, WB, dual Kc, Richards, yield, ensemble, bias correction, GPU parity, GPU math portability, metalForge dispatch, Anderson coupling, SCS-CN + Green-Ampt (coupled), VG inverse, full-season WB audit |

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
| 23 | NASS Yield Validation (Stewart 1977) — Exp 024 | 0→CPU | 41+40 | FAO-56 Table 24 + synthetic MI weather |
| 24 | Forecast Scheduling Hindcast — Exp 025 | 0→CPU | 19+19 | Synthetic (deterministic RNG) |
| 25 | USDA SCAN Soil Moisture (Richards 1D) — Exp 026 | 0→CPU | 34+34 | Carsel & Parrish (1988) + SCAN (open) |
| 26 | Multi-Crop Water Budget (5 crops) — Exp 027 | 0→CPU | 47+47 | FAO-56 Tables 12/17/24 + synthetic (open) |
| 27 | NPU Edge Inference (AKD1000) — Exp 028 | 1 | 35+21 | BrainChip AKD1000 (live hardware) |
| 28 | Funky NPU for Agricultural IoT — Exp 029 | 1 | 32/32 | AKD1000 live (streaming, evolution, LOCOMOS power) |
| 29 | High-Cadence NPU Pipeline — Exp 029b | 1 | 28/28 | AKD1000 live (1-min cadence, burst, fusion, hot-swap) |
| 30 | AmeriFlux Eddy Covariance ET (Baldocchi 2003) — Exp 030 | 0→CPU | 27+27 | AmeriFlux (free registration) |
| 31 | Hargreaves-Samani (1985) Temperature ET₀ — Exp 031 | 0→CPU | 24+24 | FAO-56 Eq. 52 (open literature) |
| 32 | Ecological Diversity Indices — Exp 032 | 0→CPU | 22+22 | Analytical (published formulas) |
| 33 | Makkink (1957) Radiation-Based ET₀ — Exp 033 | 0→CPU | 21+16 | Lysimeter + de Bruin 1987 |
| 34 | Turc (1961) Temperature-Radiation ET₀ — Exp 034 | 0→CPU | 22+17 | Annales Agronomiques + pyet |
| 35 | Hamon (1961) Temperature-Based PET — Exp 035 | 0→CPU | 20+19 | Lu et al. 2005 (T + day length) |
| 36 | biomeOS Neural API Round-Trip — Exp 036 | 0→CPU | 14+29 | JSON-RPC 2.0 parity |
| 37 | ET₀ Ensemble Consensus (6-Method) — Exp 037 | 0→CPU | 9+17 | PM/PT/HG/Mak/Turc/Hamon |
| 38 | Pedotransfer → Richards Coupling — Exp 038 | 0→CPU | 29+32 | Saxton-Rawls → VG → Richards |
| 39 | Cross-Method ET₀ Bias Correction — Exp 039 | 0→CPU | 24+24 | Linear correction factors |
| 40 | CPU vs GPU Parity — Exp 040 | 0→GPU | 22+26 | Bit-identical CPU fallback |
| 41 | metalForge Mixed-Hardware Dispatch — Exp 041 | 0→GPU | 14+18 | 18 workloads, cross-system routing |
| 42 | Seasonal Batch ET₀ at GPU Scale — Exp 042 | 0→GPU | 18+21 | 1,460 station-days batch |
| 43 | Titan V GPU Live Dispatch — Exp 043 | GPU | 24 | WGSL shader on GV100, 0.04% parity |
| 44 | metalForge Live Hardware Probe — Exp 044 | GPU+NPU | 17 | 5 substrates discovered live |
| 45 | Anderson Soil-Moisture Coupling — Exp 045 | 0→CPU | 55+95 | θ→S_e→d_eff→QS regime, cross-spring |
| 46 | Atlas Stream Real Data Validation — Exp 046 | GPU | 73/73 | 80yr Open-Meteo, seasonal pipeline + atlas stream |
| 47 | GPU Math Portability Validation — Exp 047 | 0→GPU | 21+46 | All 13 GPU modules CPU↔GPU identical |
| 48 | NCBI 16S + Soil Moisture Coupling — Exp 048 | 0→CPU | 14+29 | Cross-spring: soil moisture → 16S diversity |
| 49 | Blaney-Criddle (1950) Temperature PET — Exp 049 | 0→CPU | 18+18 | 8th ET₀ method, USDA-SCS TP-96 |
| 50 | SCS Curve Number Runoff (USDA 1972) — Exp 050 | 0→CPU | 38+38 | Industry-standard rainfall-runoff Q = (P-Ia)²/(P-Ia+S) |
| 51 | Green-Ampt (1911) Infiltration — Exp 051 | 0→CPU | 37+37 | Newton-Raphson implicit, 7 Rawls soils, ponding time |
| 52 | SCS-CN + Green-Ampt Coupled Runoff-Infiltration — Exp 052 | 0→CPU | 292+292 | Rainfall → runoff → infiltration → surface storage partitioning |
| 53 | Van Genuchten Inverse Parameter Estimation — Exp 053 | 0→CPU | 84+84 | Forward VG, Mualem K(h), θ→h→θ round-trip via Brent inversion |
| 54 | Full-Season Irrigation Water Budget — Exp 054 | 0→CPU | 34+34 | Synthetic weather → PM ET₀ → Kc → WB → Stewart yield, 4 crops |

### Rust Validation (Phase 1+3) — 59 binaries + 30 cross-spring benchmarks

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
| `validate_scheduling` | 28 | 5 irrigation strategies, mass balance, yield ordering |
| `validate_lysimeter` | 25 | mass→ET, temp compensation, calibration, diurnal |
| `validate_sensitivity` | 23 | OAT ±10%, 3 climatic zones, monotonicity, ranking |
| `validate_priestley_taylor` | 32 | PT α=1.26, analytical, cross-val vs PM |
| `validate_et0_intercomparison` | 36 | PM/PT/HG, 6 MI stations, R², bias, RMSE |
| `validate_thornthwaite` | 50 | Thornthwaite monthly ET₀ |
| `validate_gdd` | 26 | GDD accumulation, kc_from_gdd |
| `validate_pedotransfer` | 58 | Saxton-Rawls 2006 |
| `validate_nass_yield` | 40 | Stewart pipeline, 5 MI crops, drought ranking |
| `validate_forecast` | 19 | Forecast vs perfect knowledge, noise sensitivity |
| `validate_scan_moisture` | 34 | Richards vs SCAN, VG θ/K, 3 soil textures |
| `validate_multicrop` | 47 | 5-crop water budget, irrigated/rainfed/dual Kc |
| `validate_npu_eco` | 35 | AKD1000 crop stress/irrigation/anomaly classifiers |
| `validate_npu_funky_eco` | 32 | Streaming, evolution, LOCOMOS power, multi-crop |
| `validate_npu_high_cadence` | 28 | 1-min cadence, burst, fusion, ensemble, hot-swap |
| `validate_ameriflux` | 27 | AmeriFlux eddy covariance ET validation |
| `validate_hargreaves` | 24 | Hargreaves-Samani temperature-based ET₀ |
| `validate_diversity` | 22 | Shannon, Simpson, richness diversity indices |
| `validate_gpu_math` | 46 | All 13 GPU orchestrators CPU↔GPU parity |
| `cross_validate` | 75 | Python↔Rust exact match (tol=1e-5) |
| `validate_coupled_runoff` | 292 | SCS-CN + Green-Ampt coupled rainfall partitioning |
| `validate_vg_inverse` | 84 | VG forward, Mualem K(h), θ→h→θ Brent round-trip |
| `validate_season_wb` | 34 | Full-season: weather → ET₀ → Kc → WB → yield (4 crops) |
| `validate_dispatch_routing` | 21 | metalForge CPU+GPU+NPU dispatch routing (forge) |

### GPU Orchestrators (Phase 2+2.5) — 11 Tier A + 4 Tier B + 3 pipeline

| Orchestrator | BarraCuda Primitive | Cross-Spring Provenance | Status |
|-------------|--------------------|----|---|
| `BatchedEt0` | `batched_elementwise_f64` (op=0) | hotSpring `pow_f64` fix (TS-001) | **GPU-FIRST** |
| `BatchedWaterBalance` | `batched_elementwise_f64` (op=1) | Multi-spring shared | **GPU-STEP** |
| `BatchedDualKc` | `batched_elementwise_f64` (op=8, pending) | airSpring v0.5.2 | **Wired** (Tier B) |
| `BatchedHargreaves` | `batched_elementwise_f64` (op=6, pending) | FAO-56 Eq. 52 | **Wired** (Tier B) |
| `BatchedKcClimate` | `batched_elementwise_f64` (op=7, pending) | FAO-56 Eq. 62 | **Wired** (Tier B) |
| `BatchedSensorCal` | `batched_elementwise_f64` (op=5, pending) | Dong et al. 2024 | **Wired** (Tier B) |
| `KrigingInterpolator` | `kriging_f64::KrigingF64` | wetSpring spatial interpolation | **Integrated** |
| `SeasonalReducer` | `fused_map_reduce_f64` | wetSpring + airSpring TS-004 fix | **GPU N≥1024** |
| `StreamSmoother` | `moving_window_stats` | wetSpring S28+ environmental | **Wired** |
| `BatchedRichards` | `pde::richards::solve_richards` | airSpring→ToadStool S40 absorption | **Wired** |
| `fit_*_nm/global` | `optimize::nelder_mead` + `multi_start` | neuralSpring optimizer | **Wired** |
| `SeasonalPipeline` | Chains ops 0→7→1→yield | Zero round-trip architecture | **CPU chained** |
| `AtlasStream` | `UnidirectionalPipeline` (pending) | Multi-year regional ET₀ | **CPU chained** |
| `mc_et0_gpu` | `mc_et0_propagate_f64.wgsl` (pending) | groundSpring xoshiro + Box-Muller | **Wired** (Tier B) |

### CPU Benchmarks (v0.5.3) — Rust 25.9× Faster Than Python (8/8 Parity)

| Algorithm | Speedup | Parity | Cross-Spring Provenance |
|-----------|:-------:|:------:|------------------------|
| FAO-56 PM ET₀ | **15×** | ✓ | hotSpring df64, multi-spring elementwise |
| Hargreaves-Samani | **114×** | ✓ | FAO-56 Eq. 52, temperature-only |
| Water Balance Step | **190×** | ✓ | FAO-56 Ch. 8, stress coefficient |
| Anderson Coupling | **94×** | ✓ | Cross-spring: θ→S_e→d_eff→QS |
| Season Sim (153d) | **44×** | ✓ | Full pipeline: ET₀→Kc→WB→yield |
| Shannon Diversity | **26×** | ✓ | wetSpring diversity metrics |
| Van Genuchten θ(h) | **6×** | ✓ | hotSpring df64 precision |
| Thornthwaite PET | **1×** | ✓ | Higher-fidelity Rust (365 vs 12 trig) |

### Evolution Documents

| Document | Purpose |
|----------|---------|
| `barracuda/EVOLUTION_READINESS.md` | Tier A/B/C breakdown, absorbed vs stays-local, quality gates |
| `metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66) |
| `wateringHole/handoffs/` | V034 active — experiment buildout + deep debt resolution |
| `specs/CROSS_SPRING_EVOLUTION.md` | 774 WGSL shader provenance across all Springs |

### Next Steps (Dong Lab)

- **GPU math portability proven**: All 13 GPU modules validated CPU↔GPU identical (46/46 PASS, Exp 047)
- **ToadStool absorption**: Ops 5-8 pending absorption; CPU fallback validated, GPU dispatch activates automatically
- **GPU at scale**: Profile `compute_gpu()` at N=100K+ (multi-year regional grids, crossover point via `AtlasStream`)
- **Richards + isotherm on GPU**: Move remaining Tier B modules to pure GPU via ToadStool shaders
- **NUCLEUS local deployment**: Tower → Node → Nest on Eastgate, then LAN HPC across gates
- **NestGate data pipeline**: Open-Meteo + NCBI 16S coupling for baseCamp 06 extension
- **biomeOS integration**: Neural API `capability.call` for cross-primal compute orchestration
- **Paper 12+**: Multi-sensor calibration network (awaiting field data from new lab)
- **Coverage**: 97.06% → target 98%+ (remaining gaps: GPU-dependent code paths now Tier B wired)

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

With 54 experiments validated and the full Python → Rust CPU → Titan V GPU live →
GPU math portability (13 modules, 46/46) → metalForge mixed hardware pipeline proven,
airSpring can now extend beyond reproduction into new science. These explorations use the validated stack to answer
questions the original papers did not.

| File | Title | Data Tier | Status |
|------|-------|:---------:|--------|
| [open_data_atlas.md](open_data_atlas.md) | Michigan Crop Water Atlas (80yr, 100 stations) | Tier 1 (free) | **Complete** — 1393/1393 atlas checks, 15,300 station-days |
| [yield_validation.md](yield_validation.md) | Stewart Yield Model vs USDA NASS Reality | Tier 1 (free) | **Validated** — Exp 024, 40/40 Rust checks |
| [et_gold_standard.md](et_gold_standard.md) | Direct ET Validation via AmeriFlux Eddy Covariance | Tier 2 (registration) | **Complete** — Exp 030, 27+27 checks |
| [forecast_scheduling.md](forecast_scheduling.md) | Predictive Irrigation with Weather Forecasts | Tier 2 (API key) | **Validated** — Exp 025, 19/19 Rust checks |
| [soil_moisture_validation.md](soil_moisture_validation.md) | Richards theta(t) vs USDA SCAN In-Situ | Tier 2 (free) | **Validated** — Exp 026, 34/34 Rust checks |
| [npu_iot_locomos.md](npu_iot_locomos.md) | NPU-Accelerated Agricultural IoT (LOCOMOS → Edge Sovereign) | Tier 0 (hardware) | **Validated** — Exp 028+029, 32+35+21 checks, live AKD1000 |

| [ncbi_16s_coupling.md](ncbi_16s_coupling.md) | NCBI 16S + Soil Moisture Coupling (baseCamp 06 ext.) | Tier 2 (NCBI free) | **Planned** — providers validated (23/23), pipeline designed |

Cross-spring explorations (no-till Anderson coupling, soil microbiome response)
are documented in `ecoPrimals/whitePaper/gen3/baseCamp/06_notill_anderson.md`.
NPU agricultural IoT is in `ecoPrimals/whitePaper/gen3/baseCamp/08_npu_agricultural_iot.md`.

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
Step 0: ✓ DONE — metalForge cross-system routing (GPU+NPU+CPU, 18 workloads, 29/29)
Step 1: NestGate data providers (Open-Meteo, NOAA CDO, USDA NASS)
        Model after ncbi_live_provider.rs — download once, store with provenance
        NestGate already has: NCBI, Ensembl, HuggingFace, OpenMeteo, NOAA CDO, USDA NASS
Step 2: Local NUCLEUS on Eastgate (Tower + Node + Nest)
        biomeOS graphs orchestrate: airSpring workloads via capability.call
        Tower atomic: BearDog (crypto/TLS) + Songbird (mesh/discovery)
        Node atomic: Tower + ToadStool (compute/GPU)
        Nest atomic: Tower + NestGate (storage/provenance)
Step 3: LAN HPC (Plasmodium across gates via 10G backbone)
        Westgate=Heavy Nest (76TB ZFS), Southgate=Node (RTX 3090), Eastgate=Node+NPU
        PCIe bypass: NPU→GPU direct (skip CPU roundtrip) via metalForge substrate routing
Step 4: Full scale
        Strandgate (dual EPYC, 256GB ECC), Northgate (RTX 5090, 192GB DDR5)
        Cross-spring pipelines: airSpring+wetSpring+groundSpring via biomeOS graphs
```
