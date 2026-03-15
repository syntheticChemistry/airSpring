# airSpring — Paper Review Queue

**Last Updated**: March 15, 2026
**Purpose**: Track papers for reproduction/review, ordered by priority
**Status**: 87 experiments (1284/1284 Python + 851 Rust lib + 280 integration + 61 forge tests + 381/381 validation checks + 146/146 cross-spring evolution + 33/33 cross-validation). barraCuda 0.3.5 (wgpu 28), Edition 2024, 94 binaries. 14.3× Rust-vs-Python geometric mean speedup (24/24 algorithms, 21/21 CPU-GPU parity modules). All 20 ops upstream (`BatchedElementwiseF64`), `local_dispatch` retired. metalForge 66/66 mixed pipeline. V082 handoff. biomeOS niche: 41 capabilities, 4 deploy graphs. Deep code quality complete. Exp 084 CPU/GPU parity (21/21), Exp 085 toadStool dispatch (19/19), Exp 086 metalForge NUCLEUS (17/17), Exp 087 graph coordination (22/22).

---

## Completed Reproductions

| # | Paper | Phase | Checks | Faculty | Control File | Open Data |
|---|-------|:-----:|:------:|---------|-------------|:---------:|
| 1 | Allen et al. (1998) FAO-56 Penman-Monteith — Ch 2/4 | 0 | 64/64 | Standard | `benchmark_fao56.json` | FAO-56 tables (open literature) |
| 2 | Dong et al. (2020) Soil sensor calibration — CS616/EC5 | 0 | 36/36 | Dong | `benchmark_dong2020.json` | Published Tables 3-4 |
| 3 | Dong et al. (2024) IoT irrigation pipeline — SoilWatch 10 | 0 | 24/24 | Dong | `benchmark_dong2024.json` | Published tables/equations |
| 4 | FAO-56 Chapter 8 — Water balance scheduling | 0 | 18/18 | Standard | `benchmark_water_balance.json` | FAO-56 Ch 8 + USDA |
| 5 | Real data pipeline — 100 Michigan stations, 15,300 days | 0+ | R²=0.967 | Dong | Python scripts | Open-Meteo ERA5 (free) |
| 6 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc (Kcb+Ke) | 0 | 63/63 | Standard | `benchmark_dual_kc.json` | FAO-56 Tables 17, 19 (open literature) |
| 7 | Regional ET₀ intercomparison — 6 Michigan stations | 0 | 61/61 | Dong | `regional_et0_intercomparison.py` | Open-Meteo ERA5 (free) |
| 8 | Islam et al. (2014) No-till + Allen FAO-56 Ch 11 cover crops | 0 | 40/40 | Standard | `benchmark_cover_crop_kc.json` | ISWCR + FAO-56 (open) |
| 9 | Richards equation (van Genuchten-Mualem) | 0+1 | 14+15 | Dong | `benchmark_richards.json` | Published parameters |
| 10 | Kumari et al. (2025) Biochar P adsorption | 0+1 | 14+14 | Dong | `benchmark_biochar.json` | Representative literature data |
| 11 | 60-year water balance (OSU Triplett) | 0+1 | 10+11 | Standard | `benchmark_long_term_wb.json` | Open-Meteo ERA5 (free) |
| 12 | Stewart (1977) yield response to water stress | 0+1 | 32+32 | Standard | `benchmark_yield_response.json` | FAO-56 Table 24 (open) |
| 13 | Dong et al. (2019) CW2D Richards extension | 0+1 | 24+24 | Dong | `benchmark_cw2d.json` | HYDRUS CW2D params (published) |
| 14 | Ali, Dong & Lavely (2024) Irrigation scheduling | 0+1 | 25+28 | Dong | `benchmark_scheduling.json` | FAO-56 + synthetic (open) |
| 15 | Dong & Hansen (2023) Weighing lysimeter ET | 0+1 | 26+25 | Dong | `benchmark_lysimeter.json` | Published design params |
| 16 | FAO-56 ET₀ sensitivity analysis (Gong 2006) | 0+1 | 23+23 | Standard | `benchmark_sensitivity.json` | FAO-56 + literature (open) |
| 17 | Priestley & Taylor (1972) radiation-based ET₀ | 0+1 | 32+32 | Standard | `benchmark_priestley_taylor.json` | FAO-56 intermediates (open literature) |
| 18 | ET₀ 3-method intercomparison (PM/PT/HG) — 6 stations | 0+1 | 36+36 | Dong | `benchmark_et0_intercomparison.json` | Open-Meteo ERA5 (free) |
| 19 | Thornthwaite (1948) monthly ET₀ — Exp 021 | 0+1 | 23+50 | Standard | `benchmark_thornthwaite.json` | Temperature-based heat index (open) |
| 20 | Growing Degree Days (GDD) — Exp 022 | 0+1 | 33+26 | Standard | `benchmark_gdd.json` | Phenology, kc_from_gdd (open) |
| 21 | Saxton & Rawls (2006) pedotransfer — Exp 023 | 0+1 | 70+58 | Standard | `benchmark_pedotransfer.json` | θs/θr/Ks from texture (open) |
| 22 | NASS Yield Validation (Stewart 1977) — Exp 024 | 0+1 | 41+40 | Standard | `benchmark_nass_yield.json` | FAO-56 Table 24 + synthetic MI weather |
| 23 | Forecast Scheduling Hindcast — Exp 025 | 0+1 | 19+19 | Dong | `benchmark_forecast_scheduling.json` | Synthetic (deterministic RNG) |
| 24 | USDA SCAN Soil Moisture (Richards 1D) — Exp 026 | 0+1 | 34+34 | Dong | `benchmark_scan_moisture.json` | Carsel & Parrish (1988) + SCAN (open) |
| 25 | Multi-Crop Water Budget (5 crops) — Exp 027 | 0+1 | 47+47 | Standard | `benchmark_multicrop.json` | FAO-56 Tables 12/17/24 + synthetic (open) |
| 26 | NPU Edge Inference (AKD1000) — Exp 028 | 1 | 35+21 | — | metalForge forge crate | BrainChip AKD1000 (live hardware) |
| 27 | Funky NPU for Agricultural IoT — Exp 029 | 1 | 32/32 | Dong | `validate_npu_funky_eco` | AKD1000 live (streaming, evolution, LOCOMOS power) |
| 28 | High-Cadence NPU Pipeline — Exp 029b | 1 | 28/28 | Dong | `validate_npu_high_cadence` | AKD1000 live (1-min cadence, burst, fusion, hot-swap) |
| 29 | AmeriFlux Eddy Covariance ET (Baldocchi 2003) — Exp 030 | 0+1 | 27+27 | Standard | `benchmark_ameriflux_et.json` | AmeriFlux (free registration) |
| 30 | Hargreaves-Samani (1985) Temperature ET₀ — Exp 031 | 0+1 | 24+24 | Standard | `benchmark_hargreaves.json` | FAO-56 Eq. 52 (open literature) |
| 31 | Ecological Diversity Indices — Exp 032 | 0+1 | 22+22 | Standard | `benchmark_diversity.json` | Analytical (published formulas) |
| 32 | Makkink (1957) Radiation-Based ET₀ — Exp 033 | 0+1 | 21+16 | Standard | `benchmark_makkink.json` | De Bruin (1987) coefficients (open literature) |
| 33 | Turc (1961) Temperature-Radiation ET₀ — Exp 034 | 0+1 | 22+17 | Standard | `benchmark_turc.json` | Published equation (open literature) |
| 34 | Hamon (1961) Temperature-Based PET — Exp 035 | 0+1 | 20+19 | Standard | `benchmark_hamon.json` | Lu et al. (2005) coefficients (open literature) |
| 35 | biomeOS Neural API Round-Trip Parity — Exp 036 | 0+1 | 14+29 | — | `benchmark_neural_api.json` | Neural API spec (biomeOS architecture) |
| 36 | ET₀ Ensemble Consensus (6-Method) — Exp 037 | 0+1 | 9+17 | Standard | `benchmark_et0_ensemble.json` | Multi-method weighted consensus (open literature) |
| 37 | Pedotransfer → Richards Coupled — Exp 038 | 0+1 | 29+32 | FAO-56/SSSA | `benchmark_pedotransfer_richards.json` | SR→VG→Richards soil dynamics coupling |
| 38 | Cross-Method ET₀ Bias Correction — Exp 039 | 0+1 | 24+24 | Standard | `benchmark_et0_bias.json` | Linear bias correction factors (4 methods × 4 climates) |
| 39 | CPU vs GPU Parity Validation — Exp 040 | 1+2 | 22+26 | — | `benchmark_cpu_gpu_parity.json` | BatchedEt0 CPU↔GPU bit-identical proof |
| 40 | metalForge Mixed-Hardware Dispatch — Exp 041 | 2 | 14+18 | — | `benchmark_metalforge_dispatch.json` | GPU/NPU/Neural/CPU capability routing |
| 41 | Seasonal Batch ET₀ at GPU Scale — Exp 042 | 1+2 | 18+21 | Standard | `benchmark_seasonal_batch.json` | 365×4 station-days batch via BatchedEt0 |
| 42 | Titan V GPU Live Dispatch — Exp 043 | 3 | 24 | — | `validate_gpu_live` | Live WGSL shader on Titan V GV100, 10K batch scaling |
| 43 | metalForge Live Hardware Probe — Exp 044 | 3 | 17 | — | `validate_live_hardware` | RTX 4070 + Titan V + AKD1000 + i9-12900K discovery + dispatch |
| 44 | GPU Math Portability (all 13 modules) — Exp 047 | 1+2 | 21+46 | — | `benchmark_gpu_math.json` | All GPU orchestrators CPU↔GPU identical |
| 45 | Blaney-Criddle (1950) Temperature PET — Exp 049 | 0+1 | 18+18 | Standard | `benchmark_blaney_criddle.json` | USDA-SCS Tech Paper 96 (open literature) |
| 46 | SCS Curve Number Runoff (USDA 1972) — Exp 050 | 0+1 | 38+38 | Standard | `benchmark_scs_cn.json` | USDA NEH-4 / TR-55 (public domain) |
| 47 | Green-Ampt (1911) Infiltration — Exp 051 | 0+1 | 37+37 | Standard | `benchmark_green_ampt.json` | Rawls et al. (1983) Table 1 (open literature) |
| 48 | SCS-CN + Green-Ampt Coupled Runoff-Infiltration — Exp 052 | 0+1 | 292+292 | Standard | `benchmark_coupled_runoff.json` | NEH-4 + Rawls (1983) (open literature) |
| 49 | Van Genuchten Inverse Parameter Estimation — Exp 053 | 0+1 | 84+84 | Standard | `benchmark_vg_inverse.json` | Carsel & Parrish (1988) Table 1 (open literature) |
| 50 | Full-Season Irrigation Water Budget — Exp 054 | 0+1 | 34+34 | Standard | `benchmark_season_wb.json` | FAO-56 Ch 2-8 + Stewart (1977) (open literature) |
| 51 | Anderson Soil-Moisture Coupling — Exp 045 | 0+1 | 55+95 | Standard | `benchmark_anderson_coupling.json` | VG + Anderson localization (open literature) |
| 52 | Climate Scenario Analysis — Exp 058 | 1 | 46 | Standard | `benchmark_climate_scenario.json` | Synthetic scenarios (open) |
| 53 | NCBI 16S + Soil Microbiome Anderson — Exp 048 | 0+1 | 14+29 | Standard | `benchmark_ncbi_16s_coupling.json` | NCBI 16S (public) |
| 54 | Tissue Diversity (Paper 12) — Exp 066 | 0+1 | 30+30 | Anderson | `benchmark_tissue_diversity.json` | Analytical (open literature) |
| 55 | GPU Streaming Multi-Field — Exp 070 | 2 | 57 | — | `validate_gpu_streaming_multi_field` | Multi-field CPU parity + atlas-scale (50 stations) |
| 56 | CPU Parity & Speedup Benchmark — Exp 071 | 1+2 | 34 | Standard | `validate_cpu_parity_benchmark` | 9 domains, 10M ET₀/s, 13K× Python atlas-scale |
| 57 | Pure GPU End-to-End — Exp 072 | 2+3 | 46 | — | `validate_pure_gpu_multi_field` | 4-stage GPU, CPU↔GPU parity, 19.7× dispatch reduction |
| 58 | Monte Carlo ET₀ Uncertainty Propagation — Exp 079 | 0+1 | 12+26 | Standard | `benchmark_mc_et0.json` | FAO-56 + groundSpring UQ (open literature) |
| 59 | Bootstrap & Jackknife CI for Seasonal ET₀ — Exp 080 | 0+1 | 18+20 | Standard | `benchmark_bootstrap_jackknife.json` | Efron (1979) + synthetic ET₀ (open) |
| 60 | Standardized Precipitation Index (SPI) — Exp 081 | 0+1 | 17+20 | Standard | `benchmark_drought_index.json` | McKee (1993) / WMO-1090 (open literature) |
| 61 | Cross-Spring Modern Systems Validation — Exp 082 | 1 | 36 | — | `validate_cross_spring_modern` | Provenance registry, autocorrelation, PrecisionRoutingAdvice |
| 62 | NUCLEUS Modern Deployment — Exp 083 | 1 | 43 | — | `validate_nucleus_modern` | biomeOS NUCLEUS, Tower/Node, 35 JSON-RPC, SPI/ACF/gamma_cdf parity |

### Controls Audit

All 61 completed papers have:
- **Digitized benchmarks** in `control/*/benchmark_*.json` (59 benchmark JSONs, 59 control directories)
- **Python control scripts** that validate against benchmarks (1284/1284 Python checks)
- **Rust validation binaries** (85 barracuda + 5 forge = 90 binaries) that load the same benchmarks
- **Open or published data** (no institutional access required)
- **Cross-validation** (33/33 Python↔Rust match at 1e-5; 690 crop-station yield pairs within 0.01; PT↔PM cross-validated)
- **GPU wiring**: 25 Tier A modules + fused Welford (SeasonalReducer) + fused Pearson (pairwise_correlation_gpu) + `BrentGpu` (VG inverse) + `RichardsGpu` (Picard)
- **CPU benchmarks**: 14.3× geometric mean speedup vs Python (24/24 algorithms, 21/21 CPU-GPU parity modules), 13,000× atlas-scale
- **GPU pipeline**: Local GPU 6/6 ops parity (<0.002 error). 27 GPU dispatch failures (upstream wgpu 28 NVK).
- **Cross-spring provenance**: 146/146 evolution benchmarks across 5 springs (barraCuda 0.3.3)

### Compute Pipeline Per Paper

| Paper | Python Control | BarraCuda CPU | BarraCuda GPU | metalForge Module |
|:-----:|:--------------:|:-------------:|:-------------:|:-----------------:|
| 1 | 64/64 | 31/31 (`validate_et0`) | `BatchedEt0` GPU-FIRST | `metrics` (RMSE, R²) |
| 2 | 36/36 | 40/40 (`validate_soil`) | `fit_ridge` (ridge regression) | `regression` (4 models) |
| 3 | 24/24 | 11/11 (`validate_iot`) | `StreamSmoother` (moving window) | `moving_window_f64` |
| 4 | 18/18 | 13/13 (`validate_water_balance`) | `BatchedWaterBalance` GPU-STEP | `hydrology` (WB) |
| 5 | R²=0.967 | 23/23 (`validate_real_data`) | All 11 Tier A modules | All 4 modules |
| 6 | 63/63 | 61/61 (`validate_dual_kc`) | `BatchedDualKc` (Tier B) | `hydrology` (Kc) |
| 7 | 61/61 | 61/61 (`validate_regional_et0`) | `BatchedEt0` at scale | `metrics` (IA, NSE) |
| 8 | 40/40 | 40/40 (`validate_cover_crop`) | `BatchedDualKc` + mulch | `hydrology` (cover Kc) |
| 9 | 14/14 | 15/15 (`validate_richards`) | `BatchedRichards` **WIRED** | VG **absorbed** |
| 10 | 14/14 | 14/14 (`validate_biochar`) | `fit_*_nm` **WIRED** | isotherm **absorbed** |
| 11 | 10/10 | 11/11 (`validate_long_term_wb`) | `BatchedEt0` + `BatchedWB` | `hydrology` (60yr) |
| 12 | 32/32 | 32/32 (`validate_yield`) | `BatchedElementwise` (Tier B) | `yield_response` |
| 13 | 24/24 | 24/24 (`validate_cw2d`) | `BatchedRichards` | VG (CW2D media) |
| 14 | 25/25 | 28/28 (`validate_scheduling`) | `BatchedWB` + `BatchedEt0` | `hydrology` (scheduling) |
| 15 | 26/26 | 25/25 (`validate_lysimeter`) | `BatchedEt0` (ground truth) | `metrics` (lysimeter) |
| 16 | 23/23 | 23/23 (`validate_sensitivity`) | `BatchedEt0` (perturbation) | `metrics` (sensitivity) |
| 17 | 32/32 | 32/32 (`validate_priestley_taylor`) | `BatchedElementwise` (Tier B, op=PT) | `evapotranspiration` (PT) |
| 18 | 36/36 | 36/36 (`validate_et0_intercomparison`) | All 3 methods at scale | `evapotranspiration` (PM+PT+HG) |
| 19 | 23/23 | 50/50 (`validate_thornthwaite`) | `BatchedElementwise` (Tier B) | `evapotranspiration` (Thornthwaite) |
| 20 | 33/33 | 26/26 (`validate_gdd`) | GDD accumulation | `crop` (kc_from_gdd) |
| 21 | 70/70 | 58/58 (`validate_pedotransfer`) | Saxton-Rawls θs/θr/Ks | `soil_moisture` (pedotransfer) |
| 22 | 41/41 | 40/40 (`validate_nass_yield`) | `BatchedWB` + yield | `yield_response` + `water_balance` |
| 23 | 19/19 | 19/19 (`validate_forecast`) | `BatchedWB` + forecast | `water_balance` + forecast loop |
| 24 | 34/34 | 34/34 (`validate_scan_moisture`) | `BatchedRichards` | VG (SCAN soils) |
| 25 | 47/47 | 47/47 (`validate_multicrop`) | `BatchedWB` + `BatchedDualKc` | `hydrology` + `yield_response` |
| 26 | — | 35/35 (`validate_npu_eco`) | NPU dispatch | forge substrate + dispatch |
| 27 | — | 32/32 (`validate_npu_funky_eco`) | NPU streaming | AKD1000 DMA |
| 28 | — | 28/28 (`validate_npu_high_cadence`) | NPU high-cadence | AKD1000 + hot-swap |
| 29 | 27/27 | 27/27 (`validate_ameriflux`) | `BatchedEt0` | `metrics` (RMSE, R²) |
| 30 | 24/24 | 24/24 (`validate_hargreaves`) | `BatchedElementwise` | `evapotranspiration` (HG) |
| 31 | 22/22 | 22/22 (`validate_diversity`) | `DiversityFusionGpu` | `diversity` |
| 32 | 21/21 | 16/16 (`validate_makkink`) | `BatchedElementwise` (Tier B, op=Makkink) | `evapotranspiration` (Makkink) |
| 33 | 22/22 | 17/17 (`validate_turc`) | `BatchedElementwise` (Tier B, op=Turc) | `evapotranspiration` (Turc) |
| 34 | 20/20 | 19/19 (`validate_hamon`) | `BatchedElementwise` (Tier B, op=Hamon) | `evapotranspiration` (Hamon) |
| 35 | 14/14 | 29/29 (`validate_neural_api`) | Neural API `capability.call` | `neural` (biomeOS bridge) |
| 36 | 9/9 | 17/17 (`validate_et0_ensemble`) | Multi-method consensus | `evapotranspiration` (ensemble) |
| 37 | 29/29 | 32/32 (`validate_pedotransfer_richards`) | SR→VG→Richards coupling | `soil_moisture` + `richards` + `van_genuchten` |
| 38 | 24/24 | 24/24 (`validate_et0_bias`) | Bias correction factors | `evapotranspiration` (ensemble) |
| 45 | 18/18 | 18/18 (`validate_blaney_criddle`) | `BatchedElementwise` (Tier B, op=BC) | `evapotranspiration` (BC) |
| 46 | 38/38 | 38/38 (`validate_scs_cn`) | `BatchedElementwise` (Tier B, op=CN) | `runoff` (SCS-CN) |
| 47 | 37/37 | 37/37 (`validate_green_ampt`) | `BatchedElementwise` (Tier B, op=GA) | `infiltration` (Green-Ampt) |
| 48 | 292/292 | 292/292 (`validate_coupled_runoff`) | `BatchedElementwise` (CN+GA) | `runoff` + `infiltration` |
| 49 | 84/84 | 84/84 (`validate_vg_inverse`) | `BatchedOptimize` (Tier B, Brent) | `van_genuchten` (inverse) |
| 50 | 34/34 | 34/34 (`validate_season_wb`) | `SeasonalPipeline` (ET₀→Kc→WB→Yield) | All hydrology modules |
| 39 | 22/22 | 26/26 (`validate_cpu_gpu_parity`) | `BatchedEt0` + `BatchedWB` CPU↔GPU proof | `metrics` (parity) |
| 40 | 14/14 | 18/18 (`validate_dispatch`) | GPU/NPU/Neural/CPU routing | All substrates (5) |
| 41 | 18/18 | 21/21 (`validate_seasonal_batch`) | `BatchedEt0` 365×4 batch | `evapotranspiration` (GPU scale) |
| 42 | — | 24/24 (`validate_gpu_live`) | Titan V WGSL dispatch (live) | N/A (pure BarraCuda GPU) |
| 43 | — | 17/17 (`validate_live_hardware`) | RTX 4070 + Titan V + AKD1000 | All substrates (live) |
| 44 | 21/21 | 46/46 (`validate_gpu_math`) | All 13 GPU orchestrators | N/A (portability) |
| 51 | 55/55 | 95/95 (`validate_anderson`) | `DiversityFusionGpu` + Anderson W | `anderson` (θ→d_eff→QS) |
| 52 | — | 46/46 (`validate_climate_scenario`) | `SeasonalPipeline` (multi-scenario) | `yield_response` (climate) |
| 53 | 14/14 | 29/29 (`validate_ncbi_16s_coupling`) | `DiversityFusionGpu` + Anderson | `diversity` + `anderson` |
| 54 | 30/30 | 30/30 (`validate_tissue`) | `DiversityFusionGpu` | `tissue` (Anderson W) |
| 55 | — | 57/57 (`validate_gpu_streaming_multi_field`) | Multi-field `gpu_step()` per-day | `seasonal_pipeline` (M fields) |
| 56 | — | 34/34 (`validate_cpu_parity_benchmark`) | All 9 domains CPU parity | All modules (speedup) |
| 57 | — | 46/46 (`validate_pure_gpu_multi_field`) | All stages GPU + scaling | `seasonal_pipeline` (GPU) |
| 58 | 12/12 | 26/26 (`validate_mc_et0`) | `mc_et0_gpu` (CPU perturb → GPU batch ET₀) | `mc_et0` (UQ) |
| 59 | 18/18 | 20/20 (`validate_bootstrap_jackknife`) | `GpuBootstrap` + `GpuJackknife` (S71 shaders) | `bootstrap` + `jackknife` |
| 60 | 17/17 | 20/20 (`validate_drought_index`) | `BatchedElementwise` (Tier B, op=SPI) | `drought_index` (new) |
| 61 | — | 36/36 (`validate_cross_spring_modern`) | Provenance + autocorrelation + PrecisionRoutingAdvice | `gpu::autocorrelation` (new) |

---

## Review Queue

### Tier 1 — Direct extensions of current work

| # | Paper / Direction | Year | Faculty | Open Data? | Control Status | GPU Path |
|---|-------------------|------|---------|:----------:|:--------------:|----------|
| 6 | Dong et al. — Multi-sensor calibration network | 2024+ | Dong | Awaiting field data | None | Batch calibration (op=5) |
| 7 | Dong et al. — Full IoT irrigation + forecast | 2024+ | Dong | Awaiting field data | None | Forecast integration |

### Tier 2 — Cross-spring extensions

| # | Paper / Direction | Year | Faculty | Open Data? | Control Status | GPU Path |
|---|-------------------|------|---------|:----------:|:--------------:|----------|
| 9 | Regional ET₀ intercomparison — Michigan microclimates | — | Dong | **Yes** (80-yr Open-Meteo) | **61/61 PASS** (Phase 0, 2023) | `BatchedEt0` at scale |
| 10 | neuralSpring Exp 004 — Transfer learning MI→NM/CA | — | Cross-spring | Yes | Already validated | N/A (reference) |
| 11 | groundSpring Exp 003 — Error propagation through FAO-56 | — | Cross-spring | Yes | Already validated | N/A (reference) |

### Tier 3 — No-Till Soil Moisture & Anderson Geometry (baseCamp Sub-thesis 06)

baseCamp Sub-thesis 06 couples airSpring's soil moisture computation to the
Anderson localization model for QS prediction in no-till vs tilled soil.
Soil moisture θ(t) determines pore connectivity, which determines the effective
dimension of the Anderson lattice, which determines whether QS signals propagate.

| # | Paper / Direction | Year | Faculty | Open Data? | Control Status | GPU Path |
|---|-------------------|------|---------|:----------:|:--------------:|----------|
| 12 | Islam et al. "No-till and conservation agriculture: David Brandt farm" | 2014 | — | **Yes** (ISWCR) | **In Exp 011** (data digitized) | N/A (data extraction) |
| 13 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc for cover crops | 1998 | Standard | **Yes** (open literature) | **40/40 PASS** (Phase 0) | Batch Kc (op=7) |
| 14 | Soil moisture → Anderson d_eff coupling model — Exp 045 | — | Cross-spring | **Yes** (USDA + Open-Meteo) | **55+95 PASS** | `eco::anderson` (θ→d_eff) |
| 15 | OSU Triplett-Van Doren 60-year water balance reconstruction | — | Cross-spring | **Yes** (Open-Meteo 80-yr, USDA soils) | **10+11 PASS** (Exp 015) | `BatchedEt0` at scale |
| 16 | Cover crop water use & seasonal diversity dynamics | — | Dong | Awaiting field data | Future | Batch ET₀ with Kc schedule |

**Connection to wetSpring**: airSpring computes θ(t); wetSpring computes
Anderson r(t) from θ(t)-derived geometry. The cross-spring pipeline is:
`Open-Meteo weather → FAO-56 ET₀ → water balance θ(t) → pore_connectivity(t) → d_eff(t) → Anderson r(t) → QS_regime(t)`.

**Connection to groundSpring**: groundSpring Exp 003 already showed humidity
dominates ET₀ uncertainty at 66%. This propagates into the Anderson coupling:
moisture uncertainty → geometry uncertainty → QS prediction uncertainty.

### Tier 3.5 — NCBI 16S + Soil Moisture Coupling (baseCamp 06 extension)

Cross-spring experiment coupling NCBI soil metagenome data to airSpring's
moisture-driven Anderson QS model. Uses NestGate NCBI provider for 16S data
and Open-Meteo for weather. Goal: predict microbial community state from
soil moisture dynamics using the Anderson localization framework.

| # | Component | Data Source | Compute | Status |
|---|-----------|-------------|---------|--------|
| 19 | NCBI 16S agricultural soil metagenomes (ESearch + EFetch) | NCBI SRA (~105K entries for soil 16S) | I/O bound, NestGate pipeline | **Provider validated** — 23/23 checks |
| 20 | Open-Meteo ERA5 for study site weather reconstruction | Open-Meteo archive (free, no key) | CPU: `compute_et0_batch` | **Provider validated** — 115 CSVs, 80yr |
| 21 | θ(t) → Anderson d_eff coupling with 16S diversity overlay | airSpring `eco::anderson` + wetSpring 16S pipeline | GPU: Richards PDE + Anderson | **Exp 045 validated** — 55+95 checks |
| 22 | Shannon H′ from 16S → W(disorder) calibration | NCBI 16S + groundSpring Exp023 diversity loss | CPU: diversity indices | **Framework ready** — groundSpring Exp023 |

**Pipeline**: NCBI ESearch (soil 16S studies) → EFetch (FASTQ/metadata) →
wetSpring 16S pipeline (OTU table, Shannon H′) → airSpring Anderson coupling
(H′ → W, θ(t) → d_eff(t)) → QS regime prediction → compare tilled vs no-till.

**Data budget**: ~50 GB (16S amplicon data for 10-20 studies, ~2-5 GB each).
**Compute budget**: GPU Richards PDE + CPU 16S pipeline, ~2 hrs on Eastgate.
**Storage**: NestGate with BLAKE3 provenance on Eastgate (2TB NVMe sufficient).

### Tier 4 — Longer horizon

| # | Paper / Direction | Year | Faculty | Open Data? | Control Status | GPU Path |
|---|-------------------|------|---------|:----------:|:--------------:|----------|
| 23 | Dolson — Evolutionary optimization of sensor placement | — | Dolson | N/A | Future | `NelderMeadGpu` |
| 24 | Waters — Soil microbiome ↔ plant water dynamics | — | Waters | N/A | Future | N/A |

---

## Open Data Strategy

### Fully Open (no key, no account)

| Source | Data | Coverage |
|--------|------|----------|
| **Open-Meteo** | Historical weather (ERA5 reanalysis) | 80+ years, global, 10km resolution |
| **FAO-56** | Published equations and examples | Complete reference (open literature) |
| **USDA Web Soil Survey** | Soil properties (texture, Ksat, FC, WP) | US coverage |

### Open with Free Key

| Source | Data | Coverage |
|--------|------|----------|
| **NOAA CDO** | GHCND daily records | Global stations |
| **OpenWeatherMap** | Current + 5-day forecast | Global |

### Awaiting Access

| Source | Data | Status |
|--------|------|--------|
| **Dong lab field data** | Multi-sensor IoT, lysimeter | New lab 2026 |

---

## Notes

- Queue items 6-7 (Tier 1) depend on access to Dong lab's real field data (new lab 2026)
- Queue items 10-11 (Tier 2) are cross-spring references — already validated in their respective Springs
- Queue items 12-16 (Tier 3) support baseCamp Sub-thesis 06 (no-till Anderson QS)
- Queue items 19-22 (Tier 3.5) extend baseCamp 06 with NCBI 16S data — NestGate providers validated
- Queue items 23-24 (Tier 4) are longer-horizon explorations (evolutionary optimization, microbiome)
- All 41 paper reproductions use **open data** — zero institutional access, zero proprietary sensors
- Every completed paper has been validated through the full pipeline: Python → Rust CPU → GPU/NPU
- Three compute tiers verified: 40+ control dirs, 67 Rust binaries, 17 Tier A + 7 Tier B GPU modules, 3 NPU experiments
- CPU parity chain: Paper → Python control → Barracuda CPU → Barracuda GPU → Pure GPU workload → metalForge cross-system
- ET₀ method coverage: PM (FAO-56), Priestley-Taylor, Hargreaves, Thornthwaite, Makkink, Turc, Hamon, Blaney-Criddle — 8 independent methods
- GPU math portability: all 13 GPU orchestrator modules validated (Exp 047, 46/46 checks)
