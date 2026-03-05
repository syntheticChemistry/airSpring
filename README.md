# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**
**Date**: March 5, 2026
**Version**: 0.7.0
**License**: AGPL-3.0-or-later

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **hotSpring** validates nuclear physics (clean math, f64) and **wetSpring** validates *points in a system* (microbiome, mass spectra, PFAS), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCuda CPU)
     → GPU (ToadStool shaders, Titan V live) → metalForge (mixed hardware)
     → biomeOS (NUCLEUS atomics, deployment graphs) → Penny Irrigation
```

## Current Status (v0.7.0)

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **1,237/1,237 PASS** | 57 papers: FAO-56, soil, IoT, WB, dual Kc, Richards, biochar, yield, CW2D, 8 ET₀ methods, GDD, pedotransfer, ensemble, bias correction, parity, dispatch, Anderson coupling, SCS-CN + Green-Ampt (coupled), VG inverse, full-season WB |
| Phase 0+: Real data pipeline | **15,300 station-days** | ET₀ R²=0.97 vs Open-Meteo (100 Michigan stations) |
| Phase 1: Rust validation | **827 lib + 1498 atlas** | 86 binaries + 146/146 + 32/32 provenance cross-spring benchmarks (25 GPU fail: upstream wgpu 28 driver issue) |
| Phase 1.5: CPU Benchmark | **13,000× atlas-scale** | Rust vs Python: 10M ET₀/s, 6.8M field-days/s (34/34 parity) |
| Phase 2: Cross-validation | **75/75 MATCH** | Python↔Rust identical (tol=1e-5), Richards + isotherm included |
| Phase 2.5: Tier B→A GPU | **4 ops GPU-first** | Hargreaves (op=6), Kc climate (op=7), dual Kc (op=8), sensor cal (op=5) — ToadStool S70+ absorbed |
| Phase 2.6: Seasonal pipeline | **GPU Stages 1-3** | ET₀ + Kc + WB GPU dispatch, multi-field `gpu_step()`, streaming |
| Phase 2.7: GPU streaming multi-field | **57/57 PASS** | M fields × N days, Stage 3 GPU per-day, 6.8M field-days/s (Exp 070) |
| Phase 3: GPU live dispatch | **78/78 PASS** | Pure GPU workload validation (Exp 055), 0.04% seasonal parity |
| Phase 3.1: Pure GPU end-to-end | **46/46 PASS** | All 4 stages on GPU, 19.7× dispatch reduction (Exp 072) |
| Phase 3.2: Cross-spring rewire | **68/68 PASS** | `BrentGpu` VG inverse, `RichardsGpu` Picard, full provenance (Exp 073) |
| Phase 3.3: Paper chain validation | **79/79 PASS** | Full CPU→GPU→metalForge chain for 28 domains (22 GPU, 6 CPU-only) (Exp 074) |
| Phase 3.4: Local GPU compute | **6 ops f64 canonical** | SCS-CN, Stewart, Makkink, Turc, Hamon, Blaney-Criddle via `local_elementwise_f64.wgsl` via compile_shader_universal (Exp 075+078) |
| Phase 3.5: NPU edge | **AKD1000 live** | 3 experiments, 95/95 NPU checks, ~48µs inference |
| Phase 3.7: metalForge live | **5 substrates discovered** | RTX 4070 + Titan V + AKD1000 + i9-12900K, 27 workloads route |
| Phase 3.8: Mixed-hardware pipeline | **66/66 PASS** | 7-stage GPU→NPU PCIe bypass, NUCLEUS mesh routing (Exp 076: 60/60) |
| Phase 3.9: NUCLEUS primal | **30 capabilities** | airSpring biomeOS primal, 30 science capabilities, JSON-RPC |
| Phase 4.0: Cross-primal pipeline | **28/28 PASS** | ecology domain, capability.call routing, cross-primal forwarding |
| Phase 4.1: Full dispatch experiment | **51/51 PASS** | CPU vs GPU parity across all domains (Exp 064) |
| Phase 4.2: biomeOS graph experiment | **35/35 PASS** | Offline ecology pipeline, deployment graph validated (Exp 065) |
| Phase 4.3: Paper 12 immunological Anderson | **4 experiments** | Tissue diversity, CytokineBrain, barrier state, cross-species (Exp 066-069) |
| Phase 4.4: Penny Irrigation | Vision | Sovereign, consumer hardware |

### Code Quality

| Check | Status |
|-------|--------|
| `cargo test --lib` (barracuda) | **852 passed**, 0 failures |
| `cargo test --tests` (integration) | **33 passed** (13 GPU pipeline + 20 stats) |
| `cargo test --lib` (metalForge) | **62 passed**, 0 failures |
| `cargo llvm-cov --lib --fail-under-lines 90` | **95.66% line coverage** |
| `cargo clippy (pedantic)` | **0 warnings** (pedantic, both crates) |
| `cargo fmt --check` | **Clean** |
| `cargo doc --no-deps` | **Clean** (both crates) |
| `cargo-deny check` | **Clean** (AGPL-3.0-or-later) |
| `bench_cross_spring_evolution` | **146/146 PASS** (release, S87 sync) |
| `validate_cross_spring_provenance` | **32/32 PASS** — CPU↔GPU benchmark, 5-spring shader provenance |
| `validate_dispatch_experiment` | **51/51 PASS** — CPU/GPU/batch/absorption/pipeline |
| `validate_biome_graph` | **35/35 PASS** — graph topology, capabilities, offline pipeline |

### Hardware Validated

| Component | Specification | Status |
|-----------|--------------|--------|
| CPU | Intel i9-12900K (16C/24T, AVX2) | **Live** — all CPU paths |
| GPU #1 | NVIDIA RTX 4070 (12 GB, Vulkan, f64) | **Live** — wgpu adapter 0 |
| GPU #2 | NVIDIA TITAN V (GV100, NVK/Mesa, f64) | **Live** — 24/24 PASS, `BARRACUDA_GPU_ADAPTER=titan` |
| NPU | BrainChip AKD1000 (`/dev/akida0`) | **Live** — 95/95 NPU checks |
| RAM | 64 GB DDR5-4800 | |
| OS | Pop!_OS 22.04 (kernel 6.17.9) | |

## Evolution Architecture: Write → Absorb → Lean

```
  airSpring eco:: (CPU, validated against FAO-56 papers)
       │
       ▼
  airSpring gpu:: wrappers (domain-specific batched API)
       │
       ▼
  barracuda::ops/linalg/stats/pde/optimize (GPU dispatch + CPU fallback)
       │
       ▼
  BarraCuda WGSL shaders (f64 canonical, precision per silicon — 767 shaders)
       │
       ▼
  metalForge (mixed CPU + GPU + NPU)
       │
       ▼
  bingoCube/nautilus (evolutionary reservoir, drift detection, NPU export)
```

airSpring domain code (`eco::`) is validated against papers, then wrapped by GPU orchestrators (`gpu::`) that bridge to `barracuda` primitives. The primitives dispatch to ToadStool WGSL shaders for GPU or fall back to CPU. `metalForge` stages upstream absorption candidates following the "Write → Absorb → Lean" cycle. `bingoCube/nautilus` provides evolutionary reservoir computing for time-series prediction (ET₀ forecasting, drift detection) with AKD1000 NPU export.

### Cross-Spring Shader Evolution

BarraCuda contains **767 WGSL shaders** (S93: 5,369 tests, pure math + precision per silicon). airSpring uses 6 shared shader families, contributed **3 upstream fixes**, and had **all metalForge modules absorbed upstream** (S64 + S66):

| Spring | Shaders | What airSpring Gets | What airSpring Gave Back |
|--------|---------|--------------------|-----------------------|
| **hotSpring** | 56 | df64 core, pow/exp/log/trig f64, df64\_transcendentals — VG, atmospheric | TS-001: `pow_f64` fractional exponent fix |
| **wetSpring** | 25 | kriging\_f64, fused\_map\_reduce, moving\_window, ridge, **diversity metrics** | TS-004: reduce buffer N≥1024 fix |
| **neuralSpring** | 20 | nelder\_mead, multi\_start, ValidationHarness | TS-003: acos precision boundary fix |
| **groundSpring** | — | **MC ET₀ uncertainty propagation shader** | — |
| **airSpring** | — | Domain consumer + stats absorbed upstream | Richards PDE (S40), stats metrics (S64) |

50+ cross-spring absorptions (S42-S87). All metalForge absorbed. DF64 transcendentals complete (15 functions).
S87: ops 0-13, GPU uncertainty stack, `BrentGpu`, `RichardsGpu`, `BatchedStatefulF64`, `nautilus`, L-BFGS,
`gpu_helpers` refactor, `is_device_lost()`, MatMul validation, 845 WGSL shaders (zero f32-only).
See `specs/CROSS_SPRING_EVOLUTION.md`.

### BarraCuda Integration (25 Tier A + 6 GPU-universal + 3 pipeline)

| airSpring Module | BarraCuda Primitive | Op/Shader | Status |
|-----------------|--------------------|----|---|
| `gpu::et0` | `batched_elementwise_f64` | op=0 | **GPU-FIRST** |
| `gpu::water_balance` | `batched_elementwise_f64` | op=1 | **GPU-STEP** |
| `gpu::sensor_calibration` | `batched_elementwise_f64` | op=5 | **Integrated** |
| `gpu::hargreaves` | `HargreavesBatchGpu` | dedicated | **Integrated** |
| `gpu::kc_climate` | `batched_elementwise_f64` | op=7 | **Integrated** |
| `gpu::dual_kc` | `batched_elementwise_f64` | op=8 | **Integrated** |
| `gpu::van_genuchten` | `batched_elementwise_f64` | op=9,10 | **Integrated** (S79) |
| `gpu::thornthwaite` | `batched_elementwise_f64` | op=11 | **Integrated** (S79) |
| `gpu::gdd` | `batched_elementwise_f64` | op=12 | **Integrated** (S79) |
| `gpu::pedotransfer` | `batched_elementwise_f64` | op=13 | **Integrated** (S79) |
| `gpu::kriging` | `kriging_f64::KrigingF64` | dedicated | **Integrated** |
| `gpu::reduce` | `fused_map_reduce_f64` | dedicated | **GPU N≥1024** |
| `gpu::stream` | `moving_window_stats` | dedicated | **Integrated** |
| `gpu::richards` | `pde::richards::solve_richards` | dedicated | **Integrated** |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start` | dedicated | **Integrated** |
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` | dedicated | **Integrated** |
| `gpu::jackknife` | `JackknifeMeanGpu` | dedicated | **Integrated** (S79) |
| `gpu::bootstrap` | `BootstrapMeanGpu` | dedicated | **Integrated** (S79) |
| `gpu::diversity` | `DiversityFusionGpu` | dedicated | **Integrated** (S79) |
| `gpu::stats` | `linear_regression_f64` + `matrix_correlation_f64` | dedicated | **Integrated** |
| `gpu::runoff` | `local_elementwise_f64.wgsl` (op=0) | f64 canonical | **GPU-universal** (v0.7.0) |
| `gpu::yield_response` | `local_elementwise_f64.wgsl` (op=1) | f64 canonical | **GPU-universal** (v0.7.0) |
| `gpu::simple_et0` | `local_elementwise_f64.wgsl` (ops 2-5) | f64 canonical | **GPU-universal** (v0.7.0, 3/6 absorbed upstream) |
| `gpu::local_dispatch` | `LocalElementwise` (compile_shader_universal) | 6 ops f64→f32 | **EVOLVED** (v0.7.0) |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | CPU | **Integrated** |
| `eco::richards::inverse_vg_h` | `optimize::brent` | CPU | **Integrated** |
| `eco::diversity` | `stats::diversity` | CPU | **Leaning** |
| `gpu::seasonal_pipeline` | Chains ops 0→7→1→yield | fused | **CPU + GpuPipelined** |
| `gpu::atlas_stream` | `UnidirectionalPipeline` | streaming | **CPU chained** |

Also wired: `validation::ValidationHarness` (neuralSpring), `stats::pearson`, `spearman`, `bootstrap_ci`, `stats::metrics` re-exports (airSpring→upstream S64).

25 Tier A GPU modules (ops 0-13 + jackknife/bootstrap/diversity uncertainty stack) + 6 GPU-universal (f64 canonical via `compile_shader_universal`). ToadStool S87 synced.
See `barracuda/src/gpu/evolution_gaps.rs` and `barracuda/EVOLUTION_READINESS.md` for the full roadmap.

### CPU Benchmarks: Rust vs Python (14.5× geometric mean speedup, 21/21 parity)

Run `cargo run --release --bin bench_cpu_vs_python`. 21 algorithms, same f64
precision, same inputs, same outputs. 14.5× geometric mean includes the seasonal
pipeline (ET₀→Kc→WB→Yield chain) which gives Python a relative advantage due to
NumPy vectorization. Individual algorithm speedups range from 1× (Thornthwaite —
Rust computes daylight per-day vs Python mid-month) to 190× (water balance step).

## Quick Start

```bash
# Python baselines (Phase 0)
pip install -r control/requirements.txt
python scripts/download_open_meteo.py --all-stations --growing-season 2023
bash run_all_baselines.sh

# Rust validation (Phase 1)
cd barracuda && cargo test
cargo run --release --bin validate_et0

# Benchmarks
cargo run --release --bin bench_cpu_vs_python
cargo run --release --bin bench_airspring_gpu
```

No institutional access required. Zero synthetic data in the default pipeline.

## Data Strategy

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

**The Problem**: Agricultural irrigation consumes ~70% of global freshwater withdrawals. Current precision irrigation relies on proprietary sensor systems ($500-$5000/field) and vendor-locked software.

**Computational Methods**: FAO-56 PM ET₀, Topp equation soil calibration, IoT field monitoring, FAO-56 Ch 8 water balance scheduling, FAO-56 Ch 7 dual Kc with cover crops and no-till mulch.

### Track 2: Environmental Systems & Land Treatment

Richards equation (unsaturated flow — open-source alternative to HYDRUS), biochar adsorption isotherms (Langmuir/Freundlich), long-term water balance (60-year reconstruction via ERA5).

### Key Publications

- Dong et al. (2020) "Soil moisture sensor evaluation in Michigan soils" *Agriculture* 10(12), 598
- Dong & Hansen (2023) "Affordable weighing lysimeter design" *Smart Ag Tech* 4, 100147
- Dong et al. (2024) "In-field IoT for precision irrigation" *Frontiers in Water* 6, 1353597
- Ali, Dong & Lavely (2024) "Irrigation scheduling vs yield" *Ag Water Mgmt* 306, 109148
- Dong et al. (2019) "Land-based wastewater modeling using HYDRUS CW2D" *J. SWBE* 5(4)
- Kumari, Dong & Safferman (2025) "Phosphorus adsorption using biochar" *Applied Water Sci* 15(7)

## Directory Structure

```
airSpring/
├── control/                     # Phase 0: Python baselines (1237/1237, 44 scripts)
│   ├── fao56/ ... anderson_coupling/  # 44 paper controls
│   ├── coupled_runoff_infiltration/   # Coupled SCS-CN + Green-Ampt (292/292)
│   ├── vg_inverse/              # Van Genuchten inverse fitting (84/84)
│   ├── season_water_budget/     # Full-season irrigation WB (34/34)
│   └── requirements.txt
├── barracuda/                   # Phase 1+3: Rust validation + GPU dispatch (827 lib tests, 86 binaries, barraCuda 0.3.3 / wgpu 28)
│   ├── src/
│   │   ├── biomeos.rs           # biomeOS socket resolution + primal discovery (shared)
│   │   ├── eco/                 # Domain modules (19 validated, 8 ET₀ + runoff + infiltration + VG + Anderson + tissue + cytokine)
│   │   ├── gpu/                 # ToadStool/BarraCuda GPU bridge (25 Tier A + 6 GPU-local + BrentGpu + RichardsGpu)
│   │   ├── nautilus.rs          # bingoCube/nautilus evolutionary reservoir (AirSpringBrain)
│   │   ├── rpc.rs               # JSON-RPC 2.0 inter-primal communication
│   │   ├── npu.rs               # BrainChip AKD1000 NPU (feature-gated)
│   │   └── bin/                 # validate_*, bench_*, airspring_primal (84 declared)
│   ├── tests/                   # Integration + property tests (9 files + common/)
│   └── Cargo.toml               # v0.7.0
├── metalForge/                  # Mixed hardware dispatch (CPU+GPU+NPU)
│   ├── deploy/                  # biomeOS deployment graphs (airspring_deploy.toml)
│   └── forge/                   # airspring-forge (61 tests, 5 binaries, live hardware probe)
├── specs/                       # Specifications and requirements
│   ├── PAPER_REVIEW_QUEUE.md    # Paper reproduction queue (77 experiments)
│   ├── BARRACUDA_REQUIREMENTS.md# GPU + NPU kernel requirements
│   └── CROSS_SPRING_EVOLUTION.md # Cross-spring shader provenance (S87)
├── whitePaper/                  # Methodology and study documentation
│   └── baseCamp/                # Per-faculty research briefings + baseCamp extensions
├── experiments/                 # Experiment protocols and results (77 experiments)
├── wateringHole/                # Spring-local handoffs to ToadStool/BarraCuda
│   └── handoffs/                # Versioned handoffs (V069 current)
├── graphs/                      # biomeOS deployment graphs (TOML)
├── CHANGELOG.md                 # Keep-a-Changelog versioned history
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
| GPU layer | ToadStool (wgpu) | ToadStool | **ToadStool** |
| Success metric | chi² match | Same taxonomy/PFAS | **Same ET₀/yield** |
| Ultimate goal | Sovereign nuclear | Penny water monitoring | **Penny irrigation** |

## Document Index

| Document | Purpose |
|----------|---------|
| `CHANGELOG.md` | Versioned change history |
| `CONTROL_EXPERIMENT_STATUS.md` | Detailed experiment results (77 experiments) |
| `barracuda/EVOLUTION_READINESS.md` | Tier A/B/C GPU evolution, absorbed/stays-local |
| `metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66), 27 workloads |
| `metalForge/forge/` | Mixed hardware dispatch: live probe + capability routing |
| `specs/CROSS_SPRING_EVOLUTION.md` | Cross-spring shader provenance (S87) |
| `specs/PAPER_REVIEW_QUEUE.md` | Paper reproduction queue (77 experiments) |
| `whitePaper/baseCamp/README.md` | Faculty research briefings + baseCamp extensions |
| `wateringHole/handoffs/` | ToadStool/BarraCuda handoffs (V070 current) |

## License

AGPL-3.0-or-later

---

*March 5, 2026 — v0.7.0. barraCuda 0.3.3 rewire (wgpu 28). 78 experiments, 1237/1237 Python,
827 lib + 186 forge tests (27 GPU dispatch fail — upstream wgpu 28 NVK driver issue),
381/381 validation checks (10 binaries), 146/146 cross-spring evolution benchmarks, 19.8× speedup over Python.
Fused Welford mean+variance wired into SeasonalReducer (3 GPU passes vs 4).
Fused 5-accumulator Pearson correlation wired into gpu/stats (pairwise\_correlation\_gpu, fused\_mean\_variance\_gpu).
Cross-spring shader provenance: hotSpring (precision/DF64), wetSpring (bio/diversity), neuralSpring (ML/stats),
groundSpring (uncertainty/MC). 2 new ShaderProvenance entries documenting mean\_variance\_f64.wgsl and correlation\_full\_f64.wgsl origins.
3/6 local WGSL ops absorbed upstream (Makkink→Op14, Turc→Op15, Hamon→Op16); 3 remain local.
Fp64Strategy::Concurrent documented (NVK reliability verification). GPU Welford fallback to CPU when dispatch returns zeros.
Zero unsafe, zero clippy pedantic+nursery warnings, zero mocks in production.
cargo-deny clean, all SPDX AGPL-3.0-or-later. Pure Rust + BarraCuda.*
