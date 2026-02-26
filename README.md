# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**
**Date**: February 26, 2026
**Version**: 0.4.5
**License**: AGPL-3.0-or-later

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **hotSpring** validates nuclear physics (clean math, f64) and **wetSpring** validates *points in a system* (microbiome, mass spectra, PFAS), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCuda CPU)
     → GPU (ToadStool shaders) → metalForge (mixed hardware) → Penny Irrigation
```

## Current Status (v0.4.5)

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **474/474 PASS** | FAO-56, soil, IoT, water balance, dual Kc, cover crops, Richards, biochar, yield, CW2D, 60yr WB, scheduling, lysimeter, sensitivity |
| Phase 0+: Real data pipeline | **918 station-days** | ET₀ R²=0.967 vs Open-Meteo (6 Michigan stations) |
| Phase 1: Rust validation | **725 tests** | 21 binaries, 464 unit + 132 integration + 53 forge + 76 new binary checks |
| Phase 1.5: CPU Benchmark | **69x faster** | Rust vs Python geometric mean (20x–502x range) |
| Phase 2: Cross-validation | **75/75 MATCH** | Python↔Rust identical (tol=1e-5), Richards + isotherm included |
| Phase 3: GPU bridge | **11 Tier A modules** | S66 synced — all metalForge absorbed upstream, evolution\_gaps current |
| Phase 4: Penny Irrigation | Vision | Sovereign, consumer hardware |

### Code Quality

| Check | Status |
|-------|--------|
| `cargo test` | 464 barracuda + 53 forge + 132 integration = **649 lib/integration**, 0 failures |
| `cargo clippy -- -D warnings` | **0 warnings** (pedantic) |
| `cargo fmt --check` | **Clean** |
| `cargo doc` | **Builds** |
| `cargo llvm-cov --lib` | **96.81%** line coverage |
| Test breakdown | 464 unit, 33 eco, 21 GPU, 6 evolution, 4 determinism, 29 cross-spring, 20 stats, 11 I/O, 2 doc |

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
  ToadStool WGSL shaders (f64 precision on GPU, 774 shaders)
       │
       ▼
  metalForge (mixed CPU + GPU + future NPU)
```

airSpring domain code (`eco::`) is validated against papers, then wrapped by GPU orchestrators (`gpu::`) that bridge to `barracuda` primitives. The primitives dispatch to ToadStool WGSL shaders for GPU or fall back to CPU. `metalForge` stages upstream absorption candidates following the "Write → Absorb → Lean" cycle.

### Cross-Spring Shader Evolution

ToadStool contains **774 WGSL shaders** across 41+ categories (S66: 2,541+ barracuda tests). airSpring uses 6 shared shader families, contributed **3 upstream fixes**, and had **all metalForge modules absorbed upstream** (S64 + S66):

| Spring | Shaders | What airSpring Gets | What airSpring Gave Back |
|--------|---------|--------------------|-----------------------|
| **hotSpring** | 56 | df64 core, pow/exp/log/trig f64, df64\_transcendentals — VG, atmospheric | TS-001: `pow_f64` fractional exponent fix |
| **wetSpring** | 25 | kriging\_f64, fused\_map\_reduce, moving\_window, ridge, **diversity metrics** | TS-004: reduce buffer N≥1024 fix |
| **neuralSpring** | 20 | nelder\_mead, multi\_start, ValidationHarness | TS-003: acos precision boundary fix |
| **groundSpring** | — | **MC ET₀ uncertainty propagation shader** | — |
| **airSpring** | — | Domain consumer + stats absorbed upstream | Richards PDE (S40), stats metrics (S64) |

46+ cross-spring absorptions (S51-S66). All metalForge absorbed. See `specs/CROSS_SPRING_EVOLUTION.md`.

### BarraCuda Integration (11 Tier A modules)

| airSpring Module | BarraCuda Primitive | Origin | Status |
|-----------------|--------------------|----|---|
| `gpu::et0` | `ops::batched_elementwise_f64` (op=0) | Multi-spring | **GPU-FIRST** |
| `gpu::water_balance` | `ops::batched_elementwise_f64` (op=1) | Multi-spring | **GPU-STEP** |
| `gpu::kriging` | `ops::kriging_f64::KrigingF64` | wetSpring | **Integrated** |
| `gpu::reduce` | `ops::fused_map_reduce_f64` | wetSpring | **GPU N≥1024** |
| `gpu::stream` | `ops::moving_window_stats` | wetSpring S28+ | **Wired** |
| `gpu::richards` | `pde::richards::solve_richards` | airSpring→ToadStool S40 | **Wired** (+ CN f64 cross-val) |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start` | neuralSpring | **Wired** |
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` + `norm_ppf` | groundSpring S64 + hotSpring | **Wired** (parametric CI) |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN | **Wired** |
| `eco::richards::inverse_van_genuchten_h` | `optimize::brent` | neuralSpring | **Wired** |
| `eco::diversity` | `stats::diversity` | wetSpring S64 | **Leaning** |

Also wired: `validation::ValidationHarness` (neuralSpring), `stats::pearson`, `spearman`, `bootstrap_ci`, `stats::metrics` re-exports (airSpring→upstream S64).

Evolution gaps: 23 total (11 Tier A integrated, 11 Tier B ready, 1 Tier C pending). ToadStool S66 synced.
See `barracuda/src/gpu/evolution_gaps.rs` for the full roadmap.

### CPU Benchmarks: Rust vs Python (69x geometric mean speedup)

| Computation | Python | Rust CPU | Speedup |
|---|---:|---:|---:|
| FAO-56 ET₀ (10K) | 632K/s | 12.7M/s | **20x** |
| VG θ(h) retention (100K) | 434K/s | 35.8M/s | **83x** |
| Yield single-stage (100K) | 13.4M/s | 1.08B/s | **81x** |
| Yield multi-stage (100K) | 4.1M/s | 378M/s | **93x** |
| Richards 1D (20 nodes) | 23/s | 3,683/s | **159x** |
| Richards 1D (50 nodes) | 7/s | 3,620/s | **502x** |
| Dual Kc (3650-day) | — | 59M days/s | — |
| CW2D VG gravel (100K) | 444K/s | 35.5M/s | **80x** |

Same algorithms, same f64 precision. Python loops vs Rust `--release`. Richards
PDE sees the largest gains (up to 502x) because scipy.integrate overhead dwarfs
Rust's hand-coded implicit Euler + Thomas algorithm. See `experiments/README.md`
for full results and `scripts/bench_compare.py` to reproduce.

## Quick Start

```bash
# Python baselines (Phase 0)
pip install -r control/requirements.txt
python scripts/download_open_meteo.py --all-stations --growing-season 2023
bash scripts/run_all_baselines.sh

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
├── control/                     # Phase 0: Python baselines (474/474)
│   ├── fao56/                   # FAO-56 Penman-Monteith ET₀ (64/64)
│   ├── soil_sensors/            # Soil moisture calibration (36/36)
│   ├── iot_irrigation/          # IoT irrigation pipeline (24/24)
│   ├── water_balance/           # FAO-56 soil water balance (18/18)
│   ├── dual_kc/                 # Dual Kc + cover crops + no-till (63+40)
│   ├── regional_et0/            # Regional ET₀ intercomparison (61/61)
│   ├── richards/                # 1D Richards equation (14/14)
│   ├── biochar/                 # Biochar adsorption isotherms (14/14)
│   ├── yield_response/          # Yield response to water stress (32/32)
│   ├── cw2d/                    # CW2D Richards extension (24/24)
│   ├── long_term_wb/            # 60-year water balance (10/10)
│   ├── scheduling/              # Irrigation scheduling optimization (25/25)
│   ├── lysimeter/               # Lysimeter ET measurement (26/26)
│   ├── sensitivity/             # ET₀ sensitivity analysis (23/23)
│   └── requirements.txt
├── barracuda/                   # Phase 1: Rust validation (464 lib + 132 integration, 21 binaries)
│   ├── src/
│   │   ├── eco/                 # Domain modules (12 validated against papers, incl. diversity)
│   │   ├── io/                  # csv_ts (streaming columnar IoT parser)
│   │   ├── gpu/                 # ToadStool/BarraCuda GPU bridge (11 Tier A modules)
│   │   ├── error.rs             # AirSpringError enum
│   │   ├── validation.rs        # Re-exports barracuda::validation::ValidationHarness
│   │   ├── testutil/            # IA, NSE, RMSE, MBE, R², Pearson r, bootstrap CI
│   │   │   ├── generators.rs
│   │   │   ├── stats.rs
│   │   │   └── bootstrap.rs
│   │   └── bin/                 # 21 validate_*, bench_*, cross_validate, simulate_season
│   ├── tests/                   # 132 integration tests (7 files + common/)
│   │   ├── common/              # Shared GPU device helpers
│   │   ├── eco_integration.rs   # Eco module cross-validation
│   │   ├── gpu_integration.rs   # GPU orchestrator functional tests
│   │   ├── gpu_evolution.rs     # Evolution gap / ToadStool issue tracking
│   │   ├── gpu_determinism.rs   # Bit-identical rerun validation
│   │   ├── io_and_errors.rs     # CSV parsing, error variants
│   │   └── stats_integration.rs # Statistical metrics cross-validation
│   └── Cargo.toml               # v0.4.5
├── metalForge/                  # Upstream absorption staging (→ barracuda)
│   └── forge/                   # airspring-forge v0.2.0 (53 tests, 6 modules)
├── specs/                       # Specifications and requirements
│   ├── PAPER_REVIEW_QUEUE.md    # Paper reproduction queue (16 complete, queued)
│   ├── BARRACUDA_REQUIREMENTS.md# GPU kernel requirements
│   └── CROSS_SPRING_EVOLUTION.md# Cross-spring shader provenance
├── whitePaper/                  # Methodology and study documentation
│   ├── baseCamp/                # Per-faculty research briefings
│   ├── METHODOLOGY.md           # Multi-phase validation protocol
│   └── STUDY.md                 # Full results narrative
├── experiments/                 # Experiment protocols and results (16 complete)
├── wateringHole/                # Spring-local handoffs to ToadStool/BarraCuda
│   └── handoffs/                # Versioned (V016 active)
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

## Hardware Gate

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (16C/24T, 5.2 GHz) |
| RAM | 64 GB DDR5-4800 |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| Storage | 1 TB NVMe SSD |
| OS | Pop!_OS 22.04 (Ubuntu-based) |

## Document Index

| Document | Purpose |
|----------|---------|
| `CHANGELOG.md` | Versioned change history |
| `CONTROL_EXPERIMENT_STATUS.md` | Detailed experiment results |
| `barracuda/EVOLUTION_READINESS.md` | Tier A/B/C GPU evolution, absorbed/stays-local |
| `metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66) |
| `specs/CROSS_SPRING_EVOLUTION.md` | Cross-spring shader provenance |
| `specs/BARRACUDA_REQUIREMENTS.md` | GPU kernel requirements |
| `specs/PAPER_REVIEW_QUEUE.md` | Paper reproduction queue |
| `whitePaper/STUDY.md` | Full results narrative |
| `whitePaper/METHODOLOGY.md` | Validation protocol |
| `whitePaper/baseCamp/README.md` | Faculty research briefings |
| `wateringHole/handoffs/` | ToadStool/BarraCuda handoffs (V016 active) |

## License

AGPL-3.0-or-later

---

*February 26, 2026 — v0.4.5. 16 experiments, 474/474 Python, 725 Rust checks,
21 binaries, 75/75 cross-validation, 918 real station-days. Rust 69x faster
than Python (geometric mean). 11 Tier A wired modules. ToadStool S66 synced
(774 WGSL, all metalForge absorbed). Pure Rust + BarraCuda. AGPL-3.0-or-later.*
