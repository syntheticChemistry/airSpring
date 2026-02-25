# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**
**Date**: February 25, 2026
**Version**: 0.4.2
**License**: AGPL-3.0-or-later

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **hotSpring** validates nuclear physics (clean math, f64) and **wetSpring** validates *points in a system* (microbiome, mass spectra, PFAS), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCuda CPU)
     → GPU (ToadStool shaders) → metalForge (mixed hardware) → Penny Irrigation
```

## Current Status (v0.4.2)

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **344/344 PASS** | FAO-56, soil, IoT, water balance, dual Kc, cover crops, Richards, biochar, 60yr WB |
| Phase 0+: Real data pipeline | **918 station-days** | ET₀ R²=0.967 vs Open-Meteo (6 Michigan stations) |
| Phase 1: Rust validation | **328 tests** | 16 binaries, 231 unit + 97 integration |
| Phase 2: Cross-validation | **75/75 MATCH** | Python↔Rust identical (tol=1e-5), Richards + isotherm included |
| Phase 3: GPU bridge | **8 orchestrators** | 20 evolution gaps (8A+11B+1C) |
| Phase 4: Penny Irrigation | Vision | Sovereign, consumer hardware |

### Code Quality

| Check | Status |
|-------|--------|
| `cargo test` | 328 barracuda + 53 forge = **381 total**, 0 failures |
| `cargo clippy -- -D warnings` | **0 warnings** (pedantic) |
| `cargo fmt --check` | **Clean** |
| `cargo doc` | **Builds** |
| Test breakdown | 231 unit, 33 eco-integration, 31 GPU-integration, 11 stats, 20 I/O, 2 binary |

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
  ToadStool WGSL shaders (f64 precision on GPU, 608 shaders)
       │
       ▼
  metalForge (mixed CPU + GPU + future NPU)
```

airSpring domain code (`eco::`) is validated against papers, then wrapped by GPU orchestrators (`gpu::`) that bridge to `barracuda` primitives. The primitives dispatch to ToadStool WGSL shaders for GPU or fall back to CPU. `metalForge` stages upstream absorption candidates following the "Write → Absorb → Lean" cycle.

### Cross-Spring Shader Evolution

ToadStool contains **608 WGSL shaders** across 41 categories. airSpring uses 5 shared shader families and contributed **3 upstream fixes** that benefit ALL Springs:

| Spring | Shaders | What airSpring Gets | What airSpring Gave Back |
|--------|---------|--------------------|-----------------------|
| **hotSpring** | 56 | df64 core, pow/exp/log/trig f64 — VG retention, atmospheric pressure | TS-001: `pow_f64` fractional exponent fix |
| **wetSpring** | 25 | kriging_f64, fused_map_reduce, moving_window, ridge_regression | TS-004: reduce buffer N≥1024 fix |
| **neuralSpring** | 20 | nelder_mead, multi_start_nelder_mead, ValidationHarness | TS-003: acos precision boundary fix |
| **airSpring** | — | Domain consumer | Richards PDE → absorbed upstream (S40) |

46 cross-spring absorptions (S51-S57). See `specs/CROSS_SPRING_EVOLUTION.md`.

### BarraCuda Integration (8 GPU orchestrators)

| airSpring Module | BarraCuda Primitive | Origin | Status |
|-----------------|--------------------|----|---|
| `gpu::et0` | `ops::batched_elementwise_f64` (op=0) | Multi-spring | **GPU-FIRST** |
| `gpu::water_balance` | `ops::batched_elementwise_f64` (op=1) | Multi-spring | **GPU-STEP** |
| `gpu::kriging` | `ops::kriging_f64::KrigingF64` | wetSpring | **Integrated** |
| `gpu::reduce` | `ops::fused_map_reduce_f64` | wetSpring | **GPU N≥1024** |
| `gpu::stream` | `ops::moving_window_stats` | wetSpring S28+ | **Wired** |
| `gpu::richards` | `pde::richards::solve_richards` | airSpring→ToadStool S40 | **Wired** |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start` | neuralSpring | **Wired** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | wetSpring ESN | **Wired** |

Also wired: `validation::ValidationHarness` (neuralSpring), `stats::pearson`, `spearman`, `bootstrap_ci` (shared).

Evolution gaps: 20 total (8 Tier A integrated, 11 Tier B ready, 1 Tier C pending).
See `barracuda/src/gpu/evolution_gaps.rs` for the full roadmap.

### CPU Benchmarks (cross-spring provenance)

| Operation | N | Throughput | Provenance |
|-----------|---|-----------|------------|
| ET₀ (FAO-56) | 10,000 | 12.5M ops/sec | hotSpring `pow_f64`, multi-spring elementwise |
| VG θ(h) batch | 100,000 | 38.9M evals/sec | hotSpring df64 precision |
| Dual Kc season | 3,650 | 59M days/sec | airSpring `eco::dual_kc` |
| Reduce (seasonal) | 100,000 | 395M elem/sec | wetSpring `fused_map_reduce` |
| Stream smooth | 8,760 | 31.7M elem/sec | wetSpring `moving_window` |
| Kriging (20→500) | 500 | 26 µs/solve | wetSpring `kriging_f64` |
| Ridge regression | 5,000 | R²=1.000 | wetSpring ESN ridge |
| Richards PDE | 50 nodes | 72 sims/sec | airSpring→ToadStool S40, hotSpring df64 |
| Isotherm (linearized) | 9 pts | 8.3M fits/sec | airSpring `eco::isotherm` |
| Isotherm (NM 1-start) | 9 pts | 175K fits/sec | neuralSpring `nelder_mead` |
| Isotherm (NM 8×LHS) | 9 pts | 42.5K fits/sec | neuralSpring `multi_start_nelder_mead` |

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
├── control/                     # Phase 0: Python/R baselines (344/344)
│   ├── fao56/                   # FAO-56 Penman-Monteith ET₀ (64/64)
│   ├── soil_sensors/            # Soil moisture calibration (36/36)
│   ├── iot_irrigation/          # IoT irrigation pipeline (24/24)
│   ├── water_balance/           # FAO-56 soil water balance (18/18)
│   ├── dual_kc/                 # Dual Kc + cover crops + no-till (63+40)
│   ├── regional_et0/            # Regional ET₀ intercomparison (61/61)
│   ├── richards/                # 1D Richards equation (14/14)
│   ├── biochar/                 # Biochar adsorption isotherms (14/14)
│   ├── long_term_wb/            # 60-year water balance (10/10)
│   └── requirements.txt
├── barracuda/                   # Phase 1: Rust validation (328 tests, 16 binaries)
│   ├── src/
│   │   ├── eco/                 # Domain modules (9 validated against papers)
│   │   ├── io/                  # csv_ts (streaming columnar IoT parser)
│   │   ├── gpu/                 # ToadStool/BarraCuda GPU bridge (8 orchestrators)
│   │   ├── error.rs             # AirSpringError enum
│   │   ├── validation.rs        # Re-exports barracuda::validation::ValidationHarness
│   │   ├── testutil.rs          # IA, NSE, RMSE, MBE, R², Pearson r, bootstrap CI
│   │   └── bin/                 # 16 validate_*, bench_*, cross_validate, simulate_season
│   ├── tests/                   # 97 integration tests (4 files)
│   └── Cargo.toml               # v0.4.2
├── metalForge/                  # Upstream absorption staging (→ barracuda)
│   └── forge/                   # airspring-forge v0.2.0 (53 tests, 6 modules)
├── specs/                       # Specifications and requirements
│   ├── PAPER_REVIEW_QUEUE.md    # Paper reproduction queue (11 complete, 4 queued)
│   ├── BARRACUDA_REQUIREMENTS.md# GPU kernel requirements
│   └── CROSS_SPRING_EVOLUTION.md# Cross-spring shader provenance
├── whitePaper/                  # Methodology and study documentation
│   ├── baseCamp/                # Per-faculty research briefings
│   ├── METHODOLOGY.md           # Multi-phase validation protocol
│   └── STUDY.md                 # Full results narrative
├── experiments/                 # Experiment protocols and results (11 complete)
├── wateringHole/                # Spring-local handoffs to ToadStool/BarraCuda
│   └── handoffs/                # Versioned (V005 active)
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
| `specs/CROSS_SPRING_EVOLUTION.md` | Cross-spring shader provenance |
| `specs/BARRACUDA_REQUIREMENTS.md` | GPU kernel requirements |
| `specs/PAPER_REVIEW_QUEUE.md` | Paper reproduction queue |
| `whitePaper/STUDY.md` | Full results narrative |
| `whitePaper/METHODOLOGY.md` | Validation protocol |
| `whitePaper/baseCamp/README.md` | Faculty research briefings |
| `wateringHole/handoffs/` | ToadStool/BarraCuda handoffs (V005 active) |

## License

AGPL-3.0-or-later

---

*February 25, 2026 — v0.4.2. 328 Rust tests + 53 forge = 381 total.
344 Python checks, 75/75 cross-validation, 918 real station-days.
8 GPU orchestrators wired to ToadStool primitives. CPU benchmarks:
12.5M ET₀/s, 38.9M VG θ/s, 59M Kc/s, 175K NM fits/s, 72 Richards sims/s.
Cross-spring evolution: 608 WGSL shaders, 46 absorptions, 3 airSpring fixes upstream.
Pure Rust + BarraCuda. AGPL-3.0.*
