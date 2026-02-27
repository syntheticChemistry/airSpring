# airSpring — Ecological & Agricultural Sciences

**Sovereign compute for precision agriculture, irrigation science, and environmental systems.**
**Date**: February 27, 2026
**Version**: 0.5.1
**License**: AGPL-3.0-or-later

airSpring is the ecological sciences validation study in the [ecoPrimals](https://github.com/ecoPrimals) ecosystem. Where **hotSpring** validates nuclear physics (clean math, f64) and **wetSpring** validates *points in a system* (microbiome, mass spectra, PFAS), airSpring validates *systems themselves* — agricultural fields, soil-plant-atmosphere continua, irrigation networks, and land-water-energy interactions.

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCuda CPU)
     → GPU (ToadStool shaders, Titan V live) → metalForge (mixed hardware)
     → biomeOS (NUCLEUS atomics, deployment graphs) → Penny Irrigation
```

## Current Status (v0.5.1)

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **1,109/1,109 PASS** | 45 papers: FAO-56, soil, IoT, WB, dual Kc, Richards, biochar, yield, CW2D, 7 ET₀ methods, GDD, pedotransfer, ensemble, bias correction, parity, dispatch, Anderson coupling |
| Phase 0+: Real data pipeline | **15,300 station-days** | ET₀ R²=0.97 vs Open-Meteo (100 Michigan stations) |
| Phase 1: Rust validation | **651 tests** | 50 barracuda + 4 forge = 54 binaries |
| Phase 1.5: CPU Benchmark | **25.9× faster** | Rust vs Python geometric mean (6×–190× range, 8/8 parity) |
| Phase 2: Cross-validation | **75/75 MATCH** | Python↔Rust identical (tol=1e-5), Richards + isotherm included |
| Phase 3: GPU live dispatch | **Titan V validated** | 24/24 PASS, 0.04% seasonal parity, 10K batch scaling |
| Phase 3.5: NPU edge | **AKD1000 live** | 3 experiments, 95/95 NPU checks, ~48µs inference |
| Phase 3.7: metalForge live | **5 substrates discovered** | RTX 4070 + Titan V + AKD1000 + i9-12900K, 14 workloads route |
| Phase 4: Penny Irrigation | Vision | Sovereign, consumer hardware |

### Code Quality

| Check | Status |
|-------|--------|
| `cargo test --lib` | **527 passed**, 0 failures |
| `cargo test --tests` | **20 passed** (integration) |
| `cargo clippy (pedantic)` | **0 warnings** (pedantic + nursery) |
| `cargo fmt --check` | **Clean** |
| `cargo doc` | **Builds** |
| Forge tests | 26 (metalForge/forge — absorption staging + NPU dispatch) |

### Hardware Validated

| Component | Specification | Status |
|-----------|--------------|--------|
| CPU | Intel i9-12900K (16C/24T, AVX2) | **Live** — all CPU paths |
| GPU #1 | NVIDIA RTX 4070 (12 GB, Vulkan, f64) | **Live** — wgpu adapter 0 |
| GPU #2 | NVIDIA TITAN V (GV100, NVK/Mesa, f64) | **Live** — 24/24 PASS, `BARRACUDA_GPU_ADAPTER=titan` |
| NPU | BrainChip AKD1000 (`/dev/akida0`) | **Live** — 95/95 NPU checks |
| RAM | 64 GB DDR5-4800 | |
| OS | Pop!_OS 22.04 (kernel 6.17.4) | |

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

ToadStool contains **774 WGSL shaders** across 41+ categories (S68: 2,541+ barracuda tests). airSpring uses 6 shared shader families, contributed **3 upstream fixes**, and had **all metalForge modules absorbed upstream** (S64 + S66):

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
| `gpu::et0` | `ops::batched_elementwise_f64` (op=0) | barracuda | **GPU-FIRST** |
| `gpu::water_balance` | `ops::batched_elementwise_f64` (op=1) | barracuda | **GPU-STEP** |
| `gpu::kriging` | `ops::kriging_f64::KrigingF64` | barracuda | **Integrated** |
| `gpu::reduce` | `ops::fused_map_reduce_f64` | barracuda | **GPU N≥1024** |
| `gpu::stream` | `ops::moving_window_stats` | barracuda | **Wired** |
| `gpu::richards` | `pde::richards::solve_richards` | barracuda::pde | **Wired** (+ CN f64 cross-val) |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start` | barracuda::optimize | **Wired** |
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` + `norm_ppf` | barracuda | **Wired** (parametric CI) |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | barracuda::linalg | **Wired** |
| `eco::richards::inverse_van_genuchten_h` | `optimize::brent` | barracuda::optimize | **Wired** |
| `eco::diversity` | `stats::diversity` | barracuda::stats | **Leaning** |

Also wired: `validation::ValidationHarness` (neuralSpring), `stats::pearson`, `spearman`, `bootstrap_ci`, `stats::metrics` re-exports (airSpring→upstream S64).

Evolution gaps: 23 total (11 Tier A integrated, 11 Tier B ready, 1 Tier C pending). ToadStool S68 synced.
See `barracuda/src/gpu/evolution_gaps.rs` for the full roadmap.

### CPU Benchmarks: Rust vs Python (25.9× geometric mean speedup)

| Algorithm | N | Rust (s) | Python (s) | Speedup | Parity |
|-----------|---:|---:|---:|---:|:---:|
| FAO-56 PM ET₀ | 10K | 0.0008 | 0.012 | **15×** | ✓ |
| Hargreaves-Samani | 10K | 0.00001 | 0.001 | **114×** | ✓ |
| Water Balance Step | 10K | 0.00001 | 0.001 | **190×** | ✓ |
| Anderson Coupling | 100K | 0.0002 | 0.023 | **94×** | ✓ |
| Season Sim (153d) | 1K | 0.001 | 0.056 | **44×** | ✓ |
| Shannon Diversity | 10K | 0.0002 | 0.005 | **26×** | ✓ |
| Van Genuchten θ(h) | 100K | 0.002 | 0.015 | **6×** | ✓ |
| Thornthwaite PET | 10K | 0.084 | 0.081 | **1×** | ✓ |

Same algorithms, same f64 precision, same inputs, same outputs. 8/8 parity,
25.9× geometric mean speedup. Thornthwaite 1× is expected — Rust computes
daylight hours for every day (365 trig calls) while Python uses mid-month
approximation (12 calls). Run `cargo run --release --bin bench_cpu_vs_python`.

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
├── control/                     # Phase 0: Python baselines (1109/1109, 38 scripts)
│   ├── fao56/ ... anderson_coupling/  # 38 paper controls
│   ├── makkink/                 # Makkink (1957) radiation ET₀ (21/21)
│   ├── turc/                    # Turc (1961) temp-radiation ET₀ (22/22)
│   ├── hamon/                   # Hamon (1961) temp-based PET (20/20)
│   ├── neural_api/              # biomeOS Neural API parity (14/14)
│   ├── et0_ensemble/            # ET₀ 6-method ensemble (9/9)
│   ├── pedotransfer_richards/   # Pedotransfer → Richards coupling (29/29)
│   ├── et0_bias_correction/     # Cross-method bias correction (24/24)
│   ├── cpu_gpu_parity/          # CPU↔GPU bit-identical proof (22/22)
│   ├── metalforge_dispatch/     # Mixed-hardware dispatch routing (14/14)
│   ├── seasonal_batch_et0/      # 365×4 station-day batch (18/18)
│   └── requirements.txt
├── barracuda/                   # Phase 1+3: Rust validation + GPU dispatch (527 lib + 20 integration, 50 binaries)
│   ├── src/
│   │   ├── eco/                 # Domain modules (15 validated, 7 ET₀ methods + Anderson coupling)
│   │   ├── gpu/                 # ToadStool/BarraCuda GPU bridge (11 Tier A modules + device_info)
│   │   ├── npu.rs               # BrainChip AKD1000 NPU (feature-gated)
│   │   └── bin/                 # validate_*, bench_*, cross_validate (50 total)
│   ├── tests/                   # Integration tests (7+ files + common/)
│   └── Cargo.toml               # v0.5.1
├── metalForge/                  # Mixed hardware dispatch (CPU+GPU+NPU)
│   └── forge/                   # airspring-forge (26 tests, 4 binaries, live hardware probe)
├── specs/                       # Specifications and requirements
│   ├── PAPER_REVIEW_QUEUE.md    # Paper reproduction queue (45 complete)
│   ├── BARRACUDA_REQUIREMENTS.md# GPU + NPU kernel requirements
│   └── CROSS_SPRING_EVOLUTION.md# Cross-spring shader provenance
├── whitePaper/                  # Methodology and study documentation
│   └── baseCamp/                # Per-faculty research briefings + baseCamp extensions
├── experiments/                 # Experiment protocols and results (45 experiments)
├── wateringHole/                # Spring-local handoffs to ToadStool/BarraCuda
│   └── handoffs/                # Versioned (V030 evolution handoff active)
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
| `CONTROL_EXPERIMENT_STATUS.md` | Detailed experiment results (45 experiments) |
| `barracuda/EVOLUTION_READINESS.md` | Tier A/B/C GPU evolution, absorbed/stays-local |
| `metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66) |
| `metalForge/forge/` | Mixed hardware dispatch: live probe + capability routing |
| `specs/CROSS_SPRING_EVOLUTION.md` | Cross-spring shader provenance |
| `specs/PAPER_REVIEW_QUEUE.md` | Paper reproduction queue (45 complete) |
| `whitePaper/baseCamp/README.md` | Faculty research briefings + baseCamp extensions |
| `wateringHole/handoffs/` | ToadStool/BarraCuda handoffs (V030 evolution handoff active) |

## License

AGPL-3.0-or-later

---

*February 27, 2026 — v0.5.1. 45 experiments, 1109/1109 Python, 651 Rust tests,
1393 atlas checks, 54 binaries, 75/75 cross-validation, 15,300 station-days.
Rust 25.9× faster than Python (8/8 parity, geometric mean). 11 Tier A GPU modules.
Titan V GPU live dispatch (24/24 PASS, 0.04% seasonal parity).
AKD1000 NPU live (3 experiments, ~48µs inference).
metalForge live hardware: RTX 4070 + Titan V + AKD1000 + i9-12900K (5 substrates, 14 workloads).
Anderson soil-moisture coupling (θ→S_e→d_eff→QS regime, 55+95 PASS).
ToadStool S68 synced (774 WGSL). Pure Rust + BarraCuda. AGPL-3.0-or-later.*
