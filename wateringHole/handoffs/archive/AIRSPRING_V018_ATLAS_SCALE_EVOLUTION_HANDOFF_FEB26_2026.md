# airSpring V018 — Atlas Scale Validation & Evolution Handoff

**Date**: 2026-02-26
**From**: airSpring v0.4.6
**To**: ToadStool / BarraCuda
**ToadStool pin**: S68 (`f0feb226`)
**airSpring**: 662 Rust tests + 1302 atlas checks, 22 binaries, 97.45% coverage, 0 clippy warnings (pedantic + nursery)

---

## Executive Summary

airSpring scaled the validated ET₀ + water balance + yield pipeline from 6 stations
to **100 Michigan stations** (Exp 018: Michigan Crop Water Atlas). This is the
first large-scale deployment of the BarraCuda `eco::` stack on real Open-Meteo ERA5
data. All 1302 validation checks pass. Python cross-validation confirms Rust↔Python
parity (690 crop-station pairs within 0.01 yield ratio, 0.133% mean ET₀ diff).

**Key signal for ToadStool/BarraCuda**: The primitives work at scale. No new GPU
kernels were needed — the existing Tier A stack (BatchedEt0, BatchedWaterBalance)
handles 100 stations × 10 crops × 80 years without modification.

---

## Part 1: What Changed Since V017

### New Binary
- **`validate_atlas`** (Exp 018): 1302/1302 checks across 100 Michigan stations
  - Per-station: mass balance < 0.01 mm, ET₀ R² > 0.85 vs Open-Meteo, ET₀ range check
  - Per-crop (10 crops × 100 stations): yield ratio in [0.3, 1.0]
  - Aggregate: at least 1 station processed, statewide mean ET₀ in [500, 1200] mm

### New Scripts
- `scripts/atlas_stations.json` — 100 Michigan station definitions (lat/lon/elev/region)
- `scripts/download_atlas_80yr.py` — resilient download with 429 backpressure respect
- `scripts/download_open_meteo.py` — extended with `--atlas`, `--year-range` flags

### Python Control
- `control/atlas/atlas_water_budget.py` — independent Python baseline for cross-validation

### Scale Results (100-station pilot, 2023 growing season)
- 15,300 station-days processed (100 stations × 153 days)
- 1,000 crop-station seasonal water budgets computed
- ET₀ range: 520 mm (Marquette, UP) to 709 mm (Blissfield, SE)
- Mean R²: 0.969 vs Open-Meteo independent ET₀
- Processing time: 141 seconds (release mode, single-threaded CPU)

---

## Part 2: How airSpring Uses BarraCuda Primitives

No change from V017 — same 11 Tier A modules. Atlas reuses existing stack:

| Module | Primitive | Atlas Role |
|--------|-----------|------------|
| `eco::evapotranspiration` | FAO-56 PM (validated Exp 001) | Per-day ET₀ for each station |
| `eco::water_balance` | FAO-56 Ch 8 (validated Exp 004) | Per-crop seasonal simulation |
| `eco::yield_response` | Stewart 1977 (validated Exp 008) | Yield reduction from water stress |
| `eco::crop` | CropType + Kc curves | 10 Michigan crops with growth stages |
| `validation::ValidationHarness` | neuralSpring origin | 1302 pass/fail checks |

### GPU Promotion Opportunity

The atlas pipeline is embarrassingly parallel:
- **Station-level**: 100 independent stations → trivial batching
- **Crop-level**: 10 crops per station per year → inner batch
- **Year-level**: 80 years per station → outer batch

Estimated GPU throughput at full scale (100 × 10 × 80):
- `BatchedEt0`: 29.2M station-days → ~2.3 seconds at 12.5M ops/sec
- `BatchedWaterBalance`: 80K season simulations → trivial
- **Total**: ~5 seconds GPU vs ~2 hours CPU for full 80-year atlas

---

## Part 3: What ToadStool Should Evolve

### Priority Requests (unchanged from V017)

1. **Tier B → Tier A promotion**: Dual Kc batch (op=8), VG θ/K batch, Hargreaves batch (op=6)
2. **Batched season pipeline**: Compose ET₀ → Kc → WB → Yield into a single GPU dispatch
   (atlas proves the composition works at scale on CPU)

### New Learnings from Atlas

1. **CSV parsing is the bottleneck** — not computation. At 100 stations, I/O dominates.
   A GPU-side data ingestion pipeline (columnar float arrays) would eliminate this.
2. **Aggregation across stations** matters — statewide mean ET₀, spatial patterns.
   `gpu::kriging` (Tier A, already wired) is ready for 100-station → 10km grid interpolation.
3. **Multi-year grouping** is year-independent — each year can be processed in parallel.
   No inter-year state dependencies in the water balance (growing season only).

### Absorption Candidates (unchanged from V017)

| Function | Module | Rationale |
|----------|--------|-----------|
| `psychrometric_constant` | eco::evapotranspiration | Broadly useful atmospheric calc |
| `atmospheric_pressure` | eco::evapotranspiration | Altitude-based |
| `saturation_vapour_pressure` | eco::evapotranspiration | Tetens formula |
| `topp_equation` | eco::soil_moisture | Universal sensor calibration |
| `total_available_water` | eco::water_balance | FAO-56 standard |
| `yield_ratio_single` | eco::yield_response | Stewart 1977 |

---

## Part 4: Quality Gates

| Gate | Value |
|------|-------|
| `cargo test` | 662 PASS (464 lib + 134 integration + 64 forge) |
| `validate_atlas` | 1302/1302 PASS (100 stations × 13 checks) |
| All 22 binaries | PASS |
| `cargo clippy -- -D warnings` | 0 warnings (pedantic + nursery) |
| `cargo fmt --check` | Clean |
| `cargo llvm-cov --lib` | 97.45% |
| Python cross-validation | 690 crop-station yield pairs within 0.01 |
| P0 blockers | None |

---

## Part 5: metalForge Status

6/6 modules absorbed upstream (S64+S66). The `metalForge/forge/` crate is a fossil
record — kept for provenance but not in the dependency graph. No new metalForge
items from the atlas work.

---

*airSpring v0.4.6 → ToadStool S68. 17 experiments, 662 Rust tests + 1302 atlas
checks, 22 binaries, 15,300 station-days (100 Michigan stations). Pure Rust +
BarraCuda. AGPL-3.0-or-later.*
