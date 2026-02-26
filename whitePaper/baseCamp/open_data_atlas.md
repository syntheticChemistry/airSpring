# Michigan Crop Water Atlas (80yr, 100+ Stations)

**Date:** February 26, 2026
**Status:** Operational
**Data Tier:** 1 (free, tools exist)
**Dependencies:** Existing validated models only — no new code required
**Cross-Spring:** None (airSpring standalone)

**Operational:** 100 stations, 1354/1354 checks, 15,300 station-days (`validate_atlas` binary).

---

## What This Is

Run the validated FAO-56 ET₀ + water balance + dual Kc + yield response pipeline
across 10 crops, 100+ Michigan stations, and 80 years of Open-Meteo ERA5 data.
Produce a comprehensive, open-access Michigan crop water budget dataset.

This is not new science — it is the validated stack applied at scale. The value
is the dataset itself: no equivalent open resource exists for Michigan precision
agriculture.

## Why It Matters

Current crop water budgets for Michigan come from:
- USDA NRCS irrigation guides (coarse, state-level, infrequently updated)
- University extension publications (qualitative, crop-specific)
- Proprietary vendor platforms (locked, expensive, opaque methodology)

A sovereign, open, 80-year atlas with per-station, per-crop, daily resolution
enables any farmer, researcher, or extension agent to query historical water
demand for their specific location and crop.

## Data Requirements

| Source | Volume | Cost | Status |
|--------|--------|------|--------|
| Open-Meteo ERA5 (100 stations, 80yr) | ~600MB | Free | `download_atlas_80yr.py` + `atlas_stations.json` — 15,300 station-days |
| FAO-56 Kc table (40+ crops) | ~5KB | Free | Digitize from FAO-56 Tables 12, 17 |
| USDA soil properties (per station) | ~100KB | Free | USDA Web Soil Survey API |

Total new data: < 1GB. No API keys required.

## Compute Requirements

| Step | Scale | Eastgate CPU | Eastgate GPU |
|------|-------|-------------|-------------|
| ET₀ (100 stations x 80yr x 365d) | 2.9M | 0.2 sec | 0.01 sec |
| Water balance (10 crops x 100 stations x 80yr) | 29M | 30 sec | 1 sec |
| Yield response (10 crops x 100 stations x 80yr) | 29M | 30 sec | 1 sec |
| Kriging interpolation (100 stations per day) | 29K timesteps | ~8 hrs | ~20 min |
| **Total (without kriging)** | | **~1 min** | **~3 sec** |

## Output

1. **Per-station daily ET₀** (100 stations, 80 years) — CSV + JSON
2. **Per-crop water balance** (10 crops x 100 stations x 80 years) — daily Dr, Ks, ETa
3. **Yield response curves** (Ky x seasonal water stress) — per crop, per station
4. **Spatial ET₀ maps** (kriging interpolation) — Michigan coverage at 10km
5. **Climate trend analysis** — decadal ET₀ trends, growing season length shifts

## Implementation Plan

### Phase 1: Station expansion (Tier 1) — **Complete**
1. ~~Select 100 Michigan stations spanning agricultural regions~~ ✓
2. ~~Download 80yr Open-Meteo data using existing `download_open_meteo.py`~~ ✓
3. ~~Run `validate_atlas` across all stations (1354/1354 checks)~~ ✓
4. Store results with NestGate provenance (when available) or local JSON

**Delivered:** 100 stations, 15,300 station-days, 1354/1354 atlas checks PASS.

### Phase 2: Multi-crop water budgets
1. Digitize FAO-56 Kc for 40+ Michigan-relevant crops
2. Run water balance + dual Kc for each crop x station x year
3. Compute yield response using Stewart (1977) Ky factors

### Phase 3: Spatial interpolation
1. Use GPU kriging (`KrigingInterpolator`) for daily ET₀ surfaces
2. Produce seasonal and annual ET₀ maps

### Phase 4: Climate trend analysis
1. Decadal ET₀ trends (is Michigan getting drier?)
2. Growing season length changes (frost-free period)
3. Irrigation demand trends by crop

## Validated Modules Used

| Module | Experiment | Checks |
|--------|-----------|--------|
| `eco::evapotranspiration` | Exp 001 (FAO-56) | 64/64 Python, 31/31 Rust |
| `eco::water_balance` | Exp 004 (FAO-56 Ch 8) | 18/18 Python, 13/13 Rust |
| `eco::dual_kc` | Exp 005/011 (FAO-56 Ch 7) | 103/103 Python |
| `eco::yield_response` | Exp 012 (Stewart 1977) | 32/32 Python, 32/32 Rust |
| `gpu::kriging` | S28+ (wetSpring) | Integrated |
| `gpu::et0` | Exp 001 GPU | GPU-FIRST |

## Connection to Penny Irrigation

The atlas is the data foundation for Phase 4 (Penny Irrigation). A farmer at
any Michigan location can look up historical water demand for their crop,
see the typical stress periods, and use the scheduling optimizer (Exp 014)
to plan their season — all from a $200 sensor + free weather data + $600 GPU.
