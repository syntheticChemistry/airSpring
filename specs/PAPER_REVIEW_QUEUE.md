# airSpring — Paper Review Queue

**Last Updated**: February 25, 2026
**Purpose**: Track papers for reproduction/review, ordered by priority
**Status**: 9 completed (369/369 Python), 4 queued. All completed papers use open data and systems.

---

## Completed Reproductions

| # | Paper | Phase | Checks | Faculty | Control File | Open Data |
|---|-------|:-----:|:------:|---------|-------------|:---------:|
| 1 | Allen et al. (1998) FAO-56 Penman-Monteith — Ch 2/4 | 0 | 64/64 | Standard | `benchmark_fao56.json` | FAO-56 tables (open literature) |
| 2 | Dong et al. (2020) Soil sensor calibration — CS616/EC5 | 0 | 36/36 | Dong | `benchmark_dong2020.json` | Published Tables 3-4 |
| 3 | Dong et al. (2024) IoT irrigation pipeline — SoilWatch 10 | 0 | 24/24 | Dong | `benchmark_dong2024.json` | Published tables/equations |
| 4 | FAO-56 Chapter 8 — Water balance scheduling | 0 | 18/18 | Standard | `benchmark_water_balance.json` | FAO-56 Ch 8 + USDA |
| 5 | Real data pipeline — 6 Michigan stations, 918 days | 0+ | R²=0.967 | Dong | Python scripts | Open-Meteo ERA5 (free) |
| 6 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc (Kcb+Ke) | 0 | 63/63 | Standard | `benchmark_dual_kc.json` | FAO-56 Tables 17, 19 (open literature) |
| 7 | Regional ET₀ intercomparison — 6 Michigan stations | 0 | 61/61 | Dong | `regional_et0_intercomparison.py` | Open-Meteo ERA5 (free) |
| 8 | Islam et al. (2014) No-till + Allen FAO-56 Ch 11 cover crops | 0 | 40/40 | Standard | `benchmark_cover_crop_kc.json` | ISWCR + FAO-56 (open) |

### Controls Audit

All 6 completed papers have:
- **Digitized benchmarks** in `control/*/benchmark_*.json`
- **Python control scripts** that validate against benchmarks
- **Rust validation binaries** that load the same benchmarks
- **Open or published data** (no institutional access required)
- **Cross-validation** (65/65 Python↔Rust match at 1e-5)

### Compute Pipeline Per Paper

| Paper | Python Control | BarraCuda CPU | BarraCuda GPU | metalForge |
|:-----:|:--------------:|:-------------:|:-------------:|:----------:|
| 1 | 64/64 | 31/31 (`validate_et0`) | `BatchedEt0` GPU-FIRST | — |
| 2 | 36/36 | 26/26 (`validate_soil`) | `fit_ridge` (ridge regression) | — |
| 3 | 24/24 | 11/11 (`validate_iot`) | `StreamSmoother` (moving window) | — |
| 4 | 18/18 | 13/13 (`validate_water_balance`) | `BatchedWaterBalance` GPU-STEP | — |
| 5 | R²=0.967 | 21/21 (`validate_real_data`) | All 6 orchestrators | Future |
| 6 | 63/63 | — (next: BarraCuda CPU) | Batch Kc (op=7) | Future |
| 7 | 61/61 | — (next: BarraCuda CPU) | `BatchedEt0` at scale | Future |
| 8 | 40/40 | — (next: BarraCuda CPU) | Batch Kc (op=7) + mulch | Future |

---

## Review Queue

### Tier 1 — Direct extensions of current work

| # | Paper / Direction | Year | Faculty | Open Data? | Control Status | GPU Path |
|---|-------------------|------|---------|:----------:|:--------------:|----------|
| 6 | Dong et al. — Multi-sensor calibration network | 2024+ | Dong | Awaiting field data | None | Batch calibration (op=5) |
| 7 | Dong et al. — Full IoT irrigation + forecast | 2024+ | Dong | Awaiting field data | None | Forecast integration |
| 8 | Allen et al. (1998) FAO-56 Ch 7 — Dual Kc | 1998 | Standard | **Yes** (open literature) | **63/63 PASS** (Phase 0) | Batch Kc (op=7) |

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
| 14 | Soil moisture → Anderson d_eff coupling model | — | Cross-spring | **Yes** (USDA + Open-Meteo) | Future | `BatchedWaterBalance` → Anderson |
| 15 | OSU Triplett-Van Doren 60-year water balance reconstruction | — | Cross-spring | **Yes** (Open-Meteo 80-yr, USDA soils) | Future | `BatchedEt0` at scale |
| 16 | Cover crop water use & seasonal diversity dynamics | — | Dong | Awaiting field data | Future | Batch ET₀ with Kc schedule |

**Connection to wetSpring**: airSpring computes θ(t); wetSpring computes
Anderson r(t) from θ(t)-derived geometry. The cross-spring pipeline is:
`Open-Meteo weather → FAO-56 ET₀ → water balance θ(t) → pore_connectivity(t) → d_eff(t) → Anderson r(t) → QS_regime(t)`.

**Connection to groundSpring**: groundSpring Exp 003 already showed humidity
dominates ET₀ uncertainty at 66%. This propagates into the Anderson coupling:
moisture uncertainty → geometry uncertainty → QS prediction uncertainty.

### Tier 4 — Longer horizon

| # | Paper / Direction | Year | Faculty | Open Data? | Control Status | GPU Path |
|---|-------------------|------|---------|:----------:|:--------------:|----------|
| 17 | Dolson — Evolutionary optimization of sensor placement | — | Dolson | N/A | Future | `NelderMeadGpu` |
| 18 | Waters — Soil microbiome ↔ plant water dynamics | — | Waters | N/A | Future | N/A |

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

- Papers 6-7 depend on access to Dong lab's real field data (new lab 2026)
- Paper 8 (dual Kc) is a pure literature reproduction — all data in FAO-56 Chapter 7
- Paper 9 would use the 80-year Open-Meteo archive — massive open dataset, no key
- Papers 10-11 are cross-spring references — already validated in their respective springs
- Papers 12-16 (Tier 3) support baseCamp Sub-thesis 06 (no-till Anderson QS)
- Paper 12 (Brandt farm) and Paper 15 (OSU 60-year reconstruction) use only open data
- Paper 13 (dual Kc) is needed for cover crop water balance in the Anderson coupling
- Every completed paper has been validated through the full pipeline: Python → Rust CPU → GPU
