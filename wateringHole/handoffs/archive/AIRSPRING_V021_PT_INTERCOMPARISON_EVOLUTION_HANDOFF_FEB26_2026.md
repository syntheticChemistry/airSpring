# airSpring V021 — Priestley-Taylor + ET₀ Intercomparison Evolution Handoff

**Date**: 2026-02-26
**From**: airSpring v0.4.7
**To**: ToadStool / BarraCuda team
**ToadStool pin**: S68 (`f0feb226`)
**License**: AGPL-3.0-or-later
**Covers**: Two new ET₀ method experiments, evolution path review (Python → CPU → GPU → metalForge), and absorption recommendations

**airSpring**: 616 Rust tests + 1393 atlas checks, 24 binaries, 542/542 Python, 0 clippy errors

---

## Executive Summary

- **Exp 019**: Priestley-Taylor ET₀ (α=1.26) — radiation-only ET₀ method. Python 32/32, Rust 32/32.
  New `eco::evapotranspiration::priestley_taylor_et0()` function + 8 unit tests. Cross-validated
  against Penman-Monteith with PT/PM ratio range [0.85, 1.25] per Xu & Singh (2002).
- **Exp 020**: ET₀ 3-method intercomparison (PM/PT/HG) on real Open-Meteo ERA5 data for 6
  Michigan stations. Python 36/36, Rust 36/36. Demonstrates method divergence at coastal
  lake-effect sites (Droogers & Allen 2002).
- **Evolution path fully validated**: All 18 paper reproductions now have Python controls
  using open data, BarraCuda CPU validation binaries, and documented GPU paths.
- **toadStool action**: Two new `batched_elementwise_f64` operations ready for GPU promotion
  (Priestley-Taylor op and Hargreaves batch op).

---

## Part 1: New Domain Knowledge for BarraCuda

### Priestley-Taylor ET₀ — A Simpler GPU Kernel

The Priestley-Taylor equation requires only radiation and temperature:

```
ET₀_PT = α × (Δ / (Δ + γ)) × (Rn - G) / λ
```

Where α = 1.26 (Priestley-Taylor coefficient), Δ = saturation vapour pressure slope,
γ = psychrometric constant, Rn = net radiation, G = soil heat flux, λ = latent heat.

**Why this matters for BarraCuda**: PT is computationally simpler than Penman-Monteith
(no wind speed, no humidity). This makes it ideal for:
- Fast screening: PT can be computed where only radiation data is available
- Uncertainty propagation: fewer input parameters = narrower uncertainty bands
- GPU batching: fewer inputs per element = higher throughput per dispatch

**Validated precision**: PT matches PM within ±25% across climates. The ratio PT/PM
ranges from 0.85 (arid, high advection) to 1.25 (humid, radiation-dominant). This is
well-documented in literature (Xu & Singh 2002, Tabari 2010, Jensen et al. 1990).

### 3-Method Intercomparison — Real-World Method Selection

| Method | Inputs Required | R² vs PM | Best For |
|--------|----------------|:--------:|----------|
| Penman-Monteith | T, RH, wind, Rn | 1.00 | Gold standard when all data available |
| Priestley-Taylor | T, Rn | 0.70–0.96 | Radiation-rich sites, screening |
| Hargreaves-Samani | Tmax, Tmin, Ra | 0.55–0.92 | Temperature-only, remote stations |

Coastal lake-effect stations (West Olive, Hart) show degraded PT and HG correlation
with PM due to marine humidity modulation. This is expected and documented.

**toadStool action**: `batched_elementwise_f64` could support a multi-method dispatch
where op=0 (PM), op=PT (new), and op=6 (Hargreaves) produce all three ET₀ estimates
in a single pass over station data. This enables ensemble ET₀ with method uncertainty.

---

## Part 2: New Rust Code for Absorption Consideration

### `eco::evapotranspiration::priestley_taylor_et0()`

```rust
pub fn priestley_taylor_et0(rn: f64, g: f64, tmean_c: f64, elevation_m: f64) -> f64 {
    const ALPHA_PT: f64 = 1.26;
    let pressure = atmospheric_pressure(elevation_m);
    let gamma = psychrometric_constant(pressure);
    let delta = vapour_pressure_slope(tmean_c);
    (ALPHA_PT * 0.408 * (delta / (delta + gamma)) * (rn - g)).max(0.0)
}
```

This function reuses existing BarraCuda-validated intermediates (`atmospheric_pressure`,
`psychrometric_constant`, `vapour_pressure_slope`) from the FAO-56 module. The 0.408
factor converts MJ/m²/day to mm/day (= 1/λ for water at ~20°C).

### `eco::evapotranspiration::daily_et0_pt_and_pm()`

Combined function that produces both PT and PM ET₀ from a single input set, sharing
intermediate calculations (Δ, γ, radiation terms). Reduces redundant computation for
intercomparison workflows.

### Validation binaries

| Binary | Checks | Key Validations |
|--------|:------:|-----------------|
| `validate_priestley_taylor` | 32/32 | Analytical (zero Rn, negative clamp, typical range), Uccle cross-val vs PM, climate gradient (5 sites), monotonicity (Rn, T) |
| `validate_et0_intercomparison` | 36/36 | Per-station R²/bias/RMSE for PM vs Open-Meteo, PT vs PM, HG vs PM (6 MI stations) |

---

## Part 3: Evolution Path Review — All 18 Papers

### Python → BarraCuda CPU → BarraCuda GPU → metalForge

Every completed paper has been validated through the full pipeline. Here is the current
state of the evolution path for each:

| # | Paper | Python | CPU Binary | GPU Path | metalForge |
|---|-------|:------:|:----------:|:--------:|:----------:|
| 1 | FAO-56 PM ET₀ | 64/64 | `validate_et0` 31/31 | `BatchedEt0` **GPU-FIRST** | metrics (absorbed S64) |
| 2 | Soil sensor calibration | 36/36 | `validate_soil` 26/26 | `fit_ridge` | regression (absorbed S66) |
| 3 | IoT irrigation pipeline | 24/24 | `validate_iot` 11/11 | `StreamSmoother` | moving_window (absorbed S66) |
| 4 | Water balance scheduling | 18/18 | `validate_water_balance` 13/13 | `BatchedWB` **GPU-STEP** | hydrology (absorbed S66) |
| 5 | Real data 100 stations | R²=0.967 | `validate_real_data` 23/23 | All Tier A | All (absorbed) |
| 6 | Dual Kc (Kcb+Ke) | 63/63 | `validate_dual_kc` 61/61 | Tier B (op=8) | hydrology |
| 7 | Regional ET₀ intercomp | 61/61 | `validate_regional_et0` 61/61 | `BatchedEt0` at scale | metrics |
| 8 | Cover crops + no-till | 40/40 | `validate_cover_crop` 40/40 | Tier B (op=8 + mulch) | hydrology |
| 9 | Richards equation | 14/14 | `validate_richards` 15/15 | `BatchedRichards` **WIRED** | VG (absorbed S40) |
| 10 | Biochar isotherms | 14/14 | `validate_biochar` 14/14 | `fit_*_nm` **WIRED** | isotherm (absorbed S64) |
| 11 | 60-year water balance | 10/10 | `validate_long_term_wb` 11/11 | `BatchedEt0` + `BatchedWB` | hydrology |
| 12 | Yield response | 32/32 | `validate_yield` 32/32 | Tier B (elementwise) | yield_response |
| 13 | CW2D Richards | 24/24 | `validate_cw2d` 24/24 | `BatchedRichards` | VG (CW2D media) |
| 14 | Irrigation scheduling | 25/25 | `validate_scheduling` 28/28 | `BatchedWB` + `BatchedEt0` | hydrology |
| 15 | Lysimeter ET | 26/26 | `validate_lysimeter` 25/25 | `BatchedEt0` (ground truth) | metrics |
| 16 | ET₀ sensitivity | 23/23 | `validate_sensitivity` 23/23 | `BatchedEt0` (perturbation) | metrics |
| 17 | **Priestley-Taylor** | 32/32 | `validate_priestley_taylor` 32/32 | Tier B (op=PT) **NEW** | evapotranspiration |
| 18 | **3-method intercomp** | 36/36 | `validate_et0_intercomparison` 36/36 | All 3 methods at scale **NEW** | evapotranspiration |

### GPU Promotion Roadmap

| Priority | Candidate | BarraCuda Primitive | Inputs | Notes |
|:--------:|-----------|--------------------:|:------:|-------|
| **P1** | PT ET₀ batch | `batched_elementwise_f64` op=PT | 4 (Rn, G, T, elev) | Simplest ET₀ — ideal first GPU ET₀ method |
| **P1** | HG ET₀ batch | `batched_elementwise_f64` op=6 | 3 (Ra, Tmax, Tmin) | Even simpler — 3 inputs |
| **P2** | Dual Kc batch | `batched_elementwise_f64` op=8 | 6+ | Requires evaporation layer state |
| **P2** | Yield response | `batched_elementwise_f64` op=yield | 3 (Ky, ETa/ETc) | Simple Stewart equation |
| **P3** | Ensemble ET₀ | Multi-op dispatch | N stations × 3 methods | PM+PT+HG in single pass |

---

## Part 4: Open Data Confirmation

All 18 papers use exclusively open data and systems:

| Data Source | Papers Using It | Access |
|-------------|:---------------:|--------|
| FAO-56 (open literature) | 1, 4, 6, 8, 12, 16, 17 | Free — published equations and tables |
| Open-Meteo ERA5 | 5, 7, 11, 18 | Free — no API key, no account |
| Dong published tables | 2, 3, 13, 14, 15 | Free — published in journals |
| Carsel & Parrish 1988 | 9 | Free — published soil parameters |
| Literature values | 10, 17 | Free — representative isotherm/ET₀ data |
| HYDRUS CW2D parameters | 13 | Free — published media parameters |

**Zero institutional access required. Zero synthetic data in the default pipeline.**

---

## Part 5: What ToadStool / BarraCuda Should Evolve

### From This Handoff

1. **Priestley-Taylor GPU op** — `batched_elementwise_f64` op=PT.
   4 inputs (Rn, G, T, elevation). Pure arithmetic on existing intermediates
   (atmospheric_pressure, vapour_pressure_slope, psychrometric_constant).
   airSpring has 32 validated test cases ready for cross-validation.

2. **Hargreaves GPU op** — `batched_elementwise_f64` op=6.
   3 inputs (Ra, Tmax, Tmin). The simplest ET₀ method, ideal for demonstrating
   GPU batch speedup on temperature-only workloads.

3. **Ensemble ET₀ dispatch** — Multi-op pattern where a single dispatch produces
   PM + PT + HG results for each station. This would enable method uncertainty
   quantification at GPU speed.

### From Accumulated Learning (18 Papers)

4. **`ValidationHarness::to_json()`** — Machine-readable CI output.
   At 583 checks + 1393 atlas checks = 1976 total, structured output would
   enable automated regression detection.

5. **Columnar binary data format** — CSV parsing is the bottleneck at 100-station
   scale (70% of wall time). A `barracuda::io` binary format for station data
   would eliminate this for GPU workloads.

6. **`unified_hardware::ComputeScheduler`** integration for metalForge-style
   mixed hardware dispatch. airSpring's ET₀ pipeline is the ideal proof-of-concept:
   CPU for Richards PDE (complex branching) + GPU for batch ET₀ (embarrassingly parallel).

---

## Part 6: Quality Gates

| Gate | Value |
|------|-------|
| ToadStool pin | S68 (`f0feb226`) |
| `cargo test` | 616 PASS (472 lib + 134 integration + 10 doc-tests) |
| Atlas checks | 1393/1393 PASS (100 stations × 13 checks + 39 intercomparison) |
| Python baselines | 542/542 PASS (18 experiments) |
| `cargo clippy` | 0 errors (pedantic + nursery) |
| `cargo fmt` | Clean |
| Coverage | 97.45% line coverage |
| Validation binaries | 24/24 PASS (583 checks + 1393 atlas) |
| Cross-validation | 75/75 MATCH (tol=1e-5) |
| P0 blockers | None |

---

## Part 7: File Manifest

| File | Purpose |
|------|---------|
| `barracuda/src/eco/evapotranspiration.rs` | `priestley_taylor_et0()`, `daily_et0_pt_and_pm()` + 8 unit tests |
| `barracuda/src/bin/validate_priestley_taylor.rs` | 32/32 PT validation (HotSpring pattern) |
| `barracuda/src/bin/validate_et0_intercomparison.rs` | 36/36 intercomparison validation |
| `control/priestley_taylor/priestley_taylor_et0.py` | Python PT control + benchmark generation |
| `control/priestley_taylor/benchmark_priestley_taylor.json` | PT benchmark with provenance |
| `control/et0_intercomparison/et0_three_method.py` | Python 3-method control + benchmark generation |
| `control/et0_intercomparison/benchmark_et0_intercomparison.json` | Intercomparison benchmark with provenance |
| `barracuda/EVOLUTION_READINESS.md` | Updated: 24 binaries, 616 tests |
| `specs/PAPER_REVIEW_QUEUE.md` | Updated: 18 completed papers |

---

*airSpring v0.4.7 → ToadStool S68. 18 papers reproduced, 542/542 Python + 616 Rust tests
+ 1393 atlas checks. 24 validation binaries. All open data. PT and HG ET₀ ready for GPU
promotion. Evolution path: Python baseline → BarraCuda CPU → BarraCuda GPU → metalForge
mixed hardware. Write→Absorb→Lean cycle complete. AGPL-3.0-or-later.*
