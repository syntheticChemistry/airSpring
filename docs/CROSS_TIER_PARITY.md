# airSpring Cross-Tier Parity Status

**Date**: May 25, 2026 (Wave 50 Covalent HPC)
**Pattern**: primalSpring `docs/VALIDATION_TIERS.md` — Tier 1 (Python), Tier 2 (Rust), Tier 3 (provenance)

---

## Overview

Cross-tier parity confirms that Python baseline science and Rust implementations
agree within documented tolerances. airSpring validates this through the
`control/*/benchmark_*.json` → `validate_*` binary pattern:

1. **Tier 1**: Python control script generates `benchmark_*.json` with expected values
2. **Tier 2**: Rust binary `include_str!`s the JSON and computes results
3. **Tier 2 parity**: Binary compares Rust vs Python within `tolerances/` thresholds
4. **Tier 3**: Provenance trio records the validation chain (when live primals available)

---

## Per-Method Parity Matrix

| Method | Python NB | `benchmark_*.json` | Rust `validate_*` | Status |
|--------|:---------:|:-------------------:|:------------------:|--------|
| `science.et0_fao56` | 001 | `benchmark_fao56.json` | `validate_et0`, `cross_validate` | **Full** |
| `science.et0_hargreaves` | 031 | `benchmark_hargreaves.json` | `validate_hargreaves` | **Full** |
| `science.et0_priestley_taylor` | 019 | `benchmark_priestley_taylor.json` | `validate_priestley_taylor` | **Full** |
| `science.et0_makkink` | 033 | `benchmark_makkink.json` | `validate_makkink` | **Full** |
| `science.et0_turc` | 034 | `benchmark_turc.json` | `validate_turc` | **Full** |
| `science.et0_hamon` | 035 | `benchmark_hamon.json` | `validate_hamon` | **Full** |
| `science.et0_blaney_criddle` | 049 | `benchmark_blaney_criddle.json` | `validate_blaney_criddle` | **Full** |
| `science.thornthwaite` | 021 | `benchmark_thornthwaite.json` | `validate_thornthwaite` | **Full** |
| `science.water_balance` | 004 | `benchmark_water_balance.json` | `validate_water_balance` | **Full** |
| `science.yield_response` | 008 | `benchmark_yield_response.json` | `validate_yield` | **Full** |
| `science.richards_1d` | 006 | `benchmark_richards.json` | `validate_richards` | **Full** |
| `science.scs_cn_runoff` | 050 | `benchmark_scs_cn.json` | `validate_scs_cn` | **Full** |
| `science.green_ampt_infiltration` | 051 | `benchmark_green_ampt.json` | `validate_green_ampt` | **Full** |
| `science.dual_kc` | 009 | `benchmark_dual_kc.json` | `validate_dual_kc` | **Full** |
| `science.sensor_calibration` | 002 | `benchmark_dong2020.json` | `validate_sensor_calibration` | **Full** |
| `science.pedotransfer_saxton_rawls` | 023 | `benchmark_pedotransfer.json` | `validate_pedotransfer` | **Full** |
| `science.spi_drought_index` | 081 | `benchmark_drought_index.json` | `validate_drought_index` | **Full** |
| `science.gdd` | — | `benchmark_gdd.json` | `validate_gdd` | Tier 2 (no NB) |
| `science.shannon_diversity` | — | `benchmark_diversity.json` | `validate_diversity` | Tier 2 (no NB) |
| `science.bray_curtis` | — | `benchmark_diversity.json` | `validate_diversity` | Tier 2 (no NB) |
| `science.anderson_coupling` | — | `benchmark_anderson_coupling.json` | `validate_anderson` | Tier 2 (no NB) |
| `science.autocorrelation` | — | `benchmark_autocorrelation.json` | `validate_autocorrelation` | **Tier 2** (new) |
| `science.gamma_cdf` | — | `benchmark_gamma_cdf.json` | `validate_gamma_cdf` | **Tier 2** (new) |
| `science.soil_moisture_topp` | — | `benchmark_soil_moisture_topp.json` | `validate_soil_moisture_topp` | **Tier 2** (new) |

---

## Stability Tier Annotations

All science methods annotated with `stability = "stable"` in `capability_registry.toml`.
Inference and compute methods annotated `stability = "evolving"`.
See `capability_registry.toml` header for tier definitions.

---

## Remaining Gaps

- **Notebooks for 5 methods**: `gdd`, `shannon_diversity`/`bray_curtis`, `anderson_coupling`,
  `autocorrelation`, `gamma_cdf`, `soil_moisture_topp` have Python control scripts and Rust
  validators but no Jupyter narrative notebook. Low priority — science is validated.
- **Tier 3 provenance**: All methods can record provenance when trio primals are live.
  Currently validated in offline mode (`status: "unavailable"`). Live validation pending
  deployment of rhizoCrypt + loamSpine + sweetGrass.
