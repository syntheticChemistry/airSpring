# Tolerance Registry — airSpring v0.8.7

58 named `Tolerance` structs used in airSpring's Rust validation pipeline
and Python control baselines. Each tolerance is defined once in
`barracuda/src/tolerances/` (Rust) and mirrored in `control/tolerances.py`
(Python). Tolerances are never hardcoded inline.

## Domains

| Domain | Module | Count |
|--------|--------|-------|
| Atmospheric | `tolerances/atmospheric.rs` | 15 |
| Soil | `tolerances/soil.rs` | 19 |
| GPU | `tolerances/gpu.rs` | 11 |
| Instrument | `tolerances/instrument.rs` | 13 |
| **Total (Tolerance structs)** | | **58** |

## Atmospheric (15)

| Name | abs_tol | rel_tol | Justification |
|------|---------|---------|---------------|
| `et0_sat_vapour_pressure` | 0.01 | 1e-4 | FAO-56 Table 2.3: 3 decimal kPa; Tetens equation precision |
| `et0_slope_vapour_pressure` | 0.005 | 1e-4 | FAO-56 Table 2.4: 3 decimal kPa/°C; derivative of Tetens |
| `et0_net_radiation` | 0.5 | 0.05 | FAO-56 Ex 18: Rn chain, ±0.5 MJ/m²/day |
| `et0_reference` | 0.01 | 1e-3 | FAO-56 Examples 17-19: validated against 3-decimal tables |
| `et0_vpd` | 0.02 | 1e-3 | Combined saturation + actual vapour pressure uncertainty |
| `et0_cold_climate` | 0.5 | 0.1 | Near-zero ET₀ in cold climates; small denominator amplifies error |
| `psychrometric_constant` | 0.001 | 1e-4 | FAO-56 Eq 8: γ = 0.665e-3 × P; elevation precision |
| `thornthwaite_analytical` | 1e-4 | 1e-4 | Thornthwaite (1948) polynomial: 4-digit precision |
| `blaney_criddle_daylight` | 0.015 | 0.05 | FAO-24 Table 18 p values: ±0.015 interpolation |
| `et0_sat_vapour_pressure_wide` | 0.02 | 1e-3 | FAO-56 Ex 17 Bangkok: high-T doubles rounding |
| `et0_cross_method_pct` | 25.0 | 0.0 | Hargreaves vs PM 10-30% divergence |
| `mc_et0_propagation` | 0.5 | 0.1 | O(1/√N) CLT convergence: 16σ headroom |
| `cross_validation` | 1e-5 | 1e-5 | Rust vs Python f64: IEEE-754 conservative |
| `r2_minimum` | 0.85 | 0.0 | FAO-56 PM R² > 0.90; 0.85 allows ERA5 noise |
| `rmse_maximum` | 1.5 | 0.0 | Doorenbos & Pruitt: ±1.5 mm/day |

## Soil (19)

| Name | abs_tol | rel_tol | Justification |
|------|---------|---------|---------------|
| `water_balance_mass` | 0.01 | 1e-6 | FAO-56 Ch 8: ΔDr ≤ 0.01 mm |
| `stress_coefficient` | 0.01 | 0.01 | FAO-56 Eq 84: Ks midpoint |
| `soil_hydraulic` | 0.01 | 0.02 | USDA texture class averages |
| `soil_roundtrip` | 0.001 | 1e-4 | Newton-Raphson VWC roundtrip |
| `richards_steady_state` | 0.001 | 0.01 | HYDRUS benchmark |
| `richards_transient` | 0.005 | 0.02 | Picard + implicit Euler |
| `isotherm_parameter` | 0.01 | 0.01 | Nelder-Mead convergence |
| `isotherm_prediction` | 0.1 | 0.01 | Batch adsorption ±0.1 mg/g |
| `isotherm_mean_residual` | 0.5 | 0.05 | Kumari et al. (2025) |
| `water_balance_per_step` | 1e-6 | 1e-10 | f64 arithmetic residual |
| `topp_equation` | 0.005 | 1e-3 | Topp et al. (1980) residual |
| `analytical_computation` | 0.1 | 0.01 | Published table digitization |
| `scs_cn_analytical` | 0.01 | 1e-4 | SCS-CN integer arithmetic |
| `green_ampt_analytical` | 0.001 | 1e-4 | Newton iteration convergence |
| `dual_kc_precision` | 0.01 | 0.01 | FAO-56 Eq 72 tabulated values |
| `pedotransfer_moisture` | 1e-4 | 1e-4 | Saxton & Rawls 4-digit θ |
| `pedotransfer_ksat` | 0.5 | 0.05 | Exponential regression error |
| `gdd_exact` | 1e-10 | 1e-10 | f64-exact integer arithmetic |
| `irrigation_depth` | 0.01 | 0.01 | (FC − VWC) × depth |

## GPU (11)

| Name | abs_tol | rel_tol | Justification |
|------|---------|---------|---------------|
| `gpu_cpu_cross_validation` | 1e-5 | 1e-5 | WGSL f64 shader vs CPU; BarraCuda TS-001/003 |
| `kriging_interpolation` | 1e-6 | 1e-6 | Matrix solve; small N exact |
| `seasonal_reduction` | 1e-8 | 1e-8 | FusedMapReduceF64; TS-004 fix |
| `iot_stream_smoothing` | 0.01 | 1e-4 | f32 shader promotion rounding |
| `cross_spring_analytical` | 1e-10 | 1e-10 | Mathematical identities |
| `cross_spring_gpu_cpu` | 1e-4 | 1e-4 | DF64 compound ops chained |
| `cross_spring_evolution` | 1e-3 | 1e-3 | Rewire CPU→GPU accumulation |
| `gpu_simplified_et0` | 5e-3 | 5e-3 | Makkink/Turc: 4-6 chained DF64 ops |
| `gpu_empirical_pet` | 1e-2 | 1e-2 | Hamon/Blaney-Criddle: temperature-only, ~1% DF64 |
| `nucleus_roundtrip` | 1e-10 | 1e-10 | JSON f64 round-trip |
| `nucleus_pipeline` | 1e-6 | 1e-6 | Multi-stage JSON-RPC |

## Instrument (13)

| Name | abs_tol | rel_tol | Justification |
|------|---------|---------|---------------|
| `sensor_exact` | 1e-10 | 1e-10 | Polynomial: f64-exact |
| `iot_temperature_mean` | 2.0 | 0.1 | Synthetic 25°C diurnal |
| `iot_temperature_extremes` | 3.0 | 0.15 | Diurnal amplitude ±3°C |
| `iot_par_max` | 200.0 | 0.15 | PAR peak discretization |
| `iot_csv_roundtrip` | 0.1 | 0.01 | CSV `{:.2}` truncation |
| `npu_sigma_floor` | 1e-10 | 1e-10 | EMA variance floor |
| `bio_diversity_shannon` | 1e-8 | 1e-8 | Shannon H' vs scipy |
| `bio_diversity_simpson` | 1e-10 | 1e-10 | Simpson 1-D exact |
| `bio_bray_curtis` | 1e-8 | 1e-8 | Bray-Curtis vs scipy |
| `index_of_agreement_criterion` | 0.80 | 0.0 | Dong (2020) Table 3 |
| `p_significance` | 0.05 | 0.0 | Two-tailed α = 0.05 |
| `water_savings` | 0.1 | 0.05 | Irrigation ±10% |
| `bootstrap_jackknife_known` | 0.01 | 1e-3 | Fixed-seed resampling mean parity |

Plus 2 physical threshold constants (not `Tolerance` structs):
- `NPU_MIN_ANOMALY_SAMPLES` = 10
- `NPU_STRESS_DEPLETION_THRESHOLD` = 0.55

## Usage

**Rust:**
```rust
use barracuda::tolerances::{ET0_REFERENCE, WATER_BALANCE_MASS};
ET0_REFERENCE.check(rust_et0, python_et0, "FAO-56 Ex 18");
```

**Python:**
```python
from tolerances import ET0_REFERENCE, WATER_BALANCE_MASS
assert abs(rust_val - py_val) < ET0_REFERENCE.abs_tol
```

## Evolution

- v0.8.0: 46 tolerances in monolithic `tolerances.rs`
- v0.8.1: 52 tolerances, added GPU/cross-spring/nucleus
- v0.8.2: 52 tolerances in 4 submodules + Python mirror `control/tolerances.py`
- v0.8.7: 58 tolerances — 3 new (gpu_simplified_et0, gpu_empirical_pet, bootstrap_jackknife_known), 3 biodiversity added to count
