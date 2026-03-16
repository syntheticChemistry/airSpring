# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared tolerance vocabulary for airSpring Python control baselines.

Mirrors barracuda/src/tolerances/ (57 named Rust constants) so that
Python scripts use the same names and values as the Rust validation
pipeline.  Import from here instead of hardcoding thresholds inline.

Usage::

    from tolerances import ET0_REFERENCE, WATER_BALANCE_MASS
    assert abs(rust_et0 - python_et0) < ET0_REFERENCE.abs_tol

The Tolerance class is a lightweight container; it is not enforced at
import time but provides structured access to abs_tol, rel_tol, and
justification for documentation and assertion messages.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    """Named validation tolerance matching Rust ``barracuda::tolerances::Tolerance``."""

    name: str
    abs_tol: float
    rel_tol: float
    justification: str


# ── Atmospheric (barracuda/src/tolerances/atmospheric.rs) ──

ET0_SAT_VAPOUR_PRESSURE = Tolerance(
    "et0_sat_vapour_pressure", 0.01, 1e-4,
    "FAO-56 Table 2.3: 3 decimal kPa; Tetens equation precision")

ET0_SLOPE_VAPOUR = Tolerance(
    "et0_slope_vapour_pressure", 0.005, 1e-4,
    "FAO-56 Table 2.4: 3 decimal kPa/°C; derivative of Tetens")

ET0_NET_RADIATION = Tolerance(
    "et0_net_radiation", 0.5, 0.05,
    "FAO-56 Ex 18: Rn chain (albedo→Rns→Rnl→Rn), ±0.5 MJ/m²/day")

ET0_REFERENCE = Tolerance(
    "et0_reference", 0.01, 1e-3,
    "FAO-56 Examples 17-19: validated against 3-decimal tables")

ET0_VPD = Tolerance(
    "et0_vpd", 0.02, 1e-3,
    "Combined saturation + actual vapour pressure uncertainty")

ET0_COLD_CLIMATE = Tolerance(
    "et0_cold_climate", 0.5, 0.1,
    "Near-zero ET₀ in cold climates; small denominator amplifies error")

PSYCHROMETRIC_CONSTANT = Tolerance(
    "psychrometric_constant", 0.001, 1e-4,
    "FAO-56 Eq 8: γ = 0.665e-3 × P; elevation precision")

THORNTHWAITE_ANALYTICAL = Tolerance(
    "thornthwaite_analytical", 1e-4, 1e-4,
    "Thornthwaite (1948) polynomial coefficients: 4-digit precision")

BLANEY_CRIDDLE_DAYLIGHT = Tolerance(
    "blaney_criddle_daylight", 0.015, 0.05,
    "FAO-24 Table 18 p values: ±0.015 covers latitude interpolation")

ET0_SAT_VAPOUR_PRESSURE_WIDE = Tolerance(
    "et0_sat_vapour_pressure_wide", 0.02, 1e-3,
    "FAO-56 Ex 17 Bangkok: high-T range doubles rounding to 0.02 kPa")

ET0_CROSS_METHOD_PCT = Tolerance(
    "et0_cross_method_pct", 25.0, 0.0,
    "Literature: Hargreaves vs PM 10-30% divergence; 25% accommodates Great Lakes")

MC_ET0_PROPAGATION = Tolerance(
    "mc_et0_propagation", 0.5, 0.1,
    "O(1/√N) CLT convergence: σ/√1000 ≈ 0.03; 0.5 provides 16σ headroom")

CROSS_VALIDATION = Tolerance(
    "cross_validation", 1e-5, 1e-5,
    "Rust vs Python f64: IEEE-754 produces ~1e-10 diffs; 1e-5 is conservative")

R2_MINIMUM = Tolerance(
    "r2_minimum", 0.85, 0.0,
    "FAO-56 PM typically R² > 0.90; 0.85 allows for ERA5 reanalysis noise")

RMSE_MAXIMUM = Tolerance(
    "rmse_maximum", 1.5, 0.0,
    "Doorenbos & Pruitt (1977): ±1.5 mm/day ET₀ measurement uncertainty")

# ── Soil (barracuda/src/tolerances/soil.rs) ──

WATER_BALANCE_MASS = Tolerance(
    "water_balance_mass", 0.01, 1e-6,
    "FAO-56 Ch 8: conservation law — ΔDr ≤ 0.01 mm per step")

STRESS_COEFFICIENT = Tolerance(
    "stress_coefficient", 0.01, 0.01,
    "FAO-56 Eq 84: Ks = (TAW-Dr)/(TAW-RAW), midpoint precision")

SOIL_HYDRAULIC = Tolerance(
    "soil_hydraulic", 0.01, 0.02,
    "USDA texture class θ_FC, θ_WP averages; pedotransfer uncertainty")

SOIL_ROUNDTRIP = Tolerance(
    "soil_roundtrip", 0.001, 1e-4,
    "Newton-Raphson VWC→dielectric→VWC roundtrip convergence")

RICHARDS_STEADY = Tolerance(
    "richards_steady_state", 0.001, 0.01,
    "HYDRUS benchmark: steady-state θ(z) profile precision")

RICHARDS_TRANSIENT = Tolerance(
    "richards_transient", 0.005, 0.02,
    "Picard iteration + implicit Euler; cumulative time-step error")

ISOTHERM_PARAMETER = Tolerance(
    "isotherm_parameter", 0.01, 0.01,
    "NM simplex convergence tol=1e-8, 5000 iterations max")

ISOTHERM_PREDICTION = Tolerance(
    "isotherm_prediction", 0.1, 0.01,
    "Batch adsorption experiment precision: ±0.1 mg/g")

ISOTHERM_MEAN_RESIDUAL = Tolerance(
    "isotherm_mean_residual", 0.5, 0.05,
    "Kumari et al. (2025): mean(|qe_obs - qe_pred|) < 0.5 mg/g")

WATER_BALANCE_PER_STEP = Tolerance(
    "water_balance_per_step", 1e-6, 1e-10,
    "Per-step conservation check: f64 arithmetic residual < 1e-6 mm")

TOPP_EQUATION = Tolerance(
    "topp_equation", 0.005, 1e-3,
    "Topp et al. (1980): 0.005 m³/m³ covers polynomial regression residual")

ANALYTICAL_COMPUTATION = Tolerance(
    "analytical_computation", 0.1, 0.01,
    "Generic analytical: covers digitization precision from published tables")

SCS_CN_ANALYTICAL = Tolerance(
    "scs_cn_analytical", 0.01, 1e-4,
    "SCS-CN Q and S: integer CN → f64 arithmetic yields ±0.01 mm")

GREEN_AMPT_ANALYTICAL = Tolerance(
    "green_ampt_analytical", 0.001, 1e-4,
    "Green-Ampt Newton iteration: 0.001 cm covers soil param uncertainty")

DUAL_KC_PRECISION = Tolerance(
    "dual_kc_precision", 0.01, 0.01,
    "FAO-56 Eq 72 Kc_max: 2-decimal tabulated values")

PEDOTRANSFER_MOISTURE = Tolerance(
    "pedotransfer_moisture", 1e-4, 1e-4,
    "Saxton & Rawls (2006) regression: 4-digit θ precision")

PEDOTRANSFER_KSAT = Tolerance(
    "pedotransfer_ksat", 0.5, 0.05,
    "Saxton & Rawls (2006) Ksat: exponential amplification of regression error")

GDD_EXACT = Tolerance(
    "gdd_exact", 1e-10, 1e-10,
    "GDD avg/clamp: f64-exact integer arithmetic (max/min/midpoint)")

IRRIGATION_DEPTH = Tolerance(
    "irrigation_depth", 0.01, 0.01,
    "Depth precision: (FC − VWC) × root_zone_m × 100; ±0.01 cm")

# ── GPU (barracuda/src/tolerances/gpu.rs) ──

GPU_CPU_CROSS = Tolerance(
    "gpu_cpu_cross_validation", 1e-5, 1e-5,
    "WGSL f64 shader vs CPU f64; BarraCuda TS-001/003 S54 validated")

KRIGING_INTERPOLATION = Tolerance(
    "kriging_interpolation", 1e-6, 1e-6,
    "Kriging weights via matrix solve; small N exact to f64 precision")

SEASONAL_REDUCTION = Tolerance(
    "seasonal_reduction", 1e-8, 1e-8,
    "FusedMapReduceF64 GPU sum; TS-004 S54 buffer fix (N≥1024)")

IOT_STREAM_SMOOTHING = Tolerance(
    "iot_stream_smoothing", 0.01, 1e-4,
    "MovingWindowStats uses f32 shaders; f32→f64 promotion rounding")

CROSS_SPRING_ANALYTICAL = Tolerance(
    "cross_spring_analytical", 1e-10, 1e-10,
    "Mathematical identities: erf(1), Γ(5)=24, J₀(0)=1 — f64-exact")

CROSS_SPRING_GPU_CPU = Tolerance(
    "cross_spring_gpu_cpu", 1e-4, 1e-4,
    "DF64 compound ops (exp, pow, log): ~1e-6 per op, chained to 1e-4")

CROSS_SPRING_EVOLUTION = Tolerance(
    "cross_spring_evolution", 1e-3, 1e-3,
    "Chained rewire (CPU→GPU): accumulates DF64 rounding across 3-5 ops")

NUCLEUS_ROUNDTRIP = Tolerance(
    "nucleus_roundtrip", 1e-10, 1e-10,
    "JSON f64 round-trip: IEEE-754 double → serde_json → double is exact")

NUCLEUS_PIPELINE = Tolerance(
    "nucleus_pipeline", 1e-6, 1e-6,
    "Multi-stage JSON-RPC pipeline: f64 arithmetic per stage, 4 stages max")

# ── Instrument (barracuda/src/tolerances/instrument.rs) ──

SENSOR_EXACT = Tolerance(
    "sensor_exact", 1e-10, 1e-10,
    "Polynomial evaluation and linear regression: f64-exact")

IOT_TEMPERATURE_MEAN = Tolerance(
    "iot_temperature_mean", 2.0, 0.1,
    "Synthetic 25°C centre ± diurnal; mean within ~2°C")

IOT_TEMPERATURE_EXTREMES = Tolerance(
    "iot_temperature_extremes", 3.0, 0.15,
    "Synthetic diurnal amplitude ~8°C; extremes by up to 3°C")

IOT_PAR_MAX = Tolerance(
    "iot_par_max", 200.0, 0.15,
    "Bell-curve PAR peak ≈ 1800 µmol/m²/s; discretization ± 200")

IOT_CSV_ROUNDTRIP = Tolerance(
    "iot_csv_roundtrip", 0.1, 0.01,
    "CSV {:.2} format truncation: round-trip within 0.1 of mean")

NPU_SIGMA_FLOOR = Tolerance(
    "npu_sigma_floor", 1e-10, 1e-10,
    "EMA variance floor — prevents division by zero in z-score")

BIO_DIVERSITY_SHANNON = Tolerance(
    "bio_diversity_shannon", 1e-8, 1e-8,
    "Shannon H' summation: f64 matches scipy.stats.entropy to 1e-8")

BIO_DIVERSITY_SIMPSON = Tolerance(
    "bio_diversity_simpson", 1e-10, 1e-10,
    "Simpson 1-D summation: pure f64 matches Python exactly to 1e-10")

BIO_BRAY_CURTIS = Tolerance(
    "bio_bray_curtis", 1e-8, 1e-8,
    "Bray-Curtis: f64 matches scipy.spatial.distance.braycurtis to 1e-8")

IA_CRITERION = Tolerance(
    "index_of_agreement_criterion", 0.80, 0.0,
    "Dong et al. (2020) Table 3: IA ≥ 0.80 for sensor correction adequacy")

P_SIGNIFICANCE = Tolerance(
    "p_significance", 0.05, 0.0,
    "Standard two-tailed significance level: α = 0.05")

WATER_SAVINGS = Tolerance(
    "water_savings", 0.1, 0.05,
    "IoT irrigation savings: ±10% comparison margin (Dong 2024 Fig 7)")

# ── Physical thresholds (not validation tolerances) ──

NPU_MIN_ANOMALY_SAMPLES: int = 10
NPU_STRESS_DEPLETION_THRESHOLD: float = 0.55

# ── Registry (all 57 tolerances, for introspection) ──

ALL_TOLERANCES: list[Tolerance] = [
    v for v in globals().values() if isinstance(v, Tolerance)
]
