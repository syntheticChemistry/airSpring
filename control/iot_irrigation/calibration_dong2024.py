#!/usr/bin/env python3
"""
airSpring Experiment 003 — IoT Irrigation Pipeline Baseline (Python part)

Replicates the methodology from:
  Dong, Werling, Cao, Li (2024) "Implementation of an In-Field IoT System
  for Precision Irrigation Management" Frontiers in Water 6, 1353597.
  doi:10.3389/frwa.2024.1353597

Implements:
  1. SoilWatch 10 calibration equation (Eq. 5)
  2. Irrigation recommendation model (Eq. 1)
  3. Statistical evaluation (RMSE, IA, MBE — same as Dong 2020)
  4. Sensor performance validation against published Table 2

The ANOVA analysis is in anova_irrigation.R (R v4.3.1, matching the paper).

All open-source: numpy, scipy only.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SoilWatch 10 Calibration — Paper's Equation 5
# ---------------------------------------------------------------------------

def soilwatch10_vwc(raw_count: float) -> float:
    """
    SoilWatch 10 calibration equation (Dong et al. 2024, Eq. 5):
      VWC = 2×10⁻¹³ RC³ - 4×10⁻⁹ RC² + 4×10⁻⁵ RC - 0.0677

    Args:
        raw_count: Analog raw count from sensor at 3.3V
    Returns:
        Volumetric water content (cm³/cm³)
    """
    return (2e-13 * raw_count**3
            - 4e-9 * raw_count**2
            + 4e-5 * raw_count
            - 0.0677)


def soilwatch10_vwc_vec(raw_counts: np.ndarray) -> np.ndarray:
    """Vectorized SoilWatch 10 calibration."""
    return (2e-13 * raw_counts**3
            - 4e-9 * raw_counts**2
            + 4e-5 * raw_counts
            - 0.0677)


# ---------------------------------------------------------------------------
# Irrigation Recommendation Model — Paper's Equation 1
# ---------------------------------------------------------------------------

def irrigation_recommendation(field_capacity: float,
                               current_vwc: float,
                               depth_cm: float) -> float:
    """
    Dong et al. 2024, Eq. 1:
      IR = (FC_layer_i - θv_layer_i) × D_layer_i

    Args:
        field_capacity: Field capacity of soil layer (cm³/cm³)
        current_vwc: Current volumetric water content (cm³/cm³)
        depth_cm: Representative soil layer depth (cm)
    Returns:
        Maximum irrigation recommendation (cm)
    """
    return max(0.0, (field_capacity - current_vwc) * depth_cm)


def multi_layer_irrigation(layers: list) -> float:
    """
    Sum irrigation recommendation across multiple sensor depths.
    Each layer: {"fc": float, "vwc": float, "depth_cm": float}
    """
    return sum(irrigation_recommendation(l["fc"], l["vwc"], l["depth_cm"])
               for l in layers)


# ---------------------------------------------------------------------------
# Statistical functions (same as Dong 2020 — Eqs 2-4 in this paper)
# ---------------------------------------------------------------------------

def compute_rmse(measured: np.ndarray, predicted: np.ndarray) -> float:
    n = len(measured)
    return math.sqrt(np.sum((measured - predicted)**2) / n)


def compute_ia(measured: np.ndarray, predicted: np.ndarray) -> float:
    m_bar = np.mean(measured)
    numerator = np.sum((measured - predicted)**2)
    denominator = np.sum((np.abs(predicted - m_bar) +
                          np.abs(measured - m_bar))**2)
    if denominator == 0:
        return 1.0
    return 1.0 - numerator / denominator


def compute_mbe(measured: np.ndarray, predicted: np.ndarray) -> float:
    n = len(measured)
    return np.sum(predicted - measured) / n


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

def check(label: str, computed: float, expected: float, tol: float) -> bool:
    diff = abs(computed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  [{status}] {label}: {computed:.4f} "
          f"(expected {expected:.4f}, tol {tol:.4f})")
    return diff <= tol


def check_bool(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def validate_soilwatch10(benchmark: dict) -> tuple:
    """Validate SoilWatch 10 calibration equation."""
    passed = 0
    failed = 0

    print("\n=== SoilWatch 10 Calibration (Eq. 5) ===")
    cal = benchmark["soilwatch10_calibration"]
    vwc_min = cal["vwc_calibration_range"]["min_cm3_cm3"]
    vwc_max = cal["vwc_calibration_range"]["max_cm3_cm3"]

    # The equation should produce VWC in the calibrated range
    # for some span of raw counts
    raw_counts = np.linspace(1000, 50000, 1000)
    vwc_values = soilwatch10_vwc_vec(raw_counts)

    # Find range that produces valid VWC
    valid_mask = (vwc_values >= vwc_min) & (vwc_values <= vwc_max)
    valid_count = np.sum(valid_mask)

    if check_bool(f"Calibration produces VWC in [{vwc_min}, {vwc_max}] "
                  f"range: {valid_count} valid points", valid_count > 0):
        passed += 1
    else:
        failed += 1

    # Check monotonicity in valid range
    if valid_count > 1:
        valid_vwc = vwc_values[valid_mask]
        monotonic = np.all(np.diff(valid_vwc) >= 0)
        if check_bool("Monotonically increasing in valid range", monotonic):
            passed += 1
        else:
            failed += 1
    else:
        print("  [SKIP] Not enough valid points for monotonicity check")

    # Check equation coefficients match
    coeffs = cal["equation_coefficients"]
    test_rc = 10000.0
    expected_vwc = (coeffs["a3"] * test_rc**3
                    + coeffs["a2"] * test_rc**2
                    + coeffs["a1"] * test_rc
                    + coeffs["a0"])
    computed_vwc = soilwatch10_vwc(test_rc)
    if check("VWC(RC=10000)", computed_vwc, expected_vwc, 1e-10):
        passed += 1
    else:
        failed += 1

    # Check boundary behavior
    if check_bool("VWC(RC=0) is negative (below calibration range)",
                  soilwatch10_vwc(0) < 0):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_irrigation_model(benchmark: dict) -> tuple:
    """Validate irrigation recommendation equation."""
    passed = 0
    failed = 0

    print("\n=== Irrigation Recommendation (Eq. 1) ===")
    ir_data = benchmark["irrigation_recommendation"]
    example = ir_data["example_sandy_soil"]

    ir = irrigation_recommendation(
        example["field_capacity_cm_cm"],
        example["current_vwc_cm_cm"],
        example["depth_cm"],
    )
    if check("IR (sandy soil example)", ir,
             example["expected_ir_cm"], 0.01):
        passed += 1
    else:
        failed += 1

    # At field capacity, IR should be 0
    ir_at_fc = irrigation_recommendation(0.12, 0.12, 30)
    if check("IR at field capacity", ir_at_fc, 0.0, 1e-10):
        passed += 1
    else:
        failed += 1

    # Over field capacity should still return 0
    ir_over = irrigation_recommendation(0.12, 0.15, 30)
    if check("IR above field capacity", ir_over, 0.0, 1e-10):
        passed += 1
    else:
        failed += 1

    # Multi-layer test (corn: 15, 60, 90 cm)
    layers = [
        {"fc": 0.12, "vwc": 0.08, "depth_cm": 30},
        {"fc": 0.15, "vwc": 0.10, "depth_cm": 30},
        {"fc": 0.18, "vwc": 0.12, "depth_cm": 30},
    ]
    total_ir = multi_layer_irrigation(layers)
    expected_total = (0.04 * 30) + (0.05 * 30) + (0.06 * 30)
    if check("Multi-layer IR (3 layers)", total_ir, expected_total, 0.01):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_sensor_performance(benchmark: dict) -> tuple:
    """Validate published sensor performance meets criteria."""
    passed = 0
    failed = 0

    print("\n=== Sensor Performance (Table 2) ===")
    perf = benchmark["sensor_performance_table2"]
    criteria = benchmark["criteria"]
    mbe_thresh = criteria["mbe_threshold_cm3_cm3"]
    rmse_thresh = criteria["rmse_threshold_cm3_cm3"]

    for soil, stats in perf.items():
        if soil.startswith("_"):
            continue
        rmse = stats["rmse_cm3_cm3"]
        ia = stats["ia"]
        mbe = stats["mbe_cm3_cm3"]

        rmse_ok = rmse < rmse_thresh
        mbe_ok = abs(mbe) <= mbe_thresh

        if check_bool(f"{soil}: RMSE={rmse:.3f} < {rmse_thresh} (criteria)",
                      rmse_ok):
            passed += 1
        else:
            failed += 1

        if check_bool(f"{soil}: |MBE|={abs(mbe):.3f} ≤ {mbe_thresh} "
                      f"(criteria)", mbe_ok):
            passed += 1
        else:
            failed += 1

        if check_bool(f"{soil}: IA={ia:.2f} > 0.80 (criteria)", ia > 0.80):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_field_demonstrations(benchmark: dict) -> tuple:
    """Validate published field demonstration results."""
    passed = 0
    failed = 0

    print("\n=== Field Demonstration Results ===")

    # Blueberry
    bb = benchmark["blueberry_demonstration"]
    bb_yield_diff = (bb["treatment_recommended"]["yield_per_plant_g"] -
                     bb["treatment_farmer"]["yield_per_plant_g"])
    if check_bool(f"Blueberry: recommended yield ({bb['treatment_recommended']['yield_per_plant_g']}g) "
                  f"> farmer yield ({bb['treatment_farmer']['yield_per_plant_g']}g)",
                  bb_yield_diff > 0):
        passed += 1
    else:
        failed += 1

    if check_bool(f"Blueberry yield p={bb['anova_results']['yield_p_value']} "
                  f"< 0.05 (significant)",
                  bb["anova_results"]["yield_p_value"] < 0.05):
        passed += 1
    else:
        failed += 1

    if check_bool(f"Blueberry berry weight p={bb['anova_results']['berry_weight_p_value']} "
                  f"< 0.05 (significant)",
                  bb["anova_results"]["berry_weight_p_value"] < 0.05):
        passed += 1
    else:
        failed += 1

    # Tomato
    tom = benchmark["tomato_demonstration"]
    if check_bool(f"Tomato: count p={tom['anova_results']['marketable_count_p_value']} "
                  f"> 0.05 (not significant — same yield)",
                  tom["anova_results"]["marketable_count_p_value"] > 0.05):
        passed += 1
    else:
        failed += 1

    if check_bool(f"Tomato: weight p={tom['anova_results']['weight_p_value']} "
                  f"> 0.05 (not significant — same quality)",
                  tom["anova_results"]["weight_p_value"] > 0.05):
        passed += 1
    else:
        failed += 1

    if check("Tomato: water savings (%)",
             tom["water_savings_pct"], 30.0, 0.1):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_statistics_with_synthetic(benchmark: dict) -> tuple:
    """
    Generate synthetic sensor data matching published RMSE/MBE
    and verify our statistical computations recover them.
    """
    passed = 0
    failed = 0

    print("\n=== Synthetic Statistical Validation ===")
    rng = np.random.default_rng(42)

    perf = benchmark["sensor_performance_table2"]

    for soil, stats in perf.items():
        if soil.startswith("_"):
            continue
        target_mbe = stats["mbe_cm3_cm3"]
        target_rmse = stats["rmse_cm3_cm3"]

        # Generate N=50 synthetic data with known bias + noise
        n = 50
        measured = rng.uniform(0.05, 0.35, n)
        noise_std = math.sqrt(max(target_rmse**2 - target_mbe**2, 1e-10))
        predicted = measured + target_mbe + rng.normal(0, noise_std, n)

        computed_mbe = compute_mbe(measured, predicted)
        computed_rmse = compute_rmse(measured, predicted)

        # MBE should be close to target (within statistical noise)
        if check(f"{soil} synthetic MBE", computed_mbe,
                 target_mbe, 0.015):
            passed += 1
        else:
            failed += 1

        # RMSE should be in the right ballpark
        if check(f"{soil} synthetic RMSE", computed_rmse,
                 target_rmse, 0.015):
            passed += 1
        else:
            failed += 1

    return passed, failed


def main():
    benchmark_path = Path(__file__).parent / "benchmark_dong2024.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_failed = 0

    print("=" * 70)
    print("airSpring Exp 003: IoT Irrigation Pipeline Baseline (Python)")
    print("  Dong et al. (2024) Frontiers in Water 6, 1353597")
    print("=" * 70)

    for validator in [
        validate_soilwatch10,
        validate_irrigation_model,
        validate_sensor_performance,
        validate_field_demonstrations,
        validate_statistics_with_synthetic,
    ]:
        p, f_ = validator(benchmark)
        total_passed += p
        total_failed += f_

    total = total_passed + total_failed
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print("=" * 70)
    print("\nNote: ANOVA analysis is in anova_irrigation.R (R v4.3.1)")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
