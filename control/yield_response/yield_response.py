#!/usr/bin/env python3
"""
airSpring Experiment 008 — FAO-56 Yield Response to Water Stress

Implements the Stewart (1977) yield response model from:
  FAO Irrigation and Drainage Paper No. 56, Chapter 10
  (Allen et al. 1998)

Key equations:
  Single-stage: Ya/Ymax = 1 - Ky * (1 - ETa/ETc)          (Stewart 1977)
  Multi-stage:  Ya/Ymax = prod_i (1 - Kyi * (1 - ETai/ETci))  (FAO-56 Eq. 90)
  WUE:          WUE = Ya / ETa                              [kg/m³]

Ky values from FAO-56 Table 24 (Doorenbos & Kassam 1979).

Provenance:
  Benchmark output: control/yield_response/benchmark_yield_response.json
  Reproduction: python control/yield_response/yield_response.py
  Created: 2026-02-25
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Stewart (1977) yield response model
# ---------------------------------------------------------------------------

def yield_ratio_single(ky: float, eta_etc_ratio: float) -> float:
    """
    Single-season yield response.

    Ya/Ymax = 1 - Ky * (1 - ETa/ETc)

    Args:
        ky: Yield response factor (dimensionless)
        eta_etc_ratio: Ratio of actual to potential ET (0 to 1)

    Returns:
        Yield ratio Ya/Ymax (can be negative for extreme stress with high Ky)
    """
    return 1.0 - ky * (1.0 - eta_etc_ratio)


def yield_ratio_multistage(stages_ky: list, stages_eta_etc: list) -> float:
    """
    Multi-stage yield response (FAO-56 Eq. 90).

    Ya/Ymax = prod_i (1 - Kyi * (1 - ETai/ETci))

    Args:
        stages_ky: Ky values per growth stage
        stages_eta_etc: ETa/ETc ratios per growth stage

    Returns:
        Yield ratio Ya/Ymax
    """
    ratio = 1.0
    for ky, eta_etc in zip(stages_ky, stages_eta_etc):
        ratio *= (1.0 - ky * (1.0 - eta_etc))
    return ratio


def water_use_efficiency(yield_kg_ha: float, eta_mm: float) -> float:
    """
    Water use efficiency.

    WUE = yield / ETa  [kg/m³]

    1 mm over 1 ha = 10 m³, so:
    WUE [kg/m³] = yield_kg_ha / (eta_mm * 10)

    Args:
        yield_kg_ha: Crop yield in kg/ha
        eta_mm: Actual ET over the season in mm

    Returns:
        WUE in kg/m³
    """
    if eta_mm <= 0:
        return 0.0
    return yield_kg_ha / (eta_mm * 10.0)


# ---------------------------------------------------------------------------
# FAO-56 water balance integration (simplified for scheduling comparison)
# ---------------------------------------------------------------------------

def total_available_water(theta_fc: float, theta_wp: float,
                          root_depth_m: float) -> float:
    """FAO-56: TAW = 1000 (θFC - θWP) Zr  [mm]"""
    return 1000.0 * (theta_fc - theta_wp) * root_depth_m


def readily_available_water(taw: float, p: float) -> float:
    """FAO-56: RAW = p × TAW  [mm]"""
    return p * taw


def stress_coefficient(dr: float, taw: float, raw: float) -> float:
    """FAO-56 Eq. 84: Ks = (TAW-Dr)/(TAW-RAW) when Dr>RAW, else 1.0"""
    if dr <= raw:
        return 1.0
    if taw <= raw:
        return 0.0
    ks = (taw - dr) / (taw - raw)
    return max(0.0, min(1.0, ks))


def simulate_season_with_yield(
    season_days: int,
    et0_daily: np.ndarray,
    precip_daily: np.ndarray,
    kc: float,
    ky_total: float,
    theta_fc: float,
    theta_wp: float,
    root_depth_m: float,
    p: float,
    irrigation_threshold_frac: float = None,
    irrigation_depth_mm: float = 25.0,
) -> dict:
    """
    Run a full-season water balance with yield response.

    Returns dict with daily arrays and season summary including yield ratio.
    """
    taw = total_available_water(theta_fc, theta_wp, root_depth_m)
    raw = readily_available_water(taw, p)

    dr = 0.0
    total_eta = 0.0
    total_etc = 0.0
    total_irrig = 0.0
    stress_days = 0

    for day in range(season_days):
        et0 = et0_daily[day] if day < len(et0_daily) else 5.0
        prec = precip_daily[day] if day < len(precip_daily) else 0.0

        ks = stress_coefficient(dr, taw, raw)
        if ks < 1.0:
            stress_days += 1

        etc = kc * et0
        eta = ks * etc

        irrig = 0.0
        if irrigation_threshold_frac is not None and dr > irrigation_threshold_frac * taw:
            irrig = irrigation_depth_mm

        dr = dr - prec - irrig + eta
        dr = max(0.0, min(dr, taw))

        total_eta += eta
        total_etc += etc
        total_irrig += irrig

    eta_etc_ratio = total_eta / total_etc if total_etc > 0 else 1.0
    yield_ratio = yield_ratio_single(ky_total, eta_etc_ratio)
    yield_ratio_clamped = max(0.0, yield_ratio)

    return {
        "total_eta_mm": total_eta,
        "total_etc_mm": total_etc,
        "eta_etc_ratio": eta_etc_ratio,
        "yield_ratio": yield_ratio,
        "yield_ratio_clamped": yield_ratio_clamped,
        "stress_days": stress_days,
        "total_irrigation_mm": total_irrig,
    }


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

def check(label: str, computed: float, expected: float, tol: float) -> bool:
    diff = abs(computed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  [{status}] {label}: {computed:.6f} "
          f"(expected {expected:.6f}, tol {tol})")
    return diff <= tol


def check_range(label: str, value: float, low: float, high: float) -> bool:
    ok = low <= value <= high
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {value:.4f} (range [{low}, {high}])")
    return ok


def check_bool(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_ky_table(benchmark: dict) -> tuple:
    """Validate Ky values match FAO-56 Table 24."""
    print("\n--- Ky Table Values (FAO-56 Table 24) ---")
    passed = 0
    failed = 0

    for tc in benchmark["validation_checks"]["ky_table_values"]["test_cases"]:
        crop = tc["crop"]
        expected = tc["ky_total"]
        tol = tc["tolerance"]
        actual = benchmark["ky_values"][crop]["ky_total"]
        if check(f"Ky({crop})", actual, expected, tol):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_single_stage(benchmark: dict) -> tuple:
    """Validate single-stage Stewart equation against analytical values."""
    print("\n--- Single-Stage Yield Response (Stewart 1977) ---")
    passed = 0
    failed = 0

    for tc in benchmark["validation_checks"]["single_stage_analytical"]["test_cases"]:
        label = tc["label"]
        ky = tc["ky"]
        eta_etc = tc["eta_etc"]
        expected = tc["expected_ratio"]
        tol = tc["tolerance"]

        computed = yield_ratio_single(ky, eta_etc)
        if check(label, computed, expected, tol):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_multi_stage(benchmark: dict) -> tuple:
    """Validate multi-stage product formula against analytical values."""
    print("\n--- Multi-Stage Yield Response (FAO-56 Eq. 90) ---")
    passed = 0
    failed = 0

    for tc in benchmark["validation_checks"]["multi_stage_analytical"]["test_cases"]:
        label = tc["label"]
        stages_ky = tc["stages_ky"]
        stages_eta_etc = tc["stages_eta_etc"]
        expected = tc["expected_ratio"]
        tol = tc["tolerance"]

        computed = yield_ratio_multistage(stages_ky, stages_eta_etc)
        if check(label, computed, expected, tol):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_wue(benchmark: dict) -> tuple:
    """Validate water use efficiency calculations."""
    print("\n--- Water Use Efficiency ---")
    passed = 0
    failed = 0

    for tc in benchmark["validation_checks"]["water_use_efficiency"]["test_cases"]:
        label = tc["label"]
        y = tc["yield_kg_ha"]
        eta = tc["eta_mm"]
        expected = tc["expected_wue_kg_m3"]
        tol = tc["tolerance"]

        computed = water_use_efficiency(y, eta)
        if check(label, computed, expected, tol):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_scheduling(benchmark: dict) -> tuple:
    """Validate scheduling strategy yield impact comparison."""
    print("\n--- Scheduling Strategy Comparison (Corn, Michigan) ---")
    passed = 0
    failed = 0

    scenario = benchmark["validation_checks"]["scheduling_comparison"]["scenario"]
    strategies = benchmark["validation_checks"]["scheduling_comparison"]["strategies"]

    season_days = scenario["season_days"]
    np.random.seed(42)

    et0_base = scenario["et0_mean_mm_day"]
    et0_daily = np.maximum(0.5, np.random.normal(et0_base, 1.5, season_days))

    precip_mean = strategies["no_irrigation"]["precip_mean_mm_day"]
    precip_prob = strategies["no_irrigation"]["precip_prob"]
    rain_days = np.random.random(season_days) < precip_prob
    precip_daily = np.where(rain_days,
                            np.random.exponential(precip_mean / precip_prob, season_days),
                            0.0)

    results = {}

    for name, strat in strategies.items():
        thresh_frac = strat.get("irrigation_threshold_frac", None)
        irrig_depth = strat.get("irrigation_depth_mm", 25.0) if thresh_frac else 0.0

        result = simulate_season_with_yield(
            season_days=season_days,
            et0_daily=et0_daily,
            precip_daily=precip_daily,
            kc=1.2,
            ky_total=scenario["ky_total"],
            theta_fc=scenario["theta_fc"],
            theta_wp=scenario["theta_wp"],
            root_depth_m=scenario["root_depth_m"],
            p=scenario["p"],
            irrigation_threshold_frac=thresh_frac,
            irrigation_depth_mm=irrig_depth,
        )
        results[name] = result

        yr_range = strat["expected_yield_ratio_range"]
        sd_range = strat["expected_stress_days_range"]

        yr = result["yield_ratio_clamped"]
        sd = result["stress_days"]

        if check_range(f"{name}_yield_ratio", yr, yr_range[0], yr_range[1]):
            passed += 1
        else:
            failed += 1

        if check_range(f"{name}_stress_days", sd, sd_range[0], sd_range[1]):
            passed += 1
        else:
            failed += 1

    yr_none = results["no_irrigation"]["yield_ratio_clamped"]
    yr_mad = results["threshold_mad"]["yield_ratio_clamped"]
    if check_bool("irrigation_improves_yield (MAD > rainfed)",
                   yr_mad > yr_none):
        passed += 1
    else:
        failed += 1

    sd_none = results["no_irrigation"]["stress_days"]
    sd_mad = results["threshold_mad"]["stress_days"]
    if check_bool("irrigation_reduces_stress (MAD < rainfed)",
                   sd_mad < sd_none):
        passed += 1
    else:
        failed += 1

    return passed, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    benchmark_path = Path(__file__).parent / "benchmark_yield_response.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_failed = 0

    print("=" * 70)
    print("airSpring Exp 008: FAO-56 Yield Response Baseline Validation")
    print("  Stewart (1977) + FAO-56 Chapter 10 + Table 24")
    print("=" * 70)

    for validator in [
        validate_ky_table,
        validate_single_stage,
        validate_multi_stage,
        validate_wue,
        validate_scheduling,
    ]:
        p, f_ = validator(benchmark)
        total_passed += p
        total_failed += f_

    total = total_passed + total_failed
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
