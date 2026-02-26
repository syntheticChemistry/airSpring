#!/usr/bin/env python3
"""
airSpring Experiment 004 — FAO-56 Soil Water Balance Baseline

Replicates the daily soil water balance model from:
  FAO Irrigation and Drainage Paper No. 56, Chapter 8
  (Allen et al. 1998)

This is the standard irrigation scheduling method used by Dong et al.
in their IoT-based precision irrigation system.

Implements:
  1. Daily root zone depletion tracking (FAO-56 Eq. 85)
  2. Stress coefficient Ks (FAO-56 Eq. 84)
  3. Adjusted crop ET: ETc_adj = Ks * Kc * ET0
  4. Mass balance verification: inflow = outflow + storage_change
  5. Michigan summer scenario with synthetic weather

All open-source: numpy only.

Provenance:
  Baseline commit: 94cc51d
  Benchmark output: control/water_balance/benchmark_water_balance.json
  Reproduction: python control/water_balance/fao56_water_balance.py
  Created: 2026-02-16
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# FAO-56 Chapter 8 — Daily Soil Water Balance
# ---------------------------------------------------------------------------

def total_available_water(theta_fc: float, theta_wp: float,
                           root_depth_m: float) -> float:
    """FAO-56: TAW = 1000 (θFC - θWP) Zr  [mm]"""
    return 1000.0 * (theta_fc - theta_wp) * root_depth_m


def readily_available_water(TAW: float, p: float) -> float:
    """FAO-56: RAW = p × TAW  [mm]"""
    return p * TAW


def stress_coefficient(Dr: float, TAW: float, RAW: float) -> float:
    """
    FAO-56 Eq. 84: Water stress coefficient
      Ks = (TAW - Dr) / (TAW - RAW)  when Dr > RAW
      Ks = 1.0                        when Dr ≤ RAW
    Clamped to [0, 1].
    """
    if Dr <= RAW:
        return 1.0
    if TAW <= RAW:
        return 0.0
    ks = (TAW - Dr) / (TAW - RAW)
    return max(0.0, min(1.0, ks))


def daily_water_balance_step(Dr_prev: float, P: float, I: float,
                              ET0: float, Kc: float, Ks: float,
                              TAW: float) -> dict:
    """
    FAO-56 Eq. 85: Daily soil water balance

      Dr_i = Dr_{i-1} - (P_i - RO_i) - I_i - CR_i + ETc_adj_i + DP_i

    Simplified assumptions (following FAO-56 defaults):
      - RO = 0 (no surface runoff for well-drained fields)
      - CR = 0 (no capillary rise from shallow water table)
      - DP occurs when Dr would go negative (excess water drains)
    """
    ETc_adj = Ks * Kc * ET0

    # Apply water balance
    Dr_new = Dr_prev - P - I + ETc_adj

    # Deep percolation: if Dr goes negative, excess water drains
    DP = 0.0
    if Dr_new < 0:
        DP = -Dr_new
        Dr_new = 0.0

    # Cannot exceed TAW (fully depleted)
    if Dr_new > TAW:
        Dr_new = TAW

    return {
        "Dr": Dr_new,
        "ETc_adj": ETc_adj,
        "Ks": Ks,
        "DP": DP,
        "P_eff": P,  # effective precip (RO=0)
        "I": I,
    }


def simulate_season(et0_series: np.ndarray, precip_series: np.ndarray,
                     Kc: float, theta_fc: float, theta_wp: float,
                     root_depth_m: float, p: float,
                     irrigation_trigger: bool = False,
                     irrig_depth_mm: float = 25.0) -> dict:
    """
    Run full season water balance simulation.

    Returns dict with daily arrays and summary statistics.
    """
    n_days = len(et0_series)
    TAW = total_available_water(theta_fc, theta_wp, root_depth_m)
    RAW = readily_available_water(TAW, p)

    # Initialize at field capacity (Dr = 0)
    Dr = 0.0

    # Output arrays
    Dr_arr = np.zeros(n_days)
    Ks_arr = np.zeros(n_days)
    ETc_arr = np.zeros(n_days)
    DP_arr = np.zeros(n_days)
    I_arr = np.zeros(n_days)

    total_irrig = 0.0
    irrig_events = 0

    for i in range(n_days):
        # Compute stress coefficient
        Ks = stress_coefficient(Dr, TAW, RAW)

        # Decide irrigation
        I = 0.0
        if irrigation_trigger and Dr > RAW:
            I = min(Dr, irrig_depth_mm)
            total_irrig += I
            irrig_events += 1

        # Water balance step
        result = daily_water_balance_step(
            Dr, precip_series[i], I, et0_series[i], Kc, Ks, TAW)

        Dr = result["Dr"]
        Dr_arr[i] = Dr
        Ks_arr[i] = result["Ks"]
        ETc_arr[i] = result["ETc_adj"]
        DP_arr[i] = result["DP"]
        I_arr[i] = result["I"]

    return {
        "Dr": Dr_arr,
        "Ks": Ks_arr,
        "ETc": ETc_arr,
        "DP": DP_arr,
        "I": I_arr,
        "TAW": TAW,
        "RAW": RAW,
        "total_et": np.sum(ETc_arr),
        "total_precip": np.sum(precip_series),
        "total_dp": np.sum(DP_arr),
        "total_irrig": total_irrig,
        "irrig_events": irrig_events,
        "initial_Dr": 0.0,
        "final_Dr": Dr_arr[-1],
    }


def mass_balance_check(result: dict) -> float:
    """
    Verify mass conservation: inflow = outflow + storage_change

    Inflow  = P + I
    Outflow = ET + DP
    Storage change = initial_Dr - final_Dr  (decrease in depletion = water added)
    """
    inflow = result["total_precip"] + result["total_irrig"]
    outflow = result["total_et"] + result["total_dp"]
    storage_change = result["initial_Dr"] - result["final_Dr"]
    return abs(inflow - outflow - storage_change)


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

def check(label: str, computed: float, expected: float, tol: float) -> bool:
    diff = abs(computed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  [{status}] {label}: {computed:.6f} "
          f"(expected {expected:.6f}, tol {tol:.6f})")
    return diff <= tol


def check_bool(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def validate_taw_raw(benchmark: dict) -> tuple:
    passed = 0
    failed = 0

    print("\n=== TAW and RAW Calculations ===")
    soils = benchmark["soil_parameters"]
    crops = benchmark["crop_parameters"]

    # Sandy loam + corn
    sl = soils["sandy_loam"]
    corn = crops["corn"]
    taw = total_available_water(sl["theta_fc"], sl["theta_wp"],
                                 corn["root_depth_m"])
    raw = readily_available_water(taw, corn["depletion_fraction_p"])

    expected_taw = 1000 * (0.18 - 0.08) * 0.90
    if check("TAW (sandy loam, corn)", taw, expected_taw, 0.01):
        passed += 1
    else:
        failed += 1

    expected_raw = expected_taw * 0.55
    if check("RAW (sandy loam, corn)", raw, expected_raw, 0.01):
        passed += 1
    else:
        failed += 1

    # Loam + tomato
    loam = soils["loam"]
    tomato = crops["tomato"]
    taw2 = total_available_water(loam["theta_fc"], loam["theta_wp"],
                                  tomato["root_depth_m"])
    expected_taw2 = 1000 * (0.27 - 0.12) * 0.60
    if check("TAW (loam, tomato)", taw2, expected_taw2, 0.01):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_stress_coefficient(benchmark: dict) -> tuple:
    passed = 0
    failed = 0

    print("\n=== Stress Coefficient Ks ===")
    TAW = 90.0
    RAW = 49.5

    # No stress (Dr = 0)
    if check("Ks at field capacity (Dr=0)", stress_coefficient(0, TAW, RAW),
             1.0, 1e-10):
        passed += 1
    else:
        failed += 1

    # No stress (Dr = RAW)
    if check("Ks at RAW boundary (Dr=RAW)",
             stress_coefficient(RAW, TAW, RAW), 1.0, 1e-10):
        passed += 1
    else:
        failed += 1

    # Partial stress (Dr = 70, between RAW and TAW)
    expected_ks = (TAW - 70) / (TAW - RAW)
    if check("Ks at Dr=70 (partial stress)",
             stress_coefficient(70, TAW, RAW), expected_ks, 1e-10):
        passed += 1
    else:
        failed += 1

    # Full stress (Dr = TAW)
    if check("Ks at wilting point (Dr=TAW)",
             stress_coefficient(TAW, TAW, RAW), 0.0, 1e-10):
        passed += 1
    else:
        failed += 1

    # Over-depleted
    if check("Ks over-depleted (Dr>TAW)",
             stress_coefficient(100, TAW, RAW), 0.0, 1e-10):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_mass_balance_dry(benchmark: dict) -> tuple:
    """Validate mass conservation under dry-down scenario (no rain, no irrig)."""
    passed = 0
    failed = 0

    print("\n=== Mass Balance — Dry-Down Scenario ===")
    n_days = 30
    et0 = np.full(n_days, 5.0)
    precip = np.zeros(n_days)

    result = simulate_season(
        et0, precip, Kc=1.2,
        theta_fc=0.18, theta_wp=0.08,
        root_depth_m=0.90, p=0.55,
        irrigation_trigger=False,
    )

    mb_error = mass_balance_check(result)
    if check("Mass balance error (dry-down)", mb_error, 0.0,
             benchmark["mass_balance_test"]["tolerance"]):
        passed += 1
    else:
        failed += 1

    # Ks should decrease over time
    if check_bool("Ks decreases during dry-down",
                  result["Ks"][-1] < result["Ks"][0]):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_mass_balance_irrigated(benchmark: dict) -> tuple:
    """Validate mass conservation with irrigation triggers."""
    passed = 0
    failed = 0

    print("\n=== Mass Balance — Irrigated Scenario ===")
    n_days = 60
    et0 = np.full(n_days, 5.0)
    precip = np.zeros(n_days)

    result = simulate_season(
        et0, precip, Kc=1.2,
        theta_fc=0.18, theta_wp=0.08,
        root_depth_m=0.90, p=0.55,
        irrigation_trigger=True,
        irrig_depth_mm=25.0,
    )

    mb_error = mass_balance_check(result)
    if check("Mass balance error (irrigated)", mb_error, 0.0,
             benchmark["mass_balance_test"]["tolerance"]):
        passed += 1
    else:
        failed += 1

    if check_bool(f"Irrigation events triggered: {result['irrig_events']}",
                  result["irrig_events"] > 0):
        passed += 1
    else:
        failed += 1

    # Ks should stay higher with irrigation
    if check_bool("Mean Ks > 0.5 with irrigation",
                  np.mean(result["Ks"]) > 0.5):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_michigan_summer(benchmark: dict) -> tuple:
    """Simulate a Michigan summer with synthetic weather."""
    passed = 0
    failed = 0

    print("\n=== Michigan Summer Scenario (90 days) ===")
    scenario = benchmark["michigan_summer_scenario"]
    rng = np.random.default_rng(42)

    n_days = 90
    et0 = rng.normal(scenario["et0_mean_mm_day"],
                      scenario["et0_std_mm_day"], n_days)
    et0 = np.maximum(et0, 0.5)

    rain_days = rng.random(n_days) < scenario["precip_prob"]
    precip = np.zeros(n_days)
    precip[rain_days] = rng.exponential(
        scenario["precip_depth_when_rain_mm"],
        np.sum(rain_days))

    sl = benchmark["soil_parameters"]["sandy_loam"]
    corn = benchmark["crop_parameters"]["corn"]

    result = simulate_season(
        et0, precip, Kc=scenario["kc"],
        theta_fc=sl["theta_fc"], theta_wp=sl["theta_wp"],
        root_depth_m=corn["root_depth_m"], p=corn["depletion_fraction_p"],
        irrigation_trigger=True,
        irrig_depth_mm=25.0,
    )

    mb_error = mass_balance_check(result)
    if check("Mass balance error (MI summer)", mb_error, 0.0,
             benchmark["mass_balance_test"]["tolerance"]):
        passed += 1
    else:
        failed += 1

    et_range = scenario["expected_seasonal_et_range_mm"]
    total_et = result["total_et"]
    if check_bool(f"Seasonal ET={total_et:.0f} mm in expected range "
                  f"[{et_range[0]}, {et_range[1]}]",
                  et_range[0] <= total_et <= et_range[1]):
        passed += 1
    else:
        failed += 1

    if check_bool(f"Irrigation events: {result['irrig_events']}",
                  result["irrig_events"] > 0):
        passed += 1
    else:
        failed += 1

    print(f"\n  Summary:")
    print(f"    Total ET:     {total_et:.1f} mm")
    print(f"    Total precip: {result['total_precip']:.1f} mm")
    print(f"    Total irrig:  {result['total_irrig']:.1f} mm")
    print(f"    Total DP:     {result['total_dp']:.1f} mm")
    print(f"    Irrig events: {result['irrig_events']}")
    print(f"    Final Dr:     {result['final_Dr']:.1f} mm")
    print(f"    Mean Ks:      {np.mean(result['Ks']):.3f}")

    return passed, failed


def validate_mass_balance_with_rain(benchmark: dict) -> tuple:
    """Validate mass balance under heavy rain scenario."""
    passed = 0
    failed = 0

    print("\n=== Mass Balance — Heavy Rain Scenario ===")
    n_days = 14
    et0 = np.full(n_days, 3.0)
    precip = np.zeros(n_days)
    precip[2] = 50.0  # 50 mm rain event
    precip[7] = 30.0

    result = simulate_season(
        et0, precip, Kc=1.0,
        theta_fc=0.27, theta_wp=0.12,
        root_depth_m=0.60, p=0.40,
        irrigation_trigger=False,
    )

    mb_error = mass_balance_check(result)
    if check("Mass balance error (heavy rain)", mb_error, 0.0,
             benchmark["mass_balance_test"]["tolerance"]):
        passed += 1
    else:
        failed += 1

    if check_bool(f"Deep percolation occurred: {result['total_dp']:.1f} mm",
                  result["total_dp"] > 0):
        passed += 1
    else:
        failed += 1

    return passed, failed


def main():
    benchmark_path = Path(__file__).parent / "benchmark_water_balance.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_failed = 0

    print("=" * 70)
    print("airSpring Exp 004: FAO-56 Water Balance Baseline Validation")
    print("  FAO Paper 56, Chapter 8 (Allen et al. 1998)")
    print("=" * 70)

    for validator in [
        validate_taw_raw,
        validate_stress_coefficient,
        validate_mass_balance_dry,
        validate_mass_balance_irrigated,
        validate_michigan_summer,
        validate_mass_balance_with_rain,
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
