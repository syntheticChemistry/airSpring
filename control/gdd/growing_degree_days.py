#!/usr/bin/env python3
"""
Exp 022: Growing Degree Days (GDD) + Crop Phenology — Python Control Baseline

Implements Growing Degree Days (GDD) accumulation and crop phenological stage
determination. GDD links thermal time to crop development, enabling dynamic
Kc assignment instead of fixed calendar dates.

References:
    McMaster GS, Wilhelm WW (1997) "Growing degree-days: one equation, two
    interpretations." Agricultural and Forest Meteorology 87:291-300.

    FAO-56 Allen et al. (1998) Table 11 — crop coefficients and growth stages.

Equations:
    Method 1 (avg):   GDD = max(0, (Tmax + Tmin)/2 − Tbase)
    Method 2 (clamp): GDD = (min(Tmax, Tceil) + max(Tmin, Tbase))/2 − Tbase

Open data: Same Open-Meteo ERA5 stations as other experiments.
"""

import json
import math
import os
import sys

# Crop base temperatures (°C) from the literature
CROP_PARAMS = {
    "corn": {"tbase": 10.0, "tceil": 30.0, "maturity_gdd": 2700.0,
             "kc_stages_gdd": [0, 200, 800, 2200, 2700],
             "kc_values": [0.30, 0.30, 1.20, 1.20, 0.60]},
    "winter_wheat": {"tbase": 0.0, "tceil": 30.0, "maturity_gdd": 2100.0,
                     "kc_stages_gdd": [0, 160, 700, 1700, 2100],
                     "kc_values": [0.40, 0.40, 1.15, 1.15, 0.25]},
    "soybean": {"tbase": 10.0, "tceil": 30.0, "maturity_gdd": 2600.0,
                "kc_stages_gdd": [0, 200, 900, 2100, 2600],
                "kc_values": [0.40, 0.40, 1.15, 1.15, 0.50]},
    "alfalfa": {"tbase": 5.0, "tceil": 30.0, "maturity_gdd": 800.0,
                "kc_stages_gdd": [0, 100, 300, 650, 800],
                "kc_values": [0.40, 0.40, 1.20, 1.20, 1.05]},
}

# East Lansing 2023 daily temps (growing season Apr-Oct from Open-Meteo ERA5)
# Sampled 214 days: April 1 (DOY 91) through October 31 (DOY 304)
# Using representative monthly profiles for deterministic testing
EAST_LANSING_MONTHLY = {
    "apr": {"tmax": 13.5, "tmin": 1.5, "days": 30},
    "may": {"tmax": 21.0, "tmin": 8.5, "days": 31},
    "jun": {"tmax": 27.5, "tmin": 14.5, "days": 30},
    "jul": {"tmax": 30.0, "tmin": 17.0, "days": 31},
    "aug": {"tmax": 28.5, "tmin": 15.5, "days": 31},
    "sep": {"tmax": 24.5, "tmin": 11.5, "days": 30},
    "oct": {"tmax": 16.0, "tmin": 3.5, "days": 31},
}


def gdd_avg(tmax, tmin, tbase):
    """Method 1: Simple average method (McMaster & Wilhelm 1997)."""
    return max(0.0, (tmax + tmin) / 2.0 - tbase)


def gdd_clamp(tmax, tmin, tbase, tceil):
    """Method 2: Clamped/modified method (handles extremes)."""
    tmax_c = min(tmax, tceil)
    tmin_c = max(tmin, tbase)
    if tmin_c > tmax_c:
        tmin_c = tmax_c
    return max(0.0, (tmax_c + tmin_c) / 2.0 - tbase)


def accumulated_gdd(daily_tmax, daily_tmin, tbase, tceil=None, method="avg"):
    """Accumulate GDD over a season, returning cumulative GDD array."""
    cum = []
    total = 0.0
    for tmax, tmin in zip(daily_tmax, daily_tmin):
        if method == "avg":
            daily = gdd_avg(tmax, tmin, tbase)
        else:
            daily = gdd_clamp(tmax, tmin, tbase, tceil or 30.0)
        total += daily
        cum.append(total)
    return cum


def phenological_stage(cum_gdd, kc_stages_gdd, kc_values):
    """Determine crop coefficient from cumulative GDD using stage thresholds."""
    for i in range(len(kc_stages_gdd) - 1):
        if cum_gdd <= kc_stages_gdd[i + 1]:
            frac = (cum_gdd - kc_stages_gdd[i]) / max(1.0, kc_stages_gdd[i + 1] - kc_stages_gdd[i])
            return kc_values[i] + frac * (kc_values[i + 1] - kc_values[i])
    return kc_values[-1]


def generate_season_data():
    """Generate daily temperature arrays from monthly profiles."""
    tmax_list = []
    tmin_list = []
    for month_data in EAST_LANSING_MONTHLY.values():
        for _ in range(month_data["days"]):
            tmax_list.append(month_data["tmax"])
            tmin_list.append(month_data["tmin"])
    return tmax_list, tmin_list


def run_validation():
    """Run all GDD validation checks."""
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            print(f"  [{status}] {name}: {detail}")
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 60)
    print("  Exp 022: Growing Degree Days (GDD) Validation")
    print("=" * 60)

    # --- Section 1: Analytical tests ---
    print("\n--- Analytical Tests ---")

    # Known values: avg method
    check("gdd_avg_warm", abs(gdd_avg(30.0, 20.0, 10.0) - 15.0) < 1e-10)
    check("gdd_avg_cold", gdd_avg(8.0, 2.0, 10.0) == 0.0)
    check("gdd_avg_boundary", abs(gdd_avg(20.0, 10.0, 15.0) - 0.0) < 1e-10)
    check("gdd_avg_exact", abs(gdd_avg(30.0, 10.0, 10.0) - 10.0) < 1e-10)

    # Clamp method
    check("gdd_clamp_warm", abs(gdd_clamp(30.0, 20.0, 10.0, 30.0) - 15.0) < 1e-10)
    check("gdd_clamp_above_ceil",
          abs(gdd_clamp(40.0, 20.0, 10.0, 30.0) - 15.0) < 1e-10)
    check("gdd_clamp_below_base",
          abs(gdd_clamp(20.0, 5.0, 10.0, 30.0) - 5.0) < 1e-10)
    check("gdd_clamp_both_extreme",
          abs(gdd_clamp(40.0, 0.0, 10.0, 30.0) - 10.0) < 1e-10)
    check("gdd_clamp_cold", gdd_clamp(8.0, 2.0, 10.0, 30.0) == 0.0)

    # --- Section 2: Accumulation tests ---
    print("\n--- Accumulation Tests ---")

    # Constant temperature
    n_days = 100
    tmax_const = [30.0] * n_days
    tmin_const = [20.0] * n_days
    cum = accumulated_gdd(tmax_const, tmin_const, 10.0)
    check("constant_accumulation",
          abs(cum[-1] - 1500.0) < 1e-8,
          f"expected 1500, got {cum[-1]}")
    check("constant_monotonic", all(cum[i] <= cum[i+1] for i in range(len(cum)-1)))

    # Zero GDD days
    cum_cold = accumulated_gdd([8.0]*30, [2.0]*30, 10.0)
    check("cold_season_zero", cum_cold[-1] == 0.0)

    # --- Section 3: Seasonal pattern (East Lansing corn) ---
    print("\n--- Seasonal Pattern (East Lansing corn) ---")

    tmax, tmin = generate_season_data()
    cum_corn = accumulated_gdd(tmax, tmin, 10.0)
    total_corn_gdd = cum_corn[-1]

    check("corn_total_gdd_positive", total_corn_gdd > 0,
          f"total={total_corn_gdd:.0f}")

    # Michigan growing season: 1400-2200 GDD₁₀ (continental)
    check("corn_gdd_range", 1400 < total_corn_gdd < 2200,
          f"total={total_corn_gdd:.0f}")

    # GDD always non-decreasing
    check("corn_monotonic", all(cum_corn[i] <= cum_corn[i+1] for i in range(len(cum_corn)-1)))

    # July contributes most
    apr_days = 30
    may_days = 31
    jun_days = 30
    jul_start = apr_days + may_days + jun_days
    jul_end = jul_start + 31
    jul_gdd = cum_corn[jul_end - 1] - cum_corn[jul_start - 1]
    apr_gdd = cum_corn[apr_days - 1]
    check("july_gt_april_gdd", jul_gdd > apr_gdd,
          f"Jul={jul_gdd:.0f}, Apr={apr_gdd:.0f}")

    # --- Section 4: Multi-crop comparison ---
    print("\n--- Multi-Crop Comparison ---")

    for crop_name, params in CROP_PARAMS.items():
        cum = accumulated_gdd(tmax, tmin, params["tbase"], params["tceil"], "clamp")
        total = cum[-1]
        check(f"{crop_name}_gdd_positive", total > 0,
              f"total={total:.0f}")

    # Alfalfa (tbase=5) should accumulate more than corn (tbase=10) same data
    cum_alfalfa = accumulated_gdd(tmax, tmin, 5.0)
    cum_corn_same = accumulated_gdd(tmax, tmin, 10.0)
    check("alfalfa_gt_corn_gdd",
          cum_alfalfa[-1] > cum_corn_same[-1],
          f"alfalfa={cum_alfalfa[-1]:.0f}, corn={cum_corn_same[-1]:.0f}")

    # --- Section 5: Phenological stage mapping ---
    print("\n--- Phenological Stage (Kc from GDD) ---")

    corn_p = CROP_PARAMS["corn"]

    # At planting (GDD=0): Kc_ini
    kc_0 = phenological_stage(0, corn_p["kc_stages_gdd"], corn_p["kc_values"])
    check("corn_kc_planting", abs(kc_0 - 0.30) < 0.01,
          f"Kc={kc_0:.2f}")

    # At mid-season (GDD=1500): Kc_mid
    kc_mid = phenological_stage(1500, corn_p["kc_stages_gdd"], corn_p["kc_values"])
    check("corn_kc_midseason", abs(kc_mid - 1.20) < 0.01,
          f"Kc={kc_mid:.2f}")

    # At harvest (GDD=2700): Kc_end
    kc_end = phenological_stage(2700, corn_p["kc_stages_gdd"], corn_p["kc_values"])
    check("corn_kc_harvest", abs(kc_end - 0.60) < 0.01,
          f"Kc={kc_end:.2f}")

    # Mid-development (GDD=500): interpolating
    kc_dev = phenological_stage(500, corn_p["kc_stages_gdd"], corn_p["kc_values"])
    check("corn_kc_developing", 0.30 < kc_dev < 1.20,
          f"Kc={kc_dev:.2f}")

    # Kc monotonic through development
    gdd_series = list(range(0, 2700, 100))
    kc_series = [phenological_stage(g, corn_p["kc_stages_gdd"], corn_p["kc_values"]) for g in gdd_series]
    # Should increase from ini to mid, then decrease from mid to end
    mid_idx = gdd_series.index(800)
    increasing = all(kc_series[i] <= kc_series[i+1] + 0.001 for i in range(mid_idx))
    check("kc_increasing_to_mid", increasing)

    # --- Section 6: Method comparison (avg vs clamp) ---
    print("\n--- Method Comparison (avg vs clamp) ---")

    cum_avg = accumulated_gdd(tmax, tmin, 10.0, method="avg")
    cum_clamp = accumulated_gdd(tmax, tmin, 10.0, 30.0, method="clamp")

    # Clamp raises Tmin to Tbase on cold days and caps Tmax at Tceil on hot days.
    # In moderate climates with cool mornings, clamp can exceed avg.
    check("clamp_vs_avg_both_valid",
          cum_clamp[-1] > 0.0 and cum_avg[-1] > 0.0,
          f"clamp={cum_clamp[-1]:.0f}, avg={cum_avg[-1]:.0f}")

    # Both should be close when no extremes
    diff = abs(cum_avg[-1] - cum_clamp[-1])
    check("methods_close_moderate_climate", diff < 200.0,
          f"diff={diff:.1f}")

    # --- Section 7: Edge cases ---
    print("\n--- Edge Cases ---")

    check("single_day_gdd", abs(gdd_avg(25.0, 15.0, 10.0) - 10.0) < 1e-10)
    check("extreme_cold", gdd_avg(-20.0, -30.0, 10.0) == 0.0)
    check("extreme_hot", abs(gdd_avg(50.0, 30.0, 10.0) - 30.0) < 1e-10)
    check("tmax_eq_tmin", abs(gdd_avg(20.0, 20.0, 10.0) - 10.0) < 1e-10)
    check("clamp_tmax_at_ceil", abs(gdd_clamp(35.0, 20.0, 10.0, 30.0) - 15.0) < 1e-10)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"  Growing Degree Days: {passed}/{total} PASS, {failed}/{total} FAIL")
    print(f"{'=' * 60}")

    return failed == 0


def generate_benchmark():
    """Generate benchmark JSON for Rust validation."""
    tmax, tmin = generate_season_data()

    benchmark = {
        "_provenance": {
            "method": "Growing Degree Days (McMaster & Wilhelm 1997)",
            "baseline_script": "control/gdd/growing_degree_days.py",
            "baseline_commit": "pending",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "created": "2026-02-26",
            "references": [
                "McMaster GS, Wilhelm WW (1997) Ag Forest Met 87:291-300",
                "FAO-56 Allen et al. (1998) Table 11",
            ],
        },
        "analytical": {
            "gdd_avg": [
                {"tmax": 30.0, "tmin": 20.0, "tbase": 10.0, "expected": 15.0},
                {"tmax": 8.0, "tmin": 2.0, "tbase": 10.0, "expected": 0.0},
                {"tmax": 20.0, "tmin": 10.0, "tbase": 10.0, "expected": 5.0},
                {"tmax": 30.0, "tmin": 10.0, "tbase": 10.0, "expected": 10.0},
                {"tmax": 25.0, "tmin": 15.0, "tbase": 10.0, "expected": 10.0},
                {"tmax": -20.0, "tmin": -30.0, "tbase": 10.0, "expected": 0.0},
                {"tmax": 50.0, "tmin": 30.0, "tbase": 10.0, "expected": 30.0},
                {"tmax": 20.0, "tmin": 20.0, "tbase": 10.0, "expected": 10.0},
            ],
            "gdd_clamp": [
                {"tmax": 30.0, "tmin": 20.0, "tbase": 10.0, "tceil": 30.0, "expected": 15.0},
                {"tmax": 40.0, "tmin": 20.0, "tbase": 10.0, "tceil": 30.0, "expected": 15.0},
                {"tmax": 20.0, "tmin": 5.0, "tbase": 10.0, "tceil": 30.0, "expected": 5.0},
                {"tmax": 40.0, "tmin": 0.0, "tbase": 10.0, "tceil": 30.0, "expected": 10.0},
                {"tmax": 8.0, "tmin": 2.0, "tbase": 10.0, "tceil": 30.0, "expected": 0.0},
                {"tmax": 35.0, "tmin": 20.0, "tbase": 10.0, "tceil": 30.0, "expected": 15.0},
            ],
        },
        "crop_params": {k: v for k, v in CROP_PARAMS.items()},
        "east_lansing_season": {
            "monthly_profiles": EAST_LANSING_MONTHLY,
            "total_days": len(tmax),
        },
        "accumulation": {
            "corn_avg_total_gdd": round(accumulated_gdd(tmax, tmin, 10.0)[-1], 4),
            "corn_clamp_total_gdd": round(accumulated_gdd(tmax, tmin, 10.0, 30.0, "clamp")[-1], 4),
            "alfalfa_avg_total_gdd": round(accumulated_gdd(tmax, tmin, 5.0)[-1], 4),
            "tol": 0.01,
        },
        "phenology": {
            "corn_kc_at_gdd": [
                {"gdd": 0, "expected_kc": 0.30},
                {"gdd": 500, "expected_kc": round(phenological_stage(500, CROP_PARAMS["corn"]["kc_stages_gdd"], CROP_PARAMS["corn"]["kc_values"]), 6)},
                {"gdd": 1500, "expected_kc": 1.20},
                {"gdd": 2700, "expected_kc": 0.60},
            ],
            "tol": 0.01,
        },
        "thresholds": {
            "corn_gdd_range": [1400.0, 2200.0],
            "method_diff_max": 200.0,
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_gdd.json")
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  Benchmark written to {out_path}")

    return benchmark


if __name__ == "__main__":
    benchmark = generate_benchmark()
    success = run_validation()
    sys.exit(0 if success else 1)
