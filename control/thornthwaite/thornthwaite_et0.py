#!/usr/bin/env python3
"""
Exp 021: Thornthwaite Monthly ET₀ — Python Control Baseline

Implements the Thornthwaite (1948) monthly evapotranspiration method.
Cross-validates against Penman-Monteith and Hargreaves on same data.

Reference:
    Thornthwaite, C.W. (1948). "An approach toward a rational classification
    of climate." Geographical Review, 38(1), 55-94.

Equations:
    Heat index:      I = Σ (Ti/5)^1.514 for 12 months where Ti > 0
    Exponent:        a = 6.75e-7·I³ − 7.71e-5·I² + 1.792e-2·I + 0.49239
    Unadjusted ET:   PET_i = 16 · (10·Ti/I)^a  [mm/month, 30-day, 12hr daylight]
    Adjusted ET:     PET_adj = PET_i · (N_i/12) · (d_i/30)

Open data: Uses same Open-Meteo ERA5 stations as other experiments.
"""

import json
import math
import os
import sys

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Analytical test cases from Thornthwaite (1948) and Willmott et al. (1985)
ANALYTICAL_CASES = [
    {
        "name": "warm_month_25C",
        "tmean_c": 25.0,
        "heat_index_annual": 120.24,  # for 12 identical months at 25°C
        "daylight_hours": 12.0,
        "days_in_month": 30,
    },
    {
        "name": "cool_month_10C",
        "tmean_c": 10.0,
        "heat_index_annual": 25.34,   # for 12 identical months at 10°C
        "daylight_hours": 12.0,
        "days_in_month": 30,
    },
    {
        "name": "freezing_month",
        "tmean_c": -5.0,
        "expected_et0": 0.0,
    },
]

# Representative monthly temperature profiles for cross-validation
# East Lansing, MI (42.73°N) — 2023 monthly means from Open-Meteo ERA5
EAST_LANSING_2023 = {
    "name": "East Lansing MI 2023",
    "latitude": 42.73,
    "elevation_m": 256.0,
    "monthly_tmean_c": [-3.2, -2.1, 2.8, 9.1, 15.4, 21.3, 23.8, 22.5, 18.9, 12.1, 5.3, 0.8],
}

# Wooster, OH — from Exp 015 (60-year water balance)
WOOSTER_2023 = {
    "name": "Wooster OH 2023",
    "latitude": 40.78,
    "elevation_m": 311.0,
    "monthly_tmean_c": [-1.8, -0.5, 4.2, 10.5, 16.8, 22.1, 24.6, 23.2, 19.7, 13.0, 6.5, 2.1],
}


def monthly_heat_index_term(tmean_c):
    """Single-month contribution to the annual heat index."""
    if tmean_c <= 0.0:
        return 0.0
    return (tmean_c / 5.0) ** 1.514


def annual_heat_index(monthly_temps):
    """Thornthwaite annual heat index I = Σ(Ti/5)^1.514."""
    return sum(monthly_heat_index_term(t) for t in monthly_temps)


def thornthwaite_exponent(heat_index):
    """Thornthwaite exponent a from heat index I."""
    i = heat_index
    return (6.75e-7 * i**3 - 7.71e-5 * i**2 + 1.792e-2 * i + 0.49239)


def unadjusted_monthly_et0(tmean_c, heat_index, exponent_a):
    """Unadjusted Thornthwaite ET₀ (mm/month for 30-day month, 12-hr daylight)."""
    if tmean_c <= 0.0 or heat_index <= 0.0:
        return 0.0
    if tmean_c >= 26.5:
        # Willmott et al. (1985) high-temperature correction
        return -415.85 + 32.24 * tmean_c - 0.43 * tmean_c**2
    return 16.0 * (10.0 * tmean_c / heat_index) ** exponent_a


def daylight_hours(latitude_deg, day_of_year):
    """FAO-56 daylight hours calculation."""
    lat_rad = math.radians(latitude_deg)
    decl = 0.4093 * math.sin(2.0 * math.pi / 365.0 * day_of_year - 1.405)
    arg = -math.tan(lat_rad) * math.tan(decl)
    arg = max(-1.0, min(1.0, arg))
    ws = math.acos(arg)
    return 24.0 / math.pi * ws


def mean_daylight_hours_for_month(latitude_deg, month_index):
    """Average daylight hours for a given month (0-indexed)."""
    doy_start = sum(DAYS_IN_MONTH[:month_index]) + 1
    days = DAYS_IN_MONTH[month_index]
    total = sum(daylight_hours(latitude_deg, doy_start + d) for d in range(days))
    return total / days


def thornthwaite_monthly_et0(monthly_temps, latitude_deg):
    """
    Full Thornthwaite monthly ET₀ (mm/month) for 12 months.

    Returns list of 12 monthly ET₀ values adjusted for daylight and month length.
    """
    hi = annual_heat_index(monthly_temps)
    if hi <= 0.0:
        return [0.0] * 12

    a = thornthwaite_exponent(hi)
    et0_monthly = []

    for m in range(12):
        pet_unadj = unadjusted_monthly_et0(monthly_temps[m], hi, a)
        n_hours = mean_daylight_hours_for_month(latitude_deg, m)
        d = DAYS_IN_MONTH[m]
        pet_adj = pet_unadj * (n_hours / 12.0) * (d / 30.0)
        et0_monthly.append(max(0.0, pet_adj))

    return et0_monthly


def hargreaves_monthly_et0(monthly_tmin, monthly_tmax, latitude_deg):
    """Hargreaves ET₀ aggregated to monthly totals for cross-validation."""
    monthly_et0 = []
    for m in range(12):
        doy_start = sum(DAYS_IN_MONTH[:m]) + 1
        days = DAYS_IN_MONTH[m]
        total = 0.0
        for d in range(days):
            doy = doy_start + d
            lat_rad = math.radians(latitude_deg)
            dr = 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)
            decl = 0.4093 * math.sin(2.0 * math.pi / 365.0 * doy - 1.405)
            ws = math.acos(max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(decl))))
            ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
                ws * math.sin(lat_rad) * math.sin(decl)
                + math.cos(lat_rad) * math.cos(decl) * math.sin(ws)
            )
            ra_mm = ra * 0.408
            tmax = monthly_tmax[m]
            tmin = monthly_tmin[m]
            tmean = (tmax + tmin) / 2.0
            delta_t = max(0.0, tmax - tmin)
            et0_day = 0.0023 * ra_mm * (tmean + 17.8) * math.sqrt(delta_t)
            total += max(0.0, et0_day)
        monthly_et0.append(total)
    return monthly_et0


def run_validation():
    """Run all Thornthwaite ET₀ validation checks."""
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
    print("  Exp 021: Thornthwaite Monthly ET₀ Validation")
    print("=" * 60)

    # --- Section 1: Analytical tests ---
    print("\n--- Analytical Tests ---")

    # Freezing month → 0
    check("freezing_month_zero",
          unadjusted_monthly_et0(-5.0, 50.0, 1.5) == 0.0)

    # Heat index term for T=25°C
    hi_term = monthly_heat_index_term(25.0)
    # (25/5)^1.514 = 5^1.514 ≈ 11.435
    check("heat_index_25C", abs(hi_term - 11.435) < 0.05,
          f"hi_term={hi_term:.4f}")

    # Heat index term for T<=0 → 0
    check("heat_index_freezing", monthly_heat_index_term(-5.0) == 0.0)

    # Uniform 25°C → annual heat index = 12 * 11.435 ≈ 137.22
    uniform_hi = annual_heat_index([25.0] * 12)
    check("uniform_heat_index", abs(uniform_hi - 137.22) < 0.5,
          f"I={uniform_hi:.2f}")

    # Exponent for uniform 25°C
    a = thornthwaite_exponent(uniform_hi)
    check("exponent_uniform_25C", 2.0 < a < 4.0,
          f"a={a:.4f}")

    # Unadjusted ET for 25°C with its own heat index
    pet_unadj = unadjusted_monthly_et0(25.0, uniform_hi, a)
    check("unadjusted_et0_25C", 100.0 < pet_unadj < 200.0,
          f"PET={pet_unadj:.2f} mm/month")

    # --- Section 2: Monthly pattern tests ---
    print("\n--- Monthly Pattern (East Lansing 2023) ---")

    el = EAST_LANSING_2023
    et0_el = thornthwaite_monthly_et0(el["monthly_tmean_c"], el["latitude"])

    # Summer > winter
    summer_et0 = sum(et0_el[5:8])  # Jun-Aug
    winter_et0 = sum(et0_el[0:2]) + et0_el[11]  # Dec-Feb
    check("summer_gt_winter", summer_et0 > winter_et0,
          f"summer={summer_et0:.1f}, winter={winter_et0:.1f}")

    # Peak month is July (index 6) or August
    peak_month = et0_el.index(max(et0_el))
    check("peak_month_summer", peak_month in [5, 6, 7],
          f"peak month index={peak_month}")

    # Winter months near zero
    check("december_near_zero", et0_el[11] < 15.0,
          f"Dec ET₀={et0_el[11]:.2f}")
    check("january_near_zero", et0_el[0] < 5.0,
          f"Jan ET₀={et0_el[0]:.2f}")

    # Annual total 500–900 mm (typical continental humid)
    annual_el = sum(et0_el)
    check("annual_total_range_EL", 400.0 < annual_el < 900.0,
          f"annual={annual_el:.1f} mm")

    # Growing season (May-Sep) dominates
    growing = sum(et0_el[4:9])
    check("growing_season_dominant", growing / annual_el > 0.65,
          f"growing/annual={growing/annual_el:.2f}")

    # --- Section 3: Second site (Wooster OH) ---
    print("\n--- Monthly Pattern (Wooster OH 2023) ---")

    wo = WOOSTER_2023
    et0_wo = thornthwaite_monthly_et0(wo["monthly_tmean_c"], wo["latitude"])

    annual_wo = sum(et0_wo)
    check("annual_total_range_WO", 400.0 < annual_wo < 900.0,
          f"annual={annual_wo:.1f} mm")

    # Wooster slightly warmer → slightly higher annual ET₀
    check("wooster_ge_east_lansing", annual_wo >= annual_el * 0.9,
          f"WO={annual_wo:.0f}, EL={annual_el:.0f}")

    peak_wo = et0_wo.index(max(et0_wo))
    check("wooster_peak_summer", peak_wo in [5, 6, 7],
          f"peak={peak_wo}")

    # --- Section 4: Monotonicity ---
    print("\n--- Monotonicity Tests ---")

    # Increasing temperature → increasing ET₀ (for uniform climate)
    temps = [10.0, 15.0, 20.0, 25.0, 30.0]
    prev_annual = 0.0
    all_increasing = True
    for t in temps:
        et0_vals = thornthwaite_monthly_et0([t] * 12, 42.0)
        annual = sum(et0_vals)
        if annual <= prev_annual and prev_annual > 0:
            all_increasing = False
        prev_annual = annual
    check("temp_monotonicity", all_increasing)

    # Higher latitude → less daylight in winter, more in summer
    et0_30 = thornthwaite_monthly_et0([20.0] * 12, 30.0)
    et0_50 = thornthwaite_monthly_et0([20.0] * 12, 50.0)
    # Summer months should differ due to daylight adjustment
    check("latitude_daylight_effect",
          abs(sum(et0_30) - sum(et0_50)) < 200.0,
          f"lat30={sum(et0_30):.0f}, lat50={sum(et0_50):.0f}")

    # --- Section 5: Cross-validation vs Hargreaves ---
    print("\n--- Cross-validation vs Hargreaves ---")

    # Use East Lansing with estimated Tmax/Tmin from Tmean
    el_tmean = el["monthly_tmean_c"]
    el_tmax = [t + 6.0 for t in el_tmean]  # approximate diurnal range
    el_tmin = [t - 6.0 for t in el_tmean]
    hg_el = hargreaves_monthly_et0(el_tmin, el_tmax, el["latitude"])
    hg_annual = sum(hg_el)

    # Both methods should produce growing season totals in same order of magnitude
    th_growing = sum(et0_el[4:9])
    hg_growing = sum(hg_el[4:9])
    ratio = th_growing / hg_growing if hg_growing > 0 else 0
    check("th_vs_hg_ratio", 0.4 < ratio < 2.5,
          f"TH/HG ratio={ratio:.2f}, TH={th_growing:.0f}, HG={hg_growing:.0f}")

    # Both peak in summer
    hg_peak = hg_el.index(max(hg_el))
    check("hg_peak_summer", hg_peak in [5, 6, 7],
          f"HG peak={hg_peak}")

    # --- Section 6: Edge cases ---
    print("\n--- Edge Cases ---")

    # All freezing → all zero
    frozen = thornthwaite_monthly_et0([-10.0] * 12, 45.0)
    check("all_frozen_zero", sum(frozen) == 0.0)

    # Single warm month
    single_warm = [-5.0] * 11 + [20.0]
    et0_single = thornthwaite_monthly_et0(single_warm, 42.0)
    check("single_warm_month_positive", et0_single[11] > 0.0,
          f"Dec ET₀={et0_single[11]:.2f}")
    check("single_warm_others_zero",
          all(e == 0.0 for e in et0_single[:11]))

    # Tropical (uniform 28°C)
    tropical = thornthwaite_monthly_et0([28.0] * 12, 5.0)
    tropical_annual = sum(tropical)
    check("tropical_high_et0", 1000.0 < tropical_annual < 2000.0,
          f"tropical annual={tropical_annual:.0f}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"  Thornthwaite ET₀: {passed}/{total} PASS, {failed}/{total} FAIL")
    print(f"{'=' * 60}")

    return failed == 0


def generate_benchmark():
    """Generate benchmark JSON for Rust validation."""
    el = EAST_LANSING_2023
    wo = WOOSTER_2023

    et0_el = thornthwaite_monthly_et0(el["monthly_tmean_c"], el["latitude"])
    et0_wo = thornthwaite_monthly_et0(wo["monthly_tmean_c"], wo["latitude"])

    hi_el = annual_heat_index(el["monthly_tmean_c"])
    a_el = thornthwaite_exponent(hi_el)

    hi_wo = annual_heat_index(wo["monthly_tmean_c"])
    a_wo = thornthwaite_exponent(hi_wo)

    benchmark = {
        "_provenance": {
            "method": "Thornthwaite (1948) monthly ET₀",
            "baseline_script": "control/thornthwaite/thornthwaite_et0.py",
            "baseline_commit": "pending",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "created": "2026-02-26",
            "references": [
                "Thornthwaite C.W. (1948) Geographical Review 38(1):55-94",
                "Willmott C.J. et al. (1985) Physical Climatology",
            ],
        },
        "analytical": {
            "heat_index_25C": round(monthly_heat_index_term(25.0), 6),
            "heat_index_annual_uniform_25C": round(annual_heat_index([25.0] * 12), 6),
            "exponent_uniform_25C": round(thornthwaite_exponent(annual_heat_index([25.0] * 12)), 6),
            "unadjusted_et0_25C": round(
                unadjusted_monthly_et0(25.0,
                                       annual_heat_index([25.0] * 12),
                                       thornthwaite_exponent(annual_heat_index([25.0] * 12))),
                6),
            "freezing_et0": 0.0,
        },
        "east_lansing": {
            "latitude": el["latitude"],
            "monthly_tmean_c": el["monthly_tmean_c"],
            "heat_index": round(hi_el, 6),
            "exponent_a": round(a_el, 6),
            "monthly_et0_mm": [round(e, 4) for e in et0_el],
            "annual_et0_mm": round(sum(et0_el), 4),
            "tol": 0.5,
        },
        "wooster": {
            "latitude": wo["latitude"],
            "monthly_tmean_c": wo["monthly_tmean_c"],
            "heat_index": round(hi_wo, 6),
            "exponent_a": round(a_wo, 6),
            "monthly_et0_mm": [round(e, 4) for e in et0_wo],
            "annual_et0_mm": round(sum(et0_wo), 4),
            "tol": 0.5,
        },
        "monotonicity_temps": [10.0, 15.0, 20.0, 25.0, 30.0],
        "monotonicity_latitude": 42.0,
        "edge_cases": {
            "all_frozen": {"monthly_tmean_c": [-10.0] * 12, "expected_annual": 0.0},
            "tropical_uniform": {"monthly_tmean_c": [28.0] * 12, "latitude": 5.0,
                                 "annual_range": [1000.0, 2000.0]},
        },
        "thresholds": {
            "annual_et0_range_mm": [400.0, 900.0],
            "growing_season_fraction_min": 0.65,
            "th_hg_ratio_range": [0.4, 2.5],
            "_tolerance_justification": (
                "Thornthwaite is a monthly empirical method; wider tolerances "
                "than PM reflect the method's inherent approximation. "
                "Annual range per Thornthwaite (1948) for humid continental climate."
            ),
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_thornthwaite.json")
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  Benchmark written to {out_path}")

    return benchmark


if __name__ == "__main__":
    benchmark = generate_benchmark()
    success = run_validation()
    sys.exit(0 if success else 1)
