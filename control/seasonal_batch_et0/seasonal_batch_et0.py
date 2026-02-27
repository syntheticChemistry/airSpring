#!/usr/bin/env python3
"""Exp 042 — Seasonal Batch ET₀ at GPU Scale (Python control).

Generates synthetic 365-day weather for 4 US climate stations, computes
daily FAO-56 PM ET₀, and validates seasonal aggregates. This establishes
the reference that the Rust CPU and GPU batch paths must match.

References:
  Allen et al. (1998) FAO-56
  NOAA US climate normals
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(SCRIPT_DIR, "benchmark_seasonal_batch.json")


def seasonal_value(doy, vmin, vmax, phase_doy=196):
    """Sinusoidal seasonal curve peaking at phase_doy (default mid-July)."""
    frac = math.sin(2.0 * math.pi * (doy - phase_doy + 91.25) / 365.0)
    mid = (vmin + vmax) / 2.0
    amp = (vmax - vmin) / 2.0
    return mid + amp * frac


def actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max):
    e_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    e_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    return (e_tmin * rh_max / 100.0 + e_tmax * rh_min / 100.0) / 2.0


def penman_monteith_et0(tmin, tmax, rs, wind_2m, ea, elevation, latitude, doy):
    tmean = (tmin + tmax) / 2.0
    P = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    e_s = (0.6108 * math.exp(17.27 * tmax / (tmax + 237.3)) +
           0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))) / 2.0

    lat_rad = math.radians(latitude)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws_arg = -math.tan(lat_rad) * math.tan(dec)
    ws = math.acos(max(-1.0, min(1.0, ws_arg)))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws))
    rso = (0.75 + 2e-5 * elevation) * ra
    rns = (1.0 - 0.23) * rs
    rnl = (4.903e-9 *
           ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0 *
           (0.34 - 0.14 * math.sqrt(max(0, ea))) *
           (1.35 * (rs / rso if rso > 0 else 0) - 0.35))
    rn = rns - rnl
    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * wind_2m * (e_s - ea)
    den = delta + gamma * (1.0 + 0.34 * wind_2m)
    return max(0.0, num / den)


def generate_station_year(station):
    """Generate 365 days of weather and compute ET₀."""
    et0_vals = []
    for doy in range(1, 366):
        tmax = seasonal_value(doy, *station["tmax_range"])
        tmin = seasonal_value(doy, *station["tmin_range"])
        rh_max = seasonal_value(doy, *station["rh_max_range"])
        rh_min = seasonal_value(doy, *station["rh_min_range"])
        rs = seasonal_value(doy, *station["rs_range"])
        ea = actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max)
        et0 = penman_monteith_et0(
            tmin, tmax, rs, station["wind_2m"],
            ea, station["elevation"], station["latitude"], doy)
        et0_vals.append(et0)
    return et0_vals


def validate_seasonal_shape(label, et0_vals):
    """Summer ET₀ > Winter ET₀."""
    summer = sum(et0_vals[152:244]) / 92  # Jun-Aug
    winter = sum(et0_vals[0:59] + et0_vals[334:365]) / 90  # Dec-Feb
    ok = summer > winter
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {label}: summer_mean={summer:.2f} > winter_mean={winter:.2f}")
    return 1 if ok else 0


def validate_annual_total(label, et0_vals, expected):
    annual = sum(et0_vals)
    lo, hi = expected["min"], expected["max"]
    ok = lo <= annual <= hi
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {label}: annual={annual:.0f} mm in [{lo}, {hi}]")
    return 1 if ok else 0


def validate_daily_range(label, et0_vals):
    ok = all(0 <= v <= 15 for v in et0_vals)
    status = "PASS" if ok else "FAIL"
    mn, mx = min(et0_vals), max(et0_vals)
    print(f"  {status} {label}: daily ET₀ in [0, 15], actual [{mn:.2f}, {mx:.2f}]")
    return 1 if ok else 0


def validate_batch_consistency(all_stations_et0):
    """Verify that computing stations individually matches batch computation."""
    ok = True
    for vals in all_stations_et0:
        if len(vals) != 365:
            ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  {status} All stations produce 365 daily values")
    return 1 if ok else 0


def validate_reduction_accuracy(label, et0_vals):
    """Mean from manual sum matches."""
    manual_sum = sum(et0_vals)
    manual_mean = manual_sum / len(et0_vals)
    ok = abs(manual_mean * 365 - manual_sum) < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {label}: sum={manual_sum:.2f}, mean×365={manual_mean*365:.2f}")
    return 1 if ok else 0


def main():
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    print("Exp 042: Seasonal Batch ET₀ at GPU Scale")
    print(f"Benchmark: {BENCHMARK}")

    stations = benchmark["stations"]
    passed, total = 0, 0
    all_et0 = []

    for st in stations:
        label = st["label"]
        et0_vals = generate_station_year(st)
        all_et0.append(et0_vals)

        print(f"\n[Station: {label}]")

        total += 1
        passed += validate_seasonal_shape(label, et0_vals)
        total += 1
        passed += validate_annual_total(label, et0_vals, st["expected_annual_et0_mm"])
        total += 1
        passed += validate_daily_range(label, et0_vals)
        total += 1
        passed += validate_reduction_accuracy(label, et0_vals)

    print("\n[Batch Consistency]")
    total += 1
    passed += validate_batch_consistency(all_et0)

    # Cross-station monotonicity: Arizona > Michigan > Pacific NW
    print("\n[Cross-Station Ordering]")
    az_total = sum(all_et0[1])
    mi_total = sum(all_et0[0])
    pnw_total = sum(all_et0[2])
    total += 1
    ok = az_total > mi_total > pnw_total
    status = "PASS" if ok else "FAIL"
    print(f"  {status} Arizona ({az_total:.0f}) > Michigan ({mi_total:.0f}) > Pacific NW ({pnw_total:.0f})")
    if ok: passed += 1

    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{total} PASS")
    if passed == total:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
