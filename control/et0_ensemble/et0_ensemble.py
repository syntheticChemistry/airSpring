#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exp 037 — ET₀ Ensemble Consensus (7-Method Weighted).

Combines all 7 validated ET₀ methods into a data-adaptive weighted
consensus estimate. Methods are gated by data availability:

Full weather:    PM, PT, Hargreaves, Makkink, Turc, Hamon (6 daily methods)
Radiation+Temp:  PT, Makkink, Turc, Hamon, Hargreaves
Temperature:     Hargreaves, Hamon

Thornthwaite is excluded — it computes monthly PET (mm/month) using an
annual heat index and is not directly comparable to daily ET₀ methods.

Weighting: equal-weight mean of available methods (robust baseline).

Provenance:
  Baseline commit: af1eb97 (2026-02-26)
  Benchmark: control/et0_ensemble/benchmark_et0_ensemble.json
  Reproduce: python control/et0_ensemble/et0_ensemble.py

References:
  Oudin et al. (2005) J Hydrol 303:290-306
  Droogers & Allen (2002) Irrig Drain Syst 16:33-45
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(SCRIPT_DIR, "benchmark_et0_ensemble.json")

# ── Individual ET₀ methods (from validated controls) ─────────────────────────

def penman_monteith_et0(tmin, tmax, tmean, rs_mj, wind_2m, e_a, elev, lat_deg, doy):
    P = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    e_s = (0.6108 * math.exp(17.27 * tmax / (tmax + 237.3)) +
           0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))) / 2.0
    lat_rad = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra
    rns = (1.0 - 0.23) * rs_mj
    rnl = (4.903e-9 *
           ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0 *
           (0.34 - 0.14 * math.sqrt(e_a)) *
           (1.35 * (rs_mj / rso if rso > 0 else 0) - 0.35))
    rn = rns - rnl
    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * wind_2m * (e_s - e_a)
    den = delta + gamma * (1.0 + 0.34 * wind_2m)
    return max(0.0, num / den)


def priestley_taylor_et0(tmean, rs_mj, elevation_m, tmin, tmax, e_a, lat_deg, doy):
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    lat_rad = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws))
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rns = (1.0 - 0.23) * rs_mj
    rnl = (4.903e-9 *
           ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0 *
           (0.34 - 0.14 * math.sqrt(e_a)) *
           (1.35 * (rs_mj / rso if rso > 0 else 0) - 0.35))
    rn = rns - rnl
    return max(0.0, 1.26 * (delta / (delta + gamma)) * rn / 2.45)


def hargreaves_et0(tmin, tmax, lat_deg, doy):
    tmean = (tmin + tmax) / 2.0
    lat_rad = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws))
    return max(0.0, 0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * ra)


def thornthwaite_monthly_et0(tmean_monthly):
    """Simplified: single month, assumes 30-day month, 12-hour days."""
    if tmean_monthly <= 0.0:
        return 0.0
    heat_index = (tmean_monthly / 5.0) ** 1.514
    a = 6.75e-7 * heat_index ** 3 - 7.71e-5 * heat_index ** 2 + 1.792e-2 * heat_index + 0.49239
    return max(0.0, 16.0 * (10.0 * tmean_monthly / heat_index) ** a)


def makkink_et0(tmean, rs_mj, elevation_m):
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    return max(0.0, 0.61 * (delta / (delta + gamma)) * rs_mj / 2.45 - 0.12)


def turc_et0(tmean, rs_mj, rh):
    if tmean <= 0.0:
        return 0.0
    rs_cal = rs_mj * 23.8846
    base = 0.013 * (tmean / (tmean + 15.0)) * (rs_cal + 50.0)
    if rh < 50.0:
        base *= 1.0 + (50.0 - rh) / 70.0
    return max(0.0, base)


def hamon_pet(tmean, day_length_hours):
    if tmean <= 0.0 or day_length_hours <= 0.0:
        return 0.0
    e_s = 0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))
    rho_sat = 216.7 * e_s / (tmean + 273.3)
    return max(0.0, 0.1651 * (day_length_hours / 12.0) * rho_sat)


# ── Ensemble computation ─────────────────────────────────────────────────────

def compute_ensemble(tc):
    """Compute ET₀ from all available methods and return ensemble stats."""
    methods = {}
    tmean = tc.get("tmean")
    tmin = tc.get("tmin")
    tmax = tc.get("tmax")
    rs_mj = tc.get("rs_mj")
    rh_pct = tc.get("rh_pct")
    wind = tc.get("wind_speed_2m")
    e_a = tc.get("e_a")
    elev = tc.get("elevation_m", 100.0)
    lat = tc.get("latitude_deg", 45.0)
    doy = tc.get("day_of_year", 180)
    dl = tc.get("day_length_hours")

    has_full = all(v is not None for v in [tmin, tmax, tmean, rs_mj, wind, e_a])
    has_rad = rs_mj is not None and tmean is not None
    has_temp = tmean is not None or (tmin is not None and tmax is not None)

    if tmean is None and tmin is not None and tmax is not None:
        tmean = (tmin + tmax) / 2.0

    if has_full:
        methods["pm"] = penman_monteith_et0(tmin, tmax, tmean, rs_mj, wind, e_a, elev, lat, doy)
        methods["pt"] = priestley_taylor_et0(tmean, rs_mj, elev, tmin, tmax, e_a, lat, doy)

    if has_rad:
        methods["makkink"] = makkink_et0(tmean, rs_mj, elev)
        if rh_pct is not None:
            methods["turc"] = turc_et0(tmean, rs_mj, rh_pct)

    if has_temp:
        if tmin is not None and tmax is not None:
            methods["hargreaves"] = hargreaves_et0(tmin, tmax, lat, doy)
        if dl is not None and dl > 0:
            methods["hamon"] = hamon_pet(tmean, dl)

    values = [v for v in methods.values() if v is not None and v >= 0]
    if not values:
        return {"consensus": 0.0, "methods": methods, "n_methods": 0, "spread": 0.0}

    consensus = sum(values) / len(values)
    spread = max(values) - min(values)
    return {
        "consensus": consensus,
        "methods": methods,
        "n_methods": len(values),
        "spread": spread,
        "min": min(values),
        "max": max(values),
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_full_weather(benchmark):
    print("\n[Full Weather Ensemble]")
    tests = benchmark["validation_checks"]["full_weather_ensemble"]["test_cases"]
    passed, total = 0, 0
    for tc in tests:
        label = tc["label"]
        total += 1
        result = compute_ensemble(tc)
        consensus = result["consensus"]
        n = result["n_methods"]
        spread = result["spread"]
        pm_ref = tc.get("pm_reference")
        tol = tc["tolerance"]

        ok = True
        if pm_ref is not None:
            diff = abs(consensus - pm_ref)
            ok = diff < tol
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label}: consensus={consensus:.3f} pm={pm_ref:.3f} diff={diff:.3f} tol={tol} n={n} spread={spread:.3f}")
        else:
            ok = consensus > 0 and n >= 5
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label}: consensus={consensus:.3f} n={n} spread={spread:.3f}")

        if ok:
            passed += 1

        for name, val in sorted(result["methods"].items()):
            print(f"         {name:>14s} = {val:.3f}")

    return passed, total


def validate_temp_only(benchmark):
    print("\n[Temperature-Only Ensemble]")
    tests = benchmark["validation_checks"]["temperature_only_ensemble"]["test_cases"]
    passed, total = 0, 0
    for tc in tests:
        label = tc["label"]
        total += 1
        result = compute_ensemble(tc)
        consensus = result["consensus"]
        n = result["n_methods"]
        spread = result["spread"]

        ok = n >= 2 and consensus >= 0
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {label}: consensus={consensus:.3f} n={n} spread={spread:.3f}")
        if ok:
            passed += 1

        for name, val in sorted(result["methods"].items()):
            print(f"         {name:>14s} = {val:.3f}")

    return passed, total


def validate_method_ranking(benchmark):
    print("\n[Method Ranking]")
    tests = benchmark["validation_checks"]["method_ranking"]["test_cases"]
    passed, total = 0, 0
    for tc in tests:
        label = tc["label"]
        check = tc["check"]
        total += 1
        result = compute_ensemble(tc)

        if check == "spread_positive":
            ok = result["spread"] > 0
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label}: spread={result['spread']:.3f}")
        elif check == "consensus_in_range":
            ok = result["min"] <= result["consensus"] <= result["max"]
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label}: min={result['min']:.3f} consensus={result['consensus']:.3f} max={result['max']:.3f}")
        else:
            ok = False
            print(f"  SKIP {label}: unknown check '{check}'")

        if ok:
            passed += 1

    return passed, total


def validate_monotonicity(benchmark):
    print("\n[Monotonicity]")
    tests = benchmark["validation_checks"]["monotonicity"]["test_cases"]
    passed, total = 0, 0
    for tc in tests:
        total += 1
        tc_low = dict(tc)
        tc_high = dict(tc)
        tc_low["tmean"] = tc["base_tmean"]
        tc_low["tmin"] = tc["base_tmean"] + tc["tmin_offset"]
        tc_low["tmax"] = tc["base_tmean"] + tc["tmax_offset"]
        tc_high["tmean"] = tc["step_tmean"]
        tc_high["tmin"] = tc["step_tmean"] + tc["tmin_offset"]
        tc_high["tmax"] = tc["step_tmean"] + tc["tmax_offset"]

        r_low = compute_ensemble(tc_low)
        r_high = compute_ensemble(tc_high)
        ok = r_high["consensus"] > r_low["consensus"]
        status = "PASS" if ok else "FAIL"
        print(f"  {status} T={tc['base_tmean']}→{tc['step_tmean']}: "
              f"ET₀={r_low['consensus']:.3f}→{r_high['consensus']:.3f}")
        if ok:
            passed += 1

    return passed, total


def main():
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    print("Exp 037: ET₀ Ensemble Consensus (7-Method Weighted)")
    print(f"Benchmark: {BENCHMARK}")

    total_pass, total_tests = 0, 0

    p, t = validate_full_weather(benchmark)
    total_pass += p
    total_tests += t

    p, t = validate_temp_only(benchmark)
    total_pass += p
    total_tests += t

    p, t = validate_method_ranking(benchmark)
    total_pass += p
    total_tests += t

    p, t = validate_monotonicity(benchmark)
    total_pass += p
    total_tests += t

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass}/{total_tests} PASS")
    if total_pass == total_tests:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
