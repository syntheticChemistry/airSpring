#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exp 039 — Cross-Method ET₀ Bias Correction.

Quantifies the systematic bias of simplified ET₀ methods relative to
FAO-56 PM, and computes linear correction factors that enable PM-
equivalent estimates from data-sparse stations.

Methods tested: Hargreaves, Makkink, Turc, Hamon (all have PM as ground
truth). Priestley-Taylor is excluded since it requires the same inputs
as PM.

References:
  Trajkovic (2007) J Irrig Drain Eng 133:38-42
  Xu & Singh (2002) Hydrol Process 16:3605-3623
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(SCRIPT_DIR, "benchmark_et0_bias.json")

# ── Individual ET₀ methods ───────────────────────────────────────────────────

def penman_monteith_et0(tc):
    tmin, tmax, tmean = tc["tmin"], tc["tmax"], tc["tmean"]
    rs = tc["rs_mj"]; wind = tc["wind_speed_2m"]; e_a = tc["e_a"]
    elev = tc["elevation_m"]; lat = tc["latitude_deg"]; doy = tc["day_of_year"]

    P = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    e_s = (0.6108 * math.exp(17.27 * tmax / (tmax + 237.3)) +
           0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))) / 2.0
    lat_rad = math.radians(lat)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra
    rns = (1.0 - 0.23) * rs
    rnl = (4.903e-9 *
           ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0 *
           (0.34 - 0.14 * math.sqrt(e_a)) *
           (1.35 * (rs / rso if rso > 0 else 0) - 0.35))
    rn = rns - rnl
    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * wind * (e_s - e_a)
    den = delta + gamma * (1.0 + 0.34 * wind)
    return max(0.0, num / den)


def hargreaves_et0(tc):
    tmin, tmax = tc["tmin"], tc["tmax"]
    tmean = (tmin + tmax) / 2.0
    lat_rad = math.radians(tc["latitude_deg"])
    doy = tc["day_of_year"]
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws))
    return max(0.0, 0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * ra)


def makkink_et0(tc):
    tmean = tc["tmean"]; rs = tc["rs_mj"]; elev = tc["elevation_m"]
    P = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    return max(0.0, 0.61 * (delta / (delta + gamma)) * rs / 2.45 - 0.12)


def turc_et0(tc):
    tmean = tc["tmean"]; rs = tc["rs_mj"]; rh = tc["rh_pct"]
    if tmean <= 0: return 0.0
    rs_cal = rs * 23.8846
    base = 0.013 * (tmean / (tmean + 15.0)) * (rs_cal + 50.0)
    if rh < 50.0:
        base *= 1.0 + (50.0 - rh) / 70.0
    return max(0.0, base)


def hamon_pet(tc):
    tmean = tc["tmean"]; dl = tc["day_length_hours"]
    if tmean <= 0 or dl <= 0: return 0.0
    e_s = 0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))
    rho_sat = 216.7 * e_s / (tmean + 273.3)
    return max(0.0, 0.1651 * (dl / 12.0) * rho_sat)


# ── Bias analysis ────────────────────────────────────────────────────────────

def compute_all_methods(tc):
    pm = penman_monteith_et0(tc)
    return {
        "pm": pm,
        "hargreaves": hargreaves_et0(tc),
        "makkink": makkink_et0(tc),
        "turc": turc_et0(tc),
        "hamon": hamon_pet(tc),
    }


def compute_bias_table(scenarios):
    """For each scenario × method, compute bias = method - PM."""
    rows = []
    for sc in scenarios:
        results = compute_all_methods(sc)
        pm = results["pm"]
        row = {"label": sc["label"], "pm": pm}
        for method in ["hargreaves", "makkink", "turc", "hamon"]:
            val = results[method]
            row[method] = val
            row[f"{method}_bias"] = val - pm
            row[f"{method}_pct"] = 100 * (val - pm) / pm if pm > 0 else 0
        rows.append(row)
    return rows


def compute_correction_factors(bias_table):
    """Compute per-method linear correction factor across all scenarios.

    Factor = mean(PM) / mean(method) for each method.
    """
    factors = {}
    for method in ["hargreaves", "makkink", "turc", "hamon"]:
        pm_vals = [r["pm"] for r in bias_table if r["pm"] > 0]
        method_vals = [r[method] for r in bias_table if r[method] > 0]
        if method_vals:
            factors[method] = sum(pm_vals) / sum(method_vals)
        else:
            factors[method] = 1.0
    return factors


# ── Validation ────────────────────────────────────────────────────────────────

def validate_bias_quantification(benchmark):
    print("\n[Bias Quantification]")
    scenarios = benchmark["validation_checks"]["bias_quantification"]["climate_scenarios"]
    bias_table = compute_bias_table(scenarios)

    passed, total = 0, 0
    for row in bias_table:
        total += 1
        pm = row["pm"]
        ok = pm > 0
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {row['label']}: PM={pm:.3f} mm/day")
        for m in ["hargreaves", "makkink", "turc", "hamon"]:
            print(f"         {m:>12s} = {row[m]:.3f}  bias={row[f'{m}_bias']:+.3f}  ({row[f'{m}_pct']:+.1f}%)")
        if ok:
            passed += 1

    return passed, total, bias_table


def validate_correction_factors(benchmark, bias_table):
    print("\n[Correction Factor Validation]")
    tol_pct = benchmark["validation_checks"]["correction_factor_validation"]["tolerance_pct"]
    scenarios = benchmark["validation_checks"]["bias_quantification"]["climate_scenarios"]
    factors = compute_correction_factors(bias_table)

    print(f"  Correction factors (mean PM / mean method):")
    for m, f in sorted(factors.items()):
        print(f"    {m:>12s}: {f:.4f}")

    passed, total = 0, 0
    for sc in scenarios:
        pm = penman_monteith_et0(sc)
        if pm <= 0:
            continue
        for method in ["hargreaves", "makkink", "turc", "hamon"]:
            val = {"hargreaves": hargreaves_et0, "makkink": makkink_et0,
                   "turc": turc_et0, "hamon": hamon_pet}[method](sc)
            corrected = val * factors[method]
            error_pct = abs(100 * (corrected - pm) / pm)
            total += 1
            ok = error_pct < tol_pct
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {sc['label']}: {method} corrected={corrected:.3f} pm={pm:.3f} err={error_pct:.1f}%")
            if ok:
                passed += 1

    return passed, total


def validate_structural(benchmark, bias_table):
    print("\n[Structural Checks]")
    factors = compute_correction_factors(bias_table)
    passed, total = 0, 0

    # All correction factors positive
    total += 1
    ok = all(f > 0 for f in factors.values())
    print(f"  {'PASS' if ok else 'FAIL'} All correction factors positive")
    if ok: passed += 1

    # Hargreaves overestimates in humid
    total += 1
    humid = [r for r in bias_table if "humid" in r["label"].lower()]
    ok = all(r["hargreaves_bias"] > 0 for r in humid)
    print(f"  {'PASS' if ok else 'FAIL'} Hargreaves overestimates in humid climates")
    if ok: passed += 1

    # Hamon underestimates everywhere
    total += 1
    ok = all(r["hamon_bias"] < 0 for r in bias_table if r["pm"] > 0)
    print(f"  {'PASS' if ok else 'FAIL'} Hamon underestimates everywhere")
    if ok: passed += 1

    # Makkink close to PM (|bias%| < 50% on average)
    total += 1
    avg_abs_bias = sum(abs(r["makkink_pct"]) for r in bias_table) / len(bias_table)
    ok = avg_abs_bias < 50
    print(f"  {'PASS' if ok else 'FAIL'} Makkink avg |bias| = {avg_abs_bias:.1f}% < 50%")
    if ok: passed += 1

    return passed, total


def main():
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    print("Exp 039: Cross-Method ET₀ Bias Correction")
    print(f"Benchmark: {BENCHMARK}")

    total_pass, total_tests = 0, 0

    p, t, bias_table = validate_bias_quantification(benchmark)
    total_pass += p; total_tests += t

    p, t = validate_correction_factors(benchmark, bias_table)
    total_pass += p; total_tests += t

    p, t = validate_structural(benchmark, bias_table)
    total_pass += p; total_tests += t

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass}/{total_tests} PASS")
    if total_pass == total_tests:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
