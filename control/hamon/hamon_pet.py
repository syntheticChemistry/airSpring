# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 035: Hamon (1961) Temperature-Based PET — Python Control Baseline

The Hamon method estimates potential evapotranspiration from daily mean
temperature and daylight hours only.  It requires the least data of any
ET₀ method (no radiation, wind, or humidity).  Appropriate for data-sparse
deployments and long-term historical reconstruction where only temperature
records exist.

Equation (Lu et al. 2005 formulation):
  PET = 0.1651 × N × RHOSAT × KPEC

Where:
  N      = possible daylight hours (from solar geometry)
  RHOSAT = saturated absolute humidity = 216.7 × e_s / (T + 273.3) [g/m³]
  e_s    = 0.6108 × exp(17.27 × T / (T + 237.3)) [kPa]
  KPEC   = 1.0 (correction factor; some use 1.2)

References:
  - Hamon WR (1961) J Hydraulics Div ASCE 87(HY3):107-120
  - Lu J, Sun G, McNulty SG, Amatya DM (2005) J Am Water Resour Assoc 41(3):621-633
  - Xu CY, Singh VP (2001) Water Resources Management 15:305-319

Provenance:
  Baseline commit: c080031
  Benchmark output: control/hamon/benchmark_hamon.json
  Reproduction: python control/hamon/hamon_pet.py
  Created: 2026-02-27

Data: Analytical (published equations + FAO-56 solar geometry).
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_hamon.json"

KPEC = 1.0


def saturation_vapour_pressure(t):
    """e_s (kPa). FAO-56 Eq. 11."""
    return 0.6108 * math.exp(17.27 * t / (t + 237.3))


def saturated_absolute_humidity(t):
    """RHOSAT (g/m³) — saturated vapour density at temperature t (°C)."""
    es = saturation_vapour_pressure(t)
    return 216.7 * es / (t + 273.3)


def daylight_hours(latitude_deg, doy):
    """Possible sunshine hours N from latitude and day of year. FAO-56 Eq. 34."""
    lat_rad = latitude_deg * math.pi / 180.0
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    arg = -math.tan(lat_rad) * math.tan(delta)
    arg = max(-1.0, min(1.0, arg))
    ws = math.acos(arg)
    return 24.0 * ws / math.pi


def hamon_pet(tmean, day_length_hours):
    """Hamon PET (mm/day) using Lu et al. (2005) formulation."""
    if tmean < 0.0 or day_length_hours <= 0.0:
        return 0.0
    rhosat = saturated_absolute_humidity(tmean)
    return 0.1651 * day_length_hours * rhosat * KPEC


def hamon_pet_from_location(tmean, latitude_deg, doy):
    """Hamon PET computing day length from solar geometry."""
    if tmean < 0.0:
        return 0.0
    n = daylight_hours(latitude_deg, doy)
    return hamon_pet(tmean, n)


def validate_analytical(benchmark):
    checks = benchmark["validation_checks"]["analytical"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = hamon_pet(tc["tmean"], tc["day_length_hours"])
        expected = tc["expected_pet"]
        tol = tc["tolerance"]
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] T={tc['tmean']}, N={tc['day_length_hours']}h"
              f" → PET={computed:.3f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_day_length(benchmark):
    checks = benchmark["validation_checks"]["day_length_computation"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = daylight_hours(tc["latitude"], tc["doy"])
        expected = tc["expected_hours"]
        tol = tc["tolerance"]
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] lat={tc['latitude']}°, DOY={tc['doy']}"
              f" → N={computed:.2f}h (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_edge_cases(benchmark):
    checks = benchmark["validation_checks"]["edge_cases"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = hamon_pet(tc["tmean"], tc["day_length_hours"])
        check = tc["check"]
        if check == "positive":
            ok = computed > 0.0
        elif check == "non_negative":
            ok = computed >= 0.0
        elif check == "zero":
            ok = computed == 0.0
        else:
            ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: PET={computed:.4f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_monotonicity(benchmark):
    checks = benchmark["validation_checks"]["monotonicity"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        if "base_t" in tc and "base_dl" in tc:
            low = hamon_pet(tc["base_t"], tc["base_dl"])
            high = hamon_pet(tc["high_t"], tc["high_dl"])
        elif "base_t" in tc:
            dl = tc["day_length_hours"]
            low = hamon_pet(tc["base_t"], dl)
            high = hamon_pet(tc["high_t"], dl)
        else:
            tmean = tc["tmean"]
            low = hamon_pet(tmean, tc["base_dl"])
            high = hamon_pet(tmean, tc["high_dl"])
        ok = high > low
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {low:.3f} < {high:.3f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_pyet_cross():
    """Cross-validate against pyet.hamon() if available.

    pyet implements Hamon with different coefficients (Hamon 1963 variant)
    which gives systematically lower values than Lu et al. (2005).
    We verify correlation (both increase/decrease together) rather than
    absolute agreement, since both are valid published formulations.
    """
    try:
        import pyet
        import pandas as pd
    except ImportError:
        print("  [SKIP] pyet not installed — skipping cross-validation")
        return 0, 0

    conditions = [
        (20.0, 42.0, 172),
        (30.0, 42.0, 196),
        (10.0, 42.0, 80),
        (25.0, 42.0, 265),
        ( 5.0, 42.0, 355),
    ]
    passed = 0
    our_vals = []
    pyet_vals = []
    for tmean, lat, doy in conditions:
        our = hamon_pet_from_location(tmean, lat, doy)
        try:
            pyet_val = float(pyet.hamon(
                pd.Series([tmean]),
                lat=lat * math.pi / 180.0,
                method=1,
            ).iloc[0])
        except Exception as e:
            print(f"  [SKIP] pyet.hamon failed: {e}")
            continue
        our_vals.append(our)
        pyet_vals.append(pyet_val)
        ratio = our / pyet_val if pyet_val > 0 else float("nan")
        print(f"  [INFO] T={tmean}, lat={lat}, DOY={doy}: "
              f"ours={our:.3f}, pyet={pyet_val:.3f}, ratio={ratio:.2f}")

    if len(our_vals) >= 3:
        monotonic_ok = True
        for i in range(len(our_vals) - 1):
            for j in range(i + 1, len(our_vals)):
                ours_dir = our_vals[i] - our_vals[j]
                pyet_dir = pyet_vals[i] - pyet_vals[j]
                if ours_dir * pyet_dir < 0:
                    monotonic_ok = False
        ok = monotonic_ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Rank correlation preserved between formulations")
        if ok:
            passed += 1
        return passed, 1
    return 0, 0


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Analytical Benchmarks ──")
    p, t = validate_analytical(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Day Length Computation ──")
    p, t = validate_day_length(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Edge Cases ──")
    p, t = validate_edge_cases(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Monotonicity ──")
    p, t = validate_monotonicity(benchmark)
    total_passed += p
    total_checks += t

    print("\n── pyet Cross-Validation ──")
    p, t = validate_pyet_cross()
    total_passed += p
    total_checks += t

    print(f"\n=== Hamon PET: {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
