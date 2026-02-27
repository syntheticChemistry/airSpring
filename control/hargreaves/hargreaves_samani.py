# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 031: Hargreaves-Samani (1985) Temperature-Only ET₀ — Python Control Baseline

Validates the Hargreaves-Samani method against published analytical benchmarks
and FAO-56 Chapter 4 cross-comparison data.  The Hargreaves method requires
only Tmin, Tmax, and extraterrestrial radiation Ra — making it the fallback
when humidity, wind, or radiation data are unavailable.

Equation:
  ET₀_HG = 0.0023 × (Tmean + 17.8) × (Tmax − Tmin)^0.5 × Ra

References:
  - Hargreaves GH, Samani ZA (1985) Reference crop evapotranspiration from
    temperature. Applied Eng Agric 1(2):96-99. DOI:10.13031/2013.26773
  - Allen et al. (1998) FAO-56 Eq. 52 (Hargreaves as data-limited alternative)
  - Droogers & Allen (2002) Estimating reference evapotranspiration under
    inaccurate data. Irrigation & Drainage Systems 16:33-45.

Data: Analytical + FAO-56 Table 2 (published reference cities).
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_hargreaves.json"


def hargreaves_et0(tmin, tmax, ra_mm_day):
    """Hargreaves-Samani ET₀ (mm/day)."""
    tmean = (tmin + tmax) / 2.0
    dt = max(tmax - tmin, 0.0)
    return max(0.0, 0.0023 * (tmean + 17.8) * math.sqrt(dt) * ra_mm_day)


def extraterrestrial_radiation_ra(lat_deg, doy):
    """Extraterrestrial radiation Ra (MJ/m²/day) from FAO-56 Eq. 21.

    Converts to mm/day equivalent via: Ra_mm = Ra_MJ / 2.45.
    """
    lat_rad = lat_deg * math.pi / 180.0
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    gsc = 0.0820
    ra_mj = (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * math.sin(lat_rad) * math.sin(delta)
        + math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    return ra_mj / 2.45


def validate_analytical(benchmark):
    checks = benchmark["validation_checks"]["analytical"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        tmin = tc["tmin"]
        tmax = tc["tmax"]
        ra_mm = tc["ra_mm_day"]
        expected = tc["expected_et0"]
        tol = tc["tolerance"]
        computed = hargreaves_et0(tmin, tmax, ra_mm)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] Tmin={tmin}, Tmax={tmax}, Ra={ra_mm} "
            f"→ ET₀={computed:.4f} (expected {expected}, tol {tol})"
        )
        if ok:
            passed += 1
    return passed, total


def validate_ra_computation(benchmark):
    checks = benchmark["validation_checks"]["ra_computation"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        lat = tc["latitude"]
        doy = tc["doy"]
        expected = tc["expected_ra_mm"]
        tol = tc["tolerance"]
        computed = extraterrestrial_radiation_ra(lat, doy)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] lat={lat}°, DOY={doy} → Ra={computed:.3f} mm/d "
            f"(expected {expected}, tol {tol})"
        )
        if ok:
            passed += 1
    return passed, total


def validate_fao56_cities(benchmark):
    checks = benchmark["validation_checks"]["fao56_cross_comparison"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        city = tc["city"]
        tmin = tc["tmin"]
        tmax = tc["tmax"]
        lat = tc["latitude"]
        doy = tc["doy"]
        pm_et0 = tc["fao56_pm_et0"]
        max_ratio_diff = tc["max_ratio_diff"]
        ra_mm = extraterrestrial_radiation_ra(lat, doy)
        hg_et0 = hargreaves_et0(tmin, tmax, ra_mm)
        ratio = hg_et0 / pm_et0 if pm_et0 > 0 else float("nan")
        ratio_diff = abs(ratio - 1.0)
        ok = ratio_diff <= max_ratio_diff
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {city}: HG={hg_et0:.2f} vs PM={pm_et0:.2f} "
            f"(ratio={ratio:.3f}, max_diff={max_ratio_diff})"
        )
        if ok:
            passed += 1
    return passed, total


def validate_edge_cases(benchmark):
    checks = benchmark["validation_checks"]["edge_cases"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        label = tc["label"]
        tmin = tc["tmin"]
        tmax = tc["tmax"]
        ra_mm = tc["ra_mm_day"]
        computed = hargreaves_et0(tmin, tmax, ra_mm)
        check_type = tc["check"]
        if check_type == "zero":
            ok = computed == 0.0
        elif check_type == "positive":
            ok = computed > 0.0
        elif check_type == "ge":
            ok = computed >= tc["bound"]
        else:
            ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: ET₀={computed:.4f}")
        if ok:
            passed += 1
    return passed, total


def validate_monotonicity(benchmark):
    checks = benchmark["validation_checks"]["monotonicity"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        label = tc["label"]
        base = tc["base"]
        increased = tc["increased"]
        ra = tc["ra_mm_day"]
        et0_base = hargreaves_et0(base["tmin"], base["tmax"], ra)
        et0_inc = hargreaves_et0(increased["tmin"], increased["tmax"], ra)
        ok = et0_inc > et0_base
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {et0_base:.3f} < {et0_inc:.3f}")
        if ok:
            passed += 1
    return passed, total


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Analytical Benchmarks ──")
    p, t = validate_analytical(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Extraterrestrial Radiation Ra ──")
    p, t = validate_ra_computation(benchmark)
    total_passed += p
    total_checks += t

    print("\n── FAO-56 Cross-Comparison (HG vs PM) ──")
    p, t = validate_fao56_cities(benchmark)
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

    print(f"\n=== Hargreaves-Samani ET₀: {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
