# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 033: Makkink (1957) Radiation-Based ET₀ — Python Control Baseline

The Makkink method estimates reference ET₀ from solar radiation and
temperature only (no wind or humidity).  Widely used in the Netherlands
(KNMI standard) and Northern Europe.

Equation (de Bruin 1987 coefficients):
  ET₀ = C1 × (Δ/(Δ+γ)) × Rs/λ + C2

Where:
  C1 = 0.61, C2 = -0.12 (de Bruin 1987)
  Δ  = slope of saturation vapour pressure curve (kPa/°C)
  γ  = psychrometric constant (kPa/°C)
  Rs = incoming solar radiation (MJ/m²/day)
  λ  = 2.45 MJ/kg (latent heat of vaporization)

References:
  - Makkink GF (1957) J Inst Water Eng 11:277-288
  - de Bruin HAR (1987) From Penman to Makkink, TNO, The Hague, pp 5-31
  - Xu CY, Singh VP (2002) Water Resources Management 16:197-219

Data: Analytical (published equations). Open-Meteo for cross-comparison.
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_makkink.json"

C1 = 0.61
C2 = -0.12
LAMBDA = 2.45


def saturation_vapour_pressure(t):
    """e_s (kPa) at temperature t (°C). FAO-56 Eq. 11."""
    return 0.6108 * math.exp(17.27 * t / (t + 237.3))


def vapour_pressure_slope(t):
    """Δ (kPa/°C) — slope of sat. vapour pressure curve. FAO-56 Eq. 13."""
    es = saturation_vapour_pressure(t)
    return 4098.0 * es / (t + 237.3) ** 2


def atmospheric_pressure(elevation_m):
    """Atmospheric pressure (kPa) from elevation. FAO-56 Eq. 7."""
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa):
    """γ (kPa/°C). FAO-56 Eq. 8."""
    return 0.665e-3 * pressure_kpa


def makkink_et0(tmean, rs_mj, elevation_m):
    """Makkink ET₀ (mm/day) with de Bruin (1987) coefficients."""
    p = atmospheric_pressure(elevation_m)
    gamma = psychrometric_constant(p)
    delta = vapour_pressure_slope(tmean)
    et0 = C1 * (delta / (delta + gamma)) * (rs_mj / LAMBDA) + C2
    return max(0.0, et0)


def validate_analytical(benchmark):
    checks = benchmark["validation_checks"]["analytical"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = makkink_et0(tc["tmean"], tc["rs_mj"], tc["elevation_m"])
        expected = tc["expected_et0"]
        tol = tc["tolerance"]
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] T={tc['tmean']}, Rs={tc['rs_mj']}, z={tc['elevation_m']}"
              f" → ET₀={computed:.3f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_pm_cross(benchmark):
    checks = benchmark["validation_checks"]["pm_cross_comparison"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = makkink_et0(tc["tmean"], tc["rs_mj"], tc["elevation_m"])
        ratio = computed / tc["approx_pm_et0"] if tc["approx_pm_et0"] > 0 else 0.0
        ok = tc["min_ratio"] <= ratio <= tc["max_ratio"]
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: Makkink={computed:.2f} / PM≈{tc['approx_pm_et0']}"
              f" = {ratio:.3f} (range [{tc['min_ratio']}, {tc['max_ratio']}])")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_edge_cases(benchmark):
    checks = benchmark["validation_checks"]["edge_cases"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = makkink_et0(tc["tmean"], tc["rs_mj"], tc["elevation_m"])
        check = tc["check"]
        if check == "non_negative":
            ok = computed >= 0.0
        elif check == "positive":
            ok = computed > 0.0
        elif check == "zero":
            ok = computed == 0.0
        else:
            ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: ET₀={computed:.4f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_monotonicity(benchmark):
    checks = benchmark["validation_checks"]["monotonicity"]["test_cases"]
    passed = 0
    for tc in checks:
        if "base_rs" in tc:
            low = makkink_et0(tc["tmean"], tc["base_rs"], tc["elevation_m"])
            high = makkink_et0(tc["tmean"], tc["high_rs"], tc["elevation_m"])
        else:
            low = makkink_et0(tc["base_t"], tc["rs_mj"], tc["elevation_m"])
            high = makkink_et0(tc["high_t"], tc["rs_mj"], tc["elevation_m"])
        ok = high > low
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: {low:.3f} < {high:.3f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_pyet_cross(benchmark):
    """Cross-validate against pyet.makkink() if available."""
    try:
        import pyet
        import pandas as pd
    except ImportError:
        print("  [SKIP] pyet not installed — skipping cross-validation")
        return 0, 0

    section = benchmark["validation_checks"]["pyet_cross_validation"]
    tol = section["tolerance"]
    conditions = section["test_conditions"]
    passed = 0
    for tc in conditions:
        our_et0 = makkink_et0(tc["tmean"], tc["rs_mj"], tc["elevation_m"])
        tmean_s = pd.Series([tc["tmean"]])
        rs_s = pd.Series([tc["rs_mj"]])
        p = atmospheric_pressure(tc["elevation_m"])
        try:
            pyet_val = float(pyet.makkink(tmean_s, rs_s, elevation=tc["elevation_m"]).iloc[0])
        except Exception as e:
            print(f"  [SKIP] pyet.makkink failed: {e}")
            continue
        diff = abs(our_et0 - pyet_val)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] T={tc['tmean']}, Rs={tc['rs_mj']}: "
              f"ours={our_et0:.3f}, pyet={pyet_val:.3f}, diff={diff:.4f} (tol {tol})")
        if ok:
            passed += 1
    return passed, len(conditions)


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Analytical Benchmarks ──")
    p, t = validate_analytical(benchmark)
    total_passed += p
    total_checks += t

    print("\n── PM Cross-Comparison ──")
    p, t = validate_pm_cross(benchmark)
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
    p, t = validate_pyet_cross(benchmark)
    total_passed += p
    total_checks += t

    print(f"\n=== Makkink ET₀: {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
