# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 034: Turc (1961) Temperature-Radiation ET₀ — Python Control Baseline

The Turc method estimates reference ET₀ from temperature and solar radiation,
with a humidity correction for arid climates (RH < 50%).  Developed for
French conditions and widely used in Western Europe and Mediterranean regions.

Equations:
  RH ≥ 50%:  ET₀ = 0.013 × T/(T+15) × (23.8846 × Rs + 50)
  RH <  50%:  ET₀ = 0.013 × T/(T+15) × (23.8846 × Rs + 50) × (1 + (50−RH)/70)

Where:
  T  = mean daily temperature (°C)
  Rs = solar radiation (MJ/m²/day)
  RH = mean relative humidity (%)
  23.8846 converts MJ/m²/day → cal/cm²/day

References:
  - Turc L (1961) Annales Agronomiques 12:13-49
  - Jensen ME, Burman RD, Allen RG (1990) ASCE Manual 70
  - Xu CY, Singh VP (2002) Water Resources Management 16:197-219

Provenance:
  Baseline commit: c080031
  Benchmark output: control/turc/benchmark_turc.json
  Reproduction: python control/turc/turc_et0.py
  Created: 2026-02-27

Data: Analytical (published equations).
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_turc.json"

MJ_TO_CAL_CM2 = 23.8846


def turc_et0(tmean, rs_mj, rh):
    """Turc (1961) ET₀ (mm/day)."""
    if tmean + 15.0 == 0.0:
        return 0.0
    t_factor = tmean / (tmean + 15.0)
    if t_factor < 0.0:
        return 0.0
    rs_cal = MJ_TO_CAL_CM2 * rs_mj + 50.0
    et0 = 0.013 * t_factor * rs_cal
    if rh < 50.0:
        et0 *= 1.0 + (50.0 - rh) / 70.0
    return max(0.0, et0)


def validate_analytical_high_rh(benchmark):
    checks = benchmark["validation_checks"]["analytical_high_rh"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = turc_et0(tc["tmean"], tc["rs_mj"], tc["rh"])
        expected = tc["expected_et0"]
        tol = tc["tolerance"]
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] T={tc['tmean']}, Rs={tc['rs_mj']}, RH={tc['rh']}%"
              f" → ET₀={computed:.3f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_analytical_low_rh(benchmark):
    checks = benchmark["validation_checks"]["analytical_low_rh"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = turc_et0(tc["tmean"], tc["rs_mj"], tc["rh"])
        expected = tc["expected_et0"]
        tol = tc["tolerance"]
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] T={tc['tmean']}, Rs={tc['rs_mj']}, RH={tc['rh']}%"
              f" → ET₀={computed:.3f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_humidity_boundary(benchmark):
    checks = benchmark["validation_checks"]["humidity_boundary"]["test_cases"]
    passed = 0
    for tc in checks:
        tmean = tc["tmean"]
        rs_mj = tc["rs_mj"]
        tol = tc["tolerance"]
        et0_at_50 = turc_et0(tmean, rs_mj, 50.0)
        et0_at_49 = turc_et0(tmean, rs_mj, 49.99)
        diff = abs(et0_at_50 - et0_at_49)
        ok = diff < tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] RH=50: {et0_at_50:.4f}, RH=49.99: {et0_at_49:.4f}, "
              f"diff={diff:.6f} (tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_edge_cases(benchmark):
    checks = benchmark["validation_checks"]["edge_cases"]["test_cases"]
    passed = 0
    for tc in checks:
        computed = turc_et0(tc["tmean"], tc["rs_mj"], tc["rh"])
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
        print(f"  [{status}] {tc['label']}: ET₀={computed:.4f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_monotonicity(benchmark):
    checks = benchmark["validation_checks"]["monotonicity"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        if "base_rs" in tc:
            low = turc_et0(tc["tmean"], tc["base_rs"], tc["rh"])
            high = turc_et0(tc["tmean"], tc["high_rs"], tc["rh"])
        elif "base_t" in tc:
            low = turc_et0(tc["base_t"], tc["rs_mj"], tc["rh"])
            high = turc_et0(tc["high_t"], tc["rs_mj"], tc["rh"])
        else:
            low = turc_et0(tc["tmean"], tc["rs_mj"], tc["base_rh"])
            high = turc_et0(tc["tmean"], tc["rs_mj"], tc["low_rh"])
        ok = high > low
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {low:.3f} < {high:.3f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_pyet_cross():
    """Cross-validate against pyet.turc() if available."""
    try:
        import pyet
        import pandas as pd
    except ImportError:
        print("  [SKIP] pyet not installed — skipping cross-validation")
        return 0, 0

    conditions = [
        (20.0, 15.0, 70.0),
        (30.0, 25.0, 55.0),
        (10.0,  8.0, 80.0),
        (30.0, 25.0, 40.0),
        (25.0, 20.0, 20.0),
    ]
    tol = 0.1
    passed = 0
    for tmean, rs, rh in conditions:
        our = turc_et0(tmean, rs, rh)
        try:
            pyet_val = float(pyet.turc(
                pd.Series([tmean]), pd.Series([rs]), pd.Series([rh])
            ).iloc[0])
        except Exception as e:
            print(f"  [SKIP] pyet.turc failed: {e}")
            continue
        diff = abs(our - pyet_val)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] T={tmean}, Rs={rs}, RH={rh}: "
              f"ours={our:.3f}, pyet={pyet_val:.3f}, diff={diff:.4f}")
        if ok:
            passed += 1
    return passed, len(conditions)


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Analytical (RH ≥ 50%) ──")
    p, t = validate_analytical_high_rh(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Analytical (RH < 50%) ──")
    p, t = validate_analytical_low_rh(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Humidity Boundary (RH=50%) ──")
    p, t = validate_humidity_boundary(benchmark)
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

    print(f"\n=== Turc ET₀: {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
