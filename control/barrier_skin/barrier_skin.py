# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 068: Barrier State Model — Python Control Baseline

Applies van Genuchten θ(h)/K(h) to model skin barrier permeability.
The epidermis is a biological porous medium: cytokine diffusion through
the extracellular matrix follows analogous physics to water flow through
soil (van Genuchten 1980).

Anderson connection (Paper 12 §2.3):
  - Barrier integrity maps to normalized θ(h): intact=θ_s, breached=θ_r
  - Breach fraction = 1 - normalized_integrity → d_eff = 2 + breach_fraction
  - This is the "dimensional promotion" — inverse of Paper 06 tillage collapse

References:
  - van Genuchten (1980) SSSA J 44:892-898
  - Paper 12: Immunological Anderson
  - Paper 06: No-till dimensional collapse
  - Tagami H (2008) Br J Dermatol 158:431-436
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_barrier_skin.json"


def vg_theta(h, theta_r, theta_s, alpha, n_vg):
    """Van Genuchten water retention θ(h)."""
    if h >= 0.0:
        return theta_s
    m = 1.0 - 1.0 / n_vg
    ah = alpha * abs(h)
    x = ah ** n_vg
    se = 1.0 / (1.0 + x) ** m
    return theta_r + (theta_s - theta_r) * se


def vg_k(h, ks, theta_r, theta_s, alpha, n_vg):
    """Van Genuchten-Mualem hydraulic conductivity K(h)."""
    if h >= 0.0:
        return ks
    m = 1.0 - 1.0 / n_vg
    theta = vg_theta(h, theta_r, theta_s, alpha, n_vg)
    se = (theta - theta_r) / (theta_s - theta_r)
    if se <= 0.0:
        return 0.0
    if se >= 1.0:
        return ks
    term = 1.0 - se ** (1.0 / m)
    if term <= 0.0:
        return ks
    kr = math.sqrt(se) * (1.0 - term ** m) ** 2
    return ks * max(0.0, min(1.0, kr))


def normalize_barrier(theta, theta_r, theta_s):
    """Normalize θ to [0,1] barrier integrity."""
    if theta_s <= theta_r:
        return 0.0
    return (theta - theta_r) / (theta_s - theta_r)


def breach_fraction(barrier_integrity):
    """Convert barrier integrity to breach fraction."""
    return max(0.0, min(1.0, 1.0 - barrier_integrity))


def d_eff(barrier_integrity):
    """Effective dimension from barrier integrity."""
    f = breach_fraction(barrier_integrity)
    return 2.0 + f


HEALTHY_HUMAN = {"theta_r": 0.05, "theta_s": 1.0, "alpha": 0.01, "n_vg": 1.8, "ks": 50.0}


def validate_retention(benchmark):
    checks = benchmark["validation_checks"]["barrier_vg_retention"]["test_cases"]
    passed = 0
    total = 0
    p = HEALTHY_HUMAN
    for tc in checks:
        label = tc["label"]
        h = tc["h"]
        theta = vg_theta(h, p["theta_r"], p["theta_s"], p["alpha"], p["n_vg"])
        barrier = normalize_barrier(theta, p["theta_r"], p["theta_s"])
        total += 1
        if "expected_barrier" in tc:
            expected = tc["expected_barrier"]
            tol = tc["tolerance"]
            ok = abs(barrier - expected) <= tol
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: barrier={barrier:.6f} (expected {expected}, tol {tol})")
        elif "expected_barrier_range" in tc:
            lo, hi = tc["expected_barrier_range"]
            ok = lo <= barrier <= hi
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: barrier={barrier:.6f} (range [{lo}, {hi}])")
        else:
            ok = True
            print(f"  [PASS] {label}: barrier={barrier:.6f}")
        if ok:
            passed += 1
    return passed, total


def validate_conductivity(benchmark):
    checks = benchmark["validation_checks"]["barrier_conductivity"]["test_cases"]
    passed = 0
    total = 0
    p = HEALTHY_HUMAN
    ks = p["ks"]
    for tc in checks:
        label = tc["label"]
        h = tc["h"]
        k = vg_k(h, ks, p["theta_r"], p["theta_s"], p["alpha"], p["n_vg"])
        k_ratio = k / ks
        total += 1
        if "expected_k_ratio" in tc:
            expected = tc["expected_k_ratio"]
            tol = tc["tolerance"]
            ok = abs(k_ratio - expected) <= tol
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: K_ratio={k_ratio:.6f} (expected {expected})")
        elif "expected_k_less_than_max" in tc:
            ok = k < ks
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: K={k:.4f} < Ks={ks}")
        elif "expected_k_near_zero" in tc:
            ok = k < 0.001 * ks
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: K={k:.6f} ≈ 0")
        else:
            ok = True
        if ok:
            passed += 1
    return passed, total


def validate_d_eff_mapping(benchmark):
    checks = benchmark["validation_checks"]["barrier_to_d_eff_mapping"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        bi = tc["barrier_integrity"]
        expected_b = tc["expected_breach"]
        expected_d = tc["expected_d_eff"]
        tol = tc["tolerance"]
        computed_b = breach_fraction(bi)
        computed_d = d_eff(bi)
        ok_b = abs(computed_b - expected_b) <= tol
        ok_d = abs(computed_d - expected_d) <= tol
        ok = ok_b and ok_d
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: breach={computed_b:.4f} d_eff={computed_d:.4f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_skin_params(benchmark):
    checks = benchmark["validation_checks"]["skin_vg_params"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        p = tc
        theta_intact = vg_theta(0.0, p["theta_r"], p["theta_s"], p["alpha"], p["n_vg"])
        theta_stressed = vg_theta(-100.0, p["theta_r"], p["theta_s"], p["alpha"], p["n_vg"])
        ok = theta_intact > theta_stressed
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: θ(0)={theta_intact:.4f} > θ(-100)={theta_stressed:.4f}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_duality(benchmark):
    checks = benchmark["validation_checks"]["duality_check"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        d_before = tc["d_before"]
        d_after = tc["d_after"]
        direction = tc["direction"]
        if direction == "collapse":
            ok = d_after < d_before
        else:
            ok = d_after > d_before
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {d_before}→{d_after} ({direction})")
        if ok:
            passed += 1
    return passed, len(checks)


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Barrier VG Retention ──")
    p, t = validate_retention(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Barrier Conductivity ──")
    p, t = validate_conductivity(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Barrier → d_eff Mapping ──")
    p, t = validate_d_eff_mapping(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Skin VG Parameters ──")
    p, t = validate_skin_params(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Dimensional Duality ──")
    p, t = validate_duality(benchmark)
    total_passed += p
    total_checks += t

    print(f"\n=== Barrier Skin (Exp 068): {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
