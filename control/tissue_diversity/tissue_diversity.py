# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 066: Tissue Diversity Profiling — Python Control Baseline

Maps cell-type heterogeneity in biological tissue to the Anderson disorder
parameter W. Uses Shannon, Simpson, and Pielou evenness to characterize
skin tissue composition for cytokine propagation modeling (Paper 12).

Anderson mapping:
  W_effective = (1 - Pielou J') * ln(S)
  where S = species richness (number of cell types)

Regime classification (from Paper 01, validated in wetSpring Exp107-156):
  d=2: W_c ≈ 4.0
  d=3: W_c ≈ 16.26 ± 0.95

References:
  - Paper 01: Anderson QS — W_c thresholds
  - Paper 06: No-till dimensional collapse
  - Paper 12: Immunological Anderson
  - Pielou (1966) J Theoretical Biology 13:131-144
  - McCandless et al. (2014) Vet Immunol Immunopathol 157:42-48

Data: Analytical (skin cell-type abundances from dermatological literature).
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_tissue_diversity.json"


def shannon(counts):
    """Shannon entropy H' = -Σ pᵢ ln(pᵢ)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def pielou_evenness(counts):
    """Pielou evenness J' = H' / ln(S)."""
    s = sum(1 for c in counts if c > 0)
    if s <= 1:
        return 0.0
    return shannon(counts) / math.log(s)


def anderson_w_effective(counts):
    """Effective Anderson disorder W = (1 - J') * ln(S)."""
    s = sum(1 for c in counts if c > 0)
    if s <= 1:
        return 0.0
    j = pielou_evenness(counts)
    return (1.0 - j) * math.log(s)


def classify_regime(w, d):
    """Classify Anderson regime from effective W and dimensionality."""
    w_c = 4.0 if d < 2.5 else 16.26
    margin = 0.1 * w_c
    if w > w_c + margin:
        return "Localized"
    elif w < w_c - margin:
        return "Extended"
    else:
        return "Critical"


def barrier_d_eff(breach_fraction):
    """d_eff = 2.0 + clamp(breach_fraction, 0, 1)."""
    f = max(0.0, min(1.0, breach_fraction))
    return 2.0 + f


COMPARTMENT_D = {
    "Epidermis": 2.0,
    "PapillaryDermis": 3.0,
    "ReticularDermis": 3.0,
}


def validate_tissue_shannon(benchmark):
    checks = benchmark["validation_checks"]["tissue_shannon"]["test_cases"]
    passed = 0
    for tc in checks:
        expected = tc["expected_shannon"]
        tol = tc["tolerance"]
        computed = shannon(tc["abundances"])
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: H'={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_tissue_pielou(benchmark):
    checks = benchmark["validation_checks"]["tissue_pielou"]["test_cases"]
    passed = 0
    for tc in checks:
        expected = tc["expected_pielou"]
        tol = tc["tolerance"]
        computed = pielou_evenness(tc["abundances"])
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: J'={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_anderson_w(benchmark):
    checks = benchmark["validation_checks"]["anderson_w_effective"]["test_cases"]
    passed = 0
    for tc in checks:
        expected = tc["expected_w"]
        tol = tc["tolerance"]
        computed = anderson_w_effective(tc["abundances"])
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: W={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_regime(benchmark):
    checks = benchmark["validation_checks"]["anderson_regime"]["test_cases"]
    passed = 0
    for tc in checks:
        w = tc["w_effective"]
        d = tc["d_eff"]
        expected = tc["expected_regime"]
        computed = classify_regime(w, d)
        ok = computed == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: regime={computed} (expected {expected})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_barrier(benchmark):
    checks = benchmark["validation_checks"]["barrier_disruption"]["test_cases"]
    passed = 0
    for tc in checks:
        expected = tc["expected_d_eff"]
        tol = tc["tolerance"]
        computed = barrier_d_eff(tc["breach_fraction"])
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: d_eff={computed:.4f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_compartments(benchmark):
    checks = benchmark["validation_checks"]["compartment_dimensions"]["test_cases"]
    passed = 0
    for tc in checks:
        compartment = tc["compartment"]
        expected = tc["expected_d"]
        tol = tc["tolerance"]
        computed = COMPARTMENT_D[compartment]
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: d={computed} (expected {expected})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_multi_compartment(benchmark):
    checks = benchmark["validation_checks"]["multi_compartment"]["test_cases"]
    passed = 0
    total = 0
    for tc in checks:
        compartments = tc["compartments"]
        n = len(compartments)
        expected_count = tc["expected_count"]
        ok = n == expected_count
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: count={n} (expected {expected_count})")
        total += 1
        if ok:
            passed += 1
        for comp in compartments:
            w = anderson_w_effective(comp["abundances"])
            print(f"    {comp['compartment']}: W={w:.6f}")
    return passed, total


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Tissue Shannon Entropy ──")
    p, t = validate_tissue_shannon(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Tissue Pielou Evenness ──")
    p, t = validate_tissue_pielou(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Anderson W Effective ──")
    p, t = validate_anderson_w(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Anderson Regime Classification ──")
    p, t = validate_regime(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Barrier Disruption d_eff ──")
    p, t = validate_barrier(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Compartment Dimensions ──")
    p, t = validate_compartments(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Multi-Compartment Analysis ──")
    p, t = validate_multi_compartment(benchmark)
    total_passed += p
    total_checks += t

    print(f"\n=== Tissue Diversity (Exp 066): {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
