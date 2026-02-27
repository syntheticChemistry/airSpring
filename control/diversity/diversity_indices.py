# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 032: Ecological Diversity Indices — Python Control Baseline

Validates Shannon entropy, Simpson index, Chao1 richness estimator,
Pielou evenness, and Bray-Curtis dissimilarity for agroecosystem
assessment (cover crop mixtures, soil microbiome, pollinator habitat).

All expected values are analytically derived from published formulas.

References:
  - Shannon CE (1948) A mathematical theory of communication.
    Bell System Technical Journal 27(3):379-423.
  - Simpson EH (1949) Measurement of diversity. Nature 163:688.
  - Chao A (1984) Nonparametric estimation of the number of classes
    in a population. Scandinavian J Statistics 11(4):265-270.
  - Pielou EC (1966) The measurement of diversity in different types
    of biological collections. J Theoretical Biology 13:131-144.
  - Bray JR, Curtis JT (1957) An ordination of the upland forest
    communities of southern Wisconsin. Ecological Monographs 27(4):325-349.

Data: Analytical (cover crop species abundance profiles).
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_diversity.json"


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


def simpson(counts):
    """Simpson diversity 1 - Σ pᵢ²."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts)


def chao1(counts):
    """Bias-corrected Chao1 richness estimator.

    Matches upstream barracuda::stats::diversity::chao1:
      S_chao1 = S_obs + f₁(f₁-1) / (2(f₂+1))  when f₂ > 0
      S_chao1 = S_obs + f₁(f₁-1) / 2           when f₂ = 0
    where f₁ = singletons, f₂ = doubletons (within ±0.5 tolerance).
    """
    halfwidth = 0.5
    s_obs = sum(1 for c in counts if c > 0)
    f1 = sum(1 for c in counts if abs(c - 1.0) < halfwidth)
    f2 = sum(1 for c in counts if abs(c - 2.0) < halfwidth)
    if f2 > 0:
        return s_obs + f1 * (f1 - 1) / (2.0 * (f2 + 1))
    if f1 > 0:
        return s_obs + f1 * (f1 - 1) / 2.0
    return float(s_obs)


def pielou_evenness(counts):
    """Pielou evenness J' = H' / ln(S)."""
    s = sum(1 for c in counts if c > 0)
    if s <= 1:
        return 0.0
    h = shannon(counts)
    return h / math.log(s)


def observed_species(counts):
    """Number of non-zero abundance species."""
    return sum(1 for c in counts if c > 0)


def bray_curtis(a, b):
    """Bray-Curtis dissimilarity = Σ|aᵢ-bᵢ| / Σ(aᵢ+bᵢ)."""
    num = sum(abs(ai - bi) for ai, bi in zip(a, b))
    den = sum(ai + bi for ai, bi in zip(a, b))
    if den == 0:
        return 0.0
    return num / den


def validate_shannon(benchmark):
    checks = benchmark["validation_checks"]["shannon"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        counts = tc["counts"]
        expected = tc["expected"]
        tol = tc["tolerance"]
        computed = shannon(counts)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: H'={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, total


def validate_simpson(benchmark):
    checks = benchmark["validation_checks"]["simpson"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        counts = tc["counts"]
        expected = tc["expected"]
        tol = tc["tolerance"]
        computed = simpson(counts)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: D={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, total


def validate_chao1(benchmark):
    checks = benchmark["validation_checks"]["chao1"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        counts = tc["counts"]
        expected = tc["expected"]
        tol = tc["tolerance"]
        computed = chao1(counts)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: Chao1={computed:.4f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, total


def validate_pielou(benchmark):
    checks = benchmark["validation_checks"]["pielou"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        counts = tc["counts"]
        expected = tc["expected"]
        tol = tc["tolerance"]
        computed = pielou_evenness(counts)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: J'={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, total


def validate_bray_curtis(benchmark):
    checks = benchmark["validation_checks"]["bray_curtis"]["test_cases"]
    passed = 0
    total = len(checks)
    for tc in checks:
        a = tc["sample_a"]
        b = tc["sample_b"]
        expected = tc["expected"]
        tol = tc["tolerance"]
        computed = bray_curtis(a, b)
        ok = abs(computed - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tc['label']}: BC={computed:.6f} (expected {expected}, tol {tol})")
        if ok:
            passed += 1
    return passed, total


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Shannon Entropy ──")
    p, t = validate_shannon(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Simpson Diversity ──")
    p, t = validate_simpson(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Chao1 Richness ──")
    p, t = validate_chao1(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Pielou Evenness ──")
    p, t = validate_pielou(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Bray-Curtis Dissimilarity ──")
    p, t = validate_bray_curtis(benchmark)
    total_passed += p
    total_checks += t

    print(f"\n=== Diversity Indices: {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
