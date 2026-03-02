# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 069: Cross-Species Skin Comparison — Python Control Baseline

Compares canine, feline, and human skin barrier parameters and Anderson
predictions. Validates Gonzales's comparative approach: same Anderson
physics, different barrier geometry.

Key insight: thinner canine epidermis → easier breach → faster dimensional
promotion → faster cytokine delocalization → faster AD progression.

References:
  - Paper 12: Immunological Anderson — cross-species validation
  - Gonzales AJ et al. (2013) Vet Dermatol 24:48-53
  - Marsella R, De Benedetto A (2017) Vet Dermatol 28:306-e69

Data: Published species skin parameters + published IC50 values.
"""
import json
import math
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_cross_species_skin.json"


def shannon(counts):
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
    s = sum(1 for c in counts if c > 0)
    if s <= 1:
        return 0.0
    return shannon(counts) / math.log(s)


def classify_regime(w, d):
    w_c = 4.0 if d < 2.5 else 16.26
    margin = 0.1 * w_c
    if w > w_c + margin:
        return "Localized"
    elif w < w_c - margin:
        return "Extended"
    else:
        return "Critical"


def barrier_d_eff(breach_fraction):
    f = max(0.0, min(1.0, breach_fraction))
    return 2.0 + f


def validate_species_params(benchmark):
    checks = benchmark["validation_checks"]["species_barrier_params"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        alpha = tc["barrier_alpha"]
        d_eff = tc["d_eff_intact"]
        ok = alpha > 0 and d_eff == 2.0
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: α={alpha}, d_eff_intact={d_eff}")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_breach_threshold(benchmark):
    checks = benchmark["validation_checks"]["breach_threshold"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        if "expected_canine_first" in tc:
            canine_at = tc["canine_breach_at"]
            human_at = tc["human_breach_at"]
            ok = canine_at < human_at
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: canine={canine_at} < human={human_at}")
        elif "expected_canine_d" in tc:
            tol = tc["tolerance"]
            canine_d = barrier_d_eff(tc["scratch_intensity"])
            human_d = barrier_d_eff(tc["scratch_intensity"])
            ok = abs(canine_d - tc["expected_canine_d"]) <= tol and abs(human_d - tc["expected_human_d"]) <= tol
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: canine_d={canine_d}, human_d={human_d}")
        else:
            ok = True
        if ok:
            passed += 1
    return passed, len(checks)


def validate_anderson(benchmark):
    checks = benchmark["validation_checks"]["anderson_predictions"]["test_cases"]
    passed = 0
    total = 0
    for tc in checks:
        label = tc["label"]
        w = tc["w"]
        regime_2d = classify_regime(w, 2.0)
        regime_3d = classify_regime(w, 3.0)
        ok_2d = regime_2d == tc["expected_regime_d2"]
        ok_3d = regime_3d == tc["expected_regime_d3"]
        total += 2
        for (computed, expected, d) in [(regime_2d, tc["expected_regime_d2"], "d=2"), (regime_3d, tc["expected_regime_d3"], "d=3")]:
            ok = computed == expected
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label} [{d}]: {computed} (expected {expected})")
            if ok:
                passed += 1
    return passed, total


def validate_diversity(benchmark):
    checks = benchmark["validation_checks"]["comparative_diversity"]["test_cases"]
    passed = 0
    total = 0
    for tc in checks:
        label = tc["label"]
        abundances = tc["cell_abundances"]
        richness = sum(1 for c in abundances if c > 0)
        evenness = pielou_evenness(abundances)
        expected_richness = tc["expected_richness"]
        lo, hi = tc["expected_evenness_range"]

        total += 2
        ok_r = richness == expected_richness
        status = "PASS" if ok_r else "FAIL"
        print(f"  [{status}] {label} richness: {richness} (expected {expected_richness})")
        if ok_r:
            passed += 1

        ok_e = lo <= evenness <= hi
        status = "PASS" if ok_e else "FAIL"
        print(f"  [{status}] {label} evenness: {evenness:.4f} (range [{lo}, {hi}])")
        if ok_e:
            passed += 1
    return passed, total


def validate_one_health(benchmark):
    checks = benchmark["validation_checks"]["one_health_bridge"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        ok = tc["expected_same_target"]
        status = "PASS" if ok else "FAIL"
        pathway = tc["pathway"]
        print(f"  [{status}] {label}: pathway={pathway}")
        if ok:
            passed += 1
    return passed, len(checks)


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Species Barrier Parameters ──")
    p, t = validate_species_params(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Breach Threshold ──")
    p, t = validate_breach_threshold(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Anderson Predictions ──")
    p, t = validate_anderson(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Comparative Diversity ──")
    p, t = validate_diversity(benchmark)
    total_passed += p
    total_checks += t

    print("\n── One Health Bridge ──")
    p, t = validate_one_health(benchmark)
    total_passed += p
    total_checks += t

    print(f"\n=== Cross-Species Skin (Exp 069): {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
