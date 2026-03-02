# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 067: CytokineBrain Interface Validation — Python Control Baseline

The CytokineBrain uses bingocube-nautilus (Rust-only evolutionary reservoir).
Python cannot replicate the reservoir computation, so this control validates:
  1. Input normalization bounds (feature scaling)
  2. Prediction → Anderson regime classification
  3. Data profile structure (Gonzales G3/G4 time series)

The Rust validation binary (`validate_cytokine`) performs the full lifecycle
tests (observe → train → predict → export → import).

References:
  - Paper 12: Immunological Anderson
  - Gonzales AJ et al. (2016) Vet Dermatol 27:34-e10
  - Fleck TJ, Gonzales AJ (2021) Vet Dermatol 32:681-e182
"""
import json
import sys
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "benchmark_cytokine_brain.json"

NORMALIZERS = {
    "time_hours": 720.0,
    "il31_level": 500.0,
    "il4_level": 200.0,
    "il13_level": 200.0,
    "pruritus_score": 10.0,
    "tewl": 100.0,
    "pielou_evenness": 1.0,
}


def normalize_feature(name, value):
    return value / NORMALIZERS[name]


def classify_regime(signal_extent, barrier_integrity):
    if signal_extent > 0.7 and barrier_integrity < 0.3:
        return "Extended"
    elif signal_extent < 0.3 and barrier_integrity > 0.7:
        return "Localized"
    else:
        return "Critical"


def validate_normalization(benchmark):
    checks = benchmark["validation_checks"]["input_normalization"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        expected = tc["expected_normalized"]
        tol = tc["tolerance"]
        for key in NORMALIZERS:
            if key in tc:
                computed = normalize_feature(key, tc[key])
                ok = abs(computed - expected) <= tol
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {label}: {computed:.6f} (expected {expected}, tol {tol})")
                if ok:
                    passed += 1
                break
    return passed, len(checks)


def validate_regime(benchmark):
    checks = benchmark["validation_checks"]["prediction_regime"]["test_cases"]
    passed = 0
    for tc in checks:
        label = tc["label"]
        expected = tc["expected_regime"]
        computed = classify_regime(tc["signal_extent"], tc["barrier_integrity"])
        ok = computed == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {computed} (expected {expected})")
        if ok:
            passed += 1
    return passed, len(checks)


def validate_data_profile(benchmark):
    checks = benchmark["validation_checks"]["gonzales_data_profile"]["test_cases"]
    passed = 0
    total = 0
    for tc in checks:
        label = tc["label"]
        n_features = tc["expected_n_features"]
        n_targets = tc["expected_n_targets"]
        n_points = len(tc["time_points_hours"])

        ok_feat = n_features == 7
        ok_targ = n_targets == 3
        ok_pts = n_points > 0

        ok = ok_feat and ok_targ and ok_pts
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {n_points} time points, {n_features} features, {n_targets} targets")
        total += 1
        if ok:
            passed += 1
    return passed, total


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_checks = 0

    print("\n── Input Normalization ──")
    p, t = validate_normalization(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Prediction Regime Classification ──")
    p, t = validate_regime(benchmark)
    total_passed += p
    total_checks += t

    print("\n── Gonzales Data Profile ──")
    p, t = validate_data_profile(benchmark)
    total_passed += p
    total_checks += t

    print(f"\n=== CytokineBrain Interface (Exp 067): {total_passed}/{total_checks} PASS ===")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
