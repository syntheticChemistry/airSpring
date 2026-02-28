#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exp 041 — metalForge Mixed-Hardware Dispatch Validation (Python control).

Simulates metalForge's capability-based workload routing. Validates that:
1. GPU workloads route to GPU (f64 + shader dispatch)
2. NPU workloads route to NPU (quantized inference)
3. CPU workloads route to CPU (sequential I/O-bound)
4. Priority chain: GPU > NPU > Neural > CPU
5. Graceful fallback when preferred substrate is unavailable

References:
  metalForge substrate model (forge/src/substrate.rs)
  metalForge dispatch logic (forge/src/dispatch.rs)
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(SCRIPT_DIR, "benchmark_metalforge_dispatch.json")

# ── Simulated metalForge substrate model ─────────────────────────────────────

WORKLOAD_CAPS = {
    # GPU batch compute (absorbed by ToadStool)
    "et0_batch": {"F64", "ShaderDispatch"},
    "water_balance_batch": {"F64", "ShaderDispatch"},
    "richards_pde": {"F64", "ShaderDispatch"},
    "yield_response_surface": {"F64", "ShaderDispatch"},
    "monte_carlo_uq": {"F64", "ShaderDispatch"},
    "isotherm_batch": {"F64", "ShaderDispatch"},
    "gdd_accumulate": {"F64", "ShaderDispatch"},
    "dual_kc_batch": {"F64", "ShaderDispatch"},
    "forecast_scheduling": {"F64", "ShaderDispatch"},
    # Tier B GPU orchestrators (pending absorption)
    "hargreaves_et0_batch": {"F64", "ShaderDispatch"},
    "kc_climate_batch": {"F64", "ShaderDispatch"},
    "sensor_calibration_batch": {"F64", "ShaderDispatch"},
    "seasonal_pipeline": {"F64", "ShaderDispatch"},
    # NPU-native classifiers (AKD1000 int8)
    "crop_stress_classifier": {"QuantizedInference"},
    "irrigation_decision": {"QuantizedInference"},
    "sensor_anomaly": {"QuantizedInference"},
    # CPU-only domains
    "validation_harness": {"CpuCompute"},
    "weather_ingest": {"CpuCompute"},
}

SUBSTRATE_CAPS = {
    "GPU": {"F64", "F32", "ShaderDispatch", "ScalarReduce", "TimestampQuery"},
    "NPU": {"F32", "QuantizedInference", "BatchInference"},
    "Neural": {"F64", "F32", "NeuralApiRoute"},
    "CPU": {"F64", "F32", "CpuCompute", "SimdVector", "ShaderDispatch"},
}

PRIORITY = ["GPU", "NPU", "Neural", "CPU"]


def route(workload, available_substrates):
    """Route a workload to the highest-priority capable substrate."""
    required = WORKLOAD_CAPS.get(workload, set())
    for substrate in PRIORITY:
        if substrate not in available_substrates:
            continue
        caps = SUBSTRATE_CAPS.get(substrate, set())
        if required.issubset(caps):
            return substrate
    return None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_workload_routing(benchmark):
    print("\n[Workload Routing]")
    tests = benchmark["validation_checks"]["workload_routing"]["test_cases"]
    passed, total = 0, 0
    all_substrates = set(SUBSTRATE_CAPS.keys())

    for tc in tests:
        wl = tc["workload"]
        expected = tc["expected_substrate"]
        total += 1
        actual = route(wl, all_substrates)
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {wl} → {actual} (expected {expected})")
        if ok: passed += 1

    return passed, total


def validate_priority_chain(benchmark):
    print("\n[Priority Chain]")
    tests = benchmark["validation_checks"]["priority_chain"]["test_cases"]
    passed, total = 0, 0
    all_substrates = set(SUBSTRATE_CAPS.keys())

    for tc in tests:
        label = tc["label"]
        expected = tc["expected"]
        caps = set(tc["caps"])
        total += 1
        for substrate in PRIORITY:
            if substrate not in all_substrates:
                continue
            if caps.issubset(SUBSTRATE_CAPS[substrate]):
                actual = substrate
                break
        else:
            actual = None

        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {label}: → {actual} (expected {expected})")
        if ok: passed += 1

    return passed, total


def validate_fallback(benchmark):
    print("\n[Fallback Behavior]")
    tests = benchmark["validation_checks"]["fallback_behavior"]["test_cases"]
    passed, total = 0, 0

    for tc in tests:
        label = tc["label"]
        wl = tc["workload"]
        available = set(tc["available"])
        should_route = tc["should_route"]
        total += 1
        result = route(wl, available)
        ok = (result is not None) == should_route
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {label}: routed={result} (should_route={should_route})")
        if ok: passed += 1

    return passed, total


def validate_inventory(benchmark):
    print("\n[Inventory Completeness]")
    expected = benchmark["validation_checks"]["inventory_completeness"]["expected_count"]
    passed, total = 0, 0

    total += 1
    actual = len(WORKLOAD_CAPS)
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {actual} workloads defined (expected {expected})")
    if ok: passed += 1

    total += 1
    all_have_caps = all(len(caps) > 0 for caps in WORKLOAD_CAPS.values())
    status = "PASS" if all_have_caps else "FAIL"
    print(f"  {status} All workloads have capability requirements")
    if all_have_caps: passed += 1

    return passed, total


def main():
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    print("Exp 041: metalForge Mixed-Hardware Dispatch Validation")
    print(f"Benchmark: {BENCHMARK}")

    total_pass, total_tests = 0, 0

    p, t = validate_workload_routing(benchmark)
    total_pass += p; total_tests += t

    p, t = validate_priority_chain(benchmark)
    total_pass += p; total_tests += t

    p, t = validate_fallback(benchmark)
    total_pass += p; total_tests += t

    p, t = validate_inventory(benchmark)
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
