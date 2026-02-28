#!/usr/bin/env python3
"""
SCS Curve Number Runoff Method — Experiment 050.

Paper: USDA Soil Conservation Service (1972) National Engineering Handbook,
Section 4: Hydrology. USDA-SCS TR-55 (1986).

The SCS-CN method estimates direct runoff from rainfall:

    Q = (P - Ia)² / (P - Ia + S)    when P > Ia
    Q = 0                            when P ≤ Ia

where:
    S = (25400 / CN) - 254   [mm]   (potential maximum retention)
    Ia = λ × S               [mm]   (initial abstraction, λ=0.2 standard)
    CN = Curve Number (0-100, higher = more runoff)

Curve numbers depend on land use, hydrologic soil group (A-D), and antecedent
moisture condition (AMC I-III).

Data: All benchmarks from published tables. USDA public domain.
"""

import json
import sys
from pathlib import Path

BENCHMARK_FILE = Path(__file__).parent / "benchmark_scs_cn.json"

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, observed: float, expected: float, tol: float = 0.1) -> bool:
    global PASS_COUNT, FAIL_COUNT
    ok = abs(observed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{status}] {name}: observed={observed:.4f}, expected={expected:.4f}, tol={tol}")
    return ok


# ── Core SCS-CN equations ────────────────────────────────────────────────────

def potential_retention(cn: float) -> float:
    """Potential maximum retention S (mm) from curve number."""
    if cn <= 0:
        return float("inf")
    return (25400.0 / cn) - 254.0


def initial_abstraction(s_mm: float, ia_ratio: float = 0.2) -> float:
    """Initial abstraction Ia (mm)."""
    return ia_ratio * s_mm


def scs_cn_runoff(precip_mm: float, cn: float, ia_ratio: float = 0.2) -> float:
    """SCS Curve Number runoff Q (mm).

    Args:
        precip_mm: Precipitation (mm).
        cn: Curve number (0-100).
        ia_ratio: Initial abstraction ratio (default 0.2, updated 0.05).

    Returns:
        Direct runoff Q (mm).
    """
    if precip_mm <= 0 or cn <= 0:
        return 0.0
    s = potential_retention(cn)
    ia = initial_abstraction(s, ia_ratio)
    if precip_mm <= ia:
        return 0.0
    pe = precip_mm - ia
    return (pe * pe) / (pe + s)


def amc_cn_dry(cn_ii: float) -> float:
    """AMC-I (dry conditions) CN from AMC-II (Hawkins 1985)."""
    return cn_ii / (2.281 - 0.01281 * cn_ii)


def amc_cn_wet(cn_ii: float) -> float:
    """AMC-III (wet conditions) CN from AMC-II (Hawkins 1985)."""
    return cn_ii / (0.4036 + 0.0059 * cn_ii)


# ── Main validation ──────────────────────────────────────────────────────────

def main():
    with open(BENCHMARK_FILE) as f:
        bench = json.load(f)

    print("=" * 68)
    print("  Exp 050: SCS Curve Number Runoff — Python Control")
    print("=" * 68)

    # ── Analytical benchmarks ────────────────────────────────────────────
    print("\n── Analytical Benchmarks ───────────────────────────────────────\n")

    for case in bench["analytical_benchmarks"]:
        inputs = case["inputs"]
        expected_q = case["expected_Q_mm"]
        tol = case["tolerance"]
        cn = inputs["cn"]
        p = inputs["precip_mm"]
        ia_ratio = inputs.get("ia_ratio", 0.2)

        computed_q = scs_cn_runoff(p, cn, ia_ratio)
        check(case["name"], computed_q, expected_q, tol)

        # Also check S and Ia if provided
        if "S_mm" in case:
            computed_s = potential_retention(cn)
            check(f"{case['name']}_S", computed_s, case["S_mm"], 0.01)

    # ── AMC adjustment ───────────────────────────────────────────────────
    print("\n── Antecedent Moisture Condition Adjustments ────────────────────\n")

    for tc in bench["amc_adjustment"]["test_cases"]:
        cn_ii = tc["cn_ii"]
        cn_i = amc_cn_dry(cn_ii)
        cn_iii = amc_cn_wet(cn_ii)
        check(f"AMC-I CN_II={cn_ii}", cn_i, tc["expected_cn_i"], tc["tolerance"])
        check(f"AMC-III CN_II={cn_ii}", cn_iii, tc["expected_cn_iii"], tc["tolerance"])

    # ── Monotonicity checks ──────────────────────────────────────────────
    print("\n── Monotonicity Checks ─────────────────────────────────────────\n")

    # CN monotonic: higher CN → more runoff at P=50mm
    cns = [30, 50, 65, 75, 85, 90, 95, 98]
    qs_cn = [scs_cn_runoff(50.0, cn) for cn in cns]
    cn_mono = all(qs_cn[i] <= qs_cn[i + 1] for i in range(len(qs_cn) - 1))
    check("cn_monotonic", float(cn_mono), 1.0, 0.0)

    # Precip monotonic: higher P → more runoff at CN=75
    ps = [0, 10, 20, 30, 50, 75, 100, 150]
    qs_p = [scs_cn_runoff(p, 75) for p in ps]
    p_mono = all(qs_p[i] <= qs_p[i + 1] for i in range(len(qs_p) - 1))
    check("precip_monotonic", float(p_mono), 1.0, 0.0)

    # Q ≤ P always
    q_leq_p = all(scs_cn_runoff(p, cn) <= p + 0.001 for p in ps for cn in cns)
    check("Q_leq_P", float(q_leq_p), 1.0, 0.0)

    # Higher Ia ratio → less runoff
    q_02 = scs_cn_runoff(50.0, 75, 0.2)
    q_05 = scs_cn_runoff(50.0, 75, 0.05)
    check("ia_ratio_monotonic", float(q_05 > q_02), 1.0, 0.0)

    # ── CN table validation ──────────────────────────────────────────────
    print("\n── CN Table Soil Group Ordering ─────────────────────────────────\n")

    cn_table = bench["cn_table"]["land_use"]
    for land_use, groups in cn_table.items():
        cn_a = groups["A"]
        cn_d = groups["D"]
        check(f"{land_use}_A<D", float(cn_a <= cn_d), 1.0, 0.0)

    # ── Extreme values ───────────────────────────────────────────────────
    print("\n── Edge Cases ──────────────────────────────────────────────────\n")

    check("cn100_all_runoff", scs_cn_runoff(50.0, 100), 50.0, 0.01)
    check("cn1_no_runoff", scs_cn_runoff(50.0, 1), 0.0, 0.01)

    # ── Summary ──────────────────────────────────────────────────────────
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'=' * 68}")
    print(f"  SCS Curve Number: {PASS_COUNT}/{total} PASS, {FAIL_COUNT} FAIL")
    print(f"{'=' * 68}")

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
