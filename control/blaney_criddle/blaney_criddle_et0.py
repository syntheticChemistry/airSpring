#!/usr/bin/env python3
"""
Blaney-Criddle (1950) PET estimation — Experiment 049.

Paper: Blaney HF, Criddle WD (1950) Determining water requirements in irrigated
areas from climatological and irrigation data. USDA-SCS Tech Paper 96.

The original Blaney-Criddle equation estimates monthly PET from temperature and
daylight hours only:

    ET₀ = p × (0.46 × T + 8.13)   [mm/day]

where:
    p = mean daily percentage of annual daytime hours (depends on latitude + month)
    T = mean monthly temperature (°C)

This is the simplest widely-used ET₀ method, requiring only temperature and location.
FAO-24 added humidity/wind corrections (not implemented here — we validate the
original 1950 form used in western US irrigation districts).

Data: All benchmarks from published equations. Open literature, no proprietary data.
"""

import json
import math
import sys
from pathlib import Path

BENCHMARK_FILE = Path(__file__).parent / "benchmark_blaney_criddle.json"

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, observed: float, expected: float, tol: float = 0.02) -> bool:
    global PASS_COUNT, FAIL_COUNT
    ok = abs(observed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{status}] {name}: observed={observed:.4f}, expected={expected:.4f}, tol={tol}")
    return ok


# ── Core equation ────────────────────────────────────────────────────────────

def blaney_criddle_et0(tmean_c: float, p: float) -> float:
    """Original Blaney-Criddle (1950) ET₀ estimate (mm/day).

    Args:
        tmean_c: Mean monthly temperature (°C).
        p: Mean daily percentage of annual daytime hours (0-1 fraction).

    Returns:
        ET₀ in mm/day (clamped to >= 0).
    """
    return max(0.0, p * (0.46 * tmean_c + 8.13))


def daylight_hours(latitude_rad: float, doy: int) -> float:
    """Compute daylight hours N from latitude and day-of-year (FAO-56 Eq. 34)."""
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)  # noqa: F841
    delta = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.405)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(latitude_rad) * math.tan(delta))))
    return (24.0 / math.pi) * ws


def daylight_fraction(latitude_deg: float, doy: int) -> float:
    """Compute Blaney-Criddle p factor from latitude and DOY.

    p = N / 43.80, where N is daylight hours and 43.80 ≈ 4380/100.
    Total annual daylight is ~4380 hrs at any latitude (summer/winter cancel).
    This gives p ≈ 0.274 at equator (12/43.80), matching FAO-24 Table 18.
    """
    lat_rad = math.radians(latitude_deg)
    N = daylight_hours(lat_rad, doy)
    return N / 43.80


def blaney_criddle_from_location(tmean_c: float, latitude_deg: float, doy: int) -> float:
    """Blaney-Criddle with daylight computed from latitude + DOY."""
    p = daylight_fraction(latitude_deg, doy)
    return blaney_criddle_et0(tmean_c, p)


# ── Antecedent Moisture Condition for SCS-CN (shared utility) ─────────────

def amc_cn_dry(cn_ii: float) -> float:
    """AMC-I (dry) curve number from AMC-II."""
    return 4.2 * cn_ii / (10.0 - 0.058 * cn_ii)


def amc_cn_wet(cn_ii: float) -> float:
    """AMC-III (wet) curve number from AMC-II."""
    return 23.0 * cn_ii / (10.0 + 0.13 * cn_ii)


# ── Main validation ──────────────────────────────────────────────────────────

def main():
    with open(BENCHMARK_FILE) as f:
        bench = json.load(f)

    print("=" * 68)
    print("  Exp 049: Blaney-Criddle (1950) PET — Python Control")
    print("=" * 68)

    # ── Analytical benchmarks ────────────────────────────────────────────
    print("\n── Analytical Benchmarks ───────────────────────────────────────\n")

    for case in bench["analytical_benchmarks"]:
        inputs = case["inputs"]
        expected = case["expected_et0_mm_day"]
        tol = case["tolerance"]
        computed = blaney_criddle_et0(inputs["tmean_c"], inputs["p"])
        check(case["name"], computed, expected, tol)

    # ── Daylight fraction computation ────────────────────────────────────
    print("\n── Daylight Fraction from Location ─────────────────────────────\n")

    # Summer solstice at 40°N: p should be in 0.30-0.35 range (long days)
    p_summer = daylight_fraction(40.0, 172)  # June 21
    check("p_summer_40N", p_summer, 0.333, 0.015)

    # Winter solstice at 40°N: p should be in 0.20-0.23 range (short days)
    p_winter = daylight_fraction(40.0, 356)  # Dec 22
    check("p_winter_40N", p_winter, 0.222, 0.015)

    # Equator: p should be ~0.274 year-round
    p_equator = daylight_fraction(0.0, 172)
    check("p_equator", p_equator, 0.274, 0.005)

    # ── Monotonicity ─────────────────────────────────────────────────────
    print("\n── Monotonicity Checks ─────────────────────────────────────────\n")

    # Temperature monotonic at constant p
    temps = [-10, 0, 10, 20, 30, 40]
    et0_temp = [blaney_criddle_et0(t, 0.274) for t in temps]
    temp_mono = all(et0_temp[i] <= et0_temp[i + 1] for i in range(len(et0_temp) - 1))
    check("temperature_monotonic", float(temp_mono), 1.0, 0.0)

    # Daylight monotonic at constant T
    ps = [0.199, 0.222, 0.274, 0.333, 0.366]
    et0_p = [blaney_criddle_et0(25.0, p) for p in ps]
    p_mono = all(et0_p[i] <= et0_p[i + 1] for i in range(len(et0_p) - 1))
    check("daylight_monotonic", float(p_mono), 1.0, 0.0)

    # Summer p > winter p for 40°N
    check("summer_gt_winter_p", float(p_summer > p_winter), 1.0, 0.0)

    # ── Cross-method comparison ──────────────────────────────────────────
    print("\n── Cross-Method Comparison (Michigan July) ─────────────────────\n")

    bc_michigan = blaney_criddle_from_location(22.0, 42.7, 195)  # mid-July
    check("bc_michigan_july_range_low", float(bc_michigan > 5.0), 1.0, 0.0)
    check("bc_michigan_july_range_high", float(bc_michigan < 7.5), 1.0, 0.0)
    print(f"    BC ET₀ = {bc_michigan:.2f} mm/day (Michigan July, T=22°C)")

    # ── Non-negative constraint ──────────────────────────────────────────
    print("\n── Non-Negative Constraint ─────────────────────────────────────\n")
    check("non_negative_at_minus20", blaney_criddle_et0(-20.0, 0.199), 0.0, 0.01)
    check("non_negative_at_minus30", blaney_criddle_et0(-30.0, 0.199), 0.0, 0.01)

    # ── Summary ──────────────────────────────────────────────────────────
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'=' * 68}")
    print(f"  Blaney-Criddle: {PASS_COUNT}/{total} PASS, {FAIL_COUNT} FAIL")
    print(f"{'=' * 68}")

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
