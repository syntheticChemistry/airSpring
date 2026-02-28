#!/usr/bin/env python3
"""
Green-Ampt (1911) Infiltration Model — Experiment 051.

Paper: Green WH, Ampt GA (1911) Studies on Soil Physics: Flow of air and water
through soils. J Agricultural Science 4(1):1-24.

Parameters: Rawls WJ, Brakensiek DL, Miller N (1983) Green-Ampt infiltration
parameters from soils data. J Hydraul Eng 109(1):62-70.

The Green-Ampt model estimates cumulative infiltration F(t) and infiltration
rate f(t) from soil hydraulic properties:

    F(t) = Ks × t + ψ × Δθ × ln(1 + F(t)/(ψ × Δθ))   [implicit]
    f(t) = Ks × (1 + ψ × Δθ / F(t))                     [rate from cumulative]

where:
    Ks = saturated hydraulic conductivity (cm/hr)
    ψ  = wetting front suction head (cm, positive)
    Δθ = θs - θi (moisture deficit)

Ponding time under constant rainfall intensity i:
    tp = Ks × ψ × Δθ / (i × (i - Ks))    when i > Ks

Data: Rawls et al. (1983) Table 1 soil parameters (open literature).
"""

import json
import math
import sys
from pathlib import Path

BENCHMARK_FILE = Path(__file__).parent / "benchmark_green_ampt.json"

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


# ── Core Green-Ampt equations ────────────────────────────────────────────────

def green_ampt_cumulative(ks: float, psi: float, delta_theta: float,
                          t_hr: float, max_iter: int = 100, tol: float = 1e-8) -> float:
    """Solve implicit Green-Ampt equation for cumulative infiltration F(t) (cm).

    Uses Newton-Raphson iteration on:
        g(F) = F - Ks*t - ψ*Δθ*ln(1 + F/(ψ*Δθ)) = 0

    Args:
        ks: Saturated hydraulic conductivity (cm/hr).
        psi: Wetting front suction head (cm, positive).
        delta_theta: Moisture deficit θs - θi.
        t_hr: Time (hours).

    Returns:
        Cumulative infiltration F (cm).
    """
    if t_hr <= 0:
        return 0.0

    psi_dt = psi * delta_theta

    # Initial guess: F ≈ Ks * t (steady-state infiltration)
    f_guess = ks * t_hr + math.sqrt(2.0 * ks * psi_dt * t_hr)

    for _ in range(max_iter):
        if f_guess <= 0:
            f_guess = ks * t_hr * 0.01
        g = f_guess - ks * t_hr - psi_dt * math.log(1.0 + f_guess / psi_dt)
        dg = 1.0 - psi_dt / (psi_dt + f_guess)
        if abs(dg) < 1e-15:
            break
        f_new = f_guess - g / dg
        if f_new < 0:
            f_new = f_guess * 0.5
        if abs(f_new - f_guess) < tol:
            f_guess = f_new
            break
        f_guess = f_new

    return max(0.0, f_guess)


def green_ampt_rate(ks: float, psi: float, delta_theta: float, f_cum: float) -> float:
    """Green-Ampt infiltration rate f (cm/hr) from cumulative F.

    f = Ks × (1 + ψ×Δθ/F)

    At F=0 the rate is theoretically infinite; we return a large value.
    """
    if f_cum <= 0:
        return float("inf")
    return ks * (1.0 + psi * delta_theta / f_cum)


def ponding_time(ks: float, psi: float, delta_theta: float,
                 rain_intensity: float) -> float:
    """Time to ponding (hr) under constant rainfall intensity.

    tp = Ks × ψ × Δθ / (i × (i - Ks))

    Only valid when i > Ks. If i ≤ Ks, ponding never occurs.
    """
    if rain_intensity <= ks:
        return float("inf")
    return ks * psi * delta_theta / (rain_intensity * (rain_intensity - ks))


# ── Main validation ──────────────────────────────────────────────────────────

def main():
    with open(BENCHMARK_FILE) as f:
        bench = json.load(f)

    print("=" * 68)
    print("  Exp 051: Green-Ampt (1911) Infiltration — Python Control")
    print("=" * 68)

    # ── Analytical benchmarks ────────────────────────────────────────────
    print("\n── Analytical Benchmarks ───────────────────────────────────────\n")

    for case in bench["analytical_benchmarks"]:
        inputs = case["inputs"]
        ks = inputs["Ks_cm_hr"]
        psi = inputs["psi_cm"]
        dt = inputs["delta_theta"]

        if "t_hr" in inputs:
            t = inputs["t_hr"]

            if "expected_F_cm" in case and not case.get("expected_f_infinite", False):
                expected_f = case["expected_F_cm"]
                tol = case["tolerance"]
                computed_f = green_ampt_cumulative(ks, psi, dt, t)
                check(f"{case['name']}_F", computed_f, expected_f, tol)

            if "expected_f_cm_hr" in case:
                f_cum = green_ampt_cumulative(ks, psi, dt, t)
                if f_cum > 0:
                    computed_rate = green_ampt_rate(ks, psi, dt, f_cum)
                    check(f"{case['name']}_f", computed_rate,
                          case["expected_f_cm_hr"], case["tolerance"])

            if case.get("expected_f_approaches_Ks", False):
                f_cum = green_ampt_cumulative(ks, psi, dt, t)
                if f_cum > 0:
                    rate = green_ampt_rate(ks, psi, dt, f_cum)
                    ratio = rate / ks
                    tol_r = case.get("tolerance_ratio", 0.05)
                    check(f"{case['name']}_ratio", ratio, 1.0, tol_r)

            if t == 0.0 and case.get("expected_f_infinite", False):
                check(f"{case['name']}_F_zero", green_ampt_cumulative(ks, psi, dt, 0.0), 0.0, 0.001)

        if "rain_intensity_cm_hr" in inputs:
            i_rain = inputs["rain_intensity_cm_hr"]
            expected_tp = case["expected_tp_hr"]
            computed_tp = ponding_time(ks, psi, dt, i_rain)
            check(f"{case['name']}_tp", computed_tp, expected_tp, case["tolerance"])

    # ── Soil parameter table validation ──────────────────────────────────
    print("\n── Soil Parameter Table (Rawls 1983) ───────────────────────────\n")

    soils = bench["soil_parameters"]
    for soil_name in soils:
        if not isinstance(soils[soil_name], dict):
            continue
        params = soils[soil_name]
        ks = params["Ks_cm_hr"]
        psi = params["psi_cm"]
        porosity = params["porosity"]
        # All soils: Ks > 0, ψ > 0, porosity in (0.3, 0.6)
        check(f"{soil_name}_Ks_positive", float(ks > 0), 1.0, 0.0)
        check(f"{soil_name}_psi_positive", float(psi > 0), 1.0, 0.0)
        check(f"{soil_name}_porosity_physical", float(0.3 < porosity < 0.6), 1.0, 0.0)

    # ── Monotonicity checks ──────────────────────────────────────────────
    print("\n── Monotonicity Checks ─────────────────────────────────────────\n")

    # Cumulative F is increasing with time
    ks, psi, dt = 1.09, 11.01, 0.312
    times = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
    f_vals = [green_ampt_cumulative(ks, psi, dt, t) for t in times]
    f_mono = all(f_vals[i] <= f_vals[i + 1] for i in range(len(f_vals) - 1))
    check("cumulative_monotonic", float(f_mono), 1.0, 0.0)

    # Rate f is decreasing with time
    rates = [green_ampt_rate(ks, psi, dt, f) for f in f_vals if f > 0]
    rate_dec = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
    check("rate_decreasing", float(rate_dec), 1.0, 0.0)

    # Rate bounded below by Ks
    rate_geq_ks = all(r >= ks - 1e-10 for r in rates)
    check("rate_bounded_below_by_Ks", float(rate_geq_ks), 1.0, 0.0)

    # Higher Ks → more infiltration
    f_sand = green_ampt_cumulative(11.78, 4.95, 0.367, 1.0)
    f_clay = green_ampt_cumulative(0.03, 31.63, 0.285, 1.0)
    check("higher_Ks_more_infiltration", float(f_sand > f_clay), 1.0, 0.0)

    # ── Summary ──────────────────────────────────────────────────────────
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'=' * 68}")
    print(f"  Green-Ampt: {PASS_COUNT}/{total} PASS, {FAIL_COUNT} FAIL")
    print(f"{'=' * 68}")

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
