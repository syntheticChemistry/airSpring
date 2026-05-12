#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
airSpring Python Benchmark — Direct Comparison with Rust CPU

Measures wall-clock time for the same computations benchmarked by
`cargo run --release --bin bench_cpu_vs_python`, using the Python
implementations from the control/ baselines.

Usage:
    python3 scripts/bench_python_baselines.py
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Import control modules by adding their paths
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "control" / "fao56"))
sys.path.insert(0, str(ROOT / "control" / "yield_response"))
sys.path.insert(0, str(ROOT / "control" / "richards"))
sys.path.insert(0, str(ROOT / "control" / "cw2d"))

from penman_monteith import (
    saturation_vapour_pressure,
    slope_vapour_pressure_curve,
    atmospheric_pressure,
    psychrometric_constant,
    solar_declination,
)
from yield_response import (
    yield_ratio_single,
    yield_ratio_multistage,
    water_use_efficiency,
)
from richards_1d import (
    van_genuchten_theta as vg_theta,
    van_genuchten_K as vg_k,
    solve_richards_1d,
)

WARMUP = 2
MEASURE = 5

results = []


def bench(label, n, func):
    """Benchmark a callable, report throughput, accumulate results."""
    for _ in range(WARMUP):
        func()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        func()
    elapsed = time.perf_counter() - t0

    per_iter = elapsed / MEASURE
    throughput = n / per_iter if per_iter > 0 else float("inf")
    results.append({
        "label": label,
        "n": n,
        "per_iter_s": per_iter,
        "throughput": throughput,
    })
    print(f"  {label:<44} {n:>8} items  {per_iter:>10.4f}s/iter  {throughput:>12.0f} items/s")


# ---------------------------------------------------------------------------
# FAO-56 PM ET₀ computation (scalar loop, matching Rust scalar calls)
# ---------------------------------------------------------------------------

def compute_et0_scalar(n):
    """Replicate the Rust bench_cpu_vs_python ET₀ loop."""
    elev = 190.0
    lat = 42.5
    pressure = atmospheric_pressure(elev)
    gamma = psychrometric_constant(pressure)
    lat_rad = math.radians(lat)

    for i in range(n):
        day = float(i)
        tmax = 30.0 + 5.0 * math.sin(day * 0.017)
        tmin = 15.0 + 3.0 * math.cos(day * 0.017)
        tmean = (tmax + tmin) / 2.0
        rs = 18.0 + 4.0 * math.sin(day * 0.017)
        u2 = 2.0 + 0.5 * math.sin(day * 0.05)
        ea = saturation_vapour_pressure(tmin) * 0.6
        doy = (i % 365) + 1

        delta_svp = slope_vapour_pressure_curve(tmean)
        es = (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin)) / 2.0
        decl = solar_declination(doy)
        ws = math.acos(-math.tan(lat_rad) * math.tan(decl))
        dr = 1.0 + 0.033 * math.cos(2 * math.pi * doy / 365)
        ra = (24 * 60 / math.pi) * 0.0820 * dr * (
            ws * math.sin(lat_rad) * math.sin(decl)
            + math.cos(lat_rad) * math.cos(decl) * math.sin(ws)
        )
        rso = (0.75 + 2e-5 * elev) * ra
        rns = (1 - 0.23) * rs
        rnl = (4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
               * (0.34 - 0.14 * math.sqrt(ea))
               * (1.35 * (rs / max(rso, 0.01)) - 0.35))
        rn = rns - rnl
        g = 0.0
        _et0 = (
            (0.408 * delta_svp * (rn - g) + gamma * (900 / (tmean + 273)) * u2 * (es - ea))
            / (delta_svp + gamma * (1 + 0.34 * u2))
        )


# ---------------------------------------------------------------------------
# Yield response (Stewart 1977)
# ---------------------------------------------------------------------------

def yield_single_batch(n):
    for i in range(n):
        eta_etc = (i + 1) / (n + 1)
        yield_ratio_single(1.25, eta_etc)


def yield_multi_batch(n):
    stages_ky = [0.40, 1.50, 0.50, 0.20]
    stages_eta = [0.90, 0.85, 0.95, 0.98]
    for _ in range(n):
        yield_ratio_multistage(stages_ky, stages_eta)


def wue_batch(n):
    for i in range(n):
        y = 8000.0 + i * 0.04
        eta = 300.0 + i * 0.002
        water_use_efficiency(y, eta)


# ---------------------------------------------------------------------------
# Water balance + yield integration (140-day season)
# ---------------------------------------------------------------------------

def season_yield_batch(n):
    theta_fc, theta_wp, root_depth = 0.18, 0.08, 900.0
    taw = (theta_fc - theta_wp) * root_depth
    raw = taw * 0.55
    for i in range(n):
        dr = 0.0
        actual_sum = 0.0
        potential_sum = 0.0
        for day in range(140):
            ks = max(0.0, min(1.0, (taw - dr) / (taw - raw))) if dr > raw else 1.0
            et0 = 5.0 + math.sin(day * 0.04)
            etc = 1.2 * et0
            eta = ks * etc
            precip = 8.0 if (day + i) % 5 == 0 else 0.0
            dr = max(0.0, min(taw, dr - precip + eta))
            actual_sum += eta
            potential_sum += etc
        ratio = actual_sum / potential_sum
        yield_ratio_single(1.25, ratio)


# ---------------------------------------------------------------------------
# Van Genuchten retention (batch)
# ---------------------------------------------------------------------------

def vg_theta_batch(n, params):
    theta_r, theta_s, alpha, n_vg = params
    for i in range(n):
        h = -0.01 * (i + 1)
        vg_theta(h, theta_r, theta_s, alpha, n_vg)


# ---------------------------------------------------------------------------
# Richards equation
# ---------------------------------------------------------------------------

def richards_bench(n_nodes, params, duration_h, h_init, h_top, top_flux):
    solve_richards_1d(params, h_init, h_top, duration_h,
                      n_nodes=n_nodes, col_depth_cm=100.0)


# ---------------------------------------------------------------------------
# CW2D media parameters
# ---------------------------------------------------------------------------

CW2D_GRAVEL = {"theta_r": 0.025, "theta_s": 0.40, "alpha": 0.100,
                "n_vg": 3.00, "Ks_cm_day": 5000.0, "column_depth_cm": 60.0}
CW2D_ORGANIC = {"theta_r": 0.100, "theta_s": 0.60, "alpha": 0.050,
                 "n_vg": 1.50, "Ks_cm_day": 50.0, "column_depth_cm": 60.0}

SAND = {"theta_r": 0.045, "theta_s": 0.43, "alpha": 0.145,
        "n_vg": 2.68, "Ks_cm_day": 712.8, "column_depth_cm": 100.0}


def main():
    print("═══════════════════════════════════════════════════════════")
    print("  airSpring Python Benchmark — vs Rust CPU baseline")
    print("═══════════════════════════════════════════════════════════\n")

    # --- ET₀ ---
    print("── FAO-56 PM ET₀ computation (scalar Python) ──\n")
    for n in [100, 1_000, 10_000]:
        bench(f"ET₀ ({n} station-days)", n, lambda n=n: compute_et0_scalar(n))

    # --- VG retention ---
    print("\n── Van Genuchten retention (batch) ──\n")
    sand_p = (0.045, 0.43, 0.145, 2.68)
    for n in [1_000, 10_000, 100_000]:
        bench(f"VG theta ({n} evaluations)", n, lambda n=n: vg_theta_batch(n, sand_p))

    # --- Yield response ---
    print("\n── Yield response (Stewart 1977, single-stage) ──\n")
    for n in [1_000, 10_000, 100_000]:
        bench(f"Yield single ({n} evaluations)", n, lambda n=n: yield_single_batch(n))

    print("\n── Yield response (multi-stage, 4-stage corn) ──\n")
    for n in [1_000, 10_000, 100_000]:
        bench(f"Yield multi-stage ({n} corn seasons)", n, lambda n=n: yield_multi_batch(n))

    print("\n── Water use efficiency ──\n")
    for n in [1_000, 10_000, 100_000]:
        bench(f"WUE ({n} calculations)", n, lambda n=n: wue_batch(n))

    print("\n── Yield + WB integration (140-day season) ──\n")
    for n in [100, 1_000]:
        bench(f"Season yield ({n} scenarios)", n, lambda n=n: season_yield_batch(n))

    # --- Richards ---
    print("\n── Richards equation (1D infiltration) ──\n")
    for n_nodes in [20, 50]:
        bench(
            f"Richards 1D ({n_nodes} nodes, 0.1d)",
            n_nodes,
            lambda nn=n_nodes: solve_richards_1d(
                SAND, -20.0, 0.0, 2.4, n_nodes=nn),
        )

    # --- CW2D VG ---
    print("\n── CW2D VG retention (gravel + organic batch) ──\n")
    gravel_p = (0.025, 0.40, 0.100, 3.00)
    organic_p = (0.100, 0.60, 0.050, 1.50)
    for n in [10_000, 100_000]:
        bench(f"CW2D VG gravel ({n} evaluations)", n, lambda n=n: vg_theta_batch(n, gravel_p))
        bench(f"CW2D VG organic ({n} evaluations)", n, lambda n=n: vg_theta_batch(n, organic_p))

    # --- Summary ---
    print()
    print("═══════════════════════════════════════════════════════════")
    print("  Summary: Python throughput (items/s)")
    print("═══════════════════════════════════════════════════════════")
    for r in results:
        print(f"  {r['label']:<44}  {r['throughput']:>12.0f}")
    print()

    json_path = ROOT / "scripts" / "bench_python_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path.relative_to(ROOT)}")
    print("Compare with: cargo run --release --bin bench_cpu_vs_python")


if __name__ == "__main__":
    main()
