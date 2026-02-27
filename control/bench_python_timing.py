#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Python timing harness for CPU vs Rust benchmark.

Runs the same algorithms as barracuda's bench_cpu_vs_python at identical
scale, outputting JSON timing results for direct comparison.
"""

import json
import math
import sys
import time


def bench(name, func, n, *args):
    """Run func n times, return (name, n, total_secs, result_sample)."""
    # Warm up
    sample = func(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        func(*args)
    elapsed = time.perf_counter() - t0
    return {"name": name, "n": n, "secs": elapsed, "sample": sample}


# ─── FAO-56 PM ET₀ ──────────────────────────────────────────────────────────

def fao56_et0(t_max, t_min, rh_mean, u2, r_s, lat_rad, doy, altitude):
    t_mean = (t_max + t_min) / 2.0
    P = 101.3 * ((293.0 - 0.0065 * altitude) / 293.0) ** 5.26
    gamma = 0.000665 * P
    e_sat_max = 0.6108 * math.exp(17.27 * t_max / (t_max + 237.3))
    e_sat_min = 0.6108 * math.exp(17.27 * t_min / (t_min + 237.3))
    e_s = (e_sat_max + e_sat_min) / 2.0
    e_a = e_s * rh_mean / 100.0
    delta = 4098.0 * 0.6108 * math.exp(17.27 * t_mean / (t_mean + 237.3)) / (t_mean + 237.3) ** 2
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    decl = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(decl))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(decl) +
        math.cos(lat_rad) * math.cos(decl) * math.sin(ws)
    )
    rso = (0.75 + 2e-5 * altitude) * ra
    rns = (1.0 - 0.23) * r_s
    rnl = 4.903e-9 * ((t_max + 273.16) ** 4 + (t_min + 273.16) ** 4) / 2.0 * \
          (0.34 - 0.14 * math.sqrt(e_a)) * (1.35 * r_s / max(rso, 0.01) - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900.0 / (t_mean + 273.0) * u2 * (e_s - e_a)) / \
          (delta + gamma * (1.0 + 0.34 * u2))
    return max(et0, 0.0)


# ─── Thornthwaite ────────────────────────────────────────────────────────────

def thornthwaite_pet(monthly_temps, lat):
    I = sum(max(0, (t / 5.0) ** 1.514) for t in monthly_temps)
    if I <= 0:
        return [0.0] * 12
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    pet = []
    for m, t in enumerate(monthly_temps):
        if t <= 0:
            pet.append(0.0)
            continue
        p = 16.0 * (10.0 * t / I) ** a
        doy_mid = 15 + m * 30
        decl = 0.409 * math.sin(2 * math.pi * doy_mid / 365.0 - 1.39)
        lat_r = math.radians(lat)
        cos_ws = -math.tan(lat_r) * math.tan(decl)
        cos_ws = max(-1.0, min(1.0, cos_ws))
        ws = math.acos(cos_ws)
        daylight = 24.0 * ws / math.pi
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
        pet.append(p * (daylight / 12.0) * (days / 30.0))
    return pet


# ─── Hargreaves-Samani ───────────────────────────────────────────────────────

def hargreaves_et0(t_max, t_min, ra):
    t_mean = (t_max + t_min) / 2.0
    return 0.0023 * ra * (t_mean + 17.8) * max(0, t_max - t_min) ** 0.5


# ─── Van Genuchten θ(h) ─────────────────────────────────────────────────────

def vg_theta(h, theta_r, theta_s, alpha, n):
    if h >= 0:
        return theta_s
    m = 1.0 - 1.0 / n
    se = 1.0 / (1.0 + (alpha * abs(h)) ** n) ** m
    return theta_r + (theta_s - theta_r) * se


# ─── Water balance step ─────────────────────────────────────────────────────

def daily_water_balance(dr_prev, precip, irrig, et0, kc, ks, taw):
    etc = et0 * kc
    actual_et = etc * ks
    new_dr = dr_prev - precip - irrig + actual_et
    dp = 0.0
    if new_dr < 0:
        dp = -new_dr
        new_dr = 0.0
    new_dr = min(new_dr, taw)
    return new_dr, actual_et, dp


# ─── Anderson coupling ──────────────────────────────────────────────────────

def anderson_coupling(theta, theta_r, theta_s):
    se = max(0, min(1, (theta - theta_r) / (theta_s - theta_r))) if theta_s > theta_r else 0
    pc = se ** 0.5 if se > 0 else 0
    z = 6.0 * pc
    d_eff = z / 2.0
    w = 12.0 * (1.0 - se)
    return d_eff, w


# ─── Shannon diversity ──────────────────────────────────────────────────────

def shannon_diversity(abundances):
    total = sum(abundances)
    if total <= 0:
        return 0.0
    h = 0.0
    for a in abundances:
        if a > 0:
            p = a / total
            h -= p * math.log(p)
    return h


# ─── Main benchmark ─────────────────────────────────────────────────────────

def main():
    results = []
    N = 10_000

    # ET₀ PM
    results.append(bench("fao56_et0", fao56_et0, N,
                         34.8, 19.6, 65.0, 1.8, 20.5, math.radians(42.0), 180, 200.0))

    # Thornthwaite
    temps = [2.0, 4.0, 9.0, 14.0, 19.0, 24.0, 27.0, 26.0, 22.0, 15.0, 8.0, 3.0]
    results.append(bench("thornthwaite", thornthwaite_pet, N, temps, 42.0))

    # Hargreaves
    results.append(bench("hargreaves", hargreaves_et0, N, 34.8, 19.6, 38.5))

    # VG theta — 100K
    results.append(bench("van_genuchten", vg_theta, N * 10,
                         -100.0, 0.078, 0.43, 0.036, 1.56))

    # Water balance step — 10K
    results.append(bench("water_balance_step", daily_water_balance, N,
                         20.0, 5.0, 0.0, 4.5, 1.05, 0.9, 120.0))

    # Anderson coupling — 100K
    results.append(bench("anderson_coupling", anderson_coupling, N * 10,
                         0.25, 0.078, 0.43))

    # Shannon diversity — 10K
    abun = [45.0, 30.0, 15.0, 8.0, 2.0]
    results.append(bench("shannon_diversity", shannon_diversity, N, abun))

    # Season simulation (153-day water balance)
    def run_season():
        dr = 0.0
        taw = 120.0
        for d in range(153):
            et0 = 3.0 + 2.0 * math.sin(2 * math.pi * d / 153)
            p = 2.0 if d % 7 == 0 else 0.0
            ks = max(0, (taw - dr) / (taw - 60.0)) if dr > 60.0 else 1.0
            dr, _, _ = daily_water_balance(dr, p, 0.0, et0, 1.0, ks, taw)
        return dr
    results.append(bench("season_simulation", run_season, N // 10))

    out = {"benchmarks": results}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
