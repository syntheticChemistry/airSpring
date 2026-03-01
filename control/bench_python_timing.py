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

    # SCS-CN Runoff — 100K
    def scs_cn_runoff(P, CN):
        if CN <= 0 or CN >= 100:
            return 0.0
        S = 25400.0 / CN - 254.0
        Ia = 0.2 * S
        return ((P - Ia) ** 2 / (P - Ia + S)) if P > Ia else 0.0
    results.append(bench("scs_cn_runoff", scs_cn_runoff, N * 10, 50.0, 75.0))

    # Green-Ampt infiltration — 100K (Newton-Raphson)
    def green_ampt_F(ks, psi_dt, t):
        if t <= 0:
            return 0.0
        F = ks * t
        for _ in range(50):
            F_new = ks * t + psi_dt * math.log(1 + F / psi_dt)
            if abs(F_new - F) < 1e-10:
                break
            F = F_new
        return F
    results.append(bench("green_ampt", green_ampt_F, N * 10,
                         1.09, 11.01 * 0.34, 1.0))

    # Saxton-Rawls pedotransfer — 100K
    def saxton_rawls_fc(sand, clay, om):
        S, C, OM = sand * 100, clay * 100, om
        fc_33t = (-0.251 * S + 0.195 * C + 0.011 * OM +
                  0.006 * S * OM - 0.027 * C * OM +
                  0.452 * S * C / 100 + 0.299)
        fc_33 = fc_33t + (1.283 * fc_33t**2 - 0.374 * fc_33t - 0.015)
        return fc_33
    results.append(bench("saxton_rawls", saxton_rawls_fc, N * 10,
                         0.40, 0.20, 2.5))

    # Langmuir isotherm fit — 10K
    def fit_langmuir(Ce, qe):
        Ce_qe = [c / q for c, q in zip(Ce, qe) if q > 0]
        Ce_f = [c for c, q in zip(Ce, qe) if q > 0]
        n = len(Ce_f)
        sx = sum(Ce_f)
        sy = sum(Ce_qe)
        sxy = sum(x * y for x, y in zip(Ce_f, Ce_qe))
        sxx = sum(x * x for x in Ce_f)
        slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
        intercept = (sy - slope * sx) / n
        qmax = 1.0 / slope
        KL = slope / intercept
        y_mean = sy / n
        ss_tot = sum((y - y_mean) ** 2 for y in Ce_qe)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(Ce_f, Ce_qe))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return r2
    Ce = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0]
    qe = [2.8, 4.9, 8.5, 11.2, 14.0, 16.1, 17.0, 17.6, 17.8]
    results.append(bench("langmuir_fit", fit_langmuir, N, Ce, qe))

    # Priestley-Taylor ET₀ — 10K
    def priestley_taylor(Rn, G, T, alt):
        P = 101.3 * ((293 - 0.0065 * alt) / 293) ** 5.26
        gamma = 0.000665 * P
        delta = 4098 * 0.6108 * math.exp(17.27 * T / (T + 237.3)) / (T + 237.3) ** 2
        return max(0, 1.26 * 0.408 * delta / (delta + gamma) * (Rn - G))
    results.append(bench("priestley_taylor", priestley_taylor, N,
                         15.0, 0.5, 25.0, 200.0))

    # Richards 1D — 1K (simple implicit Euler)
    def richards_1d(n_nodes, dt, n_steps):
        h = [-100.0] * n_nodes
        dz = 30.0 / (n_nodes - 1)
        alpha, n_vg, theta_r, theta_s = 0.145, 2.68, 0.045, 0.43
        ks = 712.8
        m = 1 - 1 / n_vg
        h[0] = 0.0
        for _ in range(n_steps):
            for node in range(1, n_nodes - 1):
                Se = (1 + (alpha * abs(h[node])) ** n_vg) ** (-m) if h[node] < 0 else 1.0
                K = ks * Se ** 0.5 * (1 - (1 - Se ** (1/m)) ** m) ** 2
                C = alpha * m * (theta_s - theta_r) * n_vg * (alpha * abs(h[node])) ** (n_vg - 1) / (1 + (alpha * abs(h[node])) ** n_vg) ** (m + 1) if h[node] < 0 else 0.0
                flux = K * ((h[node-1] - 2*h[node] + h[node+1]) / dz**2 + 1.0)
                h[node] += dt * flux / max(C, 1e-10) if C > 0 else 0
        Se_top = (1 + (alpha * abs(h[0])) ** n_vg) ** (-m) if h[0] < 0 else 1.0
        return theta_r + (theta_s - theta_r) * Se_top
    results.append(bench("richards_1d", richards_1d, N // 10, 20, 0.001, 10))

    # Yield response — 100K
    def stewart_yield(ky, eta_etc):
        return 1 - ky * (1 - eta_etc)
    results.append(bench("yield_response", stewart_yield, N * 10, 1.25, 0.75))

    # Dual Kc 7-day — 10K
    def dual_kc_7day():
        de = 0.0
        tew, rew = 22.0, 9.0
        kcb, kc_max, few = 1.0, 1.2, 0.5
        for d in range(7):
            et0 = 4.5
            P = 15.0 if d == 0 else 0.0
            de = max(0, de - P)
            kr = max(0, (tew - de) / (tew - rew)) if de > rew else 1.0
            ke = min(few * (kc_max - kcb), kr * (kc_max - kcb))
            etc = (kcb + ke) * et0
            de = min(tew, de + etc * few - P * max(0, 1 - de/tew))
        return de
    results.append(bench("dual_kc_step", dual_kc_7day, N))

    # Makkink ET₀ — 100K
    def makkink_et0(T, Rs, alt):
        P = 101.3 * ((293 - 0.0065 * alt) / 293) ** 5.26
        gamma = 0.000665 * P
        delta = 4098 * 0.6108 * math.exp(17.27 * T / (T + 237.3)) / (T + 237.3) ** 2
        return max(0, 0.61 * delta / (delta + gamma) * Rs / 2.45 - 0.12)
    results.append(bench("makkink_et0", makkink_et0, N * 10, 25.0, 20.0, 200.0))

    # Blaney-Criddle ET₀ — 100K
    def blaney_criddle(T, lat_rad, doy):
        decl = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
        cos_ws = -math.tan(lat_rad) * math.tan(decl)
        cos_ws = max(-1, min(1, cos_ws))
        ws = math.acos(cos_ws)
        n = 24 * ws / math.pi
        p = n / (365 * 12)
        return max(0, p * (0.46 * T + 8.13))
    results.append(bench("blaney_criddle", blaney_criddle, N * 10,
                         25.0, math.radians(42.0), 180))

    out = {"benchmarks": results}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
