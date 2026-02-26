#!/usr/bin/env python3
"""
airSpring Experiment 012 — CW2D Richards Equation Extension

Validates the 1D Richards equation solver with constructed wetland media
parameters from Dong et al. (2019) and HYDRUS CW2D standard databases.

Constructed wetland substrates (gravel, coarse sand, organic layers) have
extreme van Genuchten parameters compared to natural soils:
  - Gravel: very high Ks (5000 cm/d), steep retention curve (high n)
  - Organic: high porosity (θs=0.60), high retention, moderate Ks

This experiment extends Exp 006 (natural soils) to CW2D media, validating
that the same solver handles the full parameter range.

References:
    Dong et al. (2019) J Sustainable Water in the Built Environment 5(4):04019005
    Šimůnek et al. (2012) HYDRUS software
    van Genuchten (1980) SSSA J 44:892-898

Provenance:
  Benchmark output: control/cw2d/benchmark_cw2d.json
  Reproduction: python control/cw2d/cw2d_richards.py
  Created: 2026-02-25
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
from scipy.integrate import solve_ivp


# ── Van Genuchten-Mualem hydraulics (same as Exp 006) ─────────────────────

def van_genuchten_theta(h, theta_r, theta_s, alpha, n):
    if h >= 0:
        return theta_s
    h_safe = min(abs(h), 1e4)
    m = 1.0 - 1.0 / n
    x = min((alpha * h_safe) ** n, 1e10)
    se = 1.0 / (1.0 + x) ** m
    theta = theta_r + (theta_s - theta_r) * se
    return float(np.clip(theta, theta_r, theta_s))


def van_genuchten_K(h, Ks, theta_r, theta_s, alpha, n):
    if h >= 0:
        return Ks
    if h < -1e4:
        return 0.0
    m = 1.0 - 1.0 / n
    theta = van_genuchten_theta(h, theta_r, theta_s, alpha, n)
    se = (theta - theta_r) / (theta_s - theta_r)
    if se <= 0:
        return 0.0
    if se >= 1:
        return Ks
    term = 1.0 - se ** (1.0 / m)
    if term <= 0:
        return Ks
    kr = np.sqrt(se) * (1.0 - term ** m) ** 2
    return float(Ks * np.clip(kr, 0.0, 1.0))


def dtheta_dh(h, theta_r, theta_s, alpha, n):
    if h >= 0:
        return 1e-6
    h_safe = max(abs(h), 0.1)
    h_safe = min(h_safe, 1e4)
    m = 1.0 - 1.0 / n
    x = min((alpha * h_safe) ** n, 1e10)
    denom = (1.0 + x) ** (m + 1)
    if denom <= 0 or not np.isfinite(denom):
        return 1e-6
    dse_dh = m * n * (alpha ** n) * (h_safe ** (n - 1)) / denom
    result = (theta_s - theta_r) * dse_dh
    return float(np.clip(result, 1e-10, 1e2))


# ── Richards equation solver ──────────────────────────────────────────────

def _richards_rhs(t, h_vec, params):
    dz = params["dz"]
    nn = params["n_nodes"]
    theta_r = params["theta_r"]
    theta_s = params["theta_s"]
    alpha = params["alpha"]
    n_vg = params["n_vg"]
    Ks = params["Ks_cm_day"]
    h_top = params.get("h_top", 0.0)
    zero_flux_top = params.get("zero_flux_top", False)

    h = np.clip(np.asarray(h_vec).flatten(), -1e3, 50.0)

    K = np.array([van_genuchten_K(h[i], Ks, theta_r, theta_s, alpha, n_vg)
                  for i in range(nn)])
    C = np.array([dtheta_dh(h[i], theta_r, theta_s, alpha, n_vg)
                  for i in range(nn)])

    q = np.zeros(nn + 1)

    if zero_flux_top:
        q[0] = 0.0
    else:
        K_top = van_genuchten_K(h_top, Ks, theta_r, theta_s, alpha, n_vg)
        q[0] = K_top * ((h_top - h[0]) / (0.5 * dz) + 1.0)

    for i in range(nn - 1):
        K_mid = 0.5 * (K[i] + K[i + 1])
        q[i + 1] = K_mid * ((h[i + 1] - h[i]) / dz + 1.0)

    q[nn] = K[nn - 1]

    dtheta_dt = (q[:-1] - q[1:]) / dz
    C_safe = np.maximum(C, 1e-10)
    dh_dt = np.where(np.isfinite(dtheta_dt / C_safe), dtheta_dt / C_safe, 0.0)
    return dh_dt


def solve_richards_1d(params, h_initial, h_top, duration_hours,
                      n_nodes=50, zero_flux_top=False):
    duration_days = duration_hours / 24.0
    column_depth = params["column_depth_cm"]
    dz = column_depth / n_nodes

    sim_params = dict(params)
    sim_params["dz"] = dz
    sim_params["n_nodes"] = n_nodes
    sim_params["h_top"] = h_top
    sim_params["zero_flux_top"] = zero_flux_top

    h0 = np.full(n_nodes, h_initial)
    t_span = (0.0, duration_days)

    sol = solve_ivp(
        _richards_rhs, t_span, h0,
        args=(sim_params,),
        method="BDF",
        rtol=1e-4, atol=1e-6,
        max_step=duration_days / 10,
    )

    if not sol.success:
        sol = solve_ivp(
            _richards_rhs, t_span, h0,
            args=(sim_params,),
            method="Radau",
            rtol=1e-3, atol=1e-5,
            max_step=duration_days / 5,
        )

    z = np.linspace(dz / 2, column_depth - dz / 2, n_nodes)
    h_final = sol.y[:, -1] if sol.success else h0
    theta_final = np.array([
        van_genuchten_theta(h_final[i], params["theta_r"], params["theta_s"],
                            params["alpha"], params["n_vg"])
        for i in range(n_nodes)
    ])

    cum_drainage = 0.0
    if sol.success:
        for j in range(sol.y.shape[1]):
            h_bot = sol.y[-1, j]
            k_bot = van_genuchten_K(h_bot, params["Ks_cm_day"],
                                    params["theta_r"], params["theta_s"],
                                    params["alpha"], params["n_vg"])
            dt_step = (sol.t[j] - sol.t[j - 1]) if j > 0 else sol.t[0]
            cum_drainage += k_bot * dt_step

    return {
        "success": sol.success,
        "z": z,
        "h_final": h_final,
        "theta_final": theta_final,
        "cum_drainage": cum_drainage,
    }


# ── Validation harness ────────────────────────────────────────────────────

def check(label, computed, expected, tol):
    diff = abs(computed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  [{status}] {label}: {computed:.6f} "
          f"(expected {expected:.6f}, tol {tol})")
    return diff <= tol


def check_range(label, value, low, high):
    ok = low <= value <= high
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {value:.6f} (range [{low}, {high}])")
    return ok


def check_bool(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def get_media_params(benchmark, media_name):
    m = benchmark["cw2d_media"][media_name]
    return {
        "theta_r": m["theta_r"],
        "theta_s": m["theta_s"],
        "alpha": m["alpha"],
        "n_vg": m["n_vg"],
        "Ks_cm_day": m["Ks_cm_day"],
    }


# ── Validators ────────────────────────────────────────────────────────────

def validate_retention_curves(benchmark):
    print("\n--- CW2D Retention Curves ---")
    passed = failed = 0

    for tc in benchmark["validation_checks"]["cw2d_retention_curves"]["test_cases"]:
        media = tc["media"]
        h_cm = tc["h_cm"]
        expected = tc["expected_theta"]
        tol = tc["tolerance"]

        p = get_media_params(benchmark, media)
        computed = van_genuchten_theta(h_cm, p["theta_r"], p["theta_s"],
                                      p["alpha"], p["n_vg"])
        if check(f"θ({media}, h={h_cm})", computed, expected, tol):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_conductivity(benchmark):
    print("\n--- CW2D Hydraulic Conductivity ---")
    passed = failed = 0

    for tc in benchmark["validation_checks"]["cw2d_conductivity"]["test_cases"]:
        media = tc["media"]
        h_cm = tc["h_cm"]
        p = get_media_params(benchmark, media)
        ks = p["Ks_cm_day"]

        k = van_genuchten_K(h_cm, ks, p["theta_r"], p["theta_s"],
                            p["alpha"], p["n_vg"])
        k_ratio = k / ks

        if "expected_K_ratio" in tc:
            if check(f"K_ratio({media}, h={h_cm})", k_ratio,
                     tc["expected_K_ratio"], tc["tolerance"]):
                passed += 1
            else:
                failed += 1
        elif "K_ratio_range" in tc:
            lo, hi = tc["K_ratio_range"]
            if check_range(f"K_ratio({media}, h={h_cm})", k_ratio, lo, hi):
                passed += 1
            else:
                failed += 1

    return passed, failed


def validate_gravel_infiltration(benchmark):
    print("\n--- CW2D Gravel Infiltration ---")
    passed = failed = 0

    spec = benchmark["validation_checks"]["cw2d_gravel_infiltration"]
    p = get_media_params(benchmark, "gravel")
    p["column_depth_cm"] = spec["column_depth_cm"]

    result = solve_richards_1d(
        p,
        h_initial=spec["initial_h_cm"],
        h_top=spec["top_h_cm"],
        duration_hours=spec["duration_hours"],
        n_nodes=spec["n_nodes"],
    )

    for chk in spec["checks"]:
        cid = chk["id"]
        if cid == "solver_converges":
            if check_bool(f"gravel_{cid}", result["success"]):
                passed += 1
            else:
                failed += 1
        elif cid == "surface_wets":
            min_theta = chk["min_theta"]
            theta_surf = result["theta_final"][0]
            if check_bool(f"gravel_{cid} (θ_surf={theta_surf:.4f} >= {min_theta})",
                         theta_surf >= min_theta):
                passed += 1
            else:
                failed += 1
        elif cid == "drainage_starts":
            if check_bool(f"gravel_{cid} (drainage={result['cum_drainage']:.4f})",
                         result["cum_drainage"] > 0):
                passed += 1
            else:
                failed += 1

    return passed, failed


def validate_organic_drainage(benchmark):
    print("\n--- CW2D Organic Layer Drainage ---")
    passed = failed = 0

    spec = benchmark["validation_checks"]["cw2d_organic_drainage"]
    p = get_media_params(benchmark, "organic_substrate")
    p["column_depth_cm"] = spec["column_depth_cm"]

    result = solve_richards_1d(
        p,
        h_initial=spec["initial_h_cm"],
        h_top=spec["initial_h_cm"],
        duration_hours=spec["duration_hours"],
        n_nodes=spec["n_nodes"],
        zero_flux_top=True,
    )

    for chk in spec["checks"]:
        cid = chk["id"]
        if cid == "solver_converges":
            if check_bool(f"organic_{cid}", result["success"]):
                passed += 1
            else:
                failed += 1
        elif cid == "drainage_positive":
            if check_bool(f"organic_{cid} (drainage={result['cum_drainage']:.4f})",
                         result["cum_drainage"] > 0):
                passed += 1
            else:
                failed += 1
        elif cid == "high_retention":
            avg_theta = float(np.mean(result["theta_final"]))
            threshold = p["theta_r"] + 0.1
            if check_bool(f"organic_{cid} (avg_θ={avg_theta:.4f} >= {threshold:.4f})",
                         avg_theta >= threshold):
                passed += 1
            else:
                failed += 1

    return passed, failed


def validate_mass_balance(benchmark):
    print("\n--- CW2D Mass Balance ---")
    passed = failed = 0

    for tc in benchmark["validation_checks"]["cw2d_mass_balance"]["test_cases"]:
        media = tc["media"]
        p = get_media_params(benchmark, media)
        p["column_depth_cm"] = tc["column_depth_cm"]
        max_err = tc["max_balance_error_pct"]

        result = solve_richards_1d(
            p,
            h_initial=tc["initial_h_cm"],
            h_top=tc["top_h_cm"],
            duration_hours=tc["duration_hours"],
            n_nodes=tc["n_nodes"],
            zero_flux_top=tc["zero_flux_top"],
        )

        if not result["success"]:
            print(f"  [FAIL] {media}_mass_balance: solver failed")
            failed += 1
            continue

        theta_init = van_genuchten_theta(
            tc["initial_h_cm"], p["theta_r"], p["theta_s"],
            p["alpha"], p["n_vg"])
        theta_final_avg = float(np.mean(result["theta_final"]))
        dz = tc["column_depth_cm"] / tc["n_nodes"]
        storage_change = (theta_final_avg - theta_init) * tc["column_depth_cm"]
        drainage = result["cum_drainage"]
        total_water = abs(storage_change) + abs(drainage)

        if total_water < 1e-10:
            err_pct = 0.0
        else:
            imbalance = abs(storage_change + drainage)
            err_pct = 100.0 * imbalance / total_water

        if check_bool(
            f"{media}_mass_balance (err={err_pct:.1f}% <= {max_err}%)",
            err_pct <= max_err
        ):
            passed += 1
        else:
            failed += 1

    return passed, failed


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    benchmark_path = Path(__file__).parent / "benchmark_cw2d.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_failed = 0

    print("=" * 70)
    print("airSpring Exp 012: CW2D Richards Extension Validation")
    print("  Dong et al. (2019) + HYDRUS CW2D Media Parameters")
    print("=" * 70)

    for validator in [
        validate_retention_curves,
        validate_conductivity,
        validate_gravel_infiltration,
        validate_organic_drainage,
        validate_mass_balance,
    ]:
        p, f_ = validator(benchmark)
        total_passed += p
        total_failed += f_

    total = total_passed + total_failed
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
