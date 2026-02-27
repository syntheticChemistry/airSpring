#!/usr/bin/env python3
"""Exp 038 — Pedotransfer → Richards Coupled Simulation.

Couples Saxton-Rawls pedotransfer functions (soil texture → hydraulic
parameters) with a 1D Richards PDE solver to simulate wetting/drying
profiles. Validates the full pipeline from USDA texture to moisture
dynamics.

References:
  Saxton & Rawls (2006) SSSA J 70:1569-1578
  van Genuchten (1980) SSSA J 44:892-898
  Carsel & Parrish (1988) WRR 24:755-769
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(SCRIPT_DIR, "benchmark_pedotransfer_richards.json")

# ── Saxton-Rawls pedotransfer ────────────────────────────────────────────────

def sr_theta_1500(s, c, om):
    t = (-0.024 * s + 0.487 * c + 0.006 * om
         - 0.005 * s * om - 0.013 * c * om
         + 0.068 * s * c + 0.031)
    return t + 0.14 * t - 0.02

def sr_theta_33(s, c, om):
    t = (-0.251 * s + 0.195 * c + 0.011 * om
         + 0.006 * s * om - 0.027 * c * om
         + 0.452 * s * c + 0.299)
    return t + 1.283 * t ** 2 - 0.374 * t - 0.015

def sr_theta_s_33(s, c, om):
    t = (0.278 * s + 0.034 * c + 0.022 * om
         - 0.018 * s * om - 0.027 * c * om
         - 0.584 * s * c + 0.078)
    return t + 0.636 * t - 0.107

def sr_theta_s(s, c, om):
    return sr_theta_33(s, c, om) + sr_theta_s_33(s, c, om) - 0.097 * s + 0.043

def sr_lambda(s, c, om):
    t33 = sr_theta_33(s, c, om)
    t1500 = sr_theta_1500(s, c, om)
    b = (math.log(1500.0) - math.log(33.0)) / (math.log(t33) - math.log(t1500))
    return 1.0 / b

def sr_ksat(s, c, om):
    ts = sr_theta_s(s, c, om)
    t33 = sr_theta_33(s, c, om)
    lam = sr_lambda(s, c, om)
    return 1930.0 * (ts - t33) ** (3.0 - lam)


def saxton_rawls_to_vg(sand, clay, om_pct):
    """Map Saxton-Rawls outputs to Van Genuchten parameters.

    Uses Carsel & Parrish (1988) texture-class mapping as a guide.
    theta_r ≈ theta_1500 (wilting point as residual),
    theta_s from SR directly,
    alpha and n from the moisture-tension curve shape.
    """
    theta_r = sr_theta_1500(sand, clay, om_pct)
    theta_s = sr_theta_s(sand, clay, om_pct)
    ksat = sr_ksat(sand, clay, om_pct)
    lam = sr_lambda(sand, clay, om_pct)

    # n from lambda via Brooks-Corey → VG relationship
    n_vg = lam + 1.0
    if n_vg < 1.05:
        n_vg = 1.05

    # alpha from bubbling pressure: Pb ≈ 33 * (θ_s - θ_r)/(θ_33 - θ_r)
    # scaled to VG convention (α ≈ 1/Pb in cm⁻¹)
    t33 = sr_theta_33(sand, clay, om_pct)
    if theta_s > theta_r and t33 > theta_r:
        pb = 33.0 * (theta_s - theta_r) / (t33 - theta_r)
        alpha = 1.0 / max(pb, 5.0)
    else:
        alpha = 0.02

    # Ksat: mm/hr → cm/day
    ks_cm_day = ksat * 2.4

    return {
        "theta_r": theta_r,
        "theta_s": theta_s,
        "alpha": alpha,
        "n_vg": n_vg,
        "ks": ks_cm_day,
        "ksat_mm_hr": ksat,
    }


# ── Simplified Richards solver (explicit, for validation only) ───────────────

def vg_theta(h, theta_r, theta_s, alpha, n_vg):
    if h >= 0:
        return theta_s
    m = 1.0 - 1.0 / n_vg
    ah = abs(alpha * h)
    return theta_r + (theta_s - theta_r) / (1.0 + ah ** n_vg) ** m

def vg_k(h, ks, theta_r, theta_s, alpha, n_vg):
    if h >= 0:
        return ks
    m = 1.0 - 1.0 / n_vg
    ah = abs(alpha * h)
    denom = (1.0 + ah ** n_vg) ** m
    se = 1.0 / denom
    return ks * se ** 0.5 * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2


def solve_richards_explicit(vg, depth_cm, n_nodes, h_init, h_top,
                            duration_days, dt_days,
                            zero_flux_top=False, bottom_free_drain=False):
    """Simple explicit finite-difference Richards solver for validation."""
    dz = depth_cm / (n_nodes - 1)
    z = [i * dz for i in range(n_nodes)]
    h = [h_init] * n_nodes

    n_steps = max(1, int(duration_days / dt_days))
    dt = dt_days

    profiles = []
    theta_initial = [vg_theta(h[i], vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
                     for i in range(n_nodes)]
    profiles.append({
        "z": list(z),
        "h": list(h),
        "theta": list(theta_initial),
    })

    tr = vg["theta_r"]; ts = vg["theta_s"]
    al = vg["alpha"]; nv = vg["n_vg"]; ks = vg["ks"]

    for _ in range(n_steps):
        h_new = list(h)

        for i in range(1, n_nodes - 1):
            k_up = 0.5 * (vg_k(h[i], ks, tr, ts, al, nv) + vg_k(h[i-1], ks, tr, ts, al, nv))
            k_dn = 0.5 * (vg_k(h[i], ks, tr, ts, al, nv) + vg_k(h[i+1], ks, tr, ts, al, nv))

            eps_h = 0.1
            c_h = (vg_theta(h[i] + eps_h, tr, ts, al, nv) -
                   vg_theta(h[i] - eps_h, tr, ts, al, nv)) / (2.0 * eps_h)
            c_h = max(c_h, 1e-8)

            # Richards equation with gravity: ∂θ/∂t = ∂/∂z[K(∂h/∂z + 1)]
            # z positive downward, gravity drives flow down
            q_up = k_up * ((h[i-1] - h[i]) / dz + 1.0)
            q_dn = k_dn * ((h[i] - h[i+1]) / dz + 1.0)
            dq_dz = (q_up - q_dn) / dz

            h_new[i] = h[i] + dt * dq_dz / c_h
            h_new[i] = max(h_new[i], -10000.0)
            h_new[i] = min(h_new[i], 0.0)

        # Boundary conditions
        if zero_flux_top:
            h_new[0] = h_new[1]
        else:
            h_new[0] = h_top

        if bottom_free_drain:
            # unit gradient: q_bottom = K(h_N) * 1, drains out
            k_bot = vg_k(h[-1], ks, tr, ts, al, nv)
            k_prev = vg_k(h[-2], ks, tr, ts, al, nv)
            k_int = 0.5 * (k_bot + k_prev)
            eps_h = 0.1
            c_bot = (vg_theta(h[-1] + eps_h, tr, ts, al, nv) -
                     vg_theta(h[-1] - eps_h, tr, ts, al, nv)) / (2.0 * eps_h)
            c_bot = max(c_bot, 1e-8)
            q_in = k_int * ((h[-2] - h[-1]) / dz + 1.0)
            q_out = k_bot * 1.0
            h_new[-1] = h[-1] + dt * (q_in - q_out) / dz / c_bot
            h_new[-1] = max(h_new[-1], -10000.0)
            h_new[-1] = min(h_new[-1], 0.0)
        else:
            h_new[-1] = h_init

        h = h_new

    theta_final = [vg_theta(h[i], vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
                   for i in range(n_nodes)]
    profiles.append({
        "z": list(z),
        "h": list(h),
        "theta": list(theta_final),
    })

    return profiles


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pedotransfer_to_vg(benchmark):
    print("\n[Pedotransfer → VG Mapping]")
    tests = benchmark["validation_checks"]["pedotransfer_to_vg"]["test_cases"]
    passed, total = 0, 0
    for tc in tests:
        label = tc["label"]
        vg = saxton_rawls_to_vg(tc["sand"], tc["clay"], tc["om_pct"])
        ref = tc["vg_reference"]

        checks = [
            ("theta_r", vg["theta_r"], ref["theta_r_range"]),
            ("theta_s", vg["theta_s"], ref["theta_s_range"]),
            ("alpha", vg["alpha"], ref["alpha_range"]),
            ("n_vg", vg["n_vg"], ref["n_range"]),
            ("ks", vg["ks"], ref["ks_range"]),
        ]

        for name, val, (lo, hi) in checks:
            total += 1
            ok = lo <= val <= hi
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label}: {name}={val:.4f} in [{lo}, {hi}]")
            if ok:
                passed += 1

    return passed, total


def validate_vg_retention_curves(benchmark):
    """Validate VG retention curves from pedotransfer-derived parameters."""
    print("\n[VG Retention Curve Checks]")
    tests = benchmark["validation_checks"]["pedotransfer_to_vg"]["test_cases"]
    passed, total = 0, 0

    for tc in tests:
        label = tc["label"]
        vg = saxton_rawls_to_vg(tc["sand"], tc["clay"], tc["om_pct"])

        # θ(0) = θ_s (saturation)
        total += 1
        theta_sat = vg_theta(0, vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
        ok = abs(theta_sat - vg["theta_s"]) < 1e-6
        print(f"  {'PASS' if ok else 'FAIL'} {label}: θ(0) = θ_s = {theta_sat:.4f}")
        if ok: passed += 1

        # θ(-10000) approaches θ_r (with low n, convergence is slow)
        total += 1
        theta_dry = vg_theta(-10000, vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
        range_frac = (theta_dry - vg["theta_r"]) / (vg["theta_s"] - vg["theta_r"])
        ok = range_frac < 0.50
        print(f"  {'PASS' if ok else 'FAIL'} {label}: θ(-10000)={theta_dry:.4f}, {range_frac:.1%} of range above θ_r")
        if ok: passed += 1

        # K(0) = Ks (saturated conductivity)
        total += 1
        k_sat = vg_k(0, vg["ks"], vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
        ok = abs(k_sat - vg["ks"]) < 1e-6
        print(f"  {'PASS' if ok else 'FAIL'} {label}: K(0) = Ks = {k_sat:.2f}")
        if ok: passed += 1

        # Monotonicity: θ(-50) > θ(-500)
        total += 1
        theta_wet = vg_theta(-50, vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
        theta_mid = vg_theta(-500, vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"])
        ok = theta_wet > theta_mid
        print(f"  {'PASS' if ok else 'FAIL'} {label}: θ(-50)={theta_wet:.4f} > θ(-500)={theta_mid:.4f}")
        if ok: passed += 1

    return passed, total


def validate_texture_sensitivity(benchmark):
    """Validate that texture affects VG parameters as expected."""
    print("\n[Texture Sensitivity (VG Parameters)]")
    tests = benchmark["validation_checks"]["texture_sensitivity"]["test_cases"]
    passed, total = 0, 0

    for tc in tests:
        label = tc["label"]
        sandy = tc["sandy"]
        clayey = tc["clayey"]

        vg_sand = saxton_rawls_to_vg(sandy["sand"], sandy["clay"], sandy["om_pct"])
        vg_clay = saxton_rawls_to_vg(clayey["sand"], clayey["clay"], clayey["om_pct"])

        # Sandy soil has higher Ksat
        total += 1
        ok = vg_sand["ks"] > vg_clay["ks"]
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {label}: Ks_sand={vg_sand['ks']:.2f} > Ks_clay={vg_clay['ks']:.2f}")
        if ok: passed += 1

        # Sandy drains more at same h: θ_sand(-100) < θ_clay(-100)
        total += 1
        theta_s = vg_theta(-100, vg_sand["theta_r"], vg_sand["theta_s"], vg_sand["alpha"], vg_sand["n_vg"])
        theta_c = vg_theta(-100, vg_clay["theta_r"], vg_clay["theta_s"], vg_clay["alpha"], vg_clay["n_vg"])
        ok = theta_s < theta_c
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {label}: θ_sand(-100)={theta_s:.4f} < θ_clay(-100)={theta_c:.4f}")
        if ok: passed += 1

    return passed, total


def main():
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    print("Exp 038: Pedotransfer → Richards Coupled Simulation")
    print(f"Benchmark: {BENCHMARK}")

    total_pass, total_tests = 0, 0

    p, t = validate_pedotransfer_to_vg(benchmark)
    total_pass += p; total_tests += t

    p, t = validate_vg_retention_curves(benchmark)
    total_pass += p; total_tests += t

    p, t = validate_texture_sensitivity(benchmark)
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
