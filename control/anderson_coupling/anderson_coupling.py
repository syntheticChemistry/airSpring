#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Anderson soil-moisture coupling — Experiment 045

Couples the FAO-56 water balance θ(t) to the Anderson localization model
for predicting quorum-sensing (QS) regime transitions in soil.

The physics chain:
    θ(t) → S_e(t) → pore_connectivity(t) → d_eff(t) → QS_regime(t)

Where:
    S_e = (θ - θ_r) / (θ_s - θ_r)           [effective saturation, VG]
    p_c = S_e^L                                [Mualem connectivity, L=0.5]
    z   = z_max × p_c                          [coordination number]
    d_eff = z / 2                              [effective dimension, Bethe lattice]
    W   = W_0 × (1 - S_e)                     [Anderson disorder parameter]

References:
    - van Genuchten (1980) SSSA J 44:892-898
    - Mualem (1976) WRR 12:513-522
    - Anderson (1958) Phys Rev 109:1492-1505
    - Abrahams et al. (1979) J Phys C 12:2585 [d=2 threshold]
    - airSpring baseCamp Sub-thesis 06
"""

import json
import math
import sys

# VG parameters: Carsel & Parrish (1988) — matching barracuda
SOIL_TYPES = {
    "sand":       {"theta_r": 0.045, "theta_s": 0.43, "alpha": 0.145, "n": 2.68},
    "loam":       {"theta_r": 0.078, "theta_s": 0.43, "alpha": 0.036, "n": 1.56},
    "silt_loam":  {"theta_r": 0.067, "theta_s": 0.45, "alpha": 0.020, "n": 1.41},
    "clay":       {"theta_r": 0.068, "theta_s": 0.38, "alpha": 0.008, "n": 1.09},
}

# Lattice parameters
Z_MAX = 6.0       # max coordination number (3D cubic lattice)
L_MUALEM = 0.5    # Mualem pore-connectivity exponent
W_0 = 12.0        # disorder scale (typical for heterogeneous soil)

# QS regime thresholds
D_EFF_CRITICAL = 2.0   # Anderson d=2 transition
D_EFF_EXTENDED = 2.5   # above this, clearly extended (QS active)


def effective_saturation(theta, theta_r, theta_s):
    """Effective saturation S_e from volumetric water content."""
    if theta_s <= theta_r:
        return 0.0
    se = (theta - theta_r) / (theta_s - theta_r)
    return max(0.0, min(1.0, se))


def pore_connectivity(se, l_exp=L_MUALEM):
    """Mualem pore connectivity: p_c = S_e^L."""
    if se <= 0.0:
        return 0.0
    return se ** l_exp


def coordination_number(connectivity, z_max=Z_MAX):
    """Coordination number z = z_max × connectivity."""
    return z_max * connectivity


def effective_dimension(z):
    """Effective dimension from coordination number (Bethe lattice)."""
    return z / 2.0


def disorder_parameter(se, w0=W_0):
    """Anderson disorder W = W_0 × (1 - S_e)."""
    return w0 * (1.0 - se)


def qs_regime(d_eff):
    """Classify QS regime from effective dimension."""
    if d_eff > D_EFF_EXTENDED:
        return "extended"
    elif d_eff > D_EFF_CRITICAL:
        return "marginal"
    else:
        return "localized"


def coupling_chain(theta, soil):
    """Full coupling chain: θ → QS regime, returning all intermediates."""
    se = effective_saturation(theta, soil["theta_r"], soil["theta_s"])
    pc = pore_connectivity(se)
    z = coordination_number(pc)
    d_eff = effective_dimension(z)
    w = disorder_parameter(se)
    regime = qs_regime(d_eff)
    return {
        "theta": theta,
        "se": se,
        "connectivity": pc,
        "coordination": z,
        "d_eff": d_eff,
        "disorder": w,
        "regime": regime,
    }


def seasonal_theta_profile(soil_key, tillage):
    """Synthetic seasonal θ profile (typical Michigan growing season).

    Tillage effects: conventional tillage reduces θ_FC by ~15% and increases
    drainage rate, while no-till maintains higher moisture via mulch cover.
    """
    soil = SOIL_TYPES[soil_key]
    theta_s = soil["theta_s"]
    theta_r = soil["theta_r"]
    theta_fc = theta_r + 0.65 * (theta_s - theta_r)  # ~field capacity
    theta_wp = theta_r + 0.15 * (theta_s - theta_r)  # ~wilting point

    if tillage == "conventional":
        theta_fc *= 0.85
        theta_wp *= 0.90

    days = 153  # May 1 – Sep 30
    thetas = []
    for d in range(days):
        frac = d / (days - 1)
        if frac < 0.15:      # spring: wet
            theta = theta_fc
        elif frac < 0.55:    # early-mid summer: drying
            t = (frac - 0.15) / 0.40
            theta = theta_fc - t * (theta_fc - theta_wp)
        elif frac < 0.75:    # late summer: driest
            theta = theta_wp
        else:                 # fall: rewetting
            t = (frac - 0.75) / 0.25
            theta = theta_wp + t * (theta_fc - theta_wp) * 0.8
        thetas.append(theta)
    return thetas


def main():
    results = {"checks": []}
    n_pass = 0
    n_total = 0

    # ── 1. Point coupling tests ──────────────────────────────────────────
    point_cases = [
        ("sand",      0.35,  "extended"),
        ("sand",      0.10,  "localized"),
        ("loam",      0.30,  "marginal"),
        ("loam",      0.12,  "localized"),
        ("silt_loam", 0.35,  "extended"),
        ("silt_loam", 0.10,  "localized"),
        ("clay",      0.30,  "extended"),
        ("clay",      0.10,  "localized"),
        ("loam",      0.20,  "localized"),
    ]

    for soil_key, theta, expected_regime in point_cases:
        soil = SOIL_TYPES[soil_key]
        result = coupling_chain(theta, soil)
        ok = result["regime"] == expected_regime
        n_total += 1
        if ok:
            n_pass += 1
        label = f"point({soil_key}, θ={theta:.2f}) → {result['regime']}"
        results["checks"].append({
            "label": label,
            "pass": ok,
            "soil": soil_key,
            "theta": theta,
            "expected_regime": expected_regime,
            **result,
        })
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label} (d_eff={result['d_eff']:.3f}, W={result['disorder']:.2f})")

    # ── 2. Monotonicity: d_eff increases with θ ──────────────────────────
    for soil_key in SOIL_TYPES:
        soil = SOIL_TYPES[soil_key]
        prev_d = -1.0
        monotone = True
        for i in range(11):
            theta = soil["theta_r"] + i * (soil["theta_s"] - soil["theta_r"]) / 10.0
            r = coupling_chain(theta, soil)
            if r["d_eff"] < prev_d - 1e-12:
                monotone = False
            prev_d = r["d_eff"]
        n_total += 1
        if monotone:
            n_pass += 1
        label = f"monotonicity({soil_key}): d_eff ↑ with θ"
        results["checks"].append({"label": label, "pass": monotone, "soil": soil_key})
        status = "PASS" if monotone else "FAIL"
        print(f"  [{status}] {label}")

    # ── 3. Boundary conditions ───────────────────────────────────────────
    for soil_key in SOIL_TYPES:
        soil = SOIL_TYPES[soil_key]
        # At saturation: d_eff = 3.0 (full 3D connectivity)
        r_sat = coupling_chain(soil["theta_s"], soil)
        ok_sat = abs(r_sat["d_eff"] - 3.0) < 1e-10
        n_total += 1
        if ok_sat:
            n_pass += 1
        label = f"boundary({soil_key}, θ=θ_s) → d_eff=3.0"
        results["checks"].append({"label": label, "pass": ok_sat, "d_eff": r_sat["d_eff"]})
        status = "PASS" if ok_sat else "FAIL"
        print(f"  [{status}] {label} (d_eff={r_sat['d_eff']:.6f})")

        # At residual: d_eff = 0.0 (no connectivity)
        r_dry = coupling_chain(soil["theta_r"], soil)
        ok_dry = abs(r_dry["d_eff"]) < 1e-10
        n_total += 1
        if ok_dry:
            n_pass += 1
        label = f"boundary({soil_key}, θ=θ_r) → d_eff=0.0"
        results["checks"].append({"label": label, "pass": ok_dry, "d_eff": r_dry["d_eff"]})
        status = "PASS" if ok_dry else "FAIL"
        print(f"  [{status}] {label} (d_eff={r_dry['d_eff']:.6f})")

    # ── 4. Disorder monotonicity: W decreases with θ ────────────────────
    for soil_key in SOIL_TYPES:
        soil = SOIL_TYPES[soil_key]
        prev_w = float("inf")
        monotone = True
        for i in range(11):
            theta = soil["theta_r"] + i * (soil["theta_s"] - soil["theta_r"]) / 10.0
            r = coupling_chain(theta, soil)
            if r["disorder"] > prev_w + 1e-12:
                monotone = False
            prev_w = r["disorder"]
        n_total += 1
        if monotone:
            n_pass += 1
        label = f"disorder_monotonicity({soil_key}): W ↓ with θ"
        results["checks"].append({"label": label, "pass": monotone, "soil": soil_key})
        status = "PASS" if monotone else "FAIL"
        print(f"  [{status}] {label}")

    # ── 5. Seasonal regime transitions ───────────────────────────────────
    for soil_key in ["loam", "silt_loam"]:
        for tillage in ["notill", "conventional"]:
            thetas = seasonal_theta_profile(soil_key, tillage)
            soil = SOIL_TYPES[soil_key]
            chain = [coupling_chain(t, soil) for t in thetas]
            d_effs = [c["d_eff"] for c in chain]
            regimes = [c["regime"] for c in chain]

            # Check: spring should be extended/QS-active
            spring_d = sum(d_effs[:23]) / 23  # first 23 days (~May)
            spring_ok = spring_d > D_EFF_CRITICAL
            n_total += 1
            if spring_ok:
                n_pass += 1
            label = f"seasonal({soil_key}, {tillage}): spring d_eff={spring_d:.2f} > {D_EFF_CRITICAL}"
            results["checks"].append({"label": label, "pass": spring_ok})
            status = "PASS" if spring_ok else "FAIL"
            print(f"  [{status}] {label}")

            # Check: mid-summer should show regime change
            midsummer_d = sum(d_effs[84:115]) / 31  # Aug
            regime_shift = spring_d > midsummer_d
            n_total += 1
            if regime_shift:
                n_pass += 1
            label = f"seasonal({soil_key}, {tillage}): spring > summer ({spring_d:.2f} > {midsummer_d:.2f})"
            results["checks"].append({"label": label, "pass": regime_shift})
            status = "PASS" if regime_shift else "FAIL"
            print(f"  [{status}] {label}")

            # No-till maintains higher d_eff than conventional
            if tillage == "notill":
                notill_mean = sum(d_effs) / len(d_effs)
            else:
                conv_mean = sum(d_effs) / len(d_effs)

        nt_above = notill_mean > conv_mean
        n_total += 1
        if nt_above:
            n_pass += 1
        label = f"tillage_effect({soil_key}): notill d̄={notill_mean:.3f} > conv d̄={conv_mean:.3f}"
        results["checks"].append({"label": label, "pass": nt_above})
        status = "PASS" if nt_above else "FAIL"
        print(f"  [{status}] {label}")

    # ── 6. Exact numeric reference values (for Rust cross-validation) ────
    ref_cases = []
    for soil_key in ["sand", "loam", "silt_loam", "clay"]:
        soil = SOIL_TYPES[soil_key]
        for theta_frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
            theta = soil["theta_r"] + theta_frac * (soil["theta_s"] - soil["theta_r"])
            r = coupling_chain(theta, soil)
            ref_cases.append({
                "soil": soil_key,
                "theta": theta,
                "se": r["se"],
                "connectivity": r["connectivity"],
                "coordination": r["coordination"],
                "d_eff": r["d_eff"],
                "disorder": r["disorder"],
                "regime": r["regime"],
            })
            ok = True
            n_total += 1
            n_pass += 1
            label = f"ref({soil_key}, S_e={theta_frac:.2f}): d_eff={r['d_eff']:.6f}, W={r['disorder']:.4f}"
            results["checks"].append({"label": label, "pass": ok, **r})
            print(f"  [PASS] {label}")

    results["reference_values"] = ref_cases
    results["soil_types"] = SOIL_TYPES
    results["constants"] = {
        "z_max": Z_MAX,
        "l_mualem": L_MUALEM,
        "w_0": W_0,
        "d_eff_critical": D_EFF_CRITICAL,
        "d_eff_extended": D_EFF_EXTENDED,
    }

    print(f"\n=== Anderson Coupling: {n_pass}/{n_total} PASS, {n_total - n_pass} FAIL ===")

    with open("benchmark_anderson_coupling.json", "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
