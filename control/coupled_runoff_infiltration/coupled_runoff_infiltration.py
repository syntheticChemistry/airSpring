#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Exp 052 — Coupled SCS-CN + Green-Ampt Rainfall Partitioning.

Couples the SCS Curve Number runoff method (USDA-SCS 1972) with the
Green-Ampt infiltration model (Green & Ampt 1911) to partition rainfall
into three components:

    P = Q + F + ΔS_surface

where Q is direct runoff (SCS-CN), F is cumulative infiltration (Green-Ampt),
and ΔS_surface is surface storage (P - Q - F, clamped ≥ 0).

This coupling is standard in SWMM, HEC-HMS, and TR-20 watershed models.

References:
    USDA-SCS (1972) NEH-4 Section 4: Hydrology.
    Green WH, Ampt GA (1911) J Agr Sci 4(1):1-24.
    Rawls WJ et al. (1983) J Hydraul Eng 109(1):62-70.
"""
import json
import math
import sys


# ── SCS-CN ──────────────────────────────────────────────────────────

def potential_retention(cn: float) -> float:
    if cn <= 0:
        return float('inf')
    return (25400.0 / cn) - 254.0


def scs_cn_runoff(precip_mm: float, cn: float, ia_ratio: float = 0.2) -> float:
    if precip_mm <= 0.0 or cn <= 0.0:
        return 0.0
    s = potential_retention(cn)
    ia = ia_ratio * s
    if precip_mm <= ia:
        return 0.0
    pe = precip_mm - ia
    return pe * pe / (pe + s)


# ── Green-Ampt ──────────────────────────────────────────────────────

def cumulative_infiltration(ks: float, psi: float, delta_theta: float, t_hr: float) -> float:
    if t_hr <= 0.0:
        return 0.0
    psi_dt = psi * delta_theta
    f = ks * t_hr + math.sqrt(2 * ks * psi_dt * t_hr)
    for _ in range(100):
        if f <= 0.0:
            f = ks * t_hr * 0.01
        g = f - ks * t_hr - psi_dt * math.log1p(f / psi_dt)
        dg = 1.0 - psi_dt / (psi_dt + f)
        if abs(dg) < 1e-15:
            break
        f_new = f - g / dg
        if f_new < 0:
            f_new = f * 0.5
        if abs(f_new - f) < 1e-10:
            f = f_new
            break
        f = f_new
    return max(f, 0.0)


def infiltration_rate(ks: float, psi: float, delta_theta: float, f_cum: float) -> float:
    if f_cum <= 0.0:
        return float('inf')
    return ks * (1.0 + psi * delta_theta / f_cum)


# ── Coupled model ───────────────────────────────────────────────────

SOILS = {
    "sandy_loam": {"ks": 1.09, "psi": 11.01, "delta_theta": 0.312},
    "loam":       {"ks": 0.34, "psi":  8.89, "delta_theta": 0.405},
    "silt_loam":  {"ks": 0.65, "psi": 16.68, "delta_theta": 0.400},
    "clay_loam":  {"ks": 0.10, "psi": 20.88, "delta_theta": 0.309},
}


def partition_rainfall(precip_mm, cn, soil_name, storm_duration_hr):
    """Partition rainfall into runoff, infiltration, and surface storage."""
    soil = SOILS[soil_name]
    q_mm = scs_cn_runoff(precip_mm, cn)
    p_net_mm = precip_mm - q_mm
    p_net_cm = p_net_mm / 10.0
    f_cm = cumulative_infiltration(soil["ks"], soil["psi"], soil["delta_theta"], storm_duration_hr)
    f_mm = f_cm * 10.0
    f_actual_mm = min(f_mm, p_net_mm)
    surface_mm = max(p_net_mm - f_actual_mm, 0.0)
    return {
        "precip_mm": precip_mm,
        "runoff_mm": round(q_mm, 6),
        "infiltration_mm": round(f_actual_mm, 6),
        "surface_storage_mm": round(surface_mm, 6),
        "mass_balance_error": round(abs(precip_mm - q_mm - f_actual_mm - surface_mm), 10),
        "runoff_fraction": round(q_mm / precip_mm, 6) if precip_mm > 0 else 0.0,
        "infiltration_fraction": round(f_actual_mm / precip_mm, 6) if precip_mm > 0 else 0.0,
    }


def run_storm_matrix():
    """Run a matrix of storm × soil × land-use scenarios."""
    storms = [
        {"name": "light", "precip_mm": 15.0, "duration_hr": 6.0},
        {"name": "moderate", "precip_mm": 40.0, "duration_hr": 4.0},
        {"name": "heavy", "precip_mm": 80.0, "duration_hr": 3.0},
        {"name": "extreme", "precip_mm": 150.0, "duration_hr": 2.0},
    ]
    land_uses = {
        "row_crops_B": 81,
        "pasture_B": 61,
        "woods_B": 55,
    }
    results = []
    for storm in storms:
        for soil_name in SOILS:
            for lu_name, cn in land_uses.items():
                r = partition_rainfall(storm["precip_mm"], cn, soil_name, storm["duration_hr"])
                results.append({
                    "storm": storm["name"],
                    "soil": soil_name,
                    "land_use": lu_name,
                    "cn": cn,
                    "storm_duration_hr": storm["duration_hr"],
                    **r,
                })
    return results


def run_conservation_checks():
    """Verify mass conservation for all scenarios."""
    checks = []
    for p in [5, 10, 25, 50, 75, 100, 150, 200]:
        for cn in [55, 65, 75, 85, 95]:
            for soil_name in ["sandy_loam", "clay_loam"]:
                r = partition_rainfall(float(p), float(cn), soil_name, 4.0)
                checks.append({
                    "precip_mm": float(p),
                    "cn": cn,
                    "soil": soil_name,
                    "mass_balance_error": r["mass_balance_error"],
                    "q_leq_p": r["runoff_mm"] <= float(p) + 0.001,
                    "f_leq_pnet": r["infiltration_mm"] <= float(p) - r["runoff_mm"] + 0.001,
                    "all_non_negative": all(v >= -1e-10 for v in [
                        r["runoff_mm"], r["infiltration_mm"], r["surface_storage_mm"]
                    ]),
                })
    return checks


def main():
    results = run_storm_matrix()
    conservation = run_conservation_checks()

    all_pass = True
    n_pass = 0
    n_total = 0

    print("=" * 72)
    print("Exp 052: Coupled SCS-CN + Green-Ampt Rainfall Partitioning")
    print("=" * 72)

    # Storm matrix
    for r in results:
        n_total += 1
        ok = r["mass_balance_error"] < 0.01
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{('PASS' if ok else 'FAIL')}] {r['storm']:>8s} {r['soil']:>12s} "
              f"{r['land_use']:>14s} CN={r['cn']:2d}: "
              f"Q={r['runoff_mm']:7.2f} F={r['infiltration_mm']:7.2f} "
              f"S={r['surface_storage_mm']:7.2f} err={r['mass_balance_error']:.2e}")

    # Conservation
    for c in conservation:
        n_total += 3
        for check_name, check_val in [
            ("mass_balance", c["mass_balance_error"] < 0.01),
            ("q_leq_p", c["q_leq_p"]),
            ("all_non_neg", c["all_non_negative"]),
        ]:
            ok = check_val
            if ok:
                n_pass += 1
            else:
                all_pass = False

    # Monotonicity: more precip → more runoff
    for cn in [65, 85]:
        precips = [10, 25, 50, 100]
        qs = [scs_cn_runoff(float(p), float(cn)) for p in precips]
        n_total += 1
        ok = all(qs[i] <= qs[i+1] for i in range(len(qs)-1))
        if ok:
            n_pass += 1
        else:
            all_pass = False

    # Soil ordering: sandy loam infiltrates more than clay loam
    for storm_mm in [40.0, 80.0]:
        r_sl = partition_rainfall(storm_mm, 75, "sandy_loam", 4.0)
        r_cl = partition_rainfall(storm_mm, 75, "clay_loam", 4.0)
        n_total += 1
        ok = r_sl["infiltration_mm"] > r_cl["infiltration_mm"]
        if ok:
            n_pass += 1
        else:
            all_pass = False

    print(f"\nResult: {n_pass}/{n_total} checks passed")

    # Build benchmark JSON
    import subprocess
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    ).stdout.strip() or "unknown"

    benchmark = {
        "_provenance": {
            "paper": "USDA-SCS (1972) NEH-4; Green WH, Ampt GA (1911) J Agr Sci 4(1):1-24.",
            "supplementary": [
                "Rawls WJ et al. (1983) J Hydraul Eng 109(1):62-70.",
                "Chow VT et al. (1988) Applied Hydrology, McGraw-Hill."
            ],
            "data_source": "Published equations and soil parameters (open literature)",
            "experiment": "052",
            "baseline_script": "control/coupled_runoff_infiltration/coupled_runoff_infiltration.py",
            "baseline_commit": commit,
            "baseline_command": "python control/coupled_runoff_infiltration/coupled_runoff_infiltration.py",
            "baseline_date": "2026-02-28",
            "baseline_result": f"{n_pass}/{n_total} PASS"
        },
        "storm_matrix": results[:12],
        "conservation_checks": conservation[:10],
        "_tolerance_justification": "SCS-CN and Green-Ampt are analytical: f64 arithmetic yields < 0.01 mm mass balance error"
    }

    with open("control/coupled_runoff_infiltration/benchmark_coupled_runoff.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Wrote benchmark JSON with {len(results)} storm matrix + {len(conservation)} conservation checks")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
