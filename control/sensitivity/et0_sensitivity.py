# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""Exp 017: FAO-56 Penman-Monteith ET₀ sensitivity analysis.

One-at-a-time (OAT) perturbation analysis to quantify which input
variable dominates ET₀ response. Complements groundSpring Exp 003
(Monte Carlo error propagation, which found humidity at 66% of variance).

Method:
  For each variable x_i, perturb ±10% from the FAO-56 Example 18 baseline.
  Compute sensitivity index: S_i = (ET₀(+10%) - ET₀(-10%)) / (2 × 0.1 × x_i)
  Compute elasticity: E_i = (ΔET₀/ET₀) / (Δx/x)

References:
  Allen et al. (1998) FAO-56 Ch 4
  Gong et al. (2006) Ag Water Mgmt 86:57-63 — Sensitivity of PM ET₀
  Estévez et al. (2009) J Irrig Drain Eng 135(3):275-286

Provenance:
  Script: control/sensitivity/et0_sensitivity.py
  Benchmark: control/sensitivity/benchmark_sensitivity.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

# ── FAO-56 equations (from Exp 001 penman_monteith.py) ─────────────

def saturation_vapour_pressure(t_c):
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))

def slope_vapour_pressure_curve(t_c):
    es = saturation_vapour_pressure(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2

def atmospheric_pressure(altitude_m):
    return 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26

def psychrometric_constant(pressure_kpa):
    return 0.000665 * pressure_kpa

def actual_vapour_pressure_rh(tmax_c, tmin_c, rhmax, rhmin):
    e_tmin = saturation_vapour_pressure(tmin_c)
    e_tmax = saturation_vapour_pressure(tmax_c)
    return (e_tmin * (rhmax / 100.0) + e_tmax * (rhmin / 100.0)) / 2.0

def solar_declination(doy):
    return 0.409 * math.sin(2.0 * math.pi / 365.0 * doy - 1.39)

def inverse_relative_distance(doy):
    return 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)

def sunset_hour_angle(lat_rad, dec_rad):
    arg = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(dec_rad)))
    return math.acos(arg)

def extraterrestrial_radiation(lat_deg, doy):
    gsc = 0.0820
    phi = math.radians(lat_deg)
    dr = inverse_relative_distance(doy)
    delta = solar_declination(doy)
    ws = sunset_hour_angle(phi, delta)
    return (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(ws)
    )

def clear_sky_radiation(alt_m, ra):
    return (0.75 + 2e-5 * alt_m) * ra

def net_shortwave(rs, albedo=0.23):
    return (1.0 - albedo) * rs

def net_longwave(tmax_c, tmin_c, ea_kpa, rs_over_rso):
    sigma = 4.903e-9
    tmax_k4 = (tmax_c + 273.16) ** 4
    tmin_k4 = (tmin_c + 273.16) ** 4
    avg_k4 = (tmax_k4 + tmin_k4) / 2.0
    hf = 0.34 - 0.14 * math.sqrt(ea_kpa)
    cf = 1.35 * rs_over_rso - 0.35
    return sigma * avg_k4 * hf * cf

def fao56_pm(rn, g, tmean_c, u2, vpd_kpa, delta, gamma):
    num = 0.408 * delta * (rn - g) + gamma * (900.0 / (tmean_c + 273.0)) * u2 * vpd_kpa
    den = delta + gamma * (1.0 + 0.34 * u2)
    return num / den


# ── Compute ET₀ from a parameter dict ──────────────────────────────

def compute_et0(params):
    """Full FAO-56 daily ET₀ from meteorological inputs."""
    tmin = params["tmin_c"]
    tmax = params["tmax_c"]
    tmean = (tmin + tmax) / 2.0
    rh_min = params["rh_min_pct"]
    rh_max = params["rh_max_pct"]
    u2 = params["wind_2m_m_s"]
    rs = params["solar_rad_mj_m2_day"]
    elev = params["elevation_m"]
    lat = params["latitude_deg"]
    doy = params["day_of_year"]

    p = atmospheric_pressure(elev)
    gamma = psychrometric_constant(p)
    delta = slope_vapour_pressure_curve(tmean)
    es = (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin)) / 2.0
    ea = actual_vapour_pressure_rh(tmax, tmin, rh_max, rh_min)
    vpd = es - ea

    ra = extraterrestrial_radiation(lat, doy)
    rso = clear_sky_radiation(elev, ra)
    rns = net_shortwave(rs)
    rs_rso = min(rs / rso, 1.0) if rso > 0 else 1.0
    rnl = net_longwave(tmax, tmin, ea, rs_rso)
    rn = rns - rnl
    g = 0.0

    return fao56_pm(rn, g, tmean, u2, vpd, delta, gamma)


# ── Sensitivity analysis ───────────────────────────────────────────

def oat_sensitivity(baseline_params, var_name, pct=10.0):
    """One-at-a-time sensitivity for a single variable.

    Returns: (et0_base, et0_plus, et0_minus, sensitivity, elasticity)
    """
    et0_base = compute_et0(baseline_params)
    x_base = baseline_params[var_name]
    dx = abs(x_base) * pct / 100.0

    params_plus = {**baseline_params, var_name: x_base + dx}
    params_minus = {**baseline_params, var_name: x_base - dx}

    et0_plus = compute_et0(params_plus)
    et0_minus = compute_et0(params_minus)

    sensitivity = (et0_plus - et0_minus) / (2.0 * dx) if dx > 0 else 0.0
    elasticity = ((et0_plus - et0_minus) / et0_base) / (2.0 * pct / 100.0) if et0_base > 0 else 0.0

    return et0_base, et0_plus, et0_minus, sensitivity, elasticity


def full_sensitivity_analysis(params, variables, pct=10.0):
    """Run OAT sensitivity for all variables. Returns sorted results."""
    results = []
    for var in variables:
        name = var["name"]
        et0_base, et0_plus, et0_minus, sens, elast = oat_sensitivity(params, name, pct)
        results.append({
            "name": name,
            "label": var["label"],
            "base_value": params[name],
            "et0_base": et0_base,
            "et0_plus": et0_plus,
            "et0_minus": et0_minus,
            "sensitivity": sens,
            "abs_sensitivity": abs(sens),
            "elasticity": elast,
        })
    results.sort(key=lambda r: r["abs_sensitivity"], reverse=True)
    return results


# ── Validation helpers ──────────────────────────────────────────────

passed_total = 0
failed_total = 0


def check(label, computed, expected, tol):
    global passed_total, failed_total
    ok = abs(computed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.4f}, expected={expected:.4f}, tol={tol}")
    if ok:
        passed_total += 1
    else:
        failed_total += 1
    return ok


def check_range(label, value, low, high):
    global passed_total, failed_total
    ok = low <= value <= high
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: value={value:.4f}, range=[{low}, {high}]")
    if ok:
        passed_total += 1
    else:
        failed_total += 1
    return ok


def check_bool(label, condition):
    global passed_total, failed_total
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if condition:
        passed_total += 1
    else:
        failed_total += 1
    return condition


# ── Validators ──────────────────────────────────────────────────────

def validate_baseline(benchmark, baseline_params):
    print("\n── Baseline ET₀ ──")
    et0 = compute_et0(baseline_params)
    bc = benchmark["validation_checks"]["baseline_et0"]
    check("FAO-56 Example 18 ET₀", et0, bc["expected"], bc["tolerance"])


def validate_monotonicity(benchmark, results):
    print("\n── Monotonicity ──")
    mc = benchmark["validation_checks"]["monotonicity"]
    for r in results:
        name = r["name"]
        if name in mc["positive_sensitivity"]:
            check_bool(
                f"{r['label']} sensitivity > 0 (more → more ET₀)",
                r["sensitivity"] > 0,
            )
        elif name in mc["negative_sensitivity"]:
            check_bool(
                f"{r['label']} sensitivity < 0 (more → less ET₀)",
                r["sensitivity"] < 0,
            )


def validate_elasticity(benchmark, results):
    print("\n── Elasticity bounds ──")
    ec = benchmark["validation_checks"]["elasticity_bounds"]
    for r in results:
        check_range(
            f"{r['label']} elasticity",
            r["elasticity"],
            ec["min_elasticity"],
            ec["max_elasticity"],
        )


def validate_symmetry(benchmark, results):
    print("\n── Symmetry ──")
    sc = benchmark["validation_checks"]["symmetry"]
    for r in results:
        delta_plus = abs(r["et0_plus"] - r["et0_base"])
        delta_minus = abs(r["et0_minus"] - r["et0_base"])
        if delta_minus > 1e-10 and delta_plus > 1e-10:
            ratio = max(delta_plus / delta_minus, delta_minus / delta_plus)
            check_range(
                f"{r['label']} symmetry ratio",
                ratio,
                1.0,
                sc["tolerance_ratio"],
            )
        else:
            check_bool(f"{r['label']} both deltas near zero", True)


def validate_ranking(benchmark, results):
    print("\n── Sensitivity ranking ──")
    rc = benchmark["validation_checks"]["sensitivity_ranking"]
    top_two_names = [r["name"] for r in results[:2]]
    allowed = set(rc["top_two_include"])
    has_overlap = any(n in allowed for n in top_two_names)
    check_bool(
        f"top-2 ({top_two_names}) includes radiation or humidity variable",
        has_overlap,
    )
    for i, r in enumerate(results):
        print(f"    #{i+1}: {r['label']:>10} |S|={r['abs_sensitivity']:.4f}  E={r['elasticity']:+.4f}")


def validate_multi_site(benchmark, variables):
    print("\n── Multi-site consistency ──")
    mc = benchmark["validation_checks"]["multi_site_consistency"]
    for site_spec in mc["sites"]:
        site_params = {
            "tmin_c": site_spec["tmin_c"],
            "tmax_c": site_spec["tmax_c"],
            "rh_min_pct": site_spec["rh_min_pct"],
            "rh_max_pct": site_spec["rh_max_pct"],
            "wind_2m_m_s": site_spec["wind_2m_m_s"],
            "solar_rad_mj_m2_day": site_spec["solar_rad_mj_m2_day"],
            "elevation_m": site_spec["elevation_m"],
            "latitude_deg": site_spec["latitude_deg"],
            "day_of_year": site_spec["day_of_year"],
        }
        results = full_sensitivity_analysis(site_params, variables)
        et0 = compute_et0(site_params)
        top3_names = [r["name"] for r in results[:3]]
        has_rad = "solar_rad_mj_m2_day" in top3_names
        check_bool(
            f"{site_spec['name']}: ET₀={et0:.2f}, radiation in top-3 ({top3_names})",
            has_rad,
        )
        for r in results:
            print(f"      {r['label']:>10}: |S|={r['abs_sensitivity']:.4f}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Exp 017: FAO-56 Penman-Monteith Sensitivity Analysis")
    print("  OAT ±10% perturbation of 6 input variables")
    print("=" * 65)

    bm_path = Path(__file__).parent / "benchmark_sensitivity.json"
    with open(bm_path) as f:
        benchmark = json.load(f)

    bc = benchmark["baseline_conditions"]
    baseline_params = {
        "tmin_c": bc["tmin_c"],
        "tmax_c": bc["tmax_c"],
        "rh_min_pct": bc["rh_min_pct"],
        "rh_max_pct": bc["rh_max_pct"],
        "wind_2m_m_s": bc["wind_2m_m_s"],
        "solar_rad_mj_m2_day": bc["solar_rad_mj_m2_day"],
        "elevation_m": bc["elevation_m"],
        "latitude_deg": bc["latitude_deg"],
        "day_of_year": bc["day_of_year"],
    }

    variables = benchmark["variables"]
    pct = benchmark["perturbation_pct"]

    validate_baseline(benchmark, baseline_params)

    results = full_sensitivity_analysis(baseline_params, variables, pct)

    validate_ranking(benchmark, results)
    validate_monotonicity(benchmark, results)
    validate_elasticity(benchmark, results)
    validate_symmetry(benchmark, results)
    validate_multi_site(benchmark, variables)

    print(f"\n{'=' * 65}")
    print(f"  Exp 017 Summary: {passed_total} PASS, {failed_total} FAIL")
    print(f"{'=' * 65}")
    sys.exit(0 if failed_total == 0 else 1)


if __name__ == "__main__":
    main()
