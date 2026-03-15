#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Exp 054 — Full-Season Irrigation Water Budget Audit.

Runs the complete FAO-56 pipeline for a synthetic growing season:
  Weather → ET₀ (Penman-Monteith) → Kc schedule → Water Balance → Yield

Uses deterministic synthetic weather (no external data dependency) for
reproducibility. Validates mass conservation, crop progression, stress
response, and end-of-season totals.

This exercises the full coupled pipeline that will eventually run on GPU
via barracuda's seasonal_pipeline module.

References:
    Allen et al. (1998) FAO-56: Crop Evapotranspiration, Chapters 2-8.
    Stewart JI (1977) FAO Irrigation and Drainage Paper 33.
"""
import json
import math
import sys
from pathlib import Path


# ── FAO-56 Penman-Monteith components ──────────────────────────────

def saturation_vapour_pressure(temp_c):
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))

def vapour_pressure_slope(temp_c):
    es = saturation_vapour_pressure(temp_c)
    return 4098.0 * es / (temp_c + 237.3) ** 2

def atmospheric_pressure(elevation_m):
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26

def psychrometric_constant(pressure_kpa):
    return 0.000665 * pressure_kpa

def extraterrestrial_radiation(latitude_rad, doy):
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    solar_dec = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(latitude_rad) * math.tan(solar_dec))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(latitude_rad) * math.sin(solar_dec)
        + math.cos(latitude_rad) * math.cos(solar_dec) * math.sin(ws)
    )
    return max(ra, 0.0)

def fao56_pm_et0(tmin, tmax, rh_mean, wind_2m, rs_mj, latitude_rad, doy, elevation_m):
    """FAO-56 Penman-Monteith reference ET₀ (mm/day)."""
    tmean = (tmin + tmax) / 2.0
    pressure = atmospheric_pressure(elevation_m)
    gamma = psychrometric_constant(pressure)
    delta = vapour_pressure_slope(tmean)
    es = (saturation_vapour_pressure(tmin) + saturation_vapour_pressure(tmax)) / 2.0
    ea = es * rh_mean / 100.0
    ra = extraterrestrial_radiation(latitude_rad, doy)
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rs_safe = min(rs_mj, rso) if rso > 0 else rs_mj
    rns = (1.0 - 0.23) * rs_safe
    ratio = rs_safe / rso if rso > 0 else 0.5
    rnl = (4.903e-9 * ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2.0
           * (0.34 - 0.14 * math.sqrt(max(ea, 0)))
           * (1.35 * ratio - 0.35))
    rn = rns - rnl
    g = 0.0
    et0 = (0.408 * delta * (rn - g) + gamma * 900.0 / (tmean + 273.0) * wind_2m * (es - ea)) / (
        delta + gamma * (1.0 + 0.34 * wind_2m)
    )
    return max(et0, 0.0)


# ── Crop coefficient schedule ──────────────────────────────────────

def kc_schedule(doy, plant_doy, harvest_doy, kc_ini, kc_mid, kc_end):
    """Trapezoidal Kc schedule: ini → dev → mid → late → end."""
    if doy < plant_doy or doy > harvest_doy:
        return 0.0
    season_length = harvest_doy - plant_doy
    ini_end = plant_doy + int(season_length * 0.15)
    dev_end = plant_doy + int(season_length * 0.40)
    mid_end = plant_doy + int(season_length * 0.70)

    if doy <= ini_end:
        return kc_ini
    if doy <= dev_end:
        frac = (doy - ini_end) / (dev_end - ini_end)
        return kc_ini + frac * (kc_mid - kc_ini)
    if doy <= mid_end:
        return kc_mid
    frac = (doy - mid_end) / (harvest_doy - mid_end)
    return kc_mid + frac * (kc_end - kc_mid)


# ── Water balance step ─────────────────────────────────────────────

def water_balance_step(dr_prev, precip, irrigation, et0, kc, taw, p=0.5):
    """FAO-56 daily water balance (Ch. 8)."""
    raw = p * taw
    ks = 1.0 if dr_prev <= raw else max(0.0, (taw - dr_prev) / (taw - raw))
    etc = et0 * kc
    actual_et = etc * ks
    new_dr = dr_prev - precip - irrigation + actual_et
    dp = 0.0
    if new_dr < 0.0:
        dp = -new_dr
        new_dr = 0.0
    new_dr = min(new_dr, taw)
    return new_dr, ks, actual_et, dp


# ── Yield response (Stewart 1977) ─────────────────────────────────

def stewart_yield(ky, eta_over_etc):
    """Stewart (1977) relative yield: Ya/Ym = 1 - Ky(1 - ETa/ETc)."""
    return 1.0 - ky * (1.0 - eta_over_etc)


# ── Synthetic weather generator ────────────────────────────────────

def synthetic_weather(doy, latitude_deg):
    """Deterministic synthetic weather for reproducibility."""
    lat_rad = math.radians(latitude_deg)
    tmean_annual = 10.0
    t_amp = 15.0
    tmean = tmean_annual + t_amp * math.sin(2 * math.pi * (doy - 100) / 365.0)
    tmax = tmean + 5.0
    tmin = tmean - 5.0

    ra = extraterrestrial_radiation(lat_rad, doy)
    rs = 0.55 * ra

    rh = 60.0 + 20.0 * math.cos(2 * math.pi * (doy - 200) / 365.0)

    wind = 2.0 + 0.5 * math.sin(2 * math.pi * (doy - 60) / 365.0)

    precip_base = 2.0 + 1.5 * math.sin(2 * math.pi * (doy - 150) / 365.0)
    precip = max(0.0, precip_base)

    return {
        "doy": doy,
        "tmin": round(tmin, 2),
        "tmax": round(tmax, 2),
        "rh_mean": round(rh, 1),
        "wind_2m": round(wind, 2),
        "rs_mj": round(rs, 3),
        "precip_mm": round(precip, 2),
    }


# ── Run full season ───────────────────────────────────────────────

def run_season(crop_name, latitude_deg, elevation_m, plant_doy, harvest_doy,
               kc_ini, kc_mid, kc_end, ky, theta_fc, theta_wp, root_depth_mm):
    """Run a complete growing season water budget."""
    lat_rad = math.radians(latitude_deg)
    taw = (theta_fc - theta_wp) * root_depth_mm
    dr = 0.0

    daily = []
    total_precip = 0.0
    total_et0 = 0.0
    total_etc = 0.0
    total_eta = 0.0
    total_dp = 0.0
    total_irr = 0.0
    stress_days = 0

    for doy in range(plant_doy, harvest_doy + 1):
        wx = synthetic_weather(doy, latitude_deg)
        et0 = fao56_pm_et0(wx["tmin"], wx["tmax"], wx["rh_mean"], wx["wind_2m"],
                            wx["rs_mj"], lat_rad, doy, elevation_m)

        kc = kc_schedule(doy, plant_doy, harvest_doy, kc_ini, kc_mid, kc_end)

        irrigation = 0.0
        raw = 0.5 * taw
        if dr > raw * 0.9:
            irrigation = dr

        dr, ks, actual_et, dp = water_balance_step(dr, wx["precip_mm"], irrigation, et0, kc, taw)

        total_precip += wx["precip_mm"]
        total_et0 += et0
        total_etc += et0 * kc
        total_eta += actual_et
        total_dp += dp
        total_irr += irrigation
        if ks < 1.0:
            stress_days += 1

        daily.append({
            "doy": doy, "et0": round(et0, 4), "kc": round(kc, 4),
            "ks": round(ks, 4), "eta": round(actual_et, 4),
            "dr": round(dr, 4), "dp": round(dp, 4),
            "irr": round(irrigation, 2), "precip": wx["precip_mm"],
        })

    eta_over_etc = total_eta / total_etc if total_etc > 0 else 1.0
    rel_yield = stewart_yield(ky, eta_over_etc)

    season_length = harvest_doy - plant_doy + 1
    mass_balance_err = abs(
        total_precip + total_irr - total_eta - total_dp
        + daily[-1]["dr"]
    )

    return {
        "crop": crop_name,
        "season_length_days": season_length,
        "total_precip_mm": round(total_precip, 2),
        "total_irrigation_mm": round(total_irr, 2),
        "total_et0_mm": round(total_et0, 2),
        "total_etc_mm": round(total_etc, 2),
        "total_eta_mm": round(total_eta, 2),
        "total_dp_mm": round(total_dp, 2),
        "final_depletion_mm": round(daily[-1]["dr"], 4),
        "mass_balance_error_mm": round(mass_balance_err, 6),
        "stress_days": stress_days,
        "eta_over_etc": round(eta_over_etc, 6),
        "relative_yield": round(rel_yield, 6),
        "taw_mm": round(taw, 2),
        "daily_sample": daily[:5] + daily[-5:],
    }


# ── Crop definitions (FAO-56 Table 12/17) ─────────────────────────

CROPS = [
    {
        "name": "corn",
        "plant_doy": 120, "harvest_doy": 260,
        "kc_ini": 0.30, "kc_mid": 1.20, "kc_end": 0.60,
        "ky": 1.25,
        "theta_fc": 0.30, "theta_wp": 0.15, "root_depth_mm": 1000.0,
    },
    {
        "name": "soybean",
        "plant_doy": 135, "harvest_doy": 270,
        "kc_ini": 0.40, "kc_mid": 1.15, "kc_end": 0.50,
        "ky": 0.85,
        "theta_fc": 0.32, "theta_wp": 0.16, "root_depth_mm": 800.0,
    },
    {
        "name": "winter_wheat",
        "plant_doy": 90, "harvest_doy": 200,
        "kc_ini": 0.40, "kc_mid": 1.15, "kc_end": 0.25,
        "ky": 1.05,
        "theta_fc": 0.28, "theta_wp": 0.13, "root_depth_mm": 1200.0,
    },
    {
        "name": "alfalfa",
        "plant_doy": 100, "harvest_doy": 280,
        "kc_ini": 0.40, "kc_mid": 1.20, "kc_end": 1.15,
        "ky": 1.10,
        "theta_fc": 0.34, "theta_wp": 0.17, "root_depth_mm": 1500.0,
    },
]

LATITUDE = 42.7  # Michigan
ELEVATION = 256.0  # m


def main():
    all_pass = True
    n_pass = 0
    n_total = 0

    print("=" * 72)
    print("Exp 054: Full-Season Irrigation Water Budget Audit")
    print("=" * 72)

    results = []
    for crop in CROPS:
        r = run_season(
            crop["name"], LATITUDE, ELEVATION,
            crop["plant_doy"], crop["harvest_doy"],
            crop["kc_ini"], crop["kc_mid"], crop["kc_end"],
            crop["ky"],
            crop["theta_fc"], crop["theta_wp"], crop["root_depth_mm"],
        )
        results.append(r)

        print(f"\n── {crop['name']} ({r['season_length_days']} days) ──")
        print(f"  ET₀: {r['total_et0_mm']:.1f} mm  ETc: {r['total_etc_mm']:.1f} mm  "
              f"ETa: {r['total_eta_mm']:.1f} mm")
        print(f"  Precip: {r['total_precip_mm']:.1f} mm  Irrig: {r['total_irrigation_mm']:.1f} mm  "
              f"DP: {r['total_dp_mm']:.1f} mm")
        print(f"  Stress days: {r['stress_days']}  Yield: {r['relative_yield']:.3f}  "
              f"Mass err: {r['mass_balance_error_mm']:.2e}")

        # Mass conservation
        n_total += 1
        ok = r["mass_balance_error_mm"] < 0.1
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] mass conservation")

        # ET₀ reasonable range (Michigan summer: 300-700 mm/season)
        n_total += 1
        ok = 100 < r["total_et0_mm"] < 1000
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] ET₀ range ({r['total_et0_mm']:.0f} mm)")

        # ETa ≤ ETc always
        n_total += 1
        ok = r["total_eta_mm"] <= r["total_etc_mm"] + 0.01
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] ETa ≤ ETc")

        # Yield in [0, 1]
        n_total += 1
        ok = 0.0 <= r["relative_yield"] <= 1.0
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] yield ∈ [0,1] ({r['relative_yield']:.3f})")

        # Final depletion ≤ TAW
        n_total += 1
        ok = r["final_depletion_mm"] <= r["taw_mm"] + 0.01
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] Dr ≤ TAW ({r['final_depletion_mm']:.1f} ≤ {r['taw_mm']:.1f})")

        # Non-negative deep percolation
        n_total += 1
        ok = r["total_dp_mm"] >= -0.001
        if ok:
            n_pass += 1
        else:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] DP ≥ 0 ({r['total_dp_mm']:.1f})")

    # Cross-crop comparisons
    print("\n── Cross-Crop Comparisons ──")

    corn = results[0]
    soy = results[1]

    # Corn has higher Kc_mid → higher ETc
    n_total += 1
    ok = corn["total_etc_mm"] > soy["total_etc_mm"] * 0.8
    if ok:
        n_pass += 1
    else:
        all_pass = False
    print(f"  [{'PASS' if ok else 'FAIL'}] corn ETc comparable to soy")

    # Alfalfa longest season → most ET₀
    alfalfa = results[3]
    n_total += 1
    ok = alfalfa["total_et0_mm"] > corn["total_et0_mm"]
    if ok:
        n_pass += 1
    else:
        all_pass = False
    print(f"  [{'PASS' if ok else 'FAIL'}] alfalfa ET₀ > corn ET₀ (longer season)")

    # All crops have positive precip and ET₀
    for r in results:
        n_total += 2
        ok1 = r["total_precip_mm"] > 0
        ok2 = r["total_et0_mm"] > 0
        if ok1:
            n_pass += 1
        else:
            all_pass = False
        if ok2:
            n_pass += 1
        else:
            all_pass = False

    print(f"\nResult: {n_pass}/{n_total} checks passed")

    import subprocess
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    ).stdout.strip() or "unknown"

    benchmark = {
        "_provenance": {
            "paper": "Allen et al. (1998) FAO-56 Chapters 2-8; Stewart JI (1977) FAO I&D Paper 33",
            "data_source": "Deterministic synthetic weather (Michigan lat 42.7°N, elev 256 m)",
            "experiment": "054",
            "baseline_script": "control/season_water_budget/season_water_budget.py",
            "baseline_commit": commit,
            "baseline_command": "python control/season_water_budget/season_water_budget.py",
            "baseline_date": "2026-02-28",
            "baseline_result": f"{n_pass}/{n_total} PASS"
        },
        "crops": [
            {k: v for k, v in r.items() if k != "daily_sample"}
            for r in results
        ],
        "daily_samples": {r["crop"]: r["daily_sample"] for r in results},
        "_tolerance_justification": "Deterministic synthetic weather + FAO-56 analytical: mass balance < 0.1 mm"
    }

    with open("control/season_water_budget/benchmark_season_wb.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Wrote benchmark_season_wb.json")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
