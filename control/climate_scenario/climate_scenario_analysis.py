# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 058: Climate Scenario Water Demand Analysis for airSpring.

Assesses how Michigan crop water demand changes under warming scenarios.
Uses the validated FAO-56 chain with synthetic CMIP6-like temperature offsets
applied to a baseline Michigan growing season.

Scenarios: Baseline (0°C), SSP2-4.5 (+1.5°C), SSP3-7.0 (+2.5°C), SSP5-8.5 (+4.0°C)
Crops: Corn, Soybean, Winter Wheat
Soil: Loam (FC=0.28, WP=0.14)
Season: 153-day Michigan growing season (May 1 – Sep 30)

References:
    Allen RG et al. (1998) FAO-56 Crop Evapotranspiration.
    Stewart JI (1977) Yield Response to Water. FAO Irrig Drain Paper 33.
    IPCC AR6 WG1 Ch4 — CMIP6 scenario projections.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


# ── Constants (FAO-56, CMIP6 AR6 WG1 Ch4) ────────────────────────────────

# Michigan growing season: May 1 (doy 121) to Sep 30 (doy 273) = 153 days
SEASON_START_DOY = 121
SEASON_DAYS = 153
LATITUDE_DEG = 42.5
ELEVATION_M = 250.0

# FAO-56 Table 12 crop coefficients (match Rust CropType)
CROP_PARAMS = {
    "corn": {
        "kc_ini": 0.30,
        "kc_mid": 1.20,
        "kc_end": 0.60,
        "root_depth_m": 0.90,
        "p": 0.55,
        "ky": 1.25,
    },
    "soybean": {
        "kc_ini": 0.40,
        "kc_mid": 1.15,
        "kc_end": 0.50,
        "root_depth_m": 0.60,
        "p": 0.50,
        "ky": 0.85,
    },
    "winter_wheat": {
        "kc_ini": 0.70,
        "kc_mid": 1.15,
        "kc_end": 0.25,
        "root_depth_m": 1.50,
        "p": 0.55,
        "ky": 1.00,
    },
}

# Crop height for Kc climate adjustment (FAO-56 Eq. 62)
CROP_HEIGHT_M = 2.0


# ── FAO-56 Penman-Monteith (match Rust eco::evapotranspiration) ───────────

def saturation_vapour_pressure(t_c):
    """FAO-56 Eq. 11."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def vapour_pressure_slope(t_c):
    """FAO-56 Eq. 13."""
    es = saturation_vapour_pressure(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2


def atmospheric_pressure(altitude_m):
    """FAO-56 Eq. 7."""
    return 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa):
    """FAO-56 Eq. 8."""
    return 0.000665 * pressure_kpa


def actual_vapour_pressure_rh(tmin_c, tmax_c, rh_min, rh_max):
    """FAO-56 Eq. 17: ea = [e°(Tmin)*RHmax + e°(Tmax)*RHmin] / 200."""
    e_tmin = saturation_vapour_pressure(tmin_c)
    e_tmax = saturation_vapour_pressure(tmax_c)
    return (e_tmin * rh_max / 100.0 + e_tmax * rh_min / 100.0) / 2.0


def solar_declination(doy):
    """FAO-56 Eq. 24."""
    return 0.409 * math.sin(2.0 * math.pi / 365.0 * doy - 1.39)


def inverse_relative_distance(doy):
    """FAO-56 Eq. 23."""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)


def sunset_hour_angle(lat_rad, decl_rad):
    """FAO-56 Eq. 25."""
    arg = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(decl_rad)))
    return math.acos(arg)


def extraterrestrial_radiation(lat_deg, doy):
    """FAO-56 Eq. 21."""
    phi = math.radians(lat_deg)
    dr = inverse_relative_distance(doy)
    delta = solar_declination(doy)
    ws = sunset_hour_angle(phi, delta)
    gsc = 0.0820
    ra = (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )
    return max(ra, 0.0)


def clear_sky_radiation(altitude_m, ra):
    """FAO-56 Eq. 37."""
    return (0.75 + 2e-5 * altitude_m) * ra


def net_shortwave_radiation(rs, albedo=0.23):
    """FAO-56 Eq. 38."""
    return (1.0 - albedo) * rs


def net_longwave_radiation(tmin_c, tmax_c, ea_kpa, rs, rso):
    """FAO-56 Eq. 39."""
    sigma = 4.903e-9
    tmax_k4 = (tmax_c + 273.16) ** 4
    tmin_k4 = (tmin_c + 273.16) ** 4
    avg_k4 = (tmax_k4 + tmin_k4) / 2.0
    humidity_factor = 0.34 - 0.14 * math.sqrt(max(ea_kpa, 0.01))
    ratio = rs / rso if rso > 0 else 0.5
    cloudiness_factor = 1.35 * ratio - 0.35
    return sigma * avg_k4 * humidity_factor * cloudiness_factor


def fao56_daily_et0(tmin, tmax, rh_min, rh_max, wind_2m, rs_mj, lat_deg, doy, elev_m):
    """FAO-56 Penman-Monteith daily ET₀ (mm/day)."""
    tmean = (tmin + tmax) / 2.0
    pressure = atmospheric_pressure(elev_m)
    gamma = psychrometric_constant(pressure)
    delta = vapour_pressure_slope(tmean)
    es = (saturation_vapour_pressure(tmin) + saturation_vapour_pressure(tmax)) / 2.0
    ea = actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max)
    vpd = es - ea

    ra = extraterrestrial_radiation(lat_deg, doy)
    rso = clear_sky_radiation(elev_m, ra)
    rs_safe = min(rs_mj, rso) if rso > 0 else rs_mj
    rns = net_shortwave_radiation(rs_safe)
    rnl = net_longwave_radiation(tmin, tmax, ea, rs_safe, rso)
    rn = rns - rnl
    g = 0.0

    et0 = (
        0.408 * delta * (rn - g)
        + gamma * (900.0 / (tmean + 273.0)) * wind_2m * vpd
    ) / (delta + gamma * (1.0 + 0.34 * wind_2m))
    return max(et0, 0.0)


# ── Kc schedule (match Rust stage_kc: frac = day_idx / total_days) ────────

def stage_kc(day_idx, total_days, kc_ini, kc_mid, kc_end):
    """Trapezoidal Kc by season fraction (Rust seasonal_pipeline::stage_kc)."""
    frac = day_idx / total_days
    if frac < 0.2:
        return kc_ini
    if frac < 0.7:
        return kc_mid
    return kc_end


def adjust_kc_for_climate(kc_table, u2, rh_min, crop_height_m):
    """FAO-56 Eq. 62: Kc climate adjustment."""
    adj = (0.04 * (u2 - 2.0) - 0.004 * (rh_min - 45.0)) * (crop_height_m / 3.0) ** 0.3
    return max(kc_table + adj, 0.0)


# ── Water balance (FAO-56 Ch. 8) ─────────────────────────────────────────

def water_balance_season(et0_daily, precip_daily, kc_daily, theta_fc, theta_wp,
                         root_m, p, irrigation_depth_mm=0.0):
    """Run full season water balance. Rainfed when irrigation_depth_mm=0."""
    taw = 1000.0 * (theta_fc - theta_wp) * root_m
    raw = p * taw
    dr = 0.0
    total_eta = 0.0
    total_etc = 0.0
    total_dp = 0.0
    total_irrig = 0.0
    stress_days = 0

    for d in range(len(et0_daily)):
        ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / (taw - raw))
        if ks < 1.0:
            stress_days += 1

        etc = kc_daily[d] * et0_daily[d]
        eta = ks * etc

        irrigation = irrigation_depth_mm if dr > raw else 0.0
        total_irrig += irrigation

        dr_new = dr - precip_daily[d] - irrigation + eta
        dp = 0.0
        if dr_new < 0.0:
            dp = -dr_new
            dr_new = 0.0
        dr = min(dr_new, taw)
        total_dp += dp

        total_eta += eta
        total_etc += etc

    ratio = total_eta / total_etc if total_etc > 0 else 1.0
    inflow = np.sum(precip_daily) + total_irrig
    outflow = total_eta + total_dp
    storage_change = 0.0 - dr  # Dr_initial=0, Dr_final=dr
    mass_err = abs(inflow - outflow - storage_change)
    return ratio, total_eta, total_etc, stress_days, mass_err


# ── Stewart yield response ──────────────────────────────────────────────

def yield_ratio_single(ky, eta_etc_ratio):
    """Stewart (1977): Ya/Ymax = 1 - Ky(1 - ETa/ETc)."""
    return max(0.0, min(1.0, 1.0 - ky * (1.0 - eta_etc_ratio)))


# ── Michigan baseline weather (deterministic, no API) ─────────────────────

def michigan_baseline_weather(delta_t, seed=42):
    """
    Generate deterministic Michigan growing season weather.

    Baseline Tmax: 25.0 + 7.0 * sin(π * doy_frac) — published Michigan normals.
    Tmin = Tmax - 10.0
    RH: 65%, Wind: 2.0 m/s
    Solar: 18.0 MJ/m²/day with seasonal variation
    Precip: 3.0 mm/day on 40% of days (deterministic seed)
    """
    rng = np.random.default_rng(seed)
    doys = np.arange(SEASON_START_DOY, SEASON_START_DOY + SEASON_DAYS)
    day_indices = np.arange(SEASON_DAYS)
    doy_frac = day_indices / (SEASON_DAYS - 1)  # 0 to 1 over season

    tmax_base = 25.0 + 7.0 * np.sin(np.pi * doy_frac)
    tmin_base = tmax_base - 10.0
    tmax = tmax_base + delta_t
    tmin = tmin_base + delta_t

    rh_min = np.full(SEASON_DAYS, 65.0)
    rh_max = np.full(SEASON_DAYS, 65.0)
    wind_2m = np.full(SEASON_DAYS, 2.0)

    solar_base = 18.0
    solar = solar_base * (0.85 + 0.15 * np.sin(np.pi * doy_frac))

    rain_occurs = rng.random(SEASON_DAYS) < 0.40
    precip = np.where(rain_occurs, 3.0, 0.0)

    return {
        "doy": doys,
        "tmin": tmin,
        "tmax": tmax,
        "rh_min": rh_min,
        "rh_max": rh_max,
        "wind_2m": wind_2m,
        "solar_rad": solar,
        "precip": precip,
    }


# ── Run scenario for one crop ─────────────────────────────────────────────

def run_crop_scenario(weather, crop_name, theta_fc, theta_wp):
    """Run full pipeline for one crop-scenario."""
    cp = CROP_PARAMS[crop_name]
    et0_daily = []
    kc_daily = []

    for i, doy in enumerate(weather["doy"]):
        et0 = fao56_daily_et0(
            weather["tmin"][i],
            weather["tmax"][i],
            weather["rh_min"][i],
            weather["rh_max"][i],
            weather["wind_2m"][i],
            weather["solar_rad"][i],
            LATITUDE_DEG,
            int(doy),
            ELEVATION_M,
        )
        kc_base = stage_kc(i, SEASON_DAYS, cp["kc_ini"], cp["kc_mid"], cp["kc_end"])
        kc = adjust_kc_for_climate(kc_base, weather["wind_2m"][i],
                                   weather["rh_min"][i], CROP_HEIGHT_M)
        et0_daily.append(et0)
        kc_daily.append(kc)

    eta_etc, total_eta, total_etc, stress_days, mass_err = water_balance_season(
        np.array(et0_daily),
        weather["precip"],
        np.array(kc_daily),
        theta_fc,
        theta_wp,
        cp["root_depth_m"],
        cp["p"],
        irrigation_depth_mm=0.0,
    )
    yr = yield_ratio_single(cp["ky"], eta_etc)
    total_et0 = sum(et0_daily)

    return {
        "total_et0": total_et0,
        "total_etc": total_etc,
        "total_eta": total_eta,
        "stress_days": stress_days,
        "yield_ratio": yr,
        "mass_balance_error": mass_err,
        "et0_daily": et0_daily,
    }


# ── Validation helpers ───────────────────────────────────────────────────

def check(label, computed, expected, tol):
    """Absolute tolerance check."""
    diff = abs(computed - expected)
    ok = diff <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {computed:.6f} (expected {expected:.6f}, tol {tol})")
    return ok


def check_range(label, value, lo, hi):
    """Range check."""
    ok = lo <= value <= hi
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {value:.4f} (range [{lo}, {hi}])")
    return ok


def check_bool(label, condition):
    """Boolean check."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ── Main validation ─────────────────────────────────────────────────────

def main():
    benchmark_path = Path(__file__).parent / "benchmark_climate_scenario.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    theta_fc = benchmark["soil"]["field_capacity"]
    theta_wp = benchmark["soil"]["wilting_point"]
    scenarios = benchmark["scenarios"]
    crops = benchmark["crops"]
    et0_range = benchmark["et0_pct_increase_per_degC"]
    mb_tol = benchmark["tolerance_mass_balance_mm"]

    total_passed = total_failed = 0

    print("=" * 70)
    print("  airSpring Exp 058: Climate Scenario Water Demand Analysis")
    print("  FAO-56 + Stewart + CMIP6-like temperature offsets")
    print("=" * 70)

    results = {}
    for scen in scenarios:
        delta_t = scen["delta_t"]
        weather = michigan_baseline_weather(delta_t)
        results[scen["name"]] = {}
        for crop in crops:
            results[scen["name"]][crop] = run_crop_scenario(
                weather, crop, theta_fc, theta_wp
            )

    # 1. ET₀ increases monotonically with temperature offset
    print("\n── ET₀ Monotonicity with Warming ──")
    for crop in crops:
        et0_vals = [results[s["name"]][crop]["total_et0"] for s in scenarios]
        monotonic = all(et0_vals[i] <= et0_vals[i + 1] for i in range(len(et0_vals) - 1))
        if check_bool(f"{crop}: ET₀ increases with delta_T", monotonic):
            total_passed += 1
        else:
            total_failed += 1

    # 2. Water demand (total ETc) increases with warming
    print("\n── Water Demand (ETc) Increases with Warming ──")
    for crop in crops:
        etc_vals = [results[s["name"]][crop]["total_etc"] for s in scenarios]
        monotonic = all(etc_vals[i] <= etc_vals[i + 1] for i in range(len(etc_vals) - 1))
        if check_bool(f"{crop}: ETc increases with delta_T", monotonic):
            total_passed += 1
        else:
            total_failed += 1

    # 3. Stress days increase with warming for rainfed crops
    print("\n── Stress Days Increase with Warming ──")
    for crop in crops:
        sd_vals = [results[s["name"]][crop]["stress_days"] for s in scenarios]
        monotonic = all(sd_vals[i] <= sd_vals[i + 1] for i in range(len(sd_vals) - 1))
        if check_bool(f"{crop}: stress_days increase with delta_T", monotonic):
            total_passed += 1
        else:
            total_failed += 1

    # 4. Yield ratio decreases with warming for rainfed crops
    print("\n── Yield Ratio Decreases with Warming ──")
    for crop in crops:
        yr_vals = [results[s["name"]][crop]["yield_ratio"] for s in scenarios]
        monotonic = all(yr_vals[i] >= yr_vals[i + 1] for i in range(len(yr_vals) - 1))
        if check_bool(f"{crop}: yield_ratio decreases with delta_T", monotonic):
            total_passed += 1
        else:
            total_failed += 1

    # 5. Corn more sensitive than soybean (higher Ky)
    print("\n── Corn More Sensitive Than Soybean ──")
    baseline_corn = results["baseline"]["corn"]["yield_ratio"]
    baseline_soy = results["baseline"]["soybean"]["yield_ratio"]
    ssp585_corn = results["ssp585_2050"]["corn"]["yield_ratio"]
    ssp585_soy = results["ssp585_2050"]["soybean"]["yield_ratio"]
    corn_drop = baseline_corn - ssp585_corn
    soy_drop = baseline_soy - ssp585_soy
    if check_bool("corn yield drop > soybean under SSP5-8.5", corn_drop > soy_drop):
        total_passed += 1
    else:
        total_failed += 1

    # 6. All yield ratios in [0, 1]
    print("\n── Yield Ratios in [0, 1] ──")
    for scen in scenarios:
        for crop in crops:
            yr = results[scen["name"]][crop]["yield_ratio"]
            if check_range(f"{scen['name']} {crop} yield_ratio", yr, 0.0, 1.0):
                total_passed += 1
            else:
                total_failed += 1

    # 7. Mass balance conservation
    print("\n── Mass Balance Conservation ──")
    for scen in scenarios:
        for crop in crops:
            mb = results[scen["name"]][crop]["mass_balance_error"]
            if check(f"{scen['name']} {crop} mass balance", mb, 0.0, mb_tol):
                total_passed += 1
            else:
                total_failed += 1

    # 8. ET₀ per-degree increase in plausible range (3–8% per °C)
    print("\n── ET₀ Per-Degree Increase (FAO-56 Literature) ──")
    baseline_et0 = sum(results["baseline"][c]["total_et0"] for c in crops) / 3
    ssp245_et0 = sum(results["ssp245_2050"][c]["total_et0"] for c in crops) / 3
    pct_per_deg = (ssp245_et0 / baseline_et0 - 1.0) / 1.5 * 100.0
    if check_range("ET₀ % increase per °C", pct_per_deg,
                   et0_range["lo"], et0_range["hi"]):
        total_passed += 1
    else:
        total_failed += 1

    # 9. Cross-crop ET₀ identical (same weather, different Kc/Ky)
    print("\n── Cross-Crop ET₀ Identical (Same Weather) ──")
    for scen in scenarios:
        et0_corn = results[scen["name"]]["corn"]["total_et0"]
        et0_soy = results[scen["name"]]["soybean"]["total_et0"]
        et0_wheat = results[scen["name"]]["winter_wheat"]["total_et0"]
        if check_bool(f"{scen['name']}: corn ET₀ ≈ soybean", abs(et0_corn - et0_soy) < 0.1):
            total_passed += 1
        else:
            total_failed += 1
        if check_bool(f"{scen['name']}: corn ET₀ ≈ wheat", abs(et0_corn - et0_wheat) < 0.1):
            total_passed += 1
        else:
            total_failed += 1

    total = total_passed + total_failed
    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
