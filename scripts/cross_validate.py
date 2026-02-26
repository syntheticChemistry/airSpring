# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
airSpring Phase 2 — Python side of the cross-validation harness.

Outputs a JSON object with computed values from identical inputs so that
the Rust binary `cross_validate` can compare its results. The inputs are
hardcoded here and in the Rust binary; they come from FAO-56 Example 18
(Uccle, Belgium, 6 July) and Dong et al. (2024) sensor data.

Usage:
    python scripts/cross_validate.py > /tmp/airspring_python.json
    cd barracuda && cargo run --release --bin cross_validate > /tmp/airspring_rust.json
    diff /tmp/airspring_python.json /tmp/airspring_rust.json
"""

import json
import math
import sys

sys.path.insert(0, "control/fao56")
sys.path.insert(0, "control/soil_sensors")
sys.path.insert(0, "control/iot_irrigation")

from penman_monteith import (
    atmospheric_pressure,
    psychrometric_constant,
    saturation_vapour_pressure,
    slope_vapour_pressure_curve,
    mean_saturation_vapour_pressure,
    actual_vapour_pressure_rh,
    wind_speed_at_2m,
    solar_declination,
    inverse_relative_distance,
    sunset_hour_angle,
    extraterrestrial_radiation,
    daylight_hours,
    solar_radiation_from_sunshine,
    solar_radiation_from_temp,
    clear_sky_radiation,
    net_shortwave_radiation,
    net_longwave_radiation,
    soil_heat_flux_monthly,
    fao56_penman_monteith,
)
from calibration_dong2020 import (
    topp_equation,
    compute_rmse,
    compute_ia,
    compute_mbe,
    compute_r2,
)
from calibration_dong2024 import (
    soilwatch10_vwc,
    irrigation_recommendation,
    multi_layer_irrigation,
)
import numpy as np

# ── Fixed inputs loaded from benchmark JSON (single source of truth) ─

import os as _os

_benchmark_path = _os.path.join(
    _os.path.dirname(__file__), "..", "control", "fao56", "benchmark_fao56.json"
)
with open(_benchmark_path) as _f:
    _bm = json.load(_f)
_uccle_json = _bm["example_18_uccle_daily"]

UCCLE = {
    "tmin": _uccle_json["inputs"]["tmin_c"],
    "tmax": _uccle_json["inputs"]["tmax_c"],
    "tmean": _uccle_json["intermediates"]["tmean_c"],
    "rs": _uccle_json["intermediates"]["rs_mj_m2_day"],
    "wind_10m": _uccle_json["inputs"]["wind_speed_10m_km_h"] / 3.6,
    "rh_min": _uccle_json["inputs"]["rhmin_pct"],
    "rh_max": _uccle_json["inputs"]["rhmax_pct"],
    "elevation_m": _uccle_json["inputs"]["altitude_m"],
    "latitude_deg": _uccle_json["inputs"]["latitude_deg_n"],
    "doy": _uccle_json["inputs"]["day_of_year"],
}

# Soil dielectric values for Topp equation
EPSILON_VALUES = [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# SoilWatch 10 raw counts
RAW_COUNTS = [5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0]

# Statistical test vectors
OBS = [0.10, 0.15, 0.20, 0.25, 0.30]
SIM = [0.11, 0.14, 0.21, 0.24, 0.31]


def round6(x):
    """Round to 6 decimal places for comparison."""
    return round(float(x), 6)


def main():
    u = UCCLE
    results = {}

    # ── Atmospheric ──────────────────────────────────────────────────
    pressure = atmospheric_pressure(u["elevation_m"])
    gamma = psychrometric_constant(pressure)
    delta = slope_vapour_pressure_curve(u["tmean"])
    es = mean_saturation_vapour_pressure(u["tmax"], u["tmin"])
    ea = actual_vapour_pressure_rh(u["tmax"], u["tmin"], u["rh_max"], u["rh_min"])
    u2 = wind_speed_at_2m(u["wind_10m"], 10.0)

    results["atmospheric"] = {
        "pressure_kpa": round6(pressure),
        "gamma_kpa_c": round6(gamma),
        "delta_kpa_c": round6(delta),
        "es_kpa": round6(es),
        "ea_kpa": round6(ea),
        "u2_ms": round6(u2),
    }

    # ── Solar geometry ───────────────────────────────────────────────
    lat_rad = math.radians(u["latitude_deg"])
    dr = inverse_relative_distance(u["doy"])
    decl = solar_declination(u["doy"])
    ws = sunset_hour_angle(lat_rad, decl)
    ra = extraterrestrial_radiation(u["latitude_deg"], u["doy"])
    n_hours = daylight_hours(u["latitude_deg"], u["doy"])

    results["solar"] = {
        "dr": round6(dr),
        "declination_rad": round6(decl),
        "sunset_hour_angle_rad": round6(ws),
        "ra_mj_m2_day": round6(ra),
        "daylight_hours": round6(n_hours),
    }

    # ── Radiation ────────────────────────────────────────────────────
    rso = clear_sky_radiation(u["elevation_m"], ra)
    rns = net_shortwave_radiation(u["rs"], 0.23)
    rs_over_rso = min(u["rs"] / rso, 1.0) if rso > 0 else 0.05
    rnl = net_longwave_radiation(u["tmax"], u["tmin"], ea, rs_over_rso)
    rn = rns - rnl

    results["radiation"] = {
        "rso_mj_m2_day": round6(rso),
        "rns_mj_m2_day": round6(rns),
        "rnl_mj_m2_day": round6(rnl),
        "rn_mj_m2_day": round6(rn),
    }

    # ── ET₀ ──────────────────────────────────────────────────────────
    g = 0.0  # Daily G = 0
    vpd = es - ea
    et0 = fao56_penman_monteith(rn, g, u["tmean"], u2, vpd, delta, gamma)

    results["et0"] = {
        "vpd_kpa": round6(vpd),
        "et0_mm_day": round6(et0),
    }

    # ── Topp equation ────────────────────────────────────────────────
    results["topp"] = {
        f"theta_eps_{int(e)}": round6(topp_equation(e)) for e in EPSILON_VALUES
    }

    # ── SoilWatch 10 ─────────────────────────────────────────────────
    results["soilwatch10"] = {
        f"vwc_rc_{int(rc)}": round6(soilwatch10_vwc(rc)) for rc in RAW_COUNTS
    }

    # ── Irrigation recommendation ─────────────────────────────────────
    ir_single = irrigation_recommendation(0.12, 0.08, 30.0)
    layers = [
        {"fc": 0.12, "vwc": 0.08, "depth_cm": 30.0},
        {"fc": 0.15, "vwc": 0.10, "depth_cm": 30.0},
        {"fc": 0.18, "vwc": 0.12, "depth_cm": 30.0},
    ]
    ir_multi = multi_layer_irrigation(layers)

    results["irrigation"] = {
        "ir_single_cm": round6(ir_single),
        "ir_multi_cm": round6(ir_multi),
    }

    # ── Statistical measures ──────────────────────────────────────────
    obs = np.array(OBS)
    sim = np.array(SIM)

    results["statistics"] = {
        "rmse": round6(compute_rmse(obs, sim)),
        "mbe": round6(compute_mbe(obs, sim)),
        "ia": round6(compute_ia(obs, sim)),
        "r2": round6(compute_r2(obs, sim)),
    }

    # ── Sunshine-based radiation (FAO-56 Eq. 35) ─────────────────────
    n_sunshine = 9.25  # Uccle, July
    rs_sunshine = solar_radiation_from_sunshine(n_sunshine, n_hours, ra)
    results["sunshine_radiation"] = {
        "rs_sunshine_mj": round6(rs_sunshine),
    }

    # ── Temperature-based radiation (FAO-56 Eq. 50, Hargreaves) ─────
    rs_temp_interior = solar_radiation_from_temp(u["tmax"], u["tmin"], ra, 0.16)
    rs_temp_coastal = solar_radiation_from_temp(u["tmax"], u["tmin"], ra, 0.19)
    results["temp_radiation"] = {
        "rs_temp_interior_mj": round6(rs_temp_interior),
        "rs_temp_coastal_mj": round6(rs_temp_coastal),
    }

    # ── Monthly soil heat flux (FAO-56 Eq. 43) ──────────────────────
    g_warming = soil_heat_flux_monthly(25.0, 22.0)
    g_cooling = soil_heat_flux_monthly(18.0, 25.0)
    results["soil_heat_flux"] = {
        "g_warming_mj": round6(g_warming),
        "g_cooling_mj": round6(g_cooling),
    }

    # ── Hargreaves ET₀ (FAO-56 Eq. 52) ──────────────────────────────
    ra_mm = ra / 2.45  # MJ → mm/day equivalent
    tmean_h = (u["tmax"] + u["tmin"]) / 2.0
    harg_et0 = 0.0023 * (tmean_h + 17.8) * math.sqrt(u["tmax"] - u["tmin"]) * ra_mm
    results["hargreaves"] = {
        "ra_mm_day": round6(ra_mm),
        "et0_hargreaves_mm": round6(harg_et0),
    }

    # ── SVP table (for intermediate cross-check) ─────────────────────
    temps = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    results["svp_table"] = {
        f"es_{int(t)}c": round6(saturation_vapour_pressure(t)) for t in temps
    }

    # ── Low-level PM (same result, different API path) ──────────────
    pm_lowlevel = fao56_penman_monteith(rn, 0.0, u["tmean"], u2, vpd, delta, gamma)
    results["lowlevel_pm"] = {
        "et0_lowlevel_mm": round6(pm_lowlevel),
    }

    # ── Standalone water balance functions ───────────────────────────
    sys.path.insert(0, "control/water_balance")
    from fao56_water_balance import (
        total_available_water,
        readily_available_water,
        stress_coefficient,
        daily_water_balance_step,
    )

    # Python uses root_depth_m (meters), Rust uses root_depth_mm
    # 0.5 m = 500 mm → TAW should be 100 mm
    taw = total_available_water(0.30, 0.10, 0.5)
    raw_wb = readily_available_water(taw, 0.5)
    ks_at_raw = stress_coefficient(raw_wb, taw, raw_wb)
    ks_at_midpoint = stress_coefficient((taw + raw_wb) / 2.0, taw, raw_wb)
    wb_result = daily_water_balance_step(20.0, 5.0, 0.0, 4.0, 1.0, 1.0, taw)

    results["water_balance_standalone"] = {
        "taw_mm": round6(taw),
        "raw_mm": round6(raw_wb),
        "ks_at_raw": round6(ks_at_raw),
        "ks_at_midpoint": round6(ks_at_midpoint),
        "dr_new_mm": round6(wb_result["Dr"]),
        "actual_et_mm": round6(wb_result["ETc_adj"]),
        "deep_percolation_mm": round6(wb_result["DP"]),
    }

    # ── Correction model evaluation ─────────────────────────────────
    from calibration_dong2020 import (
        linear_model,
        quadratic_model,
        exponential_model,
        logarithmic_model,
    )

    lin_val = linear_model(0.15, 1.2, 0.01)
    quad_val = quadratic_model(0.15, 2.0, 1.0, 0.05)
    exp_val = exponential_model(0.15, 0.1, 3.0)
    log_val = logarithmic_model(0.15, 0.2, 0.1)

    results["correction_models"] = {
        "linear_val": round6(lin_val),
        "quadratic_val": round6(quad_val),
        "exponential_val": round6(exp_val),
        "logarithmic_val": round6(log_val),
    }

    # ── Richards van Genuchten retention (Exp 006) ────────────────────
    def vg_theta(h, theta_r, theta_s, alpha, n):
        """Van Genuchten water retention (Eq. 1)."""
        if h >= 0:
            return theta_s
        m = 1.0 - 1.0 / n
        x = (alpha * abs(h)) ** n
        se = 1.0 / (1.0 + x) ** m
        return theta_r + (theta_s - theta_r) * se

    def vg_k(h, ks, theta_r, theta_s, alpha, n):
        """Van Genuchten-Mualem conductivity."""
        if h >= 0:
            return ks
        m = 1.0 - 1.0 / n
        theta = vg_theta(h, theta_r, theta_s, alpha, n)
        se = (theta - theta_r) / (theta_s - theta_r)
        if se <= 0:
            return 0.0
        if se >= 1:
            return ks
        term = 1.0 - se ** (1.0 / m)
        kr = se ** 0.5 * (1.0 - term ** m) ** 2
        return ks * max(0.0, min(1.0, kr))

    sand = {"theta_r": 0.045, "theta_s": 0.43, "alpha": 0.145, "n": 2.68, "ks": 712.8}

    results["richards"] = {
        "theta_h0": round6(vg_theta(0.0, **{k: sand[k] for k in ["theta_r", "theta_s", "alpha", "n"]})),
        "theta_h10": round6(vg_theta(-10.0, **{k: sand[k] for k in ["theta_r", "theta_s", "alpha", "n"]})),
        "theta_h100": round6(vg_theta(-100.0, **{k: sand[k] for k in ["theta_r", "theta_s", "alpha", "n"]})),
        "k_h0": round6(vg_k(0.0, **sand)),
        "k_h10": round6(vg_k(-10.0, **sand)),
    }

    # ── Biochar isotherm predictions (Exp 007) ──────────────────────
    def langmuir_pred(ce, qmax, kl):
        return qmax * kl * ce / (1.0 + kl * ce)

    def freundlich_pred(ce, kf, n_inv):
        return kf * max(ce, 1e-10) ** n_inv

    def langmuir_rl(kl, c0):
        return 1.0 / (1.0 + kl * c0)

    qmax, kl = 18.0, 0.05
    kf, n_iso = 2.0, 2.0
    ce_vals = [1.0, 10.0, 50.0, 100.0]

    results["isotherm"] = {
        "langmuir": {f"ce_{int(c)}": round6(langmuir_pred(c, qmax, kl)) for c in ce_vals},
        "freundlich": {f"ce_{int(c)}": round6(freundlich_pred(c, kf, 1.0 / n_iso)) for c in ce_vals},
        "rl_c0_100": round6(langmuir_rl(kl, 100.0)),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
