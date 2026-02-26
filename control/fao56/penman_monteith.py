#!/usr/bin/env python3
"""
airSpring Experiment 001 — FAO-56 Penman-Monteith ET₀ Baseline

Replicates FAO Irrigation and Drainage Paper No. 56 (Allen et al. 1998)
reference evapotranspiration calculations using:

  1. Manual numpy implementation following the paper's calculation sheets
  2. Cross-validation with the pyet library

Benchmark data digitized from FAO-56 Chapter 4 Examples 17, 18, and 20.

Reference:
  Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). Crop
  evapotranspiration — Guidelines for computing crop water requirements.
  FAO Irrigation and Drainage Paper 56. Rome: FAO.
  https://www.fao.org/4/X0490E/x0490e00.htm

Provenance:
  Baseline commit: 94cc51d
  Benchmark output: control/fao56/benchmark_fao56.json
  Reproduction: python control/fao56/penman_monteith.py
  Created: 2026-02-16
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# FAO-56 equations — manual implementation following the paper exactly
# ---------------------------------------------------------------------------

def saturation_vapour_pressure(t_c: float) -> float:
    """FAO-56 Eq. 11: e°(T) = 0.6108 exp(17.27 T / (T + 237.3))"""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def slope_vapour_pressure_curve(t_c: float) -> float:
    """FAO-56 Eq. 13: Δ = 4098 * e°(T) / (T + 237.3)²"""
    es = saturation_vapour_pressure(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2


def atmospheric_pressure(altitude_m: float) -> float:
    """FAO-56 Eq. 7: P = 101.3 ((293 - 0.0065 z) / 293)^5.26"""
    return 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa: float) -> float:
    """FAO-56 Eq. 8: γ = 0.000665 P"""
    return 0.000665 * pressure_kpa


def wind_speed_at_2m(uz: float, z: float) -> float:
    """FAO-56 Eq. 47: u₂ = uz * 4.87 / ln(67.8z - 5.42)"""
    return uz * 4.87 / math.log(67.8 * z - 5.42)


def mean_saturation_vapour_pressure(tmax_c: float, tmin_c: float) -> float:
    """FAO-56 Eq. 12: es = [e°(Tmax) + e°(Tmin)] / 2"""
    return (saturation_vapour_pressure(tmax_c) +
            saturation_vapour_pressure(tmin_c)) / 2.0


def actual_vapour_pressure_rh(tmax_c: float, tmin_c: float,
                               rhmax: float, rhmin: float) -> float:
    """FAO-56 Eq. 17: ea from RHmax and RHmin"""
    e_tmin = saturation_vapour_pressure(tmin_c)
    e_tmax = saturation_vapour_pressure(tmax_c)
    return (e_tmin * (rhmax / 100.0) + e_tmax * (rhmin / 100.0)) / 2.0


def solar_declination(day_of_year: int) -> float:
    """FAO-56 Eq. 24: δ = 0.409 sin(2π/365 J - 1.39)"""
    return 0.409 * math.sin(2.0 * math.pi / 365.0 * day_of_year - 1.39)


def inverse_relative_distance(day_of_year: int) -> float:
    """FAO-56 Eq. 23: dr = 1 + 0.033 cos(2π/365 J)"""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * day_of_year)


def sunset_hour_angle(latitude_rad: float, declination_rad: float) -> float:
    """FAO-56 Eq. 25: ωs = arccos(-tan(φ) tan(δ))"""
    arg = -math.tan(latitude_rad) * math.tan(declination_rad)
    arg = max(-1.0, min(1.0, arg))
    return math.acos(arg)


def extraterrestrial_radiation(latitude_deg: float, day_of_year: int) -> float:
    """FAO-56 Eq. 21: Ra (MJ m⁻² day⁻¹)"""
    gsc = 0.0820  # solar constant
    phi = math.radians(latitude_deg)
    dr = inverse_relative_distance(day_of_year)
    delta = solar_declination(day_of_year)
    ws = sunset_hour_angle(phi, delta)

    return (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(ws)
    )


def daylight_hours(latitude_deg: float, day_of_year: int) -> float:
    """FAO-56 Eq. 34: N = 24/π ωs"""
    phi = math.radians(latitude_deg)
    delta = solar_declination(day_of_year)
    ws = sunset_hour_angle(phi, delta)
    return 24.0 / math.pi * ws


def solar_radiation_from_sunshine(n: float, N: float, Ra: float) -> float:
    """FAO-56 Eq. 35: Rs = (as + bs n/N) Ra, default as=0.25, bs=0.50"""
    return (0.25 + 0.50 * n / N) * Ra


def solar_radiation_from_temp(tmax_c: float, tmin_c: float,
                                Ra: float, krs: float = 0.16) -> float:
    """FAO-56 Eq. 50: Rs = kRs √(Tmax - Tmin) Ra (Hargreaves radiation)"""
    return krs * math.sqrt(tmax_c - tmin_c) * Ra


def clear_sky_radiation(altitude_m: float, Ra: float) -> float:
    """FAO-56 Eq. 37: Rso = (0.75 + 2×10⁻⁵ z) Ra"""
    return (0.75 + 2e-5 * altitude_m) * Ra


def net_shortwave_radiation(Rs: float, albedo: float = 0.23) -> float:
    """FAO-56 Eq. 38: Rns = (1 - α) Rs"""
    return (1.0 - albedo) * Rs


def net_longwave_radiation(tmax_c: float, tmin_c: float,
                            ea_kpa: float, Rs_over_Rso: float) -> float:
    """FAO-56 Eq. 39: Rnl"""
    sigma = 4.903e-9  # Stefan-Boltzmann (MJ m⁻² day⁻¹ K⁻⁴)
    tmax_k4 = (tmax_c + 273.16) ** 4
    tmin_k4 = (tmin_c + 273.16) ** 4
    avg_k4 = (tmax_k4 + tmin_k4) / 2.0
    humidity_factor = 0.34 - 0.14 * math.sqrt(ea_kpa)
    cloudiness_factor = 1.35 * Rs_over_Rso - 0.35
    return sigma * avg_k4 * humidity_factor * cloudiness_factor


def soil_heat_flux_monthly(t_month: float, t_month_prev: float) -> float:
    """FAO-56 Eq. 43: G = 0.14 (Ti - Ti-1)"""
    return 0.14 * (t_month - t_month_prev)


def fao56_penman_monteith(rn: float, G: float, tmean_c: float,
                           u2: float, vpd_kpa: float,
                           delta: float, gamma: float) -> float:
    """
    FAO-56 Eq. 6: Reference evapotranspiration ET₀ (mm/day)

      ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)]
            / [Δ + γ(1 + 0.34 u₂)]
    """
    numerator = (0.408 * delta * (rn - G) +
                 gamma * (900.0 / (tmean_c + 273.0)) * u2 * vpd_kpa)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Full ET₀ computation wrappers for each example type
# ---------------------------------------------------------------------------

def compute_example_17_bangkok(inputs: dict) -> dict:
    """Monthly ET₀ with measured vapour pressure (Example 17 pattern)."""
    tmax = inputs["tmax_c"]
    tmin = inputs["tmin_c"]
    ea = inputs["ea_kpa"]
    u2 = inputs["u2_m_s"]
    n = inputs["sunshine_hours"]
    lat = inputs["latitude_deg_n"]
    alt = inputs["altitude_m"]
    doy = inputs["day_of_year"]
    t_month = inputs["t_month_c"]
    t_prev = inputs["t_month_prev_c"]

    tmean = (tmax + tmin) / 2.0
    delta = slope_vapour_pressure_curve(tmean)
    P = atmospheric_pressure(alt)
    gamma = psychrometric_constant(P)
    es = mean_saturation_vapour_pressure(tmax, tmin)
    vpd = es - ea

    Ra = extraterrestrial_radiation(lat, doy)
    N = daylight_hours(lat, doy)
    Rs = solar_radiation_from_sunshine(n, N, Ra)
    Rso = clear_sky_radiation(alt, Ra)
    Rns = net_shortwave_radiation(Rs)
    Rnl = net_longwave_radiation(tmax, tmin, ea, Rs / Rso)
    Rn = Rns - Rnl
    G = soil_heat_flux_monthly(t_month, t_prev)

    et0 = fao56_penman_monteith(Rn, G, tmean, u2, vpd, delta, gamma)

    return {
        "tmean_c": tmean,
        "delta_kpa_per_c": delta,
        "pressure_kpa": P,
        "gamma_kpa_per_c": gamma,
        "es_kpa": es,
        "vpd_kpa": vpd,
        "ra_mj_m2_day": Ra,
        "daylight_hours": N,
        "rs_mj_m2_day": Rs,
        "rso_mj_m2_day": Rso,
        "rns_mj_m2_day": Rns,
        "rnl_mj_m2_day": Rnl,
        "rn_mj_m2_day": Rn,
        "G_mj_m2_day": G,
        "et0_mm_day": et0,
    }


def compute_example_18_uccle(inputs: dict) -> dict:
    """Daily ET₀ with RH data and wind conversion (Example 18 pattern)."""
    tmax = inputs["tmax_c"]
    tmin = inputs["tmin_c"]
    rhmax = inputs["rhmax_pct"]
    rhmin = inputs["rhmin_pct"]
    wind_10m_kmh = inputs["wind_speed_10m_km_h"]
    n = inputs["sunshine_hours"]
    lat = inputs["latitude_deg_n"]
    alt = inputs["altitude_m"]
    doy = inputs["day_of_year"]

    tmean = (tmax + tmin) / 2.0
    uz_ms = wind_10m_kmh / 3.6  # km/h -> m/s
    u2 = wind_speed_at_2m(uz_ms, 10.0)

    delta = slope_vapour_pressure_curve(tmean)
    P = atmospheric_pressure(alt)
    gamma = psychrometric_constant(P)
    es = mean_saturation_vapour_pressure(tmax, tmin)
    ea = actual_vapour_pressure_rh(tmax, tmin, rhmax, rhmin)
    vpd = es - ea

    Ra = extraterrestrial_radiation(lat, doy)
    N = daylight_hours(lat, doy)
    Rs = solar_radiation_from_sunshine(n, N, Ra)
    Rso = clear_sky_radiation(alt, Ra)
    Rns = net_shortwave_radiation(Rs)
    Rnl = net_longwave_radiation(tmax, tmin, ea, Rs / Rso)
    Rn = Rns - Rnl
    G = 0.0  # daily time step

    et0 = fao56_penman_monteith(Rn, G, tmean, u2, vpd, delta, gamma)

    return {
        "tmean_c": tmean,
        "u2_m_s": u2,
        "delta_kpa_per_c": delta,
        "pressure_kpa": P,
        "gamma_kpa_per_c": gamma,
        "es_kpa": es,
        "ea_kpa": ea,
        "vpd_kpa": vpd,
        "ra_mj_m2_day": Ra,
        "daylight_hours": N,
        "rs_mj_m2_day": Rs,
        "rso_mj_m2_day": Rso,
        "rns_mj_m2_day": Rns,
        "rnl_mj_m2_day": Rnl,
        "rn_mj_m2_day": Rn,
        "G_mj_m2_day": G,
        "et0_mm_day": et0,
    }


def compute_example_20_lyon(inputs: dict) -> dict:
    """Monthly ET₀ with only Tmax/Tmin (missing data, Example 20 pattern)."""
    tmax = inputs["tmax_c"]
    tmin = inputs["tmin_c"]
    lat = inputs["latitude_deg_n"]
    alt = inputs["altitude_m"]
    doy = inputs["day_of_year"]
    u2 = inputs["u2_m_s_estimated"]

    tmean = (tmax + tmin) / 2.0
    delta = slope_vapour_pressure_curve(tmean)
    P = atmospheric_pressure(alt)
    gamma = psychrometric_constant(P)

    # Missing humidity: estimate ea from Tmin (FAO-56 Eq. 48)
    ea = saturation_vapour_pressure(tmin)
    es = mean_saturation_vapour_pressure(tmax, tmin)
    vpd = es - ea

    # Missing radiation: estimate Rs from temperature range (Eq. 50)
    Ra = extraterrestrial_radiation(lat, doy)
    Rs = solar_radiation_from_temp(tmax, tmin, Ra, krs=0.16)
    Rso = clear_sky_radiation(alt, Ra)
    Rns = net_shortwave_radiation(Rs)
    Rnl = net_longwave_radiation(tmax, tmin, ea, Rs / Rso)
    Rn = Rns - Rnl
    G = 0.0  # assume negligible

    et0 = fao56_penman_monteith(Rn, G, tmean, u2, vpd, delta, gamma)

    return {
        "tmean_c": tmean,
        "delta_kpa_per_c": delta,
        "pressure_kpa": P,
        "gamma_kpa_per_c": gamma,
        "ea_kpa": ea,
        "es_kpa": es,
        "vpd_kpa": vpd,
        "ra_mj_m2_day": Ra,
        "rs_mj_m2_day": Rs,
        "rso_mj_m2_day": Rso,
        "rns_mj_m2_day": Rns,
        "rnl_mj_m2_day": Rnl,
        "rn_mj_m2_day": Rn,
        "et0_mm_day": et0,
    }


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

def check(label: str, computed: float, expected: float, tol: float) -> bool:
    diff = abs(computed - expected)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"  [{status}] {label}: {computed:.4f} "
          f"(expected {expected:.4f}, tol {tol:.4f}, diff {diff:.4f})")
    return diff <= tol


def validate_component_tables(benchmark: dict) -> tuple:
    """Validate saturation vapour pressure and slope against FAO-56 tables."""
    passed = 0
    failed = 0

    print("\n=== Saturation Vapour Pressure (FAO-56 Table 2.3) ===")
    es_table = benchmark["saturation_vapour_pressure_table"]
    es_tol = es_table["tolerance_kpa"]
    for row in es_table["data"]:
        computed = saturation_vapour_pressure(row["temp_c"])
        if check(f"e°({row['temp_c']:.0f}°C)", computed,
                 row["es_kpa"], es_tol):
            passed += 1
        else:
            failed += 1

    print("\n=== Slope Vapour Pressure Curve (FAO-56 Table 2.4) ===")
    delta_table = benchmark["slope_vapour_pressure_table"]
    delta_tol = delta_table["tolerance_kpa_per_c"]
    for row in delta_table["data"]:
        computed = slope_vapour_pressure_curve(row["temp_c"])
        if check(f"Δ({row['temp_c']:.0f}°C)", computed,
                 row["delta_kpa_per_c"], delta_tol):
            passed += 1
        else:
            failed += 1

    return passed, failed


def validate_example(name: str, compute_fn, example_data: dict) -> tuple:
    """Run a full example and check ET₀ + key intermediates."""
    passed = 0
    failed = 0

    print(f"\n=== {name} ===")
    result = compute_fn(example_data["inputs"])
    expected = example_data["intermediates"]

    # Check key intermediates
    intermediate_tol = {
        "tmean_c": 0.1,
        "delta_kpa_per_c": 0.005,
        "gamma_kpa_per_c": 0.002,
        "es_kpa": 0.02,
        "vpd_kpa": 0.02,
        "ea_kpa": 0.02,
        "ra_mj_m2_day": 0.5,
        "rs_mj_m2_day": 0.3,
        "rso_mj_m2_day": 0.3,
        "rns_mj_m2_day": 0.3,
        "rnl_mj_m2_day": 0.3,
        "rn_mj_m2_day": 0.5,
        "u2_m_s": 0.01,
        "pressure_kpa": 0.2,
        "daylight_hours": 0.2,
    }

    for key, tol in intermediate_tol.items():
        if key in result and key in expected:
            if check(key, result[key], expected[key], tol):
                passed += 1
            else:
                failed += 1

    # Check final ET₀
    et0_expected = example_data["expected_et0_mm_day"]
    et0_tol = example_data["tolerance_mm_day"]
    if check("ET₀ (mm/day)", result["et0_mm_day"], et0_expected, et0_tol):
        passed += 1
    else:
        failed += 1

    return passed, failed


def main():
    benchmark_path = Path(__file__).parent / "benchmark_fao56.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    total_passed = 0
    total_failed = 0

    print("=" * 70)
    print("airSpring Exp 001: FAO-56 Penman-Monteith Baseline Validation")
    print("=" * 70)

    # 1. Component tables
    p, f_ = validate_component_tables(benchmark)
    total_passed += p
    total_failed += f_

    # 2. Example 17 — Bangkok (monthly, measured ea)
    p, f_ = validate_example(
        "Example 17: Bangkok, April (monthly, measured ea)",
        compute_example_17_bangkok,
        benchmark["example_17_bangkok_monthly"],
    )
    total_passed += p
    total_failed += f_

    # 3. Example 18 — Uccle (daily, RH data)
    p, f_ = validate_example(
        "Example 18: Uccle, 6 July (daily, RH + wind conversion)",
        compute_example_18_uccle,
        benchmark["example_18_uccle_daily"],
    )
    total_passed += p
    total_failed += f_

    # 4. Example 20 — Lyon (missing data)
    p, f_ = validate_example(
        "Example 20: Lyon, July (missing data — Tmax/Tmin only)",
        compute_example_20_lyon,
        benchmark["example_20_lyon_missing_data"],
    )
    total_passed += p
    total_failed += f_

    # Summary
    total = total_passed + total_failed
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
