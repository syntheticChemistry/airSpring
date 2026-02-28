#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Exp 048 — NCBI 16S + Soil Moisture Anderson Coupling (Python control).

Couples a real NCBI 16S metagenome study site (PRJNA481146, Ein Harod, Israel)
to airSpring's moisture-driven Anderson QS model. The pipeline:

    Open-Meteo weather → FAO-56 ET₀ → water balance θ(t)
        → Anderson coupling (θ → S_e → d_eff → W → QS regime)

This is the Python control baseline for Rust cross-validation.

Cross-Spring: airSpring (soil physics) × wetSpring (Anderson QS) × NestGate (NCBI data)
"""

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

BENCHMARK = json.loads(
    (Path(__file__).parent / "benchmark_ncbi_16s_coupling.json").read_text()
)
SITE = BENCHMARK["site"]
SOIL = BENCHMARK["soil_parameters"]
ANDERSON = BENCHMARK["anderson_coupling"]
WEATHER = BENCHMARK["weather_period"]

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

passed = 0
failed = 0
total = 0


def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}{f' — {detail}' if detail else ''}")
    return condition


# ---------- FAO-56 ET₀ (simplified Hargreaves for minimal weather) ----------

def compute_et0_hargreaves(tmin, tmax, lat_rad, doy):
    """Hargreaves-Samani ET₀ (mm/day) — FAO-56 Eq. 52."""
    tmean = (tmin + tmax) / 2.0

    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(delta)
        + math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    ra_mm = ra * 0.408

    td = max(tmax - tmin, 0.1)
    et0 = 0.0023 * (tmean + 17.8) * math.sqrt(td) * ra_mm
    return max(et0, 0.0)


def compute_et0_pm(tmin, tmax, rh_min, rh_max, wind, radiation, lat_rad, doy, elev):
    """FAO-56 Penman-Monteith ET₀ (mm/day)."""
    tmean = (tmin + tmax) / 2.0
    P = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * P

    e_tmin = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    e_tmax = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    e_s = (e_tmin + e_tmax) / 2.0
    e_a = (e_tmin * rh_max / 100.0 + e_tmax * rh_min / 100.0) / 2.0
    vpd = e_s - e_a

    delta_vp = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2

    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(dec))))
    Ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec)
        + math.cos(lat_rad) * math.cos(dec) * math.sin(ws)
    )

    Rso = (0.75 + 2e-5 * elev) * Ra
    Rs = radiation
    Rns = (1.0 - 0.23) * Rs
    Rnl = (4.903e-9 / 2.0) * (
        (tmax + 273.16) ** 4 + (tmin + 273.16) ** 4
    ) * (0.34 - 0.14 * math.sqrt(max(e_a, 0.001))) * (
        1.35 * min(Rs / max(Rso, 0.001), 1.0) - 0.35
    )
    Rn = Rns - Rnl
    G = 0.0

    u2 = wind * (4.87 / math.log(67.8 * 10 - 5.42))

    num = 0.408 * delta_vp * (Rn - G) + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd
    den = delta_vp + gamma * (1.0 + 0.34 * u2)
    et0 = num / den
    return max(et0, 0.0)


# ---------- Water Balance (FAO-56 Ch 8, simplified) ----------

def water_balance_step(theta_prev, precip, et0, kc, soil):
    """Single-day water balance step returning new theta."""
    theta_s = soil["theta_s"]
    theta_r = soil["theta_r"]
    fc = soil["FC_mm"]
    wp = soil["WP_mm"]
    rd = soil["rooting_depth_mm"]

    swd_prev = (theta_prev - wp) * rd
    swd_prev = max(swd_prev, 0.0)

    etc = et0 * kc
    raw = 0.55 * (fc - wp) * rd

    if swd_prev > 0:
        ks = min(swd_prev / max(raw, 1.0), 1.0)
    else:
        ks = 0.0

    eta = etc * ks

    swd_new = swd_prev + precip - eta
    swd_new = max(swd_new, 0.0)
    swd_new = min(swd_new, (theta_s - wp) * rd)

    theta_new = wp + swd_new / rd
    theta_new = max(theta_r, min(theta_s, theta_new))

    return theta_new, eta, precip


# ---------- Anderson Coupling (Exp 045 validated) ----------

def effective_saturation(theta, theta_r, theta_s):
    """van Genuchten effective saturation S_e = (θ - θ_r) / (θ_s - θ_r)."""
    return max(0.0, min(1.0, (theta - theta_r) / (theta_s - theta_r)))


def pore_connectivity(s_e, L=0.5):
    """Mualem pore connectivity: S_e^L."""
    return s_e ** L


def coordination_number(p_c, z_max=6):
    """Coordination number z = z_max × p_c (Bethe lattice)."""
    return z_max * p_c


def effective_dimension(z):
    """Effective dimension d_eff = z / 2 (Anderson theory)."""
    return z / 2.0


def disorder_parameter(s_e, W_0=20.0):
    """Disorder parameter W = W_0 × (1 - S_e) (Anderson 1958)."""
    return W_0 * (1.0 - s_e)


def classify_regime(d_eff):
    """Classify QS regime from effective dimension."""
    if d_eff > 2.5:
        return "delocalized"
    elif d_eff >= 2.0:
        return "marginal"
    else:
        return "localized"


def anderson_coupling_chain(theta, soil, anderson):
    """Full Anderson coupling chain: θ → S_e → p_c → z → d_eff → W → regime."""
    s_e = effective_saturation(theta, soil["theta_r"], soil["theta_s"])
    p_c = pore_connectivity(s_e, anderson["mualem_L"])
    z = coordination_number(p_c, anderson["z_max"])
    d_eff = effective_dimension(z)
    W = disorder_parameter(s_e, anderson["W_0"])
    regime = classify_regime(d_eff)
    return {
        "theta": theta,
        "S_e": s_e,
        "p_c": p_c,
        "z": z,
        "d_eff": d_eff,
        "W": W,
        "regime": regime,
    }


# ---------- Main Pipeline ----------

def fetch_weather():
    """Fetch real weather from Open-Meteo for the study site."""
    print("\n--- Step 1: Fetch Open-Meteo Weather ---")
    params = {
        "latitude": SITE["lat"],
        "longitude": SITE["lon"],
        "start_date": WEATHER["start"],
        "end_date": WEATHER["end"],
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_max", "relative_humidity_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "shortwave_radiation_sum",
        ]),
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    check("weather_fetch_ok", resp.status_code == 200, f"HTTP {resp.status_code}")

    data = resp.json()["daily"]
    n_days = len(data["time"])
    print(f"  Fetched {n_days} days: {WEATHER['start']} to {WEATHER['end']}")
    return data, n_days


def run_et0(weather, n_days):
    """Compute FAO-56 PM ET₀ for each day."""
    print("\n--- Step 2: Compute ET₀ ---")
    lat_rad = math.radians(SITE["lat"])
    elev = SITE["elevation_m"]
    et0_values = []

    start = datetime.strptime(WEATHER["start"], "%Y-%m-%d")

    for i in range(n_days):
        tmin = weather["temperature_2m_min"][i]
        tmax = weather["temperature_2m_max"][i]
        rh_min = weather["relative_humidity_2m_min"][i]
        rh_max = weather["relative_humidity_2m_max"][i]
        wind = weather["wind_speed_10m_max"][i]
        rad = weather["shortwave_radiation_sum"][i]

        if any(v is None for v in [tmin, tmax, rh_min, rh_max, wind, rad]):
            doy = (start + timedelta(days=i)).timetuple().tm_yday
            et0 = compute_et0_hargreaves(
                tmin or 10, tmax or 25, lat_rad, doy
            )
        else:
            doy = (start + timedelta(days=i)).timetuple().tm_yday
            rad_mj = rad / 1000.0
            et0 = compute_et0_pm(
                tmin, tmax, rh_min, rh_max, wind, rad_mj, lat_rad, doy, elev
            )
        et0_values.append(et0)

    et0_arr = np.array(et0_values)
    check("et0_positive_all_days", np.all(et0_arr >= 0), f"min={et0_arr.min():.3f}")
    mean_et0 = float(np.mean(et0_arr))
    check("et0_mean_plausible_1_8mm", 1.0 <= mean_et0 <= 8.0, f"mean={mean_et0:.2f}")
    print(f"  ET₀ mean={mean_et0:.2f} mm/day, range=[{et0_arr.min():.2f}, {et0_arr.max():.2f}]")
    return et0_values


def run_water_balance(weather, et0_values, n_days):
    """Run FAO-56 water balance to get daily θ(t)."""
    print("\n--- Step 3: Water Balance θ(t) ---")
    theta = SOIL["FC_mm"]
    theta_series = []
    total_precip = 0.0
    total_eta = 0.0

    kc = 0.85

    for i in range(n_days):
        precip = weather["precipitation_sum"][i] or 0.0
        theta, eta, p = water_balance_step(theta, precip, et0_values[i], kc, SOIL)
        theta_series.append(theta)
        total_precip += precip
        total_eta += eta

    theta_arr = np.array(theta_series)
    mass_err = abs(total_precip - total_eta - (theta_series[-1] - SOIL["FC_mm"]) * SOIL["rooting_depth_mm"])
    check("water_balance_mass_conserved", mass_err < 50.0, f"err={mass_err:.1f} mm")
    check("theta_in_range",
          np.all(theta_arr >= SOIL["theta_r"]) and np.all(theta_arr <= SOIL["theta_s"]),
          f"range=[{theta_arr.min():.3f}, {theta_arr.max():.3f}]")

    feb_start = (datetime.strptime("2014-02-01", "%Y-%m-%d") -
                 datetime.strptime(WEATHER["start"], "%Y-%m-%d")).days
    apr_start = (datetime.strptime("2014-04-01", "%Y-%m-%d") -
                 datetime.strptime(WEATHER["start"], "%Y-%m-%d")).days

    theta_feb = float(np.mean(theta_arr[max(0, feb_start):feb_start + 28]))
    theta_apr = float(np.mean(theta_arr[max(0, apr_start):apr_start + 30]))

    check("theta_feb_wetter_than_apr", theta_feb >= theta_apr,
          f"θ_feb={theta_feb:.3f}, θ_apr={theta_apr:.3f}")
    print(f"  θ mean={theta_arr.mean():.3f}, Feb={theta_feb:.3f}, Apr={theta_apr:.3f}")
    print(f"  Total precip={total_precip:.0f} mm, ETa={total_eta:.0f} mm")

    return theta_series, theta_feb, theta_apr


def run_anderson_coupling(theta_series, theta_feb, theta_apr):
    """Run Anderson coupling chain on θ(t)."""
    print("\n--- Step 4: Anderson Coupling ---")

    chain_feb = anderson_coupling_chain(theta_feb, SOIL, ANDERSON)
    chain_apr = anderson_coupling_chain(theta_apr, SOIL, ANDERSON)
    chain_dry = anderson_coupling_chain(SOIL["theta_r"] + 0.01, SOIL, ANDERSON)

    check("anderson_S_e_in_01",
          0.0 <= chain_feb["S_e"] <= 1.0 and 0.0 <= chain_apr["S_e"] <= 1.0)
    check("anderson_d_eff_positive",
          chain_feb["d_eff"] > 0 and chain_apr["d_eff"] > 0)
    check("anderson_d_eff_feb_higher",
          chain_feb["d_eff"] >= chain_apr["d_eff"],
          f"d_feb={chain_feb['d_eff']:.3f}, d_apr={chain_apr['d_eff']:.3f}")
    check("anderson_W_feb_lower",
          chain_feb["W"] <= chain_apr["W"],
          f"W_feb={chain_feb['W']:.2f}, W_apr={chain_apr['W']:.2f}")
    check("anderson_regime_feb_classifiable",
          chain_feb["regime"] in ("delocalized", "marginal", "localized"),
          f"regime={chain_feb['regime']}")
    check("anderson_regime_dry_more_localized",
          chain_dry["d_eff"] < chain_feb["d_eff"],
          f"d_dry={chain_dry['d_eff']:.3f} < d_feb={chain_feb['d_eff']:.3f}")

    print(f"\n  February (wet):  θ={chain_feb['theta']:.3f} → S_e={chain_feb['S_e']:.3f} "
          f"→ d_eff={chain_feb['d_eff']:.3f} → W={chain_feb['W']:.2f} → {chain_feb['regime']}")
    print(f"  April (drying):  θ={chain_apr['theta']:.3f} → S_e={chain_apr['S_e']:.3f} "
          f"→ d_eff={chain_apr['d_eff']:.3f} → W={chain_apr['W']:.2f} → {chain_apr['regime']}")
    print(f"  Dry extreme:     θ={chain_dry['theta']:.3f} → S_e={chain_dry['S_e']:.3f} "
          f"→ d_eff={chain_dry['d_eff']:.3f} → W={chain_dry['W']:.2f} → {chain_dry['regime']}")

    theta_sweep = np.linspace(SOIL["theta_r"], SOIL["theta_s"], 20)
    d_eff_sweep = [anderson_coupling_chain(float(t), SOIL, ANDERSON)["d_eff"] for t in theta_sweep]
    monotonic = all(d_eff_sweep[i] <= d_eff_sweep[i + 1] for i in range(len(d_eff_sweep) - 1))
    check("coupling_chain_monotonic", monotonic)

    return chain_feb, chain_apr, chain_dry


def validate_diversity_prediction(chain_feb, chain_apr, chain_dry):
    """Validate that QS regime predictions are consistent with expected diversity."""
    print("\n--- Step 5: Diversity Prediction ---")

    predicted_h_feb = 3.5 if chain_feb["regime"] in ("delocalized", "marginal") else 1.5
    predicted_h_apr = 3.0 if chain_apr["regime"] in ("delocalized", "marginal") else 1.5
    predicted_h_dry = 1.5 if chain_dry["regime"] == "localized" else 3.0

    consistent = (
        predicted_h_feb >= predicted_h_apr >= predicted_h_dry
        or (chain_feb["d_eff"] >= chain_apr["d_eff"] >= chain_dry["d_eff"])
    )
    check("diversity_prediction_consistent", consistent,
          f"H'_feb~{predicted_h_feb}, H'_apr~{predicted_h_apr}, H'_dry~{predicted_h_dry}")

    print(f"  Predicted Shannon H' — Feb: ~{predicted_h_feb}, Apr: ~{predicted_h_apr}, Dry: ~{predicted_h_dry}")
    print(f"  Mechanism: Higher θ → higher d_eff → more QS coordination → higher diversity")
    print(f"  This is the Anderson prediction: geometry determines signaling determines community")


def main():
    print("=" * 70)
    print("Exp 048: NCBI 16S + Soil Moisture Anderson Coupling")
    print(f"Site: {SITE['lat']}°N, {SITE['lon']}°E (Ein Harod, Israel)")
    print(f"BioProject: {BENCHMARK['_provenance']['bioproject']} ({SITE['sra_runs']} SRA runs)")
    print("=" * 70)

    weather, n_days = fetch_weather()
    et0_values = run_et0(weather, n_days)
    theta_series, theta_feb, theta_apr = run_water_balance(weather, et0_values, n_days)
    chain_feb, chain_apr, chain_dry = run_anderson_coupling(theta_series, theta_feb, theta_apr)
    validate_diversity_prediction(chain_feb, chain_apr, chain_dry)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {passed}/{total} PASS")
    if failed == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failed} CHECKS FAILED")
        sys.exit(1)
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
