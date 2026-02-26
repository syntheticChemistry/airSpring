# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
airSpring Experiment 015 — OSU Triplett-Van Doren 60-Year Water Balance Reconstruction

Applies the validated FAO-56 water balance (Experiment 004) to a multi-decade
historical record using Open-Meteo's ERA5 archive. Demonstrates the water balance
pipeline at scale over long time horizons with real open data.

Site: Wooster, OH (40.78°N, 81.93°W) — OSU OARDC, Triplett-Van Doren tillage study
Period: 1960-2023 growing seasons (May 1 - Sep 30)
Soil: Wooster silt loam (FC=0.33, WP=0.13)
Crop: Corn (continuous)

Usage:
    python control/long_term_wb/long_term_water_balance.py

Self-contained: uses only numpy, requests. No imports from other control scripts.

Provenance:
  Baseline commit: 3afc229
  Benchmark output: control/long_term_wb/benchmark_long_term_wb.json
  Reproduction: python control/long_term_wb/long_term_water_balance.py
  Created: 2026-02-25
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    requests = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = SCRIPT_DIR / "benchmark_long_term_wb.json"
DATA_DIR = SCRIPT_DIR / "data"
CACHE_FILE = DATA_DIR / "wooster_era5_1960_2023.json"

# ---------------------------------------------------------------------------
# FAO-56 equations — self-contained reimplementation
# ---------------------------------------------------------------------------

def saturation_vapour_pressure(t_c: float) -> float:
    """FAO-56 Eq. 11: e°(T) = 0.6108 exp(17.27 T / (T + 237.3))"""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def slope_vapour_pressure_curve(t_c: float) -> float:
    """FAO-56 Eq. 13: Δ = 4098 * e°(T) / (T + 237.3)²"""
    es = saturation_vapour_pressure(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2


def atmospheric_pressure(altitude_m: float) -> float:
    """FAO-56 Eq. 7"""
    return 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa: float) -> float:
    """FAO-56 Eq. 8: γ = 0.000665 P"""
    return 0.000665 * pressure_kpa


def wind_speed_at_2m(uz_ms: float, z: float = 10.0) -> float:
    """FAO-56 Eq. 47: u₂ = uz * 4.87 / ln(67.8z - 5.42)"""
    return uz_ms * 4.87 / math.log(67.8 * z - 5.42)


def mean_saturation_vapour_pressure(tmax_c: float, tmin_c: float) -> float:
    """FAO-56 Eq. 12"""
    return (saturation_vapour_pressure(tmax_c) + saturation_vapour_pressure(tmin_c)) / 2.0


def solar_declination(day_of_year: int) -> float:
    """FAO-56 Eq. 24"""
    return 0.409 * math.sin(2.0 * math.pi / 365.0 * day_of_year - 1.39)


def inverse_relative_distance(day_of_year: int) -> float:
    """FAO-56 Eq. 23"""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * day_of_year)


def sunset_hour_angle(latitude_rad: float, declination_rad: float) -> float:
    """FAO-56 Eq. 25"""
    arg = max(-1.0, min(1.0, -math.tan(latitude_rad) * math.tan(declination_rad)))
    return math.acos(arg)


def extraterrestrial_radiation(latitude_deg: float, day_of_year: int) -> float:
    """FAO-56 Eq. 21: Ra (MJ m⁻² day⁻¹)"""
    gsc = 0.0820
    phi = math.radians(latitude_deg)
    dr = inverse_relative_distance(day_of_year)
    delta = solar_declination(day_of_year)
    ws = sunset_hour_angle(phi, delta)
    return (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * math.sin(phi) * math.sin(delta) +
        math.cos(phi) * math.cos(delta) * math.sin(ws)
    )


def clear_sky_radiation(altitude_m: float, Ra: float) -> float:
    """FAO-56 Eq. 37"""
    return (0.75 + 2e-5 * altitude_m) * Ra


def net_shortwave_radiation(Rs: float, albedo: float = 0.23) -> float:
    """FAO-56 Eq. 38"""
    return (1.0 - albedo) * Rs


def net_longwave_radiation(tmax_c: float, tmin_c: float, ea_kpa: float,
                           Rs_over_Rso: float) -> float:
    """FAO-56 Eq. 39"""
    sigma = 4.903e-9
    avg_k4 = ((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) / 2.0
    humidity_factor = 0.34 - 0.14 * math.sqrt(ea_kpa)
    cloudiness_factor = 1.35 * Rs_over_Rso - 0.35
    return sigma * avg_k4 * humidity_factor * cloudiness_factor


def fao56_penman_monteith(rn: float, G: float, tmean_c: float, u2: float,
                          vpd_kpa: float, delta: float, gamma: float) -> float:
    """FAO-56 Eq. 6: ET₀ (mm/day)"""
    num = 0.408 * delta * (rn - G) + gamma * (900.0 / (tmean_c + 273.0)) * u2 * vpd_kpa
    den = delta + gamma * (1.0 + 0.34 * u2)
    return num / den


def daily_et0_pm(tmax: float, tmin: float, rh_mean: float, u10_kmh: float,
                 Rs_mj: float, lat_deg: float, doy: int, elevation_m: float) -> float:
    """
    Daily ET₀ via FAO-56 Penman-Monteith.
    Uses rh_mean (approximates ea = es * rh/100 at Tmean), Rs from Open-Meteo.
    u10 in km/h, converted to u2.
    """
    tmean = (tmax + tmin) / 2.0
    es = mean_saturation_vapour_pressure(tmax, tmin)
    ea = es * (rh_mean / 100.0)  # approximation when only mean RH available
    vpd = max(0.01, es - ea)

    delta = slope_vapour_pressure_curve(tmean)
    P = atmospheric_pressure(elevation_m)
    gamma = psychrometric_constant(P)

    Ra = extraterrestrial_radiation(lat_deg, doy)
    Rso = clear_sky_radiation(elevation_m, Ra)
    Rs_over_Rso = min(1.0, Rs_mj / Rso) if Rso > 0 else 0.5
    Rns = net_shortwave_radiation(Rs_mj)
    Rnl = net_longwave_radiation(tmax, tmin, ea, Rs_over_Rso)
    Rn = Rns - Rnl
    G = 0.0

    u10_ms = u10_kmh / 3.6
    u2 = wind_speed_at_2m(u10_ms, 10.0)

    return fao56_penman_monteith(Rn, G, tmean, u2, vpd, delta, gamma)


def hargreaves_et0(tmax: float, tmin: float, lat_deg: float, doy: int) -> float:
    """Hargreaves ET₀ fallback: ET₀ = 0.0023 * Ra * (Tmax-Tmin)^0.5 * (Tmean+17.8)"""
    Ra = extraterrestrial_radiation(lat_deg, doy)
    tmean = (tmax + tmin) / 2.0
    td = max(0.1, tmax - tmin)
    return 0.0023 * Ra * (td ** 0.5) * (tmean + 17.8)


# ---------------------------------------------------------------------------
# Water balance (FAO-56 Chapter 8)
# ---------------------------------------------------------------------------

def total_available_water(theta_fc: float, theta_wp: float,
                          root_depth_m: float) -> float:
    """TAW = 1000 (θFC - θWP) Zr [mm]"""
    return 1000.0 * (theta_fc - theta_wp) * root_depth_m


def readily_available_water(TAW: float, p: float) -> float:
    """RAW = p × TAW [mm]"""
    return p * TAW


def stress_coefficient(Dr: float, TAW: float, RAW: float) -> float:
    """FAO-56 Eq. 84: Ks"""
    if Dr <= RAW:
        return 1.0
    if TAW <= RAW:
        return 0.0
    return max(0.0, min(1.0, (TAW - Dr) / (TAW - RAW)))


def daily_water_balance_step(Dr_prev: float, P: float, I: float, ET0: float,
                              Kc: float, Ks: float, TAW: float) -> dict:
    """FAO-56 Eq. 85: daily balance"""
    ETc_adj = Ks * Kc * ET0
    Dr_new = Dr_prev - P - I + ETc_adj
    DP = 0.0
    if Dr_new < 0:
        DP = -Dr_new
        Dr_new = 0.0
    if Dr_new > TAW:
        Dr_new = TAW
    return {"Dr": Dr_new, "ETc_adj": ETc_adj, "Ks": Ks, "DP": DP, "I": I}


def simulate_season(et0_series: np.ndarray, precip_series: np.ndarray,
                    Kc: float, theta_fc: float, theta_wp: float,
                    root_depth_m: float, p: float,
                    irrigation_trigger: bool = False,
                    irrig_depth_mm: float = 25.0) -> dict:
    """Run full season water balance."""
    n_days = len(et0_series)
    TAW = total_available_water(theta_fc, theta_wp, root_depth_m)
    RAW = readily_available_water(TAW, p)

    Dr = 0.0
    Dr_arr = np.zeros(n_days)
    Ks_arr = np.zeros(n_days)
    ETc_arr = np.zeros(n_days)
    DP_arr = np.zeros(n_days)
    I_arr = np.zeros(n_days)
    total_irrig = 0.0
    irrig_events = 0

    for i in range(n_days):
        Ks = stress_coefficient(Dr, TAW, RAW)
        I = 0.0
        if irrigation_trigger and Dr > RAW:
            I = min(Dr, irrig_depth_mm)
            total_irrig += I
            irrig_events += 1

        result = daily_water_balance_step(
            Dr, precip_series[i], I, et0_series[i], Kc, Ks, TAW)
        Dr = result["Dr"]
        Dr_arr[i] = Dr
        Ks_arr[i] = result["Ks"]
        ETc_arr[i] = result["ETc_adj"]
        DP_arr[i] = result["DP"]
        I_arr[i] = result["I"]

    return {
        "Dr": Dr_arr, "Ks": Ks_arr, "ETc": ETc_arr, "DP": DP_arr, "I": I_arr,
        "TAW": TAW, "RAW": RAW,
        "total_et": np.sum(ETc_arr), "total_precip": np.sum(precip_series),
        "total_dp": np.sum(DP_arr), "total_irrig": total_irrig,
        "irrig_events": irrig_events,
        "initial_Dr": 0.0, "final_Dr": Dr_arr[-1],
    }


def mass_balance_error(result: dict) -> float:
    """|inflow - outflow - storage_change|"""
    inflow = result["total_precip"] + result["total_irrig"]
    outflow = result["total_et"] + result["total_dp"]
    storage_change = result["initial_Dr"] - result["final_Dr"]
    return abs(inflow - outflow - storage_change)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def fetch_open_meteo_chunk(lat: float, lon: float, start: str, end: str,
                          max_retries: int = 3) -> dict | None:
    """Fetch one chunk from Open-Meteo archive API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "et0_fao_evapotranspiration,windspeed_10m_max,relative_humidity_2m_mean,"
        "shortwave_radiation_sum"
        "&timezone=America/New_York"
    )
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  API error ({start}–{end}): {e}")
            if attempt < max_retries - 1:
                time.sleep(30)
    return None


def download_wooster_data(lat: float, lon: float, start_year: int,
                         end_year: int) -> dict | None:
    """Download in 10-year chunks, merge, return daily dict or None."""
    all_times = []
    all_tmax = []
    all_tmin = []
    all_precip = []
    all_et0_om = []
    all_u10 = []
    all_rh = []
    all_Rs = []

    for chunk_start in range(start_year, end_year + 1, 5):
        chunk_end = min(chunk_start + 4, end_year)
        start = f"{chunk_start}-05-01"
        end = f"{chunk_end}-09-30"
        data = fetch_open_meteo_chunk(lat, lon, start, end)
        if data is None:
            return None
        daily = data.get("daily", {})
        all_times.extend(daily.get("time", []))
        all_tmax.extend(daily.get("temperature_2m_max", []))
        all_tmin.extend(daily.get("temperature_2m_min", []))
        all_precip.extend(daily.get("precipitation_sum", []))
        all_et0_om.extend(daily.get("et0_fao_evapotranspiration", []))
        all_u10.extend(daily.get("windspeed_10m_max", []))
        all_rh.extend(daily.get("relative_humidity_2m_mean", []))
        all_Rs.extend(daily.get("shortwave_radiation_sum", []))
        time.sleep(3.0)  # rate limit (Open-Meteo archive: 5 req/min)

    return {
        "time": all_times,
        "temperature_2m_max": all_tmax,
        "temperature_2m_min": all_tmin,
        "precipitation_sum": all_precip,
        "et0_fao_evapotranspiration": all_et0_om,
        "windspeed_10m_max": all_u10,
        "relative_humidity_2m_mean": all_rh,
        "shortwave_radiation_sum": all_Rs,
    }


def date_to_doy(date_str: str) -> int:
    """ISO date string -> day of year (1-365)."""
    from datetime import datetime
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return dt.timetuple().tm_yday


# ---------------------------------------------------------------------------
# Synthetic fallback (when API unreachable)
# ---------------------------------------------------------------------------

def make_synthetic_seasons(n_seasons: int = 10, rng_seed: int = 42) -> list[dict]:
    """Generate synthetic season results for offline testing."""
    rng = np.random.default_rng(rng_seed)
    seasons = []
    for year in range(1960, 1960 + n_seasons):
        n_days = 153  # May 1 - Sep 30
        et0 = np.maximum(rng.normal(4.5, 1.2, n_days), 0.5)
        precip = np.where(rng.random(n_days) < 0.25,
                          rng.exponential(12, n_days), 0.0)
        result = simulate_season(
            et0, precip, Kc=1.2,
            theta_fc=0.33, theta_wp=0.13,
            root_depth_m=0.60, p=0.50,
            irrigation_trigger=True, irrig_depth_mm=25.0)
        result["year"] = year
        result["total_et0"] = np.sum(et0)
        result["total_precip"] = np.sum(precip)
        result["mb_error"] = mass_balance_error(result)
        result["drought_days"] = np.sum(result["Ks"] < 1.0)
        result["seasons_with_dp"] = 1 if result["total_dp"] > 0 else 0
        seasons.append(result)
    return seasons


# ---------------------------------------------------------------------------
# Main: load data, run balance, validate
# ---------------------------------------------------------------------------

def run_long_term_simulation(benchmark: dict) -> tuple[list[dict], bool]:
    """
    Run water balance for all seasons. Returns (season_results, used_real_data).
    """
    site = benchmark["site"]
    lat = site["latitude"]
    lon = site["longitude"]
    elev = site["elevation_m"]
    fc = site["field_capacity"]
    wp = site["wilting_point"]
    zr = site["root_depth_m"]
    p = site["depletion_fraction"]
    Kc = 1.2  # corn mid-season

    # Try to load or download data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    used_real = False
    raw = None

    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                raw = json.load(f)
            used_real = True
            print("  Using cached Open-Meteo data.")
        except Exception as e:
            print(f"  Cache read failed: {e}")

    if raw is None and requests is not None:
        print("  Downloading from Open-Meteo archive API (10-year chunks)...")
        raw = download_wooster_data(lat, lon, 1960, 2023)
        if raw is not None:
            with open(CACHE_FILE, "w") as f:
                json.dump(raw, f, indent=0)
            used_real = True
            print("  Downloaded and cached.")

    if raw is None:
        print("  WARNING: API unreachable. Using SYNTHETIC data (marked in output).")
        return make_synthetic_seasons(10), False

    # Parse into seasons (May 1 - Sep 30)
    times = raw["time"]
    tmax = np.array(raw["temperature_2m_max"], dtype=float)
    tmin = np.array(raw["temperature_2m_min"], dtype=float)
    precip = np.array(raw["precipitation_sum"], dtype=float)
    u10 = np.array(raw["windspeed_10m_max"], dtype=float)
    rh = np.array(raw["relative_humidity_2m_mean"], dtype=float)
    Rs = np.array(raw["shortwave_radiation_sum"], dtype=float)

    # Handle missing
    tmax = np.nan_to_num(tmax, nan=20.0)
    tmin = np.nan_to_num(tmin, nan=12.0)
    precip = np.nan_to_num(precip, nan=0.0)
    u10 = np.nan_to_num(u10, nan=15.0)
    rh = np.nan_to_num(rh, nan=70.0)
    Rs = np.nan_to_num(Rs, nan=20.0)

    seasons_by_year = {}
    for i, t in enumerate(times):
        try:
            y = int(t[:4])
            m = int(t[5:7])
            d = int(t[8:10])
        except (ValueError, IndexError):
            continue
        if 5 <= m <= 9 or (m == 5 and d >= 1) or (m == 9 and d <= 30):
            if y not in seasons_by_year:
                seasons_by_year[y] = []
            seasons_by_year[y].append({
                "tmax": tmax[i], "tmin": tmin[i], "precip": precip[i],
                "u10": u10[i], "rh": rh[i], "Rs": Rs[i],
                "doy": date_to_doy(t), "date": t,
            })

    # Keep only full seasons (May 1 - Sep 30)
    season_results = []
    for year in sorted(seasons_by_year.keys()):
        if year < 1960 or year > 2023:
            continue
        days = seasons_by_year[year]
        if len(days) < 100:  # skip truncated
            continue
        days = sorted(days, key=lambda x: x["date"])
        et0_arr = np.array([
            daily_et0_pm(d["tmax"], d["tmin"], d["rh"], d["u10"], d["Rs"],
                         lat, d["doy"], elev)
            for d in days
        ])
        precip_arr = np.array([d["precip"] for d in days])
        result = simulate_season(
            et0_arr, precip_arr, Kc, fc, wp, zr, p,
            irrigation_trigger=True, irrig_depth_mm=25.0)
        result["year"] = year
        result["total_et0"] = np.sum(et0_arr)
        result["total_precip"] = np.sum(precip_arr)
        result["mb_error"] = mass_balance_error(result)
        result["drought_days"] = int(np.sum(result["Ks"] < 1.0))
        result["seasons_with_dp"] = 1 if result["total_dp"] > 0 else 0
        season_results.append(result)

    return season_results, used_real


def validate_checks(benchmark: dict, seasons: list[dict],
                   used_real: bool) -> tuple[int, int]:
    """Run all validation checks. Return (passed, failed)."""
    checks = benchmark["validation_checks"]
    passed = 0
    failed = 0

    def check_pass(label: str, ok: bool) -> bool:
        nonlocal passed, failed
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if ok:
            passed += 1
        else:
            failed += 1
        return ok

    if not seasons:
        print("  [FAIL] No season data")
        return 0, 1

    n = len(seasons)
    et0_seasonal = np.array([s["total_et0"] for s in seasons])
    precip_seasonal = np.array([s["total_precip"] for s in seasons])
    et_seasonal = np.array([s["total_et"] for s in seasons])
    mb_errors = np.array([s["mb_error"] for s in seasons])
    stress_pct = 100 * np.mean([np.mean(s["Ks"] < 1.0) for s in seasons])
    seasons_with_stress = sum(1 for s in seasons if np.any(s["Ks"] < 1.0))
    stress_season_pct = 100 * seasons_with_stress / n
    seasons_with_dp = sum(s["total_dp"] > 0 for s in seasons)

    # Physical reasonableness
    pr = checks["physical_reasonableness"]["checks"]
    for c in pr:
        cid = c["id"]
        if cid == "annual_et0_range":
            mean_et0 = np.mean(et0_seasonal)
            ok = c["min"] <= mean_et0 <= c["max"]
            check_pass(f"{c['description']}: {mean_et0:.0f} mm", ok)
        elif cid == "seasonal_et0_mean":
            ok = c["min"] <= np.mean(et0_seasonal) <= c["max"]
            check_pass(f"{c['description']}: {np.mean(et0_seasonal):.0f} mm", ok)
        elif cid == "mass_balance":
            max_mb = np.max(mb_errors)
            ok = max_mb <= c["tolerance"]
            check_pass(f"{c['description']}: max error {max_mb:.6f} mm", ok)
        elif cid == "stress_fraction":
            ok = c["min_pct"] <= stress_season_pct <= c["max_pct"]
            check_pass(f"{c['description']}: {stress_season_pct:.1f}% of seasons", ok)
        elif cid == "deep_percolation":
            ok = seasons_with_dp >= c["min_seasons_with_dp"]
            check_pass(f"{c['description']}: {seasons_with_dp} seasons", ok)

    # Climate trends
    ct = checks["climate_trends"]["checks"]
    for c in ct:
        cid = c["id"]
        if cid == "et0_trend":
            x = np.arange(n)
            slope = np.polyfit(x, et0_seasonal, 1)[0]
            ok = slope >= -0.5  # allow small negative
            check_pass(f"{c['description']}: slope {slope:.4f} mm/yr", ok)
        elif cid == "precip_variability":
            cv = 100 * np.std(precip_seasonal) / np.mean(precip_seasonal)
            ok = c["min_cv"] <= cv <= c["max_cv"]
            check_pass(f"{c['description']}: CV {cv:.1f}%", ok)
        elif cid == "decade_means_stable":
            years = np.array([s["year"] for s in seasons])
            decade_means = []
            for d in range(1960, 2024, 10):
                mask = (years >= d) & (years < d + 10)
                if np.sum(mask) > 0:
                    decade_means.append(np.mean(et0_seasonal[mask]))
            decade_cv = 100 * np.std(decade_means) / np.mean(decade_means)
            ok = decade_cv <= c["max_decade_cv"]
            check_pass(f"{c['description']}: decade CV {decade_cv:.1f}%", ok)

    # Cross-validation
    cv_checks = checks["cross_validation"]["checks"]
    for c in cv_checks:
        cid = c["id"]
        if cid == "et_precip_ratio":
            ratio = np.mean(et_seasonal) / np.mean(precip_seasonal)
            ok = c["min"] <= ratio <= c["max"]
            check_pass(f"{c['description']}: {ratio:.2f}", ok)
        elif cid == "irrigation_need":
            irrig_needed = sum(1 for s in seasons if s["irrig_events"] > 0)
            pct = 100 * irrig_needed / n
            ok = c["min_pct"] <= pct <= c["max_pct"]
            check_pass(f"{c['description']}: {pct:.1f}% of seasons", ok)

    return passed, failed


def print_decade_table(seasons: list[dict]) -> None:
    """Print decade-averaged summary table."""
    if not seasons:
        return
    years = np.array([s["year"] for s in seasons])
    et0 = np.array([s["total_et0"] for s in seasons])
    precip = np.array([s["total_precip"] for s in seasons])
    et = np.array([s["total_et"] for s in seasons])
    irrig = np.array([s["total_irrig"] for s in seasons])
    dp = np.array([s["total_dp"] for s in seasons])

    print("\n  Decade-averaged results (mm, growing season):")
    print("  " + "-" * 70)
    print(f"  {'Decade':<12} {'ET₀':>8} {'Precip':>8} {'ETc':>8} {'Irrig':>8} {'DP':>8}")
    print("  " + "-" * 70)
    for d in range(1960, 2024, 10):
        mask = (years >= d) & (years < d + 10)
        if np.sum(mask) == 0:
            continue
        print(f"  {d}-{d+9:<6} {np.mean(et0[mask]):>8.0f} {np.mean(precip[mask]):>8.0f} "
              f"{np.mean(et[mask]):>8.0f} {np.mean(irrig[mask]):>8.0f} {np.mean(dp[mask]):>8.0f}")
    print("  " + "-" * 70)


def main() -> int:
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    print("=" * 70)
    print("airSpring Exp 015: OSU Triplett-Van Doren 60-Year Water Balance")
    print("  Wooster, OH | May 1 - Sep 30 | Corn on Wooster silt loam")
    print("=" * 70)

    seasons, used_real = run_long_term_simulation(benchmark)
    if not used_real:
        print("\n  *** SYNTHETIC DATA MODE — API unreachable ***")

    print(f"\n  Seasons processed: {len(seasons)}")
    if seasons:
        print_decade_table(seasons)

    print("\n=== Validation Checks ===")
    passed, failed = validate_checks(benchmark, seasons, used_real)

    total = passed + failed
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} PASS, {failed}/{total} FAIL")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
