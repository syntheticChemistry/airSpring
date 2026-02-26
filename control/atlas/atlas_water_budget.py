# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Experiment 018: Michigan Crop Water Atlas — Python control baseline.

Runs the validated ET₀ + water balance + yield response pipeline on
Open-Meteo data for cross-validation against the Rust validate_atlas binary.

This is NOT new science. It applies the Python baselines from Exp 001
(penman_monteith.py), Exp 004 (fao56_water_balance.py), and Exp 008
(yield_response.py) at scale.

Usage:
    python control/atlas/atlas_water_budget.py
    python control/atlas/atlas_water_budget.py --station east_lansing
    python control/atlas/atlas_water_budget.py --all-stations

Output:
    control/atlas/benchmark_atlas.json  (per-station summary for cross-validation)

Provenance:
  Baseline commit: cb59873
  Created: 2026-02-26
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "fao56"))
from penman_monteith import (
    atmospheric_pressure,
    clear_sky_radiation,
    extraterrestrial_radiation,
    fao56_penman_monteith,
    mean_saturation_vapour_pressure,
    net_longwave_radiation,
    net_shortwave_radiation,
    psychrometric_constant,
    saturation_vapour_pressure,
    slope_vapour_pressure_curve as vapour_pressure_slope,
)

CROPS = {
    "Corn": {"kc_ini": 0.30, "kc_mid": 1.20, "kc_end": 0.60, "root_m": 0.90, "p": 0.55, "ky": 1.25},
    "Soybean": {"kc_ini": 0.40, "kc_mid": 1.15, "kc_end": 0.50, "root_m": 0.60, "p": 0.50, "ky": 0.85},
    "WinterWheat": {"kc_ini": 0.70, "kc_mid": 1.15, "kc_end": 0.25, "root_m": 1.50, "p": 0.55, "ky": 1.00},
    "SugarBeet": {"kc_ini": 0.35, "kc_mid": 1.20, "kc_end": 0.70, "root_m": 0.70, "p": 0.55, "ky": 1.10},
    "DryBean": {"kc_ini": 0.40, "kc_mid": 1.15, "kc_end": 0.35, "root_m": 0.60, "p": 0.45, "ky": 1.15},
    "Potato": {"kc_ini": 0.50, "kc_mid": 1.15, "kc_end": 0.75, "root_m": 0.40, "p": 0.35, "ky": 1.10},
    "Tomato": {"kc_ini": 0.60, "kc_mid": 1.15, "kc_end": 0.80, "root_m": 0.60, "p": 0.40, "ky": 1.05},
    "Blueberry": {"kc_ini": 0.30, "kc_mid": 1.05, "kc_end": 0.65, "root_m": 0.40, "p": 0.50, "ky": 0.80},
    "Alfalfa": {"kc_ini": 0.40, "kc_mid": 1.20, "kc_end": 1.15, "root_m": 1.00, "p": 0.55, "ky": 1.10},
    "Turfgrass": {"kc_ini": 0.90, "kc_mid": 0.95, "kc_end": 0.95, "root_m": 0.30, "p": 0.40, "ky": 0.80},
}

FC = 0.30
WP = 0.12
IRRIG_DEPTH = 25.0


def compute_et0_day(row):
    """Compute FAO-56 PM ET₀ for a single day of Open-Meteo data."""
    tmin = row["tmin_c"]
    tmax = row["tmax_c"]
    tmean = row.get("tmean_c", (tmin + tmax) / 2)
    rs = row["solar_rad_mj_m2"]
    u2 = row.get("wind_2m_m_s", 2.0)
    rh_min = row.get("rh_min_pct", 50)
    rh_max = row.get("rh_max_pct", 80)
    lat = row["lat"]
    elev = row["elevation_m"]
    doy = row["doy"]

    if any(math.isnan(v) for v in [tmin, tmax, rs]):
        return float("nan")

    P = atmospheric_pressure(elev)
    gamma = psychrometric_constant(P)
    delta = vapour_pressure_slope(tmean)
    es = mean_saturation_vapour_pressure(tmin, tmax)

    e_tmin = saturation_vapour_pressure(tmin)
    e_tmax = saturation_vapour_pressure(tmax)
    ea = (e_tmin * rh_max / 100 + e_tmax * rh_min / 100) / 2

    ra = extraterrestrial_radiation(lat, doy)
    rso = clear_sky_radiation(elev, ra)
    rns = net_shortwave_radiation(rs)
    rs_rso = min(rs / rso, 1.0) if rso > 0 else 0.7
    rnl = net_longwave_radiation(tmin, tmax, ea, rs_rso)
    rn = rns - rnl

    vpd = es - ea
    et0 = fao56_penman_monteith(rn, 0.0, tmean, u2, vpd, delta, gamma)
    return max(0.0, et0)


def run_water_balance_season(et0_series, precip_series, crop_info):
    """Run single-season water balance for one crop."""
    root_mm = crop_info["root_m"] * 1000
    p = crop_info["p"]
    ky = crop_info["ky"]

    taw = (FC - WP) * root_mm
    raw = p * taw

    depletion = 0.0
    total_et = 0.0
    total_irrig = 0.0
    stress_days = 0
    n = len(et0_series)

    for i in range(n):
        frac = i / n
        if frac < 0.2:
            kc = crop_info["kc_ini"]
        elif frac < 0.7:
            kc = crop_info["kc_mid"]
        else:
            kc = crop_info["kc_end"]

        etc = et0_series[i] * kc
        ks = 1.0 if depletion <= raw else max(0.0, (taw - depletion) / (taw - raw))
        actual_et = etc * ks

        irr = IRRIG_DEPTH if depletion > raw else 0.0
        total_irrig += irr

        depletion = depletion - precip_series[i] - irr + actual_et
        if depletion < 0:
            depletion = 0
        depletion = min(depletion, taw)

        total_et += actual_et
        if ks < 1.0:
            stress_days += 1

    total_etc_season = sum(
        et0_series[i] * (crop_info["kc_ini"] if i / n < 0.2 else crop_info["kc_mid"] if i / n < 0.7 else crop_info["kc_end"])
        for i in range(n)
    )
    eta_etc_ratio = total_et / total_etc_season if total_etc_season > 0 else 1.0
    yield_ratio = max(0.0, min(1.0, 1 - ky * (1 - eta_etc_ratio)))

    return {
        "total_et_mm": round(total_et, 1),
        "total_precip_mm": round(sum(precip_series), 1),
        "stress_days": stress_days,
        "yield_ratio": round(yield_ratio, 4),
        "total_irrig_mm": round(total_irrig, 1),
    }


def date_to_doy(date_str):
    """Convert YYYY-MM-DD to day of year."""
    parts = date_str.split("-")
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    doy = days_before[month - 1] + day
    is_leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    if is_leap and month > 2:
        doy += 1
    return year, doy


def process_station(csv_path):
    """Process one station CSV file through the full pipeline."""
    df = pd.read_csv(csv_path)
    required = ["tmax_c", "tmin_c", "solar_rad_mj_m2", "precip_mm", "lat", "elevation_m", "date"]
    for col in required:
        if col not in df.columns:
            return None

    years_doys = [date_to_doy(d) for d in df["date"]]
    df["year"] = [yd[0] for yd in years_doys]
    df["doy"] = [yd[1] for yd in years_doys]

    valid = df.dropna(subset=["tmax_c", "solar_rad_mj_m2"])
    if valid.empty:
        return None

    et0_all = [compute_et0_day(row) for _, row in valid.iterrows()]
    valid = valid.copy()
    valid["et0_rust_equiv"] = et0_all

    season = valid[(valid["doy"] >= 121) & (valid["doy"] <= 273)]

    results = {"n_days": len(valid), "n_years": valid["year"].nunique()}

    total_et0 = valid.groupby("year")["et0_rust_equiv"].sum().mean()
    results["mean_annual_et0"] = round(total_et0, 1)

    crop_results = {}
    for crop_name, crop_info in CROPS.items():
        yearly_results = []
        for year, ydata in season.groupby("year"):
            if len(ydata) < 60:
                continue
            et0_series = ydata["et0_rust_equiv"].tolist()
            precip_series = ydata["precip_mm"].fillna(0).clip(lower=0).tolist()
            yr = run_water_balance_season(et0_series, precip_series, crop_info)
            yearly_results.append(yr)

        if yearly_results:
            crop_results[crop_name] = {
                "n_seasons": len(yearly_results),
                "mean_et_mm": round(np.mean([r["total_et_mm"] for r in yearly_results]), 1),
                "mean_precip_mm": round(np.mean([r["total_precip_mm"] for r in yearly_results]), 1),
                "mean_stress_days": round(np.mean([r["stress_days"] for r in yearly_results]), 1),
                "mean_yield_ratio": round(np.mean([r["yield_ratio"] for r in yearly_results]), 4),
                "mean_irrig_mm": round(np.mean([r["total_irrig_mm"] for r in yearly_results]), 1),
            }

    results["crops"] = crop_results
    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 018: Michigan Crop Water Atlas (Python control)")
    parser.add_argument("--station", default=None, help="Single station CSV to process")
    parser.add_argument("--all-stations", action="store_true", help="Process all stations in data/open_meteo/")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: data/open_meteo)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent.parent.parent / "data" / "open_meteo"
    if not data_dir.exists():
        print(f"[SKIP] No data at {data_dir}")
        print("Run: python scripts/download_open_meteo.py --all-stations --growing-season 2023")
        return

    if args.station:
        csvs = list(data_dir.glob(f"{args.station}_*_daily.csv"))
    else:
        csvs = sorted(data_dir.glob("*_daily.csv"))
        csvs = [c for c in csvs if not c.name.startswith("all_stations")]

    print(f"  Processing {len(csvs)} station files from {data_dir}")

    benchmark = {"stations": {}, "provenance": {"script": __file__, "date": "2026-02-26"}}

    for csv_path in csvs:
        station_id = csv_path.stem.split("_")[0]
        if any(c.isdigit() for c in station_id):
            parts = csv_path.stem.split("_")
            station_parts = []
            for p in parts:
                if p[0].isdigit():
                    break
                station_parts.append(p)
            station_id = "_".join(station_parts)

        print(f"  {station_id}...", end="")
        result = process_station(csv_path)
        if result:
            benchmark["stations"][station_id] = result
            print(f" {result['n_days']} days, ET₀={result['mean_annual_et0']:.0f} mm/yr")
            for crop, cr in result["crops"].items():
                print(f"    {crop}: ET={cr['mean_et_mm']:.0f}mm, yield={cr['mean_yield_ratio']:.3f}, "
                      f"stress={cr['mean_stress_days']:.0f}d, irrig={cr['mean_irrig_mm']:.0f}mm")
        else:
            print(" skipped")

    out_path = Path(__file__).parent / "benchmark_atlas.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  Benchmark saved: {out_path}")
    print(f"  Stations: {len(benchmark['stations'])}")


if __name__ == "__main__":
    main()
