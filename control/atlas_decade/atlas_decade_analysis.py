#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 059: Atlas 80-Year Decade Analysis — Python control baseline.

Analyses ET₀ and water balance trends by decade across Michigan stations.
Uses the same validated FAO-56 pipeline from Exp 018 but aggregates results
into decade bins (1950s, 1960s, ..., 2020s) to detect climate signal.

Key metrics per station per decade:
  - Mean seasonal ET₀ (mm)
  - Mean seasonal precipitation (mm)
  - Mean water deficit (ET₀ - precip, mm)
  - Decade-over-decade ET₀ trend (mm/decade via linear regression)

Uses Open-Meteo ERA5 data from data/open_meteo/ (80yr downloads).

Usage:
    python control/atlas_decade/atlas_decade_analysis.py

Output:
    control/atlas_decade/benchmark_atlas_decade.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "fao56"))
from penman_monteith import (
    actual_vapour_pressure_rh,
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

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "open_meteo"
OUT_DIR = Path(__file__).parent

DECADES = [
    (1950, 1959),
    (1960, 1969),
    (1970, 1979),
    (1980, 1989),
    (1990, 1999),
    (2000, 2009),
    (2010, 2019),
    (2020, 2024),
]

SEASON_START_DOY = 121  # May 1
SEASON_END_DOY = 273    # Sep 30

CORE_STATIONS = [
    "east_lansing",
    "grand_junction",
    "sparta",
    "hart",
    "west_olive",
    "manchester",
]

STATION_ELEVATIONS = {
    "east_lansing": 256.0,
    "grand_junction": 197.0,
    "sparta": 262.0,
    "hart": 244.0,
    "west_olive": 192.0,
    "manchester": 290.0,
}

STATION_LATS = {
    "east_lansing": 42.727,
    "grand_junction": 42.375,
    "sparta": 43.160,
    "hart": 43.698,
    "west_olive": 42.917,
    "manchester": 42.153,
}


def compute_et0_day(row, lat_deg, elevation_m):
    """Compute FAO-56 PM ET₀ for a single day."""
    doy = row["doy"]
    tmax = row["tmax_c"]
    tmin = row["tmin_c"]
    rh_max = row.get("rh_max_pct", 70.0)
    rh_min = row.get("rh_min_pct", 40.0)
    wind_2m = row.get("wind_2m_m_s", 2.0)
    rs_mj = row.get("solar_rad_mj_m2", 15.0)

    if any(math.isnan(x) for x in [tmax, tmin]):
        return float("nan")

    rh_max = min(max(rh_max if not math.isnan(rh_max) else 70.0, 20.0), 100.0)
    rh_min = min(max(rh_min if not math.isnan(rh_min) else 40.0, 10.0), 100.0)
    wind_2m = max(wind_2m if not math.isnan(wind_2m) else 2.0, 0.5)
    rs_mj = max(rs_mj if not math.isnan(rs_mj) else 15.0, 0.1)

    tmean = (tmax + tmin) / 2.0
    P = atmospheric_pressure(elevation_m)
    gamma = psychrometric_constant(P)
    delta = vapour_pressure_slope(tmean)
    es = mean_saturation_vapour_pressure(tmax, tmin)
    ea = actual_vapour_pressure_rh(tmax, tmin, rh_max, rh_min)

    ra = extraterrestrial_radiation(lat_deg, doy)
    rso = clear_sky_radiation(elevation_m, ra)
    rns = net_shortwave_radiation(rs_mj)
    rs_over_rso = rs_mj / max(rso, 0.1)
    rnl = net_longwave_radiation(tmin, tmax, ea, rs_over_rso)
    rn = rns - rnl

    vpd = es - ea
    G = 0.0  # FAO-56 daily time step
    et0 = fao56_penman_monteith(rn, G, tmean, wind_2m, vpd, delta, gamma)
    return max(et0, 0.0)


def process_station(station_id, df):
    """Process a single station's 80yr data into decade summaries."""
    lat_deg = STATION_LATS[station_id]
    elev = STATION_ELEVATIONS[station_id]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear

    season = df[(df["doy"] >= SEASON_START_DOY) & (df["doy"] <= SEASON_END_DOY)].copy()
    season["et0_mm"] = season.apply(lambda r: compute_et0_day(r, lat_deg, elev), axis=1)

    decade_results = []
    for dec_start, dec_end in DECADES:
        dec = season[(season["year"] >= dec_start) & (season["year"] <= dec_end)]
        if dec.empty:
            continue

        yearly_et0 = dec.groupby("year")["et0_mm"].sum()
        yearly_precip = dec.groupby("year")["precip_mm"].sum()
        yearly_deficit = yearly_et0 - yearly_precip

        n_years = len(yearly_et0)
        if n_years < 2:
            continue

        decade_results.append({
            "decade_label": f"{dec_start}s",
            "decade_start": dec_start,
            "decade_end": dec_end,
            "n_years": n_years,
            "mean_seasonal_et0_mm": float(yearly_et0.mean()),
            "std_seasonal_et0_mm": float(yearly_et0.std()),
            "mean_seasonal_precip_mm": float(yearly_precip.mean()),
            "mean_water_deficit_mm": float(yearly_deficit.mean()),
            "min_et0_year": int(yearly_et0.idxmin()),
            "max_et0_year": int(yearly_et0.idxmax()),
        })

    # Compute trend: linear regression of decade-mean ET₀ vs decade midpoint
    if len(decade_results) >= 3:
        x = np.array([d["decade_start"] + 5 for d in decade_results], dtype=float)
        y = np.array([d["mean_seasonal_et0_mm"] for d in decade_results], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend_mm_per_decade = slope * 10.0
    else:
        trend_mm_per_decade = 0.0

    return {
        "station_id": station_id,
        "lat": STATION_LATS[station_id],
        "elevation_m": elev,
        "decades": decade_results,
        "trend_mm_per_decade": float(trend_mm_per_decade),
    }


def main():
    results = []
    for sid in CORE_STATIONS:
        csv_path = DATA_DIR / f"{sid}_1945-01-01_2024-12-31_daily.csv"
        if not csv_path.exists():
            csv_path = DATA_DIR / f"{sid}_2023-05-01_2023-09-30_daily.csv"
            if not csv_path.exists():
                print(f"  SKIP {sid}: no data file")
                continue
            print(f"  {sid}: using single-season data (limited decades)")

        print(f"  Processing {sid}...")
        df = pd.read_csv(csv_path)
        station_result = process_station(sid, df)
        results.append(station_result)
        print(f"    {len(station_result['decades'])} decades, "
              f"trend={station_result['trend_mm_per_decade']:.2f} mm/decade")

    benchmark = {
        "experiment": "Exp 059: Atlas 80yr Decade Analysis",
        "provenance": "Open-Meteo ERA5 + FAO-56 PM ET₀",
        "season": "May 1 – Sep 30 (DOY 121–273)",
        "n_stations": len(results),
        "stations": results,
    }

    out_path = OUT_DIR / "benchmark_atlas_decade.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"  {len(results)} stations × {sum(len(s['decades']) for s in results)} decades")


if __name__ == "__main__":
    main()
