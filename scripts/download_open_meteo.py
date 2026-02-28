# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Download REAL historical weather data from Open-Meteo API.

Open-Meteo (https://open-meteo.com/) provides:
  - 80+ years of hourly weather data
  - 10 km resolution globally
  - ALL variables needed for FAO-56 ET₀
  - FREE, open-source, NO API KEY required

This is our PRIMARY historical data source. It gives us real Michigan
weather for any date range, enabling:
  - Validation of ET₀ against real growing-season data
  - Water balance simulations with actual precip/temp
  - Large sweeps across years, stations, seasons

The paper benchmark data (benchmark_*.json) validates that our ET₀
implementation is correct; this real data is what we actually compute on.

Usage:
    python scripts/download_open_meteo.py --station east_lansing --start 2023-05-01 --end 2023-09-30
    python scripts/download_open_meteo.py --all-stations --start 2020-01-01 --end 2024-12-31
    python scripts/download_open_meteo.py --all-stations --growing-season 2023
    python scripts/download_open_meteo.py --atlas --year-range 1945-2024
    python scripts/download_open_meteo.py --atlas --growing-season 2023 --batch-size 10

Output:
    data/open_meteo/<station>_<start>_<end>_daily.csv
    data/open_meteo/<station>_<start>_<end>_hourly.csv  (if --hourly)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Michigan stations — same locations as Enviro-weather but now with
# years of REAL historical data behind them
STATIONS = {
    "east_lansing": {
        "name": "East Lansing (MSU)",
        "lat": 42.727,
        "lon": -84.474,
        "elevation_m": 256,
        "description": "MSU campus — primary reference station",
    },
    "grand_junction": {
        "name": "Grand Junction",
        "lat": 42.375,
        "lon": -86.060,
        "elevation_m": 197,
        "description": "Southwest MI — blueberry region",
    },
    "sparta": {
        "name": "Sparta",
        "lat": 43.160,
        "lon": -85.710,
        "elevation_m": 262,
        "description": "West MI — fruit region",
    },
    "hart": {
        "name": "Hart",
        "lat": 43.698,
        "lon": -86.364,
        "elevation_m": 244,
        "description": "Tomato demonstration site (Dong 2024)",
    },
    "west_olive": {
        "name": "West Olive",
        "lat": 42.917,
        "lon": -86.167,
        "elevation_m": 192,
        "description": "Blueberry demonstration site (Dong 2024)",
    },
    "manchester": {
        "name": "Manchester",
        "lat": 42.153,
        "lon": -84.037,
        "elevation_m": 290,
        "description": "Corn demonstration site (Dong 2024)",
    },
}

ATLAS_STATIONS_FILE = Path(__file__).parent / "atlas_stations.json"

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables needed for FAO-56 Penman-Monteith ET₀
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
    "windspeed_10m_max",
    "windspeed_10m_mean",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "et0_fao_evapotranspiration",  # Open-Meteo computes ET₀ too — perfect cross-check
    "sunshine_duration",
    "pressure_msl_mean",
]

# Hourly variables (for detailed analysis)
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "shortwave_radiation",
    "precipitation",
    "surface_pressure",
    "cloudcover",
]


def load_atlas_stations() -> dict:
    """Load the full 100-station atlas from atlas_stations.json."""
    if not ATLAS_STATIONS_FILE.exists():
        print(f"  ERROR: Atlas station file not found: {ATLAS_STATIONS_FILE}")
        sys.exit(1)
    with open(ATLAS_STATIONS_FILE) as f:
        data = json.load(f)
    return data["stations"]


def _fetch_with_retry(params):
    """GET with retry on 429 rate-limit."""
    for attempt in range(8):
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=120)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"    Rate limited (429), waiting {wait}s (attempt {attempt+1}/8)...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def fetch_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily weather data from Open-Meteo Archive API.

    No API key needed. Free for non-commercial use.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/Detroit",
        "windspeed_unit": "ms",
        "precipitation_unit": "mm",
    }

    data = _fetch_with_retry(params)

    if "daily" not in data:
        raise ValueError(f"No daily data returned: {data.get('reason', 'unknown')}")

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    df["source"] = "open_meteo_archive"

    # Rename columns to our standard names
    rename_map = {
        "temperature_2m_max": "tmax_c",
        "temperature_2m_min": "tmin_c",
        "temperature_2m_mean": "tmean_c",
        "relative_humidity_2m_max": "rh_max_pct",
        "relative_humidity_2m_min": "rh_min_pct",
        "relative_humidity_2m_mean": "rh_mean_pct",
        "windspeed_10m_max": "wind_10m_max_m_s",
        "windspeed_10m_mean": "wind_10m_mean_m_s",
        "shortwave_radiation_sum": "solar_rad_mj_m2",
        "precipitation_sum": "precip_mm",
        "et0_fao_evapotranspiration": "et0_openmeteo_mm",
        "sunshine_duration": "sunshine_seconds",
        "pressure_msl_mean": "pressure_msl_hpa",
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert sunshine from seconds to hours
    if "sunshine_seconds" in df.columns:
        df["sunshine_hours"] = df["sunshine_seconds"] / 3600.0
        df.drop(columns=["sunshine_seconds"], inplace=True)

    # Convert shortwave radiation from W·h/m² to MJ/m²/day
    # Open-Meteo returns daily sum in MJ/m² already when using shortwave_radiation_sum
    # (Actually it's in MJ/m², no conversion needed)

    # Wind speed at 10m → 2m correction (FAO-56 Eq. 47)
    # u2 = u10 * 4.87 / ln(67.8 * 10 - 5.42)
    if "wind_10m_mean_m_s" in df.columns:
        df["wind_2m_m_s"] = df["wind_10m_mean_m_s"] * 4.87 / np.log(67.8 * 10 - 5.42)

    return df


def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/Detroit",
        "windspeed_unit": "ms",
    }

    resp = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly data returned: {data.get('reason', 'unknown')}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["source"] = "open_meteo_archive"
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download real historical Michigan weather from Open-Meteo (free, no key)")
    parser.add_argument("--station", default="east_lansing",
                        choices=list(STATIONS.keys()),
                        help="Station location")
    parser.add_argument("--all-stations", action="store_true",
                        help="Download for all 6 core Michigan stations")
    parser.add_argument("--atlas", action="store_true",
                        help="Download for all 100 atlas stations (from atlas_stations.json)")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--growing-season", type=int, default=None,
                        help="Download May-Sep for given year")
    parser.add_argument("--full-year", type=int, default=None,
                        help="Download full year Jan-Dec")
    parser.add_argument("--year-range", default=None,
                        help="Year range for atlas (e.g. 1945-2024)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Stations per batch for atlas mode (default: 20)")
    parser.add_argument("--hourly", action="store_true",
                        help="Also download hourly data")
    args = parser.parse_args()

    # Date range resolution
    if args.year_range:
        yr_start, yr_end = args.year_range.split("-")
        start = f"{yr_start}-01-01"
        end = f"{yr_end}-12-31"
    elif args.growing_season:
        start = f"{args.growing_season}-05-01"
        end = f"{args.growing_season}-09-30"
    elif args.full_year:
        start = f"{args.full_year}-01-01"
        end = f"{args.full_year}-12-31"
    elif args.start and args.end:
        start = args.start
        end = args.end
    else:
        start = "2023-05-01"
        end = "2023-09-30"

    out_dir = Path(__file__).parent.parent / "data" / "open_meteo"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.atlas:
        atlas_stations = load_atlas_stations()
        active_stations = atlas_stations
    elif args.all_stations:
        active_stations = STATIONS
    else:
        active_stations = {args.station: STATIONS[args.station]}

    stations_to_fetch = list(active_stations.keys())

    print("=" * 65)
    print("  airSpring — Open-Meteo Historical Weather Download")
    print("=" * 65)
    print(f"  Source: Open-Meteo Archive API (REAL DATA, no key needed)")
    print(f"  Period: {start} to {end}")
    print(f"  Stations: {len(stations_to_fetch)}")
    print(f"  Variables: {len(DAILY_VARS)} daily" +
          (f" + {len(HOURLY_VARS)} hourly" if args.hourly else ""))

    all_daily = []

    for idx, station_id in enumerate(stations_to_fetch):
        info = active_stations[station_id]
        name = info.get("name", station_id)
        print(f"\n--- [{idx+1}/{len(stations_to_fetch)}] {name} ({station_id}) ---")
        print(f"  Lat: {info['lat']}, Lon: {info['lon']}, "
              f"Elev: {info['elevation_m']}m")

        daily_path = out_dir / f"{station_id}_{start}_{end}_daily.csv"
        if daily_path.exists():
            print(f"  SKIP: {daily_path.name} already exists")
            try:
                all_daily.append(pd.read_csv(daily_path))
            except Exception:
                pass
            continue

        if idx > 0:
            delay = 10.0  # Rate limit: Open-Meteo free tier (80yr requests are heavy)
            print(f"  (waiting {delay}s for rate limit...)")
            time.sleep(delay)

        try:
            # Daily data
            print(f"  Fetching daily data ({start} to {end})...")
            df_daily = fetch_daily(info["lat"], info["lon"], start, end)
            df_daily["station"] = station_id
            df_daily["lat"] = info["lat"]
            df_daily["lon"] = info["lon"]
            df_daily["elevation_m"] = info["elevation_m"]

            daily_path = out_dir / f"{station_id}_{start}_{end}_daily.csv"
            df_daily.to_csv(daily_path, index=False)

            n_days = len(df_daily)
            n_valid = df_daily["tmax_c"].notna().sum()
            print(f"  Daily: {n_days} days ({n_valid} with valid tmax)")
            print(f"  Tmax range: {df_daily['tmax_c'].min():.1f} to "
                  f"{df_daily['tmax_c'].max():.1f} °C")
            print(f"  Total precip: {df_daily['precip_mm'].sum():.1f} mm")
            if "et0_openmeteo_mm" in df_daily.columns:
                et0_total = df_daily["et0_openmeteo_mm"].sum()
                print(f"  Total ET₀ (Open-Meteo): {et0_total:.1f} mm")
            print(f"  Saved: {daily_path}")

            all_daily.append(df_daily)

            # Hourly data (optional)
            if args.hourly:
                print(f"  Fetching hourly data...")
                df_hourly = fetch_hourly(info["lat"], info["lon"], start, end)
                df_hourly["station"] = station_id
                hourly_path = out_dir / f"{station_id}_{start}_{end}_hourly.csv"
                df_hourly.to_csv(hourly_path, index=False)
                print(f"  Hourly: {len(df_hourly)} records → {hourly_path}")

            delay = 2.0 if len(stations_to_fetch) > 20 else 0.5
            time.sleep(delay)

        except Exception as e:
            print(f"  ERROR: {e}")
            print("  (Open-Meteo may limit requests; try again in a moment)")

    # Combined daily
    if len(all_daily) > 1:
        combined = pd.concat(all_daily, ignore_index=True)
        combined_path = out_dir / f"all_stations_{start}_{end}_daily.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined: {combined_path} ({len(combined)} records)")

    # Station metadata
    meta_path = out_dir / "stations.json"
    with open(meta_path, "w") as f:
        json.dump(STATIONS, f, indent=2)

    # Summary
    if all_daily:
        print(f"\n{'=' * 65}")
        print("  DATA SUMMARY")
        print(f"{'=' * 65}")
        total = sum(len(d) for d in all_daily)
        print(f"  Total records: {total} station-days")
        print(f"  Source: REAL HISTORICAL OBSERVATIONS (Open-Meteo reanalysis)")
        print(f"  Ready for: ET₀ calculation, water balance, large sweeps")
        print(f"\n  Key columns for FAO-56:")
        print(f"    tmax_c, tmin_c    — daily temperature extremes")
        print(f"    rh_max_pct, rh_min_pct — humidity for VPD")
        print(f"    wind_2m_m_s       — 2m wind (corrected from 10m)")
        print(f"    solar_rad_mj_m2   — shortwave radiation sum")
        print(f"    sunshine_hours    — for Rs estimation where needed")
        print(f"    et0_openmeteo_mm  — Open-Meteo's own ET₀ (cross-check)")
        print(f"    precip_mm         — for water balance model")


if __name__ == "__main__":
    main()
