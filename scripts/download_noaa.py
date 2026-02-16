#!/usr/bin/env python3
"""
Download historical weather data from NOAA Climate Data Online (CDO).

NOAA CDO (https://www.ncdc.noaa.gov/cdo-web/) provides free access to
historical weather observations from US stations — years of daily data
for Michigan agricultural locations.

This is the REAL HISTORICAL DATA source for water balance simulations,
ET₀ validation against actual Michigan seasons, and large-sweep analysis.

API Key:
    Free token required — register at: https://www.ncdc.noaa.gov/cdo-web/token
    Add to testing-secrets/api-keys.toml as: noaa_cdo_token = "YOUR_TOKEN"
    Or set env: NOAA_CDO_TOKEN=YOUR_TOKEN

Usage:
    python scripts/download_noaa.py --station USW00014836 --start 2023-05-01 --end 2023-09-30
    python scripts/download_noaa.py --all-stations --start 2023-01-01 --end 2023-12-31

Output:
    data/noaa/<station>_<start>_<end>.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# NOAA GHCND stations near Dr. Dong's Michigan research sites
STATIONS = {
    "USW00014836": {
        "name": "Lansing Capital City Airport",
        "lat": 42.779,
        "lon": -84.587,
        "description": "Nearest to MSU campus / East Lansing",
    },
    "USW00094860": {
        "name": "Grand Rapids Gerald Ford Airport",
        "lat": 42.881,
        "lon": -85.523,
        "description": "West Michigan — near blueberry/fruit regions",
    },
    "USW00014840": {
        "name": "Muskegon County Airport",
        "lat": 43.170,
        "lon": -86.238,
        "description": "Near Hart, MI — tomato demonstration site",
    },
}


def generate_synthetic_noaa(station_id: str, start_date: str,
                             end_date: str, rng=None) -> pd.DataFrame:
    """
    Generate synthetic GHCND-format daily weather data representative
    of Michigan conditions for use when API token is not available.

    GHCND variables:
      TMAX (°C/10), TMIN (°C/10), PRCP (mm/10), AWND (m/s/10)
    """
    if rng is None:
        rng = np.random.default_rng(42 + hash(station_id) % 1000)

    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values

    # Temperature (sinusoidal + noise) — Michigan climate normals
    t_mean = 8.0 + 15.0 * np.sin(2 * np.pi * (doy - 100) / 365)
    t_range = 10.0 + 2.0 * rng.standard_normal(n)
    t_range = np.maximum(t_range, 3.0)

    tmax = t_mean + t_range / 2 + rng.normal(0, 3, n)
    tmin = t_mean - t_range / 2 + rng.normal(0, 3, n)
    tmin = np.minimum(tmin, tmax - 2.0)

    # Precipitation — Michigan average ~830 mm/year
    rain_prob = 0.35 - 0.10 * np.cos(2 * np.pi * (doy - 180) / 365)
    rain_days = rng.random(n) < rain_prob
    precip = np.zeros(n)
    precip[rain_days] = rng.exponential(6.5, np.sum(rain_days))

    # Wind speed (m/s) — Michigan avg ~4 m/s
    wind = 4.0 + 2.0 * rng.standard_normal(n)
    wind = np.maximum(wind, 0.5)

    df = pd.DataFrame({
        "date": dates,
        "station": station_id,
        "tmax_c": np.round(tmax, 1),
        "tmin_c": np.round(tmin, 1),
        "precip_mm": np.round(precip, 1),
        "wind_m_s": np.round(wind, 1),
    })

    return df


def download_noaa_cdo(station_id: str, start_date: str,
                       end_date: str, token: str) -> pd.DataFrame:
    """
    Download from NOAA CDO REST API.
    Requires a free token from https://www.ncdc.noaa.gov/cdo-web/token
    """
    import requests

    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token}

    all_data = []
    offset = 1

    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": f"GHCND:{station_id}",
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX,TMIN,PRCP,AWND",
            "units": "metric",
            "limit": 1000,
            "offset": offset,
        }

        resp = requests.get(base_url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"API error: {resp.status_code} — {resp.text}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        all_data.extend(results)
        offset += len(results)

        if offset > data.get("metadata", {}).get("resultset", {}).get("count", 0):
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    pivot = df.pivot_table(index="date", columns="datatype",
                            values="value", aggfunc="first")
    pivot = pivot.reset_index()
    pivot.columns.name = None
    pivot["date"] = pd.to_datetime(pivot["date"])

    rename = {"TMAX": "tmax_c", "TMIN": "tmin_c",
              "PRCP": "precip_mm", "AWND": "wind_m_s"}
    pivot = pivot.rename(columns=rename)
    pivot["station"] = station_id

    return pivot


SECRETS_PATH = Path(__file__).parent.parent.parent / "testing-secrets" / "api-keys.toml"


def load_noaa_token() -> str:
    """Load NOAA CDO token from testing-secrets/ or environment."""
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH) as f:
            for line in f:
                if "noaa_cdo_token" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"')

    return os.environ.get("NOAA_CDO_TOKEN", "")


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA CDO historical weather data for Michigan")
    parser.add_argument("--station", default="USW00014836",
                        help="GHCND station ID")
    parser.add_argument("--all-stations", action="store_true",
                        help="Download for all Michigan stations")
    parser.add_argument("--start", default="2023-05-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-09-30",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--token", default=None,
                        help="NOAA CDO API token")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Force synthetic fallback (not recommended)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "data" / "noaa"
    out_dir.mkdir(parents=True, exist_ok=True)

    station_info = STATIONS.get(args.station, {
        "name": args.station, "description": "Unknown station"})
    print(f"Station: {station_info['name']} ({args.station})")
    print(f"  Date range: {args.start} to {args.end}")

    token = args.token or load_noaa_token()

    if args.synthetic or not token:
        if not token and not args.synthetic:
            print("\n  *** NOAA CDO API token needed for REAL historical data ***")
            print("  Register (free, instant): https://www.ncdc.noaa.gov/cdo-web/token")
            print("  Then add to testing-secrets/api-keys.toml:")
            print('    noaa_cdo_token = "YOUR_TOKEN_HERE"')
            print("  Or: export NOAA_CDO_TOKEN=YOUR_TOKEN_HERE")
            print("\n  Falling back to SYNTHETIC data (not real observations).\n")
        print("Generating synthetic NOAA-format data (Michigan normals)...")
        df = generate_synthetic_noaa(args.station, args.start, args.end)
    else:
        print("\nDownloading from NOAA CDO API...")
        df = download_noaa_cdo(args.station, args.start, args.end, token)
        if df.empty:
            print("No data returned. Generating synthetic fallback.")
            df = generate_synthetic_noaa(args.station, args.start, args.end)

    safe_start = args.start.replace("-", "")
    safe_end = args.end.replace("-", "")
    out_path = out_dir / f"{args.station}_{safe_start}_{safe_end}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Records: {len(df)}")
    print(f"  Tmax range: {df['tmax_c'].min():.1f} to {df['tmax_c'].max():.1f} °C")
    print(f"  Precip total: {df['precip_mm'].sum():.1f} mm")

    meta_path = out_dir / "stations.json"
    with open(meta_path, "w") as f:
        json.dump(STATIONS, f, indent=2)


if __name__ == "__main__":
    main()
