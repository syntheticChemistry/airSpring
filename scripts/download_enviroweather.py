# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Download real weather data for Michigan agricultural stations.

Data sources (in priority order):
  1. OpenWeatherMap API — real weather, have key in testing-secrets/
  2. MSU Enviro-weather — real Michigan ag stations (JavaScript-only site)
  3. Synthetic fallback — LAST RESORT, clearly flagged

The paper data (digitized in benchmark_*.json) validates our open data
is correct. The open data allows larger sweeps than paper data alone.

Usage:
    python scripts/download_enviroweather.py [--station STATION] [--days N]
    python scripts/download_enviroweather.py --all-stations

Output:
    data/enviroweather/<station>_realtime.csv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Michigan stations relevant to Dr. Dong's research
STATIONS = {
    "east_lansing": {
        "name": "East Lansing",
        "lat": 42.727,
        "lon": -84.474,
        "description": "MSU campus — primary reference station",
    },
    "grand_junction": {
        "name": "Grand Junction",
        "lat": 42.375,
        "lon": -86.060,
        "description": "Southwest MI — blueberry region",
    },
    "sparta": {
        "name": "Sparta",
        "lat": 43.160,
        "lon": -85.710,
        "description": "West MI — fruit region",
    },
    "hart": {
        "name": "Hart",
        "lat": 43.698,
        "lon": -86.364,
        "description": "Tomato demonstration site (Dong 2024)",
    },
    "west_olive": {
        "name": "West Olive",
        "lat": 42.917,
        "lon": -86.167,
        "description": "Blueberry demonstration site (Dong 2024)",
    },
    "manchester": {
        "name": "Manchester",
        "lat": 42.153,
        "lon": -84.037,
        "description": "Corn demonstration site (Dong 2024)",
    },
}

SECRETS_PATH = Path(__file__).parent.parent.parent / "testing-secrets" / "api-keys.toml"


def load_openweather_key() -> str:
    """Load OpenWeatherMap API key from testing-secrets/."""
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH) as f:
            for line in f:
                if "openweather_api_key" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"')

    env_key = os.environ.get("OPENWEATHER_API_KEY")
    if env_key:
        return env_key

    return ""


def fetch_openweather_current(lat: float, lon: float,
                                api_key: str) -> dict:
    """Fetch current weather from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_openweather_5day(lat: float, lon: float,
                            api_key: str) -> dict:
    """Fetch 5-day / 3-hour forecast from OpenWeatherMap (free tier)."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def parse_openweather_to_daily(current: dict, forecast: dict,
                                station_name: str) -> pd.DataFrame:
    """
    Convert OpenWeatherMap responses to daily weather records
    matching the format needed for ET0 calculations.
    """
    rows = []

    # Current observation
    now = datetime.utcfromtimestamp(current["dt"])
    rows.append({
        "date": now.strftime("%Y-%m-%d"),
        "source": "openweather_current",
        "temp_c": current["main"]["temp"],
        "tmax_c": current["main"]["temp_max"],
        "tmin_c": current["main"]["temp_min"],
        "rh_pct": current["main"]["humidity"],
        "wind_m_s": current["wind"]["speed"],
        "pressure_hpa": current["main"]["pressure"],
        "clouds_pct": current["clouds"]["all"],
        "precip_mm": current.get("rain", {}).get("1h", 0.0),
        "description": current["weather"][0]["description"],
    })

    # 5-day forecast: aggregate 3-hour slots into daily
    if "list" in forecast:
        daily = {}
        for slot in forecast["list"]:
            dt = datetime.utcfromtimestamp(slot["dt"])
            day_key = dt.strftime("%Y-%m-%d")

            if day_key not in daily:
                daily[day_key] = {
                    "temps": [], "tmaxs": [], "tmins": [],
                    "rhs": [], "winds": [], "pressures": [],
                    "clouds": [], "precip": 0.0,
                }

            d = daily[day_key]
            d["temps"].append(slot["main"]["temp"])
            d["tmaxs"].append(slot["main"]["temp_max"])
            d["tmins"].append(slot["main"]["temp_min"])
            d["rhs"].append(slot["main"]["humidity"])
            d["winds"].append(slot["wind"]["speed"])
            d["pressures"].append(slot["main"]["pressure"])
            d["clouds"].append(slot["clouds"]["all"])
            d["precip"] += slot.get("rain", {}).get("3h", 0.0)

        for day_key in sorted(daily.keys()):
            d = daily[day_key]
            rows.append({
                "date": day_key,
                "source": "openweather_forecast",
                "temp_c": np.mean(d["temps"]),
                "tmax_c": max(d["tmaxs"]),
                "tmin_c": min(d["tmins"]),
                "rh_pct": np.mean(d["rhs"]),
                "wind_m_s": np.mean(d["winds"]),
                "pressure_hpa": np.mean(d["pressures"]),
                "clouds_pct": np.mean(d["clouds"]),
                "precip_mm": d["precip"],
                "description": "",
            })

    df = pd.DataFrame(rows)
    df["station"] = station_name
    return df


def generate_synthetic_fallback(station: str, n_days: int = 153,
                                 rng=None) -> pd.DataFrame:
    """
    FALLBACK ONLY: Generate synthetic data when API is unavailable.
    Clearly flagged as synthetic in the 'source' column.
    """
    if rng is None:
        rng = np.random.default_rng(42 + hash(station) % 1000)

    dates = pd.date_range(datetime.now() - timedelta(days=n_days),
                           periods=n_days, freq="D")
    doy = dates.dayofyear.values
    n = len(dates)

    t_mean = 8.0 + 15.0 * np.sin(2 * np.pi * (doy - 100) / 365)
    t_range = 10.0 + 2.0 * rng.standard_normal(n)
    t_range = np.maximum(t_range, 3.0)
    tmax = t_mean + t_range / 2 + rng.normal(0, 2, n)
    tmin = t_mean - t_range / 2 + rng.normal(0, 2, n)
    tmin = np.minimum(tmin, tmax - 2.0)

    rh = 65 + 10 * rng.standard_normal(n)
    rh = np.clip(rh, 30, 95)
    wind = 3.0 + 1.5 * rng.standard_normal(n)
    wind = np.maximum(wind, 0.5)

    rain_mask = rng.random(n) < 0.30
    precip = np.zeros(n)
    precip[rain_mask] = rng.exponential(8.0, np.sum(rain_mask))

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "source": "SYNTHETIC_FALLBACK",
        "station": station,
        "temp_c": np.round((tmax + tmin) / 2, 1),
        "tmax_c": np.round(tmax, 1),
        "tmin_c": np.round(tmin, 1),
        "rh_pct": np.round(rh, 1),
        "wind_m_s": np.round(wind, 2),
        "pressure_hpa": 1013.0,
        "clouds_pct": np.round(rng.uniform(10, 90, n), 0),
        "precip_mm": np.round(precip, 1),
        "description": "synthetic",
    })


def main():
    parser = argparse.ArgumentParser(
        description="Download Michigan weather data for airSpring experiments")
    parser.add_argument("--station", default="east_lansing",
                        choices=list(STATIONS.keys()),
                        help="Weather station")
    parser.add_argument("--all-stations", action="store_true",
                        help="Download for all stations")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic fallback (not recommended)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "data" / "enviroweather"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_openweather_key()
    stations_to_fetch = list(STATIONS.keys()) if args.all_stations \
        else [args.station]

    print("=" * 65)
    print("  airSpring — Michigan Weather Data Download")
    print("=" * 65)

    if api_key and not args.synthetic:
        print(f"  API key: OpenWeatherMap (from testing-secrets/)")
        print(f"  Data: REAL weather observations + 5-day forecast")
    else:
        if not api_key:
            print("  WARNING: No OpenWeatherMap API key found")
            print("  Check testing-secrets/api-keys.toml or set OPENWEATHER_API_KEY")
        print("  Data: SYNTHETIC FALLBACK (not real observations)")

    all_dfs = []

    for station_id in stations_to_fetch:
        info = STATIONS[station_id]
        print(f"\n--- {info['name']} ({station_id}) ---")
        print(f"  Lat: {info['lat']}, Lon: {info['lon']}")

        if api_key and not args.synthetic:
            try:
                print("  Fetching current weather...")
                current = fetch_openweather_current(
                    info["lat"], info["lon"], api_key)
                print(f"  Current: {current['main']['temp']:.1f}°C, "
                      f"{current['weather'][0]['description']}")

                print("  Fetching 5-day forecast...")
                forecast = fetch_openweather_5day(
                    info["lat"], info["lon"], api_key)

                df = parse_openweather_to_daily(current, forecast, station_id)
                print(f"  Got {len(df)} daily records (REAL DATA)")

                # Rate limit: 60 calls/min on free tier
                time.sleep(1.5)

            except Exception as e:
                print(f"  API error: {e}")
                print("  Falling back to synthetic data")
                df = generate_synthetic_fallback(station_id)
        else:
            df = generate_synthetic_fallback(station_id)

        out_path = out_dir / f"{station_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        all_dfs.append(df)

    # Combine all stations
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = out_dir / "all_stations.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined: {combined_path} ({len(combined)} records)")

    # Save station metadata
    meta_path = out_dir / "stations.json"
    with open(meta_path, "w") as f:
        json.dump(STATIONS, f, indent=2)

    # Report data source
    sources = set()
    for df in all_dfs:
        sources.update(df["source"].unique())

    print(f"\n  Data sources used: {', '.join(sorted(sources))}")
    if "SYNTHETIC_FALLBACK" in sources:
        print("  *** WARNING: Some data is synthetic. Get API key for real data. ***")


if __name__ == "__main__":
    main()
