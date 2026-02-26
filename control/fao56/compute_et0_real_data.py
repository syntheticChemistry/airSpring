# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Compute FAO-56 Penman-Monteith ET₀ on REAL Michigan weather data.

This script:
  1. Loads real Open-Meteo historical data from data/open_meteo/
  2. Runs our validated penman_monteith.py implementation on each day
  3. Cross-checks our ET₀ against Open-Meteo's own ET₀ values
  4. Outputs the results with per-day comparisons

The penman_monteith.py was validated against FAO-56 paper examples
(the "benchmark/scanned data"). Now we compute on open data.

Usage:
    python control/fao56/compute_et0_real_data.py
    python control/fao56/compute_et0_real_data.py --station east_lansing
    python control/fao56/compute_et0_real_data.py --all-stations

Provenance:
  Baseline commit: cb59873
  Created: 2026-02-26
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import our validated FAO-56 functions
sys.path.insert(0, str(Path(__file__).parent))
from penman_monteith import (
    atmospheric_pressure,
    clear_sky_radiation,
    extraterrestrial_radiation,
    fao56_penman_monteith,
    mean_saturation_vapour_pressure,
    actual_vapour_pressure_rh,
    net_longwave_radiation,
    net_shortwave_radiation,
    psychrometric_constant,
    saturation_vapour_pressure,
    slope_vapour_pressure_curve,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "open_meteo"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "et0_results"


def compute_daily_et0(row: pd.Series, altitude_m: float,
                       latitude_deg: float) -> dict:
    """
    Compute ET₀ for a single day using FAO-56 Penman-Monteith.

    Inputs from Open-Meteo daily data:
      tmax_c, tmin_c, rh_max_pct, rh_min_pct,
      wind_2m_m_s, solar_rad_mj_m2
    """
    tmax = row["tmax_c"]
    tmin = row["tmin_c"]

    if pd.isna(tmax) or pd.isna(tmin):
        return {"et0_ours_mm": np.nan, "method": "skipped_missing_temp"}

    tmean = (tmax + tmin) / 2.0

    # Date → day of year
    date = pd.Timestamp(row["date"])
    doy = date.dayofyear

    # Atmospheric parameters
    P = atmospheric_pressure(altitude_m)
    gamma = psychrometric_constant(P)
    delta = slope_vapour_pressure_curve(tmean)
    es = mean_saturation_vapour_pressure(tmax, tmin)

    # Actual vapour pressure
    rh_max = row.get("rh_max_pct")
    rh_min = row.get("rh_min_pct")
    if pd.notna(rh_max) and pd.notna(rh_min) and rh_max > 0 and rh_min > 0:
        ea = actual_vapour_pressure_rh(tmax, tmin, rh_max, rh_min)
        ea_method = "rh_max_min"
    else:
        # Fallback: estimate from Tmin (FAO-56 Eq. 48)
        ea = saturation_vapour_pressure(tmin)
        ea_method = "tmin_estimate"

    vpd = es - ea
    if vpd < 0:
        vpd = 0.0

    # Wind speed at 2m (Open-Meteo data has 10m wind, we already converted)
    u2 = row.get("wind_2m_m_s")
    if pd.isna(u2) or u2 <= 0:
        u2 = 2.0  # FAO-56 default when missing
        wind_method = "default_2m_s"
    else:
        wind_method = "from_10m_corrected"

    # Radiation
    Ra = extraterrestrial_radiation(latitude_deg, doy)
    Rso = clear_sky_radiation(altitude_m, Ra)

    Rs = row.get("solar_rad_mj_m2")
    if pd.isna(Rs) or Rs <= 0:
        # Estimate from temperature range (Hargreaves)
        krs = 0.16  # inland
        Rs = krs * math.sqrt(max(tmax - tmin, 0.1)) * Ra
        rs_method = "hargreaves_temp"
    else:
        rs_method = "measured"

    # Clamp Rs/Rso ratio
    if Rso > 0:
        rs_rso = min(Rs / Rso, 1.0)
    else:
        rs_rso = 0.75

    Rns = net_shortwave_radiation(Rs)
    Rnl = net_longwave_radiation(tmax, tmin, ea, rs_rso)
    Rn = Rns - Rnl
    G = 0.0  # daily time step

    et0 = fao56_penman_monteith(Rn, G, tmean, u2, vpd, delta, gamma)

    return {
        "et0_ours_mm": round(et0, 2),
        "ea_method": ea_method,
        "wind_method": wind_method,
        "rs_method": rs_method,
        "rn_mj_m2": round(Rn, 2),
        "vpd_kpa": round(vpd, 3),
    }


def process_station(station_id: str, csv_path: Path) -> pd.DataFrame:
    """Load real data and compute ET₀ for every day."""
    df = pd.read_csv(csv_path)

    # Get station metadata
    altitude_m = df["elevation_m"].iloc[0] if "elevation_m" in df.columns else 250
    latitude_deg = df["lat"].iloc[0] if "lat" in df.columns else 42.7

    print(f"\n  Station: {station_id}")
    print(f"  Altitude: {altitude_m}m, Latitude: {latitude_deg}°N")
    print(f"  Records: {len(df)} days")

    results = []
    for _, row in df.iterrows():
        r = compute_daily_et0(row, altitude_m, latitude_deg)
        results.append(r)

    result_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    # Comparison with Open-Meteo's ET₀
    if "et0_openmeteo_mm" in df.columns:
        mask = df["et0_ours_mm"].notna() & df["et0_openmeteo_mm"].notna()
        ours = df.loc[mask, "et0_ours_mm"].values
        theirs = df.loc[mask, "et0_openmeteo_mm"].values

        if len(ours) > 0:
            diff = ours - theirs
            rmse = np.sqrt(np.mean(diff ** 2))
            mbe = np.mean(diff)
            r2 = np.corrcoef(ours, theirs)[0, 1] ** 2

            print(f"\n  Cross-check vs Open-Meteo ET₀:")
            print(f"    RMSE: {rmse:.3f} mm/day")
            print(f"    MBE:  {mbe:+.3f} mm/day")
            print(f"    R²:   {r2:.4f}")
            print(f"    Our total:         {ours.sum():.1f} mm")
            print(f"    Open-Meteo total:  {theirs.sum():.1f} mm")
            print(f"    Days compared:     {len(ours)}")

            # Percentile analysis
            abs_diff = np.abs(diff)
            print(f"    |diff| p50: {np.percentile(abs_diff, 50):.3f} mm/day")
            print(f"    |diff| p95: {np.percentile(abs_diff, 95):.3f} mm/day")
            print(f"    |diff| max: {abs_diff.max():.3f} mm/day")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute FAO-56 ET₀ on real Michigan weather data")
    parser.add_argument("--station", default="east_lansing",
                        help="Station to process")
    parser.add_argument("--all-stations", action="store_true",
                        help="Process all stations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find data files
    if not DATA_DIR.exists():
        print(f"ERROR: No data directory at {DATA_DIR}")
        print("Run: python scripts/download_open_meteo.py --all-stations --growing-season 2023")
        return 1

    csv_files = sorted(DATA_DIR.glob("*_daily.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("all_")]

    if not csv_files:
        print(f"ERROR: No daily CSV files in {DATA_DIR}")
        return 1

    if args.all_stations:
        to_process = csv_files
    else:
        to_process = [f for f in csv_files if args.station in f.name]
        if not to_process:
            print(f"No data file found for station '{args.station}'")
            print(f"Available: {[f.stem for f in csv_files]}")
            return 1

    print("=" * 70)
    print("  airSpring — FAO-56 ET₀ on Real Michigan Data")
    print("=" * 70)
    print(f"  Implementation: penman_monteith.py (validated against FAO-56 paper)")
    print(f"  Data source: Open-Meteo Archive (REAL observations)")
    print(f"  Cross-check: Our ET₀ vs Open-Meteo's ET₀")

    all_results = []

    for csv_path in to_process:
        station_id = csv_path.stem.split("_")[0]
        df = process_station(station_id, csv_path)
        all_results.append(df)

        out_path = OUTPUT_DIR / f"et0_{csv_path.name}"
        df.to_csv(out_path, index=False)
        print(f"  Output: {out_path}")

    # Overall summary
    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = OUTPUT_DIR / "et0_all_stations.csv"
        combined.to_csv(combined_path, index=False)

        mask = combined["et0_ours_mm"].notna() & combined["et0_openmeteo_mm"].notna()
        ours = combined.loc[mask, "et0_ours_mm"].values
        theirs = combined.loc[mask, "et0_openmeteo_mm"].values
        diff = ours - theirs

        print(f"\n{'=' * 70}")
        print(f"  OVERALL CROSS-CHECK ({len(ours)} station-days)")
        print(f"{'=' * 70}")
        print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.3f} mm/day")
        print(f"  MBE:  {np.mean(diff):+.3f} mm/day")
        print(f"  R²:   {np.corrcoef(ours, theirs)[0, 1]**2:.4f}")
        print(f"  Combined: {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
