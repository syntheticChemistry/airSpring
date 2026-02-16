#!/usr/bin/env python3
"""
Run FAO-56 water balance simulation on REAL Michigan weather data.

This connects:
  - Real ET₀ (from compute_et0_real_data.py output)
  - Real precipitation (from Open-Meteo archive)
  - Validated water balance model (fao56_water_balance.py)

Simulates irrigation scheduling for blueberry, tomato, and corn
at their actual Michigan demonstration sites from Dong et al. (2024).

Usage:
    python control/water_balance/simulate_real_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fao56_water_balance import (
    simulate_season,
    mass_balance_check,
    total_available_water,
    readily_available_water,
)

ET0_DIR = Path(__file__).parent.parent.parent / "data" / "et0_results"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "water_balance_results"

# Crop and soil parameters from Dong et al. (2024) and FAO-56
SCENARIOS = {
    "blueberry_west_olive": {
        "station": "west_olive",
        "crop": "Blueberry",
        "Kc": 0.85,
        "theta_fc": 0.30,
        "theta_wp": 0.12,
        "root_depth_m": 0.40,
        "p": 0.50,
        "soil_type": "Sandy loam",
        "irrig_depth_mm": 15.0,
    },
    "tomato_hart": {
        "station": "hart",
        "crop": "Tomato",
        "Kc": 1.05,
        "theta_fc": 0.36,
        "theta_wp": 0.15,
        "root_depth_m": 0.60,
        "p": 0.40,
        "soil_type": "Loam",
        "irrig_depth_mm": 25.0,
    },
    "corn_manchester": {
        "station": "manchester",
        "crop": "Corn",
        "Kc": 1.15,
        "theta_fc": 0.33,
        "theta_wp": 0.13,
        "root_depth_m": 0.80,
        "p": 0.55,
        "soil_type": "Loamy sand",
        "irrig_depth_mm": 30.0,
    },
    "reference_east_lansing": {
        "station": "east_lansing",
        "crop": "Reference grass",
        "Kc": 1.00,
        "theta_fc": 0.32,
        "theta_wp": 0.14,
        "root_depth_m": 0.50,
        "p": 0.50,
        "soil_type": "Sandy loam",
        "irrig_depth_mm": 25.0,
    },
}


def load_et0_data(station: str) -> pd.DataFrame:
    """Load computed ET₀ results for a station."""
    candidates = sorted(ET0_DIR.glob(f"et0_{station}*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  airSpring — Water Balance on Real Michigan Data")
    print("=" * 70)
    print("  Model: FAO-56 Chapter 8 daily soil water balance")
    print("  Data: Real ET₀ + precipitation (Open-Meteo 2023)")
    print("  Crops: Blueberry, Tomato, Corn, Reference grass")

    for scenario_name, params in SCENARIOS.items():
        print(f"\n{'─' * 70}")
        print(f"  {params['crop']} @ {params['station']}")
        print(f"  Soil: {params['soil_type']}, "
              f"Kc={params['Kc']}, p={params['p']}")

        df = load_et0_data(params["station"])
        if df is None:
            print(f"  SKIP: No ET₀ data for {params['station']}")
            print(f"  Run: python control/fao56/compute_et0_real_data.py --all-stations")
            continue

        et0 = df["et0_ours_mm"].values
        precip = df["precip_mm"].values
        dates = pd.to_datetime(df["date"])

        # Replace NaN ET0 with 0 (missing days)
        et0 = np.nan_to_num(et0, nan=0.0)
        precip = np.nan_to_num(precip, nan=0.0)

        TAW = total_available_water(params["theta_fc"], params["theta_wp"],
                                     params["root_depth_m"])
        RAW = readily_available_water(TAW, params["p"])
        print(f"  TAW: {TAW:.1f} mm, RAW: {RAW:.1f} mm")

        # Run without irrigation (rain-fed)
        rainfed = simulate_season(
            et0, precip, params["Kc"],
            params["theta_fc"], params["theta_wp"],
            params["root_depth_m"], params["p"],
            irrigation_trigger=False)

        mb_rainfed = mass_balance_check(rainfed)

        # Run with smart irrigation
        irrigated = simulate_season(
            et0, precip, params["Kc"],
            params["theta_fc"], params["theta_wp"],
            params["root_depth_m"], params["p"],
            irrigation_trigger=True,
            irrig_depth_mm=params["irrig_depth_mm"])

        mb_irrigated = mass_balance_check(irrigated)

        # Results
        print(f"\n  Rain-fed scenario:")
        print(f"    Total ET:     {rainfed['total_et']:.1f} mm")
        print(f"    Total precip: {rainfed['total_precip']:.1f} mm")
        print(f"    Days stressed (Ks<1): "
              f"{np.sum(rainfed['Ks'] < 1.0)}/{len(et0)}")
        print(f"    Min Ks:       {rainfed['Ks'].min():.3f}")
        print(f"    Mass balance:  {mb_rainfed:.4f} mm")

        print(f"\n  Irrigated scenario:")
        print(f"    Total ET:     {irrigated['total_et']:.1f} mm")
        print(f"    Total irrig:  {irrigated['total_irrig']:.1f} mm "
              f"({irrigated['irrig_events']} events)")
        print(f"    Total precip: {irrigated['total_precip']:.1f} mm")
        print(f"    Days stressed (Ks<1): "
              f"{np.sum(irrigated['Ks'] < 1.0)}/{len(et0)}")
        print(f"    Mass balance:  {mb_irrigated:.4f} mm")

        # Water savings: irrigated total water vs naive schedule
        # Naive: irrigate 25mm every 5 days regardless
        naive_irrig = (len(et0) // 5) * 25.0
        smart_irrig = irrigated['total_irrig']
        if naive_irrig > 0:
            savings = (naive_irrig - smart_irrig) / naive_irrig * 100
            print(f"\n  Water savings vs naive (25mm/5d):")
            print(f"    Naive total:  {naive_irrig:.0f} mm")
            print(f"    Smart total:  {smart_irrig:.0f} mm")
            print(f"    Savings:      {savings:.1f}%")

        # Save daily results
        result_df = pd.DataFrame({
            "date": dates,
            "et0_mm": et0,
            "precip_mm": precip,
            "Dr_rainfed_mm": rainfed["Dr"],
            "Ks_rainfed": rainfed["Ks"],
            "ETc_rainfed_mm": rainfed["ETc"],
            "Dr_irrigated_mm": irrigated["Dr"],
            "Ks_irrigated": irrigated["Ks"],
            "ETc_irrigated_mm": irrigated["ETc"],
            "irrigation_mm": irrigated["I"],
        })
        out_path = OUTPUT_DIR / f"wb_{scenario_name}_2023.csv"
        result_df.to_csv(out_path, index=False)
        print(f"\n  Output: {out_path}")

    print(f"\n{'=' * 70}")
    print("  All simulations use REAL weather data — zero synthetic.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    sys.exit(main())
