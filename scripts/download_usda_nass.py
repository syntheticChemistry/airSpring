# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Download REAL crop yield data from USDA NASS Quick Stats API.

USDA NASS (https://quickstats.nass.usda.gov/) provides:
  - County-level crop yields for all US states
  - Annual survey data going back decades
  - FREE with instant API key registration

This data enables validation of our Stewart (1977) yield response predictions
against actual Michigan crop harvests. See whitePaper/baseCamp/yield_validation.md.

Usage:
    python scripts/download_usda_nass.py --api-key YOUR_KEY
    python scripts/download_usda_nass.py --api-key-file testing-secrets/nass_api_key.txt
    python scripts/download_usda_nass.py --api-key YOUR_KEY --crop CORN --years 2000-2023

Output:
    data/usda_nass/michigan_yields.csv
    data/usda_nass/michigan_yields.json  (structured for validation pipeline)

API Key:
    Register (free, instant) at https://quickstats.nass.usda.gov/api/
    Store key in testing-secrets/nass_api_key.txt (gitignored)

Baseline commit: cb59873
Created: 2026-02-26
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"

MICHIGAN_CROPS = {
    "CORN": {
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "ky_mid": 1.25,
        "crop_type": "Corn",
    },
    "SOYBEANS": {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "ky_mid": 0.85,
        "crop_type": "Soybean",
    },
    "WHEAT": {
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "ky_mid": 1.00,
        "crop_type": "WinterWheat",
    },
    "SUGARBEETS": {
        "commodity_desc": "SUGARBEETS",
        "statisticcat_desc": "YIELD",
        "unit_desc": "TONS / ACRE",
        "ky_mid": 1.10,
        "crop_type": "SugarBeet",
    },
    "BEANS, DRY EDIBLE": {
        "commodity_desc": "BEANS, DRY EDIBLE",
        "statisticcat_desc": "YIELD",
        "unit_desc": "CWT / ACRE",
        "ky_mid": 1.15,
        "crop_type": "DryBean",
    },
}


def download_crop_yields(api_key, crop_name, crop_info, year_start, year_end):
    """Download county-level yields for a single crop from USDA NASS."""
    params = {
        "key": api_key,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": crop_info["commodity_desc"],
        "statisticcat_desc": crop_info["statisticcat_desc"],
        "unit_desc": crop_info["unit_desc"],
        "state_alpha": "MI",
        "agg_level_desc": "COUNTY",
        "year__GE": str(year_start),
        "year__LE": str(year_end),
        "format": "JSON",
    }

    print(f"  Downloading {crop_name} yields ({year_start}-{year_end})...")
    resp = requests.get(API_BASE, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if "data" not in data:
        print(f"    Warning: no data returned for {crop_name}")
        return []

    records = []
    for row in data["data"]:
        value_str = row.get("Value", "").replace(",", "").strip()
        if value_str in ("", "(D)", "(NA)", "(S)", "(Z)"):
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue

        records.append({
            "crop": crop_name,
            "crop_type": crop_info["crop_type"],
            "ky_mid": crop_info["ky_mid"],
            "year": int(row["year"]),
            "county": row.get("county_name", "").title(),
            "county_code": row.get("county_code", ""),
            "state": "MI",
            "yield_value": value,
            "unit": crop_info["unit_desc"],
        })

    print(f"    Got {len(records)} county-year records for {crop_name}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Download Michigan crop yields from USDA NASS Quick Stats"
    )
    parser.add_argument("--api-key", help="NASS API key")
    parser.add_argument(
        "--api-key-file",
        default="testing-secrets/nass_api_key.txt",
        help="File containing NASS API key",
    )
    parser.add_argument(
        "--crop",
        help="Single crop to download (default: all Michigan crops)",
    )
    parser.add_argument(
        "--years",
        default="2000-2023",
        help="Year range (default: 2000-2023)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/usda_nass",
        help="Output directory (default: data/usda_nass)",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        key_path = Path(args.api_key_file)
        if key_path.exists():
            api_key = key_path.read_text().strip()
        else:
            print(
                f"Error: No API key. Provide --api-key or create {args.api_key_file}"
            )
            print("Register (free) at https://quickstats.nass.usda.gov/api/")
            sys.exit(1)

    year_start, year_end = (int(y) for y in args.years.split("-"))

    crops = MICHIGAN_CROPS
    if args.crop:
        crop_upper = args.crop.upper()
        if crop_upper in crops:
            crops = {crop_upper: crops[crop_upper]}
        else:
            print(f"Error: Unknown crop '{args.crop}'. Available: {list(crops.keys())}")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for crop_name, crop_info in crops.items():
        records = download_crop_yields(api_key, crop_name, crop_info, year_start, year_end)
        all_records.extend(records)
        time.sleep(1)

    if not all_records:
        print("No data downloaded.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    csv_path = output_dir / "michigan_yields.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} records to {csv_path}")

    summary = {
        "source": "USDA NASS Quick Stats",
        "state": "MI",
        "years": f"{year_start}-{year_end}",
        "crops": list(crops.keys()),
        "total_records": len(all_records),
        "counties": sorted(df["county"].unique().tolist()),
        "provenance": {
            "api": API_BASE,
            "download_date": pd.Timestamp.now().isoformat(),
            "script": "scripts/download_usda_nass.py",
        },
    }

    json_path = output_dir / "michigan_yields.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

    print(f"\nSummary:")
    print(f"  Crops: {len(crops)}")
    print(f"  Counties: {df['county'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Records: {len(df)}")
    for crop_name in crops:
        crop_df = df[df["crop"] == crop_name]
        if not crop_df.empty:
            print(
                f"  {crop_name}: {len(crop_df)} records, "
                f"mean yield {crop_df['yield_value'].mean():.1f} {crop_df['unit'].iloc[0]}"
            )


if __name__ == "__main__":
    main()
