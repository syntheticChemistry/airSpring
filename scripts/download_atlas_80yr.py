# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Download 80 years of ERA5 reanalysis for 100 Michigan atlas stations.

Designed to run unattended in the background. Treats HTTP 429 as a
throttle signal and backs off exponentially — the robot respects the API.

  - Resumes automatically (skips stations already on disk)
  - Exponential backoff: 30s → 60s → 120s → 300s → 600s on 429
  - Resets backoff after each successful download
  - Logs progress to stdout (redirect to file for unattended runs)

Usage:
    # Foreground (watch progress)
    python scripts/download_atlas_80yr.py

    # Background (unattended, logs to file)
    nohup python scripts/download_atlas_80yr.py > atlas_download.log 2>&1 &

    # Check progress anytime
    ls data/open_meteo/*1945-01-01*daily.csv | wc -l

Baseline commit: cb59873
Created: 2026-02-26
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_open_meteo import fetch_daily  # noqa: E402

ATLAS_FILE = Path(__file__).parent / "atlas_stations.json"
DATA_DIR = Path(__file__).parent.parent / "data" / "open_meteo"

BACKOFF_SCHEDULE = [30, 60, 120, 300, 600]


def main():
    parser = argparse.ArgumentParser(description="Respectful 80yr atlas download")
    parser.add_argument("--start", type=int, default=1945)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--delay", type=float, default=20.0,
                        help="Base seconds between successful requests (default 20)")
    args = parser.parse_args()

    start_date = f"{args.start}-01-01"
    end_date = f"{args.end}-12-31"
    suffix = f"_{start_date}_{end_date}_daily.csv"

    with open(ATLAS_FILE) as f:
        stations = json.load(f)["stations"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def count_done():
        return len(glob.glob(str(DATA_DIR / f"*{suffix}")))

    all_ids = list(stations.keys())

    while True:
        existing = {
            Path(p).name.replace(suffix, "")
            for p in glob.glob(str(DATA_DIR / f"*{suffix}"))
        }
        missing = [s for s in all_ids if s not in existing]

        if not missing:
            print(f"All {len(all_ids)} stations complete!")
            break

        print(f"\n[{count_done()}/{len(all_ids)}] {len(missing)} remaining")

        for sid in missing:
            info = stations[sid]
            backoff_idx = 0

            while True:
                print(f"  {sid}...", end="", flush=True)
                try:
                    df = fetch_daily(info["lat"], info["lon"], start_date, end_date)
                    df["station"] = sid
                    df["lat"] = info["lat"]
                    df["lon"] = info["lon"]
                    df["elevation_m"] = info["elevation_m"]
                    path = DATA_DIR / f"{sid}{suffix}"
                    df.to_csv(path, index=False)
                    valid = df["tmax_c"].notna().sum()
                    print(f" OK ({len(df)} days, {valid} valid) [{count_done()}/{len(all_ids)}]")
                    time.sleep(args.delay)
                    break
                except Exception as e:
                    if "429" in str(e):
                        wait = BACKOFF_SCHEDULE[min(backoff_idx, len(BACKOFF_SCHEDULE) - 1)]
                        backoff_idx += 1
                        print(f" 429 → sleeping {wait}s", flush=True)
                        time.sleep(wait)
                    else:
                        print(f" ERROR: {e}")
                        time.sleep(60)
                        break

        if missing and count_done() < len(all_ids):
            existing_now = {
                Path(p).name.replace(suffix, "")
                for p in glob.glob(str(DATA_DIR / f"*{suffix}"))
            }
            still_missing = [s for s in all_ids if s not in existing_now]
            if len(still_missing) == len(missing):
                print("No progress this pass — cooling down 10 minutes")
                time.sleep(600)

    print(f"\nDone. {count_done()} station files in {DATA_DIR}")


if __name__ == "__main__":
    main()
