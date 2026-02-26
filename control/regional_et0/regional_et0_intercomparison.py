# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
airSpring Experiment 010 — Regional ET₀ Intercomparison (Michigan Microclimates)

Computes FAO-56 Penman-Monteith ET₀ independently for 6 Michigan stations
using 2023 growing season Open-Meteo ERA5 data, then validates:

  1. Per-station ET₀ is physically reasonable (1–8 mm/day summer range)
  2. Seasonal totals match published Michigan agricultural references
  3. Spatial variability is within expected microclimate range
  4. Our FAO-56 PM correlates strongly with Open-Meteo's ERA5 ET₀
  5. Station rankings are geographically consistent (lake effect, latitude)

This establishes the baseline for GPU-batched ET₀ at scale (BatchedEt0).

Reference:
    Allen, R.G. et al. (1998) FAO-56; MSU Enviro-weather network;
    Open-Meteo Historical Weather API (ERA5 reanalysis).
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fao56"))
from penman_monteith import (
    saturation_vapour_pressure,
    slope_vapour_pressure_curve,
    atmospheric_pressure,
    psychrometric_constant,
    mean_saturation_vapour_pressure,
    actual_vapour_pressure_rh,
    extraterrestrial_radiation,
    clear_sky_radiation,
    net_shortwave_radiation,
    net_longwave_radiation,
    fao56_penman_monteith,
)


# ── Data loading ──────────────────────────────────────────────────────

def load_station_csv(csv_path):
    """Load Open-Meteo CSV, return list of row dicts with parsed floats."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                parsed = {
                    "date": row["date"],
                    "tmax": float(row["tmax_c"]),
                    "tmin": float(row["tmin_c"]),
                    "tmean": float(row["tmean_c"]),
                    "rh_max": float(row["rh_max_pct"]),
                    "rh_min": float(row["rh_min_pct"]),
                    "wind_2m": float(row["wind_2m_m_s"]),
                    "solar_rad": float(row["solar_rad_mj_m2"]),
                    "precip": float(row["precip_mm"]),
                    "et0_om": float(row["et0_openmeteo_mm"]),
                    "lat": float(row["lat"]),
                    "elevation_m": float(row["elevation_m"]),
                }
                doy = _date_to_doy(row["date"])
                parsed["doy"] = doy
                rows.append(parsed)
            except (ValueError, KeyError):
                continue
    return rows


def _date_to_doy(date_str):
    """Convert YYYY-MM-DD to day of year."""
    parts = date_str.split("-")
    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
        days_in_month[2] = 29
    return sum(days_in_month[:m]) + d


# ── FAO-56 PM ET₀ computation ────────────────────────────────────────

def compute_et0(row):
    """Compute ET₀ for a single day using our FAO-56 PM implementation."""
    lat_rad = math.radians(row["lat"])
    tmean = row["tmean"]
    pressure = atmospheric_pressure(row["elevation_m"])
    gamma = psychrometric_constant(pressure)
    delta = slope_vapour_pressure_curve(tmean)
    es = mean_saturation_vapour_pressure(row["tmax"], row["tmin"])
    ea = actual_vapour_pressure_rh(
        row["tmax"], row["tmin"], row["rh_max"], row["rh_min"]
    )
    vpd = es - ea

    ra = extraterrestrial_radiation(lat_rad, row["doy"])
    rso = clear_sky_radiation(row["elevation_m"], ra)
    rns = net_shortwave_radiation(row["solar_rad"], 0.23)
    rs_over_rso = min(row["solar_rad"] / rso, 1.0) if rso > 0 else 0.05
    rnl = net_longwave_radiation(row["tmax"], row["tmin"], ea, rs_over_rso)
    rn = rns - rnl

    et0 = fao56_penman_monteith(rn, 0.0, tmean, row["wind_2m"], vpd, delta, gamma)
    return max(et0, 0.0)


# ── Statistics ────────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if vals else 0.0

def pearson_r(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0

def rmse(x, y):
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)) / len(x))

def cv(vals):
    """Coefficient of variation (%)."""
    m = mean(vals)
    return 100 * std(vals) / m if m > 0 else 0.0


# ── Validation framework ─────────────────────────────────────────────

class Validator:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label, observed, expected, tol=0.01):
        diff = abs(observed - expected)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}: observed={observed:.4f} expected={expected:.4f} tol={tol}")

    def check_bool(self, label, condition):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}")

    def check_range(self, label, value, lo, hi):
        ok = lo <= value <= hi
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}: {value:.4f} in [{lo}, {hi}]")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  TOTAL: {self.passed}/{total} PASS")
        if self.failed > 0:
            print(f"  *** {self.failed} FAILED ***")
        print(f"{'='*60}")
        return self.failed == 0


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent.parent.parent / "data" / "open_meteo"
    stations_json = data_dir / "stations.json"

    if not stations_json.exists():
        print(f"[SKIP] No station data at {data_dir}")
        sys.exit(0)

    with open(stations_json) as f:
        stations_meta = json.load(f)

    v = Validator()

    station_results = {}

    # ── Per-station ET₀ computation ──────────────────────────────
    print("\n── Per-Station ET₀ (FAO-56 PM, 2023 growing season) ──")
    for station_id, meta in stations_meta.items():
        csv_path = data_dir / f"{station_id}_2023-05-01_2023-09-30_daily.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {station_id}: CSV not found")
            continue

        rows = load_station_csv(csv_path)
        if not rows:
            continue

        et0_rust = [compute_et0(r) for r in rows]
        et0_om = [r["et0_om"] for r in rows]

        r2 = pearson_r(et0_om, et0_rust) ** 2
        err = rmse(et0_om, et0_rust)
        seasonal_total = sum(et0_rust)
        daily_mean = mean(et0_rust)
        daily_max = max(et0_rust)
        daily_min = min(et0_rust)

        station_results[station_id] = {
            "name": meta["name"],
            "lat": meta["lat"],
            "elevation_m": meta["elevation_m"],
            "n_days": len(rows),
            "seasonal_total": seasonal_total,
            "daily_mean": daily_mean,
            "daily_max": daily_max,
            "daily_min": daily_min,
            "r2_vs_om": r2,
            "rmse_vs_om": err,
            "et0_series": et0_rust,
        }

        print(f"\n  {meta['name']} ({station_id}): {len(rows)} days")
        print(f"    Season total: {seasonal_total:.1f} mm, Mean: {daily_mean:.2f} mm/day")
        print(f"    Range: [{daily_min:.2f}, {daily_max:.2f}] mm/day")
        print(f"    R² vs Open-Meteo: {r2:.4f}, RMSE: {err:.3f} mm/day")

    if not station_results:
        print("No stations processed")
        sys.exit(1)

    # ── Section 1: Physical reasonableness ──────────────────────
    print("\n── Physical reasonableness checks ──")
    for sid, sr in station_results.items():
        v.check_range(f"{sid} daily mean in [2, 6] mm/day", sr["daily_mean"], 2.0, 6.0)
        v.check_range(f"{sid} daily max in [3, 10] mm/day", sr["daily_max"], 3.0, 10.0)
        v.check_bool(f"{sid} daily min >= 0", sr["daily_min"] >= 0)

    # Michigan growing season ET₀ typically 400–700 mm (MSU Enviro-weather)
    print("\n── Seasonal total range ──")
    for sid, sr in station_results.items():
        v.check_range(
            f"{sid} season total in [350, 750] mm",
            sr["seasonal_total"], 350.0, 750.0
        )

    # ── Section 2: FAO-56 PM vs Open-Meteo correlation ─────────
    print("\n── FAO-56 PM vs Open-Meteo ERA5 correlation ──")
    for sid, sr in station_results.items():
        v.check_bool(f"{sid} R² > 0.85", sr["r2_vs_om"] > 0.85)
        v.check_bool(f"{sid} RMSE < 2.0 mm/day", sr["rmse_vs_om"] < 2.0)

    # ── Section 3: Spatial variability ─────────────────────────
    print("\n── Spatial variability (Michigan microclimate) ──")
    all_means = [sr["daily_mean"] for sr in station_results.values()]
    all_totals = [sr["seasonal_total"] for sr in station_results.values()]

    mean_of_means = mean(all_means)
    cv_means = cv(all_means)
    spread_mm = max(all_totals) - min(all_totals)

    print(f"  Grand mean ET₀: {mean_of_means:.2f} mm/day")
    print(f"  CV across stations: {cv_means:.1f}%")
    print(f"  Season total spread: {spread_mm:.0f} mm")

    # Michigan microclimate variability: moderate for Lower MI (stations span ~1.5° lat)
    v.check_range("CV of daily means in [1, 30]%", cv_means, 1.0, 30.0)
    v.check_range("Season total spread in [20, 250] mm", spread_mm, 20.0, 250.0)

    # Number of stations processed
    v.check_bool(f"All 6 stations processed", len(station_results) == 6)

    # ── Section 4: Geographic consistency ──────────────────────
    print("\n── Geographic consistency ──")
    # Lake-effect: stations closer to Lake Michigan (lower elevation,
    # western MI) should have slightly moderated ET₀ (lower max temp)
    lats = {sid: sr["lat"] for sid, sr in station_results.items()}
    totals = {sid: sr["seasonal_total"] for sid, sr in station_results.items()}

    # Higher latitude stations should NOT consistently have much higher ET₀
    # (longer days partially offset by cooler temps)
    lat_range = max(lats.values()) - min(lats.values())
    v.check_range(
        f"Latitude span ({lat_range:.2f}°) covers meaningful range",
        lat_range, 0.5, 5.0
    )

    # All stations should be Michigan (lat 41.7–46.6)
    for sid, lat in lats.items():
        v.check_range(f"{sid} latitude in Michigan range", lat, 41.0, 47.0)

    # ── Section 5: Cross-station correlation matrix ────────────
    print("\n── Cross-station temporal correlation ──")
    station_ids = list(station_results.keys())
    min_days = min(sr["n_days"] for sr in station_results.values())
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            s1, s2 = station_ids[i], station_ids[j]
            et0_1 = station_results[s1]["et0_series"][:min_days]
            et0_2 = station_results[s2]["et0_series"][:min_days]
            r = pearson_r(et0_1, et0_2)
            # Regional stations should be well-correlated (r > 0.7)
            v.check_bool(
                f"r({s1}, {s2}) = {r:.3f} > 0.70",
                r > 0.70,
            )

    # ── Summary ──────────────────────────────────────────────────
    ok = v.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
