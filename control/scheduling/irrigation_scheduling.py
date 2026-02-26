# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""Exp 014: Irrigation scheduling optimization — strategy comparison.

Compares five irrigation scheduling strategies for corn on silt loam using
FAO-56 water balance + Stewart yield equation. This is the computational
core of "Penny Irrigation" — sovereign scheduling on consumer hardware.

Strategies:
  1. Rainfed (no irrigation)
  2. MAD 50% (irrigate when depletion > 50% of TAW)
  3. MAD 60% (moderate deficit)
  4. MAD 70% (aggressive deficit)
  5. Growth-stage (irrigate only during critical mid-season period)

References:
  - Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148
  - Allen et al. (1998) FAO-56 Ch 8 (water balance), Ch 10 (yield response)
  - Stewart (1977) Yield–water relationship

Provenance:
  Script: control/scheduling/irrigation_scheduling.py
  Benchmark: control/scheduling/benchmark_scheduling.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


# ── Synthetic weather generation ────────────────────────────────────

def generate_michigan_season(n_days, seed=42):
    """Generate synthetic Michigan growing-season weather.

    Based on Open-Meteo ERA5 climatology for East Lansing MI, 2023.
    ET₀ follows a sinusoidal pattern peaking mid-season; precipitation
    is stochastic with wet/dry spells.
    """
    rng = np.random.RandomState(seed)

    day_frac = np.arange(n_days, dtype=float) / n_days

    et0 = 3.0 + 2.5 * np.sin(np.pi * day_frac) + rng.normal(0, 0.4, n_days)
    et0 = np.clip(et0, 0.5, 8.0)

    precip = np.zeros(n_days)
    for i in range(n_days):
        if rng.random() < 0.30:
            precip[i] = rng.exponential(8.0)
    precip = np.clip(precip, 0, 50.0)

    return et0, precip


# ── Crop coefficient schedule ───────────────────────────────────────

def kc_schedule(n_days, kc_ini, kc_mid, kc_end, stages):
    """Generate daily Kc from crop growth stages (FAO-56 Ch 6)."""
    kc = np.zeros(n_days)
    l_ini = stages[0]["length_days"]
    l_dev = stages[1]["length_days"]
    l_mid = stages[2]["length_days"]
    l_late = stages[3]["length_days"]
    total = l_ini + l_dev + l_mid + l_late

    for d in range(min(n_days, total)):
        if d < l_ini:
            kc[d] = kc_ini
        elif d < l_ini + l_dev:
            frac = (d - l_ini) / l_dev
            kc[d] = kc_ini + (kc_mid - kc_ini) * frac
        elif d < l_ini + l_dev + l_mid:
            kc[d] = kc_mid
        else:
            frac = (d - l_ini - l_dev - l_mid) / l_late
            kc[d] = kc_mid + (kc_end - kc_mid) * frac

    for d in range(total, n_days):
        kc[d] = kc_end

    return kc


# ── Water balance simulation ────────────────────────────────────────

def simulate_season(et0, precip, kc, soil, strategy, params):
    """Run daily water balance with given scheduling strategy.

    Returns dict with: total_et, total_precip, total_irrigation,
    total_dp, stress_days, daily arrays, mass_balance_error, yield_ratio.
    """
    n = len(et0)
    taw = soil["taw_mm"]
    raw = soil["raw_mm"]
    irrig_depth = params.get("irrigation_depth_mm", 25.0)

    dr = np.zeros(n + 1)
    dr[0] = raw * 0.5

    actual_et = np.zeros(n)
    deep_perc = np.zeros(n)
    irrigation = np.zeros(n)
    ks_arr = np.zeros(n)
    eta_etc_arr = np.zeros(n)

    for d in range(n):
        if dr[d] < raw:
            ks = 1.0
        elif dr[d] >= taw:
            ks = 0.0
        else:
            ks = (taw - dr[d]) / (taw - raw)
        ks_arr[d] = ks

        etc = kc[d] * et0[d]
        eta = ks * etc
        actual_et[d] = eta
        eta_etc_arr[d] = eta / etc if etc > 0 else 1.0

        irr = 0.0
        if strategy == "rainfed":
            irr = 0.0
        elif strategy.startswith("mad_"):
            mad_frac = params["mad_fraction"]
            threshold = taw * mad_frac
            if dr[d] > threshold:
                irr = irrig_depth
        elif strategy == "growth_stage":
            crit_start = params["critical_start_day"]
            crit_end = params["critical_end_day"]
            mad_frac = params["mad_fraction"]
            threshold = taw * mad_frac
            if crit_start <= d <= crit_end and dr[d] > threshold:
                irr = irrig_depth

        irrigation[d] = irr

        dr_new = dr[d] - precip[d] - irr + eta
        if dr_new < 0:
            deep_perc[d] = -dr_new
            dr_new = 0.0
        elif dr_new > taw:
            dr_new = taw
        dr[d + 1] = dr_new

    # Mass balance: P + I + Dr_n - Dr_0 = ETa + DP  (FAO-56 Ch 8 Eq 85)
    mb_error = abs(
        precip.sum() + irrigation.sum() + dr[-1] - dr[0]
        - actual_et.sum() - deep_perc.sum()
    )

    # Stress days
    stress_days = int(np.sum(ks_arr < 0.99))

    # Yield ratio — Stewart single-stage
    season_eta_etc = actual_et.sum() / max(np.sum(kc * et0), 1e-10)
    ky = params.get("ky_total", 1.25)
    yield_ratio = max(0.0, 1.0 - ky * (1.0 - season_eta_etc))
    yield_ratio = min(yield_ratio, 1.0)

    return {
        "total_et": actual_et.sum(),
        "total_precip": precip.sum(),
        "total_irrigation": irrigation.sum(),
        "total_dp": deep_perc.sum(),
        "stress_days": stress_days,
        "mass_balance_error": mb_error,
        "yield_ratio": yield_ratio,
        "season_eta_etc": season_eta_etc,
        "final_depletion": dr[-1],
    }


# ── Stewart yield equation ──────────────────────────────────────────

def stewart_yield_ratio(ky, eta_etc_ratio):
    """Ya/Ym = 1 - Ky(1 - ETa/ETc). Stewart (1977)."""
    ratio = 1.0 - ky * (1.0 - eta_etc_ratio)
    return max(0.0, min(1.0, ratio))


def water_use_efficiency(yield_kg_ha, total_water_mm):
    """WUE = yield / total water applied (irrigation + effective precip)."""
    if total_water_mm <= 0:
        return 0.0
    return yield_kg_ha / total_water_mm


# ── Validation helpers ──────────────────────────────────────────────

passed_total = 0
failed_total = 0


def check(label, computed, expected, tol):
    global passed_total, failed_total
    ok = abs(computed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.4f}, expected={expected:.4f}, tol={tol}")
    if ok:
        passed_total += 1
    else:
        failed_total += 1
    return ok


def check_range(label, value, low, high):
    global passed_total, failed_total
    ok = low <= value <= high
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: value={value:.4f}, range=[{low:.1f}, {high:.1f}]")
    if ok:
        passed_total += 1
    else:
        failed_total += 1
    return ok


def check_bool(label, condition):
    global passed_total, failed_total
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if condition:
        passed_total += 1
    else:
        failed_total += 1
    return condition


# ── Validators ──────────────────────────────────────────────────────

def validate_analytical(benchmark):
    print("\n── Analytical Stewart equation checks ──")
    checks = benchmark["validation_checks"]["analytical_checks"]
    for tc in checks["test_cases"]:
        computed = stewart_yield_ratio(tc["ky"], tc["eta_etc"])
        check(tc["label"], computed, tc["expected_yield_ratio"], tc["tolerance"])


def validate_strategies(benchmark, results):
    print("\n── Mass balance checks ──")
    for name in benchmark["validation_checks"]["mass_balance"]["strategies"]:
        tol = benchmark["validation_checks"]["mass_balance"]["tolerance_mm"]
        r = results[name]
        check(f"{name} mass balance", r["mass_balance_error"], 0.0, tol)

    print("\n── Yield physical bounds ──")
    for name, r in results.items():
        check_range(f"{name} yield_ratio", r["yield_ratio"], 0.0, 1.0)

    print("\n── Irrigation totals ──")
    irrig_checks = benchmark["validation_checks"]["irrigation_totals"]["strategies"]
    for name, spec in irrig_checks.items():
        r = results[name]
        if "expected_mm" in spec:
            check(f"{name} irrigation", r["total_irrigation"], spec["expected_mm"], spec["tolerance_mm"])
        else:
            check_range(f"{name} irrigation", r["total_irrigation"], spec["min_mm"], spec["max_mm"])

    print("\n── Yield ordering ──")
    order = benchmark["validation_checks"]["yield_ordering"]["expected_order"]
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        check_bool(
            f"yield({b}) >= yield({a})",
            results[b]["yield_ratio"] >= results[a]["yield_ratio"] - 0.001,
        )

    print("\n── Stress days ──")
    sc = benchmark["validation_checks"]["stress_days"]
    check_bool(
        f"rainfed stress_days >= {sc['rainfed_min_stress_days']}",
        results["rainfed"]["stress_days"] >= sc["rainfed_min_stress_days"],
    )
    check_bool(
        "mad_50 fewer stress days than rainfed",
        results["mad_50"]["stress_days"] < results["rainfed"]["stress_days"],
    )

    print("\n── WUE analysis ──")
    target_yield = benchmark["season_parameters"]["target_yield_kg_ha"]
    for name, r in results.items():
        ya = r["yield_ratio"] * target_yield
        total_water = r["total_precip"] + r["total_irrigation"]
        wue = water_use_efficiency(ya, total_water)
        print(f"  {name:>15}: Ya={ya:.0f} kg/ha, water={total_water:.0f} mm, WUE={wue:.1f} kg/ha/mm")

    wue_rainfed = (results["rainfed"]["yield_ratio"] * target_yield) / max(results["rainfed"]["total_precip"], 1)
    wue_mad50 = (results["mad_50"]["yield_ratio"] * target_yield) / (results["mad_50"]["total_precip"] + results["mad_50"]["total_irrigation"])
    check_bool("irrigated WUE defined (non-zero)", wue_mad50 > 0)


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Exp 014: Irrigation Scheduling Optimization")
    print("  Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148")
    print("=" * 65)

    bm_path = Path(__file__).parent / "benchmark_scheduling.json"
    with open(bm_path) as f:
        benchmark = json.load(f)

    crop = benchmark["crop_parameters"]
    soil = benchmark["soil_parameters"]
    season = benchmark["season_parameters"]
    strategies_spec = benchmark["scheduling_strategies"]

    et0, precip = generate_michigan_season(season["length_days"])
    kc = kc_schedule(
        season["length_days"],
        crop["kc_ini"],
        crop["kc_mid"],
        crop["kc_end"],
        crop["ky_stages"],
    )

    print(f"\n  Season: {season['length_days']} days, mean ET₀={et0.mean():.2f} mm/d, total P={precip.sum():.0f} mm")
    print(f"  Crop: {crop['crop']} (Ky={crop['ky_total']}), Soil: {soil['texture']}")
    print(f"  TAW={soil['taw_mm']:.0f} mm, RAW={soil['raw_mm']:.0f} mm\n")

    # Run all strategies
    results = {}
    for name, spec in strategies_spec.items():
        params = {**spec, "ky_total": crop["ky_total"]}
        r = simulate_season(et0, precip, kc, soil, name, params)
        results[name] = r
        print(f"  {name:>15}: I={r['total_irrigation']:6.0f} mm, "
              f"ET={r['total_et']:6.1f} mm, stress={r['stress_days']:3d} d, "
              f"Ya/Ym={r['yield_ratio']:.3f}, MB={r['mass_balance_error']:.4f} mm")

    validate_analytical(benchmark)
    validate_strategies(benchmark, results)

    print(f"\n{'=' * 65}")
    print(f"  Exp 014 Summary: {passed_total} PASS, {failed_total} FAIL")
    print(f"{'=' * 65}")
    sys.exit(0 if failed_total == 0 else 1)


if __name__ == "__main__":
    main()
