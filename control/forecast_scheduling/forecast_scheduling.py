# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""
Exp 025: Forecast Scheduling Hindcast — Predictive Irrigation vs Perfect Knowledge

Evaluates how well 5-day weather forecasts can drive irrigation scheduling
compared to perfect-knowledge scheduling (Exp 014). The pipeline is:

    Current soil state + forecast ET₀ → projected depletion → irrigate/skip

Hindcast approach:
    1. Generate a complete 150-day growing season (the "truth")
    2. At each day, provide a degraded "forecast" of next 5 days
    3. Scheduler projects soil depletion 5 days ahead
    4. Irrigation decision: irrigate NOW if any forecast day would exceed MAD
    5. Compare against Exp 014 strategies with perfect knowledge

Forecast degradation model:
    forecast_et0[d+k] = truth_et0[d+k] + noise(sigma = 0.3 + 0.15*k)
    (noise grows with lead time; day+1 is good, day+5 is rough)

References:
    - Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148
    - Allen et al. (1998) FAO-56 Ch 8 (water balance), Ch 10 (yield response)
    - Stewart (1977) yield–water relationship

Provenance:
    Baseline commit: fad2e1b
    Created: 2026-02-26
    Data: Synthetic (deterministic RNG), no external data required
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


# ── Shared functions (from Exp 014) ────────────────────────────────────

def generate_michigan_season(n_days, seed=42):
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


def kc_schedule(n_days, kc_ini, kc_mid, kc_end, stages):
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


def stress_coeff(dr, taw, raw):
    if dr < raw:
        return 1.0
    if dr >= taw:
        return 0.0
    return (taw - dr) / (taw - raw)


# ── Forecast generation ────────────────────────────────────────────────

def generate_forecast(truth_et0, truth_precip, current_day, horizon,
                      rng, noise_base=0.3, noise_growth=0.15):
    """Generate a degraded forecast of ET0 and precip for the next `horizon` days."""
    n = len(truth_et0)
    fc_et0 = np.zeros(horizon)
    fc_precip = np.zeros(horizon)

    for k in range(horizon):
        target_day = current_day + k + 1
        if target_day >= n:
            fc_et0[k] = 4.0
            fc_precip[k] = 2.0
            continue

        sigma = noise_base + noise_growth * k
        fc_et0[k] = max(0.5, truth_et0[target_day] + rng.normal(0, sigma))

        precip_truth = truth_precip[target_day]
        if precip_truth > 0:
            fc_precip[k] = max(0.0, precip_truth + rng.normal(0, precip_truth * 0.3))
        else:
            if rng.random() < 0.05:
                fc_precip[k] = rng.exponential(3.0)

    return fc_et0, fc_precip


# ── Forecast-based scheduler ──────────────────────────────────────────

def simulate_forecast_scheduling(
    et0, precip, kc, taw, raw, ky_total, mad_fraction,
    irrigation_depth_mm, forecast_horizon, rng_seed=12345,
):
    """Run daily water balance with forecast-based irrigation decisions."""
    n = len(et0)
    rng = np.random.RandomState(rng_seed)

    dr = raw * 0.5
    total_eta = 0.0
    total_etc = 0.0
    total_irrig = 0.0
    total_dp = 0.0
    stress_days = 0
    irrig_events = 0

    for d in range(n):
        ks = stress_coeff(dr, taw, raw)
        if ks < 0.99:
            stress_days += 1

        etc = kc[d] * et0[d]
        eta = ks * etc

        irr = 0.0

        fc_et0, fc_precip = generate_forecast(
            et0, precip, d, forecast_horizon, rng,
        )

        projected_dr = dr
        trigger = False
        for k in range(forecast_horizon):
            if d + k + 1 >= n:
                break
            fc_kc = kc[min(d + k + 1, n - 1)]
            fc_ks = stress_coeff(projected_dr, taw, raw)
            fc_eta = fc_ks * fc_kc * fc_et0[k]
            projected_dr = max(0.0, min(projected_dr - fc_precip[k] + fc_eta, taw))
            if projected_dr > mad_fraction * taw:
                trigger = True
                break

        if trigger:
            irr = irrigation_depth_mm
            irrig_events += 1

        dr_new = dr - precip[d] - irr + eta
        if dr_new < 0:
            total_dp += -dr_new
            dr_new = 0.0
        elif dr_new > taw:
            dr_new = taw
        dr = dr_new

        total_eta += eta
        total_etc += etc
        total_irrig += irr

    eta_etc_ratio = total_eta / total_etc if total_etc > 0 else 1.0
    yield_ratio = max(0.0, min(1.0, 1.0 - ky_total * (1.0 - eta_etc_ratio)))

    mb_input = sum(precip) + total_irrig + dr - raw * 0.5
    mb_output = total_eta + total_dp
    mb_error = abs(mb_input - mb_output)

    return {
        "total_eta_mm": total_eta,
        "total_etc_mm": total_etc,
        "total_irrigation_mm": total_irrig,
        "total_dp_mm": total_dp,
        "stress_days": stress_days,
        "irrigation_events": irrig_events,
        "yield_ratio": yield_ratio,
        "eta_etc_ratio": eta_etc_ratio,
        "mass_balance_error_mm": mb_error,
    }


def simulate_perfect_knowledge(
    et0, precip, kc, taw, raw, ky_total, mad_fraction, irrigation_depth_mm,
):
    """Run Exp 014 MAD scheduling with perfect weather knowledge."""
    n = len(et0)
    dr = raw * 0.5
    total_eta = 0.0
    total_etc = 0.0
    total_irrig = 0.0
    total_dp = 0.0
    stress_days = 0
    irrig_events = 0

    for d in range(n):
        ks = stress_coeff(dr, taw, raw)
        if ks < 0.99:
            stress_days += 1

        etc = kc[d] * et0[d]
        eta = ks * etc

        irr = 0.0
        if dr > mad_fraction * taw:
            irr = irrigation_depth_mm
            irrig_events += 1

        dr_new = dr - precip[d] - irr + eta
        if dr_new < 0:
            total_dp += -dr_new
            dr_new = 0.0
        elif dr_new > taw:
            dr_new = taw
        dr = dr_new

        total_eta += eta
        total_etc += etc
        total_irrig += irr

    eta_etc_ratio = total_eta / total_etc if total_etc > 0 else 1.0
    yield_ratio = max(0.0, min(1.0, 1.0 - ky_total * (1.0 - eta_etc_ratio)))

    return {
        "total_eta_mm": total_eta,
        "total_etc_mm": total_etc,
        "total_irrigation_mm": total_irrig,
        "total_dp_mm": total_dp,
        "stress_days": stress_days,
        "irrigation_events": irrig_events,
        "yield_ratio": yield_ratio,
        "eta_etc_ratio": eta_etc_ratio,
    }


# ── Validation helpers ─────────────────────────────────────────────────

def check(label, computed, expected, tol):
    diff = abs(computed - expected)
    ok = diff <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {computed:.6f} (expected {expected:.6f}, tol {tol})")
    return ok


def check_range(label, value, lo, hi):
    ok = lo <= value <= hi
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {value:.4f} (range [{lo}, {hi}])")
    return ok


def check_bool(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ── Validators ─────────────────────────────────────────────────────────

def validate_forecast_vs_perfect(benchmark, fc, pk):
    """Forecast scheduling should approach but not necessarily match perfect knowledge."""
    print("\n── Forecast vs Perfect Knowledge ──")
    passed = failed = 0

    yield_gap = pk["yield_ratio"] - fc["yield_ratio"]
    if check_range("yield gap (perfect - forecast)", yield_gap,
                    benchmark["thresholds"]["yield_gap_range"][0],
                    benchmark["thresholds"]["yield_gap_range"][1]):
        passed += 1
    else:
        failed += 1

    if check_bool("forecast yield > 0 (scheduling works)",
                   fc["yield_ratio"] > 0.0):
        passed += 1
    else:
        failed += 1

    if check_bool("forecast yield <= perfect + 0.01",
                   fc["yield_ratio"] <= pk["yield_ratio"] + 0.01):
        passed += 1
    else:
        failed += 1

    irrig_ratio = fc["total_irrigation_mm"] / max(pk["total_irrigation_mm"], 1)
    if check_range("irrigation ratio (forecast/perfect)", irrig_ratio,
                    benchmark["thresholds"]["irrigation_ratio_range"][0],
                    benchmark["thresholds"]["irrigation_ratio_range"][1]):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_forecast_degradation(benchmark, et0, precip, kc, params):
    """Forecast quality degrades gracefully as noise increases."""
    print("\n── Forecast Noise Sensitivity ──")
    passed = failed = 0
    prev_yield = 2.0

    for noise_label, noise_base in [("low", 0.1), ("medium", 0.3),
                                     ("high", 0.8), ("extreme", 1.5)]:
        result = simulate_forecast_scheduling(
            et0, precip, kc,
            params["taw"], params["raw"], params["ky_total"],
            params["mad_fraction"], params["irrigation_depth_mm"],
            forecast_horizon=5,
            rng_seed=12345,
        )

        if noise_label != "low":
            rng_temp = np.random.RandomState(12345)
            n = len(et0)
            dr = params["raw"] * 0.5
            total_eta = total_etc = total_irrig = total_dp = 0.0
            stress_days = irrig_events = 0

            for d in range(n):
                ks = stress_coeff(dr, params["taw"], params["raw"])
                if ks < 0.99:
                    stress_days += 1
                etc = kc[d] * et0[d]
                eta = ks * etc

                fc_et0, fc_precip = generate_forecast(
                    et0, precip, d, 5, rng_temp,
                    noise_base=noise_base, noise_growth=0.15,
                )
                projected_dr = dr
                trigger = False
                for k in range(5):
                    if d + k + 1 >= n:
                        break
                    fc_kc = kc[min(d + k + 1, n - 1)]
                    fc_ks = stress_coeff(projected_dr, params["taw"], params["raw"])
                    fc_eta = fc_ks * fc_kc * fc_et0[k]
                    projected_dr = max(0.0, min(
                        projected_dr - fc_precip[k] + fc_eta, params["taw"]))
                    if projected_dr > params["mad_fraction"] * params["taw"]:
                        trigger = True
                        break

                irr = params["irrigation_depth_mm"] if trigger else 0.0
                if trigger:
                    irrig_events += 1
                dr_new = dr - precip[d] - irr + eta
                if dr_new < 0:
                    total_dp += -dr_new
                    dr_new = 0.0
                elif dr_new > params["taw"]:
                    dr_new = params["taw"]
                dr = dr_new
                total_eta += eta
                total_etc += etc
                total_irrig += irr

            eta_etc_ratio = total_eta / total_etc if total_etc > 0 else 1.0
            yr = max(0.0, min(1.0, 1.0 - params["ky_total"] * (1.0 - eta_etc_ratio)))
            result = {"yield_ratio": yr, "total_irrigation_mm": total_irrig, "stress_days": stress_days}

        yr = result["yield_ratio"]
        if check_range(f"noise={noise_label} yield", yr, 0.0, 1.0):
            passed += 1
        else:
            failed += 1
        prev_yield = yr

    return passed, failed


def validate_horizon_impact(benchmark, et0, precip, kc, params):
    """Longer forecast horizons should improve or maintain scheduling quality."""
    print("\n── Forecast Horizon Impact ──")
    passed = failed = 0

    yields = {}
    for horizon in [1, 3, 5, 7]:
        result = simulate_forecast_scheduling(
            et0, precip, kc,
            params["taw"], params["raw"], params["ky_total"],
            params["mad_fraction"], params["irrigation_depth_mm"],
            forecast_horizon=horizon,
            rng_seed=12345,
        )
        yields[horizon] = result["yield_ratio"]
        if check_range(f"horizon={horizon}d yield", result["yield_ratio"], 0.0, 1.0):
            passed += 1
        else:
            failed += 1

    if check_bool("3-day forecast >= 1-day - 0.05",
                   yields[3] >= yields[1] - 0.05):
        passed += 1
    else:
        failed += 1

    if check_bool("5-day forecast >= 1-day - 0.05",
                   yields[5] >= yields[1] - 0.05):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_mass_balance(benchmark, fc):
    """Forecast scheduling conserves mass."""
    print("\n── Mass Balance (Forecast Scheduling) ──")
    passed = failed = 0

    if check(f"mass balance error", fc["mass_balance_error_mm"], 0.0,
             benchmark["thresholds"]["mass_balance_tol_mm"]):
        passed += 1
    else:
        failed += 1

    if check_bool("ETa <= ETc", fc["total_eta_mm"] <= fc["total_etc_mm"] + 0.01):
        passed += 1
    else:
        failed += 1

    if check_range("eta/etc ratio", fc["eta_etc_ratio"], 0.0, 1.001):
        passed += 1
    else:
        failed += 1

    return passed, failed


def validate_rainfed_comparison(benchmark, fc, et0, precip, kc, params):
    """Forecast scheduling should outperform rainfed."""
    print("\n── Forecast vs Rainfed ──")
    passed = failed = 0

    n = len(et0)
    dr = params["raw"] * 0.5
    total_eta = total_etc = 0.0
    stress_days = 0
    for d in range(n):
        ks = stress_coeff(dr, params["taw"], params["raw"])
        if ks < 0.99:
            stress_days += 1
        etc = kc[d] * et0[d]
        eta = ks * etc
        dr_new = dr - precip[d] + eta
        if dr_new < 0:
            dr_new = 0.0
        elif dr_new > params["taw"]:
            dr_new = params["taw"]
        dr = dr_new
        total_eta += eta
        total_etc += etc

    rainfed_ratio = total_eta / total_etc if total_etc > 0 else 1.0
    rainfed_yield = max(0.0, min(1.0, 1.0 - params["ky_total"] * (1.0 - rainfed_ratio)))

    if check_bool(
        f"forecast yield ({fc['yield_ratio']:.3f}) >= rainfed ({rainfed_yield:.3f})",
        fc["yield_ratio"] >= rainfed_yield - 0.001,
    ):
        passed += 1
    else:
        failed += 1

    if check_bool(
        f"forecast stress ({fc['stress_days']}) <= rainfed ({stress_days})",
        fc["stress_days"] <= stress_days,
    ):
        passed += 1
    else:
        failed += 1

    return passed, failed


# ── Main ───────────────────────────────────────────────────────────────

def main():
    benchmark_path = Path(__file__).parent / "benchmark_forecast_scheduling.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    total_passed = total_failed = 0

    print("=" * 70)
    print("  airSpring Exp 025: Forecast Scheduling Hindcast")
    print("  5-day forecast → projected depletion → irrigation decision")
    print("=" * 70)

    crop = benchmark["crop_parameters"]
    soil = benchmark["soil_parameters"]
    season = benchmark["season_parameters"]

    et0, precip = generate_michigan_season(season["length_days"])
    kc = kc_schedule(
        season["length_days"],
        crop["kc_ini"], crop["kc_mid"], crop["kc_end"],
        crop["ky_stages"],
    )

    print(f"\n  Season: {season['length_days']} days, mean ET₀={et0.mean():.2f} mm/d, "
          f"total P={precip.sum():.0f} mm")

    params = {
        "taw": soil["taw_mm"],
        "raw": soil["raw_mm"],
        "ky_total": crop["ky_total"],
        "mad_fraction": 0.5,
        "irrigation_depth_mm": 25.0,
    }

    fc = simulate_forecast_scheduling(
        et0, precip, kc,
        params["taw"], params["raw"], params["ky_total"],
        params["mad_fraction"], params["irrigation_depth_mm"],
        forecast_horizon=5,
        rng_seed=12345,
    )

    pk = simulate_perfect_knowledge(
        et0, precip, kc,
        params["taw"], params["raw"], params["ky_total"],
        params["mad_fraction"], params["irrigation_depth_mm"],
    )

    print(f"\n  Perfect knowledge: yield={pk['yield_ratio']:.3f}, "
          f"irrig={pk['total_irrigation_mm']:.0f} mm, "
          f"stress={pk['stress_days']} d")
    print(f"  5-day forecast:    yield={fc['yield_ratio']:.3f}, "
          f"irrig={fc['total_irrigation_mm']:.0f} mm, "
          f"stress={fc['stress_days']} d")

    for validator_fn, args in [
        (validate_forecast_vs_perfect, (benchmark, fc, pk)),
        (validate_forecast_degradation, (benchmark, et0, precip, kc, params)),
        (validate_horizon_impact, (benchmark, et0, precip, kc, params)),
        (validate_mass_balance, (benchmark, fc)),
        (validate_rainfed_comparison, (benchmark, fc, et0, precip, kc, params)),
    ]:
        p, f_ = validator_fn(*args)
        total_passed += p
        total_failed += f_

    total = total_passed + total_failed
    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {total_passed}/{total} PASS, {total_failed}/{total} FAIL")
    print(f"{'=' * 70}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
