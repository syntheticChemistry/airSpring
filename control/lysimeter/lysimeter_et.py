# SPDX-License-Identifier: AGPL-3.0-or-later
#!/usr/bin/env python3
"""Exp 016: Lysimeter ET direct measurement.

Implements the mass-change ET measurement pipeline from Dong & Hansen (2023):
  1. Load cell calibration (known masses → linear fit)
  2. Temperature compensation (thermal drift correction)
  3. Mass-to-ET conversion: ET_mm = -ΔM_kg / (A_m² × ρ_w_kg/m³) × 1000
  4. Data quality filtering (rain events, below-resolution changes)
  5. Hourly diurnal ET pattern (sinusoidal, night ≈ 0)
  6. Synthetic daily comparison: lysimeter ET vs FAO-56 ET₀

This is the direct measurement counterpart to the FAO-56 equation-based
ET₀ from Exp 001. Ground truth for "Penny Irrigation" calibration.

References:
  Dong & Hansen (2023) Smart Ag Tech 4:100147
  López-Urrea et al. (2006) Ag Water Mgmt 82(1-2):13-24

Provenance:
  Baseline commit: cb59873
  Script: control/lysimeter/lysimeter_et.py
  Benchmark: control/lysimeter/benchmark_lysimeter.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


# ── Mass-to-ET conversion ──────────────────────────────────────────

def mass_to_et_mm(mass_change_kg, area_m2):
    """Convert lysimeter mass change to ET depth.

    Standard lysimeter simplification: 1 kg water / 1 m² = 1 mm depth.
    ET_mm = -ΔM_kg / A_m²
    Negative mass change (evaporation loss) → positive ET.
    """
    return -mass_change_kg / area_m2


# ── Temperature compensation ───────────────────────────────────────

def compensate_temperature(mass_raw_kg, temp_c, alpha_g_per_c, t_ref_c):
    """Correct load cell reading for thermal drift.

    M_corrected = M_raw - α(T - T_ref) / 1000
    α in g/°C, conversion to kg via /1000.
    """
    correction_kg = alpha_g_per_c * (temp_c - t_ref_c) / 1000.0
    return mass_raw_kg - correction_kg


# ── Data quality filtering ─────────────────────────────────────────

def is_valid_reading(delta_g, resolution_g=10, rain_threshold_g=500):
    """Check if a mass-change reading is valid for ET computation.

    Reject:
    - Readings below instrument resolution (noise)
    - Large positive jumps (rain/irrigation events)
    """
    if abs(delta_g) < resolution_g:
        return False
    if delta_g > rain_threshold_g:
        return False
    return True


# ── Calibration ────────────────────────────────────────────────────

def linear_regression(x, y):
    """Simple least-squares linear fit: y = slope*x + intercept."""
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy * sxx - sx * sxy) / denom

    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    y_mean = sy / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return slope, intercept, r_squared


# ── Diurnal ET pattern ─────────────────────────────────────────────

def hourly_et_fraction(hour):
    """Fraction of daily ET occurring at a given hour.

    Sinusoidal model: ET peaks at solar noon (hour 12), zero at night.
    ET_frac(h) = max(0, sin(π(h-6)/12)) for 6≤h≤18, else 0.
    """
    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, math.sin(math.pi * (hour - 6) / 12.0))


def generate_hourly_et(daily_et_mm, noise_std=0.02):
    """Generate 24-hour ET profile from daily total."""
    rng = np.random.RandomState(123)
    fracs = np.array([hourly_et_fraction(h) for h in range(24)])
    total = fracs.sum()
    if total > 0:
        fracs = fracs / total * daily_et_mm
    noise = rng.normal(0, noise_std, 24)
    fracs = np.clip(fracs + noise, 0, None)
    return fracs


# ── Synthetic daily comparison ─────────────────────────────────────

def generate_synthetic_comparison(n_days, mean_et0, kc, noise_std, seed=42):
    """Generate synthetic lysimeter and FAO-56 daily ET for comparison.

    Lysimeter ET = Kc * ET₀ + noise (direct measurement with instrument error).
    """
    rng = np.random.RandomState(seed)
    day_frac = np.arange(n_days, dtype=float) / n_days
    et0 = mean_et0 + 1.5 * np.sin(np.pi * day_frac) + rng.normal(0, 0.3, n_days)
    et0 = np.clip(et0, 0.5, 8.0)

    et_lysimeter = kc * et0 + rng.normal(0, noise_std, n_days)
    et_lysimeter = np.clip(et_lysimeter, 0.0, None)

    return et0, et_lysimeter


def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))


def pearson_r(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.corrcoef(a, b)[0, 1])


def bias(a, b):
    return float(np.mean(np.array(a) - np.array(b)))


# ── Validation helpers ──────────────────────────────────────────────

passed_total = 0
failed_total = 0


def check(label, computed, expected, tol):
    global passed_total, failed_total
    ok = abs(computed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.6f}, expected={expected:.6f}, tol={tol}")
    if ok:
        passed_total += 1
    else:
        failed_total += 1
    return ok


def check_range(label, value, low, high):
    global passed_total, failed_total
    ok = low <= value <= high
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: value={value:.4f}, range=[{low}, {high}]")
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

def validate_mass_to_et(benchmark):
    print("\n── Mass-to-ET conversion ──")
    for tc in benchmark["validation_checks"]["mass_to_et_conversion"]["test_cases"]:
        computed = mass_to_et_mm(tc["mass_change_kg"], tc["area_m2"])
        check(tc["label"], computed, tc["expected_et_mm"], tc["tolerance"])


def validate_temperature_compensation(benchmark):
    print("\n── Temperature compensation ──")
    tc_params = benchmark["temperature_compensation"]
    alpha = tc_params["alpha_g_per_c"]
    t_ref = tc_params["t_ref_c"]
    for tc in benchmark["validation_checks"]["temperature_compensation"]["test_cases"]:
        computed = compensate_temperature(tc["mass_raw_kg"], tc["temp_c"], alpha, t_ref)
        check(tc["label"], computed, tc["expected_corr_kg"], tc["tolerance"])


def validate_data_quality(benchmark):
    print("\n── Data quality filtering ──")
    for tc in benchmark["validation_checks"]["data_quality_filter"]["test_cases"]:
        computed = is_valid_reading(tc["delta_g"])
        check_bool(
            f"{tc['label']}: valid={computed}, expected={tc['expected_valid']}",
            computed == tc["expected_valid"],
        )


def validate_calibration(benchmark):
    print("\n── Load cell calibration ──")
    cal = benchmark["calibration"]
    slope, intercept, r2 = linear_regression(cal["known_masses_kg"], cal["measured_readings_kg"])
    check(
        "calibration R²",
        r2,
        cal["expected_r_squared"],
        benchmark["validation_checks"]["calibration_linearity"]["tolerance"],
    )
    check_range("calibration slope", slope, 0.995, 1.005)
    check_range("calibration intercept", intercept, -0.02, 0.02)


def validate_daily_comparison(benchmark):
    print("\n── Synthetic daily lysimeter vs ET₀ ──")
    spec = benchmark["validation_checks"]["daily_et_synthetic"]
    et0, et_lys = generate_synthetic_comparison(
        spec["days"], spec["mean_et0_mm"], spec["kc"], spec["noise_std_mm"],
    )

    r = pearson_r(et0, et_lys)
    rms = rmse(et0, et_lys)
    b = abs(bias(et_lys, et0))

    check_range("correlation (r)", r, spec["expected_correlation_min"], 1.0)
    check_range("RMSE", rms, 0.0, spec["expected_rmse_max_mm"])
    check_range("abs(bias)", b, 0.0, spec["expected_bias_max_mm"])

    print(f"    r={r:.4f}, RMSE={rms:.3f} mm, bias={b:.3f} mm")


def validate_hourly_pattern(benchmark):
    print("\n── Hourly diurnal ET pattern ──")
    for tc in benchmark["validation_checks"]["hourly_et_pattern"]["test_cases"]:
        h = tc["hour"]
        frac = hourly_et_fraction(h)
        if tc.get("expected_low"):
            check_bool(f"{tc['label']}: ET_frac({h})={frac:.4f} ≈ 0 (night)", frac < 0.05)
        elif tc.get("expected_peak"):
            check_bool(f"{tc['label']}: ET_frac({h})={frac:.4f} is peak", frac > 0.95)
        else:
            check_bool(f"{tc['label']}: ET_frac({h})={frac:.4f} > 0 (daytime)", frac > 0.1)


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Exp 016: Lysimeter ET Direct Measurement")
    print("  Dong & Hansen (2023) Smart Ag Tech 4:100147")
    print("=" * 65)

    bm_path = Path(__file__).parent / "benchmark_lysimeter.json"
    with open(bm_path) as f:
        benchmark = json.load(f)

    design = benchmark["lysimeter_design"]
    print(f"\n  Lysimeter: {design['surface_area_m2']} m², depth={design['soil_depth_m']} m")
    print(f"  Load cell: {design['load_cell_capacity_kg']} kg cap, "
          f"{design['load_cell_resolution_g']} g resolution")

    validate_mass_to_et(benchmark)
    validate_temperature_compensation(benchmark)
    validate_data_quality(benchmark)
    validate_calibration(benchmark)
    validate_daily_comparison(benchmark)
    validate_hourly_pattern(benchmark)

    print(f"\n{'=' * 65}")
    print(f"  Exp 016 Summary: {passed_total} PASS, {failed_total} FAIL")
    print(f"{'=' * 65}")
    sys.exit(0 if failed_total == 0 else 1)


if __name__ == "__main__":
    main()
