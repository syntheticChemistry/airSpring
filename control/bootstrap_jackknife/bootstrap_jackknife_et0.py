#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 080: Bootstrap & Jackknife Confidence Intervals for Seasonal ET₀.

Validates statistical resampling methods (bootstrap mean CI, jackknife
variance) applied to daily ET₀ time series from a Michigan growing season.

Bootstrap: resample N daily ET₀ values B times → mean distribution → 95% CI.
Jackknife: leave-one-out → estimate mean and variance of the estimator.

The Rust implementation uses gpu::bootstrap (BarraCuda S71 WGSL shader)
and gpu::jackknife (BarraCuda S71 WGSL shader) with CPU fallback.

References:
    Efron B (1979) Bootstrap methods: another look at the jackknife.
    Quenouille MH (1956) Notes on bias in estimation.
    Allen RG et al. (1998) FAO-56 Crop Evapotranspiration.

Usage:
    python3 control/bootstrap_jackknife/bootstrap_jackknife_et0.py

Output:
    control/bootstrap_jackknife/benchmark_bootstrap_jackknife.json
"""

import json
import math
import sys
from pathlib import Path

# ── FAO-56 ET₀ (same minimal implementation as mc_et0) ──────────────

def saturation_vapour_pressure(t_c):
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))

def psychrometric_constant(elev_m):
    p = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    return 0.000665 * p

def extraterrestrial_radiation(lat_deg, doy):
    lat = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    decl = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(lat) * math.tan(decl))))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.sin(ws)
    )
    return max(ra, 0.0)

def daily_et0(tmin, tmax, rs, u2, ea, elev, lat, doy):
    tmean = (tmin + tmax) / 2.0
    es_tmax = saturation_vapour_pressure(tmax)
    es_tmin = saturation_vapour_pressure(tmin)
    delta = 4098.0 * (es_tmax + es_tmin) / 2.0 / (tmean + 237.3) ** 2
    gamma = psychrometric_constant(elev)
    es = (es_tmax + es_tmin) / 2.0
    vpd = max(es - ea, 0.0)
    ra = extraterrestrial_radiation(lat, doy)
    rso = (0.75 + 2e-5 * elev) * ra
    ratio = min(max(rs / rso if rso > 0 else 0.7, 0.25), 1.0)
    rns = (1.0 - 0.23) * rs
    sigma = 4.903e-9
    rnl = sigma * ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2.0 * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd) / (delta + gamma * (1.0 + 0.34 * u2))
    return max(et0, 0.0)


# ── Deterministic synthetic ET₀ season ──────────────────────────────

def generate_season(n_days=153, lat=42.73, elev=256.0, seed=42):
    """Generate a deterministic seasonal ET₀ series (Michigan growing season)."""
    import random
    rng = random.Random(seed)
    et0_series = []
    for i in range(n_days):
        doy = 121 + i
        season_frac = i / n_days
        tmax = 22.0 + 10.0 * math.sin(math.pi * season_frac) + rng.gauss(0, 1.5)
        tmin = tmax - 8.0 - rng.gauss(0, 1.0)
        rs = 15.0 + 10.0 * math.sin(math.pi * season_frac) + rng.gauss(0, 2.0)
        rs = max(rs, 2.0)
        u2 = 2.0 + rng.gauss(0, 0.5)
        u2 = max(u2, 0.3)
        ea = 1.2 + 0.5 * season_frac + rng.gauss(0, 0.3)
        ea = max(ea, 0.3)
        val = daily_et0(tmin, tmax, rs, u2, ea, elev, lat, doy)
        et0_series.append(val)
    return et0_series


# ── Bootstrap ────────────────────────────────────────────────────────

class LehmerRng:
    def __init__(self, seed):
        self.state = int(seed) & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def next_u32(self):
        self.state = (self.state * 48271) % 0x7FFFFFFF
        return self.state

    def next_index(self, n):
        return self.next_u32() % n


def bootstrap_mean_ci(data, n_bootstrap=1000, confidence=0.95, seed=42):
    """Bootstrap 95% CI for the mean."""
    n = len(data)
    rng = LehmerRng(seed)
    boot_means = []

    for _ in range(n_bootstrap):
        sample_sum = 0.0
        for _ in range(n):
            idx = rng.next_index(n)
            sample_sum += data[idx]
        boot_means.append(sample_sum / n)

    boot_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(math.floor(alpha * (len(boot_means) - 1)))
    hi_idx = int(math.ceil((1.0 - alpha) * (len(boot_means) - 1)))
    ci_lower = boot_means[lo_idx]
    ci_upper = boot_means[hi_idx]

    point_mean = sum(data) / len(data)
    boot_mean = sum(boot_means) / len(boot_means)
    var_val = sum((x - boot_mean) ** 2 for x in boot_means) / len(boot_means)
    std_error = math.sqrt(var_val)

    return {
        "mean": point_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_bootstrap": n_bootstrap,
    }


# ── Jackknife ────────────────────────────────────────────────────────

def jackknife_mean_variance(data):
    """Leave-one-out jackknife for mean."""
    n = len(data)
    total = sum(data)
    full_mean = total / n
    jack_means = [(total - data[i]) / (n - 1) for i in range(n)]
    jack_mean_of_means = sum(jack_means) / n
    variance = ((n - 1) / n) * sum((jm - jack_mean_of_means) ** 2 for jm in jack_means)
    std_error = math.sqrt(variance)
    return {
        "mean": full_mean,
        "variance": variance,
        "std_error": std_error,
    }


# ── Tests ────────────────────────────────────────────────────────────

def main():
    et0_series = generate_season()
    n = len(et0_series)

    # Test 1: Bootstrap on full season
    boot_full = bootstrap_mean_ci(et0_series, n_bootstrap=1000, seed=42)
    print(f"Test 1 (bootstrap, N={n}, B=1000): mean={boot_full['mean']:.6f}, "
          f"CI=[{boot_full['ci_lower']:.4f}, {boot_full['ci_upper']:.4f}], "
          f"SE={boot_full['std_error']:.6f}")

    # Test 2: Jackknife on full season
    jack_full = jackknife_mean_variance(et0_series)
    print(f"Test 2 (jackknife, N={n}): mean={jack_full['mean']:.6f}, "
          f"var={jack_full['variance']:.6f}, SE={jack_full['std_error']:.6f}")

    # Test 3: Known analytical — uniform [1..10]
    uniform_data = list(range(1, 11))
    uniform_data_f = [float(x) for x in uniform_data]
    boot_known = bootstrap_mean_ci(uniform_data_f, n_bootstrap=2000, seed=42)
    jack_known = jackknife_mean_variance(uniform_data_f)
    print(f"Test 3 (known, [1..10]): boot_mean={boot_known['mean']:.4f}, "
          f"jack_mean={jack_known['mean']:.4f}, jack_var={jack_known['variance']:.6f}")

    # Test 4: Small sample (N=5)
    small = et0_series[:5]
    boot_small = bootstrap_mean_ci(small, n_bootstrap=500, seed=42)
    jack_small = jackknife_mean_variance(small)
    print(f"Test 4 (small N=5): boot_CI=[{boot_small['ci_lower']:.4f}, {boot_small['ci_upper']:.4f}], "
          f"jack_SE={jack_small['std_error']:.6f}")

    # Test 5: Constant data → zero variance
    const_data = [3.5] * 20
    jack_const = jackknife_mean_variance(const_data)
    boot_const = bootstrap_mean_ci(const_data, n_bootstrap=200, seed=42)
    print(f"Test 5 (constant=3.5): jack_var={jack_const['variance']:.10f}, "
          f"boot_CI_width={boot_const['ci_upper'] - boot_const['ci_lower']:.10f}")

    checks = 0
    fails = 0

    def check(name, cond):
        nonlocal checks, fails
        checks += 1
        status = "PASS" if cond else "FAIL"
        if not cond:
            fails += 1
        print(f"  [{status}] {name}")

    print(f"\n{'='*60}")
    print(f"  Experiment 080: Bootstrap & Jackknife CI for Seasonal ET₀")
    print(f"{'='*60}\n")

    # Bootstrap checks
    check("boot mean plausible (2-7 mm)", 2.0 < boot_full["mean"] < 7.0)
    check("boot CI contains mean", boot_full["ci_lower"] < boot_full["mean"] < boot_full["ci_upper"])
    check("boot CI width > 0", boot_full["ci_upper"] > boot_full["ci_lower"])
    check("boot CI width < 2mm (narrow for N=153)", boot_full["ci_upper"] - boot_full["ci_lower"] < 2.0)
    check("boot SE > 0", boot_full["std_error"] > 0.0)
    check("boot SE < 1mm", boot_full["std_error"] < 1.0)

    # Jackknife checks
    check("jack mean == boot mean", abs(jack_full["mean"] - boot_full["mean"]) < 1e-10)
    check("jack variance > 0", jack_full["variance"] > 0.0)
    check("jack SE > 0", jack_full["std_error"] > 0.0)
    check("jack SE < 1mm", jack_full["std_error"] < 1.0)

    # Known value [1..10]: mean=5.5
    check("known mean = 5.5", abs(boot_known["mean"] - 5.5) < 0.01)
    check("known jack_mean = 5.5", abs(jack_known["mean"] - 5.5) < 0.01)
    # Analytical jackknife variance for mean of [1..10]:
    # σ² = Var(X)/n = 8.25/10 = 0.825, jack_var ≈ σ²
    check("known jack_var near 0.825", abs(jack_known["variance"] - 0.825) < 0.1)

    # Small sample has wider CI
    check("small CI wider than full", 
          (boot_small["ci_upper"] - boot_small["ci_lower"]) > (boot_full["ci_upper"] - boot_full["ci_lower"]))

    # Constant data: zero variance
    check("const jack_var ≈ 0", jack_const["variance"] < 1e-10)
    check("const boot_CI_width ≈ 0", (boot_const["ci_upper"] - boot_const["ci_lower"]) < 1e-10)
    check("const jack_mean = 3.5", abs(jack_const["mean"] - 3.5) < 1e-10)

    # Bootstrap & jackknife SE should agree (asymptotically)
    check("boot SE and jack SE within 50%",
          0.5 < boot_full["std_error"] / max(jack_full["std_error"], 1e-10) < 2.0)

    print(f"\n  {checks - fails}/{checks} PASS")

    # Build benchmark JSON
    benchmark = {
        "_provenance": {
            "experiment": "Exp 080",
            "method": "Bootstrap + Jackknife resampling for seasonal ET₀ CI",
            "created": "2026-03-07",
            "baseline_script": "control/bootstrap_jackknife/bootstrap_jackknife_et0.py",
            "baseline_command": "python3 control/bootstrap_jackknife/bootstrap_jackknife_et0.py",
            "baseline_commit": "1c11763",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "repository": "ecoPrimals/airSpring",
            "references": [
                "Efron B (1979) Bootstrap methods: another look at the jackknife",
                "Quenouille MH (1956) Notes on bias in estimation",
                "Allen RG et al. (1998) FAO-56 Crop Evapotranspiration"
            ],
            "reproduction_note": "Deterministic RNG seed=42; fully reproducible"
        },
        "season_params": {
            "n_days": 153,
            "latitude_deg": 42.73,
            "elevation_m": 256.0,
            "location": "East Lansing MI",
            "season": "May-Sep 2023 (synthetic deterministic)"
        },
        "et0_series": et0_series,
        "bootstrap_full_season": boot_full,
        "jackknife_full_season": jack_full,
        "bootstrap_known_1_10": boot_known,
        "jackknife_known_1_10": jack_known,
        "bootstrap_small_n5": boot_small,
        "jackknife_small_n5": jack_small,
        "jackknife_constant": jack_const,
        "bootstrap_constant": boot_const,
        "thresholds": {
            "mean_plausible_min_mm": 2.0,
            "mean_plausible_max_mm": 7.0,
            "ci_width_max_mm": 2.0,
            "se_max_mm": 1.0,
            "known_mean": 5.5,
            "known_jack_var": 0.825,
            "boot_jack_se_ratio_min": 0.5,
            "boot_jack_se_ratio_max": 2.0,
        }
    }

    out_path = Path(__file__).parent / "benchmark_bootstrap_jackknife.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  Wrote: {out_path}")

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
