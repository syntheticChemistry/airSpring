#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 079: Monte Carlo ET₀ Uncertainty Propagation.

Propagates measurement uncertainties through the FAO-56 Penman-Monteith
equation via Monte Carlo sampling.  Establishes Python baselines for:
  - Central ET₀ (unperturbed)
  - MC mean, standard deviation, 5th/95th percentiles
  - Parametric 90% confidence interval
  - Convergence: std stabilises as N increases
  - Sensitivity: higher σ → wider spread

Uncertainties follow groundSpring V73 (Exp 003, Table 2):
  - T:  σ = 0.4 °C  (radiation-shielded thermistor)
  - RH: σ = 4.0 %   (capacitive humidity sensor)
  - u₂: σ_frac = 0.08 (cup / sonic anemometer)
  - Rs: σ_frac = 0.07 (pyranometer calibration drift)

Reference:
    Allen RG et al. (1998) FAO-56 Crop Evapotranspiration.
    Gong L et al. (2006) Sensitivity of Penman-Monteith ET₀.

Usage:
    python3 control/mc_et0/mc_et0_propagation.py

Output:
    control/mc_et0/benchmark_mc_et0.json
"""

import json
import math
import sys
from pathlib import Path

# ── FAO-56 Penman-Monteith (simplified, self-contained) ─────────────

def saturation_vapour_pressure(t_c):
    """Tetens formula (FAO-56 Eq. 11)."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))

def slope_vapour_pressure(t_c):
    """Slope Δ (FAO-56 Eq. 13)."""
    es = saturation_vapour_pressure(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2

def psychrometric_constant(elev_m):
    """γ (FAO-56 Eq. 8)."""
    p = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    return 0.000665 * p

def actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max):
    """ea from RH (FAO-56 Eq. 17)."""
    es_tmin = saturation_vapour_pressure(tmin)
    es_tmax = saturation_vapour_pressure(tmax)
    return (es_tmin * rh_max / 100.0 + es_tmax * rh_min / 100.0) / 2.0

def extraterrestrial_radiation(lat_deg, doy):
    """Ra (FAO-56 Eq. 21)."""
    lat = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    decl = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(-math.tan(lat) * math.tan(decl))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.sin(ws)
    )
    return max(ra, 0.0)

def daily_et0(tmin, tmax, rs, u2, ea, elev, lat, doy):
    """FAO-56 PM daily reference ET₀ (mm/day)."""
    tmean = (tmin + tmax) / 2.0
    delta = slope_vapour_pressure(tmean)
    gamma = psychrometric_constant(elev)
    es = (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin)) / 2.0
    vpd = max(es - ea, 0.0)

    ra = extraterrestrial_radiation(lat, doy)
    rso = (0.75 + 2e-5 * elev) * ra
    ratio = rs / rso if rso > 0 else 0.7
    ratio = min(max(ratio, 0.25), 1.0)

    rns = (1.0 - 0.23) * rs

    tmax_k4 = (tmax + 273.16) ** 4
    tmin_k4 = (tmin + 273.16) ** 4
    sigma = 4.903e-9
    rnl = sigma * (tmax_k4 + tmin_k4) / 2.0 * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl

    g = 0.0
    et0 = (
        (0.408 * delta * (rn - g) + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd)
        / (delta + gamma * (1.0 + 0.34 * u2))
    )
    return max(et0, 0.0)


# ── Monte Carlo engine ──────────────────────────────────────────────

class LehmerRng:
    """Deterministic Lehmer LCG matching the Rust implementation."""
    def __init__(self, seed):
        self.state = (seed + 1) & 0x7FFFFFFFFFFFFFFF

    def next_uniform(self):
        self.state = (self.state * 48271) % 0x7FFFFFFF
        return self.state / 0x7FFFFFFF

    def next_normal(self):
        u1 = max(self.next_uniform(), 1e-300)
        u2 = self.next_uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def mc_et0_propagation(tmin, tmax, rs, u2, ea, elev, lat, doy,
                        sigma_tmax, sigma_tmin, sigma_rh_max, sigma_rh_min,
                        sigma_wind_frac, sigma_rs_frac,
                        n_samples, seed):
    """Run MC propagation, return dict of summary statistics."""
    central = daily_et0(tmin, tmax, rs, u2, ea, elev, lat, doy)
    rng = LehmerRng(seed)
    samples = []

    rh_max_base = ea / saturation_vapour_pressure(tmin)
    rh_min_base = ea / saturation_vapour_pressure(tmax)

    for _ in range(n_samples):
        z_tmax = rng.next_normal()
        z_tmin = rng.next_normal()
        z_rh_max = rng.next_normal()
        z_rh_min = rng.next_normal()
        z_wind = rng.next_normal()
        z_rs = rng.next_normal()

        tp_max = tmax + z_tmax * sigma_tmax
        tp_min = tmin + z_tmin * sigma_tmin

        rh_max_p = max(1.0, min(100.0, rh_max_base * 100.0 + z_rh_max * sigma_rh_max))
        rh_min_p = max(1.0, min(100.0, rh_min_base * 100.0 + z_rh_min * sigma_rh_min))

        ea_p = actual_vapour_pressure_rh(tp_min, tp_max, rh_min_p, rh_max_p)
        wind_p = max(0.01, u2 * (1.0 + z_wind * sigma_wind_frac))
        rs_p = max(0.01, rs * (1.0 + z_rs * sigma_rs_frac))

        val = daily_et0(tp_min, tp_max, rs_p, wind_p, ea_p, elev, lat, doy)
        if math.isfinite(val) and val > 0:
            samples.append(val)

    n = len(samples)
    if n == 0:
        return {"et0_central": central, "et0_mean": central, "et0_std": 0.0,
                "et0_p05": central, "et0_p95": central, "n_samples": 0}

    mean_val = sum(samples) / n
    var_val = sum((x - mean_val) ** 2 for x in samples) / n
    std_val = math.sqrt(var_val)
    samples_sorted = sorted(samples)

    def percentile(pct):
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return samples_sorted[int(k)]
        return samples_sorted[int(f)] * (c - k) + samples_sorted[int(c)] * (k - f)

    return {
        "et0_central": central,
        "et0_mean": mean_val,
        "et0_std": std_val,
        "et0_p05": percentile(5),
        "et0_p95": percentile(95),
        "n_samples": n,
    }


# ── Run experiments ─────────────────────────────────────────────────

def main():
    # Uccle, Belgium — FAO-56 Example 18 (July 6)
    base = dict(tmin=12.3, tmax=21.5, rs=22.07, u2=2.078, ea=1.409,
                elev=100.0, lat=50.80, doy=187)

    default_unc = dict(sigma_tmax=0.4, sigma_tmin=0.4,
                       sigma_rh_max=4.0, sigma_rh_min=4.0,
                       sigma_wind_frac=0.08, sigma_rs_frac=0.07)

    # Test 1: Default uncertainty, N=2000, seed=42
    r1 = mc_et0_propagation(**base, **default_unc, n_samples=2000, seed=42)
    print(f"Test 1 (default, N=2000): central={r1['et0_central']:.6f}, "
          f"mean={r1['et0_mean']:.6f}, std={r1['et0_std']:.6f}, "
          f"CI90=[{r1['et0_p05']:.4f}, {r1['et0_p95']:.4f}]")

    # Test 2: Zero uncertainty → zero spread
    zero_unc = dict(sigma_tmax=0.0, sigma_tmin=0.0,
                    sigma_rh_max=0.0, sigma_rh_min=0.0,
                    sigma_wind_frac=0.0, sigma_rs_frac=0.0)
    r2 = mc_et0_propagation(**base, **zero_unc, n_samples=500, seed=42)
    print(f"Test 2 (zero unc, N=500):  central={r2['et0_central']:.6f}, "
          f"std={r2['et0_std']:.6f}")

    # Test 3: High uncertainty
    high_unc = dict(sigma_tmax=1.0, sigma_tmin=1.0,
                    sigma_rh_max=10.0, sigma_rh_min=10.0,
                    sigma_wind_frac=0.15, sigma_rs_frac=0.15)
    r3 = mc_et0_propagation(**base, **high_unc, n_samples=2000, seed=42)
    print(f"Test 3 (high unc, N=2000): central={r3['et0_central']:.6f}, "
          f"mean={r3['et0_mean']:.6f}, std={r3['et0_std']:.6f}")

    # Test 4: Arid climate (Phoenix-like, DOY 200)
    arid = dict(tmin=28.0, tmax=42.0, rs=28.0, u2=1.5, ea=0.8,
                elev=340.0, lat=33.45, doy=200)
    r4 = mc_et0_propagation(**arid, **default_unc, n_samples=2000, seed=42)
    print(f"Test 4 (arid, N=2000):     central={r4['et0_central']:.6f}, "
          f"mean={r4['et0_mean']:.6f}, std={r4['et0_std']:.6f}")

    # Test 5: Humid climate (East Lansing, DOY 182)
    humid = dict(tmin=18.0, tmax=29.0, rs=20.0, u2=2.5, ea=2.0,
                 elev=256.0, lat=42.73, doy=182)
    r5 = mc_et0_propagation(**humid, **default_unc, n_samples=2000, seed=42)
    print(f"Test 5 (humid, N=2000):    central={r5['et0_central']:.6f}, "
          f"mean={r5['et0_mean']:.6f}, std={r5['et0_std']:.6f}")

    # Test 6: Convergence — std should stabilise with increasing N
    convergence = []
    for n in [100, 500, 1000, 2000, 5000]:
        r = mc_et0_propagation(**base, **default_unc, n_samples=n, seed=42)
        convergence.append({"n": n, "mean": r["et0_mean"], "std": r["et0_std"]})
    print(f"Test 6 (convergence):      stds = {[c['std'] for c in convergence]}")

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
    print(f"  Experiment 079: MC ET₀ Uncertainty Propagation")
    print(f"{'='*60}\n")

    # Validation checks
    check("central ET₀ plausible (2-6 mm/day)", 2.0 < r1["et0_central"] < 6.0)
    check("MC mean near central (<0.5 mm)", abs(r1["et0_mean"] - r1["et0_central"]) < 0.5)
    check("MC std reasonable (0.05-2.0 mm)", 0.05 < r1["et0_std"] < 2.0)
    check("90% CI non-trivial (>0.1 mm)", r1["et0_p95"] - r1["et0_p05"] > 0.1)
    check("p05 < mean < p95", r1["et0_p05"] < r1["et0_mean"] < r1["et0_p95"])
    check("zero uncertainty gives ~zero std (<0.01)", r2["et0_std"] < 0.01)
    check("high unc wider than default", r3["et0_std"] > r1["et0_std"])
    check("arid ET₀ > humid ET₀ (central)", r4["et0_central"] > r5["et0_central"])
    check("arid ET₀ plausible (5-12 mm)", 5.0 < r4["et0_central"] < 12.0)
    check("humid ET₀ plausible (3-7 mm)", 3.0 < r5["et0_central"] < 7.0)
    check("convergence: std stabilises (last 2 within 20%)",
          abs(convergence[-1]["std"] - convergence[-2]["std"]) / max(convergence[-2]["std"], 1e-10) < 0.20)
    check("all MC samples valid (n_samples == 2000)", r1["n_samples"] == 2000)

    print(f"\n  {checks - fails}/{checks} PASS")

    # Build benchmark JSON
    benchmark = {
        "_provenance": {
            "experiment": "Exp 079",
            "method": "Monte Carlo uncertainty propagation through FAO-56 PM",
            "created": "2026-03-07",
            "baseline_script": "control/mc_et0/mc_et0_propagation.py",
            "baseline_command": "python3 control/mc_et0/mc_et0_propagation.py",
            "baseline_commit": "1c11763",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "repository": "ecoPrimals/airSpring",
            "references": [
                "Allen RG et al. (1998) FAO-56 Crop Evapotranspiration",
                "Gong L et al. (2006) Sensitivity of Penman-Monteith ET₀",
                "groundSpring Exp 003: Humidity dominates ET₀ uncertainty at 66%"
            ],
            "reproduction_note": "Deterministic Lehmer LCG seed=42; fully reproducible"
        },
        "base_input": {
            "description": "FAO-56 Example 18 (Uccle, Belgium, July 6)",
            "tmin": 12.3, "tmax": 21.5, "rs_mj": 22.07,
            "u2_ms": 2.078, "ea_kpa": 1.409,
            "elevation_m": 100.0, "latitude_deg": 50.80, "doy": 187
        },
        "default_uncertainties": {
            "sigma_tmax_c": 0.4, "sigma_tmin_c": 0.4,
            "sigma_rh_max_pct": 4.0, "sigma_rh_min_pct": 4.0,
            "sigma_wind_frac": 0.08, "sigma_rs_frac": 0.07,
            "source": "groundSpring V73 Exp 003 Table 2"
        },
        "test_default_n2000": r1,
        "test_zero_uncertainty": r2,
        "test_high_uncertainty": r3,
        "test_arid_climate": {**r4, "location": "Phoenix AZ (synthetic)"},
        "test_humid_climate": {**r5, "location": "East Lansing MI"},
        "convergence": convergence,
        "thresholds": {
            "mean_near_central_mm": 0.5,
            "std_min_mm": 0.05,
            "std_max_mm": 2.0,
            "ci90_min_width_mm": 0.1,
            "zero_unc_std_max": 0.01,
            "convergence_stability_pct": 20.0
        }
    }

    out_path = Path(__file__).parent / "benchmark_mc_et0.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  Wrote: {out_path}")

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
