#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment 081: Standardized Precipitation Index (SPI).

Validates the SPI drought classification method (McKee et al. 1993).
SPI transforms accumulated precipitation to a standard normal distribution
by fitting a gamma distribution, enabling objective drought categorization.

Algorithm:
    1. Accumulate precipitation over k months (SPI-1, SPI-3, SPI-6, SPI-12)
    2. Fit gamma distribution (α, β) via maximum likelihood
    3. Transform to standard normal via CDF → inverse normal

Categories (WMO):
    SPI ≥ 2.0   : Extremely wet
    1.5 ≤ SPI < 2.0 : Very wet
    1.0 ≤ SPI < 1.5 : Moderately wet
    -1.0 < SPI < 1.0 : Near normal
    -1.5 < SPI ≤ -1.0 : Moderately dry
    -2.0 < SPI ≤ -1.5 : Severely dry
    SPI ≤ -2.0  : Extremely dry

References:
    McKee TB, Doesken NJ, Kleist J (1993) The relationship of drought
        frequency and duration to time scales. AMS 8th Conf Applied Climatology.
    Edwards DC, McKee TB (1997) Characteristics of 20th century drought.
        Climatology Report 97-2, Colorado State University.
    WMO (2012) Standardized Precipitation Index User Guide. WMO-No. 1090.

Usage:
    python3 control/drought_index/drought_index_spi.py

Output:
    control/drought_index/benchmark_drought_index.json
"""

import json
import math
import sys
from pathlib import Path


# ── Gamma distribution fitting (MLE via Thom's approximation) ────────

def gamma_mle_fit(data):
    """Fit gamma(α, β) via Thom (1958) MLE approximation.

    Returns (alpha, beta) or None if data invalid.
    """
    positive = [x for x in data if x > 0]
    n = len(positive)
    if n < 3:
        return None

    mean_val = sum(positive) / n
    log_mean = sum(math.log(x) for x in positive) / n
    A = math.log(mean_val) - log_mean

    if A <= 0:
        return None

    # Thom (1958) initial estimate
    alpha = (1.0 / (4.0 * A)) * (1.0 + math.sqrt(1.0 + 4.0 * A / 3.0))
    beta = mean_val / alpha

    return (alpha, beta)


def gamma_cdf(x, alpha, beta):
    """Regularized incomplete gamma function P(a, x/β)."""
    if x <= 0:
        return 0.0
    return regularized_gamma_p(alpha, x / beta)


def regularized_gamma_p(a, x):
    """Regularized lower incomplete gamma P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < a + 1:
        return gamma_series(a, x)
    return 1.0 - gamma_cf(a, x)


def gamma_series(a, x):
    """Series expansion for P(a, x)."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(200):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gamma_cf(a, x):
    """Continued fraction for Q(a, x) = 1 - P(a, x)."""
    b = x + 1 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def norm_ppf(p):
    """Inverse standard normal (Abramowitz & Stegun 26.2.23)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

    if p < 0.5:
        return -x
    return x


# ── SPI computation ─────────────────────────────────────────────────

def compute_spi(monthly_precip, scale=1):
    """Compute SPI at given time scale (months).

    Returns list of SPI values (NaN for insufficient history).
    """
    n = len(monthly_precip)
    spi = [float('nan')] * n

    # Accumulate precipitation over `scale` months
    accum = []
    for i in range(n):
        if i < scale - 1:
            accum.append(float('nan'))
        else:
            total = sum(monthly_precip[i - scale + 1:i + 1])
            accum.append(total)

    # Get valid (non-NaN) accumulated values
    valid = [x for x in accum if not math.isnan(x)]
    if len(valid) < 3:
        return spi

    # Fit gamma distribution
    fit = gamma_mle_fit(valid)
    if fit is None:
        return spi

    alpha, beta = fit

    # Proportion of zeros
    q = sum(1 for x in valid if x == 0) / len(valid)

    # Transform each accumulated value to SPI
    for i in range(n):
        if math.isnan(accum[i]):
            continue
        if accum[i] == 0:
            prob = q
        else:
            prob = q + (1.0 - q) * gamma_cdf(accum[i], alpha, beta)
        prob = max(1e-10, min(1.0 - 1e-10, prob))
        spi[i] = norm_ppf(prob)

    return spi


def classify_spi(spi_value):
    """WMO drought classification."""
    if math.isnan(spi_value):
        return "insufficient_data"
    if spi_value >= 2.0:
        return "extremely_wet"
    if spi_value >= 1.5:
        return "very_wet"
    if spi_value >= 1.0:
        return "moderately_wet"
    if spi_value > -1.0:
        return "near_normal"
    if spi_value > -1.5:
        return "moderately_dry"
    if spi_value > -2.0:
        return "severely_dry"
    return "extremely_dry"


# ── Test data: 5 years of monthly precip (East Lansing MI, synthetic) ──

def generate_monthly_precip(n_years=5, seed=42):
    """Deterministic synthetic monthly precipitation (Michigan pattern)."""
    import random
    rng = random.Random(seed)

    monthly_means = [50, 45, 55, 75, 85, 80, 75, 80, 85, 70, 65, 55]
    precip = []
    for year in range(n_years):
        for month in range(12):
            base = monthly_means[month]
            val = max(0.0, base * (0.5 + rng.random()))
            precip.append(round(val, 1))
    return precip


def main():
    precip = generate_monthly_precip(n_years=5)
    n_months = len(precip)

    # Test 1: SPI-1
    spi1 = compute_spi(precip, scale=1)
    valid_spi1 = [x for x in spi1 if not math.isnan(x)]
    print(f"Test 1 (SPI-1, N={n_months}): {len(valid_spi1)} valid values, "
          f"mean={sum(valid_spi1)/len(valid_spi1):.4f}")

    # Test 2: SPI-3
    spi3 = compute_spi(precip, scale=3)
    valid_spi3 = [x for x in spi3 if not math.isnan(x)]
    print(f"Test 2 (SPI-3, N={n_months}): {len(valid_spi3)} valid values, "
          f"mean={sum(valid_spi3)/len(valid_spi3):.4f}")

    # Test 3: SPI-6
    spi6 = compute_spi(precip, scale=6)
    valid_spi6 = [x for x in spi6 if not math.isnan(x)]
    print(f"Test 3 (SPI-6, N={n_months}): {len(valid_spi6)} valid values")

    # Test 4: SPI-12
    spi12 = compute_spi(precip, scale=12)
    valid_spi12 = [x for x in spi12 if not math.isnan(x)]
    print(f"Test 4 (SPI-12, N={n_months}): {len(valid_spi12)} valid values")

    # Test 5: Gamma fit on known data
    known = [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0]
    fit = gamma_mle_fit(known)
    print(f"Test 5 (gamma fit): alpha={fit[0]:.6f}, beta={fit[1]:.6f}")

    # Test 6: Classification
    classes = [classify_spi(v) for v in spi1 if not math.isnan(v)]
    class_counts = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    print(f"Test 6 (classification): {class_counts}")

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
    print(f"  Experiment 081: Standardized Precipitation Index (SPI)")
    print(f"{'='*60}\n")

    # SPI-1 checks
    check("SPI-1: all 60 months valid", len(valid_spi1) == 60)
    check("SPI-1: mean near 0", abs(sum(valid_spi1) / len(valid_spi1)) < 0.5)
    check("SPI-1: min > -4", min(valid_spi1) > -4.0)
    check("SPI-1: max < 4", max(valid_spi1) < 4.0)
    check("SPI-1: std near 1.0",
          0.5 < (sum((x - sum(valid_spi1)/len(valid_spi1))**2 for x in valid_spi1) / len(valid_spi1))**0.5 < 1.5)

    # SPI-3 checks
    check("SPI-3: 58 months valid", len(valid_spi3) == 58)
    check("SPI-3: mean near 0", abs(sum(valid_spi3) / len(valid_spi3)) < 0.5)

    # SPI-6 checks
    check("SPI-6: 55 months valid", len(valid_spi6) == 55)

    # SPI-12 checks
    check("SPI-12: 49 months valid", len(valid_spi12) == 49)
    check("SPI-12: first 11 months NaN", all(math.isnan(spi12[i]) for i in range(11)))

    # Gamma fit checks
    check("gamma alpha > 0", fit[0] > 0)
    check("gamma beta > 0", fit[1] > 0)
    check("gamma alpha*beta = mean",
          abs(fit[0] * fit[1] - sum(known) / len(known)) < 0.1)

    # Classification checks
    check("near_normal is dominant class",
          class_counts.get("near_normal", 0) > len(classes) * 0.4)
    check("at least 2 classes present", len(class_counts) >= 2)

    # Scale ordering: longer SPI → smoother (lower variance)
    std1 = (sum((x - sum(valid_spi1)/len(valid_spi1))**2 for x in valid_spi1) / len(valid_spi1))**0.5
    std3 = (sum((x - sum(valid_spi3)/len(valid_spi3))**2 for x in valid_spi3) / len(valid_spi3))**0.5
    check("SPI-3 smoother than SPI-1 (lower or similar std)", std3 <= std1 * 1.2)

    # SPI values should be bounded
    all_valid = valid_spi1 + valid_spi3 + valid_spi6 + valid_spi12
    check("all SPI in [-4, 4]", all(-4.0 <= v <= 4.0 for v in all_valid))

    print(f"\n  {checks - fails}/{checks} PASS")

    # Collect first few SPI values for benchmark comparison
    benchmark = {
        "_provenance": {
            "experiment": "Exp 081",
            "method": "Standardized Precipitation Index (McKee et al. 1993)",
            "created": "2026-03-07",
            "baseline_script": "control/drought_index/drought_index_spi.py",
            "baseline_command": "python3 control/drought_index/drought_index_spi.py",
            "baseline_commit": "1c11763",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "repository": "ecoPrimals/airSpring",
            "references": [
                "McKee TB et al. (1993) Drought frequency and duration to time scales",
                "Edwards DC, McKee TB (1997) Characteristics of 20th century drought",
                "WMO (2012) Standardized Precipitation Index User Guide. WMO-No. 1090"
            ],
            "reproduction_note": "Deterministic RNG seed=42; fully reproducible"
        },
        "monthly_precip_mm": precip,
        "season_params": {
            "n_years": 5,
            "n_months": 60,
            "location": "East Lansing MI (synthetic)",
            "monthly_means_mm": [50, 45, 55, 75, 85, 80, 75, 80, 85, 70, 65, 55],
        },
        "gamma_fit_known": {
            "data": known,
            "alpha": fit[0],
            "beta": fit[1],
        },
        "spi1": {
            "values": spi1,
            "n_valid": len(valid_spi1),
            "mean": sum(valid_spi1) / len(valid_spi1),
            "std": std1,
            "min": min(valid_spi1),
            "max": max(valid_spi1),
        },
        "spi3": {
            "values": spi3,
            "n_valid": len(valid_spi3),
            "mean": sum(valid_spi3) / len(valid_spi3),
        },
        "spi6": {
            "values": spi6,
            "n_valid": len(valid_spi6),
        },
        "spi12": {
            "values": spi12,
            "n_valid": len(valid_spi12),
        },
        "classification_counts": class_counts,
        "thresholds": {
            "spi_mean_max_abs": 0.5,
            "spi_range_min": -4.0,
            "spi_range_max": 4.0,
            "std_range": [0.5, 1.5],
        }
    }

    def sanitize_nan(obj):
        """Replace NaN with None for valid JSON output."""
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, list):
            return [sanitize_nan(x) for x in obj]
        if isinstance(obj, dict):
            return {k: sanitize_nan(v) for k, v in obj.items()}
        return obj

    out_path = Path(__file__).parent / "benchmark_drought_index.json"
    with open(out_path, "w") as f:
        json.dump(sanitize_nan(benchmark), f, indent=2)
    print(f"\n  Wrote: {out_path}")

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
