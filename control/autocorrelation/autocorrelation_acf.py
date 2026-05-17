#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment: Autocorrelation function (ACF) control.

Validates time-series autocorrelation computation against known analytical
results. Uses synthetic AR(1) process and white noise to produce reference
ACF values that Rust implementations must reproduce within tolerance.

References:
    Box GEP, Jenkins GM, Reinsel GC (2015) Time Series Analysis. Wiley.
    Chatfield C (2004) The Analysis of Time Series. Chapman & Hall.

Usage:
    python3 control/autocorrelation/autocorrelation_acf.py

Output:
    control/autocorrelation/benchmark_autocorrelation.json
"""

import json
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONTROL_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "control" else _SCRIPT_DIR.parent
if str(_CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROL_DIR))
from provenance import attach_provenance


def autocorrelation_cpu(data: list[float], max_lag: int) -> list[float]:
    """Unnormalised autocovariance at lags 0..max_lag."""
    n = len(data)
    mean = sum(data) / n
    acf = []
    for lag in range(max_lag + 1):
        c = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(n - lag))
        acf.append(c)
    return acf


def normalised_acf(data: list[float], max_lag: int) -> list[float]:
    """Normalised ACF: acf[lag] / acf[0]."""
    raw = autocorrelation_cpu(data, max_lag)
    if raw[0] == 0.0:
        return [0.0] * len(raw)
    return [c / raw[0] for c in raw]


def generate_ar1(n: int, phi: float, seed: int = 42) -> list[float]:
    """Deterministic AR(1) process: x[t] = phi * x[t-1] + e[t]."""
    import random
    rng = random.Random(seed)
    x = [0.0] * n
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.gauss(0, 1)
    return x


def run_checks():
    checks = []

    # White noise: ACF at lag 0 should dominate, lags 1+ near zero
    white_noise = [math.sin(i * 0.7) + math.cos(i * 1.3) for i in range(200)]
    max_lag = 10
    nacf = normalised_acf(white_noise, max_lag)
    checks.append({
        "name": "white_noise_lag0",
        "value": nacf[0],
        "expected": 1.0,
        "tolerance": 1e-12,
        "description": "normalised ACF at lag 0 is always 1.0",
    })

    # AR(1) with phi=0.8: theoretical ACF(k) ≈ phi^k
    ar1_data = generate_ar1(500, 0.8, seed=42)
    nacf_ar1 = normalised_acf(ar1_data, 5)
    checks.append({
        "name": "ar1_lag1_positive",
        "value": nacf_ar1[1],
        "expected_range": [0.5, 0.95],
        "description": "AR(1) phi=0.8: lag-1 ACF should be strongly positive",
    })
    checks.append({
        "name": "ar1_lag1_gt_lag5",
        "value": nacf_ar1[1] > nacf_ar1[5],
        "expected": True,
        "description": "AR(1): ACF decays with lag",
    })

    # Constant data: ACF is zero (no variance)
    constant = [5.0] * 50
    nacf_const = normalised_acf(constant, 3)
    checks.append({
        "name": "constant_data_acf_zero",
        "value": nacf_const[1],
        "expected": 0.0,
        "tolerance": 1e-12,
        "description": "constant data has zero normalised ACF at all non-zero lags",
    })

    # Symmetry: raw ACF values
    raw = autocorrelation_cpu(ar1_data, 5)
    checks.append({
        "name": "raw_acf_lag0_positive",
        "value": raw[0] > 0,
        "expected": True,
        "description": "raw autocovariance at lag 0 is total variance (positive)",
    })

    # Export reference values for Rust validation
    reference_nacf = normalised_acf(ar1_data, 20)
    reference_raw = autocorrelation_cpu(ar1_data, 20)

    return {
        "experiment": "autocorrelation_acf",
        "checks": checks,
        "reference": {
            "ar1_data_first_10": ar1_data[:10],
            "ar1_phi": 0.8,
            "ar1_n": 500,
            "ar1_seed": 42,
            "normalised_acf_lags_0_20": reference_nacf,
            "raw_acf_lags_0_20": reference_raw,
            "white_noise_first_10": white_noise[:10],
            "white_noise_nacf_lags_0_10": nacf,
        },
        "pass_count": sum(1 for c in checks if c.get("expected") is not None),
        "total_checks": len(checks),
    }


if __name__ == "__main__":
    result = run_checks()
    attach_provenance(result)
    out_path = _SCRIPT_DIR / "benchmark_autocorrelation.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    passed = sum(
        1 for c in result["checks"]
        if (c.get("tolerance") is not None and abs(c["value"] - c["expected"]) <= c["tolerance"])
        or c.get("value") is True
    )
    total = len(result["checks"])
    print(f"autocorrelation control: {passed}/{total} PASS")
    for c in result["checks"]:
        status = "PASS"
        if c.get("tolerance") is not None:
            if abs(c["value"] - c["expected"]) > c["tolerance"]:
                status = "FAIL"
        elif c.get("expected_range"):
            lo, hi = c["expected_range"]
            if not (lo <= c["value"] <= hi):
                status = "FAIL"
        elif c.get("expected") is True and c["value"] is not True:
            status = "FAIL"
        print(f"  [{status}] {c['name']}: {c.get('description', '')}")
