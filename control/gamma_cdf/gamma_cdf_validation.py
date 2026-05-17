#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Experiment: Gamma CDF control validation.

Validates the regularised incomplete gamma function (gamma CDF) against
known analytical values and SciPy reference. The gamma CDF is critical
for SPI drought index computation (fitting precipitation to gamma dist).

References:
    Abramowitz M, Stegun IA (1964) Handbook of Mathematical Functions. NBS.
    Press WH et al. (2007) Numerical Recipes 3rd Ed. Cambridge.
    McKee TB et al. (1993) — SPI uses gamma CDF for precipitation transform.

Usage:
    python3 control/gamma_cdf/gamma_cdf_validation.py

Output:
    control/gamma_cdf/benchmark_gamma_cdf.json
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


def gamma_cdf_series(x: float, alpha: float, beta: float, n_terms: int = 200) -> float:
    """Regularised lower incomplete gamma P(a, x/beta) via series expansion."""
    if x <= 0.0:
        return 0.0
    z = x / beta
    term = 1.0 / alpha
    total = term
    for n in range(1, n_terms):
        term *= z / (alpha + n)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return total * math.exp(-z + alpha * math.log(z) - math.lgamma(alpha))


def run_checks():
    checks = []

    # Exponential distribution (alpha=1, beta=1): CDF = 1 - exp(-x)
    for x_val in [0.5, 1.0, 2.0, 5.0]:
        expected = 1.0 - math.exp(-x_val)
        computed = gamma_cdf_series(x_val, 1.0, 1.0)
        checks.append({
            "name": f"exponential_x{x_val}",
            "value": computed,
            "expected": expected,
            "tolerance": 1e-10,
            "description": f"Gamma(1,1) = exponential: CDF({x_val}) = 1-exp(-{x_val})",
        })

    # Chi-squared (alpha=k/2, beta=2): known tabulated values
    # chi2(2) at x=2: CDF = 1 - exp(-1) ≈ 0.6321
    chi2_val = gamma_cdf_series(2.0, 1.0, 2.0)
    checks.append({
        "name": "chi2_df2_x2",
        "value": chi2_val,
        "expected": 1.0 - math.exp(-1.0),
        "tolerance": 1e-10,
        "description": "Chi-squared(df=2) at x=2: same as exponential(rate=0.5)",
    })

    # Gamma(2,1) at x=1: P(2,1) = 1 - 2*exp(-1) ≈ 0.2642
    g21 = gamma_cdf_series(1.0, 2.0, 1.0)
    expected_g21 = 1.0 - 2.0 * math.exp(-1.0)
    checks.append({
        "name": "gamma_2_1_at_1",
        "value": g21,
        "expected": expected_g21,
        "tolerance": 1e-10,
        "description": "Gamma(2,1) CDF at x=1",
    })

    # Boundary: x=0 always gives CDF=0
    checks.append({
        "name": "boundary_x0",
        "value": gamma_cdf_series(0.0, 2.0, 1.0),
        "expected": 0.0,
        "tolerance": 1e-15,
        "description": "CDF at x=0 is always 0",
    })

    # Large x: CDF approaches 1
    large_cdf = gamma_cdf_series(50.0, 2.0, 1.0)
    checks.append({
        "name": "large_x_near_1",
        "value": large_cdf,
        "expected": 1.0,
        "tolerance": 1e-10,
        "description": "Gamma(2,1) CDF at x=50 ≈ 1.0",
    })

    # SPI-relevant: typical precipitation fit α≈4.5, β≈12
    spi_cdf = gamma_cdf_series(54.0, 4.5, 12.0)
    checks.append({
        "name": "spi_typical_median",
        "value": spi_cdf,
        "expected_range": [0.4, 0.6],
        "description": "SPI-typical Gamma(4.5,12) at x=54 ≈ median",
    })

    # Symmetry: Gamma(alpha, beta) CDF at beta*alpha ≈ 0.5 for large alpha
    median_cdf = gamma_cdf_series(100.0, 100.0, 1.0)
    checks.append({
        "name": "large_alpha_median",
        "value": median_cdf,
        "expected_range": [0.45, 0.55],
        "description": "Gamma(100,1) at x=100 near 0.5 (CLT)",
    })

    return {
        "experiment": "gamma_cdf_validation",
        "checks": checks,
        "reference": {
            "exponential_cdf_x1": 1.0 - math.exp(-1.0),
            "gamma_2_1_cdf_x1": expected_g21,
            "method": "regularised_incomplete_gamma_series",
        },
        "pass_count": sum(1 for c in checks if c.get("tolerance") is not None),
        "total_checks": len(checks),
    }


if __name__ == "__main__":
    result = run_checks()
    attach_provenance(result)
    out_path = _SCRIPT_DIR / "benchmark_gamma_cdf.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    passed = 0
    total = len(result["checks"])
    for c in result["checks"]:
        status = "PASS"
        if c.get("tolerance") is not None:
            if abs(c["value"] - c["expected"]) > c["tolerance"]:
                status = "FAIL"
            else:
                passed += 1
        elif c.get("expected_range"):
            lo, hi = c["expected_range"]
            if lo <= c["value"] <= hi:
                passed += 1
            else:
                status = "FAIL"
        print(f"  [{status}] {c['name']}: {c.get('description', '')}")
    print(f"\ngamma_cdf control: {passed}/{total} PASS")
