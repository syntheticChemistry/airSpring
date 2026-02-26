#!/usr/bin/env python3
"""
Exp 023: Pedotransfer Functions (Saxton & Rawls 2006) — Python Control Baseline

Implements the Saxton & Rawls (2006) pedotransfer functions to estimate soil
hydraulic properties from texture (sand/clay percentages) and organic matter.

Reference:
    Saxton KE, Rawls WJ (2006) "Soil water characteristic estimates by texture
    and organic matter for hydrologic solutions." Soil Sci. Soc. Am. J.
    70(5): 1569-1578.

Equations estimate:
    - Wilting point (θ_wp at -1500 kPa)
    - Field capacity (θ_fc at -33 kPa)
    - Saturation (θ_s = porosity)
    - Saturated hydraulic conductivity (Ksat)

Open data: Uses USDA soil texture class definitions + published regression
    coefficients from Saxton & Rawls (2006).
"""

import json
import math
import os
import sys

# Saxton & Rawls (2006) regression coefficients
# θ_1500 (wilting point) at -1500 kPa
def theta_1500_first(S, C, OM):
    """First estimate of wilting point moisture."""
    return (-0.024 * S + 0.487 * C + 0.006 * OM
            + 0.005 * S * OM - 0.013 * C * OM
            + 0.068 * S * C + 0.031)

def theta_1500(S, C, OM):
    """Wilting point (θ_wp at -1500 kPa)."""
    t1500_first = theta_1500_first(S, C, OM)
    return t1500_first + (0.14 * t1500_first - 0.02)

# θ_33 (field capacity) at -33 kPa
def theta_33_first(S, C, OM):
    """First estimate of field capacity moisture."""
    return (-0.251 * S + 0.195 * C + 0.011 * OM
            + 0.006 * S * OM - 0.027 * C * OM
            + 0.452 * S * C + 0.299)

def theta_33(S, C, OM):
    """Field capacity (θ_fc at -33 kPa)."""
    t33_first = theta_33_first(S, C, OM)
    return t33_first + (1.283 * t33_first * t33_first - 0.374 * t33_first - 0.015)

# θ_s (saturation = porosity) and θ_33_s (saturated moisture at -33 kPa)
def theta_s_33_first(S, C, OM):
    """First estimate of moisture between saturation and field capacity."""
    t33 = theta_33(S, C, OM)
    return (0.278 * S + 0.034 * C + 0.022 * OM
            - 0.018 * S * OM - 0.027 * C * OM
            - 0.584 * S * C + 0.078)

def theta_s_33(S, C, OM):
    """Moisture between saturation and field capacity."""
    first = theta_s_33_first(S, C, OM)
    return first + (0.636 * first - 0.107)

def theta_s(S, C, OM):
    """Saturation moisture (porosity)."""
    return theta_33(S, C, OM) + theta_s_33(S, C, OM) - 0.097 * S + 0.043

# Saturated hydraulic conductivity (Ksat)
def lambda_param(S, C, OM):
    """Slope of ln(moisture) vs ln(tension) curve."""
    t33 = theta_33(S, C, OM)
    t1500 = theta_1500(S, C, OM)
    B = (math.log(1500) - math.log(33)) / (math.log(t33) - math.log(t1500))
    return 1.0 / B

def ksat(S, C, OM):
    """Saturated hydraulic conductivity (mm/hr)."""
    ts = theta_s(S, C, OM)
    t33 = theta_33(S, C, OM)
    lam = lambda_param(S, C, OM)
    return 1930.0 * (ts - t33) ** (3.0 - lam)


# USDA texture class definitions (representative sand/clay/OM)
# sand, clay fractions as proportions (0-1); OM as percentage
USDA_TEXTURES = {
    "sand":        {"S": 0.92, "C": 0.03, "OM": 1.0},
    "loamy_sand":  {"S": 0.82, "C": 0.06, "OM": 1.5},
    "sandy_loam":  {"S": 0.65, "C": 0.10, "OM": 2.0},
    "loam":        {"S": 0.40, "C": 0.20, "OM": 2.5},
    "silt_loam":   {"S": 0.20, "C": 0.15, "OM": 3.0},
    "clay_loam":   {"S": 0.30, "C": 0.35, "OM": 2.0},
    "silty_clay":  {"S": 0.07, "C": 0.45, "OM": 2.0},
    "clay":        {"S": 0.20, "C": 0.55, "OM": 2.0},
}

# Expected ranges from Saxton & Rawls (2006) Table 4
EXPECTED_RANGES = {
    "theta_wp": (0.01, 0.35),    # volumetric, 0-1
    "theta_fc": (0.05, 0.50),    # volumetric, 0-1
    "theta_s":  (0.30, 0.65),    # porosity
    "ksat":     (0.1, 500.0),    # mm/hr
}


def run_validation():
    """Run all Saxton-Rawls pedotransfer validation checks."""
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            print(f"  [{status}] {name}: {detail}")
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 60)
    print("  Exp 023: Pedotransfer (Saxton-Rawls 2006) Validation")
    print("=" * 60)

    # --- Section 1: Analytical tests ---
    print("\n--- Analytical (Loam: S=0.40, C=0.20, OM=2.5%) ---")
    S, C, OM = 0.40, 0.20, 2.5

    wp = theta_1500(S, C, OM)
    fc = theta_33(S, C, OM)
    sat = theta_s(S, C, OM)
    ks = ksat(S, C, OM)

    check("loam_wp_range", 0.08 < wp < 0.20,
          f"θ_wp={wp:.4f}")
    check("loam_fc_range", 0.20 < fc < 0.40,
          f"θ_fc={fc:.4f}")
    check("loam_sat_range", 0.35 < sat < 0.55,
          f"θ_s={sat:.4f}")
    check("loam_ksat_range", 1.0 < ks < 100.0,
          f"Ksat={ks:.2f} mm/hr")

    # Physical constraints
    check("wp_lt_fc", wp < fc, f"θ_wp={wp:.4f} < θ_fc={fc:.4f}")
    check("fc_lt_sat", fc < sat, f"θ_fc={fc:.4f} < θ_s={sat:.4f}")
    check("ksat_positive", ks > 0.0, f"Ksat={ks:.2f}")

    # --- Section 2: Texture class sweep ---
    print("\n--- All USDA Texture Classes ---")

    results = {}
    for tex_name, tex in USDA_TEXTURES.items():
        S, C, OM = tex["S"], tex["C"], tex["OM"]
        wp = theta_1500(S, C, OM)
        fc = theta_33(S, C, OM)
        sat = theta_s(S, C, OM)
        ks = ksat(S, C, OM)
        results[tex_name] = {"wp": wp, "fc": fc, "sat": sat, "ksat": ks}

        # Physical ordering
        check(f"{tex_name}: wp<fc<sat",
              wp < fc < sat,
              f"wp={wp:.3f} fc={fc:.3f} sat={sat:.3f}")

        # Range checks
        check(f"{tex_name}: wp in range",
              EXPECTED_RANGES["theta_wp"][0] < wp < EXPECTED_RANGES["theta_wp"][1],
              f"θ_wp={wp:.4f}")
        check(f"{tex_name}: fc in range",
              EXPECTED_RANGES["theta_fc"][0] < fc < EXPECTED_RANGES["theta_fc"][1],
              f"θ_fc={fc:.4f}")
        check(f"{tex_name}: sat in range",
              EXPECTED_RANGES["theta_s"][0] < sat < EXPECTED_RANGES["theta_s"][1],
              f"θ_s={sat:.4f}")
        check(f"{tex_name}: ksat in range",
              EXPECTED_RANGES["ksat"][0] < ks < EXPECTED_RANGES["ksat"][1],
              f"Ksat={ks:.2f}")

    # --- Section 3: Ordering checks ---
    print("\n--- Physical Ordering Across Textures ---")

    # Sand should have higher Ksat than clay
    check("sand_ksat_gt_clay",
          results["sand"]["ksat"] > results["clay"]["ksat"],
          f"sand={results['sand']['ksat']:.1f} clay={results['clay']['ksat']:.1f}")

    # Clay should have higher WP than sand
    check("clay_wp_gt_sand",
          results["clay"]["wp"] > results["sand"]["wp"],
          f"clay={results['clay']['wp']:.3f} sand={results['sand']['wp']:.3f}")

    # Clay should have higher FC than sand
    check("clay_fc_gt_sand",
          results["clay"]["fc"] > results["sand"]["fc"],
          f"clay={results['clay']['fc']:.3f} sand={results['sand']['fc']:.3f}")

    # --- Section 4: Organic matter sensitivity ---
    print("\n--- Organic Matter Sensitivity ---")

    S, C = 0.40, 0.20
    wp_lo = theta_1500(S, C, 0.5)
    fc_lo = theta_33(S, C, 0.5)
    wp_hi = theta_1500(S, C, 5.0)
    fc_hi = theta_33(S, C, 5.0)

    # Higher OM → higher water retention
    check("om_increases_wp", wp_hi > wp_lo,
          f"OM=0.5: wp={wp_lo:.4f}, OM=5.0: wp={wp_hi:.4f}")
    check("om_increases_fc", fc_hi > fc_lo,
          f"OM=0.5: fc={fc_lo:.4f}, OM=5.0: fc={fc_hi:.4f}")

    # --- Section 5: Available water capacity ---
    print("\n--- Available Water Capacity ---")

    for tex_name, r in results.items():
        awc = r["fc"] - r["wp"]
        check(f"{tex_name}: AWC positive",
              awc > 0.0,
              f"AWC={awc:.4f}")
        check(f"{tex_name}: AWC realistic",
              0.01 < awc < 0.25,
              f"AWC={awc:.4f}")

    # Loam/silt_loam should have highest AWC (known soil science)
    loam_awc = results["loam"]["fc"] - results["loam"]["wp"]
    sand_awc = results["sand"]["fc"] - results["sand"]["wp"]
    check("loam_awc_gt_sand", loam_awc > sand_awc,
          f"loam={loam_awc:.3f} sand={sand_awc:.3f}")

    # --- Section 6: TAW computation ---
    print("\n--- Total Available Water (TAW) ---")

    root_depth_mm = 900.0  # corn
    loam_taw = (results["loam"]["fc"] - results["loam"]["wp"]) * root_depth_mm
    check("loam_taw_realistic",
          50.0 < loam_taw < 250.0,
          f"TAW={loam_taw:.1f} mm for 900mm root depth")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"  Pedotransfer (Saxton-Rawls): {passed}/{total} PASS, {failed}/{total} FAIL")
    print(f"{'=' * 60}")

    return failed == 0


def generate_benchmark():
    """Generate benchmark JSON for Rust validation."""
    all_results = {}
    for tex_name, tex in USDA_TEXTURES.items():
        S, C, OM = tex["S"], tex["C"], tex["OM"]
        wp = theta_1500(S, C, OM)
        fc = theta_33(S, C, OM)
        sat = theta_s(S, C, OM)
        ks = ksat(S, C, OM)
        lam = lambda_param(S, C, OM)
        all_results[tex_name] = {
            "S": S, "C": C, "OM": OM,
            "theta_wp": round(wp, 6),
            "theta_fc": round(fc, 6),
            "theta_s": round(sat, 6),
            "ksat_mm_hr": round(ks, 6),
            "lambda": round(lam, 6),
        }

    # Loam analytical intermediate values
    S, C, OM = 0.40, 0.20, 2.5
    loam_intermediates = {
        "S": S, "C": C, "OM": OM,
        "theta_1500_first": round(theta_1500_first(S, C, OM), 6),
        "theta_1500": round(theta_1500(S, C, OM), 6),
        "theta_33_first": round(theta_33_first(S, C, OM), 6),
        "theta_33": round(theta_33(S, C, OM), 6),
        "theta_s_33_first": round(theta_s_33_first(S, C, OM), 6),
        "theta_s_33": round(theta_s_33(S, C, OM), 6),
        "theta_s": round(theta_s(S, C, OM), 6),
        "lambda": round(lambda_param(S, C, OM), 6),
        "ksat_mm_hr": round(ksat(S, C, OM), 6),
    }

    benchmark = {
        "_provenance": {
            "method": "Saxton & Rawls (2006) pedotransfer functions",
            "baseline_script": "control/pedotransfer/saxton_rawls.py",
            "baseline_commit": "pending",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "created": "2026-02-26",
            "references": [
                "Saxton KE, Rawls WJ (2006) SSSAJ 70(5):1569-1578",
            ],
        },
        "loam_intermediates": loam_intermediates,
        "texture_classes": all_results,
        "tol_moisture": 1e-4,
        "tol_ksat": 0.5,
        "expected_ranges": EXPECTED_RANGES,
        "thresholds": {
            "awc_range": [0.01, 0.25],
            "_tolerance_justification": (
                "Saxton-Rawls regression coefficients have 4 significant digits; "
                "intermediate values may accumulate floating-point differences. "
                "Ksat tolerance is wider due to exponential amplification."
            ),
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_pedotransfer.json")
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n  Benchmark written to {out_path}")

    return benchmark


if __name__ == "__main__":
    benchmark = generate_benchmark()
    success = run_validation()
    sys.exit(0 if success else 1)
