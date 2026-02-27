#!/usr/bin/env python3
"""Exp 036 — biomeOS Neural API Round-Trip Parity Control.

Validates that the concept of orchestrated compute via JSON-RPC produces
results identical to direct function calls. This Python control simulates
the round-trip by computing ET₀ via direct calls and via JSON serialization
/ deserialization, confirming that the data path preserves numerical fidelity.

The actual Neural API integration is tested in the Rust binary
(validate_neural_api). This Python control validates the JSON data format
and serialization parity.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(SCRIPT_DIR, "benchmark_neural_api.json")


def penman_monteith_et0(tmin, tmax, tmean, solar_rad, wind_2m, e_a, elev, lat_deg, doy):
    """FAO-56 Penman-Monteith daily ET₀ (mm/day)."""
    P = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    e_s = (0.6108 * math.exp(17.27 * tmax / (tmax + 237.3)) +
           0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))) / 2.0

    lat_rad = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    dec = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(dec))
    ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat_rad) * math.sin(dec) +
        math.cos(lat_rad) * math.cos(dec) * math.sin(ws)
    )
    rso = (0.75 + 2e-5 * elev) * ra
    rns = (1.0 - 0.23) * solar_rad
    rnl = (4.903e-9 *
           ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0 *
           (0.34 - 0.14 * math.sqrt(e_a)) *
           (1.35 * (solar_rad / rso if rso > 0 else 0) - 0.35))
    rn = rns - rnl

    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * wind_2m * (e_s - e_a)
    den = delta + gamma * (1.0 + 0.34 * wind_2m)
    return max(0.0, num / den)


def hargreaves_et0(tmin, tmax, ra_mj):
    """Hargreaves-Samani (1985) ET₀ (mm/day)."""
    tmean = (tmin + tmax) / 2.0
    return max(0.0, 0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * ra_mj)


def makkink_et0(tmean, rs_mj, elevation_m):
    """Makkink (1957) with de Bruin (1987) coefficients (mm/day)."""
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * P
    delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    return max(0.0, 0.61 * (delta / (delta + gamma)) * rs_mj / 2.45 - 0.12)


def turc_et0(tmean, rs_mj, rh):
    """Turc (1961) ET₀ (mm/day)."""
    if tmean <= 0.0:
        return 0.0
    rs_cal = rs_mj * 23.8846
    base = 0.013 * (tmean / (tmean + 15.0)) * (rs_cal + 50.0)
    if rh < 50.0:
        base *= 1.0 + (50.0 - rh) / 70.0
    return max(0.0, base)


def hamon_pet(tmean, day_length_hours):
    """Hamon (1961) PET (mm/day), Lu et al. (2005) formulation."""
    if tmean <= 0.0 or day_length_hours <= 0.0:
        return 0.0
    e_s = 0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))
    rho_sat = 216.7 * e_s / (tmean + 273.3)
    return max(0.0, 0.1651 * (day_length_hours / 12.0) * rho_sat)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def json_round_trip(value):
    """Simulate JSON serialization round-trip (as Neural API would do)."""
    return json.loads(json.dumps(value))


def validate_serialization_parity(benchmark):
    """Verify that JSON round-trip preserves numerical values."""
    print("\n[Serialization Parity]")
    tests = benchmark["validation_checks"]["et0_round_trip"]["test_cases"]
    passed, total = 0, 0
    for tc in tests:
        method = tc["method"]
        params = tc["params"]
        tol = tc["tolerance"]
        total += 1

        if method == "pm":
            direct = penman_monteith_et0(
                params["tmin"], params["tmax"], params["tmean"],
                params["solar_radiation"], params["wind_speed_2m"],
                params["actual_vapour_pressure"], params["elevation_m"],
                params["latitude_deg"], params["day_of_year"],
            )
        elif method == "hargreaves":
            direct = hargreaves_et0(params["tmin"], params["tmax"], params["ra_mj"])
        elif method == "makkink":
            direct = makkink_et0(params["tmean"], params["rs_mj"], params["elevation_m"])
        elif method == "turc":
            direct = turc_et0(params["tmean"], params["rs_mj"], params["rh"])
        elif method == "hamon":
            direct = hamon_pet(params["tmean"], params["day_length_hours"])
        else:
            print(f"  SKIP {method}: unknown method")
            continue

        rt_params = json_round_trip(params)
        if method == "pm":
            via_json = penman_monteith_et0(
                rt_params["tmin"], rt_params["tmax"], rt_params["tmean"],
                rt_params["solar_radiation"], rt_params["wind_speed_2m"],
                rt_params["actual_vapour_pressure"], rt_params["elevation_m"],
                rt_params["latitude_deg"], rt_params["day_of_year"],
            )
        elif method == "hargreaves":
            via_json = hargreaves_et0(rt_params["tmin"], rt_params["tmax"], rt_params["ra_mj"])
        elif method == "makkink":
            via_json = makkink_et0(rt_params["tmean"], rt_params["rs_mj"], rt_params["elevation_m"])
        elif method == "turc":
            via_json = turc_et0(rt_params["tmean"], rt_params["rs_mj"], rt_params["rh"])
        elif method == "hamon":
            via_json = hamon_pet(rt_params["tmean"], rt_params["day_length_hours"])

        diff = abs(direct - via_json)
        ok = diff < tol
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {method}: direct={direct:.10f} via_json={via_json:.10f} diff={diff:.2e}")
        if ok:
            passed += 1

    return passed, total


def validate_json_format(benchmark):
    """Verify benchmark JSON structure matches capability spec."""
    print("\n[JSON Format Validation]")
    passed, total = 0, 0
    checks = benchmark["validation_checks"]

    total += 1
    assert "et0_round_trip" in checks, "missing et0_round_trip"
    print("  PASS et0_round_trip section present")
    passed += 1

    total += 1
    assert "capability_discovery" in checks, "missing capability_discovery"
    caps = checks["capability_discovery"]["expected_capabilities"]
    print(f"  PASS capability_discovery: {len(caps)} capabilities listed")
    passed += 1

    total += 1
    assert "health_check" in checks, "missing health_check"
    print("  PASS health_check section present")
    passed += 1

    total += 1
    assert "substrate_detection" in checks, "missing substrate_detection"
    sub = checks["substrate_detection"]
    assert sub["expected_kind"] == "Neural"
    print(f"  PASS substrate_detection: kind={sub['expected_kind']}, caps={sub['expected_capabilities']}")
    passed += 1

    return passed, total


def validate_method_coverage(benchmark):
    """Verify all 7 ET₀ methods have round-trip test cases."""
    print("\n[Method Coverage]")
    tests = benchmark["validation_checks"]["et0_round_trip"]["test_cases"]
    methods = {tc["method"] for tc in tests}
    expected = {"pm", "hargreaves", "makkink", "turc", "hamon"}
    passed, total = 0, 0

    for m in sorted(expected):
        total += 1
        if m in methods:
            print(f"  PASS {m} has round-trip test case")
            passed += 1
        else:
            print(f"  FAIL {m} missing round-trip test case")

    return passed, total


def main():
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    print(f"Exp 036: biomeOS Neural API Round-Trip Parity")
    print(f"Benchmark: {BENCHMARK}")

    total_pass, total_tests = 0, 0

    p, t = validate_serialization_parity(benchmark)
    total_pass += p
    total_tests += t

    p, t = validate_json_format(benchmark)
    total_pass += p
    total_tests += t

    p, t = validate_method_coverage(benchmark)
    total_pass += p
    total_tests += t

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass}/{total_tests} PASS")
    if total_pass == total_tests:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
