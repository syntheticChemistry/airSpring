#!/usr/bin/env python3
"""
airSpring Experiment 011 — Dual Kc for Cover Crops + No-Till Mulch Effects

Extends Experiment 009 (dual Kc) with cover crop coefficients and the
no-till mulch reduction factor from FAO-56 Chapter 11. Validates:

  1. Cover crop Kcb values are physically reasonable
  2. No-till mulch reduces soil evaporation (Ke) proportionally
  3. Rye→corn transition scenario shows realistic seasonal ET pattern
  4. No-till system conserves water vs conventional tillage

This connects to baseCamp Sub-thesis 06 (no-till soil moisture + Anderson
geometry) via the soil moisture coupling pathway.

References:
    Allen et al. (1998) FAO-56 Ch 7 + Ch 11
    Islam & Reeder (2014) ISWCR 2(3): 176-186
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dual_crop_coefficient import (
    etc_dual,
    kc_max,
    total_evaporable_water,
    evaporation_reduction,
    soil_evaporation_ke,
    evaporation_layer_balance,
    simulate_dual_kc,
)


def mulched_ke(kr, kcb, kc_max_val, few, mulch_factor):
    """Ke with mulch reduction: Ke_mulch = Ke × mulch_factor."""
    ke = soil_evaporation_ke(kr, kcb, kc_max_val, few)
    return ke * mulch_factor


def simulate_dual_kc_mulched(et0_daily, precip_daily, kcb, kc_max_val, few,
                              tew, rew, mulch_factor, de_init=0.0):
    """Multi-day dual Kc with mulch reduction on Ke."""
    n = len(et0_daily)
    de_arr, kr_arr, ke_arr, etc_arr = [], [], [], []
    de = de_init

    for i in range(n):
        de = max(0.0, de - precip_daily[i])
        de = min(tew, de)

        kr = evaporation_reduction(tew, rew, de)
        ke_raw = soil_evaporation_ke(kr, kcb, kc_max_val, few)
        ke = ke_raw * mulch_factor
        etc = etc_dual(kcb, 1.0, ke, et0_daily[i])

        de_arr.append(round(de, 4))
        kr_arr.append(round(kr, 4))
        ke_arr.append(round(ke, 4))
        etc_arr.append(round(etc, 4))

        de = evaporation_layer_balance(de, 0.0, 0.0, ke, et0_daily[i], few, tew)

    return {
        "de": de_arr, "kr": kr_arr, "ke": ke_arr, "etc": etc_arr,
        "total_etc": round(sum(etc_arr), 4),
        "final_de": round(de, 4),
    }


class Validator:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label, observed, expected, tol=0.01):
        diff = abs(observed - expected)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}: observed={observed:.6f} expected={expected:.6f} tol={tol}")

    def check_bool(self, label, condition):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}")

    def check_range(self, label, value, lo, hi):
        ok = lo <= value <= hi
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}: {value:.4f} in [{lo}, {hi}]")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  TOTAL: {self.passed}/{total} PASS")
        if self.failed > 0:
            print(f"  *** {self.failed} FAILED ***")
        print(f"{'='*60}")
        return self.failed == 0


def main():
    bench_path = Path(__file__).parent / "benchmark_cover_crop_kc.json"
    with open(bench_path) as f:
        bench = json.load(f)

    v = Validator()

    # ── Section 1: Cover crop Kcb values ────────────────────────
    print("\n── Cover crop Kcb physical reasonableness ──")
    cc_table = bench["cover_crop_kcb"]["crops"]
    for name, cc in cc_table.items():
        v.check_bool(
            f"{name}: Kcb_ini ({cc['kcb_ini']}) < Kcb_mid ({cc['kcb_mid']})",
            cc["kcb_ini"] < cc["kcb_mid"],
        )
        v.check_range(f"{name}: Kcb_mid in [0.5, 1.3]", cc["kcb_mid"], 0.5, 1.3)
        v.check_bool(
            f"{name}: max_height > 0",
            cc["max_height_m"] > 0,
        )

    # ── Section 2: Mulch reduces evaporation ────────────────────
    print("\n── Mulch reduces Ke proportionally ──")
    for tc in bench["validation_checks"]["mulch_reduces_evaporation"]["test_cases"]:
        ke_raw = soil_evaporation_ke(tc["kr"], tc["kcb"], tc["kc_max"], tc["few"])
        ke_m = ke_raw * tc["mulch_factor"]
        v.check(tc["label"], ke_m, tc["expected_ke"])

    # ── Section 3: Mulch factor ordering ────────────────────────
    print("\n── Mulch factor ordering ──")
    residues = bench["no_till_mulch_effects"]["residue_fractions"]
    prev_mf = 1.1
    for name, rf in residues.items():
        v.check_bool(
            f"{name}: mulch_factor ({rf['mulch_factor']}) <= prev ({prev_mf:.2f})",
            rf["mulch_factor"] <= prev_mf,
        )
        prev_mf = rf["mulch_factor"]

    # ── Section 4: No-till soil moisture observations ───────────
    print("\n── No-till soil observations (Islam et al. 2014) ──")
    obs = bench["no_till_soil_moisture"]["observations"]
    v.check_bool(
        f"SOC higher in no-till ({obs['soil_organic_carbon_pct']['no_till']}%) "
        f"vs conventional ({obs['soil_organic_carbon_pct']['conventional']}%)",
        obs["soil_organic_carbon_pct"]["no_till"] > obs["soil_organic_carbon_pct"]["conventional"],
    )
    v.check_bool(
        f"Bulk density lower in no-till ({obs['bulk_density_g_cm3']['no_till']}) "
        f"vs conventional ({obs['bulk_density_g_cm3']['conventional']})",
        obs["bulk_density_g_cm3"]["no_till"] < obs["bulk_density_g_cm3"]["conventional"],
    )
    v.check_bool(
        f"Infiltration higher in no-till ({obs['infiltration_rate_mm_hr']['no_till']} mm/hr) "
        f"vs conventional ({obs['infiltration_rate_mm_hr']['conventional']} mm/hr)",
        obs["infiltration_rate_mm_hr"]["no_till"] > obs["infiltration_rate_mm_hr"]["conventional"],
    )
    v.check_bool(
        f"AWC higher in no-till ({obs['available_water_capacity_mm']['no_till']} mm) "
        f"vs conventional ({obs['available_water_capacity_mm']['conventional']} mm)",
        obs["available_water_capacity_mm"]["no_till"] > obs["available_water_capacity_mm"]["conventional"],
    )

    # ── Section 5: Rye→corn transition simulation ───────────────
    print("\n── Rye→corn no-till transition (7-day simulation) ──")
    et0_7day = [4.0, 4.5, 4.2, 5.0, 5.5, 5.0, 4.8]
    precip_7day = [10.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0]

    result_conv = simulate_dual_kc(
        et0_7day, precip_7day, kcb=0.15, kc_max_val=1.20,
        few=1.0, tew=22.5, rew=9.0,
    )
    result_notill = simulate_dual_kc_mulched(
        et0_7day, precip_7day, kcb=0.15, kc_max_val=1.20,
        few=1.0, tew=22.5, rew=9.0, mulch_factor=0.40,
    )

    v.check_bool(
        f"No-till total ETc ({result_notill['total_etc']:.2f}) < "
        f"conventional ({result_conv['total_etc']:.2f})",
        result_notill["total_etc"] < result_conv["total_etc"],
    )

    et_savings_pct = 100 * (1 - result_notill["total_etc"] / result_conv["total_etc"])
    expected_range = bench["validation_checks"]["no_till_conserves_water"]["expected_et_reduction_pct"]
    v.check_range(
        f"ET savings {et_savings_pct:.1f}% in [{expected_range['min']}, {expected_range['max']}]%",
        et_savings_pct,
        expected_range["min"],
        expected_range["max"],
    )

    # No-till should retain more moisture (higher De means less evaporated)
    # Actually — lower De means MORE water in the evap layer (less depleted)
    v.check_bool(
        f"No-till final De ({result_notill['final_de']:.2f}) < "
        f"conventional ({result_conv['final_de']:.2f})",
        result_notill["final_de"] < result_conv["final_de"],
    )

    print(f"\n  Conventional: total ETc = {result_conv['total_etc']:.2f} mm")
    print(f"  No-till:      total ETc = {result_notill['total_etc']:.2f} mm")
    print(f"  ET savings:   {et_savings_pct:.1f}%")

    # ── Section 6: Multi-phase transition ───────────────────────
    print("\n── Rye→corn transition phases ──")
    phases = bench["transition_scenarios"]["rye_to_corn"]["phases"]
    prev_kcb = None
    for phase in phases:
        kcb_val = phase["kcb"]
        mf = phase["mulch_factor"]
        v.check_range(
            f"{phase['period']}: Kcb={kcb_val} in [0, 1.5]",
            kcb_val, 0.0, 1.5,
        )
        v.check_range(
            f"{phase['period']}: mulch_factor={mf} in [0, 1]",
            mf, 0.0, 1.0,
        )

    ok = v.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
