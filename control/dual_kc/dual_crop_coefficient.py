#!/usr/bin/env python3
"""
airSpring Experiment 009 — FAO-56 Chapter 7: Dual Crop Coefficient (Kcb + Ke)

Implements the dual crop coefficient approach from FAO Irrigation and Drainage
Paper No. 56 (Allen et al. 1998), Chapter 7. The dual Kc separates crop
evapotranspiration into transpiration (Kcb) and soil evaporation (Ke):

    ETc = (Kcb × Ks + Ke) × ET₀     (Eq. 69)

This is a direct extension of Experiment 001 (single Kc ET₀) and Experiment 004
(water balance scheduling) — combining the precision of separate transpiration
and evaporation tracking.

Reference:
    Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
    FAO Irrigation and Drainage Paper 56, Chapter 7.
    https://www.fao.org/4/X0490E/x0490e00.htm
"""

import json
import math
import sys
from pathlib import Path

# ── FAO-56 Chapter 7 equations ────────────────────────────────────────

def etc_dual(kcb: float, ks: float, ke: float, et0: float) -> float:
    """FAO-56 Eq. 69: ETc = (Kcb × Ks + Ke) × ET₀"""
    return (kcb * ks + ke) * et0


def kc_max(u2: float, rh_min: float, h: float, kcb: float) -> float:
    """FAO-56 Eq. 72: upper limit on Kc (= Kcb + Ke).

    Kc_max = max(
        1.2 + [0.04(u2 - 2) - 0.004(RHmin - 45)] × (h/3)^0.3,
        Kcb + 0.05
    )

    Parameters
    ----------
    u2 : wind speed at 2 m (m/s)
    rh_min : minimum relative humidity (%)
    h : crop height (m)
    kcb : basal crop coefficient
    """
    h_clamp = max(h, 0.001)
    climate_term = 1.2 + (0.04 * (u2 - 2.0) - 0.004 * (rh_min - 45.0)) * (h_clamp / 3.0) ** 0.3
    return max(climate_term, kcb + 0.05)


def total_evaporable_water(theta_fc: float, theta_wp: float, ze: float) -> float:
    """FAO-56 Eq. 73: TEW = 1000 × (θFC - 0.5×θWP) × Ze (mm)."""
    return 1000.0 * (theta_fc - 0.5 * theta_wp) * ze


def evaporation_reduction(tew: float, rew: float, de: float) -> float:
    """FAO-56 Eq. 72: Kr — evaporation reduction coefficient.

    Kr = 1.0                          when De ≤ REW  (stage 1)
    Kr = (TEW - De) / (TEW - REW)    when De > REW  (stage 2)
    """
    if de <= rew:
        return 1.0
    if tew <= rew:
        return 0.0
    kr = (tew - de) / (tew - rew)
    return max(0.0, min(1.0, kr))


def soil_evaporation_ke(kr: float, kcb: float, kc_max_val: float, few: float) -> float:
    """FAO-56 Eq. 71: Ke = min(Kr × (Kc_max - Kcb), few × Kc_max).

    Ke represents soil evaporation. Bounded by the energy available
    for evaporation (Kc_max - Kcb) and the wetted fraction (few).
    """
    ke = kr * (kc_max_val - kcb)
    ke = min(ke, few * kc_max_val)
    return max(0.0, ke)


def evaporation_layer_balance(
    de_prev: float,
    precip: float,
    irrig: float,
    ke: float,
    et0: float,
    few: float,
    tew: float,
) -> float:
    """FAO-56 Eq. 77: Daily water balance for evaporation layer.

    De,i = De,i-1 - P_i - I_i + (Ke × ET₀) / few + DPe,i

    Simplified: ignore transpiration from evap layer (Tew) and deep
    percolation from evap layer (DPe). Clamp De to [0, TEW].
    """
    evap = ke * et0 / max(few, 0.001) if few > 0.001 else 0.0
    de_new = de_prev - precip - irrig + evap
    return max(0.0, min(tew, de_new))


def simulate_dual_kc(
    et0_daily: list,
    precip_daily: list,
    kcb: float,
    kc_max_val: float,
    few: float,
    tew: float,
    rew: float,
    de_init: float = 0.0,
) -> dict:
    """Simulate multi-day dual Kc evaporation layer water balance.

    Returns dict with daily arrays: de, kr, ke, etc_dual, and summary stats.
    """
    n = len(et0_daily)
    de_arr = []
    kr_arr = []
    ke_arr = []
    etc_arr = []
    de = de_init

    for i in range(n):
        # Wetting from precipitation resets depletion
        de = max(0.0, de - precip_daily[i])
        de = min(tew, de)

        kr = evaporation_reduction(tew, rew, de)
        ke = soil_evaporation_ke(kr, kcb, kc_max_val, few)
        etc = etc_dual(kcb, 1.0, ke, et0_daily[i])

        de_arr.append(round(de, 4))
        kr_arr.append(round(kr, 4))
        ke_arr.append(round(ke, 4))
        etc_arr.append(round(etc, 4))

        de = evaporation_layer_balance(de, 0.0, 0.0, ke, et0_daily[i], few, tew)

    return {
        "de": de_arr,
        "kr": kr_arr,
        "ke": ke_arr,
        "etc": etc_arr,
        "total_etc": round(sum(etc_arr), 4),
        "final_de": round(de, 4),
    }


# ── Validation framework ─────────────────────────────────────────────

class Validator:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label: str, observed: float, expected: float, tol: float = 0.01):
        diff = abs(observed - expected)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}: observed={observed:.6f} expected={expected:.6f} diff={diff:.6f} tol={tol}")

    def check_bool(self, label: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {label}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  TOTAL: {self.passed}/{total} PASS")
        if self.failed > 0:
            print(f"  *** {self.failed} FAILED ***")
        print(f"{'='*60}")
        return self.failed == 0


# ── Main validation ──────────────────────────────────────────────────

def main():
    benchmark_path = Path(__file__).parent / "benchmark_dual_kc.json"
    with open(benchmark_path) as f:
        bench = json.load(f)

    v = Validator()

    # ── Section 1: ETc dual (Eq. 69) ──────────────────────────────
    print("\n── Eq. 69: ETc = (Kcb × Ks + Ke) × ET₀ ──")
    for tc in bench["equations"]["eq_69"]["test_cases"]:
        result = etc_dual(tc["kcb"], tc["ks"], tc["ke"], tc["et0"])
        v.check(tc["label"], result, tc["expected_etc"])

    # ── Section 2: Kc_max (Eq. 71/72) ────────────────────────────
    print("\n── Eq. 72: Kc_max ──")
    for tc in bench["equations"]["eq_71_kc_max"]["test_cases"]:
        result = kc_max(tc["u2"], tc["rh_min"], tc["h"], tc["kcb"])
        v.check(tc["label"], result, tc["expected_kc_max"])

    # ── Section 3: TEW (Eq. 73) ──────────────────────────────────
    print("\n── Eq. 73: TEW = 1000 × (θFC - 0.5×θWP) × Ze ──")
    for tc in bench["equations"]["eq_73_tew"]["test_cases"]:
        result = total_evaporable_water(tc["theta_fc"], tc["theta_wp"], tc["ze_m"])
        v.check(tc["label"], result, tc["expected_tew"])

    # ── Section 4: Kr (Eq. 72) ───────────────────────────────────
    print("\n── Eq. 72: Kr evaporation reduction ──")
    for tc in bench["equations"]["eq_72_kr"]["test_cases"]:
        result = evaporation_reduction(tc["tew"], tc["rew"], tc["de"])
        v.check(tc["label"], result, tc["expected_kr"])

    # ── Section 5: Kcb vs Kc consistency ─────────────────────────
    print("\n── Table 17 vs Table 12: Kcb + evaporation ≈ Kc ──")
    kcb_table = bench["table_17_kcb"]["crops"]
    kc_table = bench["table_12_kc_single"]["crops"]
    for crop_name in kcb_table:
        if crop_name not in kc_table:
            continue
        kcb_mid = kcb_table[crop_name]["kcb_mid"]
        kc_mid = kc_table[crop_name]["kc_mid"]
        diff = kc_mid - kcb_mid
        v.check_bool(
            f"{crop_name}: Kc_mid ({kc_mid}) - Kcb_mid ({kcb_mid}) = {diff:.2f} in [0, 0.20]",
            0.0 <= diff <= 0.20,
        )

    # ── Section 6: Kcb ordering invariants ───────────────────────
    print("\n── Table 17: Kcb ordering invariants ──")
    for crop_name, crop in kcb_table.items():
        v.check_bool(
            f"{crop_name}: Kcb_ini ({crop['kcb_ini']}) < Kcb_mid ({crop['kcb_mid']})",
            crop["kcb_ini"] < crop["kcb_mid"],
        )

    # ── Section 7: TEW > REW for all soils (Table 19) ────────────
    print("\n── Table 19: TEW > REW for all soil types ──")
    soils = bench["table_19_rew"]["soils"]
    for soil_name, soil in soils.items():
        tew = total_evaporable_water(soil["theta_fc"], soil["theta_wp"], 0.10)
        rew = soil["rew_mm"]
        v.check_bool(
            f"{soil_name}: TEW ({tew:.1f} mm) > REW ({rew} mm)",
            tew > rew,
        )

    # ── Section 8: Ke boundaries ─────────────────────────────────
    print("\n── Ke boundary checks ──")
    ke_dry = soil_evaporation_ke(kr=0.0, kcb=1.15, kc_max_val=1.20, few=1.0)
    v.check("Ke=0 when Kr=0 (dry surface)", ke_dry, 0.0, tol=1e-10)

    ke_wet = soil_evaporation_ke(kr=1.0, kcb=0.15, kc_max_val=1.20, few=1.0)
    v.check("Ke=1.05 for bare wet soil (Kc_max-Kcb)", ke_wet, 1.05)

    ke_limited = soil_evaporation_ke(kr=1.0, kcb=0.15, kc_max_val=1.20, few=0.3)
    v.check("Ke limited by few×Kc_max", ke_limited, 0.36)

    ke_full_cover = soil_evaporation_ke(kr=1.0, kcb=1.15, kc_max_val=1.20, few=0.05)
    v.check("Ke small under full cover (Kc_max-Kcb=0.05)", ke_full_cover, 0.05)

    # ── Section 9: Bare soil drydown simulation ──────────────────
    print("\n── Scenario: Bare soil drydown (7 days) ──")
    scenario = bench["validation_scenarios"]["bare_soil_drydown"]
    result = simulate_dual_kc(
        et0_daily=scenario["et0_daily"],
        precip_daily=scenario["precip_daily"],
        kcb=scenario["kcb"],
        kc_max_val=scenario["kc_max"],
        few=scenario["few"],
        tew=scenario["tew"],
        rew=scenario["rew"],
    )
    v.check_bool(
        f"Day 1 Kr=1.0 (stage 1, after rain): Kr={result['kr'][0]}",
        result["kr"][0] == 1.0,
    )
    v.check_bool(
        f"Kr declines over drydown: Kr day1={result['kr'][0]} >= Kr day7={result['kr'][-1]}",
        result["kr"][0] >= result["kr"][-1],
    )
    v.check_bool(
        f"De increases over drydown: De day1={result['de'][0]} <= De day7={result['de'][-1]}",
        result["de"][0] <= result["de"][-1],
    )
    v.check_bool(
        f"Ke declines over drydown: Ke day1={result['ke'][0]} >= Ke day7={result['ke'][-1]}",
        result["ke"][0] >= result["ke"][-1],
    )
    v.check_bool(
        f"Total ETc > 0: {result['total_etc']:.2f} mm",
        result["total_etc"] > 0,
    )
    v.check_bool(
        f"Final De <= TEW: {result['final_de']:.2f} <= {scenario['tew']}",
        result["final_de"] <= scenario["tew"],
    )
    print(f"  Simulation detail — De: {result['de']}")
    print(f"  Simulation detail — Kr: {result['kr']}")
    print(f"  Simulation detail — Ke: {result['ke']}")

    # ── Section 10: Mid-season corn (minimal evaporation) ────────
    print("\n── Scenario: Corn mid-season (5 days, full cover) ──")
    scenario_corn = bench["validation_scenarios"]["corn_mid_season"]
    result_corn = simulate_dual_kc(
        et0_daily=scenario_corn["et0_daily"],
        precip_daily=scenario_corn["precip_daily"],
        kcb=scenario_corn["kcb"],
        kc_max_val=scenario_corn["kc_max"],
        few=scenario_corn["few"],
        tew=scenario_corn["tew"],
        rew=scenario_corn["rew"],
    )
    for i, (etc_val, et0_val) in enumerate(zip(result_corn["etc"], scenario_corn["et0_daily"])):
        ratio = etc_val / et0_val if et0_val > 0 else 0
        v.check_bool(
            f"Day {i+1}: ETc/ET₀ ratio ({ratio:.3f}) close to Kcb ({scenario_corn['kcb']})",
            abs(ratio - scenario_corn["kcb"]) < 0.10,
        )

    # ── Summary ──────────────────────────────────────────────────
    ok = v.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
