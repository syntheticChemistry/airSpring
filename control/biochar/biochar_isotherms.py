#!/usr/bin/env python3
"""
airSpring Experiment 007 — Biochar Adsorption Isotherms Python Baseline

Implements Langmuir and Freundlich isotherm fitting for phosphorus adsorption
on biochar, validating against Kumari, Dong & Safferman (2025) "Phosphorus
adsorption and recovery from waste streams using biochar" Applied Water
Science 15(7):162.

Models:
  Langmuir:  qe = qmax * KL * Ce / (1 + KL * Ce)
  Freundlich: qe = KF * Ce^(1/n)

Uses scipy.optimize.curve_fit for nonlinear least squares fitting.

Provenance:
  Baseline commit: 3afc229
  Benchmark output: control/biochar/benchmark_biochar.json
  Reproduction: python control/biochar/biochar_isotherms.py
  Created: 2026-02-25
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


# ── Isotherm models ──────────────────────────────────────────────────────

def langmuir(Ce: np.ndarray, qmax: float, KL: float) -> np.ndarray:
    """Langmuir: qe = qmax * KL * Ce / (1 + KL * Ce)"""
    return qmax * KL * Ce / (1.0 + KL * Ce)


def freundlich(Ce: np.ndarray, KF: float, n: float) -> np.ndarray:
    """Freundlich: qe = KF * Ce^(1/n). Use Ce > 0 to avoid log(0)."""
    Ce_safe = np.maximum(Ce, 1e-10)
    return KF * np.power(Ce_safe, 1.0 / n)


# ── Statistics ────────────────────────────────────────────────────────────

def compute_r2(measured: np.ndarray, predicted: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot"""
    ss_res = np.sum((measured - predicted) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def separation_factor_RL(KL: float, C0: float) -> float:
    """Langmuir separation factor: RL = 1 / (1 + KL * C0). Favorable when 0 < RL < 1."""
    return 1.0 / (1.0 + KL * C0)


# ── Fitting ──────────────────────────────────────────────────────────────

def fit_langmuir(Ce: np.ndarray, qe: np.ndarray) -> tuple:
    """Fit Langmuir model. Returns (qmax, KL), predicted qe, R²."""
    # Initial guess: qmax ~ max(qe)*1.2, KL ~ 0.1
    p0 = [np.max(qe) * 1.2, 0.1]
    popt, _ = curve_fit(langmuir, Ce, qe, p0=p0, bounds=(0, np.inf))
    qmax, KL = popt[0], popt[1]
    pred = langmuir(Ce, qmax, KL)
    r2 = compute_r2(qe, pred)
    return qmax, KL, pred, r2


def fit_freundlich(Ce: np.ndarray, qe: np.ndarray) -> tuple:
    """Fit Freundlich model. Returns (KF, n), predicted qe, R²."""
    # Initial guess: KF ~ 1, n ~ 2 (favorable)
    p0 = [1.0, 2.0]
    popt, _ = curve_fit(freundlich, Ce, qe, p0=p0, bounds=([1e-10, 0.1], np.inf))
    KF, n = popt[0], popt[1]
    pred = freundlich(Ce, KF, n)
    r2 = compute_r2(qe, pred)
    return KF, n, pred, r2


# ── Validation harness ───────────────────────────────────────────────────

class Validator:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        msg = f"  [{status}] {label}"
        if detail:
            msg += f": {detail}"
        print(msg)

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  TOTAL: {self.passed}/{total} PASS")
        if self.failed > 0:
            print(f"  *** {self.failed} FAILED ***")
        print(f"{'='*60}")
        return self.failed == 0


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    benchmark_path = Path(__file__).parent / "benchmark_biochar.json"
    with open(benchmark_path) as f:
        bench = json.load(f)

    v = Validator()
    datasets = bench["isotherm_data"]["datasets"]
    validation = bench["validation_checks"]

    # Store results for cross-checks
    results = {}

    print("=" * 70)
    print("airSpring Exp 007: Biochar Adsorption Isotherms Baseline")
    print("  Kumari, Dong & Safferman (2025) Applied Water Science 15(7):162")
    print("=" * 70)

    for ds_name, ds in datasets.items():
        Ce = np.array(ds["Ce"], dtype=float)
        qe = np.array(ds["qe"], dtype=float)

        print(f"\n── Dataset: {ds_name} ──")
        print(f"  Source: {ds['source']}")

        # Langmuir fit
        qmax, KL, pred_lang, r2_lang = fit_langmuir(Ce, qe)
        print(f"\n  Langmuir: qmax={qmax:.4f} mg/g, KL={KL:.6f} L/mg, R²={r2_lang:.4f}")

        # Freundlich fit
        KF, n, pred_freund, r2_freund = fit_freundlich(Ce, qe)
        print(f"  Freundlich: KF={KF:.4f}, n={n:.4f}, R²={r2_freund:.4f}")

        # Separation factor RL at C0=100
        RL = separation_factor_RL(KL, 100.0)
        print(f"  RL (C0=100 mg/L) = {RL:.4f}")

        results[ds_name] = {
            "qmax": qmax,
            "KL": KL,
            "r2_langmuir": r2_lang,
            "KF": KF,
            "n": n,
            "r2_freundlich": r2_freund,
            "RL": RL,
            "residuals_lang": qe - pred_lang,
            "residuals_freund": qe - pred_freund,
        }

    # ── Langmuir validation checks ───────────────────────────────────────
    print("\n── Langmuir fit validation ──")
    for c in validation["langmuir_fit"]["checks"]:
        cid = c["id"]
        if cid == "wood_qmax_range":
            qmax = results["wood_biochar_500C"]["qmax"]
            ok = c["min"] <= qmax <= c["max"]
            v.check(c["description"], ok, f"qmax={qmax:.4f} mg/g")
        elif cid == "wood_KL_positive":
            KL = results["wood_biochar_500C"]["KL"]
            v.check(c["description"], KL > 0, f"KL={KL:.6f}")
        elif cid == "wood_r2":
            r2 = results["wood_biochar_500C"]["r2_langmuir"]
            ok = r2 >= c["min_r2"]
            v.check(c["description"], ok, f"R²={r2:.4f}")
        elif cid == "sugar_qmax_range":
            qmax = results["sugar_beet_biochar"]["qmax"]
            ok = c["min"] <= qmax <= c["max"]
            v.check(c["description"], ok, f"qmax={qmax:.4f} mg/g")
        elif cid == "sugar_r2":
            r2 = results["sugar_beet_biochar"]["r2_langmuir"]
            ok = r2 >= c["min_r2"]
            v.check(c["description"], ok, f"R²={r2:.4f}")

    # ── Freundlich validation checks ─────────────────────────────────────
    print("\n── Freundlich fit validation ──")
    for c in validation["freundlich_fit"]["checks"]:
        cid = c["id"]
        if cid == "wood_KF_positive":
            KF = results["wood_biochar_500C"]["KF"]
            v.check(c["description"], KF > 0, f"KF={KF:.4f}")
        elif cid == "wood_n_favorable":
            n = results["wood_biochar_500C"]["n"]
            ok = n >= c["min_n"]
            v.check(c["description"], ok, f"n={n:.4f}")
        elif cid == "wood_r2":
            r2 = results["wood_biochar_500C"]["r2_freundlich"]
            ok = r2 >= c["min_r2"]
            v.check(c["description"], ok, f"R²={r2:.4f}")
        elif cid == "sugar_KF_positive":
            KF = results["sugar_beet_biochar"]["KF"]
            v.check(c["description"], KF > 0, f"KF={KF:.4f}")
        elif cid == "sugar_n_range":
            n = results["sugar_beet_biochar"]["n"]
            ok = c["min"] <= n <= c["max"]
            v.check(c["description"], ok, f"n={n:.4f}")

    # ── Model comparison ──────────────────────────────────────────────────
    print("\n── Model comparison ──")
    for c in validation["model_comparison"]["checks"]:
        cid = c["id"]
        if cid == "langmuir_better_wood":
            r2_l = results["wood_biochar_500C"]["r2_langmuir"]
            r2_f = results["wood_biochar_500C"]["r2_freundlich"]
            ok = r2_l >= r2_f
            v.check(c["description"], ok, f"Langmuir R²={r2_l:.4f}, Freundlich R²={r2_f:.4f}")
        elif cid == "both_positive_params":
            all_pos = True
            for ds_name, r in results.items():
                if r["qmax"] <= 0 or r["KL"] <= 0 or r["KF"] <= 0 or r["n"] <= 0:
                    all_pos = False
                    break
            v.check(c["description"], all_pos)
        elif cid == "residuals_random":
            # Mean residual (signed) near 0 indicates no systematic bias
            max_mean_res = 0
            for ds_name, r in results.items():
                mean_res_lang = np.mean(r["residuals_lang"])
                mean_res_freund = np.mean(r["residuals_freund"])
                max_mean_res = max(max_mean_res, abs(mean_res_lang), abs(mean_res_freund))
            ok = max_mean_res < 0.5
            v.check(c["description"], ok, f"max |mean residual|={max_mean_res:.4f} mg/g")

    # ── Separation factor ───────────────────────────────────────────────
    print("\n── Separation factor RL ──")
    for c in validation["separation_factor"]["checks"]:
        cid = c["id"]
        if cid == "rl_favorable":
            C0 = c.get("C0", 100)
            all_favorable = True
            for ds_name, r in results.items():
                RL = separation_factor_RL(r["KL"], C0)
                if not (0 < RL < 1):
                    all_favorable = False
                    break
            v.check(c["description"], all_favorable, f"C0={C0} mg/L")

    ok = v.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
