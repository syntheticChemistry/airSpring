#!/usr/bin/env bash
# ==========================================================================
# airSpring Phase 0 — Run All Python/R Baseline Validations
#
# Each experiment script exits 0 on all-PASS, non-zero on any FAIL.
# This script runs all baselines and reports the overall result.
#
# Usage:
#   bash scripts/run_all_baselines.sh
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PASS=0
FAIL=0
SKIP=0

run_baseline() {
    local name="$1"
    local cmd="$2"
    local is_optional="${3:-no}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$cmd"; then
        PASS=$((PASS + 1))
    else
        if [ "$is_optional" = "optional" ]; then
            echo "  [SKIPPED] $name (optional dependency not available)"
            SKIP=$((SKIP + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    fi
}

echo "======================================================================"
echo "  airSpring Phase 0 — Baseline Validation Suite"
echo "  $(date)"
echo "======================================================================"

# --- Experiment 001: FAO-56 Penman-Monteith ---
run_baseline \
    "Exp 001: FAO-56 Penman-Monteith (Python)" \
    "python control/fao56/penman_monteith.py"

# --- Experiment 002: Soil Sensor Calibration ---
run_baseline \
    "Exp 002: Soil Sensor Calibration (Python)" \
    "python control/soil_sensors/calibration_dong2020.py"

# --- Experiment 003: IoT Irrigation Pipeline (Python part) ---
run_baseline \
    "Exp 003: IoT Irrigation Pipeline (Python)" \
    "python control/iot_irrigation/calibration_dong2024.py"

# --- Experiment 003: IoT Irrigation ANOVA (R part) ---
if command -v Rscript &>/dev/null; then
    run_baseline \
        "Exp 003: IoT Irrigation ANOVA (R)" \
        "Rscript control/iot_irrigation/anova_irrigation.R"
else
    echo ""
    echo "  [SKIP] Exp 003 R ANOVA — Rscript not found (install R >= 4.0)"
    SKIP=$((SKIP + 1))
fi

# --- Experiment 004: Water Balance ---
run_baseline \
    "Exp 004: FAO-56 Water Balance (Python)" \
    "python control/water_balance/fao56_water_balance.py"

# --- Experiment 006: Richards Equation (van Genuchten) ---
run_baseline \
    "Exp 006: Richards Equation (Python)" \
    "python control/richards/richards_1d.py"

# --- Experiment 007: Biochar Isotherms (Kumari et al. 2025) ---
run_baseline \
    "Exp 007: Biochar Isotherms (Python)" \
    "python control/biochar/biochar_isotherms.py"

# --- Experiment 009: Dual Kc (FAO-56 Ch 7) ---
run_baseline \
    "Exp 009: Dual Crop Coefficient (Python)" \
    "python control/dual_kc/dual_crop_coefficient.py"

# --- Experiment 011: Cover Crops + No-Till (FAO-56 Ch 11) ---
run_baseline \
    "Exp 011: Cover Crop Dual Kc (Python)" \
    "python control/dual_kc/cover_crop_dual_kc.py"

# --- Experiment 015: 60-Year Water Balance ---
run_baseline \
    "Exp 015: 60-Year Water Balance (Python)" \
    "python control/long_term_wb/long_term_water_balance.py"

# =====================================================================
# REAL DATA PIPELINE (requires API/internet — optional but preferred)
# =====================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Real Data Pipeline (open APIs, minimal synthetic)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Download real weather: Open-Meteo (free, no key) ---
run_baseline \
    "Data: Open-Meteo historical weather (2023 growing season)" \
    "python scripts/download_open_meteo.py --all-stations --growing-season 2023" \
    "optional"

# --- Download real weather: OpenWeatherMap (needs key in testing-secrets/) ---
run_baseline \
    "Data: OpenWeatherMap current + forecast" \
    "python scripts/download_enviroweather.py --all-stations" \
    "optional"

# --- Compute ET₀ on real data ---
if [ -d data/open_meteo ]; then
    run_baseline \
        "Compute: FAO-56 ET₀ on real Michigan data" \
        "python control/fao56/compute_et0_real_data.py --all-stations"
else
    echo "  [SKIP] ET₀ on real data — run Open-Meteo download first"
    SKIP=$((SKIP + 1))
fi

# --- Water balance on real data ---
if [ -d data/et0_results ]; then
    run_baseline \
        "Compute: Water balance on real Michigan data" \
        "python control/water_balance/simulate_real_data.py"
else
    echo "  [SKIP] Water balance on real data — run ET₀ computation first"
    SKIP=$((SKIP + 1))
fi

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo ""
echo "======================================================================"
echo "  BASELINE VALIDATION SUMMARY"
echo "======================================================================"
echo "  Passed:  $PASS / $TOTAL experiments"
echo "  Failed:  $FAIL / $TOTAL experiments"
echo "  Skipped: $SKIP (optional dependencies not available)"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: ALL BASELINES PASS"
    echo "======================================================================"
    exit 0
else
    echo "  RESULT: $FAIL BASELINE(S) FAILED"
    echo "======================================================================"
    exit 1
fi
