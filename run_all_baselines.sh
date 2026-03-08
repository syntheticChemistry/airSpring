#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run all Python baselines and Rust validation binaries.
# Exit 0 only if every step passes.
set -euo pipefail

cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════"
echo "  airSpring — Full Baseline + Validation Suite"
echo "═══════════════════════════════════════════════════════════════"
echo ""

FAIL=0

run_python() {
    echo "── Python: $1"
    if python3 "$1"; then
        echo "  PASS"
    else
        echo "  FAIL"
        FAIL=1
    fi
}

run_rust() {
    echo "── Rust:   $1"
    if cargo run --release --bin "$1" --manifest-path barracuda/Cargo.toml 2>/dev/null; then
        echo "  PASS"
    else
        echo "  FAIL"
        FAIL=1
    fi
}

echo "━━━ Phase 0: Python Baselines (paper-only, no API keys) ━━━"
echo ""
run_python control/fao56/penman_monteith.py
run_python control/soil_sensors/calibration_dong2020.py
run_python control/iot_irrigation/calibration_dong2024.py
run_python control/water_balance/fao56_water_balance.py
run_python control/dual_kc/dual_crop_coefficient.py
run_python control/dual_kc/cover_crop_dual_kc.py
run_python control/richards/richards_1d.py
run_python control/biochar/biochar_isotherms.py
run_python control/yield_response/yield_response.py
run_python control/cw2d/cw2d_richards.py
run_python control/scheduling/irrigation_scheduling.py
run_python control/lysimeter/lysimeter_et.py
run_python control/sensitivity/et0_sensitivity.py
run_python control/priestley_taylor/priestley_taylor_et0.py
run_python control/et0_intercomparison/et0_three_method.py
run_python control/thornthwaite/thornthwaite_et0.py
run_python control/gdd/growing_degree_days.py
run_python control/pedotransfer/saxton_rawls.py
run_python control/makkink/makkink_et0.py
run_python control/turc/turc_et0.py
run_python control/hamon/hamon_pet.py
run_python control/neural_api/neural_api_parity.py
run_python control/et0_ensemble/et0_ensemble.py
run_python control/pedotransfer_richards/pedotransfer_richards.py
run_python control/et0_bias_correction/et0_bias_correction.py
run_python control/cpu_gpu_parity/cpu_gpu_parity.py
run_python control/metalforge_dispatch/metalforge_dispatch.py
run_python control/seasonal_batch_et0/seasonal_batch_et0.py
run_python control/nass_yield/nass_yield_validation.py
run_python control/forecast_scheduling/forecast_scheduling.py
run_python control/scan_moisture/scan_moisture_validation.py
run_python control/ameriflux_et/ameriflux_et_validation.py
run_python control/hargreaves/hargreaves_samani.py
run_python control/diversity/diversity_indices.py
run_python control/anderson_coupling/anderson_coupling.py
run_python control/gpu_math_portability/gpu_math_portability.py
run_python control/ncbi_16s_coupling/ncbi_16s_coupling.py
run_python control/blaney_criddle/blaney_criddle_et0.py
run_python control/scs_curve_number/scs_curve_number.py
run_python control/green_ampt/green_ampt_infiltration.py
run_python control/coupled_runoff_infiltration/coupled_runoff_infiltration.py
run_python control/vg_inverse/vg_inverse_fitting.py
run_python control/season_water_budget/season_water_budget.py

echo ""
echo "━━━ Phase 1: Rust Validation Binaries ━━━"
echo ""
run_rust validate_et0
run_rust validate_soil
run_rust validate_sensor_calibration
run_rust validate_water_balance
run_rust validate_dual_kc
run_rust validate_cover_crop
run_rust validate_richards
run_rust validate_cw2d
run_rust validate_yield
run_rust validate_biochar
run_rust validate_long_term_wb
run_rust validate_scheduling
run_rust validate_lysimeter
run_rust validate_sensitivity
run_rust validate_priestley_taylor
run_rust validate_et0_intercomparison
run_rust validate_thornthwaite
run_rust validate_gdd
run_rust validate_pedotransfer
run_rust validate_makkink
run_rust validate_turc
run_rust validate_hamon
run_rust validate_neural_api
run_rust validate_et0_ensemble
run_rust validate_pedotransfer_richards
run_rust validate_et0_bias
run_rust validate_cpu_gpu_parity
run_rust validate_seasonal_batch

run_rust validate_anderson
run_rust validate_regional_et0
run_rust validate_real_data
run_rust validate_iot
run_rust validate_ameriflux
run_rust validate_hargreaves
run_rust validate_diversity
run_rust validate_blaney_criddle
run_rust validate_scs_cn
run_rust validate_green_ampt
run_rust validate_coupled_runoff
run_rust validate_vg_inverse
run_rust validate_season_wb

echo ""
echo "━━━ Phase 1+: Data-Dependent Validations ━━━"
echo ""
run_rust validate_forecast
run_rust validate_multicrop
run_rust validate_nass_yield
run_rust validate_scan_moisture
run_rust validate_atlas
run_rust validate_atlas_stream

echo ""
echo "━━━ Phase 1++: metalForge Validation ━━━"
echo ""
echo "── Rust:   validate_dispatch (metalForge)"
if cargo run --release --bin validate_dispatch --manifest-path metalForge/forge/Cargo.toml 2>/dev/null; then
    echo "  PASS"
else
    echo "  FAIL"
    FAIL=1
fi
echo "── Rust:   validate_live_hardware (metalForge — live probe)"
if cargo run --release --bin validate_live_hardware --manifest-path metalForge/forge/Cargo.toml 2>/dev/null; then
    echo "  PASS"
else
    echo "  FAIL"
    FAIL=1
fi

run_rust validate_pure_gpu
run_rust validate_gpu_math
run_rust validate_ncbi_16s_coupling

echo ""
echo "━━━ Phase 5: NUCLEUS Integration (Exp 084-087) ━━━"
echo ""
run_rust validate_cpu_gpu_comprehensive
run_rust validate_toadstool_dispatch
run_rust validate_nucleus_graphs
echo "── Rust:   validate_mixed_nucleus_live (metalForge — Exp 086)"
if cargo run --release --bin validate_mixed_nucleus_live --manifest-path metalForge/forge/Cargo.toml 2>/dev/null; then
    echo "  PASS"
else
    echo "  FAIL"
    FAIL=1
fi

echo ""
echo "━━━ Phase 3: GPU Live Dispatch (Titan V) ━━━"
echo ""
echo "── Rust:   validate_gpu_live (barracuda — Titan V GPU)"
if BARRACUDA_GPU_ADAPTER=titan cargo run --release --bin validate_gpu_live --manifest-path barracuda/Cargo.toml 2>/dev/null; then
    echo "  PASS"
else
    echo "  FAIL (GPU may not be available)"
fi

echo ""
echo "━━━ Phase 2: Cross-Validation ━━━"
echo ""
run_rust cross_validate

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL BASELINES + VALIDATIONS PASSED"
else
    echo "  SOME STEPS FAILED — see above"
fi
echo "═══════════════════════════════════════════════════════════════"

exit "$FAIL"
