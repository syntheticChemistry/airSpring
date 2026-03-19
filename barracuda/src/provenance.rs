// SPDX-License-Identifier: AGPL-3.0-or-later
//! Python baseline provenance registry.
//!
//! Every validation binary in airSpring corresponds to a Python baseline
//! script. This module provides a structured registry of those baselines
//! with commit hashes, dates, and categories — the single source of truth
//! for which Rust binary validates against which Python run.

/// A Python baseline provenance record.
pub struct PythonBaseline {
    /// Rust validation binary name (e.g. `"validate_et0"`).
    pub binary: &'static str,
    /// Python control script path relative to workspace.
    pub script: Option<&'static str>,
    /// Git commit of the baseline run.
    pub commit: &'static str,
    /// Date of the baseline run (ISO 8601).
    pub date: &'static str,
    /// Baseline category.
    pub category: BaselineCategory,
}

/// Categories of validation baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineCategory {
    /// Rust reproduces Python/scipy results within tolerance.
    PythonParity,
    /// GPU reproduces CPU Rust results.
    GpuParity,
    /// Rust matches closed-form known values.
    Analytical,
    /// Rust reproduces published paper values.
    Published,
}

/// Canonical commit hashes for major baseline epochs.
pub mod commits {
    /// Initial Python parity baselines (Exp001–060, `FAO-56` + soil + `IoT` + WB).
    pub const PYTHON_PARITY_V1: &str = "e4358c5";
    /// Extended baselines (Exp045-081, Anderson + Richards + coupled + MC + drought).
    pub const PYTHON_PARITY_V2: &str = "3afc229";
    /// GPU parity baselines (Exp040-047, 055, 057).
    pub const GPU_PARITY_V1: &str = "cb59873";
    /// Simplified ET₀ + Hargreaves + diversity (v0.5.x).
    pub const SIMPLIFIED_ET0: &str = "fad2e1b";
    /// Biochar + cover crop + lysimeter + scheduling (v0.5.x).
    pub const EXPERIMENT_V3: &str = "5684b1e";
    /// Lysimeter + scheduling + sensitivity (v0.5.x).
    pub const EXPERIMENT_V4: &str = "e651409";
    /// Priestley-Taylor + `ET₀` intercomparison (v0.5.x).
    pub const INTERCOMPARISON: &str = "9a84ae5";
    /// SCS-CN + Green-Ampt + Blaney-Criddle + ensemble + bias + regional (v0.6.x).
    pub const RUNOFF_ENSEMBLE: &str = "97e7533";
    /// Makkink + Turc + Hamon + simplified (v0.5.x).
    pub const SIMPLIFIED_V2: &str = "d3ecdc8";
    /// Forecast + SCAN + multicrop + NASS + `AmeriFlux` (v0.6.x).
    pub const FIELD_DATA: &str = "8c3953b";
    /// VG inverse + season WB (v0.6.x).
    pub const VG_SEASON: &str = "6be822f";
    /// Anderson coupling (v0.6.x).
    pub const ANDERSON: &str = "0500398";
    /// Paper 12: barrier skin + cross-species + cytokine + tissue + CPU/GPU parity.
    pub const PAPER12: &str = "dbfb53a";
    /// Bootstrap/jackknife + climate scenario (v0.7.x).
    pub const STOCHASTIC_V1: &str = "1c11763";
    /// MC `ET₀` propagation + bootstrap/jackknife + SPI drought (v0.7.x stochastic).
    pub const STOCHASTIC_V2: &str = "e1754cf";
    /// NCBI diversity + atlas decade + NASS real (v0.7.x).
    pub const NCBI_ATLAS: &str = "88d07c0";
    /// NCBI 16S coupling (v0.8.x).
    pub const NCBI_16S: &str = "4c8546e";
    /// Climate scenario extended (v0.8.x).
    pub const CLIMATE_EXT: &str = "1a40b1e";
}

/// All registered Python baseline provenance records.
///
/// When a Python control script is rerun, update the commit and date here.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "provenance registry is a single const table — splitting loses locality"
)]
pub const fn python_baselines() -> &'static [PythonBaseline] {
    &[
        // ── Core FAO-56 + soil + water balance (V1 epoch) ──
        PythonBaseline {
            binary: "validate_et0",
            script: Some("control/fao56/penman_monteith.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_soil",
            script: Some("control/soil_sensors/calibration_dong2020.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_water_balance",
            script: Some("control/water_balance/fao56_water_balance.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_richards",
            script: Some("control/richards/richards_1d.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-20",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_yield",
            script: Some("control/yield_response/yield_response.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::Published,
        },
        PythonBaseline {
            binary: "validate_sensor_calibration",
            script: Some("control/iot_irrigation/calibration_dong2024.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-16",
            category: BaselineCategory::PythonParity,
        },
        // ── Extended baselines (V2 epoch) ──
        PythonBaseline {
            binary: "validate_dual_kc",
            script: Some("control/dual_kc/cover_crop_dual_kc.py"),
            commit: commits::PYTHON_PARITY_V2,
            date: "2026-02-25",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_coupled_runoff",
            script: Some("control/coupled_runoff_infiltration/coupled_runoff_infiltration.py"),
            commit: commits::PYTHON_PARITY_V2,
            date: "2026-02-25",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_mc_et0",
            script: Some("control/mc_et0/mc_et0_propagation.py"),
            commit: commits::STOCHASTIC_V2,
            date: "2026-03-07",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_drought_index",
            script: Some("control/drought_index/drought_index_spi.py"),
            commit: commits::STOCHASTIC_V2,
            date: "2026-03-07",
            category: BaselineCategory::PythonParity,
        },
        // ── GPU parity ──
        PythonBaseline {
            binary: "validate_gpu_math",
            script: None,
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-26",
            category: BaselineCategory::GpuParity,
        },
        PythonBaseline {
            binary: "validate_atlas",
            script: Some("control/atlas/atlas_water_budget.py"),
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_cpu_gpu_parity",
            script: Some("control/cpu_gpu_parity/cpu_gpu_parity.py"),
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::GpuParity,
        },
        PythonBaseline {
            binary: "validate_paper_chain",
            script: None,
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::GpuParity,
        },
        // ── Experiment V3: biochar + cover crop + long-term WB ──
        PythonBaseline {
            binary: "validate_biochar",
            script: Some("control/biochar/biochar_isotherms.py"),
            commit: commits::EXPERIMENT_V3,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_cover_crop",
            script: Some("control/dual_kc/cover_crop_dual_kc.py"),
            commit: commits::EXPERIMENT_V3,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_long_term_wb",
            script: Some("control/long_term_wb/long_term_water_balance.py"),
            commit: commits::EXPERIMENT_V3,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        // ── Experiment V4: lysimeter + scheduling + sensitivity ──
        PythonBaseline {
            binary: "validate_lysimeter",
            script: Some("control/lysimeter/lysimeter_et.py"),
            commit: commits::EXPERIMENT_V4,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_scheduling",
            script: Some("control/scheduling/irrigation_scheduling.py"),
            commit: commits::EXPERIMENT_V4,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_sensitivity",
            script: Some("control/sensitivity/et0_sensitivity.py"),
            commit: commits::EXPERIMENT_V4,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        // ── Intercomparison + Priestley-Taylor ──
        PythonBaseline {
            binary: "validate_priestley_taylor",
            script: Some("control/priestley_taylor/priestley_taylor_et0.py"),
            commit: commits::INTERCOMPARISON,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_et0_intercomparison",
            script: Some("control/et0_intercomparison/et0_three_method.py"),
            commit: commits::INTERCOMPARISON,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
        },
        // ── Simplified ET₀ + Hargreaves + diversity ──
        PythonBaseline {
            binary: "validate_hargreaves",
            script: Some("control/hargreaves/hargreaves_samani.py"),
            commit: commits::SIMPLIFIED_ET0,
            date: "2026-03-02",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_thornthwaite",
            script: Some("control/thornthwaite/thornthwaite_et0.py"),
            commit: commits::SIMPLIFIED_ET0,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_gdd",
            script: Some("control/gdd/growing_degree_days.py"),
            commit: commits::SIMPLIFIED_ET0,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_diversity",
            script: Some("control/diversity/diversity_indices.py"),
            commit: commits::SIMPLIFIED_ET0,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_pedotransfer",
            script: Some("control/pedotransfer/saxton_rawls.py"),
            commit: commits::SIMPLIFIED_ET0,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        // ── Simplified V2: Makkink + Turc + Hamon ──
        PythonBaseline {
            binary: "validate_makkink",
            script: Some("control/makkink/makkink_et0.py"),
            commit: commits::SIMPLIFIED_V2,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_turc",
            script: Some("control/turc/turc_et0.py"),
            commit: commits::SIMPLIFIED_V2,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_hamon",
            script: Some("control/hamon/hamon_pet.py"),
            commit: commits::SIMPLIFIED_V2,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        // ── Runoff + ensemble + regional ──
        PythonBaseline {
            binary: "validate_blaney_criddle",
            script: Some("control/blaney_criddle/blaney_criddle_et0.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_scs_cn",
            script: Some("control/scs_curve_number/scs_curve_number.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_green_ampt",
            script: Some("control/green_ampt/green_ampt_infiltration.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_pedotransfer_richards",
            script: Some("control/pedotransfer_richards/pedotransfer_richards.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_et0_ensemble",
            script: Some("control/et0_ensemble/et0_ensemble.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_et0_bias",
            script: Some("control/et0_bias_correction/et0_bias_correction.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_regional_et0",
            script: Some("control/regional_et0/regional_et0_intercomparison.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_seasonal_batch",
            script: Some("control/seasonal_batch_et0/seasonal_batch_et0.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_neural_api",
            script: Some("control/neural_api/neural_api_parity.py"),
            commit: commits::RUNOFF_ENSEMBLE,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        // ── VG inverse + season WB ──
        PythonBaseline {
            binary: "validate_vg_inverse",
            script: Some("control/vg_inverse/vg_inverse_fitting.py"),
            commit: commits::VG_SEASON,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_season_wb",
            script: Some("control/season_water_budget/season_water_budget.py"),
            commit: commits::VG_SEASON,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        // ── Anderson coupling ──
        PythonBaseline {
            binary: "validate_anderson",
            script: Some("control/anderson_coupling/anderson_coupling.py"),
            commit: commits::ANDERSON,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        // ── Field data: forecast + SCAN + multicrop + NASS + AmeriFlux ──
        PythonBaseline {
            binary: "validate_forecast",
            script: Some("control/forecast_scheduling/forecast_scheduling.py"),
            commit: commits::FIELD_DATA,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_scan_moisture",
            script: Some("control/scan_moisture/scan_moisture_validation.py"),
            commit: commits::FIELD_DATA,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_multicrop",
            script: Some("control/multicrop_budget/multicrop_water_budget.py"),
            commit: commits::FIELD_DATA,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_nass_yield",
            script: Some("control/nass_yield/nass_yield_validation.py"),
            commit: commits::FIELD_DATA,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_ameriflux",
            script: Some("control/ameriflux_et/ameriflux_et_validation.py"),
            commit: commits::FIELD_DATA,
            date: "2026-02-27",
            category: BaselineCategory::PythonParity,
        },
        // ── Paper 12: immunological Anderson ──
        PythonBaseline {
            binary: "validate_barrier_skin",
            script: Some("control/barrier_skin/barrier_skin.py"),
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_cross_species",
            script: Some("control/cross_species_skin/cross_species_skin.py"),
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_cytokine",
            script: Some("control/cytokine_brain/cytokine_brain.py"),
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_tissue",
            script: Some("control/tissue_diversity/tissue_diversity.py"),
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::PythonParity,
        },
        // ── Stochastic: bootstrap/jackknife + climate scenario ──
        PythonBaseline {
            binary: "validate_bootstrap_jackknife",
            script: Some("control/bootstrap_jackknife/bootstrap_jackknife_et0.py"),
            commit: commits::STOCHASTIC_V1,
            date: "2026-03-07",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_climate_scenario",
            script: Some("control/climate_scenario/climate_scenario_analysis.py"),
            commit: commits::CLIMATE_EXT,
            date: "2026-03-01",
            category: BaselineCategory::PythonParity,
        },
        // ── NCBI + atlas decade ──
        PythonBaseline {
            binary: "validate_ncbi_diversity",
            script: Some("control/ncbi_diversity/ncbi_diversity_analysis.py"),
            commit: commits::NCBI_ATLAS,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_atlas_decade",
            script: Some("control/atlas_decade/atlas_decade_analysis.py"),
            commit: commits::NCBI_ATLAS,
            date: "2026-03-01",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_ncbi_16s_coupling",
            script: Some("control/ncbi_16s_coupling/ncbi_16s_coupling.py"),
            commit: commits::NCBI_16S,
            date: "2026-02-28",
            category: BaselineCategory::PythonParity,
        },
        // ── Published / analytical-only (no Python baseline script) ──
        PythonBaseline {
            binary: "validate_nass_real",
            script: None,
            commit: commits::NCBI_ATLAS,
            date: "2026-03-01",
            category: BaselineCategory::Published,
        },
        PythonBaseline {
            binary: "validate_atlas_stream",
            script: None,
            commit: commits::SIMPLIFIED_V2,
            date: "2026-02-27",
            category: BaselineCategory::Published,
        },
        PythonBaseline {
            binary: "validate_iot",
            script: None,
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::Analytical,
        },
        PythonBaseline {
            binary: "validate_real_data",
            script: None,
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::Published,
        },
        PythonBaseline {
            binary: "validate_npu_eco",
            script: None,
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::Analytical,
        },
        PythonBaseline {
            binary: "validate_npu_funky_eco",
            script: None,
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::Analytical,
        },
        PythonBaseline {
            binary: "validate_npu_high_cadence",
            script: None,
            commit: commits::PAPER12,
            date: "2026-03-02",
            category: BaselineCategory::Analytical,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_baselines_have_commit() {
        for baseline in python_baselines() {
            assert!(
                !baseline.commit.is_empty(),
                "baseline '{}' has empty commit",
                baseline.binary
            );
        }
    }

    #[test]
    fn all_baselines_have_date() {
        for baseline in python_baselines() {
            assert!(
                baseline.date.len() == 10,
                "baseline '{}' date '{}' is not ISO 8601",
                baseline.binary,
                baseline.date
            );
        }
    }

    #[test]
    fn no_duplicate_binaries() {
        let names: Vec<&str> = python_baselines().iter().map(|b| b.binary).collect();
        for (i, name) in names.iter().enumerate() {
            assert!(
                !names[i + 1..].contains(name),
                "duplicate binary '{name}' in provenance registry"
            );
        }
    }

    #[test]
    fn baseline_count() {
        assert!(
            python_baselines().len() >= 60,
            "expected at least 60 baselines, got {}",
            python_baselines().len()
        );
    }
}
