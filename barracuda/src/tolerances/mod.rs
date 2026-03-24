// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-specific validation tolerances for agricultural science.
//!
//! Extends the [`barracuda::tolerances`] pattern (centralized, justified) with
//! airSpring-specific constants grounded in FAO-56, HYDRUS, and published
//! Python baselines. Every tolerance here has a physical or numerical
//! justification—no magic numbers.
//!
//! # Cross-Spring Provenance
//!
//! The `Tolerance` struct itself comes from `BarraCuda` S52 (M-010), inspired by
//! barracuda validation registry and adopted across all Springs.
//!
//! | Domain | Tolerance | Origin |
//! |--------|-----------|--------|
//! | ET₀ Penman-Monteith | 0.01 mm/day | FAO-56 Table 2 (worked examples) |
//! | Saturation vapor pressure | 0.01 kPa | FAO-56 Table 2.3 (3 decimals) |
//! | Water balance mass | 0.01 mm | FAO-56 Ch 8 conservation law |
//! | Richards PDE | benchmark-specified | HYDRUS numerical convergence |
//! | Isotherm fit | 0.01 mg/g | Adsorption experiment precision |
//! | Kriging interpolation | 1e-6 | Numerical linear algebra (small systems) |
//! | GPU/CPU cross-validation | 1e-5 | f64 shader rounding (`BarraCuda` TS-001 fix) |
//!
//! # Baseline Provenance
//!
//! Tolerances were validated against Python baselines with the following provenance.
//! Each tolerance domain maps to one or more control experiments whose outputs
//! confirmed the threshold is sufficient and minimal.
//!
//! | Domain | Control script | Commit | Date | Command |
//! |--------|---------------|--------|------|---------|
//! | ET₀ (PM) | `control/fao56/penman_monteith.py` | `94cc51d` | 2026-02-16 | `python3 control/fao56/penman_monteith.py` |
//! | ET₀ (intercomp.) | `control/et0_intercomparison/et0_three_method.py` | `9a84ae5` | 2026-02-26 | `python3 control/et0_intercomparison/et0_three_method.py` |
//! | Priestley-Taylor | `control/priestley_taylor/priestley_taylor_et0.py` | `9a84ae5` | 2026-02-26 | `python3 control/priestley_taylor/priestley_taylor_et0.py` |
//! | Hargreaves | `control/hargreaves/hargreaves_samani.py` | `fad2e1b` | 2026-02-26 | `python3 control/hargreaves/hargreaves_samani.py` |
//! | Thornthwaite | `control/thornthwaite/thornthwaite_et0.py` | `fad2e1b` | 2026-02-26 | `python3 control/thornthwaite/thornthwaite_et0.py` |
//! | Water balance | `control/water_balance/fao56_water_balance.py` | `94cc51d` | 2026-02-16 | `python3 control/water_balance/fao56_water_balance.py` |
//! | Richards PDE | `control/richards/richards_1d.py` | `3afc229` | 2026-02-25 | `python3 control/richards/richards_1d.py` |
//! | Soil sensors | `control/soil_sensors/calibration_dong2020.py` | `94cc51d` | 2026-02-16 | `python3 control/soil_sensors/calibration_dong2020.py` |
//! | IoT irrigation | `control/iot_irrigation/calibration_dong2024.py` | `94cc51d` | 2026-02-16 | Synthetic generator; physical plausibility checks only |
//! | Dual Kc | `control/dual_kc/cover_crop_dual_kc.py` | `3afc229` | 2026-02-25 | `python3 control/dual_kc/cover_crop_dual_kc.py` |
//! | Biochar isotherm | `control/biochar/biochar_isotherms.py` | `5684b1e` | 2026-02-26 | `python3 control/biochar/biochar_isotherms.py` |
//! | Pedotransfer | `control/pedotransfer/saxton_rawls.py` | `fad2e1b` | 2026-02-26 | `python3 control/pedotransfer/saxton_rawls.py` |
//! | GDD | `control/gdd/growing_degree_days.py` | `fad2e1b` | 2026-02-26 | `python3 control/gdd/growing_degree_days.py` |
//! | SCS-CN | `control/scs_curve_number/scs_curve_number.py` | `97e7533` | 2026-02-28 | `python3 control/scs_curve_number/scs_curve_number.py` |
//! | Green-Ampt | `control/green_ampt/green_ampt_infiltration.py` | `97e7533` | 2026-02-28 | `python3 control/green_ampt/green_ampt_infiltration.py` |
//! | Blaney-Criddle | `control/blaney_criddle/blaney_criddle_et0.py` | `97e7533` | 2026-02-28 | `python3 control/blaney_criddle/blaney_criddle_et0.py` |
//! | Anderson coupling | `control/anderson_coupling/anderson_coupling.py` | `0500398` | 2026-02-27 | `python3 control/anderson_coupling/anderson_coupling.py` |
//! | Diversity | `control/diversity/diversity_indices.py` | `fad2e1b` | 2026-02-26 | `python3 control/diversity/diversity_indices.py` |
//! | GPU/CPU parity | `control/cpu_gpu_parity/cpu_gpu_parity.py` | `fad2e1b` | 2026-03-02 | `python3 control/cpu_gpu_parity/cpu_gpu_parity.py` |
//! | Yield response | `control/yield_response/yield_response.py` | `97e7533` | 2026-02-28 | `python3 control/yield_response/yield_response.py` |
//! | Climate scenario | `control/climate_scenario/climate_scenario_analysis.py` | `1c11763` | 2026-03-01 | `python3 control/climate_scenario/climate_scenario_analysis.py` |
//! | Atlas decade | `control/atlas_decade/atlas_decade_analysis.py` | `1c11763` | 2026-03-01 | `python3 control/atlas_decade/atlas_decade_analysis.py` |
//! | Makkink ET₀ | `control/makkink/makkink_et0.py` | `8c3953b` | 2026-02-27 | `python3 control/makkink/makkink_et0.py` |
//! | Turc ET₀ | `control/turc/turc_et0.py` | `8c3953b` | 2026-02-27 | `python3 control/turc/turc_et0.py` |
//! | Hamon PET | `control/hamon/hamon_pet.py` | `8c3953b` | 2026-02-27 | `python3 control/hamon/hamon_pet.py` |
//! | MC ET₀ (stochastic) | `control/mc_et0/mc_et0_propagation.py` | `e1754cf` | 2026-03-07 | `python3 control/mc_et0/mc_et0_propagation.py` |
//! | Bootstrap/Jackknife | `control/bootstrap_jackknife/bootstrap_jackknife_et0.py` | `e1754cf` | 2026-03-07 | `python3 control/bootstrap_jackknife/bootstrap_jackknife_et0.py` |
//! | SPI drought index | `control/drought_index/drought_index_spi.py` | `e1754cf` | 2026-03-07 | `python3 control/drought_index/drought_index_spi.py` |
//! | Barrier skin (Paper 12) | `control/barrier_skin/barrier_skin.py` | `dbfb53a` | 2026-03-02 | `python3 control/barrier_skin/barrier_skin.py` |
//! | Cross-species skin (Paper 12) | `control/cross_species_skin/cross_species_skin.py` | `dbfb53a` | 2026-03-02 | `python3 control/cross_species_skin/cross_species_skin.py` |
//! | Cytokine brain (Paper 12) | `control/cytokine_brain/cytokine_brain.py` | `dbfb53a` | 2026-03-02 | `python3 control/cytokine_brain/cytokine_brain.py` |
//! | Tissue diversity (Paper 12) | `control/tissue_diversity/tissue_diversity.py` | `dbfb53a` | 2026-03-02 | `python3 control/tissue_diversity/tissue_diversity.py` |
//! | Biodiversity (Shannon/Simpson/Bray-Curtis) | `control/diversity/diversity_indices.py` | `fad2e1b` | 2026-02-26 | `python3 control/diversity/diversity_indices.py` |
//! | NPU sigma floor | (analytical) | — | — | EMA variance floor; no Python baseline (machine-precision guard) |
//! | IoT sensor validation | `control/iot_irrigation/calibration_dong2024.py` | `94cc51d` | 2026-02-16 | Synthetic generator; physical plausibility checks only |
//! | IoT CSV round-trip | (analytical) | — | — | `{:.2}` format truncation; verified via `io::csv_ts` unit tests |
//! | Analytical computation | (analytical) | — | — | Generic digitization precision; FAO/USDA published table interpolation |
//! | R² minimum | (literature) | — | — | FAO-56 PM R² > 0.90; threshold allows ERA5 reanalysis noise |
//! | RMSE maximum | (literature) | — | — | Doorenbos & Pruitt (1977) measurement uncertainty ±1.5 mm/day |
//! | ET₀ cross-method % | (literature) | — | — | Hargreaves vs PM 10–30% divergence; Great Lakes 25% |
//! | P significance | (convention) | — | — | Standard two-tailed α = 0.05; no Python baseline needed |

pub use barracuda::tolerances::{Tolerance, check};

mod atmospheric;
mod gpu;
mod instrument;
mod numerics;
mod soil;

pub use atmospheric::*;
pub use gpu::*;
pub use instrument::*;
pub use numerics::*;
pub use soil::*;

/// Every registered [`Tolerance`] constant (upstream contract inventory).
#[must_use]
pub const fn all_tolerances() -> &'static [&'static Tolerance] {
    &[
        // ET₀ and atmospheric (FAO-56)
        &ET0_SAT_VAPOUR_PRESSURE,
        &ET0_SLOPE_VAPOUR,
        &ET0_NET_RADIATION,
        &ET0_REFERENCE,
        &ET0_VPD,
        &ET0_COLD_CLIMATE,
        &PSYCHROMETRIC_CONSTANT,
        // Water balance and soil moisture
        &WATER_BALANCE_MASS,
        &WATER_BALANCE_PER_STEP,
        &STRESS_COEFFICIENT,
        &SOIL_HYDRAULIC,
        &SOIL_ROUNDTRIP,
        // Richards equation
        &RICHARDS_STEADY,
        &RICHARDS_TRANSIENT,
        // Isotherm fitting
        &ISOTHERM_PARAMETER,
        &ISOTHERM_PREDICTION,
        &ISOTHERM_MEAN_RESIDUAL,
        // GPU/CPU cross-validation
        &GPU_CPU_CROSS,
        &KRIGING_INTERPOLATION,
        &SEASONAL_REDUCTION,
        &IOT_STREAM_SMOOTHING,
        // Sensor calibration
        &SENSOR_EXACT,
        &IRRIGATION_DEPTH,
        // Thornthwaite, GDD, pedotransfer
        &THORNTHWAITE_ANALYTICAL,
        &GDD_EXACT,
        &PEDOTRANSFER_MOISTURE,
        &PEDOTRANSFER_KSAT,
        // Per-step and sensor-specific
        &TOPP_EQUATION,
        &ANALYTICAL_COMPUTATION,
        // Statistical quality criteria
        &IA_CRITERION,
        &P_SIGNIFICANCE,
        &WATER_SAVINGS,
        // Cross-method and cross-station
        &CROSS_VALIDATION,
        &ET0_SAT_VAPOUR_PRESSURE_WIDE,
        &R2_MINIMUM,
        &RMSE_MAXIMUM,
        &ET0_CROSS_METHOD_PCT,
        // Simplified ET₀ methods
        &BLANEY_CRIDDLE_DAYLIGHT,
        &SCS_CN_ANALYTICAL,
        &GREEN_AMPT_ANALYTICAL,
        &DUAL_KC_PRECISION,
        // IoT sensor data validation
        &IOT_TEMPERATURE_MEAN,
        &IOT_TEMPERATURE_EXTREMES,
        &IOT_PAR_MAX,
        &IOT_CSV_ROUNDTRIP,
        // NPU streaming classification
        &NPU_SIGMA_FLOOR,
        // Biodiversity
        &BIO_DIVERSITY_SHANNON,
        &BIO_DIVERSITY_SIMPSON,
        &BIO_BRAY_CURTIS,
        // Stochastic / Monte Carlo
        &MC_ET0_PROPAGATION,
        // Cross-spring analytical
        &CROSS_SPRING_ANALYTICAL,
        &CROSS_SPRING_GPU_CPU,
        &CROSS_SPRING_EVOLUTION,
        // GPU method-specific parity tiers
        &GPU_SIMPLIFIED_ET0,
        &GPU_EMPIRICAL_PET,
        // NUCLEUS / IPC
        &NUCLEUS_ROUNDTRIP,
        &NUCLEUS_PIPELINE,
        // Bootstrap / jackknife
        &BOOTSTRAP_JACKKNIFE_KNOWN,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_et0_tolerance_check() {
        assert!(check(2.338, 2.338, &ET0_REFERENCE));
        assert!(check(2.348, 2.338, &ET0_REFERENCE));
        assert!(!check(2.448, 2.338, &ET0_REFERENCE));
    }

    #[test]
    fn test_water_balance_mass_check() {
        assert!(check(0.0, 0.0, &WATER_BALANCE_MASS));
        assert!(check(0.005, 0.0, &WATER_BALANCE_MASS));
        assert!(!check(0.02, 0.0, &WATER_BALANCE_MASS));
    }

    #[test]
    fn test_gpu_cpu_cross_validation() {
        let cpu = 3.88;
        let gpu = 3.880_03;
        assert!(check(gpu, cpu, &GPU_CPU_CROSS));
    }

    #[test]
    fn test_sensor_exact_precision() {
        assert!(check(42.000_000_001, 42.0, &SENSOR_EXACT));
        assert!(!check(42.001, 42.0, &SENSOR_EXACT));
    }

    #[test]
    fn test_richards_steady_tolerance() {
        assert!(check(0.350, 0.351, &RICHARDS_STEADY));
        assert!(!check(0.340, 0.351, &RICHARDS_STEADY));
    }

    #[test]
    fn test_isotherm_parameter_tolerance() {
        assert!(check(10.005, 10.0, &ISOTHERM_PARAMETER));
        assert!(!check(10.5, 10.0, &ISOTHERM_PARAMETER));
    }

    #[test]
    fn test_kriging_tolerance() {
        assert!(check(25.000_000_5, 25.0, &KRIGING_INTERPOLATION));
        assert!(!check(25.001, 25.0, &KRIGING_INTERPOLATION));
    }

    #[test]
    fn test_seasonal_reduction_tolerance() {
        assert!(check(1_000.000_000_005, 1_000.0, &SEASONAL_REDUCTION));
    }

    #[test]
    fn test_cold_climate_wide_tolerance() {
        assert!(check(0.3, 0.1, &ET0_COLD_CLIMATE));
        assert!(!check(1.0, 0.1, &ET0_COLD_CLIMATE));
    }

    #[test]
    fn test_all_tolerances_have_justification() {
        for tol in all_tolerances() {
            assert!(
                !tol.name.is_empty(),
                "tolerance {} must have a name",
                tol.name
            );
            assert!(
                !tol.justification.is_empty(),
                "tolerance {} must have justification",
                tol.name
            );
            assert!(tol.abs_tol > 0.0, "{}: abs_tol must be positive", tol.name);
        }
        // 58 Tolerance structs + 1 plain threshold (NPU_STRESS_DEPLETION_THRESHOLD)
        // + 1 plain count (NPU_MIN_ANOMALY_SAMPLES)
        assert_eq!(
            all_tolerances().len(),
            58,
            "test must include every Tolerance constant defined in this file"
        );
        let threshold = NPU_STRESS_DEPLETION_THRESHOLD;
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "stress threshold must be a fraction of TAW"
        );
    }
}

#[cfg(test)]
mod upstream_contract_tests {
    //! Verify airSpring tolerance constants align with upstream barraCuda contracts.
    //! Pattern: neuralSpring S174 upstream contract pinning.

    use super::*;

    #[test]
    fn all_tolerances_have_positive_thresholds() {
        for tol in all_tolerances() {
            assert!(
                tol.abs_tol > 0.0,
                "{}: abs_tol must be positive, got {}",
                tol.name,
                tol.abs_tol
            );
            assert!(
                tol.rel_tol >= 0.0,
                "{}: rel_tol must be non-negative, got {}",
                tol.name,
                tol.rel_tol
            );
        }
    }

    #[test]
    fn no_duplicate_tolerance_names() {
        let tols = all_tolerances();
        let mut seen = std::collections::HashSet::new();
        for tol in tols {
            assert!(
                seen.insert(tol.name),
                "duplicate tolerance name: {}",
                tol.name
            );
        }
    }

    #[test]
    fn all_tolerances_have_justification() {
        for tol in all_tolerances() {
            assert!(
                !tol.justification.is_empty(),
                "{}: justification must not be empty",
                tol.name
            );
        }
    }

    #[test]
    fn gpu_cpu_parity_tolerance_is_tighter_than_science() {
        const {
            assert!(
                GPU_CPU_CROSS.abs_tol <= ET0_REFERENCE.abs_tol,
                "GPU/CPU parity should be tighter than science tolerance",
            );
        }
    }

    #[test]
    fn tolerance_count_matches_registry() {
        let count = all_tolerances().len();
        assert!(
            count >= 58,
            "expected at least 58 tolerances in registry, found {count}"
        );
    }
}
