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
}

/// All registered Python baseline provenance records.
///
/// When a Python control script is rerun, update the commit and date here.
#[must_use]
pub const fn python_baselines() -> &'static [PythonBaseline] {
    &[
        PythonBaseline {
            binary: "validate_et0",
            script: Some("control/fao56/fao56_et0.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_soil",
            script: Some("control/soil_sensors/soil_sensor_calibration.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_water_balance",
            script: Some("control/water_balance/water_balance.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_dual_kc",
            script: Some("control/dual_kc/cover_crop_dual_kc.py"),
            commit: commits::PYTHON_PARITY_V2,
            date: "2026-02-25",
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
            script: Some("control/yield_response/stewart_yield.py"),
            commit: commits::PYTHON_PARITY_V1,
            date: "2026-02-19",
            category: BaselineCategory::Published,
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
            script: Some("control/mc_et0/mc_et0_uncertainty.py"),
            commit: commits::PYTHON_PARITY_V2,
            date: "2026-02-25",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_drought_index",
            script: Some("control/drought_index/spi_drought.py"),
            commit: commits::PYTHON_PARITY_V2,
            date: "2026-02-25",
            category: BaselineCategory::PythonParity,
        },
        PythonBaseline {
            binary: "validate_gpu_math",
            script: None,
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-26",
            category: BaselineCategory::GpuParity,
        },
        PythonBaseline {
            binary: "validate_atlas",
            script: Some("control/atlas/atlas_et0.py"),
            commit: commits::GPU_PARITY_V1,
            date: "2026-02-26",
            category: BaselineCategory::PythonParity,
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
            python_baselines().len() >= 10,
            "expected at least 10 baselines, got {}",
            python_baselines().len()
        );
    }
}
