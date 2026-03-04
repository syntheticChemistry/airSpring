// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sensor calibration equations.
//!
//! Implements calibration polynomials for field-deployable soil moisture and
//! environmental sensors.  These replace the Python baselines and are
//! promotion-ready for GPU (Tier C — new WGSL shaders for `IoT` pipelines).
//!
//! # References
//!
//! - Dong J, Werling B, Cao R, Li B (2024) "Implementation of an In-Field
//!   `IoT` System for Precision Irrigation Management"
//!   *Frontiers in Water* 6, 1353597. doi:10.3389/frwa.2024.1353597
//!
//! - Dong J, Miller R, Kelley C (2020) "Performance Evaluation of Soil
//!   Moisture Sensors in Coarse- and Fine-Textured Michigan Agricultural Soils"
//!   *Agriculture* 10(12), 598. doi:10.3390/agriculture10120598

// ── SoilWatch 10 Calibration (Dong et al. 2024, Eq. 5) ─────────────

/// `SoilWatch` 10 VWC calibration (Dong et al. 2024, Eq. 5).
///
/// Converts raw analog counts at 3.3 V to volumetric water content (cm³/cm³):
///
/// ```text
/// VWC = 2×10⁻¹³ RC³ − 4×10⁻⁹ RC² + 4×10⁻⁵ RC − 0.0677
/// ```
///
/// Ported from the Python baseline:
/// `control/iot_irrigation/calibration_dong2024.py::soilwatch10_vwc`
///
/// # Arguments
///
/// * `raw_count` — Analog raw count from `SoilWatch` 10 sensor at 3.3 V.
///
/// # Returns
///
/// Volumetric water content (cm³/cm³). May be negative for raw counts
/// below the calibration range — callers should clamp if needed.
#[must_use]
pub fn soilwatch10_vwc(raw_count: f64) -> f64 {
    // Horner's method for numerical stability and FMA precision:
    // VWC = ((2e-13 × RC − 4e-9) × RC + 4e-5) × RC − 0.0677
    2e-13_f64
        .mul_add(raw_count, -4e-9)
        .mul_add(raw_count, 4e-5)
        .mul_add(raw_count, -0.0677)
}

/// Vectorised `SoilWatch` 10 calibration over a slice of raw counts.
///
/// Returns a `Vec<f64>` of VWC values, one per input raw count.
#[must_use]
pub fn soilwatch10_vwc_vec(raw_counts: &[f64]) -> Vec<f64> {
    raw_counts.iter().copied().map(soilwatch10_vwc).collect()
}

// ── Irrigation Recommendation (Dong et al. 2024, Eq. 1) ────────────

/// Single-layer irrigation recommendation (Dong et al. 2024, Eq. 1).
///
/// ```text
/// IR = max(0, (FC − θv) × D)
/// ```
///
/// # Arguments
///
/// * `field_capacity` — Field capacity of the soil layer (cm³/cm³).
/// * `current_vwc`   — Current volumetric water content (cm³/cm³).
/// * `depth_cm`      — Representative soil layer depth (cm).
///
/// # Returns
///
/// Maximum irrigation recommendation (cm). Non-negative.
#[must_use]
pub fn irrigation_recommendation(field_capacity: f64, current_vwc: f64, depth_cm: f64) -> f64 {
    ((field_capacity - current_vwc) * depth_cm).max(0.0)
}

/// A single soil layer for multi-depth irrigation calculations.
#[derive(Debug, Clone, Copy)]
pub struct SoilLayer {
    /// Field capacity (cm³/cm³).
    pub field_capacity: f64,
    /// Current volumetric water content (cm³/cm³).
    pub current_vwc: f64,
    /// Layer depth (cm).
    pub depth_cm: f64,
}

/// Multi-layer irrigation recommendation.
///
/// Sums [`irrigation_recommendation`] across all sensor depths:
/// `IR_total` = Σ max(0, (`FCᵢ` − θvᵢ) × Dᵢ)
///
/// Ported from `control/iot_irrigation/calibration_dong2024.py::multi_layer_irrigation`.
#[must_use]
pub fn multi_layer_irrigation(layers: &[SoilLayer]) -> f64 {
    layers
        .iter()
        .map(|l| irrigation_recommendation(l.field_capacity, l.current_vwc, l.depth_cm))
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_soilwatch10_reference_point() {
        // Python baseline: soilwatch10_vwc(10_000) =
        //   2e-13 * 10000^3 - 4e-9 * 10000^2 + 4e-5 * 10000 - 0.0677
        //   = 2e-13 * 1e12 - 4e-9 * 1e8 + 0.4 - 0.0677
        //   = 0.2 - 0.4 + 0.4 - 0.0677 = 0.1323
        let vwc = soilwatch10_vwc(10_000.0);
        assert!(
            (vwc - 0.1323).abs() < 1e-4,
            "VWC(10000) = {vwc}, expected ~0.1323"
        );
    }

    #[test]
    fn test_soilwatch10_boundary_below_cal_range() {
        // At RC=0 the equation gives −0.0677 (below calibration range).
        assert!(soilwatch10_vwc(0.0) < 0.0);
    }

    #[test]
    fn test_soilwatch10_monotonic_in_valid_range() {
        // Within typical raw count range (~1000–40000), VWC should increase.
        let prev = soilwatch10_vwc(1000.0);
        let next = soilwatch10_vwc(20_000.0);
        assert!(next > prev, "prev={prev}, next={next}");
    }

    #[test]
    fn test_soilwatch10_vec_matches_scalar() {
        let counts = [5000.0, 10_000.0, 20_000.0, 30_000.0];
        let vec_result = soilwatch10_vwc_vec(&counts);
        for (i, &rc) in counts.iter().enumerate() {
            assert!(
                (vec_result[i] - soilwatch10_vwc(rc)).abs() < f64::EPSILON,
                "Mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_irrigation_recommendation_basic() {
        // FC=0.12, VWC=0.08, depth=30cm → IR = 0.04 × 30 = 1.2 cm
        let ir = irrigation_recommendation(0.12, 0.08, 30.0);
        assert!((ir - 1.2).abs() < 1e-10, "IR = {ir}");
    }

    #[test]
    fn test_irrigation_recommendation_at_field_capacity() {
        assert!(irrigation_recommendation(0.12, 0.12, 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_irrigation_recommendation_above_field_capacity() {
        assert!(irrigation_recommendation(0.12, 0.15, 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_layer_irrigation() {
        let layers = [
            SoilLayer {
                field_capacity: 0.12,
                current_vwc: 0.08,
                depth_cm: 30.0,
            },
            SoilLayer {
                field_capacity: 0.15,
                current_vwc: 0.10,
                depth_cm: 30.0,
            },
            SoilLayer {
                field_capacity: 0.18,
                current_vwc: 0.12,
                depth_cm: 30.0,
            },
        ];
        let ir = multi_layer_irrigation(&layers);
        let expected = 0.06f64.mul_add(30.0, 0.04f64.mul_add(30.0, 0.05 * 30.0));
        assert!(
            (ir - expected).abs() < 1e-10,
            "IR = {ir}, expected {expected}"
        );
    }
}
