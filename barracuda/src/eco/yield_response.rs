// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crop yield response to water stress (FAO-56 Chapter 10).
//!
//! Implements the Stewart (1977) yield-water production function:
//!
//! ```text
//! Single-stage: Ya/Ymax = 1 − Ky × (1 − ETa/ETc)
//! Multi-stage:  Ya/Ymax = ∏ᵢ (1 − Kyᵢ × (1 − ETaᵢ/ETcᵢ))
//! ```
//!
//! Ky is the yield response factor from FAO-56 Table 24 (Doorenbos & Kassam 1979).
//! Values > 1.0 indicate proportionally greater yield loss than ET deficit
//! (sensitive crops/stages). Values < 1.0 indicate tolerance.
//!
//! References:
//! - Stewart (1977) Optimizing Crop Production through Control of Water
//!   and Salinity Levels in the Soil
//! - Doorenbos & Kassam (1979) FAO Irrigation and Drainage Paper 33
//! - Allen et al. (1998) FAO-56 Chapter 10, Eq. 90
//! - Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148

use crate::error::{AirSpringError, Result};

/// Yield response factor (Ky) for a crop, with optional per-stage breakdown.
#[derive(Debug, Clone)]
pub struct YieldResponseFactor {
    /// Seasonal Ky (FAO-56 Table 24).
    pub ky_total: f64,
    /// Per-growth-stage Ky values and stage length fractions.
    /// Each tuple: `(ky_stage, stage_length_fraction)`.
    pub stages: Option<Vec<(f64, f64)>>,
}

/// Result of a yield response calculation.
#[derive(Debug, Clone, Copy)]
pub struct YieldResult {
    /// Ya/Ymax ratio (can be negative for extreme stress with high Ky).
    pub yield_ratio: f64,
    /// Ya/Ymax clamped to `[0, 1]`.
    pub yield_ratio_clamped: f64,
}

/// Single-stage yield response (Stewart 1977).
///
/// `Ya/Ymax = 1 − Ky × (1 − ETa/ETc)`
///
/// # Arguments
/// * `ky` — Yield response factor (dimensionless, FAO-56 Table 24)
/// * `eta_etc_ratio` — Ratio of actual to potential ET (typically 0–1)
#[must_use]
pub fn yield_ratio_single(ky: f64, eta_etc_ratio: f64) -> f64 {
    ky.mul_add(-(1.0 - eta_etc_ratio), 1.0)
}

/// Multi-stage yield response (FAO-56 Eq. 90).
///
/// `Ya/Ymax = ∏ᵢ (1 − Kyᵢ × (1 − ETaᵢ/ETcᵢ))`
///
/// # Arguments
/// * `stages` — Slice of `(ky, eta_etc_ratio)` tuples per growth stage
///
/// # Errors
/// Returns `Err` if `stages` is empty.
pub fn yield_ratio_multistage(stages: &[(f64, f64)]) -> Result<f64> {
    if stages.is_empty() {
        return Err(AirSpringError::InvalidInput(
            "yield_ratio_multistage requires at least one stage".into(),
        ));
    }
    let ratio = stages
        .iter()
        .map(|&(ky, eta_etc)| ky.mul_add(-(1.0 - eta_etc), 1.0))
        .product();
    Ok(ratio)
}

/// Water use efficiency (WUE).
///
/// `WUE = yield_kg_ha / (eta_mm × 10)` \[kg/m³\]
///
/// Conversion: 1 mm over 1 ha = 10 m³.
///
/// # Errors
/// Returns `Err` if `eta_mm` is not positive.
pub fn water_use_efficiency(yield_kg_ha: f64, eta_mm: f64) -> Result<f64> {
    if eta_mm <= 0.0 {
        return Err(AirSpringError::InvalidInput(
            "water_use_efficiency requires positive ETa".into(),
        ));
    }
    Ok(yield_kg_ha / (eta_mm * 10.0))
}

/// Clamp a yield ratio to the physically meaningful range `[0, 1]`.
#[must_use]
pub const fn clamp_yield_ratio(ratio: f64) -> f64 {
    ratio.clamp(0.0, 1.0)
}

/// FAO-56 Table 24 Ky values for common crops.
///
/// Returns `(ky_total, Option<Vec<(ky_stage, stage_fraction)>>)`.
#[must_use]
pub fn ky_table(crop: &str) -> Option<YieldResponseFactor> {
    match crop {
        "corn" => Some(YieldResponseFactor {
            ky_total: 1.25,
            stages: Some(vec![
                (0.40, 0.25), // vegetative
                (1.50, 0.25), // flowering
                (0.50, 0.25), // yield formation
                (0.20, 0.25), // ripening
            ]),
        }),
        "wheat_spring" => Some(YieldResponseFactor {
            ky_total: 1.15,
            stages: Some(vec![(0.20, 0.25), (0.65, 0.20), (0.55, 0.30), (0.25, 0.25)]),
        }),
        "soybean" => Some(YieldResponseFactor {
            ky_total: 0.85,
            stages: Some(vec![(0.20, 0.30), (0.80, 0.30), (1.00, 0.40)]),
        }),
        "potato" => Some(YieldResponseFactor {
            ky_total: 1.10,
            stages: Some(vec![(0.45, 0.25), (0.80, 0.25), (0.70, 0.30), (0.20, 0.20)]),
        }),
        "tomato" => Some(YieldResponseFactor {
            ky_total: 1.05,
            stages: Some(vec![(0.40, 0.20), (1.10, 0.25), (0.80, 0.30), (0.40, 0.25)]),
        }),
        "alfalfa" => Some(YieldResponseFactor {
            ky_total: 1.10,
            stages: None,
        }),
        "winter_wheat" | "wheat_winter" => Some(YieldResponseFactor {
            ky_total: 1.00,
            stages: Some(vec![
                (0.20, 0.30), // vegetative
                (0.60, 0.20), // flowering
                (0.50, 0.25), // yield formation
                (0.10, 0.25), // ripening
            ]),
        }),
        "dry_bean" => Some(YieldResponseFactor {
            ky_total: 1.15,
            stages: Some(vec![
                (0.20, 0.25), // vegetative
                (1.10, 0.25), // flowering
                (0.75, 0.25), // yield formation
                (0.20, 0.25), // ripening
            ]),
        }),
        "sugarcane" => Some(YieldResponseFactor {
            ky_total: 1.20,
            stages: Some(vec![(0.75, 0.35), (0.50, 0.15), (1.00, 0.35), (0.10, 0.15)]),
        }),
        "apple" => Some(YieldResponseFactor {
            ky_total: 1.00,
            stages: None,
        }),
        "blueberry" => Some(YieldResponseFactor {
            ky_total: 0.80,
            stages: None,
        }),
        _ => None,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code may use unwrap for clarity")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_single_stage_no_stress() {
        assert!((yield_ratio_single(1.25, 1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_single_stage_mild() {
        assert!((yield_ratio_single(1.25, 0.9) - 0.875).abs() < TOL);
    }

    #[test]
    fn test_single_stage_severe() {
        assert!((yield_ratio_single(1.25, 0.5) - 0.375).abs() < TOL);
    }

    #[test]
    fn test_single_stage_total_stress() {
        assert!((yield_ratio_single(1.25, 0.0) - (-0.25)).abs() < TOL);
    }

    #[test]
    fn test_single_stage_wheat() {
        assert!((yield_ratio_single(1.15, 0.9) - 0.885).abs() < TOL);
    }

    #[test]
    fn test_multistage_corn_uniform() {
        let stages = vec![(0.40, 0.9), (1.50, 0.9), (0.50, 0.9), (0.20, 0.9)];
        let ratio = yield_ratio_multistage(&stages).unwrap();
        // 0.96 * 0.85 * 0.95 * 0.98 = 0.759696
        assert!((ratio - 0.759_696).abs() < 1e-3);
    }

    #[test]
    fn test_multistage_flower_only() {
        let stages = vec![(0.40, 1.0), (1.50, 0.7), (0.50, 1.0), (0.20, 1.0)];
        let ratio = yield_ratio_multistage(&stages).unwrap();
        assert!((ratio - 0.55).abs() < TOL);
    }

    #[test]
    fn test_multistage_empty_error() {
        assert!(yield_ratio_multistage(&[]).is_err());
    }

    #[test]
    fn test_wue_corn_irrigated() {
        let wue = water_use_efficiency(12_000.0, 500.0).unwrap();
        assert!((wue - 2.4).abs() < 0.01);
    }

    #[test]
    fn test_wue_zero_eta_error() {
        assert!(water_use_efficiency(12_000.0, 0.0).is_err());
    }

    #[test]
    fn test_clamp_negative() {
        assert!((clamp_yield_ratio(-0.25) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_clamp_normal() {
        assert!((clamp_yield_ratio(0.875) - 0.875).abs() < TOL);
    }

    #[test]
    fn test_ky_table_corn() {
        let ky = ky_table("corn").unwrap();
        assert!((ky.ky_total - 1.25).abs() < TOL);
        assert!(ky.stages.is_some());
        assert_eq!(ky.stages.unwrap().len(), 4);
    }

    #[test]
    fn test_ky_table_alfalfa() {
        let ky = ky_table("alfalfa").unwrap();
        assert!((ky.ky_total - 1.10).abs() < TOL);
        assert!(ky.stages.is_none());
    }

    #[test]
    fn test_ky_table_unknown() {
        assert!(ky_table("quinoa").is_none());
    }

    #[test]
    fn test_ky_table_all_crops() {
        let crops = [
            "corn",
            "wheat_spring",
            "soybean",
            "potato",
            "tomato",
            "alfalfa",
            "winter_wheat",
            "dry_bean",
            "sugarcane",
            "apple",
            "blueberry",
        ];
        for crop in &crops {
            assert!(ky_table(crop).is_some(), "missing Ky for {crop}");
        }
    }
}
