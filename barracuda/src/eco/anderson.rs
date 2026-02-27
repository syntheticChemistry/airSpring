// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anderson soil-moisture coupling model.
//!
//! Couples volumetric water content θ to the Anderson localization model's
//! effective dimension `d_eff`, predicting quorum-sensing regime transitions.
//!
//! # Physics chain
//!
//! ```text
//! θ → S_e → p_c → z → d_eff → QS regime
//! ```
//!
//! - `S_e` = effective saturation (van Genuchten)
//! - `p_c` = pore connectivity (Mualem, 1976): `S_e^L`
//! - `z` = coordination number: `z_max × p_c`
//! - `d_eff` = effective dimension: `z / 2` (Bethe lattice)
//!
//! # References
//!
//! - van Genuchten (1980) SSSA J 44:892-898
//! - Mualem (1976) WRR 12:513-522
//! - Anderson (1958) Phys Rev 109:1492-1505
//! - Abrahams et al. (1979) J Phys C 12:2585

/// Maximum coordination number (3D cubic lattice).
pub const Z_MAX: f64 = 6.0;

/// Mualem pore-connectivity exponent.
pub const L_MUALEM: f64 = 0.5;

/// Disorder scale for heterogeneous soil.
pub const W_0: f64 = 12.0;

/// Anderson d=2 localization threshold.
pub const D_EFF_CRITICAL: f64 = 2.0;

/// Above this, clearly extended (QS active).
pub const D_EFF_EXTENDED: f64 = 2.5;

/// Quorum-sensing regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsRegime {
    /// `d_eff` ≤ 2.0 — all states localized, QS suppressed.
    Localized,
    /// 2.0 < `d_eff` ≤ 2.5 — near critical threshold.
    Marginal,
    /// `d_eff` > 2.5 — extended states, QS active.
    Extended,
}

impl QsRegime {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Localized => "localized",
            Self::Marginal => "marginal",
            Self::Extended => "extended",
        }
    }
}

/// Full coupling result.
#[derive(Debug, Clone, Copy)]
pub struct CouplingResult {
    pub theta: f64,
    pub se: f64,
    pub connectivity: f64,
    pub coordination: f64,
    pub d_eff: f64,
    pub disorder: f64,
    pub regime: QsRegime,
}

/// Effective saturation from volumetric water content.
#[must_use]
pub fn effective_saturation(theta: f64, theta_r: f64, theta_s: f64) -> f64 {
    if theta_s <= theta_r {
        return 0.0;
    }
    ((theta - theta_r) / (theta_s - theta_r)).clamp(0.0, 1.0)
}

/// Mualem pore connectivity: `p_c` = `S_e^L`.
#[must_use]
pub fn pore_connectivity(se: f64) -> f64 {
    if se <= 0.0 {
        return 0.0;
    }
    se.sqrt()
}

/// Coordination number: z = `z_max` × connectivity.
#[must_use]
pub fn coordination_number(connectivity: f64) -> f64 {
    Z_MAX * connectivity
}

/// Effective dimension from coordination number (Bethe lattice).
#[must_use]
pub fn effective_dimension(z: f64) -> f64 {
    z / 2.0
}

/// Anderson disorder parameter: W = `W_0` × (1 - `S_e`).
#[must_use]
pub fn disorder_parameter(se: f64) -> f64 {
    W_0 * (1.0 - se)
}

/// Classify QS regime from effective dimension.
#[must_use]
pub fn classify_regime(d_eff: f64) -> QsRegime {
    if d_eff > D_EFF_EXTENDED {
        QsRegime::Extended
    } else if d_eff > D_EFF_CRITICAL {
        QsRegime::Marginal
    } else {
        QsRegime::Localized
    }
}

/// Full coupling chain: θ → QS regime with all intermediates.
#[must_use]
pub fn coupling_chain(theta: f64, theta_r: f64, theta_s: f64) -> CouplingResult {
    let se = effective_saturation(theta, theta_r, theta_s);
    let connectivity = pore_connectivity(se);
    let coordination = coordination_number(connectivity);
    let d_eff = effective_dimension(coordination);
    let disorder = disorder_parameter(se);
    let regime = classify_regime(d_eff);
    CouplingResult {
        theta,
        se,
        connectivity,
        coordination,
        d_eff,
        disorder,
        regime,
    }
}

/// Batch coupling for a θ(t) time series.
#[must_use]
pub fn coupling_series(theta_series: &[f64], theta_r: f64, theta_s: f64) -> Vec<CouplingResult> {
    theta_series
        .iter()
        .map(|&theta| coupling_chain(theta, theta_r, theta_s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAND_TR: f64 = 0.045;
    const SAND_TS: f64 = 0.43;
    const LOAM_TR: f64 = 0.078;
    const LOAM_TS: f64 = 0.43;

    #[test]
    fn saturation_boundaries() {
        assert!((effective_saturation(SAND_TS, SAND_TR, SAND_TS) - 1.0).abs() < 1e-12);
        assert!(effective_saturation(SAND_TR, SAND_TR, SAND_TS).abs() < 1e-12);
    }

    #[test]
    fn d_eff_at_saturation() {
        let r = coupling_chain(SAND_TS, SAND_TR, SAND_TS);
        assert!((r.d_eff - 3.0).abs() < 1e-10);
        assert!(r.disorder.abs() < 1e-10);
        assert_eq!(r.regime, QsRegime::Extended);
    }

    #[test]
    fn d_eff_at_residual() {
        let r = coupling_chain(SAND_TR, SAND_TR, SAND_TS);
        assert!(r.d_eff.abs() < 1e-10);
        assert!((r.disorder - W_0).abs() < 1e-10);
        assert_eq!(r.regime, QsRegime::Localized);
    }

    #[test]
    fn monotonicity() {
        let mut prev = -1.0;
        for i in 0..=10 {
            let theta = LOAM_TR + f64::from(i) * (LOAM_TS - LOAM_TR) / 10.0;
            let r = coupling_chain(theta, LOAM_TR, LOAM_TS);
            assert!(r.d_eff >= prev - 1e-12);
            prev = r.d_eff;
        }
    }

    #[test]
    fn reference_se_050() {
        let theta = 0.5f64.mul_add(SAND_TS - SAND_TR, SAND_TR);
        let r = coupling_chain(theta, SAND_TR, SAND_TS);
        let expected_d = 3.0 * 0.5_f64.sqrt();
        assert!((r.d_eff - expected_d).abs() < 1e-10);
    }

    #[test]
    fn batch_matches_single() {
        let thetas = vec![0.10, 0.20, 0.30, 0.40];
        let batch = coupling_series(&thetas, LOAM_TR, LOAM_TS);
        for (i, &theta) in thetas.iter().enumerate() {
            let single = coupling_chain(theta, LOAM_TR, LOAM_TS);
            assert!((batch[i].d_eff - single.d_eff).abs() < 1e-15);
        }
    }
}
