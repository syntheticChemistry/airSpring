// SPDX-License-Identifier: AGPL-3.0-or-later
//! Standardized Precipitation Index (SPI) for drought classification.
//!
//! Implements the SPI algorithm (`McKee` et al., 1993):
//! 1. Accumulate precipitation over k months
//! 2. Fit gamma distribution (α, β) via Thom (1958) MLE
//! 3. Transform to standard normal via gamma CDF → inverse normal
//!
//! # GPU Promotion Path
//!
//! SPI computation is embarrassingly parallel across stations and time scales.
//! Each station's precipitation series can be independently processed:
//! `BatchedElementwise` (Tier B, op=SPI) or a dedicated `BatchedSpi` shader.
//!
//! # References
//!
//! - `McKee` TB, Doesken NJ, Kleist J (1993) Drought frequency and time scales.
//! - Edwards DC, `McKee` TB (1997) Characteristics of 20th century drought.
//! - WMO (2012) SPI User Guide. WMO-No. 1090.
//! - Thom HCS (1958) A note on the gamma distribution. Monthly Weather Rev 86(4).

use barracuda::special::gamma::regularized_gamma_p as upstream_gamma_p;
use barracuda::stats::normal::norm_ppf;

/// WMO drought classification category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroughtClass {
    ExtremelyWet,
    VeryWet,
    ModeratelyWet,
    NearNormal,
    ModeratelyDry,
    SeverelyDry,
    ExtremelyDry,
}

impl DroughtClass {
    /// Classify an SPI value per WMO guidelines.
    #[must_use]
    pub fn from_spi(spi: f64) -> Self {
        if spi >= 2.0 {
            Self::ExtremelyWet
        } else if spi >= 1.5 {
            Self::VeryWet
        } else if spi >= 1.0 {
            Self::ModeratelyWet
        } else if spi > -1.0 {
            Self::NearNormal
        } else if spi > -1.5 {
            Self::ModeratelyDry
        } else if spi > -2.0 {
            Self::SeverelyDry
        } else {
            Self::ExtremelyDry
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ExtremelyWet => "extremely_wet",
            Self::VeryWet => "very_wet",
            Self::ModeratelyWet => "moderately_wet",
            Self::NearNormal => "near_normal",
            Self::ModeratelyDry => "moderately_dry",
            Self::SeverelyDry => "severely_dry",
            Self::ExtremelyDry => "extremely_dry",
        }
    }
}

/// Gamma distribution parameters.
#[derive(Debug, Clone, Copy)]
pub struct GammaParams {
    /// Shape parameter α.
    pub alpha: f64,
    /// Scale parameter β (mean = α * β).
    pub beta: f64,
}

/// Fit gamma(α, β) to positive data via Thom (1958) MLE approximation.
///
/// Returns `None` if fewer than 3 positive values or if `A ≤ 0`.
#[must_use]
pub fn gamma_mle_fit(data: &[f64]) -> Option<GammaParams> {
    let positive: Vec<f64> = data.iter().copied().filter(|&x| x > 0.0).collect();
    let n = positive.len();
    if n < 3 {
        return None;
    }

    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;
    let mean_val: f64 = positive.iter().sum::<f64>() / nf;
    let log_mean: f64 = positive.iter().map(|x| x.ln()).sum::<f64>() / nf;
    let a_param = mean_val.ln() - log_mean;

    if a_param <= 0.0 {
        return None;
    }

    let alpha = (1.0 / (4.0 * a_param)) * (1.0 + (a_param.mul_add(4.0 / 3.0, 1.0)).sqrt());
    let beta = mean_val / alpha;

    Some(GammaParams { alpha, beta })
}

/// Gamma CDF: P(X ≤ x) for X ~ Gamma(α, β).
///
/// Delegates to `barracuda::special::gamma::regularized_gamma_p` (upstream).
/// Local `gamma_series`/`gamma_cf` removed in v0.7.5 (Write→Absorb→Lean).
#[must_use]
pub fn gamma_cdf(x: f64, params: &GammaParams) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    upstream_gamma_p(params.alpha, x / params.beta).unwrap_or(0.0)
}

/// Compute SPI at a given time scale.
///
/// `monthly_precip` is the precipitation series in mm.
/// `scale` is the accumulation window in months (1, 3, 6, 12, etc.).
///
/// Returns a vector of SPI values; `f64::NAN` for months with insufficient
/// history (first `scale - 1` months).
#[must_use]
pub fn compute_spi(monthly_precip: &[f64], scale: usize) -> Vec<f64> {
    let n = monthly_precip.len();
    let mut spi = vec![f64::NAN; n];

    if scale == 0 || n == 0 {
        return spi;
    }

    let mut accum = vec![f64::NAN; n];
    for i in (scale - 1)..n {
        let total: f64 = monthly_precip[(i + 1 - scale)..=i].iter().sum();
        accum[i] = total;
    }

    let valid: Vec<f64> = accum.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.len() < 3 {
        return spi;
    }

    let Some(params) = gamma_mle_fit(&valid) else {
        return spi;
    };

    #[allow(clippy::cast_precision_loss)]
    let q = valid.iter().filter(|&&x| x == 0.0).count() as f64 / valid.len() as f64;

    for i in 0..n {
        if accum[i].is_nan() {
            continue;
        }
        let prob = if accum[i] == 0.0 {
            q
        } else {
            (1.0 - q).mul_add(gamma_cdf(accum[i], &params), q)
        };
        let prob_clamped = prob.clamp(1e-10, 1.0 - 1e-10);
        spi[i] = norm_ppf(prob_clamped);
    }

    spi
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_fit_known() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0];
        let params = gamma_mle_fit(&data).unwrap();
        assert!(params.alpha > 0.0);
        assert!(params.beta > 0.0);
        #[allow(clippy::cast_precision_loss)]
        let mean_data = data.iter().sum::<f64>() / data.len() as f64;
        assert!(params.alpha.mul_add(params.beta, -mean_data).abs() < 0.1);
    }

    #[test]
    fn test_gamma_fit_insufficient() {
        assert!(gamma_mle_fit(&[1.0, 2.0]).is_none());
        assert!(gamma_mle_fit(&[]).is_none());
    }

    #[test]
    fn test_gamma_cdf_bounds() {
        let params = GammaParams {
            alpha: 2.0,
            beta: 5.0,
        };
        assert!((gamma_cdf(0.0, &params)).abs() < 1e-10);
        assert!(gamma_cdf(1.0, &params) > 0.0);
        assert!(gamma_cdf(100.0, &params) > 0.99);
    }

    #[test]
    fn test_spi_basic() {
        let precip = vec![50.0; 24];
        let spi = compute_spi(&precip, 1);
        assert_eq!(spi.len(), 24);
        for &v in &spi {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_spi_scale_nan_prefix() {
        let precip = vec![50.0; 12];
        let spi3 = compute_spi(&precip, 3);
        assert!(spi3[0].is_nan());
        assert!(spi3[1].is_nan());
        assert!(spi3[2].is_finite());
    }

    #[test]
    fn test_classify() {
        assert_eq!(DroughtClass::from_spi(2.5), DroughtClass::ExtremelyWet);
        assert_eq!(DroughtClass::from_spi(1.7), DroughtClass::VeryWet);
        assert_eq!(DroughtClass::from_spi(1.2), DroughtClass::ModeratelyWet);
        assert_eq!(DroughtClass::from_spi(0.0), DroughtClass::NearNormal);
        assert_eq!(DroughtClass::from_spi(-1.2), DroughtClass::ModeratelyDry);
        assert_eq!(DroughtClass::from_spi(-1.7), DroughtClass::SeverelyDry);
        assert_eq!(DroughtClass::from_spi(-2.5), DroughtClass::ExtremelyDry);
    }
}
