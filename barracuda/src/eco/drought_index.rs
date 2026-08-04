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

use crate::tolerances::POSITIVE_DATA_GUARD;

/// WMO drought classification category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroughtClass {
    /// SPI ≥ 2.0.
    ExtremelyWet,
    /// 1.5 ≤ SPI < 2.0.
    VeryWet,
    /// 1.0 ≤ SPI < 1.5.
    ModeratelyWet,
    /// −1.0 < SPI < 1.0.
    NearNormal,
    /// −1.5 < SPI ≤ −1.0.
    ModeratelyDry,
    /// −2.0 < SPI ≤ −1.5.
    SeverelyDry,
    /// SPI ≤ −2.0.
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

    let nf = crate::cast::usize_f64(n);
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
/// Pure-Rust regularized lower incomplete gamma function via series expansion
/// (small x/α) or continued fraction (large x/α). Matches the Numerical
/// Recipes approach used by upstream `barracuda::special::gamma`.
#[must_use]
pub fn gamma_cdf(x: f64, params: &GammaParams) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    regularized_gamma_p(params.alpha, x / params.beta)
}

/// Regularized lower incomplete gamma P(a, x) = γ(a,x) / Γ(a).
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for P(a, x) when x < a + 1.
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..200 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 3e-14 {
            break;
        }
    }
    sum * a.mul_add(x.ln(), -x - ln_gamma_a).exp()
}

/// Continued fraction for Q(a, x) = 1 - P(a, x) when x >= a + 1.
/// Lentz's modified algorithm.
#[expect(
    clippy::many_single_char_names,
    reason = "Numerical Recipes CF variables"
)]
fn gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / POSITIVE_DATA_GUARD;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200 {
        let fi = f64::from(i);
        let an = -fi * (fi - a);
        b += 2.0;
        d = an.mul_add(d, b);
        if d.abs() < POSITIVE_DATA_GUARD {
            d = POSITIVE_DATA_GUARD;
        }
        c = b + an / c;
        if c.abs() < POSITIVE_DATA_GUARD {
            c = POSITIVE_DATA_GUARD;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-14 {
            break;
        }
    }
    h * a.mul_add(x.ln(), -x - ln_gamma_a).exp()
}

/// Lanczos approximation to ln(Γ(x)) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 7] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
    ];
    const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
    let y = x - 1.0;
    let mut s = 0.999_999_999_999_81_f64;
    for (i, &coeff) in COEFFS.iter().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "i < 7, exact")]
        let idx = i as f64;
        s += coeff / (y + idx + 1.0);
    }
    let t = y + 7.5;
    (y + 0.5).mul_add(t.ln(), -t) + LN_SQRT_2PI + s.ln()
}

/// Inverse standard normal CDF (probit function).
///
/// Abramowitz & Stegun rational approximation (26.2.23). Accurate to ~4.5e-4
/// absolute error; adequate for SPI drought classification.
fn norm_ppf(p: f64) -> f64 {
    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < f64::EPSILON {
        return 0.0;
    }

    let half = p < 0.5;
    let pp = if half { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();

    let num = C2.mul_add(t, C1).mul_add(t, C0);
    let den = D3.mul_add(t, D2).mul_add(t, D1).mul_add(t, 1.0);
    let z = t - num / den;

    if half { -z } else { z }
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

    let q = crate::cast::usize_f64(valid.iter().filter(|&&x| x == 0.0).count())
        / crate::cast::usize_f64(valid.len());

    for i in 0..n {
        if accum[i].is_nan() {
            continue;
        }
        let prob = if accum[i] == 0.0 {
            q
        } else {
            (1.0 - q).mul_add(gamma_cdf(accum[i], &params), q)
        };
        let prob_clamped = prob.clamp(POSITIVE_DATA_GUARD, 1.0 - POSITIVE_DATA_GUARD);
        spi[i] = norm_ppf(prob_clamped);
    }

    spi
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_fit_known() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0];
        let params = gamma_mle_fit(&data).unwrap();
        assert!(params.alpha > 0.0);
        assert!(params.beta > 0.0);
        #[expect(clippy::cast_precision_loss, reason = "small test array, exact cast")]
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

    #[test]
    fn ln_gamma_known_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-10, "Γ(1) = 1, ln(1) = 0");
        let ln_g5 = ln_gamma(5.0);
        let expected = (24.0_f64).ln();
        assert!((ln_g5 - expected).abs() < 1e-8, "Γ(5) = 24");
    }

    #[test]
    fn regularized_gamma_p_known() {
        let p = regularized_gamma_p(1.0, 1.0);
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((p - expected).abs() < 1e-8, "P(1,1) ≈ 1-e⁻¹ ≈ 0.6321");
    }

    #[test]
    fn regularized_gamma_p_bounds() {
        assert!((regularized_gamma_p(2.0, 0.0)).abs() < 1e-15);
        assert!(regularized_gamma_p(2.0, 100.0) > 0.999);
    }

    #[test]
    fn norm_ppf_symmetry() {
        let z_low = norm_ppf(0.025);
        let z_high = norm_ppf(0.975);
        assert!((z_low + z_high).abs() < 0.01, "ppf(0.025) + ppf(0.975) ≈ 0");
        assert!(z_low < -1.9 && z_low > -2.0, "ppf(0.025) ≈ -1.96");
    }

    #[test]
    fn norm_ppf_median() {
        assert!((norm_ppf(0.5)).abs() < 1e-10, "ppf(0.5) = 0");
    }

    #[test]
    fn norm_ppf_extremes() {
        assert!(norm_ppf(0.0).is_infinite() && norm_ppf(0.0) < 0.0);
        assert!(norm_ppf(1.0).is_infinite() && norm_ppf(1.0) > 0.0);
    }

    #[test]
    fn spi_empty_and_zero_scale() {
        assert!(compute_spi(&[], 1).is_empty());
        let spi0 = compute_spi(&[50.0; 12], 0);
        assert!(spi0.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn spi_drought_detection() {
        let mut precip = vec![80.0; 12];
        precip.extend_from_slice(&[10.0; 12]);
        let spi = compute_spi(&precip, 1);
        let last = spi[23];
        assert!(last.is_finite());
        assert!(
            last < 0.0,
            "drought months should have negative SPI: {last}"
        );
    }
}
