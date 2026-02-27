// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thornthwaite (1948) monthly reference evapotranspiration (ET₀).
//!
//! Empirical temperature-based method using mean monthly temperatures and
//! latitude. Suitable for data-sparse regions where only monthly climate
//! normals are available. Conceptually distinct from physics-based daily
//! methods (Penman-Monteith, Priestley-Taylor).
//!
//! # Reference
//!
//! Thornthwaite C.W. (1948) "An approach toward a rational classification
//! of climate." Geographical Review 38(1):55–94.

use super::solar::daylight_hours;

/// Single-month contribution to the annual Thornthwaite heat index.
///
/// `i = (T/5)^1.514` for T > 0, else 0.
///
/// # Reference
/// Thornthwaite C.W. (1948) Geographical Review, 38(1):55-94.
#[must_use]
pub fn monthly_heat_index_term(tmean_c: f64) -> f64 {
    if tmean_c <= 0.0 {
        return 0.0;
    }
    (tmean_c / 5.0).powf(1.514)
}

/// Annual Thornthwaite heat index: I = Σ (Tᵢ/5)^1.514 for 12 months.
#[must_use]
pub fn annual_heat_index(monthly_temps: &[f64; 12]) -> f64 {
    monthly_temps
        .iter()
        .map(|&t| monthly_heat_index_term(t))
        .sum()
}

/// Thornthwaite exponent from annual heat index.
///
/// `a = 6.75×10⁻⁷·I³ − 7.71×10⁻⁵·I² + 1.792×10⁻²·I + 0.49239`
#[must_use]
pub fn thornthwaite_exponent(heat_index: f64) -> f64 {
    let i = heat_index;
    6.75e-7f64
        .mul_add(i, -7.71e-5)
        .mul_add(i, 1.792e-2)
        .mul_add(i, 0.49239)
}

/// Unadjusted monthly Thornthwaite ET₀ (mm/month for a 30-day month with 12-hr daylight).
///
/// For T > 26.5°C, applies the Willmott et al. (1985) high-temperature correction.
#[must_use]
pub fn thornthwaite_unadjusted_et0(tmean_c: f64, heat_index: f64, exponent_a: f64) -> f64 {
    if tmean_c <= 0.0 || heat_index <= 0.0 {
        return 0.0;
    }
    if tmean_c >= 26.5 {
        #[allow(clippy::suboptimal_flops)]
        return (-415.85 + 32.24 * tmean_c - 0.43 * tmean_c * tmean_c).max(0.0);
    }
    16.0 * (10.0 * tmean_c / heat_index).powf(exponent_a)
}

/// Mean daylight hours for a month at a given latitude.
///
/// Averages [`daylight_hours`] over every day in the month.
#[must_use]
pub fn mean_daylight_hours_for_month(latitude_deg: f64, month_index: usize) -> f64 {
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let doy_start: u32 = DAYS_IN_MONTH[..month_index].iter().sum::<u32>() + 1;
    let days = DAYS_IN_MONTH[month_index];
    let total: f64 = (0..days)
        .map(|d| daylight_hours(latitude_deg.to_radians(), doy_start + d))
        .sum();
    total / f64::from(days)
}

/// Full Thornthwaite monthly ET₀ (mm/month) for 12 months.
///
/// Applies daylight-hour and month-length corrections to the unadjusted estimate.
///
/// # Arguments
/// * `monthly_temps` — 12 mean monthly temperatures (°C), Jan–Dec
/// * `latitude_deg` — Station latitude (degrees)
///
/// # Returns
/// Array of 12 monthly ET₀ values (mm/month), adjusted for daylight and month length.
#[must_use]
pub fn thornthwaite_monthly_et0(monthly_temps: &[f64; 12], latitude_deg: f64) -> [f64; 12] {
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let hi = annual_heat_index(monthly_temps);
    if hi <= 0.0 {
        return [0.0; 12];
    }
    let a = thornthwaite_exponent(hi);
    let mut result = [0.0; 12];
    for m in 0..12 {
        let pet_unadj = thornthwaite_unadjusted_et0(monthly_temps[m], hi, a);
        let n_hours = mean_daylight_hours_for_month(latitude_deg, m);
        let d = f64::from(DAYS_IN_MONTH[m]);
        result[m] = (pet_unadj * (n_hours / 12.0) * (d / 30.0)).max(0.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thornthwaite_heat_index_term() {
        let hi = monthly_heat_index_term(25.0);
        assert!((hi - 11.435).abs() < 0.01, "5^1.514 ≈ 11.435: got {hi}");
        assert!(monthly_heat_index_term(-5.0).abs() < f64::EPSILON);
        assert!(monthly_heat_index_term(0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thornthwaite_annual_heat_index() {
        let hi = annual_heat_index(&[25.0; 12]);
        assert!((hi - 137.22).abs() < 0.5, "12 × 11.435 ≈ 137.22: got {hi}");
    }

    #[test]
    fn test_thornthwaite_freezing_zero() {
        let et0 = thornthwaite_monthly_et0(&[-10.0; 12], 42.0);
        assert!(et0.iter().all(|&e| e.abs() < f64::EPSILON));
    }

    #[test]
    fn test_thornthwaite_summer_gt_winter() {
        let temps = [
            -3.2, -2.1, 2.8, 9.1, 15.4, 21.3, 23.8, 22.5, 18.9, 12.1, 5.3, 0.8,
        ];
        let et0 = thornthwaite_monthly_et0(&temps, 42.73);
        let summer: f64 = et0[5..8].iter().sum();
        let winter = et0[0] + et0[1] + et0[11];
        assert!(summer > winter, "summer={summer:.1} winter={winter:.1}");
    }

    #[test]
    fn test_thornthwaite_annual_range() {
        let temps = [
            -3.2, -2.1, 2.8, 9.1, 15.4, 21.3, 23.8, 22.5, 18.9, 12.1, 5.3, 0.8,
        ];
        let et0 = thornthwaite_monthly_et0(&temps, 42.73);
        let annual: f64 = et0.iter().sum();
        assert!(
            (400.0..=900.0).contains(&annual),
            "annual ET₀={annual:.0} not in [400,900]"
        );
    }

    #[test]
    fn test_thornthwaite_monotonicity() {
        let mut prev = 0.0_f64;
        for t in [10.0, 15.0, 20.0, 25.0, 30.0] {
            let et0 = thornthwaite_monthly_et0(&[t; 12], 42.0);
            let annual: f64 = et0.iter().sum();
            assert!(annual > prev, "annual should increase: T={t} → {annual:.0}");
            prev = annual;
        }
    }

    #[test]
    fn test_thornthwaite_tropical() {
        let et0 = thornthwaite_monthly_et0(&[28.0; 12], 5.0);
        let annual: f64 = et0.iter().sum();
        assert!(
            (1000.0..=2000.0).contains(&annual),
            "tropical annual={annual:.0}"
        );
    }

    #[test]
    fn test_thornthwaite_single_warm_month() {
        let mut temps = [-5.0; 12];
        temps[6] = 20.0;
        let et0 = thornthwaite_monthly_et0(&temps, 42.0);
        assert!(et0[6] > 0.0, "warm month should have ET₀>0");
        for (i, &e) in et0.iter().enumerate() {
            if i != 6 {
                assert!(e.abs() < f64::EPSILON, "month {i} should be 0");
            }
        }
    }
}
