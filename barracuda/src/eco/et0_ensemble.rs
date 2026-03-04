// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-method ET₀ ensemble consensus.
//!
//! Combines Penman-Monteith, Priestley-Taylor, Hargreaves, Makkink, Turc,
//! and Hamon into an equal-weight mean when data availability permits.

use super::evapotranspiration::{
    daily_et0, daily_et0_pt_and_pm, hamon_pet, hargreaves_et0, makkink_et0, turc_et0, DailyEt0Input,
};
use super::solar::{daylight_hours, extraterrestrial_radiation};

/// Input data for a multi-method ET₀ ensemble computation.
///
/// All fields are optional; the ensemble uses whichever methods
/// the available data supports.
pub struct EnsembleInput {
    /// Minimum temperature (°C). Enables PM, PT, Hargreaves.
    pub tmin: Option<f64>,
    /// Maximum temperature (°C). Enables PM, PT, Hargreaves.
    pub tmax: Option<f64>,
    /// Mean temperature (°C). Enables Makkink, Turc, Hamon.
    pub tmean: Option<f64>,
    /// Solar radiation Rs (MJ/m²/day). Enables PM, PT, Makkink, Turc.
    pub rs_mj: Option<f64>,
    /// Wind speed at 2 m (m/s). Enables PM.
    pub wind_speed_2m: Option<f64>,
    /// Actual vapour pressure ea (kPa). Enables PM, PT.
    pub actual_vapour_pressure: Option<f64>,
    /// Relative humidity (%). Enables Turc humidity correction.
    pub rh_pct: Option<f64>,
    /// Elevation above sea level (m).
    pub elevation_m: f64,
    /// Latitude (decimal degrees, positive = North).
    pub latitude_deg: f64,
    /// Day of year (1–366).
    pub day_of_year: u32,
    /// Day length (hours). Enables Hamon. Computed from lat/doy if absent.
    pub day_length_hours: Option<f64>,
}

/// Result of a multi-method ET₀ ensemble.
pub struct EnsembleResult {
    /// Equal-weight mean of available methods (mm/day).
    pub consensus: f64,
    /// Max − min of individual method estimates.
    pub spread: f64,
    /// Number of methods that contributed.
    pub n_methods: u8,
    /// Individual method results (NaN if method was not applicable).
    pub pm: f64,
    pub pt: f64,
    pub hargreaves: f64,
    pub makkink: f64,
    pub turc: f64,
    pub hamon: f64,
}

/// Build a [`DailyEt0Input`] from ensemble-level inputs and a pre-resolved mean temperature.
fn ensemble_daily_input(input: &EnsembleInput, tmean: f64) -> DailyEt0Input {
    DailyEt0Input {
        tmin: input.tmin.unwrap_or(0.0),
        tmax: input.tmax.unwrap_or(0.0),
        tmean: Some(tmean),
        solar_radiation: input.rs_mj.unwrap_or(0.0),
        wind_speed_2m: input.wind_speed_2m.unwrap_or(0.0),
        actual_vapour_pressure: input.actual_vapour_pressure.unwrap_or(0.0),
        elevation_m: input.elevation_m,
        latitude_deg: input.latitude_deg,
        day_of_year: input.day_of_year,
    }
}

/// Compute individual ET₀ estimates for each method, gated by data availability.
fn ensemble_methods(input: &EnsembleInput, tmean: f64) -> [f64; 6] {
    let has_full = input.tmin.is_some()
        && input.tmax.is_some()
        && input.rs_mj.is_some()
        && input.wind_speed_2m.is_some()
        && input.actual_vapour_pressure.is_some()
        && tmean.is_finite();
    let has_rad = input.rs_mj.is_some() && tmean.is_finite();

    let pm = if has_full {
        daily_et0(&ensemble_daily_input(input, tmean)).et0
    } else {
        f64::NAN
    };

    let pt = if has_full {
        daily_et0_pt_and_pm(&ensemble_daily_input(input, tmean)).0
    } else {
        f64::NAN
    };

    let mak = if has_rad {
        makkink_et0(tmean, input.rs_mj.unwrap_or(0.0), input.elevation_m)
    } else {
        f64::NAN
    };

    let trc = if has_rad && input.rh_pct.is_some() {
        turc_et0(
            tmean,
            input.rs_mj.unwrap_or(0.0),
            input.rh_pct.unwrap_or(60.0),
        )
    } else {
        f64::NAN
    };

    let hg = if input.tmin.is_some() && input.tmax.is_some() {
        let extra_rad =
            extraterrestrial_radiation(input.latitude_deg.to_radians(), input.day_of_year);
        let ra_mm = extra_rad / 2.45;
        hargreaves_et0(input.tmin.unwrap_or(0.0), input.tmax.unwrap_or(0.0), ra_mm)
    } else {
        f64::NAN
    };

    let dl = input
        .day_length_hours
        .unwrap_or_else(|| daylight_hours(input.latitude_deg.to_radians(), input.day_of_year));
    let ham = if tmean.is_finite() && tmean >= 0.0 && dl > 0.0 {
        hamon_pet(tmean, dl)
    } else {
        f64::NAN
    };

    [pm, pt, hg, mak, trc, ham]
}

/// Compute a multi-method ET₀ ensemble from available data.
///
/// Uses 6 daily ET₀ methods (Thornthwaite excluded — it's monthly).
/// Methods are gated by data availability: PM and PT need full weather,
/// Makkink/Turc need radiation, Hargreaves/Hamon need only temperature.
///
/// Returns a consensus estimate (equal-weight mean) and method spread.
#[must_use]
pub fn et0_ensemble(input: &EnsembleInput) -> EnsembleResult {
    let tmean = input
        .tmean
        .or_else(|| {
            input
                .tmin
                .zip(input.tmax)
                .map(|(lo, hi)| f64::midpoint(lo, hi))
        })
        .unwrap_or(f64::NAN);

    let [pm, pt, hg, mak, trc, ham] = ensemble_methods(input, tmean);

    let valid: Vec<f64> = [pm, pt, hg, mak, trc, ham]
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();

    let n = valid.len();
    if n == 0 {
        return EnsembleResult {
            consensus: 0.0,
            spread: 0.0,
            n_methods: 0,
            pm,
            pt,
            hargreaves: hg,
            makkink: mak,
            turc: trc,
            hamon: ham,
        };
    }

    let consensus = barracuda::stats::mean(&valid);
    let min_v = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    EnsembleResult {
        consensus,
        spread: max_v - min_v,
        n_methods: u8::try_from(n).unwrap_or(u8::MAX),
        pm,
        pt,
        hargreaves: hg,
        makkink: mak,
        turc: trc,
        hamon: ham,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_full_weather() {
        let input = EnsembleInput {
            tmin: Some(12.3),
            tmax: Some(21.5),
            tmean: Some(16.9),
            rs_mj: Some(22.07),
            wind_speed_2m: Some(2.078),
            actual_vapour_pressure: Some(1.409),
            rh_pct: Some(66.5),
            elevation_m: 100.0,
            latitude_deg: 50.8,
            day_of_year: 187,
            day_length_hours: Some(16.1),
        };
        let r = et0_ensemble(&input);
        assert_eq!(r.n_methods, 6, "Full weather should use 6 methods");
        assert!(r.consensus > 0.0, "Consensus should be positive");
        assert!(r.spread > 0.0, "Spread should be positive");
        assert!(r.pm.is_finite(), "PM should compute");
        assert!(r.pt.is_finite(), "PT should compute");
        assert!(r.hargreaves.is_finite(), "Hargreaves should compute");
        assert!(r.makkink.is_finite(), "Makkink should compute");
        assert!(r.turc.is_finite(), "Turc should compute");
        assert!(r.hamon.is_finite(), "Hamon should compute");
    }

    #[test]
    fn test_ensemble_temp_only() {
        let input = EnsembleInput {
            tmin: Some(20.0),
            tmax: Some(30.0),
            tmean: None,
            rs_mj: None,
            wind_speed_2m: None,
            actual_vapour_pressure: None,
            rh_pct: None,
            elevation_m: 100.0,
            latitude_deg: 42.0,
            day_of_year: 180,
            day_length_hours: Some(15.0),
        };
        let r = et0_ensemble(&input);
        assert!(r.n_methods >= 2, "Should use at least Hargreaves + Hamon");
        assert!(r.consensus > 0.0);
        assert!(r.pm.is_nan(), "PM needs full weather");
        assert!(r.pt.is_nan(), "PT needs full weather");
        assert!(r.makkink.is_nan(), "Makkink needs radiation");
    }

    #[test]
    fn test_ensemble_monotonicity() {
        let cool = et0_ensemble(&EnsembleInput {
            tmin: Some(5.0),
            tmax: Some(15.0),
            tmean: Some(10.0),
            rs_mj: Some(15.0),
            wind_speed_2m: Some(2.0),
            actual_vapour_pressure: Some(1.0),
            rh_pct: Some(60.0),
            elevation_m: 100.0,
            latitude_deg: 45.0,
            day_of_year: 180,
            day_length_hours: Some(15.0),
        });
        let warm = et0_ensemble(&EnsembleInput {
            tmin: Some(15.0),
            tmax: Some(25.0),
            tmean: Some(20.0),
            rs_mj: Some(15.0),
            wind_speed_2m: Some(2.0),
            actual_vapour_pressure: Some(1.5),
            rh_pct: Some(60.0),
            elevation_m: 100.0,
            latitude_deg: 45.0,
            day_of_year: 180,
            day_length_hours: Some(15.0),
        });
        assert!(
            warm.consensus > cool.consensus,
            "Warmer → higher ET₀: {} > {}",
            warm.consensus,
            cool.consensus
        );
    }

    #[test]
    fn test_ensemble_consensus_within_range() {
        let r = et0_ensemble(&EnsembleInput {
            tmin: Some(15.0),
            tmax: Some(25.0),
            tmean: Some(20.0),
            rs_mj: Some(15.0),
            wind_speed_2m: Some(2.0),
            actual_vapour_pressure: Some(1.2),
            rh_pct: Some(60.0),
            elevation_m: 100.0,
            latitude_deg: 45.0,
            day_of_year: 150,
            day_length_hours: Some(15.0),
        });
        let methods = [r.pm, r.pt, r.hargreaves, r.makkink, r.turc, r.hamon];
        let valid: Vec<f64> = methods.iter().copied().filter(|v| v.is_finite()).collect();
        let min_v = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            r.consensus >= min_v && r.consensus <= max_v,
            "Consensus {:.3} should be within [{:.3}, {:.3}]",
            r.consensus,
            min_v,
            max_v
        );
    }
}
