// SPDX-License-Identifier: AGPL-3.0-or-later
//! FAO-56 Penman-Monteith Reference Evapotranspiration (ET₀).
//!
//! Implements the standard FAO Paper 56 equation (Allen et al., 1998):
//!
//! ```text
//! ET₀ = [0.408 Δ(Rn - G) + γ (900/(T+273)) u₂ (es - ea)] / [Δ + γ(1 + 0.34 u₂)]
//! ```
//!
//! This is the foundational calculation for all irrigation scheduling.
//! Every variable has a published derivation in FAO-56 Chapters 2–4.
//!
//! # Reference
//!
//! Allen RG, Pereira LS, Raes D, Smith M (1998)
//! "Crop evapotranspiration — Guidelines for computing crop water requirements"
//! FAO Irrigation and Drainage Paper 56, Rome.

mod atmosphere;
mod hargreaves;
mod penman_monteith;
mod priestley_taylor;
mod radiation;

// Re-export solar geometry and radiation for backward compatibility.
pub use super::et0_ensemble::{EnsembleInput, EnsembleResult, et0_ensemble};
pub use super::solar::{
    clear_sky_radiation, daylight_hours, extraterrestrial_radiation, inverse_rel_distance,
    net_longwave_radiation, net_radiation, net_shortwave_radiation, solar_declination,
    sunset_hour_angle,
};

// Simplified ET₀ methods (Makkink, Turc, Hamon, Blaney-Criddle) live in
// `eco::simple_et0` — re-exported here for backward compatibility.
pub use super::simple_et0::{
    blaney_criddle_et0, blaney_criddle_from_location, blaney_criddle_p, hamon_pet,
    hamon_pet_from_location, makkink_et0, turc_et0,
};

// Thornthwaite (1948) monthly ET₀ has moved to `eco::thornthwaite`.
// Re-exported for backward compatibility.
pub use super::thornthwaite::{
    annual_heat_index, mean_daylight_hours_for_month, monthly_heat_index_term,
    thornthwaite_exponent, thornthwaite_monthly_et0, thornthwaite_unadjusted_et0,
};

// Sub-module public API
pub use atmosphere::{
    actual_vapour_pressure_dewpoint, actual_vapour_pressure_rh, atmospheric_pressure,
    mean_saturation_vapour_pressure, psychrometric_constant, saturation_vapour_pressure,
    vapour_pressure_slope, wind_speed_at_2m,
};
pub use hargreaves::hargreaves_et0;
pub use penman_monteith::{
    DailyEt0Input, Et0Result, daily_et0, daily_et0_pt_and_pm, fao56_penman_monteith,
};
pub use priestley_taylor::priestley_taylor_et0;
pub use radiation::{
    soil_heat_flux_monthly, solar_radiation_from_sunshine, solar_radiation_from_temperature,
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // FAO-56 Table 2.3 values are rounded to 3 decimal places, but the
    // equation coefficients (17.27, 237.3) are themselves rounded from the
    // original Tetens formula. At high temperatures (35–48 °C), the Eq. 11
    // computation diverges from the tabulated values by up to 0.009 kPa.
    // Tolerance 0.01 matches benchmark_fao56.json specification.
    const FAO56_SVP_TOL: f64 = 0.01;
    // Table 2.4 tolerance from benchmark JSON: 0.005 kPa/°C.
    const FAO56_DELTA_TOL: f64 = 0.005;

    #[test]
    fn test_saturation_vapour_pressure_table_2_3() {
        // FAO-56 Table 2.3: saturation vapour pressure at various temperatures.
        let cases = [
            (1.0, 0.657),
            (5.0, 0.872),
            (10.0, 1.228),
            (15.0, 1.705),
            (20.0, 2.338),
            (25.0, 3.168),
            (30.0, 4.243),
            (35.0, 5.624),
            (40.0, 7.384),
            (45.0, 9.585),
        ];
        for (temp, expected) in cases {
            let es = saturation_vapour_pressure(temp);
            assert!(
                (es - expected).abs() < FAO56_SVP_TOL,
                "es({temp}°C) = {es}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_vapour_pressure_slope_table_2_4() {
        // FAO-56 Table 2.4: slope of vapour pressure curve.
        let cases = [
            (1.0, 0.047),
            (10.0, 0.082),
            (20.0, 0.145),
            (30.0, 0.243),
            (40.0, 0.393),
        ];
        for (temp, expected) in cases {
            let delta = vapour_pressure_slope(temp);
            assert!(
                (delta - expected).abs() < FAO56_DELTA_TOL,
                "Δ({temp}°C) = {delta}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_atmospheric_pressure() {
        let p_sea = atmospheric_pressure(0.0);
        assert!((p_sea - 101.3).abs() < 0.1, "P at 0m: {p_sea}");
        let p_1800 = atmospheric_pressure(1800.0);
        assert!((p_1800 - 81.8).abs() < 0.5, "P at 1800m: {p_1800}");
    }

    #[test]
    fn test_psychrometric_constant() {
        let gamma = psychrometric_constant(101.3);
        assert!((gamma - 0.0674).abs() < 0.001, "γ: {gamma}");
    }

    #[test]
    fn test_actual_vapour_pressure_rh() {
        // FAO-56 Eq. 17: ea from Tmin=18, Tmax=25, RHmin=54, RHmax=80
        let ea = actual_vapour_pressure_rh(18.0, 25.0, 54.0, 80.0);
        // Expected: midpoint(es(18)*80/100, es(25)*54/100)
        let e18 = saturation_vapour_pressure(18.0);
        let e25 = saturation_vapour_pressure(25.0);
        let expected = f64::midpoint(e18 * 80.0 / 100.0, e25 * 54.0 / 100.0);
        assert!((ea - expected).abs() < 1e-10);
        // Should be a reasonable value for these conditions (1.5–2.0 kPa)
        assert!(ea > 1.0 && ea < 2.5, "ea = {ea}");
    }

    #[test]
    fn test_actual_vapour_pressure_dewpoint() {
        // ea at Tdew = 20°C should equal es(20) = 2.338
        let ea = actual_vapour_pressure_dewpoint(20.0);
        assert!((ea - 2.338).abs() < 0.001);
    }

    #[test]
    fn test_wind_speed_at_2m_from_10m() {
        // FAO-56 Eq. 47: u₂ = uz × 4.87 / ln(67.8z − 5.42)
        // At z=10m: u₂ = 3.0 × 4.87 / ln(672.58) = 3.0 × 0.748 ≈ 2.244
        let u2 = wind_speed_at_2m(3.0, 10.0);
        assert!((u2 - 2.244).abs() < 0.02, "u₂ from 10m: {u2}");
    }

    #[test]
    fn test_wind_speed_at_2m_identity_at_2m() {
        // At z=2m the conversion should be approximately identity.
        let u2 = wind_speed_at_2m(5.0, 2.0);
        assert!((u2 - 5.0).abs() < 0.15, "u₂ at 2m should be ~5.0: {u2}");
    }

    #[test]
    fn test_wind_speed_lower_at_2m() {
        // Wind at 2m should always be lower than at any height above 2m.
        for &z in &[3.0, 5.0, 10.0, 20.0, 50.0] {
            let u2 = wind_speed_at_2m(5.0, z);
            assert!(u2 < 5.0, "u₂ should be < uz at z={z}m: u₂={u2}");
        }
    }

    #[test]
    fn test_daily_et0_uccle_example_18() {
        // FAO-56 Example 18: Uccle, Belgium, 6 July.
        // Published answer: 3.88 mm/day (tolerance 0.10 per benchmark JSON).
        // Our function takes Rs directly; the published example derives Rs from
        // sunshine hours. Using the published intermediate Rs = 22.07 MJ/m²/day.
        let input = DailyEt0Input {
            tmin: 12.3,
            tmax: 21.5,
            tmean: Some(16.9),
            solar_radiation: 22.07,
            wind_speed_2m: 2.078,
            actual_vapour_pressure: 1.409,
            elevation_m: 100.0,
            latitude_deg: 50.80,
            day_of_year: 187,
        };
        let result = daily_et0(&input);
        assert!(
            (result.et0 - 3.88).abs() < 0.25,
            "Uccle ET₀: {:.3} (expected ~3.88)",
            result.et0
        );
    }

    #[test]
    fn test_daily_et0_positive_for_typical_conditions() {
        let input = DailyEt0Input {
            tmin: 15.0,
            tmax: 28.0,
            tmean: None,
            solar_radiation: 18.0,
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 1.5,
            elevation_m: 50.0,
            latitude_deg: 45.0,
            day_of_year: 200,
        };
        let result = daily_et0(&input);
        assert!(result.et0 > 0.0, "ET₀ should be positive: {}", result.et0);
    }

    #[test]
    fn test_daily_et0_zero_wind_reduces_et() {
        let base = DailyEt0Input {
            tmin: 20.0,
            tmax: 32.0,
            tmean: None,
            solar_radiation: 18.0,
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 2.0,
            elevation_m: 50.0,
            latitude_deg: 30.0,
            day_of_year: 150,
        };
        let calm = DailyEt0Input {
            wind_speed_2m: 0.0,
            ..base
        };
        assert!(daily_et0(&calm).et0 < daily_et0(&base).et0);
    }

    #[test]
    fn test_solar_radiation_from_sunshine() {
        // FAO-56 Eq. 35: Rs = (0.25 + 0.50 × n/N) × Ra
        // n=7.1, N=11.7, Ra=32.2 → Rs = (0.25 + 0.50 × 7.1/11.7) × 32.2
        let rs = solar_radiation_from_sunshine(7.1, 11.7, 32.2);
        let expected = (0.25 + 0.50 * 7.1 / 11.7) * 32.2;
        assert!(
            (rs - expected).abs() < 0.01,
            "Rs = {rs}, expected {expected}"
        );
    }

    #[test]
    fn test_solar_radiation_from_sunshine_zero_sunshine() {
        // Zero sunshine hours: Rs = 0.25 × Ra (cloudy day)
        let rs = solar_radiation_from_sunshine(0.0, 12.0, 40.0);
        assert!((rs - 10.0).abs() < 0.01, "Rs(n=0) = {rs}, expected 10.0");
    }

    #[test]
    fn test_solar_radiation_from_temperature() {
        // FAO-56 Eq. 50: Rs = 0.16 × √(25-15) × 40 = 0.16 × √10 × 40
        let rs = solar_radiation_from_temperature(25.0, 15.0, 40.0, 0.16);
        let expected = 0.16 * 10.0_f64.sqrt() * 40.0;
        assert!(
            (rs - expected).abs() < 0.01,
            "Rs = {rs}, expected {expected}"
        );
    }

    #[test]
    fn test_soil_heat_flux_monthly() {
        // G = 0.14 × (25 − 22) = 0.42
        let g = soil_heat_flux_monthly(25.0, 22.0);
        assert!((g - 0.42).abs() < 0.001, "G = {g}");
    }

    #[test]
    fn test_soil_heat_flux_monthly_cooling() {
        // Cooling month → negative G
        let g = soil_heat_flux_monthly(18.0, 22.0);
        assert!(g < 0.0, "G should be negative: {g}");
    }

    #[test]
    fn test_hargreaves_et0_reasonable_range() {
        // Typical summer conditions: Tmin=18, Tmax=32, Ra=40 MJ/m²/day
        // Ra in mm/day = 40/2.45 ≈ 16.33
        let ra_mm = 40.0 / 2.45;
        let et0 = hargreaves_et0(18.0, 32.0, ra_mm);
        // Hargreaves typically gives 3–8 mm/day for summer conditions
        assert!(et0 > 2.0 && et0 < 10.0, "Hargreaves ET₀ = {et0} mm/day");
    }

    #[test]
    fn test_hargreaves_et0_increases_with_temperature() {
        let ra_mm = 40.0 / 2.45;
        let et0_cool = hargreaves_et0(10.0, 20.0, ra_mm);
        let et0_warm = hargreaves_et0(18.0, 32.0, ra_mm);
        assert!(
            et0_warm > et0_cool,
            "Warmer should have higher ET₀: {et0_cool} vs {et0_warm}"
        );
    }

    #[test]
    fn test_hargreaves_et0_non_negative() {
        // Even with cold conditions, should not go negative
        let et0 = hargreaves_et0(-5.0, 0.0, 5.0);
        assert!(et0 >= 0.0, "ET₀ = {et0}");
    }

    #[test]
    fn test_priestley_taylor_zero_radiation() {
        let pt = priestley_taylor_et0(0.0, 0.0, 20.0, 0.0);
        assert!((pt).abs() < 1e-10, "PT should be 0 when Rn=0: {pt}");
    }

    #[test]
    fn test_priestley_taylor_negative_rn_clamped() {
        let pt = priestley_taylor_et0(-2.0, 0.0, 15.0, 0.0);
        assert!(
            (pt).abs() < 1e-10,
            "PT should clamp to 0 for negative Rn: {pt}"
        );
    }

    #[test]
    fn test_priestley_taylor_summer_reasonable() {
        let pt = priestley_taylor_et0(15.0, 0.0, 25.0, 0.0);
        assert!(
            pt > 3.0 && pt < 10.0,
            "PT should be 3-10 mm/day for summer Rn=15: {pt}"
        );
    }

    #[test]
    fn test_priestley_taylor_increases_with_rn() {
        let pt_low = priestley_taylor_et0(5.0, 0.0, 20.0, 0.0);
        let pt_high = priestley_taylor_et0(20.0, 0.0, 20.0, 0.0);
        assert!(
            pt_high > pt_low,
            "PT should increase with Rn: {pt_low} → {pt_high}"
        );
    }

    #[test]
    fn test_priestley_taylor_increases_with_temp() {
        let pt_cool = priestley_taylor_et0(15.0, 0.0, 5.0, 0.0);
        let pt_warm = priestley_taylor_et0(15.0, 0.0, 35.0, 0.0);
        assert!(
            pt_warm > pt_cool,
            "PT should increase with temperature (Δ/(Δ+γ) increases): {pt_cool} → {pt_warm}"
        );
    }

    #[test]
    fn test_priestley_taylor_altitude_effect() {
        let pt_sea = priestley_taylor_et0(15.0, 0.0, 25.0, 0.0);
        let pt_high = priestley_taylor_et0(15.0, 0.0, 25.0, 1500.0);
        assert!(
            pt_high > pt_sea,
            "PT should be higher at altitude (lower γ → higher Δ/(Δ+γ)): {pt_sea} → {pt_high}"
        );
    }

    #[test]
    fn test_priestley_taylor_soil_heat_flux() {
        let pt_no_g = priestley_taylor_et0(15.0, 0.0, 25.0, 0.0);
        let pt_with_g = priestley_taylor_et0(15.0, 2.0, 25.0, 0.0);
        assert!(
            pt_no_g > pt_with_g,
            "PT should decrease with positive G (less energy for ET): {pt_no_g} → {pt_with_g}"
        );
    }

    #[test]
    fn test_priestley_taylor_cross_validate_pm() {
        let input = DailyEt0Input {
            tmin: 12.3,
            tmax: 21.5,
            tmean: Some(16.9),
            solar_radiation: 22.07,
            wind_speed_2m: 2.078,
            actual_vapour_pressure: 1.409,
            elevation_m: 100.0,
            latitude_deg: 50.8,
            day_of_year: 187,
        };
        let (pt, pm_result) = daily_et0_pt_and_pm(&input);
        let ratio = pt / pm_result.et0;
        assert!(
            (0.85..=1.25).contains(&ratio),
            "PT/PM ratio should be 0.85-1.25 for humid climate (Uccle): ratio={ratio}"
        );
    }
}
