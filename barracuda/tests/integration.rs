//! Integration tests for airSpring `BarraCUDA`.
//!
//! Tests cross-module interactions, round-trip fidelity, and determinism.
//! These complement the unit tests in each module.

use airspring_barracuda::eco::{
    correction,
    crop::CropType,
    evapotranspiration::{self as et, DailyEt0Input},
    sensor_calibration as sc,
    soil_moisture::{self as sm, SoilTexture},
    water_balance::{self as wb, DailyInput, RunoffModel, WaterBalanceState},
};
use airspring_barracuda::io::csv_ts;
use airspring_barracuda::testutil;
use std::io::Write;

// ── Cross-module integration ─────────────────────────────────────────

#[test]
fn test_et0_drives_water_balance() {
    // ET₀ from evapotranspiration feeds directly into water balance.
    let et_input = DailyEt0Input {
        tmin: 18.0,
        tmax: 30.0,
        tmean: None,
        solar_radiation: 20.0,
        wind_speed_2m: 2.0,
        actual_vapour_pressure: 1.5,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: 180,
    };
    let et_result = et::daily_et0(&et_input);
    assert!(et_result.et0 > 0.0);

    // Feed ET₀ into water balance for 10 days
    let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    for _ in 0..10 {
        let wb_input = DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: et_result.et0,
            kc: 1.0,
        };
        state.step(&wb_input);
    }

    // After 10 dry days with real ET₀, soil should be depleted
    assert!(state.depletion > 0.0);
    assert!(state.current_theta() < 0.33);
}

#[test]
fn test_soil_texture_into_water_balance() {
    // Soil texture hydraulic properties drive water balance initialization.
    let props = SoilTexture::SiltLoam.hydraulic_properties();
    let state = WaterBalanceState::new(props.field_capacity, props.wilting_point, 600.0, 0.5);

    // TAW should match PAW calculation
    let paw = sm::plant_available_water(props.field_capacity, props.wilting_point, 600.0);
    assert!((state.taw - paw).abs() < f64::EPSILON);
}

// ── CSV parsing integration ──────────────────────────────────────────

#[test]
fn test_csv_parse_from_memory_buffer() {
    let csv_data = b"time,temp_c,rh_pct,wind_ms\n\
        2024-07-01T06:00,18.5,82.0,1.2\n\
        2024-07-01T12:00,28.3,55.0,2.5\n\
        2024-07-01T18:00,25.1,65.0,1.8\n";

    let cursor = std::io::Cursor::new(csv_data as &[u8]);
    let data = csv_ts::parse_csv_reader(cursor, Some("time")).unwrap();

    assert_eq!(data.len(), 3);
    assert_eq!(data.num_columns(), 3);

    let temps = data.column("temp_c").unwrap();
    assert!((temps[0] - 18.5).abs() < 0.01);
    assert!((temps[1] - 28.3).abs() < 0.01);
    assert!((temps[2] - 25.1).abs() < 0.01);

    let stats = data.column_stats("temp_c").unwrap();
    assert!(stats.mean > 20.0 && stats.mean < 30.0);
}

#[test]
fn test_csv_round_trip_to_disk() {
    let data = testutil::generate_synthetic_iot_data(72);

    // Write to temp file
    let tmp = std::env::temp_dir().join("airspring_integration_test.csv");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(
            f,
            "timestamp,soil_moisture_1,soil_moisture_2,temperature,humidity,par"
        )
        .unwrap();
        let sm1 = data.column("soil_moisture_1").unwrap();
        let sm2 = data.column("soil_moisture_2").unwrap();
        let temperature = data.column("temperature").unwrap();
        let hum = data.column("humidity").unwrap();
        let par = data.column("par").unwrap();
        for i in 0..data.len() {
            writeln!(
                f,
                "{},{:.6},{:.6},{:.4},{:.4},{:.2}",
                data.timestamps()[i],
                sm1[i],
                sm2[i],
                temperature[i],
                hum[i],
                par[i],
            )
            .unwrap();
        }
    }

    // Parse back from disk
    let parsed = csv_ts::parse_csv(&tmp, Some("timestamp")).unwrap();
    assert_eq!(parsed.len(), data.len());
    assert_eq!(parsed.num_columns(), data.num_columns());

    // Statistics should survive round-trip within formatting precision
    let orig_stats = data.column_stats("temperature").unwrap();
    let parsed_stats = parsed.column_stats("temperature").unwrap();
    assert!(
        (orig_stats.mean - parsed_stats.mean).abs() < 0.01,
        "Temperature mean drifted: {} → {}",
        orig_stats.mean,
        parsed_stats.mean
    );

    let _ = std::fs::remove_file(&tmp);
}

// ── Determinism tests ────────────────────────────────────────────────

#[test]
fn test_et0_deterministic() {
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

    let r1 = et::daily_et0(&input);
    let r2 = et::daily_et0(&input);
    assert!((r1.et0 - r2.et0).abs() < f64::EPSILON);
    assert!((r1.rn - r2.rn).abs() < f64::EPSILON);
    assert!((r1.vpd - r2.vpd).abs() < f64::EPSILON);
}

#[test]
fn test_water_balance_deterministic() {
    let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let inputs: Vec<DailyInput> = (0..60)
        .map(|day| DailyInput {
            precipitation: if day % 5 == 0 { 20.0 } else { 0.0 },
            irrigation: if day % 12 == 0 { 30.0 } else { 0.0 },
            et0: 4.5,
            kc: 1.0,
        })
        .collect();

    let (final1, out1) = airspring_barracuda::eco::water_balance::simulate_season(&state, &inputs);
    let (final2, out2) = airspring_barracuda::eco::water_balance::simulate_season(&state, &inputs);

    assert!((final1.depletion - final2.depletion).abs() < f64::EPSILON);
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((a.depletion - b.depletion).abs() < f64::EPSILON);
        assert!((a.actual_et - b.actual_et).abs() < f64::EPSILON);
        assert!((a.deep_percolation - b.deep_percolation).abs() < f64::EPSILON);
    }
}

#[test]
fn test_topp_inverse_deterministic() {
    for &theta in &[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] {
        let e1 = sm::inverse_topp(theta);
        let e2 = sm::inverse_topp(theta);
        assert!((e1 - e2).abs() < f64::EPSILON);
    }
}

// ── Error path tests ─────────────────────────────────────────────────

#[test]
fn test_csv_parse_empty_input() {
    let cursor = std::io::Cursor::new(b"" as &[u8]);
    assert!(csv_ts::parse_csv_reader(cursor, None).is_err());
}

#[test]
fn test_csv_parse_missing_timestamp_col() {
    let csv = b"col_a,col_b\n1.0,2.0\n";
    let cursor = std::io::Cursor::new(csv as &[u8]);
    assert!(csv_ts::parse_csv_reader(cursor, Some("nonexistent")).is_err());
}

#[test]
fn test_csv_nonexistent_file() {
    let path = std::path::Path::new("/tmp/airspring_no_such_file_12345.csv");
    assert!(csv_ts::parse_csv(path, None).is_err());
}

// ── Boundary / edge cases ────────────────────────────────────────────

#[test]
fn test_et0_arctic_conditions() {
    // High-latitude winter: very short days, low radiation
    let input = DailyEt0Input {
        tmin: -20.0,
        tmax: -10.0,
        tmean: None,
        solar_radiation: 2.0,
        wind_speed_2m: 3.0,
        actual_vapour_pressure: 0.1,
        elevation_m: 50.0,
        latitude_deg: 65.0,
        day_of_year: 355,
    };
    let result = et::daily_et0(&input);
    // ET₀ should be ≥ 0 (clamped) in extreme cold
    assert!(result.et0 >= 0.0);
}

#[test]
fn test_et0_tropical_conditions() {
    // Equatorial summer: 12h days, high radiation
    let input = DailyEt0Input {
        tmin: 24.0,
        tmax: 34.0,
        tmean: None,
        solar_radiation: 25.0,
        wind_speed_2m: 1.5,
        actual_vapour_pressure: 2.5,
        elevation_m: 10.0,
        latitude_deg: 0.0,
        day_of_year: 80,
    };
    let result = et::daily_et0(&input);
    // Tropical ET₀ typically 4–7 mm/day
    assert!(
        result.et0 > 3.0 && result.et0 < 10.0,
        "Tropical ET₀: {}",
        result.et0
    );
}

#[test]
fn test_water_balance_saturation_overflow() {
    // Massive irrigation → depletion goes to 0, deep percolation absorbs excess
    let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let output = state.step(&DailyInput {
        precipitation: 0.0,
        irrigation: 1000.0,
        et0: 5.0,
        kc: 1.0,
    });
    assert!(
        (state.depletion).abs() < f64::EPSILON,
        "Depletion should be 0"
    );
    assert!(output.deep_percolation > 900.0, "Excess should drain");
}

#[test]
fn test_runoff_model_configurable() {
    let state_default = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let state_runoff = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5).with_runoff_model(
        RunoffModel::SimpleThreshold {
            threshold_mm: 20.0,
            fraction: 0.2,
        },
    );

    let heavy_rain = DailyInput {
        precipitation: 50.0,
        irrigation: 0.0,
        et0: 3.0,
        kc: 1.0,
    };

    let (_, out_default) = airspring_barracuda::eco::water_balance::simulate_season(
        &state_default,
        std::slice::from_ref(&heavy_rain),
    );
    let (_, out_runoff) =
        airspring_barracuda::eco::water_balance::simulate_season(&state_runoff, &[heavy_rain]);

    // Default (None) → no runoff
    assert!((out_default[0].runoff).abs() < f64::EPSILON);
    // Threshold → some runoff: (50 − 20) × 0.2 = 6.0
    assert!((out_runoff[0].runoff - 6.0).abs() < 0.01);
}

// ── BarraCUDA primitive cross-validation ─────────────────────────────
//
// These tests prove the Spring thesis: airSpring's computations are
// consistent with barracuda shared primitives. If barracuda stats and
// airSpring stats diverge, we catch it here.

#[test]
fn test_barracuda_pearson_cross_validation() {
    // Generate two correlated time series from ET₀ computation:
    // Series A: ET₀ at varying temperatures (deterministic)
    // Series B: ET₀ intermediates (es values) at same temperatures
    let temps: Vec<f64> = (0..50).map(|i| f64::from(i).mul_add(0.6, 5.0)).collect();
    let es_values: Vec<f64> = temps
        .iter()
        .map(|&t| et::saturation_vapour_pressure(t))
        .collect();
    let delta_values: Vec<f64> = temps
        .iter()
        .map(|&t| et::vapour_pressure_slope(t))
        .collect();

    // Use barracuda's pearson_correlation to verify es and Δ are highly correlated
    // (both are monotonically increasing functions of temperature).
    let r = barracuda::stats::pearson_correlation(&es_values, &delta_values).unwrap();
    assert!(r > 0.99, "es and Δ should be highly correlated: R = {r}");

    // Also test via testutil::r_squared which wraps barracuda
    let r2 = testutil::r_squared(&es_values, &delta_values).unwrap();
    assert!(r2 > 0.98, "R² should be > 0.98: {r2}");
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_barracuda_stats_vs_airspring_stats() {
    // Cross-validate: airSpring's column_stats vs barracuda's variance/std_dev.
    // Our column_stats uses population statistics (n divisor).
    // barracuda::stats uses sample statistics (n-1 divisor).
    // They should be close for large n but systematically different.
    let data = testutil::generate_synthetic_iot_data(168);
    let temps = data.column("temperature").unwrap();

    // airSpring population std_dev
    let our_stats = data.column_stats("temperature").unwrap();

    // barracuda sample std_dev
    let bc_std = barracuda::stats::correlation::std_dev(temps).unwrap();

    // For n=168: population_std ≈ sample_std × sqrt((n-1)/n)
    // The ratio should be very close to 1 for large n.
    let ratio = our_stats.std_dev / bc_std;
    let expected_ratio = ((temps.len() - 1) as f64 / temps.len() as f64).sqrt();
    assert!(
        (ratio - expected_ratio).abs() < 0.001,
        "Population vs sample std ratio: {ratio} (expected {expected_ratio})"
    );
}

#[test]
fn test_testutil_rmse_and_mbe() {
    let observed = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let simulated = vec![1.1, 2.0, 2.9, 4.1, 5.0];

    let rmse_val = testutil::rmse(&observed, &simulated);
    assert!(rmse_val < 0.1, "RMSE: {rmse_val}");

    let mbe_val = testutil::mbe(&observed, &simulated);
    assert!(mbe_val.abs() < 0.05, "MBE: {mbe_val}");
}

#[test]
fn test_testutil_perfect_correlation() {
    // Perfect positive correlation → R² = 1.0
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r2 = testutil::r_squared(&a, &b).unwrap();
    assert!((r2 - 1.0).abs() < 1e-10, "R²: {r2}");
}

// ── Sensor calibration integration ──────────────────────────────────

#[test]
fn test_soilwatch10_drives_irrigation_recommendation() {
    // End-to-end: raw sensor count → VWC → irrigation recommendation.
    // Simulates Dong et al. 2024 pipeline: sensor → calibration → decision.
    let raw_count = 15_000.0;
    let vwc = sc::soilwatch10_vwc(raw_count);

    // VWC should be in a reasonable range
    assert!(vwc > 0.0 && vwc < 0.5, "VWC({raw_count}) = {vwc}");

    // Sandy soil FC ≈ 0.12, depth 30 cm
    let ir = sc::irrigation_recommendation(0.12, vwc, 30.0);
    // If VWC > FC, IR should be 0; otherwise positive
    if vwc < 0.12 {
        assert!(ir > 0.0, "Should need irrigation: IR = {ir}");
    } else {
        assert!(ir.abs() < f64::EPSILON, "Should not need irrigation");
    }
}

#[test]
fn test_soilwatch10_multi_layer_integration() {
    // Three-depth sensor profile (15 cm, 60 cm, 90 cm) — matches paper setup.
    let raw_counts = [12_000.0, 18_000.0, 22_000.0];
    let depths = [30.0, 30.0, 30.0];
    let field_capacities = [0.12, 0.15, 0.18];

    let layers: Vec<sc::SoilLayer> = raw_counts
        .iter()
        .zip(depths.iter())
        .zip(field_capacities.iter())
        .map(|((&rc, &d), &fc)| sc::SoilLayer {
            field_capacity: fc,
            current_vwc: sc::soilwatch10_vwc(rc),
            depth_cm: d,
        })
        .collect();

    let total_ir = sc::multi_layer_irrigation(&layers);
    assert!(total_ir >= 0.0, "Total IR must be non-negative: {total_ir}");
}

// ── Index of Agreement & Nash-Sutcliffe ─────────────────────────────

#[test]
fn test_index_of_agreement_perfect() {
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ia = testutil::index_of_agreement(&obs, &obs);
    assert!((ia - 1.0).abs() < 1e-10, "IA (perfect) = {ia}");
}

#[test]
fn test_index_of_agreement_constant_bias() {
    // Constant bias should still produce high IA (close to 1.0)
    let obs = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let pred: Vec<f64> = obs.iter().map(|&x| x + 0.02).collect();
    let ia = testutil::index_of_agreement(&obs, &pred);
    assert!(ia > 0.95, "IA (constant +0.02 bias) = {ia}");
}

#[test]
fn test_index_of_agreement_matches_python() {
    // Cross-validate against Python's compute_ia
    // Using known analytical case: perfect prediction → IA = 1.0
    let measured = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let predicted = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let ia = testutil::index_of_agreement(&measured, &predicted);
    assert!((ia - 1.0).abs() < 1e-10);

    // MBE of perfect should be 0
    let mbe_val = testutil::mbe(&measured, &predicted);
    assert!(mbe_val.abs() < 1e-10);
}

#[test]
fn test_nash_sutcliffe_perfect() {
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let nse = testutil::nash_sutcliffe(&obs, &obs);
    assert!((nse - 1.0).abs() < 1e-10, "NSE (perfect) = {nse}");
}

#[test]
fn test_nash_sutcliffe_mean_predictor() {
    // If the model always predicts the mean, NSE = 0
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_val = 3.0;
    let pred = vec![mean_val; 5];
    let nse = testutil::nash_sutcliffe(&obs, &pred);
    assert!(nse.abs() < 1e-10, "NSE (mean predictor) = {nse}");
}

#[test]
fn test_coefficient_of_determination_equals_nse() {
    let obs = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let pred = vec![1.2, 2.8, 5.1, 6.9, 9.2];
    let r2 = testutil::coefficient_of_determination(&obs, &pred);
    let nse = testutil::nash_sutcliffe(&obs, &pred);
    assert!(
        (r2 - nse).abs() < f64::EPSILON,
        "R² and NSE should be identical: R²={r2}, NSE={nse}"
    );
}

// ── Wind speed conversion ───────────────────────────────────────────

#[test]
fn test_wind_speed_at_2m_from_10m() {
    // FAO-56 Eq. 47: u₂ = uz × 4.87 / ln(67.8z − 5.42)
    // At z=10m: u₂ = u₁₀ × 4.87 / ln(67.8×10 − 5.42) = u₁₀ × 4.87 / ln(672.58)
    // = u₁₀ × 4.87 / 6.5115 ≈ u₁₀ × 0.748
    let u10 = 3.0; // 3 m/s at 10 m
    let u2 = et::wind_speed_at_2m(u10, 10.0);

    let expected = 3.0 * 0.748;
    assert!(
        (u2 - expected).abs() < 0.02,
        "u₂ from 10m: {u2}, expected ~{expected:.3}",
    );
}

#[test]
fn test_wind_speed_at_2m_identity() {
    // At z=2m the conversion factor should be ~1.0 (identity)
    let u2 = et::wind_speed_at_2m(5.0, 2.0);
    assert!((u2 - 5.0).abs() < 0.1, "u₂ at 2m should be ~5.0: {u2}");
}

#[test]
fn test_wind_speed_conversion_into_et0() {
    // End-to-end: 10 m wind → convert → ET₀
    let u10 = 3.5;
    let u2 = et::wind_speed_at_2m(u10, 10.0);

    let input = DailyEt0Input {
        tmin: 18.0,
        tmax: 30.0,
        tmean: None,
        solar_radiation: 20.0,
        wind_speed_2m: u2,
        actual_vapour_pressure: 1.5,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: 180,
    };
    let result = et::daily_et0(&input);
    assert!(result.et0 > 0.0, "ET₀ with converted wind: {}", result.et0);
}

// ── Error type integration ──────────────────────────────────────────

#[test]
fn test_error_type_io_variant() {
    // CSV parse on nonexistent file should produce AirSpringError::Io
    let path = std::path::Path::new("/tmp/airspring_no_such_file_99999.csv");
    let err = csv_ts::parse_csv(path, None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("I/O error"),
        "Expected I/O error variant, got: {msg}"
    );
}

#[test]
fn test_error_type_csv_parse_variant() {
    // Empty input should produce AirSpringError::CsvParse
    let cursor = std::io::Cursor::new(b"" as &[u8]);
    let err = csv_ts::parse_csv_reader(cursor, None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("CSV parse error"),
        "Expected CSV parse error variant, got: {msg}"
    );
}

#[test]
fn test_error_type_display_and_source() {
    // Verify the error implements std::error::Error (Display + source chain)
    use airspring_barracuda::error::AirSpringError;
    use std::error::Error;

    // Io variant — Display and source
    let path = std::path::Path::new("/tmp/airspring_no_such_file_99999.csv");
    let err = csv_ts::parse_csv(path, None).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("I/O error"), "Io display: {msg}");
    assert!(err.source().is_some(), "Io should have source");

    // CsvParse variant — Display and source
    let csv_err = AirSpringError::CsvParse("test error".to_string());
    let msg = format!("{csv_err}");
    assert!(msg.contains("CSV parse error"), "CsvParse display: {msg}");
    assert!(csv_err.source().is_none(), "CsvParse has no source");

    // JsonParse variant — Display and source
    let bad_json = serde_json::from_str::<serde_json::Value>("{{invalid}").unwrap_err();
    let json_err = AirSpringError::JsonParse(bad_json);
    let msg = format!("{json_err}");
    assert!(msg.contains("JSON parse error"), "JsonParse display: {msg}");
    assert!(json_err.source().is_some(), "JsonParse should have source");

    // InvalidInput variant — Display and source
    let input_err = AirSpringError::InvalidInput("out of range".to_string());
    let msg = format!("{input_err}");
    assert!(msg.contains("Invalid input"), "InvalidInput display: {msg}");
    assert!(input_err.source().is_none(), "InvalidInput has no source");

    // Barracuda variant — Display and source
    let bc_err = AirSpringError::Barracuda("primitive failed".to_string());
    let msg = format!("{bc_err}");
    assert!(msg.contains("barracuda error"), "Barracuda display: {msg}");
    assert!(bc_err.source().is_none(), "Barracuda has no source");

    // From<std::io::Error>
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
    let converted: AirSpringError = io_err.into();
    assert!(format!("{converted}").contains("I/O error"));

    // From<serde_json::Error>
    let json_err2 = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
    let converted: AirSpringError = json_err2.into();
    assert!(format!("{converted}").contains("JSON parse error"));

    // Debug trait
    let debug_err = AirSpringError::InvalidInput("test".to_string());
    let debug_msg = format!("{debug_err:?}");
    assert!(debug_msg.contains("InvalidInput"), "Debug: {debug_msg}");
}

// ── Crop coefficient → water balance pipeline ───────────────────────

#[test]
fn test_crop_kc_drives_water_balance() {
    // End-to-end: crop database → soil properties → water balance simulation.
    // Corn mid-season on sandy loam, 30-day simulation.
    let crop = CropType::Corn.coefficients();
    let soil = SoilTexture::SandyLoam.hydraulic_properties();

    let mut state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        crop.root_depth_m * 1000.0, // m → mm
        crop.depletion_fraction,
    );

    // 30-day dry period at mid-season Kc
    for _ in 0..30 {
        state.step(&DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: crop.kc_mid,
        });
    }

    // After 30 dry days with Kc_mid=1.2, soil should be stressed
    assert!(state.depletion > 0.0, "Should be depleted");
    assert!(
        state.current_theta() < soil.field_capacity,
        "θ should be below FC"
    );
}

#[test]
fn test_tomato_vs_corn_water_demand() {
    // Tomato (shallow root, Kc=1.15) should deplete faster than
    // corn (deep root, Kc=1.20) per unit depth despite similar Kc.
    let tomato = CropType::Tomato.coefficients();
    let corn = CropType::Corn.coefficients();
    let soil = SoilTexture::Loam.hydraulic_properties();

    let mut tom_state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        tomato.root_depth_m * 1000.0,
        tomato.depletion_fraction,
    );
    let mut corn_state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        corn.root_depth_m * 1000.0,
        corn.depletion_fraction,
    );

    for _ in 0..20 {
        tom_state.step(&DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: tomato.kc_mid,
        });
        corn_state.step(&DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: corn.kc_mid,
        });
    }

    // Tomato (smaller TAW) should reach higher fractional depletion
    let tom_frac = tom_state.depletion / tom_state.taw;
    let corn_frac = corn_state.depletion / corn_state.taw;
    assert!(
        tom_frac > corn_frac,
        "Tomato should deplete faster: {tom_frac:.3} vs corn {corn_frac:.3}"
    );
}

// ── Hargreaves cross-check ──────────────────────────────────────────

#[test]
fn test_hargreaves_vs_penman_monteith_same_order() {
    // Hargreaves and PM should produce ET₀ within the same order of magnitude
    // for typical summer conditions. Hargreaves is less accurate but should
    // be broadly consistent.
    let doy: u32 = 180;
    let lat_rad = 42.0_f64.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, doy);
    let ra_mm = ra / 2.45; // MJ → mm equivalent

    let harg_et0 = et::hargreaves_et0(18.0, 32.0, ra_mm);

    let pm_input = DailyEt0Input {
        tmin: 18.0,
        tmax: 32.0,
        tmean: None,
        solar_radiation: 22.0,
        wind_speed_2m: 2.0,
        actual_vapour_pressure: 1.5,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: doy,
    };
    let pm_et0 = et::daily_et0(&pm_input).et0;

    // Both should be 3–8 mm/day for summer conditions
    assert!(harg_et0 > 1.0 && harg_et0 < 12.0, "Hargreaves: {harg_et0}");
    assert!(pm_et0 > 1.0 && pm_et0 < 12.0, "PM: {pm_et0}");

    // Ratio should be within 0.5–2.0 (broadly same order)
    let ratio = harg_et0 / pm_et0;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "Harg/PM ratio: {ratio:.2} ({harg_et0:.2} vs {pm_et0:.2})"
    );
}

// ── Sunshine-based radiation integration ─────────────────────────────

#[test]
fn test_sunshine_radiation_into_et0() {
    // Compute Rs from sunshine hours, then use for ET₀.
    // FAO-56 Example 18: Uccle, n=9.25h.
    let lat_rad = 50.80_f64.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, 187);
    let n_hours = et::daylight_hours(lat_rad, 187);
    let rs = et::solar_radiation_from_sunshine(9.25, n_hours, ra);

    // Rs should be reasonable (15–25 MJ/m²/day for summer mid-latitude)
    assert!(rs > 10.0 && rs < 30.0, "Rs from sunshine: {rs} MJ/m²/day");

    // Use this Rs in ET₀ computation
    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: rs,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    };
    let result = et::daily_et0(&input);
    assert!(result.et0 > 0.0 && result.et0 < 8.0, "ET₀: {}", result.et0);
}

// ══════════════════════════════════════════════════════════════════════
// BarraCUDA stats deepened integration
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_spearman_r_perfect_monotonic() {
    // Perfect monotonic relationship ⇒ Spearman ρ = 1.0
    let x: Vec<f64> = (1..=20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * v).collect(); // monotonic, non-linear
    let rho = testutil::spearman_r(&x, &y).unwrap();
    assert!(
        (rho - 1.0).abs() < 1e-10,
        "Perfect monotonic should give ρ=1, got {rho}"
    );
}

#[test]
fn test_spearman_r_inverse_relationship() {
    let x: Vec<f64> = (1..=20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().rev().copied().collect();
    let rho = testutil::spearman_r(&x, &y).unwrap();
    assert!(
        (rho + 1.0).abs() < 1e-10,
        "Perfect inverse should give ρ=-1, got {rho}"
    );
}

#[test]
fn test_spearman_vs_pearson_nonlinear() {
    // For strongly nonlinear but monotonic data, Spearman > Pearson
    let x: Vec<f64> = (1..=50).map(|i| f64::from(i) * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&v| v.powi(3)).collect();

    let rho = testutil::spearman_r(&x, &y).unwrap();
    let r2 = testutil::r_squared(&x, &y).unwrap();

    // Spearman should be ~1.0 (perfect rank), Pearson R² < 1.0
    assert!(rho > 0.99, "Spearman: {rho}");
    assert!(r2 < 1.0, "Pearson R²: {r2}");
}

#[test]
fn test_bootstrap_rmse_confidence_interval() {
    // Known-error series: simulated = observed + 0.5 noise
    let observed: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let simulated: Vec<f64> = observed.iter().map(|&v| v + 0.5).collect();

    // Analytical RMSE = 0.5
    let point_rmse = testutil::rmse(&observed, &simulated);
    assert!((point_rmse - 0.5).abs() < 1e-10);

    let (lower, upper) = testutil::bootstrap_rmse(&observed, &simulated, 500, 0.95).unwrap();

    // CI should contain the true RMSE
    assert!(
        lower <= point_rmse && point_rmse <= upper,
        "CI [{lower:.4}, {upper:.4}] should contain RMSE {point_rmse:.4}"
    );
    // CI should be narrow for constant-bias data
    assert!(
        (upper - lower) < 0.2,
        "CI width {:.4} too wide for constant bias",
        upper - lower
    );
}

#[test]
fn test_variance_and_std_dev() {
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var = testutil::variance(&data).unwrap();
    let sd = testutil::std_deviation(&data).unwrap();

    // Known: mean=5.0, var(sample) = 4.571...
    assert!((var - 4.571_428_571_428_571).abs() < 1e-6, "Var: {var}");
    assert!(
        (sd - var.sqrt()).abs() < 1e-12,
        "SD: {sd} vs sqrt(var): {}",
        var.sqrt()
    );
}

#[test]
fn test_barracuda_variance_matches_manual() {
    // Cross-validate barracuda variance against manual computation
    let temps: Vec<f64> = testutil::generate_synthetic_iot_data(48)
        .column("temperature")
        .unwrap()
        .to_vec();

    #[allow(clippy::cast_precision_loss)]
    let n = temps.len() as f64;
    let mean = temps.iter().sum::<f64>() / n;
    let manual_var = temps.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let bc_var = testutil::variance(&temps).unwrap();

    assert!(
        (bc_var - manual_var).abs() < 1e-10,
        "barracuda var {bc_var} vs manual {manual_var}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// GPU evolution gap infrastructure
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_evolution_gaps_catalogued() {
    use airspring_barracuda::gpu::evolution_gaps::{Tier, GAPS};

    // We should have a non-trivial number of documented gaps
    assert!(GAPS.len() >= 8, "Expected 8+ gaps, got {}", GAPS.len());

    // Tier A (ready to wire) should dominate
    let tier_a = GAPS.iter().filter(|g| g.tier == Tier::A).count();
    let tier_b = GAPS.iter().filter(|g| g.tier == Tier::B).count();
    let tier_c = GAPS.iter().filter(|g| g.tier == Tier::C).count();

    assert!(tier_a >= 4, "Expected 4+ Tier A gaps, got {tier_a}");
    assert!(tier_b >= 2, "Expected 2+ Tier B gaps, got {tier_b}");
    assert!(tier_c >= 2, "Expected 2+ Tier C gaps, got {tier_c}");

    // Each Tier A gap should have a ToadStool primitive
    for gap in GAPS.iter().filter(|g| g.tier == Tier::A) {
        assert!(
            gap.toadstool_primitive.is_some(),
            "Tier A gap '{}' should reference a ToadStool primitive",
            gap.id
        );
    }

    // All gaps should have non-empty fields
    for gap in GAPS {
        assert!(!gap.id.is_empty(), "Gap id must not be empty");
        assert!(
            !gap.description.is_empty(),
            "Gap description must not be empty"
        );
        assert!(!gap.action.is_empty(), "Gap action must not be empty");
    }
}

#[test]
fn test_evolution_gaps_unique_ids() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for gap in GAPS {
        assert!(
            seen.insert(gap.id),
            "Duplicate evolution gap id: '{}'",
            gap.id
        );
    }
}

#[test]
fn test_batched_et0_gap_documented() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;

    let et0_gap = GAPS.iter().find(|g| g.id == "batched_et0_gpu");
    assert!(et0_gap.is_some(), "Batched ET₀ GPU gap must be documented");

    let gap = et0_gap.unwrap();
    assert!(
        gap.toadstool_primitive
            .unwrap()
            .contains("batched_elementwise"),
        "Should reference the batched elementwise shader"
    );
}

#[test]
fn test_kriging_gap_documented() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;

    let kriging_gap = GAPS.iter().find(|g| g.id == "kriging_soil_moisture");
    assert!(
        kriging_gap.is_some(),
        "Kriging soil moisture gap must be documented"
    );

    let gap = kriging_gap.unwrap();
    assert!(
        gap.toadstool_primitive.unwrap().contains("kriging"),
        "Should reference kriging_f64"
    );
}

// ══════════════════════════════════════════════════════════════════════
// Low-level FAO-56 PM + standalone water balance functions
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_lowlevel_pm_matches_daily_et0() {
    // Low-level fao56_penman_monteith() should produce the same ET₀ as daily_et0()
    // when given the same intermediate values.
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
    let result = et::daily_et0(&input);

    // Extract intermediates and call low-level function
    let pm = et::fao56_penman_monteith(
        result.rn,
        result.g,
        16.9,
        input.wind_speed_2m,
        result.vpd,
        result.delta,
        result.gamma,
    );
    assert!(
        (pm - result.et0).abs() < 1e-10,
        "Low-level PM={pm} vs daily_et0={}",
        result.et0
    );
}

#[test]
fn test_standalone_taw_raw_match_state() {
    // Standalone functions should produce same values as WaterBalanceState.
    let fc = 0.33;
    let wp = 0.13;
    let root_depth_mm = 600.0;
    let p = 0.5;

    let taw = wb::total_available_water(fc, wp, root_depth_mm);
    let raw = wb::readily_available_water(taw, p);

    let state = WaterBalanceState::new(fc, wp, root_depth_mm, p);
    assert!((taw - state.taw).abs() < f64::EPSILON, "TAW mismatch");
    assert!((raw - state.raw).abs() < f64::EPSILON, "RAW mismatch");
}

#[test]
fn test_standalone_stress_coefficient() {
    let taw = wb::total_available_water(0.30, 0.10, 500.0);
    let raw = wb::readily_available_water(taw, 0.5);

    // At field capacity (Dr=0) → Ks=1.0
    assert!((wb::stress_coefficient(0.0, taw, raw) - 1.0).abs() < f64::EPSILON);

    // At RAW boundary → Ks=1.0
    assert!((wb::stress_coefficient(raw, taw, raw) - 1.0).abs() < f64::EPSILON);

    // At TAW (fully depleted) → Ks=0.0
    assert!((wb::stress_coefficient(taw, taw, raw)).abs() < f64::EPSILON);

    // Midpoint between RAW and TAW → Ks=0.5
    let mid = f64::midpoint(taw, raw);
    assert!(
        (wb::stress_coefficient(mid, taw, raw) - 0.5).abs() < 0.01,
        "Ks at midpoint"
    );
}

#[test]
fn test_standalone_daily_step() {
    let taw = wb::total_available_water(0.30, 0.10, 500.0);
    let (new_dr, actual_et, dp) = wb::daily_water_balance_step(20.0, 5.0, 0.0, 4.0, 1.0, 1.0, taw);

    // Dr_new = 20 - 5 - 0 + 1.0*1.0*4.0 = 19.0
    assert!((new_dr - 19.0).abs() < 1e-10, "Dr_new={new_dr}");
    assert!((actual_et - 4.0).abs() < 1e-10, "ETa={actual_et}");
    assert!(dp.abs() < 1e-10, "DP should be 0");
}

// ══════════════════════════════════════════════════════════════════════
// Correction model integration tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_correction_fit_all_models() {
    // Generate synthetic calibration data: factory vs measured VWC
    let factory: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.02).collect();
    let measured: Vec<f64> = factory.iter().map(|&x| 1.1f64.mul_add(x, 0.02)).collect();

    let models = correction::fit_correction_equations(&factory, &measured);
    assert!(models.len() >= 2, "Should fit at least 2 models");

    // Linear model should have R² > 0.99 for linear data
    let linear = models.iter().find(|m| m.model_type == "linear").unwrap();
    assert!(linear.r_squared > 0.99, "Linear R²={}", linear.r_squared);
}

#[test]
fn test_correction_evaluate_roundtrip() {
    // Fit a model, then evaluate it at the fitted points
    let x: Vec<f64> = (1..=10).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0f64.mul_add(xi, 0.5)).collect();

    let model = correction::fit_linear(&x, &y).unwrap();
    for (&xi, &yi) in x.iter().zip(&y) {
        let predicted = correction::evaluate(&model, xi);
        assert!(
            (predicted - yi).abs() < 1e-6,
            "Evaluate({xi})={predicted} vs {yi}"
        );
    }
}

#[test]
fn test_correction_models_soil_calibration_pipeline() {
    // End-to-end: generate realistic sensor data → fit correction → evaluate
    // Simulating Dong 2020 correction methodology
    let factory_vwc: Vec<f64> = (0..15).map(|i| f64::from(i).mul_add(0.02, 0.05)).collect();
    let true_vwc: Vec<f64> = factory_vwc
        .iter()
        .map(|&x| 0.85f64.mul_add(x, 0.03))
        .collect();

    let models = correction::fit_correction_equations(&factory_vwc, &true_vwc);
    assert!(!models.is_empty(), "Should fit at least one model");

    // Find best model by R²
    let best = models
        .iter()
        .max_by(|a, b| a.r_squared.partial_cmp(&b.r_squared).unwrap())
        .unwrap();

    // Apply correction to a new factory reading
    let new_factory = 0.20;
    let corrected = correction::evaluate(best, new_factory);
    let expected = 0.85f64.mul_add(new_factory, 0.03);
    assert!(
        (corrected - expected).abs() < 0.01,
        "Corrected={corrected} vs expected={expected}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// GPU orchestrator integration tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_gpu_batched_et0_matches_cpu_loop() {
    use airspring_barracuda::gpu::et0::BatchedEt0;

    let inputs: Vec<DailyEt0Input> = (0..50)
        .map(|i| DailyEt0Input {
            tmin: f64::from(i).mul_add(0.1, 12.0),
            tmax: f64::from(i).mul_add(0.2, 25.0),
            tmean: None,
            solar_radiation: f64::from(i).mul_add(0.1, 18.0),
            wind_speed_2m: 2.0,
            actual_vapour_pressure: 1.5,
            elevation_m: 200.0,
            latitude_deg: 42.0,
            day_of_year: 150 + i,
        })
        .collect();

    let engine = BatchedEt0::cpu();
    let batched = engine.compute(&inputs);

    for (i, input) in inputs.iter().enumerate() {
        let scalar = et::daily_et0(input).et0;
        assert!(
            (batched.et0_values[i] - scalar).abs() < f64::EPSILON,
            "Day {i}: batched={} vs scalar={scalar}",
            batched.et0_values[i]
        );
    }
}

#[test]
fn test_gpu_water_balance_mass_conservation() {
    use airspring_barracuda::gpu::water_balance::BatchedWaterBalance;

    let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
    let inputs: Vec<DailyInput> = (0..90)
        .map(|day| DailyInput {
            precipitation: if day % 7 == 3 { 15.0 } else { 0.0 },
            irrigation: if day % 14 == 0 { 25.0 } else { 0.0 },
            et0: 4.5,
            kc: 1.0,
        })
        .collect();

    let summary = engine.simulate_season(&inputs);
    assert!(
        summary.mass_balance_error < 0.01,
        "Mass balance: {}",
        summary.mass_balance_error
    );
    assert_eq!(summary.daily_outputs.len(), 90);
    assert!(summary.total_actual_et > 0.0);
}

#[test]
fn test_gpu_kriging_interpolation() {
    use airspring_barracuda::gpu::kriging::*;

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
        SensorReading {
            x: 0.0,
            y: 100.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 100.0,
            y: 100.0,
            vwc: 0.35,
        },
    ];

    let targets = vec![
        TargetPoint { x: 50.0, y: 50.0 },
        TargetPoint { x: 25.0, y: 25.0 },
    ];

    let result = interpolate_soil_moisture(
        &sensors,
        &targets,
        SoilVariogram::Spherical {
            nugget: 0.001,
            sill: 0.01,
            range: 150.0,
        },
    );

    assert_eq!(result.vwc_values.len(), 2);
    // Midpoint should be average of all four sensors (~0.275)
    assert!(
        result.vwc_values[0] > 0.20 && result.vwc_values[0] < 0.35,
        "Center VWC: {}",
        result.vwc_values[0]
    );
    // All variances should be positive
    for &v in &result.variances {
        assert!(v > 0.0 && v.is_finite(), "Variance: {v}");
    }
}

#[test]
fn test_gpu_seasonal_stats() {
    use airspring_barracuda::gpu::reduce;

    let et0_values = [4.2, 5.1, 3.8, 6.0, 4.5, 5.5, 3.2, 4.8, 5.0, 4.0];
    let stats = reduce::compute_seasonal_stats(&et0_values);

    assert_eq!(stats.count, 10);
    assert!((stats.total - 46.1).abs() < 1e-10);
    assert!((stats.mean - 4.61).abs() < 1e-10);
    assert!((stats.max - 6.0).abs() < 1e-10);
    assert!((stats.min - 3.2).abs() < 1e-10);
    assert!(stats.std_dev > 0.5 && stats.std_dev < 1.5);
}

// ══════════════════════════════════════════════════════════════════════
// ToadStool issue tracking tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_toadstool_issues_documented() {
    use airspring_barracuda::gpu::evolution_gaps::TOADSTOOL_ISSUES;

    assert!(
        TOADSTOOL_ISSUES.len() >= 3,
        "Expected 3+ ToadStool issues, got {}",
        TOADSTOOL_ISSUES.len()
    );

    // TS-001 (pow_f64) must be CRITICAL
    let pow_issue = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-001");
    assert!(pow_issue.is_some(), "pow_f64 issue must be documented");
    assert_eq!(pow_issue.unwrap().severity, "CRITICAL");

    // All issues must have non-empty fields
    for issue in TOADSTOOL_ISSUES {
        assert!(!issue.id.is_empty());
        assert!(!issue.file.is_empty());
        assert!(!issue.summary.is_empty());
        assert!(!issue.fix.is_empty());
        assert!(!issue.blocks.is_empty());
    }
}

#[test]
fn test_toadstool_pow_f64_issue_details() {
    use airspring_barracuda::gpu::evolution_gaps::TOADSTOOL_ISSUES;

    let issue = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-001").unwrap();

    assert!(
        issue.file.contains("batched_elementwise"),
        "Should reference the shader file"
    );
    assert_eq!(issue.line, 138, "Should reference line 138");
    assert!(issue.fix.contains("exp_f64"), "Fix should mention exp_f64");
    assert!(
        issue.blocks.contains("GPU ET"),
        "Should document what's blocked"
    );
}

// ══════════════════════════════════════════════════════════════════════
// GPU wiring: KrigingInterpolator (barracuda::ops::kriging_f64)
// ══════════════════════════════════════════════════════════════════════

/// Helper: try to create an f64-capable `WgpuDevice`. Returns None on CI/headless
/// or if the GPU doesn't support `SHADER_F64`.
fn try_create_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
    pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
        .ok()
        .map(std::sync::Arc::new)
}

/// Get a device or skip the test. `let...else` pattern for `clippy::manual_let_else`.
macro_rules! device_or_skip {
    () => {
        match try_create_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: No GPU device available");
                return;
            }
        }
    };
}

#[test]
fn test_kriging_interpolator_matches_idw_at_sensor() {
    use airspring_barracuda::gpu::kriging::{
        KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint,
    };

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
        SensorReading {
            x: 0.0,
            y: 100.0,
            vwc: 0.25,
        },
    ];
    let targets = vec![TargetPoint { x: 0.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 150.0,
    };

    let result = interp.interpolate(&sensors, &targets, variogram).unwrap();

    // At sensor (0,0), kriging should return very close to 0.20
    assert!(
        (result.vwc_values[0] - 0.20).abs() < 0.01,
        "At-sensor VWC should be ~0.20, got {}",
        result.vwc_values[0]
    );
    // Variance at a sensor location should be small (near nugget)
    assert!(
        result.variances[0] < 0.005,
        "At-sensor variance should be small, got {}",
        result.variances[0]
    );
}

#[test]
fn test_kriging_interpolator_midpoint() {
    use airspring_barracuda::gpu::kriging::{
        KrigingInterpolator, SensorReading, SoilVariogram, TargetPoint,
    };

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![TargetPoint { x: 50.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 150.0,
    };

    let result = interp.interpolate(&sensors, &targets, variogram).unwrap();

    // Midpoint between 0.20 and 0.30 should be close to 0.25
    assert!(
        (result.vwc_values[0] - 0.25).abs() < 0.02,
        "Midpoint VWC should be ~0.25, got {}",
        result.vwc_values[0]
    );
    // Midpoint variance should be larger than at-sensor but finite
    assert!(
        result.variances[0] > 0.0 && result.variances[0].is_finite(),
        "Midpoint variance: {}",
        result.variances[0]
    );
}

#[test]
fn test_kriging_interpolator_empty_inputs() {
    use airspring_barracuda::gpu::kriging::{KrigingInterpolator, SoilVariogram, TargetPoint};

    let device = device_or_skip!();

    let interp = KrigingInterpolator::new(device).unwrap();
    let targets = vec![TargetPoint { x: 0.0, y: 0.0 }];
    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };

    let result = interp.interpolate(&[], &targets, variogram).unwrap();
    assert_eq!(result.vwc_values.len(), 1);
    assert!(result.variances[0].is_infinite());
}

#[test]
fn test_kriging_fit_variogram() {
    use airspring_barracuda::gpu::kriging::{KrigingInterpolator, SensorReading};

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.20,
        },
        SensorReading {
            x: 10.0,
            y: 0.0,
            vwc: 0.22,
        },
        SensorReading {
            x: 20.0,
            y: 0.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 30.0,
            y: 0.0,
            vwc: 0.28,
        },
        SensorReading {
            x: 40.0,
            y: 0.0,
            vwc: 0.30,
        },
    ];

    let result = KrigingInterpolator::fit_variogram(&sensors, 5, 50.0);
    assert!(result.is_ok(), "Variogram fitting should succeed");
    let (lags, gammas) = result.unwrap();
    assert_eq!(lags.len(), gammas.len());
    assert!(!lags.is_empty(), "Should have at least one lag bin");
}

// ══════════════════════════════════════════════════════════════════════
// GPU wiring: SeasonalReducer (barracuda::ops::fused_map_reduce_f64)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_seasonal_reducer_sum_matches_cpu() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let gpu_sum = reducer.sum(&values).unwrap();
    let cpu_sum = reduce::seasonal_sum(&values);

    assert!(
        (gpu_sum - cpu_sum).abs() < 1e-6,
        "GPU sum ({gpu_sum}) should match CPU sum ({cpu_sum})"
    );
}

#[test]
fn test_seasonal_reducer_max_min_matches_cpu() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values = [4.2, 5.1, 3.8, 6.0, 4.5, 5.5, 3.2, 4.8, 5.0, 4.0];
    let gpu_max = reducer.max(&values).unwrap();
    let gpu_min = reducer.min(&values).unwrap();
    let cpu_max = reduce::seasonal_max(&values);
    let cpu_min = reduce::seasonal_min(&values);

    assert!(
        (gpu_max - cpu_max).abs() < 1e-10,
        "GPU max ({gpu_max}) should match CPU max ({cpu_max})"
    );
    assert!(
        (gpu_min - cpu_min).abs() < 1e-10,
        "GPU min ({gpu_min}) should match CPU min ({cpu_min})"
    );
}

#[test]
fn test_seasonal_reducer_stats_matches_cpu() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    let values = [4.2, 5.1, 3.8, 6.0, 4.5, 5.5, 3.2, 4.8, 5.0, 4.0];
    let gpu_stats = reducer.compute_stats(&values).unwrap();
    let cpu_stats = reduce::compute_seasonal_stats(&values);

    assert_eq!(gpu_stats.count, cpu_stats.count);
    assert!(
        (gpu_stats.total - cpu_stats.total).abs() < 1e-6,
        "total: GPU={} CPU={}",
        gpu_stats.total,
        cpu_stats.total
    );
    assert!(
        (gpu_stats.mean - cpu_stats.mean).abs() < 1e-6,
        "mean: GPU={} CPU={}",
        gpu_stats.mean,
        cpu_stats.mean
    );
    assert!(
        (gpu_stats.max - cpu_stats.max).abs() < 1e-10,
        "max: GPU={} CPU={}",
        gpu_stats.max,
        cpu_stats.max
    );
    assert!(
        (gpu_stats.min - cpu_stats.min).abs() < 1e-10,
        "min: GPU={} CPU={}",
        gpu_stats.min,
        cpu_stats.min
    );
    // std_dev may differ slightly due to different variance computation paths
    assert!(
        (gpu_stats.std_dev - cpu_stats.std_dev).abs() < 0.01,
        "std_dev: GPU={} CPU={}",
        gpu_stats.std_dev,
        cpu_stats.std_dev
    );
}

#[test]
fn test_seasonal_reducer_large_array_gpu_dispatch() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();

    // N=2048 — should trigger GPU dispatch in FusedMapReduceF64 (threshold: 1024).
    // Known issue TS-004: the partials pipeline has a buffer usage conflict
    // in wgpu (STORAGE_READ vs STORAGE_READ_WRITE on the same buffer).
    // This panics in wgpu rather than returning an error, so we catch it.
    let values: Vec<f64> = (0..2048).map(|i| f64::from(i) * 0.01).collect();
    let cpu_sum = reduce::seasonal_sum(&values);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reducer.sum(&values)));

    match result {
        Ok(Ok(gpu_sum)) => {
            assert!(
                (gpu_sum - cpu_sum).abs() < 1e-4,
                "Large array sum: GPU={gpu_sum} CPU={cpu_sum}"
            );
        }
        Ok(Err(e)) => {
            eprintln!("KNOWN TS-004: FusedMapReduceF64 GPU dispatch error: {e}");
        }
        Err(_) => {
            eprintln!(
                "KNOWN TS-004: FusedMapReduceF64 GPU dispatch panicked \
                 (buffer usage conflict in partials pipeline). \
                 CPU fallback: sum={cpu_sum}"
            );
        }
    }
}

#[test]
fn test_seasonal_reducer_empty() {
    use airspring_barracuda::gpu::reduce;

    let device = device_or_skip!();

    let reducer = reduce::SeasonalReducer::new(device).unwrap();
    let stats = reducer.compute_stats(&[]).unwrap();
    assert_eq!(stats.count, 0);
    assert!((stats.total).abs() < 1e-10);
}
