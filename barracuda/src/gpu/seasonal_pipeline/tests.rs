// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::float_cmp,
    reason = "exact f64 comparisons verify GPU/CPU determinism"
)]
#![expect(clippy::expect_used, reason = "test assertions")]
#![expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]

use super::*;

fn sample_weather_day(doy: u32) -> WeatherDay {
    WeatherDay {
        tmax: 25.0 + f64::from(doy % 30),
        tmin: 12.0 + f64::from(doy % 15),
        rh_max: 85.0,
        rh_min: 45.0,
        wind_2m: 2.0,
        solar_rad: 22.0,
        precipitation: if doy.is_multiple_of(7) { 5.0 } else { 0.0 },
        elevation: 200.0,
        latitude_deg: 42.5,
        day_of_year: doy,
    }
}

#[test]
fn test_cpu_pipeline_construction() {
    let pipeline = SeasonalPipeline::cpu();
    assert_eq!(pipeline.backend(), Backend::Cpu);
    let dbg = format!("{pipeline:?}");
    assert!(dbg.contains("SeasonalPipeline"));
}

#[test]
fn test_crop_config_standard() {
    let config = CropConfig::standard(CropType::Corn);
    assert_eq!(config.field_capacity, 0.30);
    assert_eq!(config.wilting_point, 0.12);
    assert!(config.ky > 0.0 && config.ky < 2.0);
}

#[test]
fn test_run_season_empty_weather() {
    let pipeline = SeasonalPipeline::cpu();
    let config = CropConfig::standard(CropType::Corn);
    let result = pipeline.run_season(&[], &config);
    assert_eq!(result.n_days, 0);
    assert_eq!(result.total_et0, 0.0);
    assert_eq!(result.total_actual_et, 0.0);
    assert_eq!(result.yield_ratio, 1.0);
    assert!(result.et0_daily.is_empty());
}

#[test]
fn test_run_season_cpu_single_day() {
    let pipeline = SeasonalPipeline::cpu();
    let config = CropConfig::standard(CropType::Corn);
    let weather = vec![sample_weather_day(180)];
    let result = pipeline.run_season(&weather, &config);
    assert_eq!(result.n_days, 1);
    assert!(result.total_et0 > 0.0);
    assert!(result.total_actual_et > 0.0);
    assert!(result.yield_ratio > 0.0 && result.yield_ratio <= 1.0);
}

#[test]
fn test_streaming_et0_kc_cpu_fallback() {
    let pipeline = SeasonalPipeline::cpu();
    let config = CropConfig::standard(CropType::Soybean);
    let weather: Vec<WeatherDay> = (1..=30).map(sample_weather_day).collect();
    let result = pipeline.streaming_et0_kc(&weather, &config);
    assert_eq!(result.n_days, 30);
    assert!(result.total_et0 > 0.0);
    assert_eq!(result.et0_daily.len(), 30);
}

#[test]
fn test_run_season_with_et0_empty() {
    let config = CropConfig::standard(CropType::Corn);
    let result = SeasonalPipeline::run_season_with_et0(&[], &config, &[]);
    assert_eq!(result.n_days, 0);
    assert_eq!(result.yield_ratio, 1.0);
}

#[test]
fn test_compute_et0_batch_cpu_path() {
    let pipeline = SeasonalPipeline::cpu();
    let weather = vec![sample_weather_day(100), sample_weather_day(200)];
    let et0 = pipeline.compute_et0_batch(&weather);
    assert_eq!(et0.len(), 2);
    assert!(et0.iter().all(|&v| v.is_finite() && v >= 0.0));
}

#[test]
fn test_multi_field_cpu_fallback() {
    let pipeline = SeasonalPipeline::cpu();
    let weather: Vec<WeatherDay> = (120..=240).map(sample_weather_day).collect();
    let configs = [
        CropConfig::standard(CropType::Corn),
        CropConfig::standard(CropType::Soybean),
        CropConfig::standard(CropType::WinterWheat),
    ];
    let weather_refs: Vec<&[WeatherDay]> = vec![&weather; 3];

    let result = pipeline.run_multi_field(&weather_refs, &configs).unwrap();
    assert_eq!(result.fields.len(), 3);
    assert!(!result.gpu_wb_used, "CPU pipeline should not use GPU WB");
    assert_eq!(result.gpu_wb_dispatches, 0);

    for field in &result.fields {
        assert_eq!(field.n_days, 121);
        assert!(field.total_et0 > 0.0);
        assert!(field.yield_ratio > 0.0 && field.yield_ratio <= 1.0);
    }
}

#[test]
fn test_multi_field_empty() {
    let pipeline = SeasonalPipeline::cpu();
    let result = pipeline.run_multi_field(&[], &[]).unwrap();
    assert!(result.fields.is_empty());
    assert_eq!(result.gpu_wb_dispatches, 0);
}

#[test]
fn run_multi_field_rejects_mismatched_field_lengths() {
    let pipeline = SeasonalPipeline::cpu();
    let w1 = vec![sample_weather_day(1)];
    let w2 = vec![sample_weather_day(2), sample_weather_day(3)];
    let configs = [
        CropConfig::standard(CropType::Corn),
        CropConfig::standard(CropType::Corn),
    ];
    let err = pipeline
        .run_multi_field(&[&w1, &w2], &configs)
        .expect_err("expected InvalidConfig");
    match err {
        crate::error::AirSpringError::Pipeline(PipelineError::InvalidConfig(ref msg)) => {
            assert!(msg.contains("same number of days"), "message: {msg}");
        }
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn pipeline_error_invalid_config_display() {
    let e = PipelineError::InvalidConfig("test reason".into());
    let s = format!("{e}");
    assert!(s.contains("invalid pipeline configuration"));
    assert!(s.contains("test reason"));
}

#[test]
fn pipeline_error_device_init_display() {
    let e = PipelineError::DeviceInit(barracuda::error::BarracudaError::Device("no device".into()));
    assert!(format!("{e}").contains("init"));
    assert!(format!("{e}").contains("no device"));
}

#[test]
fn pipeline_error_shader_dispatch_display() {
    let e = PipelineError::ShaderDispatch(barracuda::error::BarracudaError::Gpu("dispatch".into()));
    let s = format!("{e}");
    assert!(s.contains("dispatch"));
    assert!(s.to_lowercase().contains("shader"));
}

#[test]
fn pipeline_error_unexpected_display() {
    let e = PipelineError::Unexpected("io".into());
    assert!(format!("{e}").contains("unexpected"));
}

#[test]
fn test_multi_field_parity_with_single() {
    let pipeline = SeasonalPipeline::cpu();
    let weather: Vec<WeatherDay> = (150..=200).map(sample_weather_day).collect();
    let config = CropConfig::standard(CropType::Corn);

    let single = pipeline.run_season(&weather, &config);
    let multi = pipeline
        .run_multi_field(&[&weather], std::slice::from_ref(&config))
        .unwrap();

    assert_eq!(multi.fields.len(), 1);
    let mf = &multi.fields[0];
    assert!(
        (single.total_et0 - mf.total_et0).abs() < 0.01,
        "ET₀ should match: single={:.2} multi={:.2}",
        single.total_et0,
        mf.total_et0
    );
    assert!(
        (single.yield_ratio - mf.yield_ratio).abs() < 0.01,
        "Yield should match: single={:.3} multi={:.3}",
        single.yield_ratio,
        mf.yield_ratio
    );
}
