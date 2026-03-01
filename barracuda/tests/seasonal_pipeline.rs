// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `gpu::seasonal_pipeline` — seasonal agricultural pipeline.

mod common;

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};

use common::try_create_device;

const fn summer_day(doy: u32) -> WeatherDay {
    WeatherDay {
        tmax: 28.0,
        tmin: 16.0,
        rh_max: 85.0,
        rh_min: 50.0,
        wind_2m: 2.0,
        solar_rad: 22.0,
        precipitation: 0.0,
        elevation: 250.0,
        latitude_deg: 42.5,
        day_of_year: doy,
    }
}

fn growing_season() -> Vec<WeatherDay> {
    (121..=273)
        .map(|doy| {
            let mut day = summer_day(doy);
            if doy % 7 == 0 {
                day.precipitation = 8.0;
            }
            day
        })
        .collect()
}

#[test]
fn single_season_corn() {
    let pipeline = SeasonalPipeline::cpu();
    let config = CropConfig::standard(CropType::Corn);
    let weather = growing_season();

    let result = pipeline.run_season(&weather, &config);

    assert_eq!(result.n_days, 153);
    assert!(result.total_et0 > 400.0, "ET₀ = {:.0}", result.total_et0);
    assert!(
        result.yield_ratio > 0.5 && result.yield_ratio <= 1.0,
        "YR = {:.3}",
        result.yield_ratio
    );
    assert!(
        result.mass_balance_error < 0.1,
        "MB = {:.6}",
        result.mass_balance_error
    );
}

#[test]
fn empty_season() {
    let pipeline = SeasonalPipeline::cpu();
    let config = CropConfig::standard(CropType::Corn);
    let result = pipeline.run_season(&[], &config);
    assert_eq!(result.n_days, 0);
    assert!((result.yield_ratio - 1.0).abs() < f64::EPSILON);
}

#[test]
fn multi_crop_comparison() {
    let pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();

    let corn = pipeline.run_season(&weather, &CropConfig::standard(CropType::Corn));
    let soy = pipeline.run_season(&weather, &CropConfig::standard(CropType::Soybean));

    assert!(
        corn.total_actual_et > soy.total_actual_et,
        "Corn ET ({:.0}) > Soy ET ({:.0})",
        corn.total_actual_et,
        soy.total_actual_et
    );
}

#[test]
fn rainfed_vs_irrigated() {
    let pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();

    let irrigated = pipeline.run_season(&weather, &CropConfig::standard(CropType::Corn));

    let mut rainfed_config = CropConfig::standard(CropType::Corn);
    rainfed_config.irrigation_depth_mm = 0.0;
    let rainfed = pipeline.run_season(&weather, &rainfed_config);

    assert!(
        irrigated.yield_ratio >= rainfed.yield_ratio,
        "Irrigated YR ({:.3}) >= rainfed ({:.3})",
        irrigated.yield_ratio,
        rainfed.yield_ratio
    );
}

#[test]
fn daily_series_correct_length() {
    let pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();
    let config = CropConfig::standard(CropType::Corn);
    let result = pipeline.run_season(&weather, &config);
    assert_eq!(result.et0_daily.len(), weather.len());
    assert_eq!(result.actual_et_daily.len(), weather.len());
}

#[test]
fn deterministic() {
    let pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();
    let config = CropConfig::standard(CropType::Corn);
    let r1 = pipeline.run_season(&weather, &config);
    let r2 = pipeline.run_season(&weather, &config);
    assert!((r1.yield_ratio - r2.yield_ratio).abs() < f64::EPSILON);
    assert!((r1.total_et0 - r2.total_et0).abs() < f64::EPSILON);
}

#[test]
fn drought_lowers_yield() {
    let pipeline = SeasonalPipeline::cpu();
    let normal = growing_season();
    let drought: Vec<WeatherDay> = normal
        .iter()
        .map(|w| WeatherDay {
            precipitation: 0.0,
            ..*w
        })
        .collect();

    let mut config = CropConfig::standard(CropType::Corn);
    config.irrigation_depth_mm = 0.0;

    let normal_result = pipeline.run_season(&normal, &config);
    let drought_result = pipeline.run_season(&drought, &config);

    assert!(
        normal_result.yield_ratio > drought_result.yield_ratio,
        "Normal YR ({:.3}) > drought ({:.3})",
        normal_result.yield_ratio,
        drought_result.yield_ratio
    );
}

#[test]
fn mass_balance_all_crops() {
    let pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();

    for crop_type in &[
        CropType::Corn,
        CropType::Soybean,
        CropType::WinterWheat,
        CropType::Alfalfa,
        CropType::Potato,
    ] {
        let config = CropConfig::standard(*crop_type);
        let result = pipeline.run_season(&weather, &config);
        assert!(
            result.mass_balance_error < 0.1,
            "{:?} MB = {:.6}",
            crop_type,
            result.mass_balance_error
        );
    }
}

#[test]
fn backend_accessor() {
    use airspring_barracuda::gpu::seasonal_pipeline::Backend;

    let cpu = SeasonalPipeline::cpu();
    assert_eq!(cpu.backend(), Backend::Cpu);
}

#[test]
fn debug_format_cpu() {
    let pipeline = SeasonalPipeline::cpu();
    let dbg = format!("{pipeline:?}");
    assert!(dbg.contains("Cpu"));
    assert!(dbg.contains("false"));
}

#[test]
fn gpu_pipeline_matches_cpu() {
    use airspring_barracuda::gpu::seasonal_pipeline::Backend;

    let Some(device) = try_create_device() else {
        eprintln!("SKIP: No GPU device for SeasonalPipeline");
        return;
    };
    let gpu_pipeline = SeasonalPipeline::gpu(device).unwrap();
    let cpu_pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();
    let config = CropConfig::standard(CropType::Corn);

    let gpu_result = gpu_pipeline.run_season(&weather, &config);
    let cpu_result = cpu_pipeline.run_season(&weather, &config);

    assert_eq!(gpu_result.n_days, cpu_result.n_days);
    let et0_diff = (gpu_result.total_et0 - cpu_result.total_et0).abs();
    let et0_pct = et0_diff / cpu_result.total_et0 * 100.0;
    assert!(
        et0_pct < 1.0,
        "GPU↔CPU ET₀ {:.1} vs {:.1} ({:.2}% > 1% threshold)",
        gpu_result.total_et0,
        cpu_result.total_et0,
        et0_pct
    );
    assert!(
        (gpu_result.yield_ratio - cpu_result.yield_ratio).abs() < 0.05,
        "GPU YR {:.3} vs CPU {:.3}",
        gpu_result.yield_ratio,
        cpu_result.yield_ratio
    );
    assert_eq!(gpu_pipeline.backend(), Backend::GpuPerStage);
}

#[test]
fn gpu_pipeline_mass_balance() {
    let Some(device) = try_create_device() else {
        eprintln!("SKIP: No GPU device for SeasonalPipeline");
        return;
    };
    let pipeline = SeasonalPipeline::gpu(device).unwrap();
    let weather = growing_season();

    for crop_type in &[CropType::Corn, CropType::Soybean, CropType::WinterWheat] {
        let config = CropConfig::standard(*crop_type);
        let result = pipeline.run_season(&weather, &config);
        assert!(
            result.mass_balance_error < 0.5,
            "{:?} GPU MB = {:.6}",
            crop_type,
            result.mass_balance_error
        );
    }
}

#[test]
fn streaming_matches_cpu() {
    use airspring_barracuda::gpu::seasonal_pipeline::Backend;

    let cpu_pipeline = SeasonalPipeline::cpu();
    let weather = growing_season();
    let config = CropConfig::standard(CropType::Corn);

    let cpu_result = cpu_pipeline.run_season(&weather, &config);

    let Some(device) = try_create_device() else {
        eprintln!("SKIP: No GPU device for streaming_matches_cpu");
        return;
    };
    let streaming_pipeline = SeasonalPipeline::streaming(device).unwrap();
    let streaming_result = streaming_pipeline.streaming_et0_kc(&weather, &config);

    assert_eq!(streaming_result.n_days, cpu_result.n_days);
    let et0_diff = (streaming_result.total_et0 - cpu_result.total_et0).abs();
    let et0_pct = et0_diff / cpu_result.total_et0.max(1e-10) * 100.0;
    assert!(
        et0_pct < 1.0,
        "Streaming↔CPU ET₀ {:.1} vs {:.1} ({:.2}% > 1% threshold)",
        streaming_result.total_et0,
        cpu_result.total_et0,
        et0_pct
    );
    assert!(
        (streaming_result.yield_ratio - cpu_result.yield_ratio).abs() < 0.05,
        "Streaming YR {:.3} vs CPU {:.3}",
        streaming_result.yield_ratio,
        cpu_result.yield_ratio
    );
    assert_eq!(streaming_pipeline.backend(), Backend::GpuPipelined);
}
