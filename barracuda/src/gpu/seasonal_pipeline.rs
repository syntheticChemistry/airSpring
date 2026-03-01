// SPDX-License-Identifier: AGPL-3.0-or-later
//! Seasonal agricultural pipeline — zero-round-trip architecture.
//!
//! Chains the complete precision agriculture pipeline in a single call:
//!
//! ```text
//! Weather → ET₀ → Kc Adjust → Water Balance → Yield Response
//!   |         |        |            |               |
//!   GPU buf → GPU buf → GPU buf → GPU buf → GPU buf (future)
//! ```
//!
//! # Evolution Path
//!
//! | Phase | Architecture | Status |
//! |-------|-------------|--------|
//! | CPU chained | Sequential CPU, single API | Available |
//! | GPU per-stage | Stages 1-2 GPU, Stages 3-4 CPU | **Current** (S70+ absorption) |
//! | GPU pipelined | `StreamingPipeline` zero round-trip | **Wired** (fused batch until staging) |
//! | GPU fused | Both ET₀ + Kc in one batch, single poll | Same as pipelined (no staging yet) |
//! | Streaming | `UnidirectionalPipeline` for multi-year | Pending (Phase 4) |
//!
//! # Streaming / Fused Batch Gap
//!
//! `barracuda::staging::PipelineBuilder` and `StreamingPipeline` are available, but
//! `BatchedElementwiseF64` does not expose its pipeline/bind groups for composition.
//! Until barracuda adds fused dispatch (e.g. `execute_fused` or stage extraction),
//! `GpuPipelined` and `GpuFused` use sequential GPU dispatch: ET₀ batch → Kc batch,
//! each with its own submit-and-poll. Kc base values are pre-computed so both
//! dispatches can be issued without CPU round-trip between stages 1-2. Stages 3-4
//! (water balance + yield) remain on CPU due to day-over-day state dependency.
//!
//! # Usage
//!
//! ```rust,ignore
//! let pipeline = SeasonalPipeline::cpu();
//! let result = pipeline.run_season(&weather_days, &crop_config);
//! println!("Yield ratio: {:.3}", result.yield_ratio);
//! ```

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::eco::crop::{adjust_kc_for_climate, CropCoefficients, CropType};
use crate::eco::evapotranspiration::{self as et, DailyEt0Input};
use crate::eco::water_balance::{self as wb, DailyInput, WaterBalanceState};
use crate::eco::yield_response;
use crate::gpu::et0::{BatchedEt0, StationDay};
use crate::gpu::kc_climate::{BatchedKcClimate, KcClimateDay};

/// Daily weather observation for the seasonal pipeline.
#[derive(Debug, Clone, Copy)]
pub struct WeatherDay {
    /// Maximum temperature (°C).
    pub tmax: f64,
    /// Minimum temperature (°C).
    pub tmin: f64,
    /// Maximum relative humidity (%).
    pub rh_max: f64,
    /// Minimum relative humidity (%).
    pub rh_min: f64,
    /// Wind speed at 2 m (m/s).
    pub wind_2m: f64,
    /// Solar radiation (MJ/m²/day).
    pub solar_rad: f64,
    /// Precipitation (mm/day).
    pub precipitation: f64,
    /// Elevation (m).
    pub elevation: f64,
    /// Latitude (decimal degrees).
    pub latitude_deg: f64,
    /// Day of year (1–366).
    pub day_of_year: u32,
}

/// Crop and soil configuration for a seasonal simulation.
#[derive(Debug, Clone)]
pub struct CropConfig {
    /// Crop type (determines Kc schedule, root depth, Ky).
    pub crop_type: CropType,
    /// Soil field capacity (cm³/cm³).
    pub field_capacity: f64,
    /// Soil wilting point (cm³/cm³).
    pub wilting_point: f64,
    /// Crop height for Kc climate adjustment (m).
    pub crop_height_m: f64,
    /// Irrigation trigger threshold (fraction of TAW).
    pub irrigation_trigger: f64,
    /// Fixed irrigation depth (mm) when triggered.
    pub irrigation_depth_mm: f64,
    /// Total-season Ky for yield response.
    pub ky: f64,
}

impl CropConfig {
    /// Create a standard config for a crop type with default MI soil.
    #[must_use]
    pub fn standard(crop_type: CropType) -> Self {
        let kc = crop_type.coefficients();
        let ky = yield_response::ky_table(kc.name).map_or(1.25, |yrf| yrf.ky_total);
        Self {
            crop_type,
            field_capacity: 0.30,
            wilting_point: 0.12,
            crop_height_m: 2.0,
            irrigation_trigger: kc.depletion_fraction,
            irrigation_depth_mm: 25.0,
            ky,
        }
    }
}

/// Result of a complete seasonal simulation.
#[derive(Debug, Clone)]
pub struct SeasonResult {
    /// Number of simulated days.
    pub n_days: usize,
    /// Total seasonal ET₀ (mm).
    pub total_et0: f64,
    /// Total seasonal actual ET (mm).
    pub total_actual_et: f64,
    /// Total seasonal precipitation (mm).
    pub total_precipitation: f64,
    /// Total irrigation applied (mm).
    pub total_irrigation: f64,
    /// Number of stress days (Ks < 1.0).
    pub stress_days: usize,
    /// Maximum mass balance error (mm).
    pub mass_balance_error: f64,
    /// Final yield ratio (Ya/Ymax, 0–1).
    pub yield_ratio: f64,
    /// Daily ET₀ series (mm/day).
    pub et0_daily: Vec<f64>,
    /// Daily actual ET series (mm/day).
    pub actual_et_daily: Vec<f64>,
}

/// Seasonal agricultural pipeline orchestrator.
///
/// Chains ET₀ → Kc adjustment → water balance → yield response in a
/// single `run_season()` call. Stages 1-2 (ET₀ + Kc) dispatch to GPU
/// when a device is available; remaining stages use CPU.
pub struct SeasonalPipeline {
    backend: Backend,
    gpu_et0: Option<BatchedEt0>,
    gpu_kc: Option<BatchedKcClimate>,
}

impl std::fmt::Debug for SeasonalPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeasonalPipeline")
            .field("backend", &self.backend)
            .field("gpu_et0", &self.gpu_et0.is_some())
            .field("gpu_kc", &self.gpu_kc.is_some())
            .finish()
    }
}

/// Backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU chained (all stages validated).
    #[default]
    Cpu,
    /// GPU per-stage (Tier B, each stage independent).
    GpuPerStage,
    /// GPU pipelined (zero round-trip when staging wired; fused batch until then).
    GpuPipelined,
    /// GPU fused: both ET₀ and Kc stages in one batch, single device poll.
    /// Same as `GpuPipelined` until `BatchedElementwiseF64` exposes fused dispatch.
    GpuFused,
}

impl SeasonalPipeline {
    /// Create a CPU-chained seasonal pipeline.
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            gpu_et0: None,
            gpu_kc: None,
        }
    }

    /// Create a GPU per-stage pipeline with GPU dispatch for Stages 1-2.
    ///
    /// Stage 1 (ET₀) dispatches via `BatchedEt0` (op=0);
    /// Stage 2 (Kc climate adjustment) dispatches via `BatchedKcClimate` (op=7).
    /// Remaining stages use CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device cannot initialise.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let gpu_et0 = BatchedEt0::gpu(Arc::clone(&device))?;
        let gpu_kc = BatchedKcClimate::gpu(device)?;
        Ok(Self {
            backend: Backend::GpuPerStage,
            gpu_et0: Some(gpu_et0),
            gpu_kc: Some(gpu_kc),
        })
    }

    /// Create a streaming/pipelined GPU pipeline (ET₀ + Kc in single batch).
    ///
    /// Uses `Backend::GpuPipelined`. Chains ET₀ (op=0) → Kc climate adjust (op=7)
    /// with minimal CPU round-trip. Stages 3-4 (water balance + yield) remain on CPU.
    ///
    /// Falls back to `GpuPerStage` if GPU init fails (same engines, different backend tag).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device cannot initialise.
    pub fn streaming(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let gpu_et0 = BatchedEt0::gpu(Arc::clone(&device))?;
        let gpu_kc = BatchedKcClimate::gpu(device)?;
        Ok(Self {
            backend: Backend::GpuPipelined,
            gpu_et0: Some(gpu_et0),
            gpu_kc: Some(gpu_kc),
        })
    }

    /// Which backend this pipeline was configured with.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Run a complete seasonal simulation.
    ///
    /// Chains: weather → ET₀ → Kc climate adjust → water balance → yield.
    /// Stages 1-2 use GPU dispatch when available; remaining stages use CPU.
    /// For `GpuPipelined` and `GpuFused`, delegates to [`streaming_et0_kc`].
    #[must_use]
    pub fn run_season(&self, weather: &[WeatherDay], config: &CropConfig) -> SeasonResult {
        if matches!(self.backend, Backend::GpuPipelined | Backend::GpuFused) {
            return self.streaming_et0_kc(weather, config);
        }
        let et0_daily = self.compute_et0_batch(weather);
        let kc_daily = self.compute_kc_batch(weather, config);
        Self::run_stages_3_and_4(weather, config, et0_daily, &kc_daily)
    }

    /// Streaming path: ET₀ + Kc in one GPU batch, then stages 3-4 on CPU.
    ///
    /// Pre-computes Kc base values so both GPU dispatches can be issued without
    /// CPU readback between stages. For `GpuPipelined`/`GpuFused` backends.
    /// Falls back to CPU for ET₀/Kc when no GPU engines are present.
    #[must_use]
    pub fn streaming_et0_kc(&self, weather: &[WeatherDay], config: &CropConfig) -> SeasonResult {
        let et0_daily = self.compute_et0_batch(weather);
        let kc_daily = self.compute_kc_batch(weather, config);
        Self::run_stages_3_and_4(weather, config, et0_daily, &kc_daily)
    }

    /// Run Stages 3–4 with pre-computed ET₀ and Kc (enables unified GPU dispatch).
    ///
    /// When [`AtlasStream`] computes ET₀ for all stations in a single GPU
    /// batch (eliminating per-station round-trips), it passes the pre-sliced
    /// ET₀ array here for the remaining CPU-bound stages.
    #[must_use]
    pub fn run_season_with_et0(
        weather: &[WeatherDay],
        config: &CropConfig,
        et0_daily: &[f64],
    ) -> SeasonResult {
        let kc = config.crop_type.coefficients();
        let n = weather.len();
        let kc_daily: Vec<f64> = weather
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let kc_base = stage_kc(&kc, i, n);
                adjust_kc_for_climate(kc_base, w.wind_2m, w.rh_min, config.crop_height_m)
            })
            .collect();
        Self::run_stages_3_and_4(weather, config, et0_daily.to_vec(), &kc_daily)
    }

    fn run_stages_3_and_4(
        weather: &[WeatherDay],
        config: &CropConfig,
        et0_daily: Vec<f64>,
        kc_daily: &[f64],
    ) -> SeasonResult {
        if weather.is_empty() {
            return SeasonResult {
                n_days: 0,
                total_et0: 0.0,
                total_actual_et: 0.0,
                total_precipitation: 0.0,
                total_irrigation: 0.0,
                stress_days: 0,
                mass_balance_error: 0.0,
                yield_ratio: 1.0,
                et0_daily: Vec::new(),
                actual_et_daily: Vec::new(),
            };
        }

        let kc = config.crop_type.coefficients();
        let n = weather.len();

        // Stage 3: Water balance (op=1) with irrigation scheduling
        let root_mm = kc.root_depth_m * 1000.0;
        let mut state = WaterBalanceState::new(
            config.field_capacity,
            config.wilting_point,
            root_mm,
            config.irrigation_trigger,
        );

        let mut wb_inputs = Vec::with_capacity(n);
        let mut wb_outputs = Vec::with_capacity(n);
        let mut total_irrigation = 0.0_f64;

        for (i, w) in weather.iter().enumerate() {
            let irr = if state.depletion > state.raw {
                config.irrigation_depth_mm
            } else {
                0.0
            };
            total_irrigation += irr;

            let input = DailyInput {
                precipitation: w.precipitation,
                irrigation: irr,
                et0: et0_daily[i],
                kc: kc_daily[i],
            };
            let output = state.step(&input);
            wb_inputs.push(input);
            wb_outputs.push(output);
        }

        // Stage 4: Yield response
        let total_actual_et: f64 = wb_outputs.iter().map(|o| o.actual_et).sum();
        let total_etc: f64 = wb_outputs.iter().map(|o| o.etc).sum();
        let eta_etc = if total_etc > 0.0 {
            total_actual_et / total_etc
        } else {
            1.0
        };
        let yield_ratio = yield_response::clamp_yield_ratio(yield_response::yield_ratio_single(
            config.ky, eta_etc,
        ));

        let mass_balance_error =
            wb::mass_balance_check(&wb_inputs, &wb_outputs, 0.0, state.depletion);

        let actual_et_daily: Vec<f64> = wb_outputs.iter().map(|o| o.actual_et).collect();
        let stress_days = wb_outputs.iter().filter(|o| o.ks < 1.0).count();
        let total_precipitation: f64 = weather.iter().map(|w| w.precipitation).sum();

        SeasonResult {
            n_days: n,
            total_et0: et0_daily.iter().sum(),
            total_actual_et,
            total_precipitation,
            total_irrigation,
            stress_days,
            mass_balance_error,
            yield_ratio,
            et0_daily,
            actual_et_daily,
        }
    }

    /// Batch-compute Kc climate adjustment, using GPU (op=7) when available.
    #[must_use]
    fn compute_kc_batch(&self, weather: &[WeatherDay], config: &CropConfig) -> Vec<f64> {
        let kc_coeff = config.crop_type.coefficients();
        let n = weather.len();

        self.gpu_kc.as_ref().map_or_else(
            || {
                weather
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        let kc_base = stage_kc(&kc_coeff, i, n);
                        adjust_kc_for_climate(kc_base, w.wind_2m, w.rh_min, config.crop_height_m)
                    })
                    .collect()
            },
            |kc_engine| {
                let days: Vec<KcClimateDay> = weather
                    .iter()
                    .enumerate()
                    .map(|(i, w)| KcClimateDay {
                        kc_table: stage_kc(&kc_coeff, i, n),
                        u2: w.wind_2m,
                        rh_min: w.rh_min,
                        crop_height_m: config.crop_height_m,
                    })
                    .collect();
                kc_engine.compute_gpu(&days).map_or_else(
                    |_| {
                        weather
                            .iter()
                            .enumerate()
                            .map(|(i, w)| {
                                let kc_base = stage_kc(&kc_coeff, i, n);
                                adjust_kc_for_climate(
                                    kc_base,
                                    w.wind_2m,
                                    w.rh_min,
                                    config.crop_height_m,
                                )
                            })
                            .collect()
                    },
                    |r| r.kc_values,
                )
            },
        )
    }

    /// Batch-compute ET₀ for all weather days, using GPU when available.
    ///
    /// Public for unified dispatch from [`super::atlas_stream::AtlasStream`].
    #[must_use]
    pub fn compute_et0_batch(&self, weather: &[WeatherDay]) -> Vec<f64> {
        self.gpu_et0.as_ref().map_or_else(
            || weather.iter().map(compute_et0).collect(),
            |et0_engine| {
                let station_days: Vec<StationDay> = weather
                    .iter()
                    .map(|w| StationDay {
                        tmax: w.tmax,
                        tmin: w.tmin,
                        rh_max: w.rh_max,
                        rh_min: w.rh_min,
                        wind_2m: w.wind_2m,
                        rs: w.solar_rad,
                        elevation: w.elevation,
                        latitude: w.latitude_deg,
                        doy: w.day_of_year,
                    })
                    .collect();
                et0_engine.compute_gpu(&station_days).map_or_else(
                    |_| weather.iter().map(compute_et0).collect(),
                    |r| r.et0_values,
                )
            },
        )
    }
}

fn compute_et0(w: &WeatherDay) -> f64 {
    let ea = et::actual_vapour_pressure_rh(w.tmin, w.tmax, w.rh_min, w.rh_max);
    let input = DailyEt0Input {
        tmin: w.tmin,
        tmax: w.tmax,
        tmean: Some(f64::midpoint(w.tmin, w.tmax)),
        solar_radiation: w.solar_rad,
        wind_speed_2m: w.wind_2m,
        actual_vapour_pressure: ea,
        elevation_m: w.elevation,
        latitude_deg: w.latitude_deg,
        day_of_year: w.day_of_year,
    };
    et::daily_et0(&input).et0.max(0.0)
}

fn stage_kc(kc: &CropCoefficients, day_idx: usize, total_days: usize) -> f64 {
    let frac = day_idx as f64 / total_days as f64;
    if frac < 0.2 {
        kc.kc_ini
    } else if frac < 0.7 {
        kc.kc_mid
    } else {
        kc.kc_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summer_day(doy: u32) -> WeatherDay {
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

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
            .ok()
            .map(std::sync::Arc::new)
    }

    #[test]
    fn gpu_pipeline_matches_cpu() {
        let Some(device) = try_device() else {
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
        let Some(device) = try_device() else {
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
        let cpu_pipeline = SeasonalPipeline::cpu();
        let weather = growing_season();
        let config = CropConfig::standard(CropType::Corn);

        let cpu_result = cpu_pipeline.run_season(&weather, &config);

        let Some(device) = try_device() else {
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
}
