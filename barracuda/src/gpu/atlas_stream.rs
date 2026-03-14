// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming orchestrator for multi-year regional ET₀.
//!
//! Wraps [`SeasonalPipeline`] for processing multiple stations and years.
//! When `BarraCuda`'s `UnidirectionalPipeline` is wired, this becomes fire-and-forget
//! streaming without blocking.
//!
//! # Streaming Architecture
//!
//! ```text
//! Host → [GpuRingBuffer] → GPU compute → [GpuRingBuffer] → Host (future)
//! Current: CPU sequential, same API shape
//! ```

use std::ops::Range;
use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::gpu::seasonal_pipeline::{CropConfig, SeasonResult, SeasonalPipeline, WeatherDay};

/// Station batch: station identifier and weather data for one season.
#[derive(Debug, Clone)]
pub struct StationBatch {
    /// Station identifier.
    pub station_id: String,
    /// Year of the season.
    pub year: u32,
    /// Daily weather observations.
    pub weather: Vec<WeatherDay>,
}

/// Configuration for atlas stream processing.
#[derive(Debug, Clone)]
pub struct AtlasStreamConfig {
    /// Crop configurations to simulate.
    pub crop_configs: Vec<CropConfig>,
    /// Year range (metadata for multi-year runs).
    pub year_range: Range<u32>,
}

/// Result of a station-season simulation.
#[derive(Debug, Clone)]
pub struct StationSeasonResult {
    /// Station identifier.
    pub station_id: String,
    /// Year of the season.
    pub year: u32,
    /// Crop name (from `CropConfig`).
    pub crop_name: String,
    /// Seasonal simulation result.
    pub result: SeasonResult,
}

/// Atlas stream orchestrator for multi-station, multi-crop seasonal simulation.
///
/// Processes multiple stations through [`SeasonalPipeline`] for each crop config.
/// Supports both CPU and GPU-accelerated backends via [`AtlasStream::with_gpu`].
///
/// # Streaming Pattern
///
/// [`AtlasStream::process_streaming`] emits results via callback as they are produced,
/// preparing for `BarraCuda`'s `UnidirectionalPipeline` (fire-and-forget
/// GPU streaming) without buffering the full result set.
#[derive(Debug)]
pub struct AtlasStream {
    pipeline: SeasonalPipeline,
}

impl AtlasStream {
    /// Create a new atlas stream (CPU backend).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pipeline: SeasonalPipeline::cpu(),
        }
    }

    /// Create an atlas stream with GPU-accelerated ET₀ dispatch.
    ///
    /// Stage 1 (ET₀) of each seasonal simulation dispatches to GPU;
    /// remaining stages use CPU until `BarraCuda` absorbs their ops.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device cannot initialise `BatchedEt0`.
    pub fn with_gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let pipeline = SeasonalPipeline::gpu(device)?;
        Ok(Self { pipeline })
    }

    /// Process a batch of stations through the seasonal pipeline.
    ///
    /// Runs each station's weather through each crop config and returns
    /// all station-season results.
    #[must_use]
    pub fn process_batch(
        &self,
        batches: &[StationBatch],
        config: &AtlasStreamConfig,
    ) -> Vec<StationSeasonResult> {
        let mut results = Vec::with_capacity(batches.len() * config.crop_configs.len());
        self.process_streaming(batches, config, |r| results.push(r));
        results
    }

    /// Process stations in a streaming pattern, emitting each result via callback.
    ///
    /// Unlike [`Self::process_batch`], this avoids buffering the full result set.
    /// When `BarraCuda`'s `UnidirectionalPipeline` is available, this method
    /// becomes fire-and-forget GPU streaming.
    pub fn process_streaming<F>(
        &self,
        batches: &[StationBatch],
        config: &AtlasStreamConfig,
        mut on_result: F,
    ) where
        F: FnMut(StationSeasonResult),
    {
        for batch in batches {
            for crop_config in &config.crop_configs {
                let result = self.pipeline.run_season(&batch.weather, crop_config);
                let crop_name = crop_config.crop_type.coefficients().name.to_string();
                on_result(StationSeasonResult {
                    station_id: batch.station_id.clone(), // justified: ownership needed for storage in StationSeasonResult
                    year: batch.year,
                    crop_name,
                    result,
                });
            }
        }
    }

    /// Unified GPU dispatch: compute ET₀ for ALL stations in a single GPU
    /// batch, then distribute per-station results through CPU stages.
    ///
    /// This eliminates N-1 CPU↔GPU round-trips when processing N stations.
    /// Equivalent to `BarraCuda`'s `UnidirectionalPipeline` pattern: all data
    /// flows GPU→host in one direction for Stage 1, then stages 2-4 proceed
    /// on CPU with the pre-computed ET₀.
    ///
    /// Falls back to per-station dispatch when no GPU engine is available.
    #[must_use]
    pub fn process_batch_unified(
        &self,
        batches: &[StationBatch],
        config: &AtlasStreamConfig,
    ) -> Vec<StationSeasonResult> {
        if batches.is_empty() || config.crop_configs.is_empty() {
            return Vec::new();
        }

        let all_et0 = self.compute_unified_et0(batches);

        let mut results = Vec::with_capacity(batches.len() * config.crop_configs.len());
        let station_lengths: Vec<usize> = batches.iter().map(|b| b.weather.len()).collect();
        let mut offset = 0;

        for (batch_idx, batch) in batches.iter().enumerate() {
            let n = station_lengths[batch_idx];
            let et0_slice = &all_et0[offset..offset + n];
            offset += n;

            for crop_config in &config.crop_configs {
                let result =
                    SeasonalPipeline::run_season_with_et0(&batch.weather, crop_config, et0_slice);
                let crop_name = crop_config.crop_type.coefficients().name.to_string();
                results.push(StationSeasonResult {
                    station_id: batch.station_id.clone(), // justified: ownership needed for storage in StationSeasonResult
                    year: batch.year,
                    crop_name,
                    result,
                });
            }
        }

        results
    }

    /// Number of simulations that will be produced for the given inputs.
    #[must_use]
    pub const fn simulation_count(batches: &[StationBatch], config: &AtlasStreamConfig) -> usize {
        batches.len() * config.crop_configs.len()
    }

    fn compute_unified_et0(&self, batches: &[StationBatch]) -> Vec<f64> {
        self.pipeline.compute_et0_batch(
            &batches
                .iter()
                .flat_map(|b| b.weather.iter().copied())
                .collect::<Vec<_>>(),
        )
    }
}

impl Default for AtlasStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Lightweight fitness-based drift detector for atlas stream processing.
///
/// Tracks mean and best fitness across generations. Drift is flagged when
/// mean fitness drops below 50% of the historical best for 3 or more
/// consecutive generations — indicating drought onset or regime shift.
///
/// Replaces the upstream `bingocube_nautilus::DriftMonitor` with a focused
/// implementation for agricultural yield-ratio monitoring.
#[derive(Debug)]
pub struct FitnessDriftMonitor {
    best_historical_mean: f64,
    consecutive_drops: usize,
    latest_ne_s: f64,
    drifting: bool,
}

impl Default for FitnessDriftMonitor {
    fn default() -> Self {
        Self {
            best_historical_mean: 0.0,
            consecutive_drops: 0,
            latest_ne_s: 1.0,
            drifting: false,
        }
    }
}

impl FitnessDriftMonitor {
    const DRIFT_THRESHOLD: f64 = 0.5;
    const CONSECUTIVE_REQUIRED: usize = 3;

    /// Record a generation's fitness statistics.
    pub fn record(&mut self, _generation: usize, pop_size: usize, mean_fit: f64, _best_fit: f64) {
        if mean_fit > self.best_historical_mean {
            self.best_historical_mean = mean_fit;
        }

        #[allow(clippy::cast_precision_loss)]
        let ne_s = if self.best_historical_mean > 0.0 {
            (mean_fit / self.best_historical_mean) * pop_size as f64
        } else {
            pop_size as f64
        };
        self.latest_ne_s = ne_s;

        if self.best_historical_mean > 0.0
            && mean_fit < self.best_historical_mean * Self::DRIFT_THRESHOLD
        {
            self.consecutive_drops += 1;
        } else {
            self.consecutive_drops = 0;
        }

        self.drifting = self.consecutive_drops >= Self::CONSECUTIVE_REQUIRED;
    }

    /// Whether the monitor is in a drifting state.
    #[must_use]
    pub const fn is_drifting(&self) -> bool {
        self.drifting
    }

    /// Latest `N_e * s` value.
    #[must_use]
    pub const fn latest_ne_s(&self) -> f64 {
        self.latest_ne_s
    }
}

/// Regime change snapshot from drift detection during multi-year atlas processing.
///
/// Emitted by [`MonitoredAtlasStream`] when the monitor detects a shift in
/// agricultural conditions (drought onset, irrigation regime change, etc.).
#[derive(Debug, Clone)]
pub struct RegimeChange {
    /// Station where the regime change was detected.
    pub station_id: String,
    /// Year where the change was flagged.
    pub year: u32,
    /// Crop name.
    pub crop_name: String,
    /// The `N_e * s` value at the time of detection.
    pub ne_s: f64,
}

/// Atlas stream with integrated [`FitnessDriftMonitor`] for regime change detection.
///
/// Wraps [`AtlasStream`] and feeds year-over-year yield/ET₀ into the
/// drift monitor, flagging stations and years where `N_e * s` drops below
/// the drift threshold for ≥3 consecutive seasons.
///
/// The "population" analogy: each station's yearly yield ratio is an
/// "organism" fitness; drought or regime shifts cause fitness collapse,
/// which the drift detector flags.
#[derive(Debug)]
pub struct MonitoredAtlasStream {
    inner: AtlasStream,
    monitor: FitnessDriftMonitor,
    regime_changes: Vec<RegimeChange>,
    generation: usize,
}

impl MonitoredAtlasStream {
    /// Create a monitored atlas stream (CPU backend).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: AtlasStream::new(),
            monitor: FitnessDriftMonitor::default(),
            regime_changes: Vec::new(),
            generation: 0,
        }
    }

    /// Create a monitored atlas stream with GPU acceleration.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device cannot initialise.
    pub fn with_gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self {
            inner: AtlasStream::with_gpu(device)?,
            monitor: FitnessDriftMonitor::default(),
            regime_changes: Vec::new(),
            generation: 0,
        })
    }

    /// Process a batch and track regime changes year-over-year.
    ///
    /// Each station-season result is fed into the drift monitor.
    /// Yield ratio is used as the "fitness" signal: `mean_fitness` is the
    /// average yield ratio across all crops for a station-year, and
    /// `best_fitness` is the maximum.
    #[must_use]
    pub fn process_monitored(
        &mut self,
        batches: &[StationBatch],
        config: &AtlasStreamConfig,
    ) -> Vec<StationSeasonResult> {
        let results = self.inner.process_batch(batches, config);

        let mut by_station_year: std::collections::HashMap<
            (String, u32),
            Vec<&StationSeasonResult>,
        > = std::collections::HashMap::new();
        for r in &results {
            by_station_year
                .entry((r.station_id.clone(), r.year)) // justified: HashMap key requires owned String
                .or_default()
                .push(r);
        }

        let pop_size = config.crop_configs.len().max(1);
        let mut keys: Vec<_> = by_station_year.keys().cloned().collect();
        keys.sort();

        for key in &keys {
            if let Some(group) = by_station_year.get(key) {
                let yields: Vec<f64> = group.iter().map(|r| r.result.yield_ratio).collect();
                let mean_yield = yields.iter().sum::<f64>() / yields.len() as f64;
                let best_yield = yields.iter().fold(0.0_f64, |a, &b| a.max(b));

                self.monitor
                    .record(self.generation, pop_size, mean_yield, best_yield);

                if self.monitor.is_drifting() {
                    let crop_name = group
                        .iter()
                        .min_by(|a, b| {
                            a.result
                                .yield_ratio
                                .partial_cmp(&b.result.yield_ratio)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or_else(String::new, |r| r.crop_name.clone());

                    self.regime_changes.push(RegimeChange {
                        station_id: key.0.clone(), // justified: ownership needed for storage in RegimeChange
                        year: key.1,
                        crop_name,
                        ne_s: self.monitor.latest_ne_s(),
                    });
                }
                self.generation += 1;
            }
        }

        results
    }

    /// Whether any regime changes have been detected.
    #[must_use]
    pub const fn has_regime_changes(&self) -> bool {
        !self.regime_changes.is_empty()
    }

    /// All detected regime changes.
    #[must_use]
    pub fn regime_changes(&self) -> &[RegimeChange] {
        &self.regime_changes
    }

    /// Reference to the underlying drift monitor.
    #[must_use]
    pub const fn drift_monitor(&self) -> &FitnessDriftMonitor {
        &self.monitor
    }

    /// Whether the monitor is currently in a drifting state.
    #[must_use]
    pub const fn is_drifting(&self) -> bool {
        self.monitor.is_drifting()
    }

    /// Reset the monitor for a new multi-year run.
    pub fn reset(&mut self) {
        self.monitor = FitnessDriftMonitor::default();
        self.regime_changes.clear();
        self.generation = 0;
    }
}

impl Default for MonitoredAtlasStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::eco::crop::CropType;

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
    fn single_station_single_crop() {
        let stream = AtlasStream::new();
        let batches = vec![StationBatch {
            station_id: "STN-001".to_string(),
            year: 2024,
            weather: growing_season(),
        }];
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2024..2025,
        };

        let results = stream.process_batch(&batches, &config);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].station_id, "STN-001");
        assert_eq!(results[0].year, 2024);
        assert_eq!(results[0].crop_name, "Corn (grain)");
        assert_eq!(results[0].result.n_days, 153);
        assert!(results[0].result.yield_ratio > 0.5 && results[0].result.yield_ratio <= 1.0);
    }

    #[test]
    fn multi_crop_batch() {
        let stream = AtlasStream::new();
        let batches = vec![StationBatch {
            station_id: "STN-002".to_string(),
            year: 2024,
            weather: growing_season(),
        }];
        let config = AtlasStreamConfig {
            crop_configs: vec![
                CropConfig::standard(CropType::Corn),
                CropConfig::standard(CropType::Soybean),
            ],
            year_range: 2024..2025,
        };

        let results = stream.process_batch(&batches, &config);

        assert_eq!(results.len(), 2);
        let corn = results
            .iter()
            .find(|r| r.crop_name == "Corn (grain)")
            .unwrap();
        let soy = results.iter().find(|r| r.crop_name == "Soybean").unwrap();
        assert!(corn.result.total_actual_et > soy.result.total_actual_et);
    }

    #[test]
    fn empty_batch_returns_empty() {
        let stream = AtlasStream::new();
        let batches: Vec<StationBatch> = vec![];
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2024..2025,
        };

        let results = stream.process_batch(&batches, &config);

        assert!(results.is_empty());
    }

    #[test]
    fn station_id_preserved() {
        let stream = AtlasStream::new();
        let batches = vec![
            StationBatch {
                station_id: "ALPHA".to_string(),
                year: 2023,
                weather: growing_season(),
            },
            StationBatch {
                station_id: "BETA".to_string(),
                year: 2024,
                weather: growing_season(),
            },
        ];
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2023..2025,
        };

        let results = stream.process_batch(&batches, &config);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].station_id, "ALPHA");
        assert_eq!(results[0].year, 2023);
        assert_eq!(results[1].station_id, "BETA");
        assert_eq!(results[1].year, 2024);
    }

    #[test]
    fn streaming_matches_batch() {
        let stream = AtlasStream::new();
        let batches = vec![
            StationBatch {
                station_id: "A".to_string(),
                year: 2024,
                weather: growing_season(),
            },
            StationBatch {
                station_id: "B".to_string(),
                year: 2024,
                weather: growing_season(),
            },
        ];
        let config = AtlasStreamConfig {
            crop_configs: vec![
                CropConfig::standard(CropType::Corn),
                CropConfig::standard(CropType::Soybean),
            ],
            year_range: 2024..2025,
        };

        let batch_results = stream.process_batch(&batches, &config);
        let mut stream_results = Vec::new();
        stream.process_streaming(&batches, &config, |r| stream_results.push(r));

        assert_eq!(batch_results.len(), stream_results.len());
        for (b, s) in batch_results.iter().zip(&stream_results) {
            assert_eq!(b.station_id, s.station_id);
            assert_eq!(b.crop_name, s.crop_name);
            assert!((b.result.yield_ratio - s.result.yield_ratio).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn simulation_count_correct() {
        let batches = vec![
            StationBatch {
                station_id: "X".to_string(),
                year: 2024,
                weather: growing_season(),
            },
            StationBatch {
                station_id: "Y".to_string(),
                year: 2024,
                weather: growing_season(),
            },
        ];
        let config = AtlasStreamConfig {
            crop_configs: vec![
                CropConfig::standard(CropType::Corn),
                CropConfig::standard(CropType::Soybean),
                CropConfig::standard(CropType::Potato),
            ],
            year_range: 2024..2025,
        };
        assert_eq!(AtlasStream::simulation_count(&batches, &config), 6);
    }

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .ok()
        .map(std::sync::Arc::new)
    }

    #[test]
    fn unified_matches_per_station() {
        let stream = AtlasStream::new();
        let batches = vec![
            StationBatch {
                station_id: "U1".to_string(),
                year: 2024,
                weather: growing_season(),
            },
            StationBatch {
                station_id: "U2".to_string(),
                year: 2024,
                weather: growing_season(),
            },
        ];
        let config = AtlasStreamConfig {
            crop_configs: vec![
                CropConfig::standard(CropType::Corn),
                CropConfig::standard(CropType::Soybean),
            ],
            year_range: 2024..2025,
        };

        let batch_results = stream.process_batch(&batches, &config);
        let unified_results = stream.process_batch_unified(&batches, &config);

        assert_eq!(batch_results.len(), unified_results.len());
        for (b, u) in batch_results.iter().zip(&unified_results) {
            assert_eq!(b.station_id, u.station_id);
            assert_eq!(b.crop_name, u.crop_name);
            assert!(
                (b.result.yield_ratio - u.result.yield_ratio).abs() < f64::EPSILON,
                "Batch YR {:.4} != unified {:.4}",
                b.result.yield_ratio,
                u.result.yield_ratio
            );
            assert!(
                (b.result.total_et0 - u.result.total_et0).abs() < f64::EPSILON,
                "Batch ET₀ {:.2} != unified {:.2}",
                b.result.total_et0,
                u.result.total_et0
            );
        }
    }

    #[test]
    fn unified_empty_batches() {
        let stream = AtlasStream::new();
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2024..2025,
        };
        let results = stream.process_batch_unified(&[], &config);
        assert!(results.is_empty());
    }

    #[test]
    fn gpu_atlas_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for AtlasStream");
            return;
        };
        let gpu_stream = AtlasStream::with_gpu(device).unwrap();
        let cpu_stream = AtlasStream::new();
        let batches = vec![StationBatch {
            station_id: "GPU-TEST".to_string(),
            year: 2024,
            weather: growing_season(),
        }];
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2024..2025,
        };

        let gpu_results = gpu_stream.process_batch(&batches, &config);
        let cpu_results = cpu_stream.process_batch(&batches, &config);

        assert_eq!(gpu_results.len(), cpu_results.len());
        let et0_diff = (gpu_results[0].result.total_et0 - cpu_results[0].result.total_et0).abs();
        let et0_pct = et0_diff / cpu_results[0].result.total_et0 * 100.0;
        assert!(
            et0_pct < 1.0,
            "GPU↔CPU ET₀ {:.1} vs {:.1} ({:.2}% > 1% threshold)",
            gpu_results[0].result.total_et0,
            cpu_results[0].result.total_et0,
            et0_pct
        );
    }

    #[test]
    fn monitored_stream_processes_ok() {
        let mut stream = MonitoredAtlasStream::new();
        let batches = vec![StationBatch {
            station_id: "MON-001".to_string(),
            year: 2024,
            weather: growing_season(),
        }];
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2024..2025,
        };

        let results = stream.process_monitored(&batches, &config);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].station_id, "MON-001");
    }

    #[test]
    fn monitored_stream_tracks_multi_year() {
        let mut stream = MonitoredAtlasStream::new();
        let config = AtlasStreamConfig {
            crop_configs: vec![
                CropConfig::standard(CropType::Corn),
                CropConfig::standard(CropType::Soybean),
            ],
            year_range: 2020..2025,
        };

        for year in 2020..2025 {
            let batches = vec![StationBatch {
                station_id: "DECADE-A".to_string(),
                year,
                weather: growing_season(),
            }];
            let _ = stream.process_monitored(&batches, &config);
        }

        let dm = stream.drift_monitor();
        assert!(!dm.is_drifting() || stream.has_regime_changes());
    }

    #[test]
    fn monitored_stream_reset() {
        let mut stream = MonitoredAtlasStream::new();
        let batches = vec![StationBatch {
            station_id: "RST".to_string(),
            year: 2024,
            weather: growing_season(),
        }];
        let config = AtlasStreamConfig {
            crop_configs: vec![CropConfig::standard(CropType::Corn)],
            year_range: 2024..2025,
        };

        let _ = stream.process_monitored(&batches, &config);
        stream.reset();

        assert!(stream.regime_changes().is_empty());
        assert!(!stream.is_drifting());
    }

    #[test]
    fn monitored_default_impl() {
        let stream = MonitoredAtlasStream::default();
        assert!(!stream.is_drifting());
        assert!(!stream.has_regime_changes());
    }
}
