// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming orchestrator for multi-year regional ET₀.
//!
//! Wraps [`SeasonalPipeline`] for processing multiple stations and years.
//! When `ToadStool`'s `UnidirectionalPipeline` is wired, this becomes fire-and-forget
//! streaming without blocking.
//!
//! # Streaming Architecture
//!
//! ```text
//! Host → [GpuRingBuffer] → GPU compute → [GpuRingBuffer] → Host (future)
//! Current: CPU sequential, same API shape
//! ```

use crate::gpu::seasonal_pipeline::{
    CropConfig, SeasonalPipeline, SeasonResult, WeatherDay,
};
use std::ops::Range;

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
/// CPU-only; same API shape as future GPU streaming.
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
        let mut results = Vec::with_capacity(
            batches.len() * config.crop_configs.len(),
        );
        for batch in batches {
            for crop_config in &config.crop_configs {
                let result = self
                    .pipeline
                    .run_season(&batch.weather, crop_config);
                let crop_name = crop_config.crop_type.coefficients().name.to_string();
                results.push(StationSeasonResult {
                    station_id: batch.station_id.clone(),
                    year: batch.year,
                    crop_name,
                    result,
                });
            }
        }
        results
    }
}

impl Default for AtlasStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
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
        let corn = results.iter().find(|r| r.crop_name == "Corn (grain)").unwrap();
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
}
