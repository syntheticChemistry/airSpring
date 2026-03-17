// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fitness-based drift detection for multi-year atlas stream processing.
//!
//! Tracks year-over-year yield / ET₀ fitness and flags regime shifts
//! (drought onset, irrigation changes) when `N_e * s` drops below
//! threshold for consecutive generations.

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use super::{AtlasStream, AtlasStreamConfig, StationBatch, StationSeasonResult};

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

        let pop_f64 = crate::cast::usize_f64(pop_size);
        let ne_s = if self.best_historical_mean > 0.0 {
            (mean_fit / self.best_historical_mean) * pop_f64
        } else {
            pop_f64
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
                .entry((r.station_id.clone(), r.year))
                .or_default()
                .push(r);
        }

        let pop_size = config.crop_configs.len().max(1);
        let mut keys: Vec<_> = by_station_year.keys().cloned().collect();
        keys.sort();

        for key in &keys {
            if let Some(group) = by_station_year.get(key) {
                let yields: Vec<f64> = group.iter().map(|r| r.result.yield_ratio).collect();
                let mean_yield = yields.iter().sum::<f64>() / crate::cast::usize_f64(yields.len());
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
                        station_id: key.0.clone(),
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
