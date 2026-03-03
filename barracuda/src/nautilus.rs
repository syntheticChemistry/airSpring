// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nautilus Shell integration for agricultural regime prediction.
//!
//! Wraps `bingocube_nautilus::NautilusShell` with agricultural domain heads:
//! - **ET₀ head**: Predict reference evapotranspiration from weather features
//! - **Soil moisture head**: Predict volumetric water content trajectory
//! - **Crop stress head**: Predict yield response (Ky-based water stress)
//!
//! The shell evolves board populations to predict agricultural observables.
//! [`DriftMonitor`] detects regime changes (drought onset, season transitions).
//! Concept edge detection identifies days where predictions fail — the most
//! scientifically interesting regime boundaries.
//!
//! # Cross-Spring Provenance
//!
//! - `bingocube-nautilus` from `primalTools/bingoCube/nautilus/`
//! - Brain pattern from hotSpring v0.6.15 (`NautilusBrain` for QCD)
//! - [`DriftMonitor`] implements `N_e*s` boundary from constrained evolution thesis
//! - Board populations map to AKD1000 int4 for edge deployment

use bingocube_nautilus::{
    DriftMonitor, EvolutionConfig, InstanceId, NautilusShell, ReservoirInput, ShellConfig,
};
use serde::{Deserialize, Serialize};

/// Number of prediction targets for the agricultural brain.
///
/// 0: ET₀ (mm/day, normalized)
/// 1: soil moisture deficit (0-1)
/// 2: crop stress factor (0-1, where 1 = no stress)
pub const N_TARGETS: usize = 3;

/// Configuration for the agricultural Nautilus brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirSpringBrainConfig {
    /// Shell configuration (population size, evolution params).
    pub shell: ShellConfig,

    /// Generations to evolve per training cycle.
    pub generations_per_cycle: u64,

    /// Minimum observations before training is allowed.
    pub min_training_points: usize,

    /// LOO error threshold for concept edge detection.
    /// Points above this threshold are flagged as regime boundaries
    /// (drought onset, frost events, seasonal transitions).
    pub concept_edge_threshold: f64,
}

impl Default for AirSpringBrainConfig {
    fn default() -> Self {
        Self {
            shell: ShellConfig {
                population_size: 24,
                n_targets: N_TARGETS,
                evolution: EvolutionConfig::default(),
                ridge_lambda: 1e-4,
                ..Default::default()
            },
            generations_per_cycle: 20,
            min_training_points: 5,
            concept_edge_threshold: 0.15,
        }
    }
}

/// A single weather/soil observation for training the agricultural brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherObservation {
    /// Day of year (1-366).
    pub doy: u16,
    /// Maximum temperature (°C).
    pub tmax: f64,
    /// Minimum temperature (°C).
    pub tmin: f64,
    /// Mean relative humidity (%).
    pub rh_mean: f64,
    /// Wind speed at 2m (m/s).
    pub wind_2m: f64,
    /// Solar radiation (MJ/m²/day).
    pub solar_rad: f64,
    /// Precipitation (mm).
    pub precip: f64,
    /// Measured ET₀ (mm/day) — target for the ET₀ head.
    pub et0_observed: f64,
    /// Measured soil moisture deficit (0-1) — target for soil head.
    pub soil_deficit: f64,
    /// Measured crop stress factor (0-1) — target for crop head.
    pub crop_stress: f64,
}

impl WeatherObservation {
    fn to_reservoir_input(&self) -> ReservoirInput {
        ReservoirInput::Continuous(vec![
            f64::from(self.doy) / 366.0,
            (self.tmax + 10.0) / 60.0,
            (self.tmin + 10.0) / 60.0,
            self.rh_mean / 100.0,
            self.wind_2m / 10.0,
            self.solar_rad / 40.0,
            self.precip / 50.0,
        ])
    }

    fn targets(&self, et0_max: f64) -> Vec<f64> {
        let et0_norm = if et0_max > 0.0 {
            self.et0_observed / et0_max
        } else {
            0.0
        };
        vec![et0_norm, self.soil_deficit, self.crop_stress]
    }
}

/// Agricultural Nautilus brain — evolutionary reservoir for weather/soil prediction.
///
/// Follows the same pattern as hotSpring's `NautilusBrain` for QCD, adapted
/// to agricultural domain heads (ET₀, soil moisture, crop stress).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirSpringBrain {
    config: AirSpringBrainConfig,
    shell: NautilusShell,
    observations: Vec<WeatherObservation>,
    drift: DriftMonitor,
    concept_edge_doys: Vec<u16>,
    trained: bool,
}

impl AirSpringBrain {
    /// Create a new agricultural brain for a named instance (e.g. "eastgate").
    #[must_use]
    pub fn new(config: AirSpringBrainConfig, instance: &str) -> Self {
        let id = InstanceId::new(instance);
        let shell = NautilusShell::from_seed(config.shell.clone(), id, 42);

        Self {
            config,
            shell,
            observations: Vec::new(),
            drift: DriftMonitor::default(),
            concept_edge_doys: Vec::new(),
            trained: false,
        }
    }

    /// Create from an inherited shell (cross-station or cross-run bootstrap).
    #[must_use]
    pub fn from_shell(config: AirSpringBrainConfig, shell: NautilusShell, instance: &str) -> Self {
        let id = InstanceId::new(instance);
        let shell = NautilusShell::continue_from(shell, id);

        Self {
            config,
            shell,
            observations: Vec::new(),
            drift: DriftMonitor::default(),
            concept_edge_doys: Vec::new(),
            trained: true,
        }
    }

    /// Record a weather/soil observation.
    pub fn observe(&mut self, obs: WeatherObservation) {
        self.observations.push(obs);
    }

    /// Number of accumulated observations.
    #[must_use]
    pub const fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Whether the brain has been trained at least once.
    #[must_use]
    pub const fn is_trained(&self) -> bool {
        self.trained
    }

    /// Whether the drift monitor detects regime change (drought, season shift).
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.drift.is_drifting()
    }

    /// Days of year flagged as concept edges (regime boundaries).
    #[must_use]
    pub fn concept_edge_doys(&self) -> &[u16] {
        &self.concept_edge_doys
    }

    /// Reference to the underlying shell (for serialization/transfer).
    #[must_use]
    pub const fn shell(&self) -> &NautilusShell {
        &self.shell
    }

    /// Reference to the drift monitor.
    #[must_use]
    pub const fn drift_monitor(&self) -> &DriftMonitor {
        &self.drift
    }

    /// Maximum observed ET₀ (for denormalization).
    fn et0_max(&self) -> f64 {
        self.observations
            .iter()
            .map(|o| o.et0_observed)
            .fold(0.0_f64, f64::max)
    }

    /// Train the shell on all accumulated observations.
    /// Returns MSE, or `None` if insufficient data.
    pub fn train(&mut self) -> Option<f64> {
        if self.observations.len() < self.config.min_training_points {
            return None;
        }

        let et0_max = self.et0_max();
        let inputs: Vec<ReservoirInput> = self
            .observations
            .iter()
            .map(WeatherObservation::to_reservoir_input)
            .collect();
        let targets: Vec<Vec<f64>> = self
            .observations
            .iter()
            .map(|o| o.targets(et0_max))
            .collect();

        let mut last_mse = 0.0;
        for gen in 0..self.config.generations_per_cycle {
            let seed = self.shell.generation() as u64 * 1000 + gen;
            last_mse = self.shell.evolve_generation_seeded(&inputs, &targets, seed);

            let traj = self.shell.fitness_trajectory();
            if let Some(&(gen_num, mean_fit, best_fit)) = traj.last() {
                self.drift.record(
                    gen_num,
                    self.config.shell.population_size,
                    mean_fit,
                    best_fit,
                );
            }
        }

        self.detect_concept_edges(&inputs);
        self.trained = true;
        Some(last_mse)
    }

    /// Predict ET₀, soil deficit, and crop stress for a weather observation.
    /// Returns `None` if untrained.
    #[must_use]
    pub fn predict(&self, obs: &WeatherObservation) -> Option<AgPrediction> {
        if !self.trained {
            return None;
        }

        let input = obs.to_reservoir_input();
        let pred = self.shell.predict(&input);

        if pred.len() >= N_TARGETS {
            let et0_max = self.et0_max();
            Some(AgPrediction {
                et0: pred[0] * et0_max,
                soil_deficit: pred[1].clamp(0.0, 1.0),
                crop_stress: pred[2].clamp(0.0, 1.0),
            })
        } else {
            None
        }
    }

    /// Export shell state as JSON for transfer to another station/instance.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the shell state cannot be serialized.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.shell)
    }

    /// Import a shell from JSON (cross-station transfer).
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the JSON cannot be deserialized into a valid shell.
    pub fn import_json(
        config: AirSpringBrainConfig,
        json: &str,
        instance: &str,
    ) -> Result<Self, serde_json::Error> {
        let shell: NautilusShell = serde_json::from_str(json)?;
        Ok(Self::from_shell(config, shell, instance))
    }

    fn detect_concept_edges(&mut self, inputs: &[ReservoirInput]) {
        self.concept_edge_doys.clear();

        for (i, input) in inputs.iter().enumerate() {
            let pred = self.shell.predict(input);
            if pred.is_empty() {
                continue;
            }
            let et0_max = self.et0_max();
            if let Some(obs) = self.observations.get(i) {
                let targets = obs.targets(et0_max);
                let mse: f64 = pred
                    .iter()
                    .zip(targets.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / pred.len() as f64;
                if mse > self.config.concept_edge_threshold {
                    self.concept_edge_doys.push(obs.doy);
                }
            }
        }

        self.concept_edge_doys.sort_unstable();
        self.concept_edge_doys.dedup();
    }
}

/// Prediction output from the agricultural brain.
#[derive(Debug, Clone, Copy)]
pub struct AgPrediction {
    /// Predicted reference evapotranspiration (mm/day).
    pub et0: f64,
    /// Predicted soil moisture deficit (0 = field capacity, 1 = wilting point).
    pub soil_deficit: f64,
    /// Predicted crop stress factor (0 = total failure, 1 = no stress).
    pub crop_stress: f64,
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::suboptimal_flops,
    reason = "test code; flops match reference reservoir formulas"
)]
mod tests {
    use super::*;

    fn make_obs(doy: u16, tmax: f64, tmin: f64, et0: f64) -> WeatherObservation {
        WeatherObservation {
            doy,
            tmax,
            tmin,
            rh_mean: 60.0,
            wind_2m: 2.0,
            solar_rad: 20.0,
            precip: 0.0,
            et0_observed: et0,
            soil_deficit: 0.3,
            crop_stress: 0.8,
        }
    }

    #[test]
    fn brain_lifecycle() {
        let config = AirSpringBrainConfig::default();
        let mut brain = AirSpringBrain::new(config, "eastgate-test");

        assert!(!brain.is_trained());
        assert_eq!(brain.observation_count(), 0);

        for doy in 1..=10 {
            let et0 = 2.0 + f64::from(doy) * 0.3;
            brain.observe(make_obs(doy * 15, 25.0 + f64::from(doy), 10.0, et0));
        }
        assert_eq!(brain.observation_count(), 10);

        let mse = brain.train();
        assert!(mse.is_some(), "training should succeed with 10 points");
        assert!(brain.is_trained());

        let pred = brain.predict(&make_obs(75, 28.0, 12.0, 0.0));
        assert!(pred.is_some());
        let p = pred.unwrap();
        assert!(p.et0.is_finite());
        assert!((0.0..=1.0).contains(&p.soil_deficit));
        assert!((0.0..=1.0).contains(&p.crop_stress));
    }

    #[test]
    fn brain_untrained_returns_none() {
        let brain = AirSpringBrain::new(AirSpringBrainConfig::default(), "test");
        assert!(brain.predict(&make_obs(100, 30.0, 15.0, 5.0)).is_none());
    }

    #[test]
    fn brain_insufficient_data_returns_none() {
        let config = AirSpringBrainConfig {
            min_training_points: 10,
            ..Default::default()
        };
        let mut brain = AirSpringBrain::new(config, "test");
        for doy in 1..=3 {
            brain.observe(make_obs(doy * 30, 25.0, 10.0, 3.0));
        }
        assert!(brain.train().is_none());
    }

    #[test]
    fn shell_serialization_roundtrip() {
        let config = AirSpringBrainConfig::default();
        let mut brain = AirSpringBrain::new(config.clone(), "station-a");

        for doy in 1..=8 {
            brain.observe(make_obs(doy * 20, 25.0, 10.0, 2.0 + f64::from(doy) * 0.2));
        }
        brain.train();

        let json = brain.export_json().unwrap();
        assert!(!json.is_empty());

        let imported = AirSpringBrain::import_json(config, &json, "station-b");
        assert!(imported.is_ok());
        let station_b = imported.unwrap();
        assert!(station_b.is_trained());
    }

    #[test]
    fn default_config_is_reasonable() {
        let config = AirSpringBrainConfig::default();
        assert_eq!(config.shell.population_size, 24);
        assert_eq!(config.shell.n_targets, N_TARGETS);
        assert_eq!(config.generations_per_cycle, 20);
        assert_eq!(config.min_training_points, 5);
    }

    #[test]
    fn observation_normalization_bounded() {
        let obs = make_obs(182, 35.0, 20.0, 7.5);
        let input = obs.to_reservoir_input();
        if let ReservoirInput::Continuous(features) = &input {
            assert_eq!(features.len(), 7);
            for &f in features {
                assert!(f.is_finite());
                assert!(
                    (0.0..=2.0).contains(&f),
                    "feature {f} out of expected range"
                );
            }
        } else {
            panic!("expected Continuous input");
        }
    }

    #[test]
    fn concept_edges_initially_empty() {
        let brain = AirSpringBrain::new(AirSpringBrainConfig::default(), "test");
        assert!(brain.concept_edge_doys().is_empty());
        assert!(!brain.is_drifting());
    }

    #[test]
    fn drift_monitor_accessible() {
        let brain = AirSpringBrain::new(AirSpringBrainConfig::default(), "test");
        let dm = brain.drift_monitor();
        assert!(!dm.is_drifting());
    }

    #[test]
    fn multiple_predictions_consistent() {
        let config = AirSpringBrainConfig::default();
        let mut brain = AirSpringBrain::new(config, "test");

        for doy in 1..=10 {
            brain.observe(make_obs(doy * 15, 25.0, 10.0, 3.0 + f64::from(doy) * 0.1));
        }
        brain.train();

        let obs = make_obs(75, 28.0, 12.0, 0.0);
        let p1 = brain.predict(&obs).unwrap();
        let p2 = brain.predict(&obs).unwrap();
        assert!(
            (p1.et0 - p2.et0).abs() < 1e-12,
            "predictions should be deterministic"
        );
    }
}
