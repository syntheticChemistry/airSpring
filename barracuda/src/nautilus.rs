// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nautilus Brain integration for agricultural regime prediction.
//!
//! Wraps `bingocube_nautilus::NautilusBrain` with agricultural domain mapping:
//! - **ET₀ head**: Predict reference evapotranspiration from weather features
//! - **Soil moisture head**: Predict volumetric water content trajectory
//! - **Crop stress head**: Predict yield response (Ky-based water stress)
//!
//! The brain evolves board populations to predict agricultural observables.
//! Built-in drift detection identifies regime changes (drought onset, season
//! transitions). Concept edge detection identifies days where predictions
//! fail — the most scientifically interesting regime boundaries.
//!
//! # Cross-Spring Provenance
//!
//! - `bingocube-nautilus` from `primalTools/bingoCube/nautilus/`
//! - Brain pattern from hotSpring v0.6.15 (`NautilusBrain` for QCD)
//! - Agricultural domain mapping: weather observations → reservoir features
//! - Board populations map to AKD1000 int4 for edge deployment

use bingocube_nautilus::{BetaObservation, NautilusBrain, NautilusBrainConfig, ShellConfig};

/// Number of prediction targets for the agricultural brain.
///
/// The upstream `NautilusBrain` returns a 3-tuple from `predict_dynamical`:
/// 0: ET₀ (mm/day, normalized)
/// 1: soil moisture deficit (0-1)
/// 2: crop stress factor (0-1, where 1 = no stress)
pub const N_TARGETS: usize = 3;

/// Configuration for the agricultural Nautilus brain.
#[derive(Debug, Clone)]
pub struct AirSpringBrainConfig {
    /// Upstream brain configuration.
    pub brain: NautilusBrainConfig,

    /// LOO error threshold for concept edge detection.
    /// Points above this threshold are flagged as regime boundaries
    /// (drought onset, frost events, seasonal transitions).
    pub concept_edge_threshold: f64,
}

impl Default for AirSpringBrainConfig {
    fn default() -> Self {
        Self {
            brain: NautilusBrainConfig {
                shell: ShellConfig {
                    population_size: 24,
                    n_targets: N_TARGETS,
                    ridge_lambda: 1e-4,
                    input_dim: 7,
                },
                generations_per_cycle: 20,
                min_training_points: 5,
                concept_edge_threshold: 0.15,
                edge_seed_count: 3,
            },
            concept_edge_threshold: 0.15,
        }
    }
}

/// A single weather/soil observation for training the agricultural brain.
#[derive(Debug, Clone)]
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
    /// Map agricultural observation to upstream `BetaObservation`.
    ///
    /// The reservoir doesn't care about physical meaning of fields — it
    /// processes normalised continuous features. We map agricultural
    /// observables onto the upstream physics-domain struct fields:
    ///
    /// | Upstream field | Agricultural mapping | Normalisation |
    /// |---------------|---------------------|---------------|
    /// | `beta` | day of year | DOY / 366 |
    /// | `plaquette` | ET₀ | ET₀ / max_et0 |
    /// | `cg_iters` | temperature range | (Tmax-Tmin) / 30 |
    /// | `acceptance` | relative humidity | RH / 100 |
    /// | `delta_h_abs` | precipitation | P / 50 |
    /// | `quenched_plaq` | solar radiation | Rs / 40 |
    /// | `quenched_plaq_var` | wind speed | u2 / 10 |
    /// | `anderson_r` | soil deficit | direct (0-1) |
    /// | `anderson_lambda_min` | crop stress | direct (0-1) |
    fn to_beta_observation(&self, et0_max: f64) -> BetaObservation {
        let et0_norm = if et0_max > 0.0 {
            self.et0_observed / et0_max
        } else {
            0.0
        };
        BetaObservation {
            beta: f64::from(self.doy) / 366.0,
            plaquette: et0_norm,
            cg_iters: (self.tmax - self.tmin) / 30.0,
            acceptance: self.rh_mean / 100.0,
            delta_h_abs: self.precip / 50.0,
            quenched_plaq: Some(self.solar_rad / 40.0),
            quenched_plaq_var: Some(self.wind_2m / 10.0),
            anderson_r: Some(self.soil_deficit),
            anderson_lambda_min: Some(self.crop_stress),
        }
    }
}

/// Agricultural Nautilus brain — evolutionary reservoir for weather/soil prediction.
///
/// Follows the same pattern as hotSpring's `NautilusBrain` for QCD, adapted
/// to agricultural domain heads (ET₀, soil moisture, crop stress).
pub struct AirSpringBrain {
    config: AirSpringBrainConfig,
    brain: NautilusBrain,
    observations: Vec<WeatherObservation>,
    concept_edge_doys: Vec<u16>,
    trained: bool,
}

impl AirSpringBrain {
    /// Create a new agricultural brain for a named instance (e.g. "eastgate").
    #[must_use]
    pub fn new(config: AirSpringBrainConfig, instance: &str) -> Self {
        let brain = NautilusBrain::new(config.brain.clone(), instance);

        Self {
            config,
            brain,
            observations: Vec::new(),
            concept_edge_doys: Vec::new(),
            trained: false,
        }
    }

    /// Record a weather/soil observation.
    pub fn observe(&mut self, obs: WeatherObservation) {
        let et0_max = self.et0_max().max(obs.et0_observed);
        let beta_obs = obs.to_beta_observation(et0_max);
        self.brain.observe(beta_obs);
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
        self.brain.is_drifting()
    }

    /// Days of year flagged as concept edges (regime boundaries).
    #[must_use]
    pub fn concept_edge_doys(&self) -> &[u16] {
        &self.concept_edge_doys
    }

    /// Maximum observed ET₀ (for denormalization).
    fn et0_max(&self) -> f64 {
        self.observations
            .iter()
            .map(|o| o.et0_observed)
            .fold(0.0_f64, f64::max)
    }

    /// Train the brain on all accumulated observations.
    /// Returns MSE, or `None` if insufficient data.
    pub fn train(&mut self) -> Option<f64> {
        if self.observations.len() < self.config.brain.min_training_points {
            return None;
        }

        let mse = self.brain.train();

        if mse.is_some() {
            self.detect_concept_edges();
            self.trained = true;
        }

        mse
    }

    /// Predict ET₀, soil deficit, and crop stress for a weather observation.
    /// Returns `None` if untrained.
    #[must_use]
    pub fn predict(&self, obs: &WeatherObservation) -> Option<AgPrediction> {
        if !self.trained {
            return None;
        }

        let doy_norm = f64::from(obs.doy) / 366.0;
        let (p0, p1, p2) = self.brain.predict_dynamical(doy_norm, None)?;

        let et0_max = self.et0_max();
        Some(AgPrediction {
            et0: p0 * et0_max,
            soil_deficit: p1.clamp(0.0, 1.0),
            crop_stress: p2.clamp(0.0, 1.0),
        })
    }

    /// Export brain state as JSON for transfer to another station/instance.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the brain state cannot be serialized.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        self.brain.to_json()
    }

    /// Import a brain from JSON (cross-station transfer).
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the JSON cannot be deserialized.
    pub fn import_json(
        config: AirSpringBrainConfig,
        json: &str,
    ) -> Result<Self, serde_json::Error> {
        let brain = NautilusBrain::from_json(json)?;
        Ok(Self {
            config,
            brain,
            observations: Vec::new(),
            concept_edge_doys: Vec::new(),
            trained: true,
        })
    }

    fn detect_concept_edges(&mut self) {
        let edges = self.brain.detect_concept_edges();
        self.concept_edge_doys.clear();

        for (beta, _error) in &edges {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let doy = (beta * 366.0).round() as u16;
            if doy > 0 && doy <= 366 {
                self.concept_edge_doys.push(doy);
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
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
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
            brain: NautilusBrainConfig {
                min_training_points: 10,
                ..AirSpringBrainConfig::default().brain
            },
            ..AirSpringBrainConfig::default()
        };
        let mut brain = AirSpringBrain::new(config, "test");
        for doy in 1..=3 {
            brain.observe(make_obs(doy * 30, 25.0, 10.0, 3.0));
        }
        assert!(brain.train().is_none());
    }

    #[test]
    fn default_config_is_reasonable() {
        let config = AirSpringBrainConfig::default();
        assert_eq!(config.brain.shell.population_size, 24);
        assert_eq!(config.brain.shell.n_targets, N_TARGETS);
        assert_eq!(config.brain.generations_per_cycle, 20);
        assert_eq!(config.brain.min_training_points, 5);
    }

    #[test]
    fn observation_mapping_bounded() {
        let obs = make_obs(182, 35.0, 20.0, 7.5);
        let beta = obs.to_beta_observation(8.0);
        assert!(beta.beta >= 0.0 && beta.beta <= 1.0);
        assert!(beta.plaquette >= 0.0);
        assert!(beta.cg_iters >= 0.0);
        assert!(beta.acceptance >= 0.0 && beta.acceptance <= 1.0);
    }

    #[test]
    fn concept_edges_initially_empty() {
        let brain = AirSpringBrain::new(AirSpringBrainConfig::default(), "test");
        assert!(brain.concept_edge_doys().is_empty());
        assert!(!brain.is_drifting());
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
