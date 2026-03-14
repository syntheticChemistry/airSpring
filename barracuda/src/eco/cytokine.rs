// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cytokine brain — evolutionary reservoir for AD flare prediction.
//!
//! Adapts the [`AirSpringBrain`](crate::nautilus::AirSpringBrain) pattern from
//! agricultural regime prediction to immunological regime prediction. Three
//! prediction heads map cytokine panel data onto Anderson localization parameters:
//!
//! - **IL-31 propagation head**: predicts signal extent (localized vs propagating)
//! - **Cell diversity head**: predicts tissue disorder `W` (Pielou evenness)
//! - **Barrier state head**: predicts effective dimension `d_eff` (intact 2D vs breached 3D)
//!
//! The brain's built-in drift detection flags AD flare onset when `N_e * s` drops
//! below threshold — the immunological equivalent of drought onset in the
//! agricultural brain.
//!
//! # Cross-Spring Provenance
//!
//! - Paper 01 (Anderson QS): Level spacing ratio `r`, disorder `W`, `W_c` thresholds
//! - Paper 06 (no-till): Dimensional collapse duality with Paper 12's dimensional promotion
//! - Paper 12 (immunological Anderson): Cytokine propagation through skin tissue
//! - `bingocube-nautilus`: Evolutionary reservoir computing
//! - hotSpring v0.6.15: `NautilusBrain` architecture
//! - Gonzales catalog (G1-G6): Empirical IL-31 dose-response data

use bingocube_nautilus::{BetaObservation, NautilusBrain, NautilusBrainConfig, ShellConfig};

use super::tissue::AndersonRegime;

/// Number of prediction targets for the cytokine brain.
///
/// 0: IL-31 signal extent (0=fully localized, 1=fully propagating)
/// 1: tissue disorder W (normalized to 0-1)
/// 2: barrier integrity (0=fully breached/3D, 1=fully intact/2D)
pub const N_CYTOKINE_TARGETS: usize = 3;

/// Configuration for the cytokine Nautilus brain.
#[derive(Debug, Clone)]
pub struct CytokineBrainConfig {
    /// Upstream brain configuration.
    pub brain: NautilusBrainConfig,
    /// MSE threshold for concept edge detection.
    /// Points above this flag AD flare boundaries.
    pub concept_edge_threshold: f64,
}

impl Default for CytokineBrainConfig {
    fn default() -> Self {
        Self {
            brain: NautilusBrainConfig {
                shell: ShellConfig {
                    population_size: 24,
                    n_targets: N_CYTOKINE_TARGETS,
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

/// A single cytokine panel observation from a tissue sample or serum measurement.
///
/// Based on Gonzales catalog: IL-31 (G1/G3), IL-4/IL-13 (G2/G6), barrier metrics.
#[derive(Debug, Clone)]
pub struct CytokineObservation {
    /// Time point (hours post-treatment or days in study).
    pub time_hours: f64,
    /// IL-31 serum/tissue level (pg/mL, normalized internally).
    pub il31_level: f64,
    /// IL-4 level (pg/mL).
    pub il4_level: f64,
    /// IL-13 level (pg/mL).
    pub il13_level: f64,
    /// Pruritus score (0-10 scale, from Gonzales G3 standardized model).
    pub pruritus_score: f64,
    /// TEWL — transepidermal water loss (g/m²/h), a proxy for barrier integrity.
    pub tewl: f64,
    /// Pielou evenness of tissue cell types (0-1).
    pub pielou_evenness: f64,

    /// Observed signal extent (0=localized, 1=propagating) — target for head 0.
    pub signal_extent_observed: f64,
    /// Observed tissue disorder W (normalized 0-1) — target for head 1.
    pub w_observed: f64,
    /// Observed barrier integrity (0=breached, 1=intact) — target for head 2.
    pub barrier_integrity_observed: f64,
}

impl CytokineObservation {
    /// Map cytokine observation to upstream `BetaObservation`.
    ///
    /// See [`crate::nautilus::WeatherObservation::to_beta_observation`] for the
    /// general mapping strategy.
    fn to_beta_observation(&self) -> BetaObservation {
        BetaObservation {
            beta: self.time_hours / 720.0,
            plaquette: self.signal_extent_observed,
            cg_iters: self.il31_level / 500.0,
            acceptance: self.pielou_evenness,
            delta_h_abs: self.pruritus_score / 10.0,
            quenched_plaq: Some(self.tewl / 100.0),
            quenched_plaq_var: Some(self.il4_level / 200.0),
            anderson_r: Some(self.w_observed),
            anderson_lambda_min: Some(self.barrier_integrity_observed),
        }
    }
}

/// Cytokine brain — evolutionary reservoir for immunological regime prediction.
///
/// Follows the same architecture as [`AirSpringBrain`](crate::nautilus::AirSpringBrain)
/// adapted for cytokine panel data instead of weather observations.
pub struct CytokineBrain {
    config: CytokineBrainConfig,
    brain: NautilusBrain,
    observations: Vec<CytokineObservation>,
    concept_edge_hours: Vec<u32>,
    trained: bool,
}

/// Prediction output from the cytokine brain.
#[derive(Debug, Clone, Copy)]
pub struct CytokinePrediction {
    /// Predicted signal extent (0=localized, 1=propagating).
    pub signal_extent: f64,
    /// Predicted tissue disorder W (normalized 0-1).
    pub w_predicted: f64,
    /// Predicted barrier integrity (0=breached, 1=intact).
    pub barrier_integrity: f64,
}

impl CytokinePrediction {
    /// Classify the predicted Anderson regime from the prediction heads.
    #[must_use]
    pub fn anderson_regime(&self) -> AndersonRegime {
        if self.signal_extent > 0.7 && self.barrier_integrity < 0.3 {
            AndersonRegime::Extended
        } else if self.signal_extent < 0.3 && self.barrier_integrity > 0.7 {
            AndersonRegime::Localized
        } else {
            AndersonRegime::Critical
        }
    }
}

impl CytokineBrain {
    /// Create a new cytokine brain for a named instance (e.g. "gonzales-lab").
    #[must_use]
    pub fn new(config: CytokineBrainConfig, instance: &str) -> Self {
        let brain = NautilusBrain::new(config.brain.clone(), instance);

        Self {
            config,
            brain,
            observations: Vec::new(),
            concept_edge_hours: Vec::new(),
            trained: false,
        }
    }

    /// Record a cytokine panel observation.
    pub fn observe(&mut self, obs: CytokineObservation) {
        let beta_obs = obs.to_beta_observation();
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

    /// Whether the drift monitor detects regime change (AD flare onset).
    #[must_use]
    pub fn is_drifting(&self) -> bool {
        self.brain.is_drifting()
    }

    /// Time points (hours) flagged as concept edges (regime boundaries).
    #[must_use]
    pub fn concept_edge_hours(&self) -> &[u32] {
        &self.concept_edge_hours
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

    /// Predict signal extent, tissue disorder, and barrier state for a cytokine observation.
    /// Returns `None` if untrained.
    #[must_use]
    pub fn predict(&self, obs: &CytokineObservation) -> Option<CytokinePrediction> {
        if !self.trained {
            return None;
        }

        let time_norm = obs.time_hours / 720.0;
        let (p0, p1, p2) = self.brain.predict_dynamical(time_norm, None)?;

        Some(CytokinePrediction {
            signal_extent: p0.clamp(0.0, 1.0),
            w_predicted: p1.clamp(0.0, 1.0),
            barrier_integrity: p2.clamp(0.0, 1.0),
        })
    }

    /// Export brain state as JSON for cross-species transfer.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the brain state cannot be serialized.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        self.brain.to_json()
    }

    /// Import a brain from JSON (cross-species or cross-study transfer).
    ///
    /// Enables One Health transfer: train on canine IL-31 data (Gonzales G1-G6),
    /// transfer to human AD data (Simpson D1, Silverberg D2).
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the JSON cannot be deserialized.
    pub fn import_json(config: CytokineBrainConfig, json: &str) -> Result<Self, serde_json::Error> {
        let brain = NautilusBrain::from_json(json)?;
        Ok(Self {
            config,
            brain,
            observations: Vec::new(),
            concept_edge_hours: Vec::new(),
            trained: true,
        })
    }

    fn detect_concept_edges(&mut self) {
        let edges = self.brain.detect_concept_edges();
        self.concept_edge_hours.clear();

        for (beta, _error) in &edges {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let hours = (beta * 720.0).round() as u32;
            self.concept_edge_hours.push(hours);
        }

        self.concept_edge_hours.sort_unstable();
        self.concept_edge_hours.dedup();
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    fn make_obs(
        time_hours: f64,
        il31: f64,
        pruritus: f64,
        signal_extent: f64,
        barrier: f64,
    ) -> CytokineObservation {
        CytokineObservation {
            time_hours,
            il31_level: il31,
            il4_level: 50.0,
            il13_level: 40.0,
            pruritus_score: pruritus,
            tewl: 25.0,
            pielou_evenness: 0.7,
            signal_extent_observed: signal_extent,
            w_observed: 0.4,
            barrier_integrity_observed: barrier,
        }
    }

    #[test]
    fn brain_lifecycle() {
        let config = CytokineBrainConfig::default();
        let mut brain = CytokineBrain::new(config, "gonzales-test");

        assert!(!brain.is_trained());
        assert_eq!(brain.observation_count(), 0);

        for i in 0..10 {
            let t = f64::from(i) * 6.0;
            let il31 = 100.0 + f64::from(i) * 20.0;
            let pruritus = 3.0 + f64::from(i) * 0.5;
            let signal = 0.3 + f64::from(i) * 0.05;
            let barrier = 0.8 - f64::from(i) * 0.03;
            brain.observe(make_obs(t, il31, pruritus, signal, barrier));
        }
        assert_eq!(brain.observation_count(), 10);

        let mse = brain.train();
        assert!(mse.is_some(), "training should succeed with 10 points");
        assert!(brain.is_trained());

        let pred = brain.predict(&make_obs(48.0, 200.0, 5.0, 0.0, 0.0));
        assert!(pred.is_some());
        let p = pred.unwrap();
        assert!(p.signal_extent.is_finite());
        assert!((0.0..=1.0).contains(&p.signal_extent));
        assert!((0.0..=1.0).contains(&p.w_predicted));
        assert!((0.0..=1.0).contains(&p.barrier_integrity));
    }

    #[test]
    fn untrained_returns_none() {
        let brain = CytokineBrain::new(CytokineBrainConfig::default(), "test");
        assert!(brain
            .predict(&make_obs(0.0, 100.0, 3.0, 0.5, 0.8))
            .is_none());
    }

    #[test]
    fn insufficient_data_returns_none() {
        let config = CytokineBrainConfig {
            brain: NautilusBrainConfig {
                min_training_points: 10,
                ..CytokineBrainConfig::default().brain
            },
            ..CytokineBrainConfig::default()
        };
        let mut brain = CytokineBrain::new(config, "test");
        for i in 0..3 {
            brain.observe(make_obs(f64::from(i) * 6.0, 100.0, 3.0, 0.5, 0.8));
        }
        assert!(brain.train().is_none());
    }

    #[test]
    fn anderson_regime_from_prediction() {
        let localized = CytokinePrediction {
            signal_extent: 0.1,
            w_predicted: 0.3,
            barrier_integrity: 0.9,
        };
        assert_eq!(localized.anderson_regime(), AndersonRegime::Localized);

        let extended = CytokinePrediction {
            signal_extent: 0.9,
            w_predicted: 0.8,
            barrier_integrity: 0.1,
        };
        assert_eq!(extended.anderson_regime(), AndersonRegime::Extended);

        let critical = CytokinePrediction {
            signal_extent: 0.5,
            w_predicted: 0.5,
            barrier_integrity: 0.5,
        };
        assert_eq!(critical.anderson_regime(), AndersonRegime::Critical);
    }

    #[test]
    fn default_config_is_reasonable() {
        let config = CytokineBrainConfig::default();
        assert_eq!(config.brain.shell.population_size, 24);
        assert_eq!(config.brain.shell.n_targets, N_CYTOKINE_TARGETS);
        assert_eq!(config.brain.generations_per_cycle, 20);
        assert_eq!(config.brain.min_training_points, 5);
    }

    #[test]
    fn concept_edges_initially_empty() {
        let brain = CytokineBrain::new(CytokineBrainConfig::default(), "test");
        assert!(brain.concept_edge_hours().is_empty());
        assert!(!brain.is_drifting());
    }

    #[test]
    fn predictions_deterministic() {
        let config = CytokineBrainConfig::default();
        let mut brain = CytokineBrain::new(config, "test");

        for i in 0..10 {
            brain.observe(make_obs(
                f64::from(i) * 6.0,
                100.0 + f64::from(i) * 10.0,
                3.0 + f64::from(i) * 0.2,
                0.3 + f64::from(i) * 0.04,
                0.8 - f64::from(i) * 0.02,
            ));
        }
        brain.train();

        let obs = make_obs(48.0, 200.0, 5.0, 0.0, 0.0);
        let p1 = brain.predict(&obs).unwrap();
        let p2 = brain.predict(&obs).unwrap();
        assert!(
            (p1.signal_extent - p2.signal_extent).abs() < 1e-12,
            "predictions should be deterministic"
        );
    }
}
