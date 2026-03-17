// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared types for dual crop coefficient (Kcb + Ke) — FAO-56 Chapters 7 + 11.

/// FAO-56 Table 17 basal crop coefficients for three growth stages.
#[derive(Debug, Clone, Copy)]
pub struct BasalCropCoefficients {
    /// Kcb during initial growth stage (bare soil dominates).
    pub kcb_ini: f64,
    /// Kcb during mid-season (full cover, transpiration dominates).
    pub kcb_mid: f64,
    /// Kcb during late season (senescence).
    pub kcb_end: f64,
    /// Maximum crop height (m) — needed for `Kc_max` calculation.
    pub max_height_m: f64,
}

/// Readily evaporable water (REW) and parameters for the evaporation layer.
#[derive(Debug, Clone, Copy)]
pub struct EvaporationParams {
    /// Field capacity of evaporation layer (m³/m³).
    pub theta_fc: f64,
    /// Wilting point of evaporation layer (m³/m³).
    pub theta_wp: f64,
    /// Readily evaporable water (mm) — stage 1 limit.
    pub rew_mm: f64,
}

/// State of the evaporation layer for multi-day simulation.
#[derive(Debug, Clone, Copy)]
pub struct EvaporationLayerState {
    /// Cumulative depth of evaporation from the soil surface (mm).
    pub de: f64,
    /// Total evaporable water (readily + slowly evaporable) (mm).
    pub tew: f64,
    /// Readily evaporable water (stage I limit) (mm).
    pub rew: f64,
}

/// Output of a single dual Kc simulation step.
#[derive(Debug, Clone, Copy)]
pub struct DualKcOutput {
    /// Cumulative evaporation depth from soil surface (mm).
    pub de: f64,
    /// Transpiration reduction coefficient (0–1).
    pub kr: f64,
    /// Evaporation coefficient for soil surface.
    pub ke: f64,
    /// Crop evapotranspiration (mm/day).
    pub etc: f64,
}

/// Daily input for dual Kc simulation.
#[derive(Debug, Clone, Copy)]
pub struct DualKcInput {
    /// Reference evapotranspiration (mm/day).
    pub et0: f64,
    /// Precipitation (mm).
    pub precipitation: f64,
    /// Irrigation (mm).
    pub irrigation: f64,
}
