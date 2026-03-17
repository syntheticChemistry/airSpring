// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dual crop coefficient (Kcb + Ke) — FAO-56 Chapters 7 + 11.
//!
//! Separates crop evapotranspiration into transpiration (Kcb) and soil
//! evaporation (Ke) for precision irrigation scheduling:
//!
//! ```text
//! ETc = (Kcb × Ks + Ke) × ET₀     (Eq. 69)
//! ```
//!
//! This module provides:
//! - [`BasalCropCoefficients`] — Table 17 Kcb values per crop
//! - [`CoverCropType`] — Cover crop Kcb values for no-till systems
//! - [`EvaporationParams`] — Table 19 REW/TEW soil parameters
//! - Pure functions matching every FAO-56 equation (69, 71–73, 77)
//! - [`mulched_ke`] — FAO-56 Ch 11 mulch reduction on soil evaporation
//! - [`EvaporationLayerState`] — stateful daily simulation (with/without mulch)
//!
//! # References
//!
//! Allen RG, Pereira LS, Raes D, Smith M (1998)
//! FAO Irrigation and Drainage Paper 56, Chapters 7 + 11.
//!
//! Islam R, Reeder R (2014) No-till and conservation agriculture.
//! ISWCR 2(3): 176-186.

mod cover_crop;
mod crop_basal;
mod equations;
mod evaporation_params;
mod mulch;
mod simulation;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public API so `crate::eco::dual_kc::*` remains unchanged.
// Note: CropType::basal_coefficients and SoilTexture::evaporation_params are
// impl blocks on types from eco::crop and eco::soil_moisture — no re-export needed.
pub use cover_crop::CoverCropType;
pub use equations::{
    etc_dual, evaporation_layer_balance, evaporation_reduction, kc_max, soil_evaporation_ke,
    total_evaporable_water,
};
pub use mulch::{ResidueLevel, mulched_ke};
pub use simulation::{simulate_dual_kc, simulate_dual_kc_mulched};
pub use types::{
    BasalCropCoefficients, DualKcInput, DualKcOutput, EvaporationLayerState, EvaporationParams,
};
