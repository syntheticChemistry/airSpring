// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soil moisture sensor calibration — dielectric permittivity to volumetric water content.
//!
//! Implements the Topp equation (Topp et al., 1980), USDA soil texture
//! classification with hydraulic properties, Saxton & Rawls (2006)
//! pedotransfer functions, and water availability calculations.
//!
//! # Reference
//!
//! Topp GC, Davis JL, Annan AP (1980) "Electromagnetic
//! determination of soil water content" Water Resources Research 16(3), 574–582.

mod saxton_rawls;
mod texture;
mod topp;
mod water;

pub use saxton_rawls::{SaxtonRawlsInput, SaxtonRawlsResult, saxton_rawls};
pub use texture::{SoilHydraulicProps, SoilTexture};
pub use topp::{inverse_topp, topp_equation};
pub use water::{irrigation_trigger, plant_available_water, soil_water_deficit};
