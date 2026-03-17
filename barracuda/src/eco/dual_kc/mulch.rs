// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mulch reduction on soil evaporation — FAO-56 Chapter 11.

use super::equations::soil_evaporation_ke;

/// No-till residue coverage levels and their mulch reduction factors.
///
/// The mulch factor reduces Ke: `Ke_mulch = Ke × mulch_factor`.
/// This accounts for surface residue blocking radiation from reaching
/// the soil, reducing stage 1 and stage 2 evaporation.
#[derive(Debug, Clone, Copy)]
pub enum ResidueLevel {
    /// Conventional tillage, bare soil.
    NoResidue,
    /// Light residue (<30% ground cover).
    Light,
    /// Moderate residue (30–60% ground cover).
    Moderate,
    /// Heavy residue (>60%, typical no-till).
    Heavy,
    /// Nearly complete cover (thick mulch).
    FullMulch,
}

impl ResidueLevel {
    /// Mulch reduction factor for soil evaporation.
    ///
    /// FAO-56 Chapter 11: surface residue reduces energy reaching the soil
    /// surface, reducing both stage 1 and stage 2 evaporation rates.
    #[must_use]
    pub const fn mulch_factor(self) -> f64 {
        match self {
            Self::NoResidue => 1.00,
            Self::Light => 0.80,
            Self::Moderate => 0.60,
            Self::Heavy => 0.40,
            Self::FullMulch => 0.25,
        }
    }
}

/// Soil evaporation with mulch reduction (FAO-56 Ch 11).
///
/// `Ke_mulch` = Ke × `mulch_factor`
#[must_use]
pub fn mulched_ke(kr: f64, kcb: f64, kc_max_val: f64, few: f64, mulch_factor: f64) -> f64 {
    soil_evaporation_ke(kr, kcb, kc_max_val, few) * mulch_factor
}
