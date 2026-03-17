// SPDX-License-Identifier: AGPL-3.0-or-later
//! FAO-56 Table 19 soil evaporation parameters.

use crate::eco::soil_moisture::SoilTexture;

use super::types::EvaporationParams;

impl SoilTexture {
    /// FAO-56 Table 19 evaporation parameters.
    ///
    /// REW values are typical midpoints for each USDA texture class.
    /// `θFC` and `θWP` are from Table 19 (may differ slightly from
    /// [`SoilTexture::hydraulic_properties`] which uses Saxton & Rawls).
    #[must_use]
    pub const fn evaporation_params(&self) -> EvaporationParams {
        match self {
            Self::Sand => EvaporationParams {
                theta_fc: 0.12,
                theta_wp: 0.04,
                rew_mm: 6.0,
            },
            Self::LoamySand => EvaporationParams {
                theta_fc: 0.16,
                theta_wp: 0.06,
                rew_mm: 6.0,
            },
            Self::SandyLoam => EvaporationParams {
                theta_fc: 0.23,
                theta_wp: 0.10,
                rew_mm: 8.0,
            },
            Self::Loam => EvaporationParams {
                theta_fc: 0.30,
                theta_wp: 0.15,
                rew_mm: 9.0,
            },
            Self::SiltLoam => EvaporationParams {
                theta_fc: 0.33,
                theta_wp: 0.13,
                rew_mm: 10.0,
            },
            Self::Silt => EvaporationParams {
                theta_fc: 0.36,
                theta_wp: 0.15,
                rew_mm: 10.0,
            },
            Self::SandyClayLoam => EvaporationParams {
                theta_fc: 0.33,
                theta_wp: 0.19,
                rew_mm: 8.0,
            },
            Self::ClayLoam => EvaporationParams {
                theta_fc: 0.36,
                theta_wp: 0.21,
                rew_mm: 9.0,
            },
            Self::SiltyClayLoam => EvaporationParams {
                theta_fc: 0.37,
                theta_wp: 0.21,
                rew_mm: 9.0,
            },
            Self::SandyClay => EvaporationParams {
                theta_fc: 0.36,
                theta_wp: 0.21,
                rew_mm: 8.0,
            },
            Self::SiltyClay => EvaporationParams {
                theta_fc: 0.40,
                theta_wp: 0.23,
                rew_mm: 10.0,
            },
            Self::Clay => EvaporationParams {
                theta_fc: 0.42,
                theta_wp: 0.25,
                rew_mm: 10.0,
            },
        }
    }
}
