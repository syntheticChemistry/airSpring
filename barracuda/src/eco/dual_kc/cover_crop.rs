// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cover crop types for no-till systems — FAO-56 Ch 11 + literature.

use super::types::BasalCropCoefficients;

/// Cover crop types for no-till systems.
///
/// Kcb values adapted from FAO-56 Table 17 and cover crop literature.
#[derive(Debug, Clone, Copy)]
pub enum CoverCropType {
    /// Winter cereal rye — dominant Midwest cover crop.
    CerealRye,
    /// Crimson clover — legume cover with moderate transpiration.
    CrimsonClover,
    /// Winter wheat used as cover crop (terminated early).
    WinterWheatCover,
    /// Hairy vetch — vining legume, good ground cover.
    HairyVetch,
    /// Daikon/tillage radish — winterkills, acts as green mulch.
    TillageRadish,
}

impl CoverCropType {
    /// Basal crop coefficients for cover crops.
    #[must_use]
    pub const fn basal_coefficients(self) -> BasalCropCoefficients {
        match self {
            Self::CerealRye => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 1.2,
            },
            Self::CrimsonClover => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 0.95,
                kcb_end: 0.30,
                max_height_m: 0.5,
            },
            Self::WinterWheatCover => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 1.0,
            },
            Self::HairyVetch => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 0.90,
                kcb_end: 0.25,
                max_height_m: 0.4,
            },
            Self::TillageRadish => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 0.85,
                kcb_end: 0.20,
                max_height_m: 0.3,
            },
        }
    }
}
