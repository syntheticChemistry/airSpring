// SPDX-License-Identifier: AGPL-3.0-or-later
//! FAO-56 Table 17 basal crop coefficients for main crops.

use crate::eco::crop::CropType;

use super::types::BasalCropCoefficients;

impl CropType {
    /// FAO-56 Table 17 basal crop coefficients.
    ///
    /// These are lower than [`CropType::coefficients`] (Table 12) because
    /// they exclude soil evaporation — the Ke component accounts for it.
    #[must_use]
    pub const fn basal_coefficients(self) -> BasalCropCoefficients {
        match self {
            Self::Corn => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.15,
                kcb_end: 0.50,
                max_height_m: 2.0,
            },
            Self::Soybean => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.30,
                max_height_m: 0.75,
            },
            Self::WinterWheat => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 1.0,
            },
            Self::Alfalfa => BasalCropCoefficients {
                kcb_ini: 0.30,
                kcb_mid: 0.90,
                kcb_end: 0.85,
                max_height_m: 0.7,
            },
            Self::Tomato => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.70,
                max_height_m: 0.6,
            },
            Self::Potato => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.65,
                max_height_m: 0.6,
            },
            Self::SugarBeet => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.15,
                kcb_end: 0.90,
                max_height_m: 0.5,
            },
            Self::DryBean => BasalCropCoefficients {
                kcb_ini: 0.15,
                kcb_mid: 1.10,
                kcb_end: 0.25,
                max_height_m: 0.4,
            },
            Self::Blueberry => BasalCropCoefficients {
                kcb_ini: 0.20,
                kcb_mid: 0.95,
                kcb_end: 0.55,
                max_height_m: 1.5,
            },
            Self::Turfgrass => BasalCropCoefficients {
                kcb_ini: 0.80,
                kcb_mid: 0.85,
                kcb_end: 0.85,
                max_height_m: 0.10,
            },
        }
    }
}
