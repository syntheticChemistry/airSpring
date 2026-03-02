// SPDX-License-Identifier: AGPL-3.0-or-later
//! SCS Curve Number runoff estimation (USDA-SCS 1972).
//!
//! The SCS-CN method estimates direct runoff Q from rainfall P:
//!
//! ```text
//! Q = (P − Ia)² / (P − Ia + S)    when P > Ia
//! Q = 0                            when P ≤ Ia
//! ```
//!
//! where S = (25400/CN) − 254 mm and Ia = λ·S (λ = 0.2 standard).
//!
//! # Hydrologic Soil Groups
//!
//! - **A**: Low runoff (sand, loamy sand). Ksat > 7.6 mm/hr.
//! - **B**: Moderate (silt loam, loam). 3.8–7.6 mm/hr.
//! - **C**: High (sandy clay loam). 1.3–3.8 mm/hr.
//! - **D**: Very high (clay). < 1.3 mm/hr.
//!
//! # Reference
//!
//! USDA-SCS (1972) National Engineering Handbook, Section 4.
//! USDA-SCS (1986) TR-55: Urban Hydrology for Small Watersheds.

/// Hydrologic Soil Group (HSG) per USDA classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoilGroup {
    A,
    B,
    C,
    D,
}

/// Land use categories with standard CN values from NEH-4 / TR-55.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandUse {
    FallowBare,
    RowCropsStraight,
    RowCropsContoured,
    SmallGrainStraight,
    PastureGood,
    Meadow,
    WoodsGood,
    Farmstead,
    Impervious,
}

impl LandUse {
    /// Standard CN for this land use and soil group (AMC-II, average conditions).
    #[must_use]
    #[allow(clippy::match_same_arms)] // justified: Impervious returns same CN for all soil groups
    pub const fn cn(self, group: SoilGroup) -> u8 {
        match (self, group) {
            (Self::FallowBare, SoilGroup::A) => 77,
            (Self::FallowBare, SoilGroup::B) => 86,
            (Self::FallowBare, SoilGroup::C) => 91,
            (Self::FallowBare, SoilGroup::D) => 94,
            (Self::RowCropsStraight, SoilGroup::A) => 72,
            (Self::RowCropsStraight, SoilGroup::B) => 81,
            (Self::RowCropsStraight, SoilGroup::C) => 88,
            (Self::RowCropsStraight, SoilGroup::D) => 91,
            (Self::RowCropsContoured, SoilGroup::A) => 67,
            (Self::RowCropsContoured, SoilGroup::B) => 78,
            (Self::RowCropsContoured, SoilGroup::C) => 85,
            (Self::RowCropsContoured, SoilGroup::D) => 89,
            (Self::SmallGrainStraight, SoilGroup::A) => 65,
            (Self::SmallGrainStraight, SoilGroup::B) => 76,
            (Self::SmallGrainStraight, SoilGroup::C) => 84,
            (Self::SmallGrainStraight, SoilGroup::D) => 88,
            (Self::PastureGood, SoilGroup::A) => 39,
            (Self::PastureGood, SoilGroup::B) => 61,
            (Self::PastureGood, SoilGroup::C) => 74,
            (Self::PastureGood, SoilGroup::D) => 80,
            (Self::Meadow, SoilGroup::A) => 30,
            (Self::Meadow, SoilGroup::B) => 58,
            (Self::Meadow, SoilGroup::C) => 71,
            (Self::Meadow, SoilGroup::D) => 78,
            (Self::WoodsGood, SoilGroup::A) => 30,
            (Self::WoodsGood, SoilGroup::B) => 55,
            (Self::WoodsGood, SoilGroup::C) => 70,
            (Self::WoodsGood, SoilGroup::D) => 77,
            (Self::Farmstead, SoilGroup::A) => 59,
            (Self::Farmstead, SoilGroup::B) => 74,
            (Self::Farmstead, SoilGroup::C) => 82,
            (Self::Farmstead, SoilGroup::D) => 86,
            (Self::Impervious, _) => 98,
        }
    }
}

/// Potential maximum retention S (mm) from curve number.
///
/// S = (25400 / CN) − 254
#[must_use]
pub fn potential_retention(cn: f64) -> f64 {
    if cn <= 0.0 {
        return f64::MAX;
    }
    (25_400.0 / cn) - 254.0
}

/// Initial abstraction Ia (mm).
///
/// Standard: Ia = 0.2 × S. Updated (Woodward 2003): Ia = 0.05 × S.
#[must_use]
pub fn initial_abstraction(s_mm: f64, ia_ratio: f64) -> f64 {
    ia_ratio * s_mm
}

/// SCS Curve Number direct runoff Q (mm).
///
/// Q = (P − Ia)² / (P − Ia + S) when P > Ia, else 0.
///
/// # Arguments
///
/// * `precip_mm` — Event precipitation (mm).
/// * `cn` — Curve number (0–100).
/// * `ia_ratio` — Initial abstraction ratio (0.2 standard, 0.05 updated).
#[must_use]
pub fn scs_cn_runoff(precip_mm: f64, cn: f64, ia_ratio: f64) -> f64 {
    if precip_mm <= 0.0 || cn <= 0.0 {
        return 0.0;
    }
    let s = potential_retention(cn);
    let ia = initial_abstraction(s, ia_ratio);
    if precip_mm <= ia {
        return 0.0;
    }
    let pe = precip_mm - ia;
    #[allow(clippy::suspicious_operation_groupings)] // justified: SCS-CN formula Q = pe²/(pe+S)
    {
        pe * pe / (pe + s)
    }
}

/// SCS-CN runoff with standard Ia = 0.2S.
#[must_use]
pub fn scs_cn_runoff_standard(precip_mm: f64, cn: f64) -> f64 {
    scs_cn_runoff(precip_mm, cn, 0.2)
}

/// Antecedent Moisture Condition I (dry) CN from AMC-II (Hawkins 1985).
#[must_use]
pub fn amc_cn_dry(cn_ii: f64) -> f64 {
    cn_ii / 0.01281f64.mul_add(-cn_ii, 2.281)
}

/// Antecedent Moisture Condition III (wet) CN from AMC-II (Hawkins 1985).
#[must_use]
pub fn amc_cn_wet(cn_ii: f64) -> f64 {
    cn_ii / 0.0059f64.mul_add(cn_ii, 0.4036)
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // justified: test assertions on runoff values
mod tests {
    use super::*;

    #[test]
    fn cn75_25mm() {
        let q = scs_cn_runoff_standard(25.0, 75.0);
        assert!((q - 0.70).abs() < 0.1, "Q={q}");
    }

    #[test]
    fn cn75_50mm() {
        let q = scs_cn_runoff_standard(50.0, 75.0);
        assert!((q - 9.27).abs() < 0.1, "Q={q}");
    }

    #[test]
    fn cn90_50mm() {
        let q = scs_cn_runoff_standard(50.0, 90.0);
        assert!((q - 27.14).abs() < 0.2, "Q={q}");
    }

    #[test]
    fn no_runoff_below_ia() {
        let q = scs_cn_runoff_standard(50.0, 50.0);
        assert!(q.abs() < 0.01, "Q={q}");
    }

    #[test]
    fn zero_precip() {
        assert!(scs_cn_runoff_standard(0.0, 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cn100_all_runoff() {
        let q = scs_cn_runoff_standard(50.0, 100.0);
        assert!((q - 50.0).abs() < 0.01, "Q={q}");
    }

    #[test]
    fn q_never_exceeds_p() {
        for cn in [30, 50, 75, 85, 90, 95, 98, 100] {
            for p in [10.0, 25.0, 50.0, 100.0, 200.0] {
                let q = scs_cn_runoff_standard(p, f64::from(cn));
                assert!(q <= p + 0.001, "Q={q} > P={p} at CN={cn}");
            }
        }
    }

    #[test]
    fn cn_monotonic() {
        let mut prev = 0.0;
        for cn in [30, 50, 65, 75, 85, 90, 95, 98] {
            let q = scs_cn_runoff_standard(50.0, f64::from(cn));
            assert!(q >= prev - 1e-10, "CN={cn}: Q={q} < prev={prev}");
            prev = q;
        }
    }

    #[test]
    fn potential_retention_values() {
        let s = potential_retention(75.0);
        assert!((s - 84.667).abs() < 0.01, "S={s}");
    }

    #[test]
    fn amc_adjustments() {
        let cn_i = amc_cn_dry(75.0);
        let cn_iii = amc_cn_wet(75.0);
        assert!((cn_i - 56.8).abs() < 0.5, "CN_I={cn_i}");
        assert!((cn_iii - 88.6).abs() < 0.5, "CN_III={cn_iii}");
        assert!(cn_i < 75.0 && cn_iii > 75.0);
    }

    #[test]
    fn soil_group_ordering() {
        let cn_a = LandUse::RowCropsStraight.cn(SoilGroup::A);
        let cn_d = LandUse::RowCropsStraight.cn(SoilGroup::D);
        assert!(cn_a < cn_d);
    }

    #[test]
    fn updated_ia_more_runoff() {
        let q_std = scs_cn_runoff(50.0, 75.0, 0.2);
        let q_upd = scs_cn_runoff(50.0, 75.0, 0.05);
        assert!(q_upd > q_std, "Updated Ia should give more runoff");
    }

    #[test]
    fn scs_runoff_p_le_ia_returns_zero() {
        let q = scs_cn_runoff(5.0, 90.0, 0.2);
        assert!(q < 0.01, "P <= Ia should give Q ≈ 0, got Q={q}");
    }

    #[test]
    fn scs_runoff_cn_zero_returns_zero() {
        assert!(scs_cn_runoff(50.0, 0.0, 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn scs_runoff_negative_precip_returns_zero() {
        assert!(scs_cn_runoff(-10.0, 75.0, 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn potential_retention_cn_zero_returns_max() {
        let s = potential_retention(0.0);
        assert_eq!(s, f64::MAX);
    }

    #[test]
    fn potential_retention_negative_cn() {
        let s = potential_retention(-10.0);
        assert_eq!(s, f64::MAX);
    }

    #[test]
    fn initial_abstraction_standard() {
        let s = potential_retention(75.0);
        let ia = initial_abstraction(s, 0.2);
        assert!((ia - 16.93).abs() < 0.1, "Ia = 0.2*S for CN=75");
    }

    #[test]
    fn land_use_impervious_all_soil_groups() {
        for group in [SoilGroup::A, SoilGroup::B, SoilGroup::C, SoilGroup::D] {
            assert_eq!(LandUse::Impervious.cn(group), 98);
        }
    }

    #[test]
    fn land_use_meadow_all_groups() {
        assert_eq!(LandUse::Meadow.cn(SoilGroup::A), 30);
        assert_eq!(LandUse::Meadow.cn(SoilGroup::B), 58);
        assert_eq!(LandUse::Meadow.cn(SoilGroup::C), 71);
        assert_eq!(LandUse::Meadow.cn(SoilGroup::D), 78);
    }

    #[test]
    fn land_use_woods_good_all_groups() {
        assert_eq!(LandUse::WoodsGood.cn(SoilGroup::A), 30);
        assert_eq!(LandUse::WoodsGood.cn(SoilGroup::B), 55);
        assert_eq!(LandUse::WoodsGood.cn(SoilGroup::C), 70);
        assert_eq!(LandUse::WoodsGood.cn(SoilGroup::D), 77);
    }

    #[test]
    fn land_use_fallow_bare_all_groups() {
        assert_eq!(LandUse::FallowBare.cn(SoilGroup::A), 77);
        assert_eq!(LandUse::FallowBare.cn(SoilGroup::D), 94);
    }

    #[test]
    fn amc_cn_dry_lower_than_ii() {
        let cn_ii = 75.0;
        let cn_i = amc_cn_dry(cn_ii);
        assert!(cn_i < cn_ii, "AMC I (dry) should be lower than AMC II");
    }

    #[test]
    fn amc_cn_wet_higher_than_ii() {
        let cn_ii = 75.0;
        let cn_iii = amc_cn_wet(cn_ii);
        assert!(cn_iii > cn_ii, "AMC III (wet) should be higher than AMC II");
    }

    #[test]
    fn scs_runoff_amc_adjusted_dry_less_runoff() {
        let cn_ii = 75.0;
        let cn_i = amc_cn_dry(cn_ii);
        let q_ii = scs_cn_runoff_standard(50.0, cn_ii);
        let q_i = scs_cn_runoff_standard(50.0, cn_i);
        assert!(q_i < q_ii, "Dry AMC should produce less runoff");
    }

    #[test]
    fn scs_runoff_amc_adjusted_wet_more_runoff() {
        let cn_ii = 75.0;
        let cn_iii = amc_cn_wet(cn_ii);
        let q_ii = scs_cn_runoff_standard(50.0, cn_ii);
        let q_iii = scs_cn_runoff_standard(50.0, cn_iii);
        assert!(q_iii > q_ii, "Wet AMC should produce more runoff");
    }

    #[test]
    fn p_exactly_at_ia_boundary() {
        let s = potential_retention(80.0);
        let ia = initial_abstraction(s, 0.2);
        let q = scs_cn_runoff(ia, 80.0, 0.2);
        assert!(q < 0.001, "P exactly at Ia should give Q ≈ 0");
    }
}
