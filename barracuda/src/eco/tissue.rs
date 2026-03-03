// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tissue diversity profiling for immunological Anderson localization.
//!
//! Maps cell-type heterogeneity in biological tissue to the Anderson disorder
//! parameter W. Uses the same diversity metrics (Shannon, Simpson, Pielou
//! evenness) validated for soil microbiome (Paper 06) to characterize skin
//! tissue composition for cytokine propagation modeling (Paper 12).
//!
//! # The Anderson Mapping
//!
//! | Diversity Metric | Anderson Meaning |
//! |------------------|-----------------|
//! | Pielou evenness J' → 1 | Low effective W (uniform cell distribution ≈ periodic lattice) |
//! | Pielou J' → 0 | High effective W (one cell type dominates ≈ strong disorder) |
//! | Shannon H' | Total disorder magnitude |
//! | Species richness S | Number of distinct lattice site types |
//!
//! # Cross-Spring Provenance
//!
//! - Paper 01 (Anderson QS): W → Pielou evenness mapping for microbial communities
//! - Paper 06 (no-till): Soil cell-type diversity as W for QS propagation
//! - Paper 12 (immunological): Skin tissue cell-type diversity as W for cytokine propagation
//! - `gpu::diversity`: GPU-accelerated Shannon/Simpson/Pielou via `DiversityFusionGpu`

use crate::gpu::diversity::{DiversityMetrics, GpuDiversity};

/// Skin tissue compartment for the multi-layer Anderson model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkinCompartment {
    /// Viable epidermis: quasi-2D (4-8 cell layers), keratinocytes dominate.
    Epidermis,
    /// Papillary dermis: 3D matrix, high cell-type diversity.
    PapillaryDermis,
    /// Reticular dermis: 3D dense matrix, fibroblasts dominate.
    ReticularDermis,
}

impl SkinCompartment {
    /// Effective Anderson dimensionality for an intact (unbreached) compartment.
    #[must_use]
    pub const fn effective_dimension_intact(&self) -> f64 {
        match self {
            Self::Epidermis => 2.0,
            Self::PapillaryDermis | Self::ReticularDermis => 3.0,
        }
    }
}

/// Cell-type abundance for a tissue sample.
///
/// Each entry is a named cell type with its abundance count in the sample.
#[derive(Debug, Clone)]
pub struct CellTypeAbundance {
    /// Cell type name.
    pub cell_type: String,
    /// Raw abundance (counts or normalized frequency).
    pub abundance: f64,
}

/// Anderson tissue disorder analysis result.
#[derive(Debug, Clone)]
pub struct TissueDisorder {
    /// Compartment analyzed.
    pub compartment: SkinCompartment,
    /// Diversity metrics (Shannon H', Simpson D, Pielou J').
    pub diversity: DiversityMetrics,
    /// Effective Anderson disorder parameter W.
    /// `W` ∝ `(1 - Pielou J') × ln(S)`, where `S` is species richness.
    pub w_effective: f64,
    /// Anderson prediction: whether cytokine signals localize (`W > W_c`) or propagate (`W < W_c`).
    /// `None` if insufficient data to determine.
    pub regime: Option<AndersonRegime>,
}

/// Anderson regime classification for cytokine propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndersonRegime {
    /// Signals localize — cytokines confined near source (homeostatic in epidermis).
    Localized,
    /// Signals propagate — cytokines spread through tissue (active signaling in dermis).
    Extended,
    /// Near the critical point `W_c` — regime unstable, small perturbations flip state.
    Critical,
}

/// Analyze tissue cell-type composition and compute the Anderson disorder parameter.
///
/// Takes a vector of cell-type abundances for a single tissue sample and returns
/// the diversity metrics mapped to Anderson disorder parameters.
///
/// # Errors
///
/// Returns an error if the diversity computation fails.
pub fn analyze_tissue_disorder(
    cell_types: &[CellTypeAbundance],
    compartment: SkinCompartment,
    diversity_engine: &GpuDiversity,
) -> crate::error::Result<TissueDisorder> {
    if cell_types.is_empty() {
        return Err(crate::error::AirSpringError::InvalidInput(
            "empty cell type abundances".to_string(),
        ));
    }

    let abundances: Vec<f64> = cell_types.iter().map(|ct| ct.abundance).collect();
    let n_species = abundances.len();

    let metrics = diversity_engine.compute_alpha(&abundances, 1, n_species)?;

    let dm = metrics.into_iter().next().ok_or_else(|| {
        crate::error::AirSpringError::InvalidInput(
            "diversity computation returned no results".to_string(),
        )
    })?;

    let richness = n_species as f64;
    let w_effective = (1.0 - dm.evenness) * richness.ln();

    let d = compartment.effective_dimension_intact();
    let regime = classify_regime(w_effective, d);

    Ok(TissueDisorder {
        compartment,
        diversity: dm,
        w_effective,
        regime: Some(regime),
    })
}

/// Classify the Anderson regime based on effective W and dimensionality.
///
/// Uses the `W_c` thresholds from Paper 01 (validated in wetSpring Exp107-156):
/// - `d=2`: `W_c` ≈ 4.0 (all states localize above this)
/// - `d=3`: `W_c` ≈ 16.26 ± 0.95 (validated in 3,100+ checks)
fn classify_regime(w: f64, d: f64) -> AndersonRegime {
    let w_c = if d < 2.5 { 4.0 } else { 16.26 };

    let margin = 0.1 * w_c;
    if w > w_c + margin {
        AndersonRegime::Localized
    } else if w < w_c - margin {
        AndersonRegime::Extended
    } else {
        AndersonRegime::Critical
    }
}

/// Compute the effective dimension of a barrier-disrupted epidermis.
///
/// Scratching opens 3D channels through the normally 2D epidermal barrier.
/// `breach_fraction` (0.0-1.0) represents the fraction of barrier that is
/// disrupted. `d_eff` interpolates between 2D (intact) and 3D (fully breached).
///
/// Paper 12 §2.3: this is the "dimensional promotion" — inverse of
/// Paper 06's tillage "dimensional collapse".
#[must_use]
pub fn barrier_disruption_d_eff(breach_fraction: f64) -> f64 {
    let f = breach_fraction.clamp(0.0, 1.0);
    2.0 + f
}

/// Compute effective disorder W for a multi-compartment tissue stack.
///
/// Returns `(compartment, w_effective, regime)` for each layer.
///
/// # Errors
///
/// Returns an error if any compartment's diversity computation fails.
pub fn multi_compartment_analysis(
    compartments: &[(SkinCompartment, Vec<CellTypeAbundance>)],
    diversity_engine: &GpuDiversity,
) -> crate::error::Result<Vec<TissueDisorder>> {
    compartments
        .iter()
        .map(|(compartment, cell_types)| {
            analyze_tissue_disorder(cell_types, *compartment, diversity_engine)
        })
        .collect()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code may use unwrap for clarity")]
mod tests {
    use super::*;

    fn cell_types(abundances: &[(&str, f64)]) -> Vec<CellTypeAbundance> {
        abundances
            .iter()
            .map(|(name, abundance)| CellTypeAbundance {
                cell_type: (*name).to_string(),
                abundance: *abundance,
            })
            .collect()
    }

    #[test]
    fn healthy_epidermis_low_disorder() {
        let engine = GpuDiversity::cpu();
        let cells = cell_types(&[
            ("keratinocyte", 85.0),
            ("langerhans", 5.0),
            ("melanocyte", 8.0),
            ("merkel", 2.0),
        ]);

        let result = analyze_tissue_disorder(&cells, SkinCompartment::Epidermis, &engine).unwrap();

        assert!(
            result.diversity.evenness < 0.8,
            "epidermis dominated by keratinocytes → low evenness"
        );
        assert_eq!(result.compartment, SkinCompartment::Epidermis);
        assert!(result.w_effective.is_finite());
    }

    #[test]
    fn inflamed_dermis_high_diversity() {
        let engine = GpuDiversity::cpu();
        let cells = cell_types(&[
            ("fibroblast", 20.0),
            ("th2_cell", 15.0),
            ("mast_cell", 12.0),
            ("eosinophil", 10.0),
            ("dendritic_cell", 8.0),
            ("nerve_ending", 5.0),
            ("macrophage", 10.0),
            ("neutrophil", 8.0),
            ("endothelial", 12.0),
        ]);

        let result =
            analyze_tissue_disorder(&cells, SkinCompartment::PapillaryDermis, &engine).unwrap();

        assert!(
            result.diversity.evenness > 0.8,
            "inflamed dermis has diverse cell population → high evenness"
        );
        assert_eq!(result.compartment, SkinCompartment::PapillaryDermis);
    }

    #[test]
    fn barrier_disruption_increases_d_eff() {
        let d_intact = barrier_disruption_d_eff(0.0);
        let d_partial = barrier_disruption_d_eff(0.5);
        let d_full = barrier_disruption_d_eff(1.0);

        assert!((d_intact - 2.0).abs() < f64::EPSILON);
        assert!((d_partial - 2.5).abs() < f64::EPSILON);
        assert!((d_full - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn barrier_disruption_clamps() {
        assert!((barrier_disruption_d_eff(-0.5) - 2.0).abs() < f64::EPSILON);
        assert!((barrier_disruption_d_eff(1.5) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn regime_classification() {
        assert_eq!(classify_regime(1.0, 3.0), AndersonRegime::Extended);
        assert_eq!(classify_regime(20.0, 3.0), AndersonRegime::Localized);
        assert_eq!(classify_regime(16.0, 3.0), AndersonRegime::Critical);

        assert_eq!(classify_regime(1.0, 2.0), AndersonRegime::Extended);
        assert_eq!(classify_regime(6.0, 2.0), AndersonRegime::Localized);
    }

    #[test]
    fn multi_compartment_stack() {
        let engine = GpuDiversity::cpu();
        let compartments = vec![
            (
                SkinCompartment::Epidermis,
                cell_types(&[
                    ("keratinocyte", 85.0),
                    ("langerhans", 5.0),
                    ("melanocyte", 8.0),
                    ("merkel", 2.0),
                ]),
            ),
            (
                SkinCompartment::PapillaryDermis,
                cell_types(&[
                    ("fibroblast", 20.0),
                    ("th2_cell", 15.0),
                    ("mast_cell", 12.0),
                    ("eosinophil", 10.0),
                    ("dendritic_cell", 8.0),
                ]),
            ),
        ];

        let results = multi_compartment_analysis(&compartments, &engine).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].compartment, SkinCompartment::Epidermis);
        assert_eq!(results[1].compartment, SkinCompartment::PapillaryDermis);
    }

    #[test]
    fn empty_cell_types_returns_error() {
        let engine = GpuDiversity::cpu();
        let result = analyze_tissue_disorder(&[], SkinCompartment::Epidermis, &engine);
        assert!(result.is_err());
    }

    #[test]
    fn compartment_dimensions() {
        assert!(
            (SkinCompartment::Epidermis.effective_dimension_intact() - 2.0).abs() < f64::EPSILON
        );
        assert!(
            (SkinCompartment::PapillaryDermis.effective_dimension_intact() - 3.0).abs()
                < f64::EPSILON
        );
        assert!(
            (SkinCompartment::ReticularDermis.effective_dimension_intact() - 3.0).abs()
                < f64::EPSILON
        );
    }
}
