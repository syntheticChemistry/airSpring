// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ecological diversity metrics for agroecosystem assessment.
//!
//! # Cross-Spring Provenance
//!
//! These metrics delegate to `barracuda::stats::diversity`, absorbed from
//! wetSpring `bio/diversity.rs` in `BarraCuda` S64. wetSpring originally
//! developed them for microbiome alpha/beta diversity (Shannon, Simpson,
//! Chao1, Bray-Curtis). airSpring reuses them for:
//!
//! - **Cover crop biodiversity** — species richness and evenness in
//!   multi-species cover crop mixtures
//! - **Soil microbiome diversity** — rhizosphere community assessment
//!   (when paired with amplicon sequencing data)
//! - **Pollinator habitat assessment** — plant community diversity indices
//!   for field margin evaluations
//!
//! # GPU Acceleration
//!
//! For large-scale diversity computation (many samples), `BarraCuda` provides
//! `barracuda::ops::bio::DiversityFusionGpu` which fuses Shannon + Simpson +
//! evenness into a single GPU dispatch. Wire through [`crate::gpu`] when
//! sample counts justify GPU overhead.

/// Alpha diversity summary for a single community sample.
///
/// Wraps [`barracuda::stats::AlphaDiversity`].
pub type AlphaDiversity = barracuda::stats::AlphaDiversity;

/// Shannon entropy H' = −Σ pᵢ ln(pᵢ).
///
/// Higher values indicate greater diversity. Input is species abundance counts.
#[must_use]
pub fn shannon(counts: &[f64]) -> f64 {
    barracuda::stats::shannon(counts)
}

/// Simpson diversity index 1 − Σ pᵢ².
///
/// Ranges from 0 (one dominant species) to ~1 (many evenly distributed species).
#[must_use]
pub fn simpson(counts: &[f64]) -> f64 {
    barracuda::stats::simpson(counts)
}

/// Chao1 richness estimator — estimates true species count from observed sample.
///
/// Accounts for unobserved rare species using singleton/doubleton ratio.
#[must_use]
pub fn chao1(counts: &[f64]) -> f64 {
    barracuda::stats::chao1(counts)
}

/// Pielou's evenness J' = H' / ln(S).
///
/// Measures how evenly individuals are distributed among species.
/// 1.0 = perfectly even, 0.0 = completely dominated by one species.
#[must_use]
pub fn pielou_evenness(counts: &[f64]) -> f64 {
    barracuda::stats::pielou_evenness(counts)
}

/// Observed species count (number of non-zero entries).
#[must_use]
pub fn observed_species(counts: &[f64]) -> f64 {
    barracuda::stats::observed_features(counts)
}

/// Compute all alpha diversity metrics in one pass.
#[must_use]
pub fn alpha_diversity(counts: &[f64]) -> AlphaDiversity {
    barracuda::stats::alpha_diversity(counts)
}

/// Bray-Curtis dissimilarity between two community samples.
///
/// BC = Σ|aᵢ − bᵢ| / Σ(aᵢ + bᵢ). Ranges from 0 (identical) to 1 (no shared
/// species). Widely used in agroecology for comparing field-margin vs interior
/// communities.
#[must_use]
pub fn bray_curtis(a: &[f64], b: &[f64]) -> f64 {
    barracuda::stats::bray_curtis(a, b)
}

/// Shannon entropy from pre-computed relative frequencies.
///
/// Use when abundances are already normalised to proportions (sum to 1.0).
/// Avoids re-normalisation overhead in pipelines where relative frequencies
/// are maintained (e.g., streaming 16S amplicon processing).
///
/// # Cross-Spring Provenance
///
/// Absorbed from wetSpring biodiversity pipeline (S66 R-S66-037).
#[must_use]
pub fn shannon_from_frequencies(freqs: &[f64]) -> f64 {
    barracuda::stats::shannon_from_frequencies(freqs)
}

/// Bray-Curtis pairwise distance matrix (condensed upper triangle).
///
/// For M samples, returns M*(M-1)/2 pairwise dissimilarities.
#[must_use]
pub fn bray_curtis_condensed(samples: &[Vec<f64>]) -> Vec<f64> {
    barracuda::stats::bray_curtis_condensed(samples)
}

/// Bray-Curtis full distance matrix (M×M).
///
/// Returns a flat M×M matrix where `matrix[i*m + j]` is the Bray-Curtis
/// dissimilarity between samples `i` and `j`. Diagonal entries are 0.0.
/// Useful for ordination (`PCoA`, `NMDS`) and cluster analysis in soil
/// microbiome community comparisons.
///
/// # Cross-Spring Provenance
///
/// Absorbed from wetSpring biodiversity pipeline (S64). Uses the same
/// kernel as `bray_curtis_condensed` but returns the full symmetric matrix.
#[must_use]
pub fn bray_curtis_matrix(samples: &[Vec<f64>]) -> Vec<f64> {
    barracuda::stats::bray_curtis_matrix(samples)
}

/// Rarefaction curve: expected species count at each subsampling depth.
///
/// Used in agroecology to compare species richness across sites with
/// unequal sampling effort.
#[must_use]
pub fn rarefaction_curve(counts: &[f64], depths: &[f64]) -> Vec<f64> {
    barracuda::stats::rarefaction_curve(counts, depths)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn cover_crop_mix() -> Vec<f64> {
        // 5-species cover crop mix: crimson clover, cereal rye, radish, vetch, oats
        vec![120.0, 85.0, 45.0, 30.0, 20.0]
    }

    fn monoculture() -> Vec<f64> {
        vec![300.0, 0.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn shannon_diversity_mix_gt_monoculture() {
        let mix_h = shannon(&cover_crop_mix());
        let mono_h = shannon(&monoculture());
        assert!(
            mix_h > mono_h,
            "Mix H'={mix_h} should exceed monoculture H'={mono_h}"
        );
        assert!(mix_h > 1.0, "5-species mix should have H' > 1.0");
        assert!(mono_h.abs() < 1e-12, "Monoculture H' should be 0.0");
    }

    #[test]
    fn simpson_diversity_mix_gt_monoculture() {
        let mix_d = simpson(&cover_crop_mix());
        let mono_d = simpson(&monoculture());
        assert!(mix_d > mono_d, "Mix D={mix_d} > mono D={mono_d}");
        assert!(mix_d > 0.5, "5-species mix should have D > 0.5");
    }

    #[test]
    fn chao1_at_least_observed() {
        let counts = cover_crop_mix();
        let c = chao1(&counts);
        let obs = observed_species(&counts);
        assert!(c >= obs, "Chao1={c} >= observed={obs}");
    }

    #[test]
    fn pielou_evenness_bounds() {
        let j = pielou_evenness(&cover_crop_mix());
        assert!((0.0..=1.0).contains(&j), "J'={j} should be in [0,1]");
    }

    #[test]
    fn alpha_diversity_all_at_once() {
        let ad = alpha_diversity(&cover_crop_mix());
        assert!((ad.observed - 5.0).abs() < 1e-10);
        assert!(ad.shannon > 1.0);
        assert!(ad.simpson > 0.5);
        assert!(ad.chao1 >= 5.0);
        assert!((0.0..=1.0).contains(&ad.evenness));
    }

    #[test]
    fn bray_curtis_identical_is_zero() {
        let a = cover_crop_mix();
        let bc = bray_curtis(&a, &a);
        assert!(bc.abs() < 1e-12, "BC={bc}");
    }

    #[test]
    fn bray_curtis_different_communities() {
        let a = cover_crop_mix();
        let b = monoculture();
        let bc = bray_curtis(&a, &b);
        assert!(bc > 0.0 && bc <= 1.0, "BC={bc} should be in (0,1]");
    }

    #[test]
    fn bray_curtis_condensed_correct_size() {
        let samples = vec![
            cover_crop_mix(),
            monoculture(),
            vec![50.0, 50.0, 50.0, 50.0, 50.0],
        ];
        let dists = bray_curtis_condensed(&samples);
        assert_eq!(dists.len(), 3, "3 samples → 3 pairwise distances");
    }

    #[test]
    fn shannon_from_frequencies_matches_counts() {
        let counts = cover_crop_mix();
        let total: f64 = counts.iter().sum();
        let freqs: Vec<f64> = counts.iter().map(|&c| c / total).collect();
        let h_counts = shannon(&counts);
        let h_freqs = shannon_from_frequencies(&freqs);
        assert!(
            (h_counts - h_freqs).abs() < 1e-10,
            "H' from counts ({h_counts}) should match H' from frequencies ({h_freqs})"
        );
    }

    #[test]
    fn bray_curtis_matrix_correct_size() {
        let samples = vec![
            cover_crop_mix(),
            monoculture(),
            vec![50.0, 50.0, 50.0, 50.0, 50.0],
        ];
        let m = 3;
        let mat = bray_curtis_matrix(&samples);
        assert_eq!(mat.len(), m * m, "3×3 matrix");
        for i in 0..m {
            assert!(mat[i * m + i].abs() < 1e-12, "Diagonal should be 0");
        }
        assert!(
            (mat[1] - mat[m]).abs() < 1e-12,
            "Symmetric: mat[0,1] == mat[1,0]"
        );
    }

    #[test]
    fn rarefaction_monotonic() {
        let counts = cover_crop_mix();
        let depths: Vec<f64> = (1..=10).map(|d| f64::from(d) * 30.0).collect();
        let curve = rarefaction_curve(&counts, &depths);
        assert_eq!(curve.len(), depths.len());
        for window in curve.windows(2) {
            assert!(
                window[1] >= window[0] - 1e-10,
                "Rarefaction should be monotonically non-decreasing"
            );
        }
    }
}
