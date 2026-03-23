// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring shader provenance for airSpring GPU modules.
//!
//! Documents the lineage of each shader primitive used by airSpring,
//! tracking which Spring originally developed it and how it was evolved.
//! The upstream `barraCuda` provenance registry is queried at runtime
//! via [`upstream_airspring_provenance`].

/// Cross-spring shader provenance record.
#[derive(Debug, Clone)]
pub struct ShaderProvenance {
    /// WGSL shader filename.
    pub shader: &'static str,
    /// Key primitives from this shader.
    pub primitives: &'static [&'static str],
    /// Spring that originally developed it.
    pub origin: &'static str,
    /// Scientific domain.
    pub domain: &'static str,
    /// Which Springs evolved it further.
    pub evolved_by: &'static [&'static str],
    /// How the local primal uses this shader.
    pub domain_use: &'static str,
}

/// Query the upstream `barraCuda` provenance registry for shaders consumed by airSpring.
///
/// Returns records from `barracuda::shaders::provenance` where airSpring is a consumer,
/// bridging the upstream registry with airSpring's local provenance table.
#[must_use]
pub fn upstream_airspring_provenance() -> Vec<&'static barracuda::shaders::provenance::ShaderRecord>
{
    barracuda::shaders::provenance::shaders_consumed_by(
        barracuda::shaders::provenance::SpringDomain::AIR_SPRING,
    )
}

/// Generate the upstream cross-spring evolution report (markdown).
#[must_use]
pub fn upstream_evolution_report() -> String {
    barracuda::shaders::provenance::evolution_report()
}

/// Query the cross-spring dependency matrix from the upstream registry.
#[must_use]
pub fn upstream_cross_spring_matrix() -> std::collections::HashMap<
    (
        barracuda::shaders::provenance::SpringDomain,
        barracuda::shaders::provenance::SpringDomain,
    ),
    usize,
> {
    barracuda::shaders::provenance::cross_spring_matrix()
}

/// Cross-spring shader provenance for airSpring GPU modules.
pub const PROVENANCE: &[ShaderProvenance] = &[
    ShaderProvenance {
        shader: "math_f64.wgsl",
        primitives: &[
            "pow_f64", "exp_f64", "log_f64", "sin_f64", "cos_f64", "acos_f64",
        ],
        origin: "hotSpring",
        domain: "Lattice QCD f64 precision",
        evolved_by: &["airSpring (TS-001 pow_f64 fix, TS-003 acos precision)"],
        domain_use: "Solar declination, atmospheric pressure, VG retention curves",
    },
    ShaderProvenance {
        shader: "df64_core.wgsl",
        primitives: &["df64_add", "df64_mul", "df64_div", "df64_neg", "df64_abs"],
        origin: "hotSpring",
        domain: "Nuclear EOS double-float arithmetic",
        evolved_by: &["hotSpring S60 (FMA optimization)"],
        domain_use: "Consumer GPU precision (RTX 4070: Df64 ~48-bit for ET₀)",
    },
    ShaderProvenance {
        shader: "df64_transcendentals.wgsl",
        primitives: &["df64_exp", "df64_log", "df64_sqrt", "df64_sin", "df64_cos"],
        origin: "hotSpring",
        domain: "FMA-optimized transcendentals for DF64",
        evolved_by: &["hotSpring S60"],
        domain_use: "Df64 precision path for ET₀ on consumer GPUs",
    },
    ShaderProvenance {
        shader: "batched_elementwise_f64.wgsl",
        primitives: &[
            "fao56_et0_batch (op=0)",
            "water_balance_batch (op=1)",
            "sensor_cal (op=5)",
            "hargreaves_et0 (op=6)",
            "kc_climate_adjust (op=7)",
            "dual_kc_ke (op=8)",
            "vg_theta (op=9)",
            "vg_k (op=10)",
            "thornthwaite_et0 (op=11)",
            "gdd (op=12)",
            "pedotransfer_poly (op=13)",
            "makkink_et0 (op=14)",
            "turc_et0 (op=15)",
            "hamon_et0 (op=16)",
            "scs_cn_runoff (op=17)",
            "stewart_yield_water (op=18)",
            "blaney_criddle_et0 (op=19)",
        ],
        origin: "multi-spring convergence",
        domain: "Precision agriculture: FAO-56 ET₀, WB, VG, Thornthwaite, GDD, pedotransfer, simple ET₀, runoff, yield",
        evolved_by: &[
            "airSpring (domain equations, ops 0-1, 5-8 → v0.5.6)",
            "hotSpring S54 (acos_f64, sin_f64 for Ra/sunset angle)",
            "neuralSpring (batch orchestrator pattern)",
            "BarraCuda S54→S70+ (ops 0-8 unified absorption)",
            "BarraCuda S79 (ops 9-13: VG, Thornthwaite, GDD, pedotransfer)",
            "airSpring v0.7.2 → BarraCuda (ops 14-19: Makkink, Turc, Hamon, SCS-CN, Stewart, Blaney-Criddle)",
        ],
        domain_use: "GPU-first dispatch: 20 ops covering all soil physics, crop science, and hydrology",
    },
    ShaderProvenance {
        shader: "kriging_f64.wgsl",
        primitives: &["ordinary_kriging", "variogram_fit"],
        origin: "wetSpring",
        domain: "Geostatistical spatial interpolation",
        evolved_by: &["wetSpring S28+"],
        domain_use: "Soil moisture spatial interpolation from sensor networks",
    },
    ShaderProvenance {
        shader: "fused_map_reduce_f64.wgsl",
        primitives: &["sum", "max", "min", "shannon_entropy", "simpson_index"],
        origin: "wetSpring",
        domain: "Biodiversity and ecological statistics",
        evolved_by: &[
            "wetSpring (Shannon/Simpson)",
            "airSpring (TS-004 buffer fix)",
        ],
        domain_use: "Seasonal ET₀ aggregation, diversity metrics",
    },
    ShaderProvenance {
        shader: "moving_window_stats.wgsl",
        primitives: &["moving_mean", "moving_std"],
        origin: "wetSpring",
        domain: "Time series IoT stream smoothing",
        evolved_by: &["wetSpring S28+", "airSpring metalForge S66 (f64 path)"],
        domain_use: "IoT sensor stream smoothing for SoilWatch data",
    },
    ShaderProvenance {
        shader: "nelder_mead.wgsl",
        primitives: &["nelder_mead", "multi_start_nelder_mead"],
        origin: "neuralSpring",
        domain: "Derivative-free optimization",
        evolved_by: &["neuralSpring S52+"],
        domain_use: "Isotherm fitting (Langmuir qm/KL, Freundlich Kf/n)",
    },
    ShaderProvenance {
        shader: "crank_nicolson_f64.wgsl",
        primitives: &["cn_step", "cyclic_reduction_f64"],
        origin: "hotSpring",
        domain: "Implicit PDE time-stepping (heat, Schrödinger)",
        evolved_by: &["hotSpring S61-63 (sovereign compiler, f64 evolution)"],
        domain_use: "Richards PDE linearised diffusion cross-validation",
    },
    ShaderProvenance {
        shader: "norm_ppf.wgsl (Moro 1995)",
        primitives: &["norm_ppf", "norm_cdf"],
        origin: "hotSpring",
        domain: "Special functions (inverse normal CDF)",
        evolved_by: &["hotSpring special-function library → barracuda S52+"],
        domain_use: "MC ET₀ parametric confidence intervals",
    },
    ShaderProvenance {
        shader: "hydrology (CPU→GPU kernel)",
        primitives: &[
            "hargreaves_et0_batch",
            "crop_coefficient",
            "soil_water_balance",
        ],
        origin: "airSpring",
        domain: "FAO-56 hydrology batch primitives (GPU-first since v0.5.6)",
        evolved_by: &[
            "airSpring metalForge → BarraCuda S66 (absorption)",
            "airSpring v0.5.6 (GPU-first rewire via ops 5-8)",
        ],
        domain_use: "Hargreaves GPU ET₀, Kc GPU adjustment, DualKc GPU Ke",
    },
    ShaderProvenance {
        shader: "stats_f64 (GPU statistics)",
        primitives: &["linear_regression", "matrix_correlation"],
        origin: "neuralSpring S69",
        domain: "GPU-accelerated OLS regression and correlation matrices",
        evolved_by: &["neuralSpring S69 → BarraCuda absorption"],
        domain_use: "Sensor calibration regression, multi-variate soil analysis",
    },
    ShaderProvenance {
        shader: "seasonal_pipeline.wgsl (fused)",
        primitives: &["fused_et0_kc_wb_stress"],
        origin: "airSpring concept → BarraCuda S70+",
        domain: "Single-dispatch seasonal pipeline: ET₀ → Kc → WB → Stress",
        evolved_by: &[
            "airSpring (domain spec)",
            "BarraCuda S70+ (WGSL implementation)",
        ],
        domain_use: "Future: fused seasonal pipeline (pending Rust executor)",
    },
    ShaderProvenance {
        shader: "brent_f64.wgsl (root-finding)",
        primitives: &["brent_vg_inverse", "brent_green_ampt"],
        origin: "airSpring concept → BarraCuda S70+",
        domain: "Brent method root-finding for VG inverse and Green-Ampt",
        evolved_by: &[
            "airSpring (VG inverse need)",
            "BarraCuda S70+ (WGSL, bug on L49)",
        ],
        domain_use: "Future: GPU VG inverse (pending barraCuda shader bug fix on L49)",
    },
    ShaderProvenance {
        shader: "diversity (CPU bio kernel)",
        primitives: &[
            "shannon",
            "simpson",
            "chao1",
            "bray_curtis",
            "bray_curtis_matrix",
            "shannon_from_frequencies",
        ],
        origin: "wetSpring",
        domain: "Microbiome alpha/beta diversity",
        evolved_by: &[
            "wetSpring S28 (bio/diversity)",
            "BarraCuda S64 (absorption)",
            "airSpring (agroecology wrappers)",
        ],
        domain_use: "Cover crop biodiversity, soil 16S microbiome, pollinator habitat",
    },
    ShaderProvenance {
        shader: "anderson (CPU coupling kernel)",
        primitives: &["coupling_chain", "coupling_series", "classify_regime"],
        origin: "groundSpring",
        domain: "Anderson localisation → soil moisture coupling",
        evolved_by: &[
            "groundSpring (physics model)",
            "airSpring Exp-048 (θ→QS regime for 16S)",
        ],
        domain_use: "Soil moisture regime classification, NCBI 16S coupling",
    },
    ShaderProvenance {
        shader: "blaney_criddle (CPU ET₀ kernel)",
        primitives: &[
            "blaney_criddle_et0",
            "blaney_criddle_p",
            "blaney_criddle_from_location",
        ],
        origin: "airSpring",
        domain: "Temperature-daylight PET (8th ET₀ method)",
        evolved_by: &["airSpring Exp-049 (USDA-SCS 1950)"],
        domain_use: "Blaney-Criddle PET for data-sparse regions",
    },
    ShaderProvenance {
        shader: "scs_cn (CPU runoff kernel)",
        primitives: &[
            "scs_cn_runoff",
            "potential_retention",
            "amc_cn_dry",
            "amc_cn_wet",
        ],
        origin: "airSpring",
        domain: "SCS Curve Number rainfall-runoff",
        evolved_by: &["airSpring Exp-050 (USDA-SCS TR-55)"],
        domain_use: "Runoff estimation for water balance, CN tables, AMC adjustment",
    },
    ShaderProvenance {
        shader: "green_ampt (CPU infiltration kernel)",
        primitives: &[
            "cumulative_infiltration",
            "infiltration_rate",
            "ponding_time",
        ],
        origin: "airSpring",
        domain: "Green-Ampt (1911) soil infiltration physics",
        evolved_by: &["airSpring Exp-051 (Rawls 1983 parameters)"],
        domain_use: "Infiltration modeling, ponding prediction, 7-soil parameter table",
    },
    ShaderProvenance {
        shader: "jackknife_mean_f64.wgsl",
        primitives: &["jackknife_leave_one_out", "jackknife_variance"],
        origin: "groundSpring",
        domain: "Leave-one-out uncertainty estimation",
        evolved_by: &[
            "groundSpring (methodology)",
            "neuralSpring (GPU dispatch pattern)",
            "BarraCuda S71 (WGSL shader)",
        ],
        domain_use: "ET₀ and yield estimate uncertainty quantification",
    },
    ShaderProvenance {
        shader: "bootstrap_mean_f64.wgsl",
        primitives: &["bootstrap_resample", "bootstrap_mean"],
        origin: "groundSpring",
        domain: "Non-parametric bootstrap confidence intervals",
        evolved_by: &[
            "groundSpring (bootstrap methodology)",
            "neuralSpring (GPU dispatch)",
            "BarraCuda S71 (xoshiro PRNG + WGSL shader)",
        ],
        domain_use: "RMSE confidence intervals, yield prediction uncertainty",
    },
    ShaderProvenance {
        shader: "diversity_fusion_f64.wgsl",
        primitives: &["shannon_gpu", "simpson_gpu", "pielou_evenness_gpu"],
        origin: "wetSpring",
        domain: "GPU-fused alpha diversity (Shannon + Simpson + evenness in one dispatch)",
        evolved_by: &[
            "wetSpring S28 (CPU diversity indices)",
            "BarraCuda S70 (GPU fusion shader)",
            "airSpring (agroecology: cover crop, soil 16S, pollinator)",
        ],
        domain_use: "Multi-sample diversity profiling for soil microbiome studies",
    },
    ShaderProvenance {
        shader: "hargreaves_batch_f64.wgsl (science shader)",
        primitives: &["hargreaves_et0_gpu", "extraterrestrial_radiation"],
        origin: "airSpring",
        domain: "Hargreaves-Samani ET₀ with internal Ra computation",
        evolved_by: &[
            "airSpring (Hargreaves domain need)",
            "BarraCuda S71 (HargreavesBatchGpu — science shader path)",
        ],
        domain_use: "Alternative to op=6 when Ra is not precomputed",
    },
    ShaderProvenance {
        shader: "mc_et0_propagate_f64.wgsl",
        primitives: &["mc_et0_propagate", "box_muller", "xoshiro128"],
        origin: "groundSpring",
        domain: "Monte Carlo ET₀ uncertainty propagation",
        evolved_by: &[
            "groundSpring (MC methodology, xoshiro PRNG)",
            "airSpring (FAO-56 domain equations)",
            "BarraCuda S66+ (WGSL shader, Box-Muller transform)",
        ],
        domain_use: "GPU Monte Carlo ET₀ uncertainty bands (N=10K+ samples)",
    },
    ShaderProvenance {
        shader: "mean_variance_f64.wgsl (fused Welford)",
        primitives: &["mean_variance", "sample_variance", "std_dev"],
        origin: "hotSpring",
        domain: "Single-pass Welford mean+variance (numerically stable)",
        evolved_by: &[
            "hotSpring S58 (lattice QCD observable statistics)",
            "neuralSpring (ML loss/gradient variance tracking)",
            "groundSpring (sensor noise quantification)",
            "barraCuda 0.3.3 (DF64 variant: mean_variance_df64.wgsl)",
        ],
        domain_use: "SeasonalReducer fused stats (3 GPU passes vs previous 4), sensor QA",
    },
    ShaderProvenance {
        shader: "correlation_full_f64.wgsl (5-accumulator Pearson)",
        primitives: &["correlation_full", "pearson_r", "mean_x", "var_x"],
        origin: "neuralSpring S69",
        domain: "Fused Pearson correlation (mean, variance, r in one pass)",
        evolved_by: &[
            "neuralSpring S69 (Kokkos parallel_reduce pattern)",
            "hotSpring (DF64 variant: correlation_full_df64.wgsl, S93)",
            "barraCuda 0.3.3 (Fp64Strategy-aware routing)",
        ],
        domain_use: "Pairwise sensor cross-correlation (VWC↔EC, temp↔ET₀)",
    },
    ShaderProvenance {
        shader: "bingocube-nautilus (Rust reservoir, no WGSL)",
        primitives: &[
            "NautilusBrain::train",
            "NautilusBrain::is_drifting",
            "EdgeSeeder::seed_boards",
        ],
        origin: "hotSpring (v0.6.15)",
        domain: "Evolutionary reservoir computing via BingoCube board populations",
        evolved_by: &[
            "hotSpring (NautilusBrain for QCD — β scan cost reduction)",
            "primalTools/bingoCube (domain-agnostic extraction)",
            "airSpring v0.6.2 (AirSpringBrain: ET₀/soil/crop heads, MonitoredAtlasStream)",
        ],
        domain_use: "Agricultural regime prediction, drift detection in 80yr atlas, cross-station transfer",
    },
    ShaderProvenance {
        shader: "tissue-cytokine (CPU immunological kernel)",
        primitives: &[
            "tissue::analyze_tissue_disorder",
            "tissue::barrier_disruption_d_eff",
            "cytokine::CytokineBrain::train",
        ],
        origin: "airSpring (Paper 12)",
        domain: "Immunological Anderson localization — cytokine propagation through skin tissue",
        evolved_by: &[
            "Paper 01 (Anderson QS — W_c thresholds, level spacing ratio)",
            "Paper 06 (no-till dimensional collapse ↔ Paper 12 dimensional promotion duality)",
            "bingocube-nautilus (CytokineBrain inherits AirSpringBrain pattern)",
            "wetSpring (GpuDiversity → Pielou evenness as disorder W)",
        ],
        domain_use: "AD flare prediction via CytokineBrain, tissue disorder profiling, Gonzales data pipeline",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_non_empty() {
        assert!(
            !PROVENANCE.is_empty(),
            "Cross-spring provenance should have entries"
        );
        for p in PROVENANCE {
            assert!(!p.shader.is_empty());
            assert!(!p.primitives.is_empty());
            assert!(!p.origin.is_empty());
        }
    }

    #[test]
    fn test_provenance_covers_all_gpu_modules() {
        let shaders: Vec<&str> = PROVENANCE.iter().map(|p| p.shader).collect();
        assert!(shaders.contains(&"batched_elementwise_f64.wgsl"));
        assert!(shaders.contains(&"kriging_f64.wgsl"));
        assert!(shaders.contains(&"fused_map_reduce_f64.wgsl"));
        assert!(shaders.contains(&"moving_window_stats.wgsl"));
        assert!(shaders.contains(&"nelder_mead.wgsl"));
        assert!(shaders.contains(&"math_f64.wgsl"));
        assert!(shaders.contains(&"df64_core.wgsl"));
        assert!(shaders.contains(&"crank_nicolson_f64.wgsl"));
        assert!(shaders.contains(&"hydrology (CPU→GPU kernel)"));
        assert!(shaders.contains(&"stats_f64 (GPU statistics)"));
        assert!(shaders.contains(&"seasonal_pipeline.wgsl (fused)"));
        assert!(shaders.contains(&"brent_f64.wgsl (root-finding)"));
        assert!(shaders.contains(&"diversity (CPU bio kernel)"));
        assert!(shaders.contains(&"anderson (CPU coupling kernel)"));
        assert!(shaders.contains(&"blaney_criddle (CPU ET₀ kernel)"));
        assert!(shaders.contains(&"scs_cn (CPU runoff kernel)"));
        assert!(shaders.contains(&"green_ampt (CPU infiltration kernel)"));
        assert!(shaders.contains(&"jackknife_mean_f64.wgsl"));
        assert!(shaders.contains(&"bootstrap_mean_f64.wgsl"));
        assert!(shaders.contains(&"diversity_fusion_f64.wgsl"));
        assert!(shaders.contains(&"hargreaves_batch_f64.wgsl (science shader)"));
        assert!(shaders.contains(&"mc_et0_propagate_f64.wgsl"));
        assert!(shaders.contains(&"bingocube-nautilus (Rust reservoir, no WGSL)"));
        assert!(shaders.contains(&"tissue-cytokine (CPU immunological kernel)"));
    }

    #[test]
    fn test_provenance_origins_multi_spring() {
        let origins: Vec<&str> = PROVENANCE.iter().map(|p| p.origin).collect();
        assert!(origins.contains(&"hotSpring"));
        assert!(origins.contains(&"wetSpring"));
        assert!(origins.contains(&"neuralSpring"));
        assert!(origins.contains(&"multi-spring convergence"));
        assert!(origins.contains(&"airSpring"));
        assert!(origins.contains(&"groundSpring"));
    }

    #[test]
    fn test_upstream_provenance_registry() {
        let records = upstream_airspring_provenance();
        assert!(
            !records.is_empty(),
            "airSpring should consume upstream shaders"
        );
        for r in &records {
            assert!(
                r.consumers
                    .contains(&barracuda::shaders::provenance::SpringDomain::AIR_SPRING),
                "shader {} should list airSpring as consumer",
                r.path
            );
        }
    }

    #[test]
    fn test_upstream_evolution_report() {
        let report = upstream_evolution_report();
        assert!(report.contains("Timeline"));
        assert!(report.contains("Dependency Matrix"));
        assert!(report.contains("airSpring"));
    }

    #[test]
    fn test_upstream_cross_spring_matrix() {
        let matrix = upstream_cross_spring_matrix();
        assert!(
            !matrix.is_empty(),
            "cross-spring matrix should be non-empty"
        );
    }
}
