// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device information and precision strategy reporting.
//!
//! Wraps `barracuda::device` to provide precision-aware device creation
//! and capability reporting for airSpring GPU modules.
//!
//! # Cross-Spring Shader Provenance
//!
//! The precision architecture was evolved across all Springs:
//! - **hotSpring**: `df64_core.wgsl`, `df64_transcendentals.wgsl`, `math_f64.wgsl` — FMA-optimized
//!   double-float arithmetic for lattice QCD and nuclear EOS.
//! - **wetSpring**: `shannon_f64.wgsl`, `kriging_f64.wgsl` — f64 diversity and spatial interpolation.
//! - **neuralSpring**: `NelderMeadGpu`, `ValidationHarness` — optimization and quality infrastructure.
//! - **airSpring**: Richards PDE, regression, hydrology — domain contributions absorbed S40+S66.
//! - **groundSpring**: MC propagation, `batched_multinomial` — uncertainty quantification.
//!
//! `Fp64Strategy` was introduced in S58, evolving from hotSpring's need for
//! native f64 in lattice QCD (where `Df64` accumulated unacceptable phase errors
//! in chained SU(3) matrix multiplications). airSpring benefits: the Titan V
//! runs our ET₀ shaders in native f64, while the RTX 4070 can use `Df64` with
//! ~48-bit mantissa precision — still adequate for FAO-56 (which only needs ~6 digits).

use std::sync::Arc;

use barracuda::device::probe::F64BuiltinCapabilities;
use barracuda::device::{Fp64Rate, Fp64Strategy, GpuDriverProfile, WgpuDevice};

/// Precision report for a GPU device.
#[derive(Debug, Clone)]
pub struct DevicePrecisionReport {
    /// Device adapter name.
    pub adapter_name: String,
    /// f64 throughput strategy.
    pub fp64_strategy: Fp64Strategy,
    /// Raw f64 rate classification.
    pub fp64_rate: Fp64Rate,
    /// f64 builtin capabilities.
    pub builtins: F64BuiltinCapabilities,
    /// Whether the device has native f64 shader support.
    pub has_f64_shaders: bool,
    /// Whether the device supports SPIR-V passthrough.
    pub has_spirv_passthrough: bool,
}

impl std::fmt::Display for DevicePrecisionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GPU: {}", self.adapter_name)?;
        writeln!(f, "  Fp64Strategy: {:?}", self.fp64_strategy)?;
        writeln!(f, "  Fp64Rate:     {:?}", self.fp64_rate)?;
        writeln!(f, "  f64 shaders:  {}", self.has_f64_shaders)?;
        writeln!(f, "  SPIR-V pass:  {}", self.has_spirv_passthrough)?;
        write!(
            f,
            "  Builtins:     exp={} log={} sin={} cos={} sqrt={} fma={}",
            self.builtins.exp,
            self.builtins.log,
            self.builtins.sin,
            self.builtins.cos,
            self.builtins.sqrt,
            self.builtins.fma
        )
    }
}

/// Probe a device and produce a precision report.
///
/// Combines `GpuDriverProfile` (for `Fp64Strategy`) with `probe_f64_builtins`
/// (for native f64 builtin availability).
#[must_use]
pub fn probe_device(device: &WgpuDevice) -> DevicePrecisionReport {
    let profile = GpuDriverProfile::from_device(device);
    let builtins = pollster::block_on(barracuda::device::probe::probe_f64_builtins(device));

    DevicePrecisionReport {
        adapter_name: device.adapter_info().name.clone(),
        fp64_strategy: profile.fp64_strategy(),
        fp64_rate: profile.fp64_rate,
        builtins,
        has_f64_shaders: device.has_f64_shaders(),
        has_spirv_passthrough: device.has_spirv_passthrough(),
    }
}

/// Try to create a GPU device respecting `BARRACUDA_GPU_ADAPTER` selection.
///
/// Uses `WgpuDevice::from_env()` which reads the environment variable:
/// - `BARRACUDA_GPU_ADAPTER=titan` → selects adapter containing "titan"
/// - `BARRACUDA_GPU_ADAPTER=0` → selects first adapter
/// - `BARRACUDA_GPU_ADAPTER=auto` or unset → wgpu `HighPerformance` default
///
/// Falls back to `new_f64_capable()` if `from_env()` fails.
///
/// # Errors
///
/// Returns `None` if no suitable GPU is available.
pub fn try_f64_device() -> Option<Arc<WgpuDevice>> {
    pollster::block_on(WgpuDevice::from_env())
        .or_else(|_| pollster::block_on(WgpuDevice::new_f64_capable()))
        .ok()
        .map(Arc::new)
}

/// Cross-spring shader provenance for airSpring GPU modules.
///
/// Documents the lineage of each shader primitive used by airSpring,
/// tracking which Spring originally developed it and how it was evolved.
pub const PROVENANCE: &[ShaderProvenance] = &[
    ShaderProvenance {
        shader: "math_f64.wgsl",
        primitives: &[
            "pow_f64", "exp_f64", "log_f64", "sin_f64", "cos_f64", "acos_f64",
        ],
        origin: "hotSpring",
        domain: "Lattice QCD f64 precision",
        evolved_by: &["airSpring (TS-001 pow_f64 fix, TS-003 acos precision)"],
        airspring_use: "Solar declination, atmospheric pressure, VG retention curves",
    },
    ShaderProvenance {
        shader: "df64_core.wgsl",
        primitives: &["df64_add", "df64_mul", "df64_div", "df64_neg", "df64_abs"],
        origin: "hotSpring",
        domain: "Nuclear EOS double-float arithmetic",
        evolved_by: &["hotSpring S60 (FMA optimization)"],
        airspring_use: "Consumer GPU precision (RTX 4070: Df64 ~48-bit for ET₀)",
    },
    ShaderProvenance {
        shader: "df64_transcendentals.wgsl",
        primitives: &["df64_exp", "df64_log", "df64_sqrt", "df64_sin", "df64_cos"],
        origin: "hotSpring",
        domain: "FMA-optimized transcendentals for DF64",
        evolved_by: &["hotSpring S60"],
        airspring_use: "Df64 precision path for ET₀ on consumer GPUs",
    },
    ShaderProvenance {
        shader: "batched_elementwise_f64.wgsl",
        primitives: &["fao56_et0_batch", "water_balance_batch"],
        origin: "airSpring + ToadStool",
        domain: "Precision agriculture: FAO-56 ET₀ + water balance",
        evolved_by: &[
            "airSpring (domain equations)",
            "ToadStool S54 (orchestrator + precision fixes)",
        ],
        airspring_use: "Primary GPU dispatch for ET₀ and water balance",
    },
    ShaderProvenance {
        shader: "kriging_f64.wgsl",
        primitives: &["ordinary_kriging", "variogram_fit"],
        origin: "wetSpring",
        domain: "Geostatistical spatial interpolation",
        evolved_by: &["wetSpring S28+"],
        airspring_use: "Soil moisture spatial interpolation from sensor networks",
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
        airspring_use: "Seasonal ET₀ aggregation, diversity metrics",
    },
    ShaderProvenance {
        shader: "moving_window_stats.wgsl",
        primitives: &["moving_mean", "moving_std"],
        origin: "wetSpring",
        domain: "Time series IoT stream smoothing",
        evolved_by: &["wetSpring S28+", "airSpring metalForge S66 (f64 path)"],
        airspring_use: "IoT sensor stream smoothing for SoilWatch data",
    },
    ShaderProvenance {
        shader: "nelder_mead.wgsl",
        primitives: &["nelder_mead", "multi_start_nelder_mead"],
        origin: "neuralSpring",
        domain: "Derivative-free optimization",
        evolved_by: &["neuralSpring S52+"],
        airspring_use: "Isotherm fitting (Langmuir qm/KL, Freundlich Kf/n)",
    },
    ShaderProvenance {
        shader: "crank_nicolson_f64.wgsl",
        primitives: &["cn_step", "cyclic_reduction_f64"],
        origin: "hotSpring",
        domain: "Implicit PDE time-stepping (heat, Schrödinger)",
        evolved_by: &["hotSpring S61-63 (sovereign compiler, f64 evolution)"],
        airspring_use: "Richards PDE linearised diffusion cross-validation",
    },
    ShaderProvenance {
        shader: "norm_ppf.wgsl (Moro 1995)",
        primitives: &["norm_ppf", "norm_cdf"],
        origin: "hotSpring",
        domain: "Special functions (inverse normal CDF)",
        evolved_by: &["hotSpring special-function library → barracuda S52+"],
        airspring_use: "MC ET₀ parametric confidence intervals",
    },
    ShaderProvenance {
        shader: "hydrology (CPU batch kernel)",
        primitives: &[
            "hargreaves_et0_batch",
            "crop_coefficient",
            "soil_water_balance",
        ],
        origin: "airSpring",
        domain: "FAO-56 hydrology batch primitives",
        evolved_by: &["airSpring metalForge → ToadStool S66 (absorption)"],
        airspring_use: "Hargreaves batch ET₀, Kc stage interpolation",
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
            "ToadStool S64 (absorption)",
            "airSpring (agroecology wrappers)",
        ],
        airspring_use: "Cover crop biodiversity, soil 16S microbiome, pollinator habitat",
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
        airspring_use: "Soil moisture regime classification, NCBI 16S coupling",
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
        airspring_use: "Blaney-Criddle PET for data-sparse regions",
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
        airspring_use: "Runoff estimation for water balance, CN tables, AMC adjustment",
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
        airspring_use: "Infiltration modeling, ponding prediction, 7-soil parameter table",
    },
];

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
    /// How airSpring uses it.
    pub airspring_use: &'static str,
}

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
        assert!(shaders.contains(&"hydrology (CPU batch kernel)"));
        assert!(shaders.contains(&"diversity (CPU bio kernel)"));
        assert!(shaders.contains(&"anderson (CPU coupling kernel)"));
        assert!(shaders.contains(&"blaney_criddle (CPU ET₀ kernel)"));
        assert!(shaders.contains(&"scs_cn (CPU runoff kernel)"));
        assert!(shaders.contains(&"green_ampt (CPU infiltration kernel)"));
    }

    #[test]
    fn test_provenance_origins_multi_spring() {
        let origins: Vec<&str> = PROVENANCE.iter().map(|p| p.origin).collect();
        assert!(origins.contains(&"hotSpring"));
        assert!(origins.contains(&"wetSpring"));
        assert!(origins.contains(&"neuralSpring"));
        assert!(origins.contains(&"airSpring + ToadStool"));
        assert!(origins.contains(&"airSpring"));
        assert!(origins.contains(&"groundSpring"));
    }

    #[test]
    fn test_try_f64_device() {
        // May return None in CI — just ensure no panic
        let _ = try_f64_device();
    }

    #[test]
    fn test_probe_device_if_available() {
        let Some(device) = try_f64_device() else {
            eprintln!("SKIP: No f64-capable GPU");
            return;
        };
        let report = probe_device(&device);
        assert!(!report.adapter_name.is_empty());
        println!("{report}");
    }

    #[test]
    fn test_device_precision_report_display() {
        let report = DevicePrecisionReport {
            adapter_name: "Test GPU".to_string(),
            fp64_strategy: Fp64Strategy::Native,
            fp64_rate: Fp64Rate::Full,
            builtins: F64BuiltinCapabilities {
                basic_f64: true,
                exp: true,
                log: true,
                exp2: true,
                log2: true,
                sin: true,
                cos: true,
                sqrt: true,
                fma: true,
                abs_min_max: true,
            },
            has_f64_shaders: true,
            has_spirv_passthrough: false,
        };
        let s = format!("{report}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("Native"));
    }
}
