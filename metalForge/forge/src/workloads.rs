// SPDX-License-Identifier: AGPL-3.0-or-later

//! Preset workloads for airSpring ecological/agricultural domains.
//!
//! Each workload declares required capabilities and shader origin. The three
//! substrates serve distinct roles:
//!
//! - **GPU**: f64 batch compute — ET₀ grids, water balance sweeps, Richards
//!   PDE, Monte Carlo uncertainty quantification, yield response surfaces.
//! - **NPU**: int8 edge classifiers — crop stress detection, irrigation
//!   scheduling decisions, sensor anomaly flags. These are small FC layers
//!   that fit in AKD1000's 8 MB SRAM with sub-millisecond latency.
//! - **CPU**: validation harnesses, I/O, sequential control (always available).
//!
//! # NPU Workloads for Agriculture
//!
//! The AKD1000 excels at binary/multi-class classification with quantized
//! weights. Agricultural edge use cases:
//!
//! 1. **Crop stress classifier**: soil depletion fraction + ET ratio → stressed
//!    vs healthy (binary int8). Runs on field-edge hardware at sensor cadence.
//! 2. **Irrigation decision**: forecast ET₀ + current soil moisture + crop
//!    stage → irrigate/hold/deficit (3-class int8). Real-time scheduling.
//! 3. **Sensor anomaly detector**: ET₀ or θ reading → normal/outlier
//!    (binary int8). Catches broken sensors before bad data propagates.

use crate::dispatch::Workload;
use crate::substrate::{Capability, SubstrateKind};

/// Where the compute implementation for a workload lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderOrigin {
    /// Absorbed by `ToadStool` — uses `barracuda::ops::*` primitives.
    Absorbed,
    /// Local WGSL shader in `barracuda/src/shaders/` — pending absorption.
    Local,
    /// CPU-only domain — no GPU shader exists or is planned.
    CpuOnly,
    /// NPU-native — int8 inference on `BrainChip` AKD1000.
    NpuNative,
}

/// An eco workload with shader provenance tracking.
#[derive(Debug)]
pub struct EcoWorkload {
    /// The dispatch workload (name + capabilities).
    pub workload: Workload,
    /// Where the implementation lives.
    pub origin: ShaderOrigin,
    /// `ToadStool` primitive name (if absorbed).
    pub primitive: Option<&'static str>,
    /// ODE/PDE system dimensions (if applicable).
    pub system_dims: Option<SystemDims>,
}

/// System dimensions for dispatch sizing.
#[derive(Debug, Clone, Copy)]
pub struct SystemDims {
    /// Number of state variables.
    pub n_vars: u32,
    /// Number of parameters per batch element.
    pub n_params: u32,
}

impl EcoWorkload {
    const fn new_static(origin: ShaderOrigin) -> Self {
        Self {
            workload: Workload {
                name: String::new(),
                required: Vec::new(),
                preferred_substrate: None,
            },
            origin,
            primitive: None,
            system_dims: None,
        }
    }

    fn named(mut self, name: &str, required: Vec<Capability>) -> Self {
        self.workload.name = name.to_string();
        self.workload.required = required;
        self
    }

    const fn with_primitive(mut self, primitive: &'static str) -> Self {
        self.primitive = Some(primitive);
        self
    }

    const fn with_dims(mut self, n_vars: u32, n_params: u32) -> Self {
        self.system_dims = Some(SystemDims { n_vars, n_params });
        self
    }

    /// Whether this workload uses a local (non-absorbed) WGSL shader.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self.origin, ShaderOrigin::Local)
    }

    /// Whether this workload has been absorbed by `ToadStool`.
    #[must_use]
    pub const fn is_absorbed(&self) -> bool {
        matches!(self.origin, ShaderOrigin::Absorbed)
    }

    /// Whether this workload is CPU-only.
    #[must_use]
    pub const fn is_cpu_only(&self) -> bool {
        matches!(self.origin, ShaderOrigin::CpuOnly)
    }

    /// Whether this workload targets the NPU natively.
    #[must_use]
    pub const fn is_npu_native(&self) -> bool {
        matches!(self.origin, ShaderOrigin::NpuNative)
    }
}

// ── GPU batch compute domains (absorbed by ToadStool) ───────────────

/// Penman-Monteith `ET₀` batch — station-parallel evapotranspiration.
#[must_use]
pub fn et0_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedEt0F64")
}

/// FAO-56 water balance batch — multi-crop seasonal simulation.
#[must_use]
pub fn water_balance_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "water_balance_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedWaterBalanceF64")
}

/// Richards 1D PDE — vadose zone flow (N nodes × M soils).
#[must_use]
pub fn richards_pde() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "richards_pde",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedRichards1dF64")
        .with_dims(50, 5)
}

/// Multi-crop yield response surface — parameter sweep.
#[must_use]
pub fn yield_response_surface() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "yield_response_surface",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4F64")
        .with_dims(1, 8)
}

/// Monte Carlo uncertainty quantification — ET₀ parameter sampling.
#[must_use]
pub fn monte_carlo_uq() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "monte_carlo_uq",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Sorption isotherm fitting — Langmuir/Freundlich batch.
#[must_use]
pub fn isotherm_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "isotherm_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedIsothermF64")
}

/// Growing degree day accumulation — station × year grid.
#[must_use]
pub fn gdd_accumulate() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "gdd_accumulate",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Dual crop coefficient batch — `Ke` + `Kcb` at field scale.
#[must_use]
pub fn dual_kc_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dual_kc_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedDualKcF64")
}

/// Forecast scheduling — ensemble NWP → irrigation plan.
#[must_use]
pub fn forecast_scheduling() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "forecast_scheduling",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedForecastF64")
}

// ── GPU orchestrators (absorbed by ToadStool S70+) ──────────────────

/// Hargreaves-Samani ET₀ batch — temperature-only ET₀ estimate.
/// Absorbed via `HargreavesBatchGpu` (S71).
#[must_use]
pub fn hargreaves_et0_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "hargreaves_et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("HargreavesBatchGpu")
}

/// Kc climate adjustment batch — FAO-56 Eq. 62 for wind/humidity.
/// Absorbed via `BatchedElementwiseF64` op=7 (S70+).
#[must_use]
pub fn kc_climate_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "kc_climate_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedElementwiseF64_op7")
}

/// Sensor calibration batch — `SoilWatch` 10 raw→VWC.
/// Absorbed via `BatchedElementwiseF64` op=5 (S70+).
#[must_use]
pub fn sensor_calibration_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "sensor_calibration_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedElementwiseF64_op5")
}

/// Seasonal pipeline — ET₀→Kc→WB→Yield chained pipeline.
/// Stages 1-2 GPU via `BatchedElementwiseF64`, stages 3-4 CPU fallback.
/// Full GPU pending `BatchedEncoder` (S80+).
#[must_use]
pub fn seasonal_pipeline() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "seasonal_pipeline",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("SeasonalPipelineF64")
}

// ── NPU-native classifiers ─────────────────────────────────────────

/// Crop stress classifier — binary (stressed / healthy).
///
/// Input: 4 features (depletion fraction, `ETa/ETc`, soil θ, Ks).
/// Output: 2-class softmax (int8 quantized FC).
/// Latency target: <1 ms on AKD1000.
#[must_use]
pub fn crop_stress_classifier() -> EcoWorkload {
    let mut w = EcoWorkload::new_static(ShaderOrigin::NpuNative).named(
        "crop_stress_classifier",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    w.workload.preferred_substrate = Some(SubstrateKind::Npu);
    w
}

/// Irrigation decision — 3-class (irrigate / hold / deficit-allow).
///
/// Input: 6 features (forecast `ET₀`, current θ, `TAW`, crop stage, `Ks`, rain probability).
/// Output: 3-class softmax (int8 quantized FC).
#[must_use]
pub fn irrigation_decision() -> EcoWorkload {
    let mut w = EcoWorkload::new_static(ShaderOrigin::NpuNative).named(
        "irrigation_decision",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    w.workload.preferred_substrate = Some(SubstrateKind::Npu);
    w
}

/// Sensor anomaly detector — binary (normal / outlier).
///
/// Input: 3 features (reading, rolling mean, rolling σ).
/// Output: 2-class softmax (int8 quantized FC).
#[must_use]
pub fn sensor_anomaly() -> EcoWorkload {
    let mut w = EcoWorkload::new_static(ShaderOrigin::NpuNative).named(
        "sensor_anomaly",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    w.workload.preferred_substrate = Some(SubstrateKind::Npu);
    w
}

// ── Paper 12 — Immunological Anderson ────────────────────────────────

/// Tissue diversity profiling — cell-type Pielou evenness → Anderson disorder W.
///
/// Dispatches to `GpuDiversity` (Shannon/Simpson/Pielou) for tissue samples.
/// GPU path available via `DiversityFusionGpu` (`ToadStool` S70+).
#[must_use]
pub fn tissue_diversity() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "tissue_diversity",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("DiversityFusionF64")
}

/// `CytokineBrain` evolutionary reservoir — AD flare regime prediction.
///
/// CPU-only: Nautilus reservoir computing (`bingocube-nautilus`). No WGSL shader.
/// NPU export available for edge deployment via `NautilusShell::export_akd1000_weights()`.
#[must_use]
pub fn cytokine_brain() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("cytokine_brain", vec![Capability::F64Compute])
}

/// AD flare classifier — NPU binary (healthy / flare) from cytokine panel.
///
/// Input: 7 features (time, IL-31, IL-4, IL-13, pruritus, TEWL, Pielou).
/// Output: 2-class softmax (int8 quantized FC from `CytokineBrain` export).
/// Enables edge deployment on AKD1000 for real-time disease state monitoring.
#[must_use]
pub fn ad_flare_classifier() -> EcoWorkload {
    let mut w = EcoWorkload::new_static(ShaderOrigin::NpuNative).named(
        "ad_flare_classifier",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    w.workload.preferred_substrate = Some(SubstrateKind::Npu);
    w
}

// ── Local WGSL shaders (pending ToadStool absorption) ───────────────

/// SCS-CN runoff batch — element-wise Q from (P, CN, Ia ratio).
///
/// Local f32 WGSL shader. Becomes `BatchedElementwiseF64` op=14 on absorption.
#[must_use]
pub fn scs_cn_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Local)
        .named(
            "scs_cn_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("LocalElementwise_op0")
}

/// Stewart yield response batch — element-wise Ya/Ymax from (Ky, `ETa/ETc`).
///
/// Local f32 WGSL shader. Becomes `BatchedElementwiseF64` op=15 on absorption.
#[must_use]
pub fn stewart_yield_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Local)
        .named(
            "stewart_yield_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("LocalElementwise_op1")
}

/// Makkink ET₀ batch — radiation-based ET₀ from (T, Rs, elev).
///
/// Local f32 WGSL shader. Becomes `BatchedElementwiseF64` op=16 on absorption.
#[must_use]
pub fn makkink_et0_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Local)
        .named(
            "makkink_et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("LocalElementwise_op2")
}

/// Turc ET₀ batch — temp-radiation-humidity from (T, Rs, RH).
///
/// Local f32 WGSL shader. Becomes `BatchedElementwiseF64` op=17 on absorption.
#[must_use]
pub fn turc_et0_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Local)
        .named(
            "turc_et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("LocalElementwise_op3")
}

/// Hamon PET batch — temperature-only PET from (T, lat, DOY).
///
/// Local f32 WGSL shader. Becomes `BatchedElementwiseF64` op=18 on absorption.
#[must_use]
pub fn hamon_pet_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Local)
        .named(
            "hamon_pet_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("LocalElementwise_op4")
}

/// Blaney-Criddle ET₀ batch — daylight-based ET₀ from (T, lat, DOY).
///
/// Local f32 WGSL shader. Becomes `BatchedElementwiseF64` op=19 on absorption.
#[must_use]
pub fn blaney_criddle_et0_batch() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::Local)
        .named(
            "blaney_criddle_et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("LocalElementwise_op5")
}

// ── CPU-only domains ────────────────────────────────────────────────

/// Validation harness — CPU sequential checks.
#[must_use]
pub fn validation_harness() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("validation_harness", vec![Capability::F64Compute])
}

/// Weather data ingest — CSV/JSON parsing, I/O-bound.
#[must_use]
pub fn weather_ingest() -> EcoWorkload {
    EcoWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("weather_ingest", vec![Capability::CpuCompute])
}

// ── Inventory ───────────────────────────────────────────────────────

/// All known eco domain workloads.
#[must_use]
pub fn all_workloads() -> Vec<EcoWorkload> {
    vec![
        // GPU batch compute (absorbed)
        et0_batch(),
        water_balance_batch(),
        richards_pde(),
        yield_response_surface(),
        monte_carlo_uq(),
        isotherm_batch(),
        gdd_accumulate(),
        dual_kc_batch(),
        forecast_scheduling(),
        // GPU orchestrators (absorbed S70+)
        hargreaves_et0_batch(),
        kc_climate_batch(),
        sensor_calibration_batch(),
        seasonal_pipeline(),
        // NPU-native classifiers
        crop_stress_classifier(),
        irrigation_decision(),
        sensor_anomaly(),
        // Paper 12 — Immunological Anderson
        tissue_diversity(),
        cytokine_brain(),
        ad_flare_classifier(),
        // Local WGSL shaders (pending ToadStool absorption)
        scs_cn_batch(),
        stewart_yield_batch(),
        makkink_et0_batch(),
        turc_et0_batch(),
        hamon_pet_batch(),
        blaney_criddle_et0_batch(),
        // CPU-only
        validation_harness(),
        weather_ingest(),
    ]
}

/// Count workloads by shader origin.
#[must_use]
pub fn origin_summary() -> (usize, usize, usize, usize) {
    let all = all_workloads();
    let absorbed = all.iter().filter(|w| w.is_absorbed()).count();
    let local = all.iter().filter(|w| w.is_local()).count();
    let npu_native = all.iter().filter(|w| w.is_npu_native()).count();
    let cpu_only = all.iter().filter(|w| w.is_cpu_only()).count();
    (absorbed, local, npu_native, cpu_only)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn all_workloads_has_entries() {
        let all = all_workloads();
        assert_eq!(all.len(), 27, "27 eco workloads (21 + 6 local WGSL)");
    }

    #[test]
    fn origin_counts_match() {
        let (absorbed, local, npu_native, cpu_only) = origin_summary();
        assert_eq!(absorbed, 14, "14 absorbed GPU domains (9 original + 4 S70+ ops + tissue)");
        assert_eq!(local, 6, "6 local WGSL shaders (SCS-CN, Stewart, Makkink, Turc, Hamon, BC)");
        assert_eq!(npu_native, 4, "4 NPU-native classifiers");
        assert_eq!(cpu_only, 3, "3 CPU-only domains");
    }

    #[test]
    fn npu_workloads_prefer_npu() {
        for w in [
            crop_stress_classifier(),
            irrigation_decision(),
            sensor_anomaly(),
            ad_flare_classifier(),
        ] {
            assert!(w.is_npu_native());
            assert_eq!(w.workload.preferred_substrate, Some(SubstrateKind::Npu));
        }
    }

    #[test]
    fn absorbed_workloads_have_primitive() {
        for w in all_workloads() {
            if !w.is_absorbed() {
                continue;
            }
            assert!(
                w.primitive.is_some(),
                "{} should have primitive name",
                w.workload.name
            );
        }
    }

    #[test]
    fn richards_has_dims() {
        let w = richards_pde();
        assert!(w.is_absorbed());
        let dims = w.system_dims.expect("should have dims");
        assert_eq!(dims.n_vars, 50);
        assert_eq!(dims.n_params, 5);
    }

    #[test]
    fn no_duplicate_names() {
        let all = all_workloads();
        let mut names: Vec<&str> = all.iter().map(|w| w.workload.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate workload names found");
    }
}
