// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dispatch routing — route ecological workloads to capable substrates.
//!
//! # airSpring Dispatch Patterns
//!
//! - **ET₀ batch**: f64 + shader → GPU (station-parallel `BatchedEt0`)
//! - **Water balance batch**: f64 + shader → GPU (`BatchedWB`)
//! - **Richards PDE**: f64 + shader → GPU (`BatchedRichards`)
//! - **Monte Carlo UQ**: f64 + shader → GPU (parallel parameter sweeps)
//! - **Crop stress classifier**: quant(8) → NPU (int8 FC inference)
//! - **Irrigation decision**: quant(8) → NPU (binary classification)
//! - **Sensor anomaly**: quant(8) → NPU (edge outlier detection)
//! - **Validation harness**: f64 → CPU (sequential, I/O-bound)

use crate::substrate::{Capability, Substrate, SubstrateKind};

/// A workload that needs to be dispatched to a substrate.
#[derive(Debug)]
pub struct Workload {
    /// Human-readable workload name.
    pub name: String,
    /// Capabilities required for this workload.
    pub required: Vec<Capability>,
    /// Preferred substrate kind (if any).
    pub preferred_substrate: Option<SubstrateKind>,
}

/// Dispatch decision — which substrate was chosen and why.
#[derive(Debug)]
pub struct Decision<'a> {
    /// The chosen substrate.
    pub substrate: &'a Substrate,
    /// Why this substrate was chosen.
    pub reason: Reason,
}

/// Why a particular substrate was chosen.
#[derive(Debug, PartialEq, Eq)]
pub enum Reason {
    /// The workload's preferred substrate had all capabilities.
    Preferred,
    /// Best capable substrate by priority (GPU > NPU > CPU).
    BestAvailable,
}

impl Workload {
    /// Create a workload with name and required capabilities.
    #[must_use]
    pub fn new(name: impl Into<String>, required: Vec<Capability>) -> Self {
        Self {
            name: name.into(),
            required,
            preferred_substrate: None,
        }
    }

    /// Set the preferred substrate kind.
    #[must_use]
    pub const fn prefer(mut self, kind: SubstrateKind) -> Self {
        self.preferred_substrate = Some(kind);
        self
    }
}

/// Route a workload to the best matching substrate.
///
/// Priority: preferred → GPU → NPU → Neural → CPU.
///
/// The Neural tier (biomeOS `capability.call`) sits between NPU and CPU
/// because it can dispatch to remote GPUs in the ecosystem but has
/// network round-trip overhead.
#[must_use]
pub fn route<'a>(workload: &Workload, substrates: &'a [Substrate]) -> Option<Decision<'a>> {
    let capable: Vec<&Substrate> = substrates
        .iter()
        .filter(|s| workload.required.iter().all(|req| s.has(req)))
        .collect();

    if capable.is_empty() {
        return None;
    }

    if let Some(pref) = workload.preferred_substrate {
        if let Some(s) = capable.iter().find(|s| s.kind == pref) {
            return Some(Decision {
                substrate: s,
                reason: Reason::Preferred,
            });
        }
    }

    let best = capable
        .iter()
        .find(|s| s.kind == SubstrateKind::Gpu)
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Npu))
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Neural))
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Cpu))?;

    Some(Decision {
        substrate: best,
        reason: Reason::BestAvailable,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::substrate::{Identity, Properties};

    fn make_gpu(name: &str, caps: Vec<Capability>) -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named(name),
            properties: Properties::default(),
            capabilities: caps,
        }
    }

    fn make_cpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("CPU"),
            properties: Properties::default(),
            capabilities: vec![Capability::F64Compute, Capability::F32Compute],
        }
    }

    fn make_npu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Npu,
            identity: Identity::named("AKD1000"),
            properties: Properties::default(),
            capabilities: vec![
                Capability::F32Compute,
                Capability::QuantizedInference { bits: 8 },
                Capability::BatchInference { max_batch: 8 },
            ],
        }
    }

    #[test]
    fn routes_et0_batch_to_gpu() {
        let gpu = make_gpu(
            "RTX 4070",
            vec![
                Capability::F64Compute,
                Capability::ScalarReduce,
                Capability::ShaderDispatch,
            ],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = Workload::new(
            "ET₀ batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
    }

    #[test]
    fn routes_crop_stress_to_npu() {
        let npu = make_npu();
        let cpu = make_cpu();
        let subs = [cpu, npu];
        let work = Workload::new(
            "crop stress classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        );

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Npu);
    }

    #[test]
    fn falls_back_to_cpu() {
        let subs = [make_cpu()];
        let work = Workload::new("validation", vec![Capability::F64Compute]);

        let d = route(&work, &subs).expect("should route to CPU");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
    }

    #[test]
    fn no_route_if_incapable() {
        let subs = [make_cpu()];
        let work = Workload::new(
            "NPU inference",
            vec![Capability::QuantizedInference { bits: 4 }],
        );

        assert!(route(&work, &subs).is_none());
    }

    #[test]
    fn respects_cpu_preference() {
        let gpu = make_gpu("GPU", vec![Capability::F64Compute]);
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work =
            Workload::new("validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
        assert_eq!(d.reason, Reason::Preferred);
    }

    #[test]
    fn gpu_over_npu_for_f64() {
        let gpu = make_gpu("GPU", vec![Capability::F64Compute, Capability::F32Compute]);
        let npu = make_npu();
        let cpu = make_cpu();
        let subs = [cpu, npu, gpu];
        let work = Workload::new("water balance batch", vec![Capability::F64Compute]);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
    }

    #[test]
    fn empty_substrate_list() {
        let work = Workload::new("anything", vec![Capability::F64Compute]);
        assert!(route(&work, &[]).is_none());
    }
}
