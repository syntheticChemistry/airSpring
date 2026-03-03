// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware probing — GPU via wgpu, NPU via `/dev/akida*`, CPU via procfs.

use crate::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use std::fs;

/// Probe all GPU adapters via wgpu.
#[must_use]
pub fn probe_gpus() -> Vec<Substrate> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let mut gpus = Vec::new();

    for (idx, adapter) in adapters.into_iter().enumerate() {
        let info = adapter.get_info();
        let features = adapter.features();

        if info.device_type == wgpu::DeviceType::Cpu {
            continue;
        }

        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let mut capabilities = vec![Capability::F32Compute, Capability::ShaderDispatch];
        if has_f64 {
            capabilities.push(Capability::F64Compute);
            capabilities.push(Capability::ScalarReduce);
        }
        if has_timestamps {
            capabilities.push(Capability::TimestampQuery);
        }

        gpus.push(Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: info.name.clone(),
                driver: Some(format!("{} ({})", info.driver, info.driver_info)),
                backend: Some(format!("{:?}", info.backend)),
                adapter_index: Some(idx),
                device_node: None,
                pci_id: None,
            },
            properties: Properties {
                has_f64,
                has_timestamps,
                ..Properties::default()
            },
            capabilities,
        });
    }

    gpus
}

/// Probe CPU via `/proc/cpuinfo` and `/proc/meminfo`.
#[must_use]
pub fn probe_cpu() -> Substrate {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let (model, cores, threads, cache_kb, has_avx2) = parse_cpuinfo(&cpuinfo);
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mem_bytes = parse_meminfo(&meminfo);

    let name = model.unwrap_or_else(|| String::from("Unknown CPU"));

    let mut capabilities = vec![
        Capability::F64Compute,
        Capability::F32Compute,
        Capability::CpuCompute,
    ];
    if has_avx2 {
        capabilities.push(Capability::SimdVector);
    }

    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named(name),
        properties: Properties {
            memory_bytes: mem_bytes,
            core_count: cores,
            thread_count: threads,
            cache_kb,
            ..Properties::default()
        },
        capabilities,
    }
}

/// Probe for NPU devices via runtime discovery.
///
/// Discovery strategy (capability-based, no hardcoded paths):
/// 1. If `NPU_DEVICE_PATH` is set, probe that single device.
/// 2. Otherwise, scan `/dev/akida*` for all available NPU devices.
///
/// Each discovered device is reported with its full capability set.
#[must_use]
pub fn probe_npus() -> Vec<Substrate> {
    if let Ok(explicit_path) = std::env::var("NPU_DEVICE_PATH") {
        return probe_npu_at(&explicit_path).into_iter().collect();
    }

    let mut npus = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/dev") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_str().is_some_and(|n| n.starts_with("akida")) {
                if let Some(s) = probe_npu_at(&entry.path().to_string_lossy()) {
                    npus.push(s);
                }
            }
        }
    }
    npus
}

fn probe_npu_at(device_path: &str) -> Option<Substrate> {
    let path = std::path::Path::new(device_path);
    if !path.exists() {
        return None;
    }
    Some(Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity {
            name: String::from("BrainChip AKD1000"),
            device_node: Some(device_path.to_string()),
            ..Identity::named("BrainChip AKD1000")
        },
        properties: Properties::default(),
        capabilities: vec![
            Capability::F32Compute,
            Capability::QuantizedInference { bits: 8 },
            Capability::QuantizedInference { bits: 4 },
            Capability::BatchInference { max_batch: 8 },
            Capability::WeightMutation,
        ],
    })
}

fn parse_cpuinfo(content: &str) -> (Option<String>, Option<u32>, Option<u32>, Option<u32>, bool) {
    let mut model = None;
    let mut cores = None;
    let mut siblings = None;
    let mut cache_kb = None;
    let mut has_avx2 = false;

    for line in content.lines() {
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "model name" if model.is_none() => model = Some(val.to_string()),
                "cpu cores" if cores.is_none() => cores = val.parse().ok(),
                "siblings" if siblings.is_none() => siblings = val.parse().ok(),
                "cache size" if cache_kb.is_none() => {
                    cache_kb = val.trim_end_matches(" KB").parse().ok();
                }
                "flags" if !has_avx2 => {
                    has_avx2 = val.split_whitespace().any(|f| f == "avx2");
                }
                _ => {}
            }
        }
    }

    (model, cores, siblings, cache_kb, has_avx2)
}

fn parse_meminfo(content: &str) -> Option<u64> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.trim().trim_end_matches(" kB").trim();
            let kb: u64 = kb_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_discovered() {
        let cpu = probe_cpu();
        assert_eq!(cpu.kind, SubstrateKind::Cpu);
        assert!(cpu.has(&Capability::F64Compute));
        assert!(!cpu.identity.name.is_empty());
    }

    #[test]
    fn gpu_probe_uses_wgpu() {
        let gpus = probe_gpus();
        for gpu in &gpus {
            assert_eq!(gpu.kind, SubstrateKind::Gpu);
            assert!(gpu.has(&Capability::ShaderDispatch));
        }
    }

    #[test]
    fn parse_cpuinfo_extracts_model() {
        let content = "model name\t: Intel(R) Core(TM) i9-12900K\ncpu cores\t: 8\nsiblings\t: 24\ncache size\t: 30720 KB\nflags\t\t: fpu vme de sse sse2 avx avx2\n";
        let (model, cores, threads, cache, avx2) = parse_cpuinfo(content);
        assert_eq!(model.unwrap(), "Intel(R) Core(TM) i9-12900K");
        assert_eq!(cores.unwrap(), 8);
        assert_eq!(threads.unwrap(), 24);
        assert_eq!(cache.unwrap(), 30720);
        assert!(avx2);
    }

    #[test]
    fn parse_cpuinfo_empty() {
        let (model, cores, threads, cache, avx2) = parse_cpuinfo("");
        assert!(model.is_none());
        assert!(cores.is_none());
        assert!(threads.is_none());
        assert!(cache.is_none());
        assert!(!avx2);
    }

    #[test]
    fn parse_meminfo_extracts_total() {
        let content = "MemTotal:       32749772 kB\nMemFree:        15000000 kB\n";
        let bytes = parse_meminfo(content).unwrap();
        assert_eq!(bytes, 32_749_772 * 1024);
    }

    #[test]
    fn parse_meminfo_empty() {
        assert!(parse_meminfo("").is_none());
    }
}
