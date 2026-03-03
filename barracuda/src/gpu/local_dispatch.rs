// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local GPU compute dispatch — airSpring-evolved shaders pending `ToadStool` absorption.
//!
//! Provides a minimal wgpu compute pipeline for element-wise agricultural
//! operations. Each operation takes three f32 inputs per element and produces
//! one f32 output. Conversions between f64 (CPU domain) and f32 (GPU) are
//! handled transparently.
//!
//! # Precision
//!
//! Local shaders operate in f32 (~7 significant digits). `ToadStool` absorption
//! upgrades these to f64 via `compile_shader_universal` with `Fp64Strategy`.
//!
//! # Operations
//!
//! | Op | Domain | Inputs |
//! |----|--------|--------|
//! | 0 | SCS-CN runoff | P(mm), CN, Ia ratio |
//! | 1 | Stewart yield | Ky, `ETa/ETc`, — |
//! | 2 | Makkink ET₀ | T(°C), Rs(MJ), elev(m) |
//! | 3 | Turc ET₀ | T(°C), Rs(MJ), RH(%) |
//! | 4 | Hamon PET | T(°C), lat(rad), DOY |
//! | 5 | Blaney-Criddle ET₀ | T(°C), lat(rad), DOY |

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use wgpu::util::DeviceExt;

use crate::error::{AirSpringError, Result};

const SHADER_SOURCE: &str = include_str!("../shaders/local_elementwise.wgsl");
const WORKGROUP_SIZE: u32 = 256;

/// Local GPU element-wise compute dispatcher.
///
/// Compiles the `local_elementwise.wgsl` shader once and reuses the pipeline
/// for multiple dispatches. Thread-safe via `Arc<WgpuDevice>`.
pub struct LocalElementwise {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl std::fmt::Debug for LocalElementwise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalElementwise")
            .field("device", &"Arc<WgpuDevice>")
            .finish()
    }
}

/// Operation selector for the WGSL shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LocalOp {
    ScsCnRunoff = 0,
    StewartYield = 1,
    MakkinkEt0 = 2,
    TurcEt0 = 3,
    HamonPet = 4,
    BlaneyCriddleEt0 = 5,
}

/// Uniform parameters buffer layout (must match WGSL `Params` struct).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    op: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

impl LocalElementwise {
    /// Compile the local WGSL shader and create the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let wgpu_device = device.device();

        let shader_module = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("airspring_local_elementwise"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout =
            wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("local_elementwise_bgl"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

        let pipeline_layout =
            wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("local_elementwise_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("local_elementwise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
        })
    }

    /// Dispatch an element-wise operation on GPU.
    ///
    /// Inputs and output are f64 on the CPU side. The GPU operates in f32;
    /// conversions are transparent. Unused input slots should be filled with 0.0.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU buffer mapping or submission fails.
    #[allow(clippy::too_many_lines)]
    pub fn dispatch(
        &self,
        op: LocalOp,
        a: &[f64],
        b: &[f64],
        c: &[f64],
    ) -> Result<Vec<f64>> {
        let n = a.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if b.len() != n || c.len() != n {
            return Err(AirSpringError::InvalidInput(
                "local_dispatch: input arrays must have equal length".into(),
            ));
        }

        let wgpu_device = self.device.device();
        let queue = self.device.queue();

        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let c_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();

        let params = GpuParams {
            op: op as u32,
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let buf_size = (n * std::mem::size_of::<f32>()) as u64;

        let params_buf = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buf_a = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in_a"),
            contents: bytemuck::cast_slice(&a_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_b = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in_b"),
            contents: bytemuck::cast_slice(&b_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_c = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in_c"),
            contents: bytemuck::cast_slice(&c_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_out = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("local_elementwise_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE);

        let mut encoder =
            wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("local_elementwise_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        buf_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        wgpu_device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| AirSpringError::barracuda_msg(format!("buffer map recv: {e}")))?
            .map_err(|e| AirSpringError::barracuda_msg(format!("buffer map: {e}")))?;

        let data = buf_staging.slice(..).get_mapped_range();
        let out_f32: &[f32] = bytemuck::cast_slice(&data);
        let out_f64: Vec<f64> = out_f32.iter().map(|&v| f64::from(v)).collect();
        drop(data);
        buf_staging.unmap();

        Ok(out_f64)
    }
}

const fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device_info::try_f64_device;

    fn try_local() -> Option<LocalElementwise> {
        let device = try_f64_device()?;
        LocalElementwise::new(device).ok()
    }

    #[test]
    fn test_scs_cn_gpu() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU for LocalElementwise");
            return;
        };
        let p = [50.0, 100.0, 25.0, 0.0];
        let cn = [75.0, 85.0, 65.0, 80.0];
        let ia = [0.2, 0.2, 0.2, 0.2];

        let gpu = le.dispatch(LocalOp::ScsCnRunoff, &p, &cn, &ia).unwrap();

        let cpu: Vec<f64> = p.iter().zip(&cn).zip(&ia)
            .map(|((&pp, &cc), &ii)| crate::eco::runoff::scs_cn_runoff(pp, cc, ii))
            .collect();

        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let tol = c.abs().mul_add(1e-3, 1e-4);
            assert!((g - c).abs() < tol, "SCS-CN[{i}] GPU={g:.4} CPU={c:.4}");
        }
    }

    #[test]
    fn test_stewart_gpu() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU for LocalElementwise");
            return;
        };
        let ky = [1.25, 0.85, 1.0];
        let ratio = [0.8, 0.7, 1.0];
        let zeros = [0.0; 3];

        let gpu = le.dispatch(LocalOp::StewartYield, &ky, &ratio, &zeros).unwrap();

        for (i, (&k, &r)) in ky.iter().zip(&ratio).enumerate() {
            let cpu = crate::eco::yield_response::yield_ratio_single(k, r);
            let tol = cpu.abs().mul_add(1e-4, 1e-6);
            assert!((gpu[i] - cpu).abs() < tol, "Stewart[{i}] GPU={:.6} CPU={cpu:.6}", gpu[i]);
        }
    }

    #[test]
    fn test_makkink_gpu() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let t = [20.0, 30.0];
        let rs = [15.0, 25.0];
        let elev = [100.0, 0.0];

        let gpu = le.dispatch(LocalOp::MakkinkEt0, &t, &rs, &elev).unwrap();

        for (i, ((&tt, &rr), &ee)) in t.iter().zip(&rs).zip(&elev).enumerate() {
            let cpu = crate::eco::simple_et0::makkink_et0(tt, rr, ee);
            let tol = cpu.abs().mul_add(2e-3, 0.01);
            assert!((gpu[i] - cpu).abs() < tol, "Makkink[{i}] GPU={:.4} CPU={cpu:.4}", gpu[i]);
        }
    }

    #[test]
    fn test_turc_gpu() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let t = [20.0, 25.0];
        let rs = [15.0, 20.0];
        let rh = [70.0, 40.0];

        let gpu = le.dispatch(LocalOp::TurcEt0, &t, &rs, &rh).unwrap();

        for (i, ((&tt, &rr), &hh)) in t.iter().zip(&rs).zip(&rh).enumerate() {
            let cpu = crate::eco::simple_et0::turc_et0(tt, rr, hh);
            let tol = cpu.abs().mul_add(2e-3, 0.01);
            assert!((gpu[i] - cpu).abs() < tol, "Turc[{i}] GPU={:.4} CPU={cpu:.4}", gpu[i]);
        }
    }

    #[test]
    fn test_hamon_gpu() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let t = [20.0, 10.0];
        let lat = [42.7_f64.to_radians(), 42.7_f64.to_radians()];
        let doy = [180.0, 90.0];

        let gpu = le.dispatch(LocalOp::HamonPet, &t, &lat, &doy).unwrap();

        for (i, ((&tt, &ll), &dd)) in t.iter().zip(&lat).zip(&doy).enumerate() {
            let cpu = crate::eco::simple_et0::hamon_pet_from_location(tt, ll, dd as u32);
            let tol = cpu.abs().mul_add(5e-3, 0.02);
            assert!((gpu[i] - cpu).abs() < tol, "Hamon[{i}] GPU={:.4} CPU={cpu:.4}", gpu[i]);
        }
    }

    #[test]
    fn test_blaney_criddle_gpu() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let t = [25.0, 5.0];
        let lat = [42.7_f64.to_radians(), 42.7_f64.to_radians()];
        let doy = [180.0, 15.0];

        let gpu = le.dispatch(LocalOp::BlaneyCriddleEt0, &t, &lat, &doy).unwrap();

        for (i, ((&tt, &ll), &dd)) in t.iter().zip(&lat).zip(&doy).enumerate() {
            let cpu = crate::eco::simple_et0::blaney_criddle_from_location(tt, ll, dd as u32);
            let tol = cpu.abs().mul_add(5e-3, 0.02);
            assert!(
                (gpu[i] - cpu).abs() < tol,
                "BC[{i}] GPU={:.4} CPU={cpu:.4}", gpu[i]
            );
        }
    }

    #[test]
    fn test_empty_dispatch() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let result = le.dispatch(LocalOp::ScsCnRunoff, &[], &[], &[]).unwrap();
        assert!(result.is_empty());
    }
}
