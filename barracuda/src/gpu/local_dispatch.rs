// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local GPU compute dispatch — f64 canonical, universal precision.
//!
//! Provides a precision-aware wgpu compute pipeline for element-wise
//! agricultural operations. Shaders are written in f64 (canonical) and
//! compiled via `BarraCuda`'s `compile_shader_universal()`:
//!
//! - **`F64`**: native f64 on pro GPUs (Titan V, A100, MI250)
//! - **`Df64`**: double-float f32-pair (~48-bit) on consumer GPUs
//! - **`F32`**: downcast to f32 on consumer GPUs (RTX 4070, Arc, RX 7900)
//!
//! "Math is universal, precision is silicon" — `BarraCuda` S67.
//!
//! # Cross-Spring Evolution Provenance
//!
//! | Component | Origin Spring | Session |
//! |-----------|---------------|---------|
//! | `compile_shader_universal` | neuralSpring → `BarraCuda` S68 | Architecture |
//! | `math_f64.wgsl` precision | hotSpring lattice QCD | S54 (pow, acos, sin) |
//! | `Fp64Strategy` probing | hotSpring → `BarraCuda` S58 | Device detection |
//! | SCS-CN, Stewart, ET₀ ops | airSpring domain science | V0.6.9 (local shaders) |
//! | `downcast_f64_to_f32` | `BarraCuda` S68 | Text compiler |
//!
//! # Operations
//!
//! | Op | Domain | Inputs | Upstream |
//! |----|--------|--------|----------|
//! | 0 | SCS-CN runoff | P(mm), CN, Ia ratio | local only |
//! | 1 | Stewart yield | Ky, `ETa/ETc`, — | local only |
//! | 2 | Makkink ET₀ | T(°C), Rs(MJ), elev(m) | absorbed: `Op::MakkinkEt0` (14) |
//! | 3 | Turc ET₀ | T(°C), Rs(MJ), RH(%) | absorbed: `Op::TurcEt0` (15) |
//! | 4 | Hamon PET | T(°C), lat(rad), DOY | absorbed: `Op::HamonEt0` (16) |
//! | 5 | Blaney-Criddle ET₀ | T(°C), lat(rad), DOY | local only |

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::Precision;
use wgpu::util::DeviceExt;

use crate::error::{AirSpringError, Result};

const F64_SHADER_SOURCE: &str = include_str!("../shaders/local_elementwise_f64.wgsl");
const WORKGROUP_SIZE: u32 = 256;

/// Local GPU element-wise compute dispatcher with universal precision.
///
/// Compiles the f64-canonical `local_elementwise_f64.wgsl` shader for the
/// device's native precision and reuses the pipeline for multiple dispatches.
pub struct LocalElementwise {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    precision: Precision,
}

impl std::fmt::Debug for LocalElementwise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalElementwise")
            .field("precision", &self.precision)
            .finish_non_exhaustive()
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
    /// Compile the f64-canonical WGSL shader for production use.
    ///
    /// "Math is universal, precision is silicon." The canonical f64 source is
    /// compiled to f32 via `compile_shader_universal` — adequate for agricultural
    /// science (FAO-56 needs ~6 digits; f32 gives ~7).
    ///
    /// Use [`with_precision`](Self::with_precision) to force a specific
    /// precision for benchmarking or pro-GPU deployments.
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Self::with_precision(device, Precision::F32)
    }

    /// Compile the f64-canonical WGSL shader for a specific precision target.
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation fails.
    pub fn with_precision(device: Arc<WgpuDevice>, precision: Precision) -> Result<Self> {
        let source_for_compile = F64_SHADER_SOURCE.replace("enable f64;\n", "");
        let shader_module = device.compile_shader_universal(
            &source_for_compile,
            precision,
            Some("local_elementwise"),
        );

        let wgpu_device = device.device();

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

        let pipeline_layout = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("local_elementwise_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("local_elementwise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            precision,
        })
    }

    /// The precision this pipeline was compiled for.
    #[must_use]
    pub const fn precision(&self) -> Precision {
        self.precision
    }

    /// Dispatch an element-wise operation on GPU.
    ///
    /// Inputs and output are f64 on the CPU side. Buffer packing adapts to
    /// the compiled precision: f64 buffers for `Precision::F64`, f32 for
    /// `Precision::F32` and `Precision::Df64` (DF64 uses f32-pair buffers).
    ///
    /// # Errors
    ///
    /// Returns an error if GPU buffer mapping or submission fails.
    pub fn dispatch(&self, op: LocalOp, a: &[f64], b: &[f64], c: &[f64]) -> Result<Vec<f64>> {
        let n = a.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if b.len() != n || c.len() != n {
            return Err(AirSpringError::InvalidInput(
                "local_dispatch: input arrays must have equal length".into(),
            ));
        }

        match self.precision {
            Precision::F64 => self.dispatch_f64(op, a, b, c, n),
            Precision::F32 | Precision::Df64 | Precision::F16 => self.dispatch_f32(op, a, b, c, n),
        }
    }

    /// F64 dispatch path — native f64 buffers, no precision loss.
    fn dispatch_f64(
        &self,
        op: LocalOp,
        a: &[f64],
        b: &[f64],
        c: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        let wgpu_device = self.device.device();
        let queue = self.device.queue();

        let params = gpu_params(op, n)?;
        let buf_size = checked_buf_size::<f64>(n)?;

        let params_buf = create_init_buf(
            wgpu_device,
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let buf_a = create_init_buf(
            wgpu_device,
            "in_a",
            bytemuck::cast_slice(a),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_b = create_init_buf(
            wgpu_device,
            "in_b",
            bytemuck::cast_slice(b),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_c = create_init_buf(
            wgpu_device,
            "in_c",
            bytemuck::cast_slice(c),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_out = create_output_buf(wgpu_device, "out", buf_size);
        let buf_staging = create_staging_buf(wgpu_device, "staging", buf_size);

        let bind_group =
            self.create_bind_group(wgpu_device, &params_buf, &buf_a, &buf_b, &buf_c, &buf_out);

        submit_and_copy(&SubmitParams {
            device: wgpu_device,
            queue,
            pipeline: &self.pipeline,
            bind_group: &bind_group,
            n,
            buf_out: &buf_out,
            buf_staging: &buf_staging,
            buf_size,
        });

        let data = map_read(wgpu_device, &buf_staging)?;
        let out: &[f64] = bytemuck::cast_slice(&data);
        let result = out.to_vec();
        drop(data);
        buf_staging.unmap();
        Ok(result)
    }

    /// F32 dispatch path — downcast f64→f32, compute, upcast f32→f64.
    fn dispatch_f32(
        &self,
        op: LocalOp,
        a: &[f64],
        b: &[f64],
        c: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        let wgpu_device = self.device.device();
        let queue = self.device.queue();

        #[allow(
            clippy::cast_possible_truncation,
            reason = "intentional f64→f32 downcast for consumer-GPU dispatch path"
        )]
        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "intentional f64→f32 downcast for consumer-GPU dispatch path"
        )]
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "intentional f64→f32 downcast for consumer-GPU dispatch path"
        )]
        let c_f32: Vec<f32> = c.iter().map(|&v| v as f32).collect();

        let params = gpu_params(op, n)?;
        let buf_size = checked_buf_size::<f32>(n)?;

        let params_buf = create_init_buf(
            wgpu_device,
            "params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let buf_a = create_init_buf(
            wgpu_device,
            "in_a",
            bytemuck::cast_slice(&a_f32),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_b = create_init_buf(
            wgpu_device,
            "in_b",
            bytemuck::cast_slice(&b_f32),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_c = create_init_buf(
            wgpu_device,
            "in_c",
            bytemuck::cast_slice(&c_f32),
            wgpu::BufferUsages::STORAGE,
        );
        let buf_out = create_output_buf(wgpu_device, "out", buf_size);
        let buf_staging = create_staging_buf(wgpu_device, "staging", buf_size);

        let bind_group =
            self.create_bind_group(wgpu_device, &params_buf, &buf_a, &buf_b, &buf_c, &buf_out);

        submit_and_copy(&SubmitParams {
            device: wgpu_device,
            queue,
            pipeline: &self.pipeline,
            bind_group: &bind_group,
            n,
            buf_out: &buf_out,
            buf_staging: &buf_staging,
            buf_size,
        });

        let data = map_read(wgpu_device, &buf_staging)?;
        let out_f32: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = out_f32.iter().map(|&v| f64::from(v)).collect();
        drop(data);
        buf_staging.unmap();
        Ok(result)
    }

    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("local_elementwise_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.as_entire_binding(),
                },
            ],
        })
    }
}

// ── Helper functions ─────────────────────────────────────────────────

fn gpu_params(op: LocalOp, n: usize) -> Result<GpuParams> {
    Ok(GpuParams {
        op: op as u32,
        n: u32::try_from(n)
            .map_err(|_| AirSpringError::InvalidInput("batch size exceeds u32::MAX".into()))?,
        _pad0: 0,
        _pad1: 0,
    })
}

fn checked_buf_size<T>(n: usize) -> Result<u64> {
    u64::try_from(n * std::mem::size_of::<T>())
        .map_err(|_| AirSpringError::InvalidInput("buffer size exceeds u64::MAX".into()))
}

fn create_init_buf(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

fn create_output_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_staging_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

struct SubmitParams<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    n: usize,
    buf_out: &'a wgpu::Buffer,
    buf_staging: &'a wgpu::Buffer,
    buf_size: u64,
}

fn submit_and_copy(p: &SubmitParams<'_>) {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "n is validated to fit in u32 by gpu_params()"
    )]
    let workgroups = (p.n as u32).div_ceil(WORKGROUP_SIZE);

    let mut encoder = p
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("local_elementwise_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(p.pipeline);
        pass.set_bind_group(0, p.bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(p.buf_out, 0, p.buf_staging, 0, p.buf_size);
    p.queue.submit(std::iter::once(encoder.finish()));
}

fn map_read(device: &wgpu::Device, staging: &wgpu::Buffer) -> Result<wgpu::BufferView> {
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    rx.recv()
        .map_err(|e| AirSpringError::barracuda_msg(format!("buffer map recv: {e}")))?
        .map_err(|e| AirSpringError::barracuda_msg(format!("buffer map: {e}")))?;

    Ok(staging.slice(..).get_mapped_range())
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::gpu::device_info::try_f64_device;

    fn try_local() -> Option<LocalElementwise> {
        let device = try_f64_device()?;
        LocalElementwise::new(device).ok()
    }

    #[test]
    fn test_precision_detected() {
        let Some(le) = try_local() else {
            eprintln!("SKIP: no GPU for LocalElementwise");
            return;
        };
        let p = le.precision();
        eprintln!("LocalElementwise precision: {p:?}");
        assert!(matches!(
            p,
            Precision::F64 | Precision::F32 | Precision::Df64
        ));
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

        let cpu: Vec<f64> = p
            .iter()
            .zip(&cn)
            .zip(&ia)
            .map(|((&pp, &cc), &ii)| crate::eco::runoff::scs_cn_runoff(pp, cc, ii))
            .collect();

        let tol = precision_tol(le.precision(), 1e-10, 1e-3);
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            assert!(
                (g - c).abs() < c.abs().mul_add(tol, 1e-10),
                "SCS-CN[{i}] GPU={g:.8} CPU={c:.8} prec={:?}",
                le.precision()
            );
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

        let gpu = le
            .dispatch(LocalOp::StewartYield, &ky, &ratio, &zeros)
            .unwrap();

        let tol = precision_tol(le.precision(), 1e-12, 1e-4);
        for (i, (&k, &r)) in ky.iter().zip(&ratio).enumerate() {
            let cpu = crate::eco::yield_response::yield_ratio_single(k, r);
            assert!(
                (gpu[i] - cpu).abs() < cpu.abs().mul_add(tol, 1e-12),
                "Stewart[{i}] GPU={:.8} CPU={cpu:.8}",
                gpu[i]
            );
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
        let tol = precision_tol(le.precision(), 1e-8, 2e-3);
        for (i, ((&tt, &rr), &ee)) in t.iter().zip(&rs).zip(&elev).enumerate() {
            let cpu = crate::eco::simple_et0::makkink_et0(tt, rr, ee);
            assert!(
                (gpu[i] - cpu).abs() < cpu.abs().mul_add(tol, 0.01),
                "Makkink[{i}] GPU={:.6} CPU={cpu:.6}",
                gpu[i]
            );
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
        let tol = precision_tol(le.precision(), 1e-8, 2e-3);
        for (i, ((&tt, &rr), &hh)) in t.iter().zip(&rs).zip(&rh).enumerate() {
            let cpu = crate::eco::simple_et0::turc_et0(tt, rr, hh);
            assert!(
                (gpu[i] - cpu).abs() < cpu.abs().mul_add(tol, 0.01),
                "Turc[{i}] GPU={:.6} CPU={cpu:.6}",
                gpu[i]
            );
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
        let tol = precision_tol(le.precision(), 1e-6, 5e-3);
        for (i, ((&tt, &ll), &dd)) in t.iter().zip(&lat).zip(&doy).enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let cpu = crate::eco::simple_et0::hamon_pet_from_location(tt, ll, dd as u32);
            assert!(
                (gpu[i] - cpu).abs() < cpu.abs().mul_add(tol, 0.02),
                "Hamon[{i}] GPU={:.6} CPU={cpu:.6}",
                gpu[i]
            );
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

        let gpu = le
            .dispatch(LocalOp::BlaneyCriddleEt0, &t, &lat, &doy)
            .unwrap();
        let tol = precision_tol(le.precision(), 1e-6, 5e-3);
        for (i, ((&tt, &ll), &dd)) in t.iter().zip(&lat).zip(&doy).enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let cpu = crate::eco::simple_et0::blaney_criddle_from_location(tt, ll, dd as u32);
            assert!(
                (gpu[i] - cpu).abs() < cpu.abs().mul_add(tol, 0.02),
                "BC[{i}] GPU={:.6} CPU={cpu:.6}",
                gpu[i]
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

    /// Select tolerance based on compiled precision.
    /// F64 native gives near-exact agreement with CPU f64 baseline.
    /// F32 gives ~7 significant digits (~1e-3 relative).
    fn precision_tol(prec: Precision, f64_tol: f64, f32_tol: f64) -> f64 {
        match prec {
            Precision::F64 => f64_tol,
            _ => f32_tol,
        }
    }
}
