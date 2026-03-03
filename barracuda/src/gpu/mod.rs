// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU acceleration layer — airSpring ↔ `ToadStool`/`BarraCuda` bridge.
//!
//! This module provides domain-specific wrappers around `ToadStool` GPU primitives
//! for precision agriculture workloads. All functions have CPU fallbacks and
//! can be used without a GPU device.
//!
//! # Modules
//!
//! | Module | Purpose | Backend |
//! |--------|---------|---------|
//! | [`device_info`] | Precision probing, `Fp64Strategy`, provenance | Device + cross-spring |
//! | [`et0`] | Batched FAO-56 ET₀ for `N` station-days | **GPU-first** (`BatchedElementwiseF64`) |
//! | [`hargreaves`] | Batched Hargreaves-Samani ET₀ (temp-only) | **GPU-first** (`BatchedElementwiseF64` op=6, S70+) |
//! | [`simple_et0`] | Batched Makkink/Turc/Hamon/Blaney-Criddle | **GPU-local** (f32 WGSL, `ToadStool` f64 pending) |
//! | [`water_balance`] | Batched season simulation + GPU step | **GPU-step** + CPU season |
//! | [`infiltration`] | Batched Green-Ampt via `BrentGpu` | **GPU** (`brent_f64.wgsl` GA residual, S83) |
//! | [`runoff`] | Batched SCS-CN runoff computation | **GPU-local** (f32 WGSL, `ToadStool` f64 pending) |
//! | [`yield_response`] | Batched Stewart yield-water function | **GPU-local** (f32 WGSL, `ToadStool` f64 pending) |
//! | [`dual_kc`] | Batched dual Kc (`Ke` + `ETc`) for M fields | **GPU-first** (`BatchedElementwiseF64` op=8, S70+) |
//! | [`kriging`] | Soil moisture spatial interpolation | **Integrated** (`KrigingF64`) |
//! | [`reduce`] | Seasonal aggregation statistics | **GPU** for N≥1024 (`FusedMapReduceF64`) |
//! | [`sensor_calibration`] | Batched `SoilWatch` 10 VWC calibration | **GPU-first** (`BatchedElementwiseF64` op=5, S70+) |
//! | [`stats`] | GPU OLS regression + correlation matrix | **GPU** (`stats_f64`, neuralSpring S69) |
//! | [`seasonal_pipeline`] | Full-season ET₀→Kc→WB→Yield pipeline | **GPU Stages 1-2** (ET₀ + Kc) + CPU stages 3-4 |
//! | [`atlas_stream`] | Multi-station multi-crop regional pipeline | **GPU-capable** + streaming callback |
//! | [`stream`] | `IoT` stream smoothing (sliding window) | **GPU** (`MovingWindowStats`, wetSpring) |
//! | [`richards`] | 1D Richards equation (vadose zone) | **Wired** (`pde::richards`) |
//! | [`isotherm`] | Batch isotherm fitting (biochar) | **Wired** (`nelder_mead` + `multi_start`) |
//! | [`van_genuchten`] | Batched VG θ(h) and K(h) | **GPU-first** (`BatchedElementwiseF64` ops 9-10, S79) |
//! | [`thornthwaite`] | Batched Thornthwaite monthly ET₀ | **GPU-first** (`BatchedElementwiseF64` op=11, S79) |
//! | [`gdd`] | Batched Growing Degree Days | **GPU-first** (`BatchedElementwiseF64` op=12, S79) |
//! | [`pedotransfer`] | Batched pedotransfer polynomial | **GPU-first** (`BatchedElementwiseF64` op=13, S79) |
//! | [`jackknife`] | Jackknife variance estimation | **GPU** (`JackknifeMeanGpu`, groundSpring→S71) |
//! | [`bootstrap`] | Bootstrap mean + CI | **GPU** (`BootstrapMeanGpu`, groundSpring→S71) |
//! | [`diversity`] | Alpha diversity fusion | **GPU** (`DiversityFusionGpu`, wetSpring→S70) |
//! | [`mc_et0`] | Monte Carlo ET₀ uncertainty bands | **Wired** (`norm_ppf` + parametric CI, GPU kernel available S66+) |
//! | [`evolution_gaps`] | Living roadmap of CPU→GPU gaps | Documentation only |
//!
//! # `ToadStool` Universal Precision Architecture (S68)
//!
//! All WGSL shaders are **f64 canonical** — written in f64, compiled to the
//! target precision via `compile_shader_universal(source, precision, label)`:
//!
//! | Precision | Target | When |
//! |-----------|--------|------|
//! | `F64` | Native f64 builtins | Compute-class GPUs (Titan V, A100) |
//! | `Df64` | Double-float f32-pair (~48-bit) | Consumer GPUs (RTX 4070, 1:64 f64 ratio) |
//! | `F32` | Downcast to f32 | Inference-only, low precision OK |
//! | `F16` | Downcast to f16 | Edge inference |
//!
//! `Fp64Strategy::Native` vs `Fp64Strategy::Hybrid` is selected per-device
//! by `GpuDriverProfile::fp64_strategy()` based on f64:f32 throughput ratio.
//!
//! # `ToadStool` Issues — All RESOLVED (S54+S66)
//!
//! | ID | Issue | Status |
//! |----|-------|--------|
//! | TS-001 | `pow_f64` non-integer exponents | **RESOLVED** (S54) |
//! | TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **RESOLVED** (S54) |
//! | TS-003 | `acos_simple`/`sin_simple` approximations | **RESOLVED** (S54) |
//! | TS-004 | `FusedMapReduceF64` buffer conflict N≥1024 | **RESOLVED** (S54) |
//! | P0 | GPU dispatch bind-group panic | **RESOLVED** (S66 explicit BGL) |
//!
//! # Architecture
//!
//! ```text
//! airSpring eco:: modules (CPU, validated against FAO-56)
//!        │
//!        ▼
//! airSpring gpu:: wrappers (domain-specific batched API)
//!        │
//!        ▼
//! barracuda::ops:: primitives (GPU dispatch + CPU fallback)
//!        │
//!        ▼
//! ToadStool WGSL shaders (f64 canonical, universal precision)
//!        │
//!  ┌─────┼─────┬──────┐
//!  F64   Df64  F32    F16
//! native pair  down   down
//! ```

pub mod atlas_stream;
pub mod bootstrap;
pub mod device_info;
pub mod diversity;
pub mod dual_kc;
pub mod et0;
pub mod evolution_gaps;
pub mod gdd;
pub mod hargreaves;
pub mod infiltration;
pub mod isotherm;
pub mod jackknife;
pub mod kc_climate;
pub mod local_dispatch;
pub mod kriging;
pub mod mc_et0;
pub mod pedotransfer;
pub mod reduce;
pub mod richards;
pub mod runoff;
pub mod seasonal_pipeline;
pub mod sensor_calibration;
pub mod simple_et0;
pub mod stats;
pub mod stream;
pub mod thornthwaite;
pub mod van_genuchten;
pub mod water_balance;
pub mod yield_response;
