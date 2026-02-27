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
//! | [`et0`] | Batched FAO-56 ET₀ for `N` station-days | **GPU-first** (`BatchedElementwiseF64`) |
//! | [`water_balance`] | Batched season simulation + GPU step | **GPU-step** + CPU season |
//! | [`dual_kc`] | Batched dual Kc (`Ke` + `ETc`) for M fields | **CPU** (Tier B → GPU pending) |
//! | [`kriging`] | Soil moisture spatial interpolation | **Integrated** (`KrigingF64`) |
//! | [`reduce`] | Seasonal aggregation statistics | **GPU** for N≥1024 (`FusedMapReduceF64`) |
//! | [`stream`] | `IoT` stream smoothing (sliding window) | **GPU** (`MovingWindowStats`, wetSpring) |
//! | [`richards`] | 1D Richards equation (vadose zone) | **Wired** (`pde::richards`) |
//! | [`isotherm`] | Batch isotherm fitting (biochar) | **Wired** (`nelder_mead` + `multi_start`) |
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

pub mod dual_kc;
pub mod et0;
pub mod evolution_gaps;
pub mod isotherm;
pub mod kriging;
pub mod mc_et0;
pub mod reduce;
pub mod richards;
pub mod stream;
pub mod water_balance;
