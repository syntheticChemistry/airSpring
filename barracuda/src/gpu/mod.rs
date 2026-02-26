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
//! | [`richards`] | 1D Richards equation (vadose zone) | **Tier B** (`pde::richards`) |
//! | [`isotherm`] | Batch isotherm fitting (biochar) | **Tier B** (`nelder_mead` + `multi_start`) |
//! | [`mc_et0`] | Monte Carlo ET₀ uncertainty bands | **CPU** (GPU kernel available, blocked by S60-S65 regression) |
//! | [`evolution_gaps`] | Living roadmap of CPU→GPU gaps | Documentation only |
//!
//! # `ToadStool` Issues — All RESOLVED
//!
//! All four issues identified during Phase 2 were fixed in `ToadStool` commit
//! `0c477306` (February 16, 2026 unified handoff):
//!
//! | ID | Issue | Status |
//! |----|-------|--------|
//! | TS-001 | `pow_f64` non-integer exponents | **RESOLVED** |
//! | TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **RESOLVED** |
//! | TS-003 | `acos_simple`/`sin_simple` approximations | **RESOLVED** |
//! | TS-004 | `FusedMapReduceF64` buffer conflict N≥1024 | **RESOLVED** |
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
//! ToadStool WGSL shaders (f64 precision on GPU)
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
