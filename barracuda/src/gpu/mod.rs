//! GPU acceleration layer — airSpring ↔ `ToadStool`/`BarraCUDA` bridge.
//!
//! This module provides domain-specific wrappers around `ToadStool` GPU primitives
//! for precision agriculture workloads. All functions have CPU fallbacks and
//! can be used without a GPU device.
//!
//! # Modules
//!
//! | Module | Purpose | Backend |
//! |--------|---------|---------|
//! | [`et0`] | Batched FAO-56 ET₀ for N station-days | CPU (GPU blocked on `pow_f64`) |
//! | [`water_balance`] | Batched season simulation | CPU (GPU needs orchestrator) |
//! | [`kriging`] | Soil moisture spatial interpolation | CPU (`KrigingInterpolator` ↔ `KrigingF64`) |
//! | [`reduce`] | Seasonal aggregation statistics | CPU/GPU (`SeasonalReducer` ↔ `FusedMapReduceF64`) |
//! | [`evolution_gaps`] | Living roadmap of CPU→GPU gaps | Documentation only |
//!
//! # `ToadStool` Issues (for next handoff)
//!
//! 1. **`pow_f64` non-integer exponents** (`batched_elementwise_f64.wgsl:138`):
//!    Returns 0.0 for exponent 5.26. Fix: `exp_f64(exp * log_f64(base))` when base > 0.
//!    **Blocks**: GPU ET₀ (atmospheric pressure calculation).
//!
//! 2. **`acos_simple` approximation**: 3-term polynomial — fine for latitude ±70° but
//!    loses accuracy near ±1. Wire `math_f64.wgsl` full `acos_f64` when available.
//!
//! 3. **`sin_simple`** Taylor series: Adequate for ET₀ but not general use.
//!
//! 4. **No Rust `ops::batched_elementwise_f64` module**: Shader exists but no
//!    orchestrator to create pipelines, pack buffers, and dispatch workgroups.
//!
//! 5. **`FusedMapReduceF64` buffer conflict** (TS-004): GPU dispatch panics for
//!    N ≥ 1024 due to conflicting buffer usages in the partials pipeline. The
//!    `SeasonalReducer` works correctly for N < 1024 (CPU path).
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
//! `ToadStool` WGSL shaders (f64 precision on GPU)
//! ```

pub mod et0;
pub mod evolution_gaps;
pub mod kriging;
pub mod reduce;
pub mod water_balance;
