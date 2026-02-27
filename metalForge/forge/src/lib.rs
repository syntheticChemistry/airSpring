// SPDX-License-Identifier: AGPL-3.0-or-later
//! airSpring metalForge — cross-system compute dispatch.
//!
//! Discovers CPU, GPU, and NPU substrates at runtime and routes
//! ecological/agricultural workloads to the best available hardware.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │ airSpring eco workloads                                       │
//! │   ET₀ batch, water balance, Richards PDE, yield response      │
//! │   Crop stress classifier (NPU), irrigation decision (NPU)    │
//! └──────────────────────────┬────────────────────────────────────┘
//!                            │
//! ┌──────────────────────────▼────────────────────────────────────┐
//! │ metalForge dispatch (capability-based routing)                │
//! │   GPU: f64 batch compute (ET₀, WB, Richards, Monte Carlo)    │
//! │   NPU: int8 classifiers (crop stress, irrigation, anomaly)   │
//! │   CPU: validation, I/O, sequential control logic             │
//! └──────────────────────────┬────────────────────────────────────┘
//!                            │
//! ┌──────────────────────────▼────────────────────────────────────┐
//! │ probe: wgpu (GPU), /proc (CPU), /dev/akida* (NPU)           │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Cross-Spring Pattern
//!
//! Same substrate/dispatch architecture as `wetSpring/metalForge`.
//! Each spring defines its own workloads; the dispatch logic is identical.

pub mod dispatch;
pub mod inventory;
pub mod neural;
pub mod probe;
pub mod substrate;
pub mod workloads;
