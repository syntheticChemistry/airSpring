// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inter-Primal Communication — biomeOS capability routing.
//!
//! All IPC goes through biomeOS `capability.call` over Unix sockets.
//! Zero compile-time coupling to external primal crates.
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`method_register`] | biomeOS v3.51 `method.register` for dynamic semantic routing |
//! | [`skunkbat`] | Audit event emission via `security.audit_log` (JH-5 forwarding) |
//! | [`mcp`] | MCP tool definitions for Squirrel AI integration (10 ecology tools) |
//! | [`neural_bridge`] | biomeOS v3.67+ observatory: routing weights, instrumented dispatch |
//! | [`provenance`] | Provenance trio (rhizoCrypt + loamSpine + sweetGrass) |
//! | [`resilience`] | Circuit breaker + retry with exponential backoff |
//! | [`timeseries`] | Cross-spring time series exchange (`ecoPrimals/time-series/v1`) |
//! | [`toadstool_validate`] | Tier 2: `toadstool.validate` workload pre-flight |
//! | [`precision_route`] | Tier 2: `precision.route` GPU precision advisory |
//! | [`nestgate_data`] | NestGate CAS: `content.store`, `content.get`, `storage.status` |
//! | [`squirrel_inference`] | Squirrel `inference.embed`, `inference.complete`, `inference.models` |

pub mod barracuda_route;
pub mod compute_dispatch;
pub mod dispatch_outcome;
pub mod mcp;
pub mod method_register;
pub mod nestgate_data;
pub mod neural_bridge;
pub mod precision_route;
pub mod provenance;
pub mod resilience;
pub mod skunkbat;
pub mod squirrel_inference;
pub mod timeseries;
pub mod toadstool_validate;

pub use dispatch_outcome::DispatchOutcome;
