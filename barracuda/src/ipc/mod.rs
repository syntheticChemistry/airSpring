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
//! | [`provenance`] | Provenance trio (rhizoCrypt + loamSpine + sweetGrass) |
//! | [`resilience`] | Circuit breaker + retry with exponential backoff |
//! | [`timeseries`] | Cross-spring time series exchange (`ecoPrimals/time-series/v1`) |

pub mod compute_dispatch;
pub mod dispatch_outcome;
pub mod mcp;
pub mod method_register;
pub mod provenance;
pub mod resilience;
pub mod skunkbat;
pub mod timeseries;

pub use dispatch_outcome::DispatchOutcome;
