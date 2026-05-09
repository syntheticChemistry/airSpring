// SPDX-License-Identifier: AGPL-3.0-or-later
//! Data provider abstraction for ecological data acquisition.
//!
//! # Architecture: Sovereign Transport
//!
//! airSpring data providers follow a trait-first design with sovereign transport:
//!
//! | Tier | Transport | TLS | Status |
//! |------|-----------|-----|--------|
//! | **Sovereign** | Songbird `network.http_request` | Pure Rust TLS 1.3 via `BearDog` | Active |
//! | **NUCLEUS** | `capability.call` → `NestGate` | Sovereign (content-addressed cache) | Planned |
//!
//! When Tower Atomic is running (`BearDog` + Songbird), HTTPS routes through
//! Songbird's pure-Rust TLS 1.3 stack, which delegates crypto to `BearDog` via
//! JSON-RPC — zero C dependencies in the TLS path.
//!
//! Discovery: check for Songbird socket → if present, use `network.http_request`
//! capability.
//!
//! # Cross-Spring Provenance
//!
//! Provider APIs mirror `NestGate`'s `OpenMeteoLiveProvider`, `UsdaNassLiveProvider`,
//! and `NCBILiveProvider` — same endpoints, same parameter semantics. The data
//! returned is byte-identical; only the transport differs.

pub mod open_meteo;
pub mod provider;
pub mod usda_nass;
pub mod weather;
