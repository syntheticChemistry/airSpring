// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inter-Primal Communication — biomeOS capability routing.
//!
//! All IPC goes through biomeOS `capability.call` over Unix sockets.
//! Zero compile-time coupling to external primal crates.
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`provenance`] | Provenance trio (rhizoCrypt + loamSpine + sweetGrass) |

pub mod provenance;
