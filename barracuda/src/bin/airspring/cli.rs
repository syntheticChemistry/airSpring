// SPDX-License-Identifier: AGPL-3.0-or-later

//! `UniBin` CLI — clap subcommands for the eukaryotic airspring binary.

use clap::{Parser, Subcommand};

/// airSpring `UniBin` — ecological & agricultural science niche.
#[derive(Parser)]
#[command(
    name = "airspring",
    version,
    about = "Eukaryotic ecology niche — certification, validation, and IPC server"
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Run niche certification (absorbed guidestone, L0-L4).
    Certify {
        /// Maximum certification layer (0-4, default 4).
        #[arg(long, value_name = "N")]
        layer: Option<u8>,
        /// Run only Layer 0 (bare structural validation, no primals needed).
        #[arg(long, default_value_t = false)]
        bare: bool,
    },
    /// Run validation scenarios (absorbed experiments).
    Validate {
        /// Filter by track (e.g. science-dispatch, composition, foundation, provenance).
        #[arg(long)]
        track: Option<String>,
        /// Run a single scenario by ID.
        #[arg(long)]
        scenario: Option<String>,
        /// Filter by tier: rust (structural), live (IPC), both.
        #[arg(long)]
        tier: Option<String>,
        /// List all available scenarios without running them.
        #[arg(long, default_value_t = false)]
        list: bool,
    },
    /// Start the JSON-RPC 2.0 IPC server (cell membrane).
    Serve,
    /// Show niche health and capability discovery status.
    Status,
    /// Show version information.
    Version,
}
