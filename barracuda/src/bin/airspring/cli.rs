// SPDX-License-Identifier: AGPL-3.0-or-later

//! `UniBin` CLI — clap subcommands for the eukaryotic airspring binary.

use clap::{Parser, Subcommand, ValueEnum};

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

/// Output format for machine-readable ingestion.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text (default).
    #[default]
    Text,
    /// Structured JSON for Tier 2 projectNUCLEUS ingestion.
    Json,
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
        /// Output format: text (default) or json (for projectNUCLEUS Tier 2 ingestion).
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
        /// Write provenance artifacts (results.json + provenance.toml) to this directory.
        /// Used by projectFOUNDATION workloads for Thread 5+6 capture.
        #[arg(long, value_name = "DIR")]
        provenance_dir: Option<String>,
    },
    /// Start the JSON-RPC 2.0 IPC server (cell membrane).
    Serve,
    /// Show niche health and capability discovery status.
    Status,
    /// Show version information.
    Version,
}
