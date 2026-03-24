# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.10.x (current) | Yes |

## Security Model

airSpring is a scientific validation Spring — it does not handle user
authentication, network-facing services, or sensitive data directly. Its
security posture derives from the ecoPrimals sovereign stack:

- **Pure Rust**: `#![forbid(unsafe_code)]` across all workspace crates.
  Zero C dependencies in application code (ecoBin compliant).
- **cargo-deny**: Continuous advisory scanning via RustSec database,
  license compliance, and source provenance checks.
- **No vendor lock-in**: Zero proprietary dependencies. All computation
  is sovereign (runs on your hardware, no cloud calls).
- **IPC isolation**: JSON-RPC 2.0 over Unix domain sockets with
  capability-based discovery. No shared memory, no global state.

## Reporting a Vulnerability

If you discover a security issue:

1. **Do not** open a public issue
2. Contact the ecoPrimals maintainers via the repository's security
   advisory feature (GitHub → Security → Advisories → New draft)
3. Include: affected component, reproduction steps, potential impact
4. Expected response time: 72 hours for acknowledgment

## Dependencies

All dependencies are Pure Rust and audited via `cargo-deny`. The dependency
tree is reviewed on every PR via CI. Transitive `-sys` crates (from `wgpu`
GPU HAL) are infrastructure-level and do not process untrusted input.

## Data Provenance

All datasets used in validation are from public repositories (Open-Meteo ERA5,
USDA NASS, AmeriFlux, NCBI BioProject) with documented accession numbers in
`specs/DATA_PROVENANCE.md` and structured `_provenance` fields in all 59
benchmark JSON files.
