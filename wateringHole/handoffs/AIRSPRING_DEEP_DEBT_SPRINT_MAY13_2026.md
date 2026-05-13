# airSpring — Deep Debt Resolution + Evolution Sprint Handoff

**Date**: May 13, 2026
**From**: airSpring (ecology / agriculture)
**For**: primalSpring (coordination), upstream primals
**License**: AGPL-3.0-or-later

---

## Executive Summary

Full deep debt audit executed per primalSpring directive. **Zero actionable debt remaining** in core library. Fixes this sprint: 9 `#[must_use]` clippy pedantic attrs, last `/tmp/` hardcoded path eliminated, `unused_must_use` in binary resolved. AG-005 (Squirrel science path) fully resolved — `inference.embed/complete/models` wired through `dispatch_science`. All audit dimensions green.

---

## Deep Debt Audit Results

| Category | Finding |
|----------|---------|
| TODO/FIXME/HACK/XXX | **0** |
| `unsafe` blocks (non-test) | **0** (`#![forbid(unsafe_code)]`) |
| `unsafe fn` | **0** |
| Production mocks | **0** (all `Mock` types in `#[cfg(test)]`) |
| Files >800 LOC | **0** (largest: 775 LOC) |
| Hardcoded primal paths | **0** (last `/tmp/` eliminated this sprint) |
| `#[allow(` in production | **0** (all `#[expect()]` with reasons) |
| `todo!()` / `unimplemented!()` | **0** |
| `.unwrap()` in lib | **0** |
| Clippy default | **0** warnings |
| Clippy pedantic+nursery | **0** warnings (9 `#[must_use]` fixed this sprint) |
| External C deps | **0** (pure Rust: serde, clap, thiserror, toml, tracing) |
| Edition | **2024** (Rust 1.92+) |

---

## Fixes Applied This Sprint

1. **9 `#[must_use]` attributes** added to satisfy clippy pedantic:
   - `ipc/barracuda_route.rs`: `try_forward`
   - `ipc/provenance.rs`: `resolve_neural_api_transport`
   - `ipc/skunkbat.rs`: `discover`, `audit_log`, `audit_certification`, `audit_startup`
   - `math.rs`: `mean`, `pearson_r`, `std_dev`

2. **Last `/tmp/` hardcoded path eliminated** in `data/provider.rs`:
   - `SongbirdTransport::discover` had a `/tmp/songbird-{FAMILY_ID}.sock` fallback
   - Replaced with `biomeos::discover_primal_socket(SONGBIRD)` — standard biomeOS discovery

---

## Audit Questions — Answers

### Python baselines for barraCuda CPU (Rust) parity

| Metric | Value |
|--------|-------|
| `bench_cpu_vs_python` algorithms | **25/25** parity (14.3× geometric mean speedup) |
| Python control checks | **1,284** from `control/` scripts (60 papers) |
| Cross-validation harness | **75/75** Python↔Rust match (tol=1e-5) |

**Missing from the 25-bench speed parity table** (validated elsewhere but not in the CPU-vs-Python timing harness): Turc, Hamon, SPI drought index, Monte Carlo ET₀, bootstrap/jackknife, kriging, autocorrelation, CN+GA coupled.

### Industry-standard GPU benchmarks

| Standard | Status |
|----------|--------|
| **Kokkos/Cabana** | **Not started** — Tier 1 performance reference gap. groundSpring V74 has 3.5×-2669× dispatch overhead data for reference. |
| **LAMMPS** | Referenced as methodology comparison — no integrated harness. |
| **SciPy** | Numerical alignment reference (not GPU benchmark). |
| **Galaxy/QIIME2** | Ecosystem comparison — not integrated. |
| **Internal GPU coverage** | 21/21 CPU-GPU parity, 46/46 validate_gpu_math, 25 Tier A upstream ops |

### What's not implemented / tested

| Gap | Status |
|-----|--------|
| AG-005: Squirrel science path | **Resolved** — `inference.embed/complete/models` wired through `dispatch_science` with 7 new tests |
| AG-006: coralReef shader compile | Open |
| AG-007: compute.dispatch typing | Open |
| AG-010: TensorSession | Open (barraCuda roadmap) |
| AG-011: Anderson WGSL shader | Open (Tier C) |
| L5 certification | Structural complete, blocked on live primals |
| L6 cross-spring pipeline | Blocked on L5 |
| Kokkos Tier 1 harness | Not started |

### Unreviewed papers

- **#6, #7** (Tier 1): Awaiting Dong lab field data (2026)
- **#16** (Tier 3): Awaiting field data
- **#23, #24** (Tier 4): Future

### Datasets to examine

- **NOAA CDO / OpenWeatherMap**: Open with API key — weather data expansion
- **Dong lab**: Multi-sensor IoT + lysimeter (awaiting 2026 access)
- **NCBI 16S**: ~50 GB metagenome pipeline for soil microbiome coupling
- **USDA SCAN**: Soil moisture validation stations (partially integrated)

---

## Metrics Snapshot

| Metric | Value |
|--------|-------|
| Lib tests | **1,057** |
| Total tests | **1,435** (1,057 lib + 316 integration + 62 forge) |
| IPC modules | **13** |
| Method constants | **61** |
| Capabilities | **49** |
| CPU vs Python parity | **25/25** |
| Python control checks | **1,284** |
| UniBin validation scenarios | **10** |
| guideStone | **L4** (structural L5) |
| Binaries | **94** |
| Edition | **2024** (Rust 1.92+) |

---

**This document consumed by primalSpring.** See also: `docs/PRIMAL_GAPS.md`
