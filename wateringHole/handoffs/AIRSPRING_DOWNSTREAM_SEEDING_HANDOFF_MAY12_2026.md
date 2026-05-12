# airSpring Downstream Seeding Sprint Handoff — May 12, 2026

**From**: airSpring (ecology / agriculture)
**To**: primalSpring (audit), all primal teams, all spring teams, downstream products (projectNUCLEUS, foundation, lithoSpore)
**Date**: May 12, 2026
**Subject**: LTEE E3 Rust validation, Tier 2 JSON output, projectNUCLEUS workload completion, GPU registry reconciliation, deep debt final sweep

---

## Current State Snapshot

| Metric | Value |
|--------|-------|
| Lib tests | **1,027** (`cargo test --features local,testutil --lib`) |
| Integration + doc tests | **316** (barracuda) |
| Forge tests | **62** (metalForge) |
| **Grand total** | **1,405** |
| Binaries | **94** (85 validation, 4 bench, 3 operational, 1 UniBin, 1 guidestone) |
| Capabilities | **46** (science + ecology + provenance + composition + infrastructure + cross-primal) |
| Deploy graphs | **7** (eco + provenance + niche + cross-primal + GPU batch + sovereign data + uncertainty) |
| GuideStone level | **L4** (targeting L5+) |
| UniBin scenarios | **10** (incl. `s_tier4_math_parity`) |
| barraCuda version | **0.4.0** (wgpu 28, Vulkan) |
| Features default | **`[]`** (Tier 4 IPC-first) |
| Clippy | **0 warnings** (IPC-only + full-feature builds) |
| `#[allow()]` in production | **0** (all use `#[expect(reason)]`) |
| `#[allow()]` in tests | **0** (evolved to `#[expect()]` in `tests/common/mod.rs`) |
| `unsafe` in production | **0** (`#![forbid(unsafe_code)]` crate-wide) |
| C dependencies | **0** (ecoBin v3.0 compliant) |
| TODO/FIXME/HACK | **0** |
| Hardcoded primal names | **0** (all use `primal_names::` constants) |
| projectNUCLEUS workloads | **6 TOMLs** (et0-validation, et0-methods, soil-physics, water-balance, atlas-pipeline, full-suite) |
| Foundation Thread 6 | **36/36 targets validated** (Agricultural Science, most target-rich thread) |
| LTEE E3 | **Python 12/12 + Rust 29/29 PASS** |
| MSRV | **1.92** (Edition 2024) |

---

## What Changed This Round

### 1. LTEE E3 Rust Validation (Complete)

`validate_ltee_fls2.rs` — **29/29 PASS**. Pure Rust reproduction of Dolgikh et al. 2025 FLS2 plant immunity sentinel analysis:
- Langmuir, Hill, and two-site binding models reproduced in Rust
- Glycosylation Kd shift (28 nM → 15 nM, sensitivity ratio 1.867)
- Soil-immune coupling model: moisture factor, Q10 temperature factor, microbial activity, flagellin exposure
- Cross-validated against Python `benchmark_ltee_fls2.json` (12 Python checks → 29 Rust checks with additional boundary and parity tests)

This closes the "Rust validation TBD" gap for LTEE E3. The soil-immune coupling model is unique to airSpring — it connects soil moisture/temperature models to plant immune activation thresholds via microbial ecology.

**For lithoSpore**: E3 data is now Tier 2 ready (Python + Rust). Could seed a new lithoSpore module (`ltee-immunity`) when the ecosystem is ready.

### 2. `--format json` on UniBin `validate` Subcommand

New `OutputFormat` enum (`text` / `json`) wired into the `validate` subcommand CLI:

```
airspring validate --format json
airspring validate --list --format json
```

`harness_to_json()` serializes `ValidationHarness` to structured JSON: suite name, pass/fail counts, per-check details (label, observed, expected, tolerance, mode). This enables projectNUCLEUS Tier 2 ingestion without text parsing.

**For all springs**: The `harness_to_json()` pattern works with the shared `ValidationHarness` struct from barraCuda. Any spring using `ValidationHarness` can add the same structured output.

### 3. projectNUCLEUS Workload Completion (1 → 6 TOMLs)

`gardens/projectNUCLEUS/workloads/airspring/` now contains 6 workload TOMLs matching the sporeGarden set:

| Workload | Command | Notes |
|----------|---------|-------|
| `airspring-et0-validation` | `validate_fao56_et0` | 75/75 Python-Rust cross-validated |
| `airspring-et0-methods` | `validate_et0_methods` | 8 ET₀ methods |
| `airspring-soil-physics` | `validate_soil_physics` | Richards, Green-Ampt, SCS-CN, Saxton-Rawls |
| `airspring-water-balance` | `validate_water_balance` | FAO-56 Ch 8 daily WB |
| `airspring-atlas-pipeline` | `validate_atlas` | 100 stations, 80 years |
| `airspring-full-suite` | `airspring validate --format json` | All scenarios, structured JSON output |

All use `${SPRINGS_ROOT:-...}` path convention with `isolation_level = "process"`.

**For projectNUCLEUS team**: airSpring is now fully routable. The `full-suite` workload produces machine-readable JSON for automated ingestion.

### 4. GPU Capability Registry Drift Fix

`capability_registry.toml` had 7 methods marked `gpu_accelerated = false` that actually have GPU paths via `gpu/simple_et0.rs`, `gpu/infiltration.rs`, and `gpu/autocorrelation.rs`:

| Method | GPU Module | Shader |
|--------|-----------|--------|
| `science.et0_makkink` | `gpu::simple_et0` | `batched_elementwise_f64.wgsl` |
| `science.et0_turc` | `gpu::simple_et0` | `batched_elementwise_f64.wgsl` |
| `science.et0_hamon` | `gpu::simple_et0` | `batched_elementwise_f64.wgsl` |
| `science.et0_blaney_criddle` | `gpu::simple_et0` | `batched_elementwise_f64.wgsl` |
| `science.green_ampt_infiltration` | `gpu::infiltration` | `brent_f64.wgsl` |
| `science.autocorrelation` | `gpu::autocorrelation` | `AutocorrelationF64` |
| `ecology.autocorrelation` | (alias) | (alias) |

**For all springs**: Check your `capability_registry.toml` GPU flags against actual GPU module implementations. Drift happens as GPU modules are added but the registry isn't updated simultaneously.

### 5. Deep Debt Final Sweep

- **Last hardcoded primal name**: `"toadstool"` in `compute_dispatch.rs` test → `primal_names::TOADSTOOL`
- **Last `#[allow()]`**: 3 instances in `tests/common/mod.rs` → `#[expect()]`
- **Comprehensive audit results**: 0 large files (>800L), 0 unsafe in production, 0 TODO/FIXME, 0 stale mocks, all deps pure Rust, `.unwrap()`/`.expect()` library-denied

### 6. Tier 2 IPC wiring (TCP round-trip probes + composition scenario)

- **`ipc::toadstool_validate`** (8 TCP round-trip lib tests): typed JSON-RPC plumbing for **`toadstool.validate`** workload pre-flight.
- **`ipc::precision_route`** (8 TCP round-trip lib tests): typed coverage for **`barracuda.precision.route`** precision advisory (`PRECISION_ROUTE` constant).
- **`methods.rs`**: **`TOADSTOOL_VALIDATE`**, **`TOADSTOOL_LIST_WORKLOADS`**, **`PRECISION_ROUTE`** added (49 centralized method constants total).
- **AG-012 resolved**: Tier 2 IPC path unblocked end-to-end in-tree (typed IPC clients exercised in tests).
- **Composition-parity** UniBin scenario extended with Tier 2 probes (`validation/scenarios/s_composition_parity.rs`).

---

## Proven Composition Patterns (for other springs)

### 1. Tier 2 Structured Validation Output

```
airspring validate --format json
```

Produces:
```json
{
  "suite": "airSpring Validation — Scenario Runner",
  "passed": 29,
  "total": 29,
  "failed": 0,
  "all_passed": true,
  "checks": [
    {
      "label": "Rust Langmuir(50nM) matches formula",
      "passed": true,
      "observed": 0.6430940329,
      "expected": 0.6430940329,
      "tolerance": 1e-12,
      "mode": "abs"
    }
  ]
}
```

### 2. projectNUCLEUS Workload Routing

Each workload TOML follows `[metadata] / [execution] / [resources] / [security]` with `type = "native"` and `${SPRINGS_ROOT:-fallback}` path convention. The `full-suite` workload uses the UniBin `--format json` for machine ingestion.

### 3. LTEE Reproduction Pattern

Python baseline generates `benchmark_*.json` + `expected_values.json` → Rust validator `include_str!`s the JSON and reproduces deterministic math independently, cross-validating against the benchmark reference values. For models requiring nonlinear fitting (scipy), the Rust validator validates invariants (R² thresholds, Kd recovery, monotonicity, boundary conditions) rather than re-fitting.

---

## Upstream Needs (Priority Order)

### P0: NestGate
`data.weather` standardization is blocked on NestGate going live. AG-008 remains open. This is the single highest-priority upstream dependency for the entire spring ecosystem.

### P1: biomeOS `composition.status` + Live NUCLEUS
GuideStone L5-L6 requires live NUCLEUS composition. airSpring has `composition.status` wired and tested structurally; live validation awaits biomeOS orchestration.

### P2: barraCuda Anderson GPU Coupling
AG-011: Anderson coupling is CPU-only. A dedicated `anderson_coupling_f64.wgsl` shader would enable GPU-first dispatch for this capability. Currently Tier C on the promotion map.

### P3: Squirrel Inference
AG-005: `inference.*` capabilities are declared but not exercised on the science path. Requires Squirrel NPU integration.

---

## Per-Primal Recommendations

| Primal | Recommendation |
|--------|---------------|
| **barraCuda** | `harness_to_json()` could be upstreamed into the shared `ValidationHarness` so all springs get `--format json` for free |
| **toadStool** | Workload dispatch should resolve `${SPRINGS_ROOT:-...}` or adopt absolute plasmidBin paths |
| **biomeOS** | Live `composition.status` and `method.register` are wired; need biomeOS live for L5+ |
| **NestGate** | Critical blocker — `data.weather` standardization |
| **skunkBat** | Deploy graph wired; audit logging ready when skunkBat comes online |
| **sweetGrass** | Provenance trio wired structurally; live validation awaits NestGate + biomeOS |

---

## For Other Springs

1. **`--format json`**: Add `OutputFormat` to your UniBin CLI and wire `harness_to_json()` for Tier 2 projectNUCLEUS ingestion
2. **GPU registry audit**: Check `capability_registry.toml` GPU flags match actual implementations
3. **LTEE baselines**: If you have Python-only LTEE reproductions, the `include_str! + ValidationHarness` pattern is proven for Rust cross-validation
4. **projectNUCLEUS workloads**: Ensure you have workload TOMLs matching your sporeGarden set
5. **`#[allow()]` → `#[expect()]`**: Modern Rust practice; catches when suppressed warnings are actually fixed

---

## Archive Review

Codebase is clean. No new debris to archive:
- `scripts/` — all still active (data downloads, benchmark helpers)
- `control/` — all Python baselines are active Phase-0 references
- `fossilRecord/` — properly catalogued superseded material
- `archive/` — 3 superseded scripts properly indexed
- Only known gap: `control/ncbi_diversity/ncbi_diversity_analysis.py` is missing (benchmark JSON works with Rust-only validator)

---

*May 12, 2026 — 94 binaries, 1,027 lib tests, 1,405 total, 46 capabilities, 7 deploy graphs, 10 UniBin scenarios, guideStone L4 (targeting L5+). LTEE E3 Python 12/12 + Rust 29/29 PASS. `--format json` for Tier 2 ingestion; Tier 2 IPC wired (`ipc::toadstool_validate`, `ipc::precision_route`, AG-012 resolved). 6 projectNUCLEUS workload TOMLs. Foundation Thread 6 complete (36/36). barraCuda 0.4.0, Tier 4 IPC-first, Edition 2024, MSRV 1.92. Zero debt. AGPL-3.0-or-later.*
