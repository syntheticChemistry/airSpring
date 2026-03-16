# AIRSPRING V084 — Deep Debt Execution + Upstream Evolution Handoff

**Date:** 2026-03-16
**From:** airSpring V084
**To:** barraCuda, toadStool, All Springs
**License:** AGPL-3.0-or-later
**Covers:** V083 → V084 (deep debt execution, code quality evolution, primal refactoring)

---

## Executive Summary

- **All 12 audit findings executed** from V083 deep audit — zero remaining debt items
- **Zero hardcoded primal names** in production code — all evolved to `primal_names::*` constants + `niche::NICHE_NAME`
- **Primal binary refactored** from 736 LOC monolith → 4 focused modules (discovery/dispatch/handlers/main)
- **`eprintln` → `tracing`** in library code (niche.rs) with structured fields
- **Python provenance standardized** across 20 control scripts (commit, date, command)
- **CI expanded**: cross-compile (aarch64 ecoBin), metalForge deny job, 11 additional validation binaries, file-size gate expanded
- **deny.toml wildcards=deny** for both barracuda and metalForge crates
- **863 tests pass**, zero new warnings, zero clippy regressions

---

## Part 1: Code Quality Evolution

### Cast Lint Strategy
- `cast_precision_loss` remains crate-level (justified: scientific code, hundreds of `n as f64` denominators)
- `cast_possible_truncation` and `cast_sign_loss` documented with reason strings; crate-level covers 91 validation binaries, library code uses per-site `#[expect()]`
- All GPU library modules verified: `.unwrap()` exclusively in `#[cfg(test)]`

### Library Tracing Evolution
- `niche.rs`: 4 `eprintln!` → `tracing::info!` / `tracing::warn!` with structured fields
  - `target: "biomeos"`, capability field, registered/total counts
- `gpu/atlas_stream/`, `gpu/device_info/`: verified `eprintln`/`println` only in test code
- `validation.rs`: `println`/`eprintln` retained by design — validation harness produces terminal output for human operators

### TODO Resolution
- `evolution_gaps.rs`: `TODO(afit)` → "async fn in dyn trait resolved (edition 2024)"
- `bench_cross_spring_evolution/pipeline.rs`: benchmark check description updated

### Primal Name Constants
8 locations evolved from hardcoded strings to constants:
- `ipc/provenance.rs`: `"neural-api"` → `primal_names::NEURAL_API`
- `validate_toadstool_dispatch.rs`: `"toadstool"` → `primal_names::TOADSTOOL`
- `validate_nucleus_modern.rs`: 5 string literals → constants
- `validate_nucleus_graphs.rs`: 4 `find_socket()` calls → constants
- `validate_nucleus_pipeline.rs`: 2 calls → constants
- `validate_nucleus.rs`: `"airspring"` → `niche::NICHE_NAME`

---

## Part 2: Validation Fidelity

### Blaney-Criddle Hardcoding → Benchmark JSON
- `validate_blaney_criddle.rs`: hardcoded daylight fraction p values (0.333, 0.222, 0.274) evolved to reads from `benchmark["monthly_daylight_pct"]["latitude_40N"]` and `["latitude_0N"]`
- FAO-24 Table 18 values now sourced from the benchmark JSON provenance chain

### Bootstrap Heuristic Documentation
- `validate_bootstrap_jackknife.rs`: mean plausibility check (4.5 ± 2.5 mm) now documented with source (Allen et al. 1998, FAO-56 Table A2 for temperate regions)

### Tolerance Sync
- `control/tolerances.py`: added `BOOTSTRAP_JACKKNIFE_KNOWN` (58th named tolerance → 58 total)
- `control/requirements.txt`: added `Requires: Python >= 3.10` pin for match types / structural pattern matching

---

## Part 3: Primal Binary Refactoring

`airspring_primal.rs` (736 LOC) → 4 focused modules:

| Module | LOC | Responsibility |
|--------|:---:|----------------|
| `discovery.rs` | 36 | Runtime primal discovery, capability-based, zero hardcoded paths |
| `dispatch.rs` | 86 | JSON-RPC method dispatch, provenance auto-recording |
| `handlers.rs` | 289 | All `handle_*` implementations (health, ecology, provenance, data, compute) |
| `main.rs` | 282 | CLI, server lifecycle, connection handling, heartbeat, metrics |

The refactoring preserves exact behavior — no functional changes. Each module has a single clear responsibility.

---

## Part 4: Infrastructure Evolution

### CI Pipeline
- **cross-compile job**: aarch64-unknown-linux-gnu cargo check (ecoBin compliance)
- **deny-forge job**: cargo-deny on metalForge/forge (was missing)
- **file-size gate**: now checks both `barracuda/src/` and `metalForge/forge/src/`
- **11 new validation binaries**: richards, biochar, long_term_wb, drought_index, bootstrap_jackknife, mc_et0, ncbi_diversity, season_wb, et0_ensemble, et0_bias, paper_chain

### deny.toml Alignment
- `barracuda/deny.toml`: `wildcards = "allow"` → `wildcards = "deny"`
- `metalForge/forge/deny.toml`: added `[graph]`, `[advisories]` db config, expanded `[licenses]` allow list, added `highlight = "all"`

### Python Provenance
20 control scripts standardized with provenance blocks (script, commit=af1eb97, date=2026-02-26, command). 9 scripts already had provenance.

---

## Part 5: Large File Assessment

| File | Total LOC | Production | Docs | Tests | Decision |
|------|:---------:|:----------:|:----:|:-----:|----------|
| `evolution_gaps.rs` | 546 | 167 | 379 | 0 | **KEEP** — living roadmap, doc is content |
| `evapotranspiration.rs` | 549 | 131 | 171 | 247 | **KEEP** — single responsibility, re-exports |
| `airspring_primal.rs` | 736→282 | 282 | 12 | 0 | **REFACTORED** → 4 modules |
| `dual_kc.rs` | 511 | 258 | 98 | 155 | **KEEP** — cohesive FAO-56 model |
| `richards.rs` | 458 | 311 | 62 | 85 | **KEEP** — single PDE solver |

---

## Part 6: barraCuda Evolution Recommendations

### Current airSpring Usage (barraCuda 0.3.5)
- **20 ops** via `BatchedElementwiseF64` (ops 0-19)
- **25 Tier A** GPU orchestrators wired
- **3 pipeline** patterns: `SeasonalPipeline`, `AtlasStream`, `MonitoredAtlasStream`
- **Pure Rust**: zero C deps, `#![forbid(unsafe_code)]`, edition 2024

### Upstream Absorption Candidates
airSpring has validated patterns that could benefit the wider ecosystem:

1. **Validation harness pattern** — `ValidationHarness` with named tolerances, `check_abs`/`check_bool`, automatic pass/fail exit codes. Could become a `barracuda::testing` module.
2. **Tolerance registry pattern** — 58 named tolerances in 4 submodules (atmospheric, soil, GPU, instrument) with justification strings. Could become `barracuda::tolerances` infrastructure.
3. **Niche self-knowledge pattern** — capability table, semantic mappings, operation dependencies, cost estimates. Other springs duplicate this; could be a shared `barracuda::niche` trait.
4. **Benchmark JSON loading** — `json_field`, `json_f64_required`, `parse_benchmark_json` helpers used by all 91 binaries. Could be a `barracuda::validation::benchmark` module.

### barraCuda Action Items
- [ ] Consider absorbing `ValidationHarness` + tolerance registry as `barracuda::testing`
- [ ] Consider `barracuda::niche::NicheCapabilities` trait for cross-spring self-knowledge
- [ ] `TensorSession` f64 support would enable fused seasonal pipelines (currently f32 only)
- [ ] `BatchedOdeRK45F64` for adaptive Richards GPU (airSpring uses fixed-step Picard)

### toadStool Action Items
- [ ] airSpring primal binary is ready for niche-graph deployment (4 deploy graphs in `graphs/`)
- [ ] 41 capabilities registered — validate routing through `biomeOS` capability registry
- [ ] Cross-compile gate (aarch64) passes — ecoBin ready for ARM deployment

### All Springs
- [ ] Adopt `primal_names` constant pattern — zero hardcoded primal strings
- [ ] Adopt `#[expect()]` with reason strings over `#[allow()]`
- [ ] Adopt `deny.toml wildcards = "deny"` (all springs)
- [ ] Python baseline provenance blocks (script, commit, date, command)

---

## Debris Status

| Category | Status |
|----------|--------|
| TODO/FIXME/HACK/XXX | **Zero** in barracuda/src/ |
| `#[allow(dead_code)]` in production | **Zero** |
| Files > 1000 LOC | **Zero** |
| .bak/.old/.tmp/.orig | **Zero** |
| __pycache__/.pyc | **Zero** |
| Hardcoded primal names | **Zero** (8 evolved to constants) |
| archive/ | 2 root + 76 handoffs (fossil record, intentional) |

---

## Metrics

| Metric | V083 | V084 |
|--------|:----:|:----:|
| Library tests | 863 | 863 |
| Validation binaries | 91 | 91 |
| Clippy (pedantic+nursery) | 0 warnings | 0 warnings |
| `#[allow()]` in production | 0 | 0 |
| Hardcoded primal names | 8 | 0 |
| Python scripts with provenance | ~50% | ~100% |
| CI validation binaries | ~10 | ~21 |
| CI deny jobs | 1 | 2 |
| Primal binary modules | 1 (736 LOC) | 4 (282+289+86+36 LOC) |
| Named tolerances | 57 Rust + 57 Python | 58 Rust + 58 Python |
