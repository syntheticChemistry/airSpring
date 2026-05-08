# AIRSPRING_V010_DEEP_DEBT_EVOLUTION_HANDOFF_APR27_2026

**Date**: April 27, 2026
**From**: airSpring v0.10.0 deep debt evolution
**For**: primalSpring, barraCuda, biomeOS, all spring teams
**License**: AGPL-3.0-or-later

---

## Summary

airSpring completed a comprehensive deep debt execution pass covering code
quality, validation fidelity, ecosystem alignment, and primal composition
readiness. This handoff documents changes, patterns, and gaps for upstream
absorption.

## Changes Executed

### 1. Capability Naming Convergence

**Pattern**: `niche.rs` is now the single source of truth for capability names.

| Surface | Before | After |
|---------|--------|-------|
| `niche::CAPABILITIES` | `science.et0_fao56` (canonical) | Unchanged — canonical |
| `metalForge/deploy/airspring_deploy.toml` | `ecology.et0.penman_monteith` (incompatible) | `science.et0_fao56` (aligned) |
| `plasmidBin/cells/airspring_cell.toml` | 4 capabilities, `by_capability = "weather"` | 14 capabilities, `by_capability = "ecology"` |
| `metalForge/deploy` health_method | `health.check` | `health.liveness` (matches dispatch) |

**For biomeOS**: `discovery.resolve` was listed in the deploy TOML but never
implemented. It has been removed. If biomeOS requires this capability, please
define the wire contract and we will implement it.

**For all springs**: The pattern of niche.rs as canonical source → all deploy
surfaces derive from it eliminates drift. Consider adopting this: one Rust
constant array (`CAPABILITIES`) that deploy TOMLs and cell graphs reference.

### 2. Tolerance Registry Evolution (58 → 60)

Two new named tolerances:
- `ATLAS_ANNUAL_ET0` (1.0 mm/day) — multi-station ERA5 reanalysis cross-validation
- `ATLAS_YIELD_RATIO` (0.001) — Stewart yield equation 3-decimal precision

Zero inline magic numbers remain in validation code. The registry claim
"tolerances are never hardcoded inline" is now literally true.

**For all springs**: The `Tolerance` struct pattern (name + abs_tol + rel_tol +
justification, centralized in `tolerances/` submodules, mirrored in Python
`control/tolerances.py`) is mature and worth adopting ecosystem-wide.

### 3. Dispatch Completeness (44/44)

`science.timeseries` and `ecology.timeseries` were the only capabilities in
`niche::CAPABILITIES` without a dispatch path. Now routed through
`ipc::timeseries::handle_timeseries`.

**For biomeOS**: All 44 registered capabilities are routable. A biomeOS
`capability.call` to any airSpring capability will receive a response (not
MethodNotFound).

### 4. Provenance Drift Fixed

| Experiment | Rust Header | JSON Authority | Fix |
|-----------|-------------|----------------|-----|
| Atlas | `e651409` → `fad2e1b` | `fad2e1b` | Aligned |
| Dual Kc | `3afc229` → `94cc51d` | `94cc51d` | Aligned |

**For all springs**: The pattern of JSON `_provenance` as ground truth (generated
by Python) with Rust `//! Provenance:` headers derived from it prevents drift.
Consider adding a CI check that diffs the two.

### 5. Large File Refactoring

| File | Before | After | Method |
|------|--------|-------|--------|
| `validate_gpu_rewire_benchmark.rs` | 829 | 45 + 774 (support module) | Bin-local submodule extraction |
| `bench_cpu_vs_python/benchmarks.rs` | 804 | 668 | `entry!` macro for registration |

**For all springs**: The bin-local submodule pattern (`mod support;` in a
`src/bin/foo.rs` that loads `src/bin/foo_support.rs`) is clean for large
validation binaries that can't easily extract to lib code.

### 6. CI Toolchain Alignment

`dtolnay/rust-toolchain@stable` → `@master` with `toolchain: "1.92"` across
all CI jobs. Now matches `rust-toolchain.toml` exactly.

### 7. `#[expect()]` Audit

All 120+ `#[expect()]` attributes verified still-fulfilled via
`RUSTFLAGS='-Dunfulfilled_lint_expectations' cargo clippy --lib`. Zero stale
expects. The Edition 2024 `#[expect(reason)]` pattern is working as designed.

## Gaps Discovered (docs/PRIMAL_GAPS.md)

| ID | Primal | Gap |
|----|--------|-----|
| AG-001 | primalSpring | `downstream_manifest.toml` not read by airSpring |
| AG-002 | primalSpring | No `primalspring` crate dependency (needed for guideStone) |
| AG-005 | Squirrel | `inference.*` not exercised in science path |
| AG-006 | coralReef | Sovereign shader compile not wired |
| AG-007 | ToadStool | `compute.dispatch` returns opaque results — no typed ecology contract |
| AG-008 | NestGate | `data.open_meteo_weather` not a standard method |
| AG-009 | petalTongue | No direct IPC wiring (graph-level only) |
| AG-010 | barraCuda | `TensorSession` / `TensorContext` not available |
| AG-011 | barraCuda | Anderson coupling needs new WGSL shader |

### For primalSpring
- AG-001/AG-002 block guideStone Level 1. airSpring needs `primalspring` as a
  path dependency and must read its `downstream_manifest.toml` entry.
- Please confirm: does our `niche::CAPABILITIES` set match what the manifest expects?

### For barraCuda
- AG-010: Seasonal GPU pipeline blocked on persistent buffer pooling (`TensorContext`).
  When available, `gpu::seasonal_pipeline` can eliminate per-step buffer allocation.
- AG-011: `science.anderson_coupling` runs CPU-only. A WGSL shader for the
  Pielou→Anderson disorder chain would enable GPU acceleration.

### For ToadStool
- AG-007: `compute.dispatch` returns generic JSON. A typed response contract
  for ecology workloads (ET₀ batch → `{et0_values: [f64], latency_us: u64}`)
  would eliminate runtime parsing.

### For NestGate
- AG-008: `data.open_meteo_weather` is not a standard NestGate method. Should
  weather data routing go through a capability-based pattern
  (`data.fetch_by_schema("ecoPrimals/weather/v1")`)?

### For Squirrel / neuralSpring
- AG-005: airSpring's cell graph includes Squirrel but no science code calls
  `inference.*`. When neuralSpring matures WGSL shader ML, airSpring will wire
  crop stress classification and ecological prediction through `inference.complete`.

## Composition Patterns for NUCLEUS Deployment

### What Works

1. **Three-tier discovery** (env → named socket → capability probe) is solid.
   `biomeos::discover_primal_by_capability()` finds any primal exposing the
   right domain.

2. **Graceful degradation** in provenance trio: science dispatch succeeds
   without provenance primals running. `is_available()` check → skip recording.

3. **Heartbeat thread**: `lifecycle.status` sent every 30s to orchestrator.
   Includes capability count, composition status, uptime.

4. **MCP tool bridge**: 10 ecology tools discoverable by Squirrel via
   `tools/list` + `tools/call` → `dispatch_science`.

### What Needs Evolution

1. **Family-aware discovery**: The v0.9.17 handoff mentions
   `{capability}-{family}.sock` naming. airSpring's `biomeos::discover_primal_socket`
   does not yet use family-scoped socket names. Needs upstream pattern.

2. **BTSP handshake**: Not implemented. airSpring connects to primals via
   plain UDS. When BTSP enforcement is live, airSpring will need the
   `BEARDOG_FAMILY_SEED` relay pattern from sourDough.

3. **guideStone binary**: Does not exist yet. Blocked on primalSpring dependency.
   Target: three-tier (Tier 1 local, Tier 2 IPC-skip, Tier 3 full NUCLEUS).

4. **Graph execution**: airSpring's deploy graphs are validated offline
   (`validate_biome_graph`), but not executed through biomeOS `graph.execute`.
   Need live graph deployment testing.

## Current guideStone Status

```
Level 0: Paths fixed, science validated (L2), no guideStone binary
→ Level 1: guideStone scaffold, bare property checks
→ Level 2: IPC-wired checks with check_skip()
→ Level 3: Full NUCLEUS deployment from plasmidBin
```

Blocked on: primalSpring clone + dependency (AG-001, AG-002).

## Cross-Spring Patterns Worth Absorbing

| Pattern | Origin | Description |
|---------|--------|-------------|
| `niche.rs` canonical capabilities | airSpring | Single Rust array → all deploy surfaces derive |
| `Tolerance` registry with Python mirror | airSpring | Named, justified, centralized, CI-gated |
| Bin-local support module | airSpring | `mod support;` for large validation binaries |
| Provenance JSON as authority | airSpring | Python `_provenance` → Rust header derived |
| `#[expect(reason)]` everywhere | airSpring | Zero `#[allow()]` in production, Edition 2024 |
| 60-tolerance registry | airSpring | Every science threshold named and justified |

## Files Changed in This Pass

| File | Change |
|------|--------|
| `barracuda/src/bin/validate_paper_chain.rs` | unwrap→expect |
| `barracuda/src/bin/validate_dispatch_experiment.rs` | unwrap→expect (2 sites) |
| `barracuda/src/bin/bench_airspring_gpu.rs` | unwrap→expect (4 sites) |
| `barracuda/src/tolerances/soil.rs` | +2 tolerances |
| `barracuda/src/tolerances/mod.rs` | Registry array + count assertions updated |
| `barracuda/src/bin/validate_atlas.rs` | Inline tolerances → registry |
| `metalForge/deploy/airspring_deploy.toml` | Capability naming converged |
| `barracuda/src/primal_science/mod.rs` | +science.timeseries dispatch |
| `barracuda/src/bin/validate_gpu_rewire_benchmark.rs` | Refactored (829→45) |
| `barracuda/src/bin/validate_gpu_rewire_support.rs` | New (774 lines, support module) |
| `barracuda/src/bin/bench_cpu_vs_python/benchmarks.rs` | Refactored (804→668) |
| `.github/workflows/ci.yml` | Toolchain 1.92 pinned |
| `deny.toml` (root) | Annotated as superseded |
| `barracuda/src/bin/validate_atlas.rs` | Provenance commit aligned |
| `barracuda/src/bin/validate_dual_kc.rs` | Provenance commit aligned |
| `specs/README.md` | Baseline lineage table corrected |
| `specs/TOLERANCE_REGISTRY.md` | Count 58→60 |
| `docs/PRIMAL_GAPS.md` | **New** — 11 gaps tracked |
| `infra/plasmidBin/cells/airspring_cell.toml` | Capabilities expanded, by_capability→ecology |

---

**Next**: Clone primalSpring, add dependency, build guideStone scaffold (gS Level 0→1).
**License**: AGPL-3.0-or-later
