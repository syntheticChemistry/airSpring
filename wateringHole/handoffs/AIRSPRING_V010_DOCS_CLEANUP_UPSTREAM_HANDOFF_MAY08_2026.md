# airSpring v0.10.0 — Docs Cleanup, Primal Evolution Review, and Upstream Handoff

**Date**: May 8, 2026
**From**: airSpring (ecology / agriculture)
**For**: primalSpring, barraCuda, toadStool, NestGate, Squirrel, projectNUCLEUS, foundation, all spring teams
**guideStone Level**: 2 (IPC-wired) → targeting 3+

---

## What This Handoff Covers

Post deep-debt evolution, airSpring completed a full docs cleanup, number reconciliation across 12 files, wateringHole archive pass, primal evolution review, and upstream gap handback. This document synthesizes what we learned and what other teams need to know.

---

## 1. airSpring Current State (May 8, 2026)

| Metric | Value |
|--------|-------|
| Experiments | **90** (87 numbered + exp001/002/003 composition crates) |
| Rust tests | **1,364** (986 lib + 316 integration + 62 forge) |
| Python baselines | **1,284/1,284** (60 papers reproduced) |
| IPC capabilities | **44** (centralized in `methods.rs`, CI sync-tested) |
| Notebooks | **25** (20 paper baseline + 5 sporePrint summary) |
| guideStone | **Level 2** (IPC-wired, composition experiments) |
| Line coverage | **90.56%** (gated at 90%) |
| CPU speedup | **14.3×** geometric mean vs Python (24/24 parity) |
| C dependencies | **Zero** (ecoBin v3.0 compliant) |

---

## 2. Primal Wiring Inventory

### Primals airSpring Discovers at Runtime

| Primal | Discovery Method | Status |
|--------|-----------------|--------|
| **biomeOS** | `BIOMEOS_ORCHESTRATOR_SOCKET` env / socket scan | **Active** — registration target |
| **ToadStool** | `AIRSPRING_COMPUTE_PRIMAL` env / `toadstool` / `compute` capability | **Active** — compute offload |
| **NestGate** | `AIRSPRING_DATA_PRIMAL` env / `nestgate` / `storage` capability | **Active** — weather data, caching |
| **neural-api** | Provenance trio `capability.call` chain | **Active** — DAG/commit/provenance |
| **BearDog** | Socket scan (NUCLEUS Tower) | **Detected** — not directly called |
| **Songbird** | Socket scan (NUCLEUS Tower) | **Detected** — not directly called |
| **coralReef** | `discover_shader_compiler()` hook | **Exists** — not called in production |
| **Squirrel** | `discover_inference_primal()` hook | **Exists** — not called in production |
| **petalTongue** | `discover_visualization_primal()` hook | **Exists** — not called in production |

### Capabilities airSpring Provides

44 methods across 7 namespaces: `science.*` (17), `ecology.*` (17), `provenance.*` (3), `primal.*` (2), `health.*` (2), `capability.*` (1), `data.*` (2). Full list in `capability_registry.toml` and `barracuda/src/methods.rs`.

### Capabilities airSpring Consumes

- **ToadStool**: `compute.offload` → `compute.{op}` (raw JSON forward)
- **NestGate**: `capability.call` with `operation: "weather.daily"` for cross-spring weather
- **neural-api**: `capability.call` for provenance trio (DAG domain health probes)

---

## 3. Active Gaps for Upstream Teams

### For primalSpring

- **AG-001**: airSpring reads `downstream_manifest.toml` via standalone TOML reader (no primalspring path dep). `airspring_guidestone` binary validates 16/16 properties. primalSpring should confirm manifest contract is stable.
- **AG-015**: barraCuda is still a mandatory path dep. Ecosystem needs `MathBackend` trait abstraction to enable `optional = true`. This blocks sovereign NUCLEUS deployment where only primals are present (no Rust compilation).

### For barraCuda

- **AG-010**: `TensorSession` / `TensorContext` not available. Seasonal GPU pipeline blocked on persistent buffer pooling. Documented in `evolution_gaps.rs`.
- **AG-011**: Anderson coupling (`science.anderson_coupling`) runs CPU-only. No upstream WGSL shader exists. Tier C in GPU promotion map.
- **AG-015**: Making barraCuda optional requires `MathBackend` trait that abstracts `barracuda::ops::*` calls. airSpring has 25 Tier A GPU modules that all route through barraCuda types.

### For toadStool

- **AG-007**: `compute.dispatch` returns opaque JSON. airSpring's `compute.offload` forwards raw results. Need typed response contract for ecology workloads (at minimum: result value, precision tier, dispatch latency).
- **AG-012**: Live Science API (`toadstool.validate` JSON-RPC) not yet available. Blocks notebook Tier 2/3 evolution where notebooks call validation directly.

### For NestGate

- **AG-008**: `data.open_meteo_weather` is not a standard NestGate method. airSpring's `data.weather` handler calls NestGate with `operation: "weather.daily"`, which is non-standard. Ecosystem needs weather data routing standard.

### For Squirrel / neuralSpring

- **AG-005**: `inference.*` capabilities not exercised in science path. The composition includes Squirrel but no science code invokes it. Waiting for neuralSpring WGSL inference evolution.

### For coralReef

- **AG-006**: Sovereign shader compile not wired. `discover_shader_compiler()` exists but no active usage. All GPU dispatch goes through barraCuda direct.

---

## 4. Composition Patterns Learned

### capability-based IPC Routing (NestGate example)

When the deep debt evolution replaced the non-existent `NestGateProvider` type, we implemented capability-based IPC routing:

```rust
let transport = rpc::resolve_transport(primal_names::NESTGATE)?;
let result = rpc::send_to(&transport, "capability.call", &params)?;
```

This pattern gracefully degrades — if NestGate is absent, the handler returns an error JSON instead of panicking. All primal interactions should follow this pattern.

### Readiness Truthfulness

airSpring's `health.readiness` always returns `ready: true` even when subsystems (provenance trio, NestGate, ToadStool) are unavailable. This is intentional — science dispatch works without primals — but may not match k8s-style readiness semantics where "not ready" means "don't route traffic." Springs should document their readiness contract explicitly.

### Method String Centralization

`barracuda/src/methods.rs` provides 44 `pub const` strings as the single source of truth for all capability method names. This eliminated 4 drift sources (niche.rs, primal_science dispatch, ipc/mcp.rs tool mapping, capability_registry.toml). Other springs should consider the same pattern.

### Test Extraction for Large Files

Extracting `#[cfg(test)]` blocks into `_tests.rs` sibling modules halved the LOC of three files while keeping tests fully functional. The pattern:
1. Create `{module}_tests.rs` with `#![expect(clippy::unwrap_used)]`
2. Add `use super::*;` for access
3. Add `#[cfg(test)] mod tests;` in the parent module

---

## 5. For Downstream Systems (projectNUCLEUS, foundation, sporeGarden)

### projectNUCLEUS

- airSpring has **6 workload TOMLs** in `projectNUCLEUS/workloads/airspring/` using `${AIRSPRING_ROOT}` portable paths
- airSpring is ready for NUCLEUS deployment validation once plasmidBin includes the airspring binary
- The `airspring_primal` binary serves 44 JSON-RPC capabilities and registers with biomeOS on startup

### foundation

- airSpring validates against **36 targets** from `foundation/data/targets/thread06_ag_targets.toml`
- exp003 reads these targets and dispatches matching science methods (4/4 PASS)
- 6 workloads created in `foundation/workloads/thread06_ag/`

### For Other Springs (patterns to absorb)

1. **`methods.rs` pattern**: Centralize all capability method strings in one module. Drift-proof.
2. **`capability_registry.toml`**: 44-method TOML at workspace root, CI sync-tested against niche.rs.
3. **Composition experiment crates**: Standalone `experiments/exp00*` crates with `[workspace]` table to opt out of parent workspace. Pattern: exp001 (local parity), exp002 (composition with graceful skip), exp003 (foundation targets).
4. **Paper baseline notebooks**: 20 `.ipynb` in `notebooks/papers/` following `PAPER_NOTEBOOK_PATTERN.md`. First spring done.
5. **deny.toml at workspace root**: ecoBin v3.0, `ring`/`openssl` banned, applies to all crates.

---

## 6. Docs Cleanup Summary

| File | Changes |
|------|---------|
| `README.md` | Capability count **46** (incl. `method.register`; registry + docs aligned), 943→986 lib, 87→90 experiments |
| `CHANGELOG.md` | Added `[Unreleased]` section covering May 2026 work |
| `CONTROL_EXPERIMENT_STATUS.md` | Updated header (90 exp, 986 lib, 1,364 total, 44 caps) and footer |
| `experiments/README.md` | Added composition crates table, fixed test breakdown counts |
| `specs/README.md` | Updated date, counts, capabilities |
| `specs/PAPER_REVIEW_QUEUE.md` | Updated header counts |
| `specs/CROSS_SPRING_EVOLUTION.md` | Updated header counts |
| `sporeprint/validation-summary.md` | Fixed 87→90 in notebook 03 and workload table |
| `whitePaper/baseCamp/README.md` | Updated counts, capabilities, experiments |
| `wateringHole/README.md` | Archived Apr 27 handoff, fixed date on May 07, removed broken links |
| `docs/PRIMAL_GAPS.md` | Reconciled gS L2 checklist with actual state |

### Archive Actions

- **Archived**: `AIRSPRING_V010_DEEP_DEBT_EVOLUTION_HANDOFF_APR27_2026.md` (superseded by May 8)
- **Removed**: 2 broken cross-spring doc links (`AIRSPRING_COMPOSITION_GUIDANCE.md`, `SPRING_EVOLUTION_ISSUES.md`)
- **Cleanup**: Zero TODO/FIXME/HACK in active code. Zero `.bak/.old/.tmp` files. Zero `__pycache__`. Zero orphaned binaries.

---

## 7. What Blocks gS Level 3+

1. **Deploy NUCLEUS from plasmidBin** — airspring binary not yet in plasmidBin
2. **Run guideStone against live primals** — exp002 Tier 2/3 currently skip when primals absent
3. **barraCuda optional** — AG-015, requires ecosystem-wide `MathBackend` trait

---

*airSpring v0.10.0 — AGPL-3.0-or-later*
