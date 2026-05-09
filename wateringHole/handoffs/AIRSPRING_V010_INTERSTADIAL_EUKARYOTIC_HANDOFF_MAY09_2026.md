# airSpring V0.10.0 — Interstadial Eukaryotic Evolution Handoff

**Date**: May 9, 2026
**From**: airSpring v0.10.0
**To**: primalSpring, upstream primal teams, sibling springs
**Triggered by**: Interstadial Primordial Extinction Wave (primalSpring v0.9.25 Phase 60+)

## What Changed

### UniBin Eukaryotic Evolution

airSpring now has a single `airspring` UniBin binary with 5 subcommands:
- `airspring certify [--layer N] [--bare]` — L0-L4 layered certification
- `airspring validate [--track T] [--scenario S] [--tier T] [--list]` — scenario runner
- `airspring serve` — JSON-RPC 2.0 IPC server
- `airspring status` — niche health and discovery status
- `airspring version` — version info

### Certification Organelle (`certification/`)

Absorbed from `airspring_guidestone` binary into library module:
- **Layer 0 (Bare)**: Manifest structural validation — identity, fragments, dependencies, capabilities
- **Layer 1 (Discovery)**: Primal discovery via biomeOS socket directory
- **Layer 2 (Health)**: health.liveness for toadStool, beardog, songbird, nestgate
- **Layer 3 (Parity)**: Science dispatch produces results (7 representative methods)
- **Layer 4 (Cross-Atomic)**: Provenance trio roundtrip (begin → record → complete)

### Validation Scenarios (`validation/scenarios/`)

Absorbed from prokaryotic experiment crates:

| Scenario ID | Track | Tier | Provenance Crate |
|-------------|-------|------|-----------------|
| `local-science-parity` | science-dispatch | rust | exp001_local_science_parity |
| `composition-parity` | composition | both | exp002_composition_parity |
| `foundation-targets` | foundation | rust | exp003_foundation_target_validation |

Registry follows primalSpring pattern: `ScenarioMeta`, `ScenarioRegistry`, `Tier`, `Track`.

### Quality Gates

- **`aws-lc-sys` + `aws-lc-rs`** banned in workspace-root and barracuda deny.toml
- **Zero bare `#[allow()]`** — all suppression with `reason`
- **Zero TODO/FIXME/HACK/DEBT** in active code
- **1001/1007 lib tests pass** (6 pre-existing failures: `data::open_meteo` + `data::usda_nass` require Songbird HTTP transport, not available in test)
- **Zero new clippy warnings** on UniBin binary
- **`fossilRecord/`** created with provenance for absorbed experiment crates
- **`PRIMAL_PROOF_IPC_MAPPING.md`** — 44 library calls → JSON-RPC mapping

## Current guideStone Level: L2 → targeting L4+

| Level | Status |
|-------|--------|
| L0 (Bare) | DONE — manifest validation |
| L1 (Discovery) | DONE — primal socket discovery |
| L2 (IPC-wired) | DONE — 3 composition crates, graceful skip |
| L3 (NUCLEUS validated) | PARTIAL — certification layers 1-4, needs live deployment |
| L4 (Primal proof) | BLOCKED — requires plasmidBin deployment for Tier 2/3 scenarios |
| L5 (Certified) | BLOCKED — requires L4 + full NUCLEUS |

## What Blocks L3+

1. **Live NUCLEUS deployment** — need plasmidBin binaries deployed and reachable
2. **CompositionContext integration** — primalSpring v0.9.25 `guidestone` feature flag is declared but not exercised by UniBin (current discovery uses native `biomeos::discover_*` functions)
3. **6 pre-existing test failures** — `data::open_meteo` and `data::usda_nass` need Songbird HTTP transport (standalone-http feature is a dead end — ureq removed during sovereignty evolution)

## For primalSpring

- airSpring's `certification/` and `validation/scenarios/` follow the v0.9.25 pattern
- `ScenarioRegistry` is local (not importing from primalSpring) — appropriate for niche-specific scenarios
- `capability_registry.toml` has 44 methods — CI cross-sync test against canonical 389 pending
- primalSpring dep is feature-gated at v0.9.25 (`guidestone` feature)

## For Sibling Springs

airSpring's eukaryotic patterns are available for reference:
- UniBin CLI: `barracuda/src/bin/airspring/{main.rs, cli.rs}`
- Certification: `barracuda/src/certification/{mod.rs, bare.rs, health.rs}`
- Scenarios: `barracuda/src/validation/scenarios/{mod.rs, registry.rs, s_*.rs}`
- IPC mapping: `docs/PRIMAL_PROOF_IPC_MAPPING.md`
- Fossil record: `fossilRecord/experiments_prokaryotic_may2026/`

## Active Gaps for Upstream Teams

| Team | Gap | Priority |
|------|-----|----------|
| **primalSpring** | CI cross-sync test (44 vs 389 capability registry) | Medium |
| **barraCuda** | `aws-lc-sys` ban now ecosystem-wide (deny.toml updated) | Info |
| **NestGate** | Weather data IPC returns empty when NestGate unavailable — no error | Low |
| **toadStool** | `compute.offload` tested structurally but not live | Medium |
| **Squirrel** | Not yet in airSpring composition — blocks AI-enhanced science | Low |

## Composition Metrics

- **44** capabilities registered (science + ecology + provenance + health + data)
- **3** validation scenarios absorbed
- **4** certification layers implemented
- **1** UniBin binary (replaces 2 separate: airspring_primal + airspring_guidestone)
- **92** total declared binaries (91 validation/bench + 1 UniBin)
- **0** deprecated IPC patterns (PrimalClient, AtomicHarness, etc.)
- **0** bare `#[allow()]` suppressions
