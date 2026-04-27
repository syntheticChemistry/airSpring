# Primal Gaps — airSpring v0.10.0

**Date**: April 27, 2026
**Spring**: airSpring (ecology / agriculture)
**guideStone Level**: 0 → targeting 1
**License**: AGPL-3.0-or-later

---

## Purpose

This document tracks gaps discovered during airSpring's evolution from
validated Rust science (L2) toward primal composition (L3–L5). Each gap
is a missing capability, wire contract issue, or primal behavior that
blocks or complicates composition. Gaps are handed back to primalSpring
for ecosystem-wide refinement.

Format follows wetSpring/hotSpring `PRIMAL_GAPS.md` pattern.

---

## Active Gaps

| ID | Primal | Gap | Impact | Status |
|----|--------|-----|--------|--------|
| AG-001 | primalSpring | `downstream_manifest.toml` not read by airSpring | Cannot validate proto-nucleate composition; niche.rs capabilities defined independently | **Open** — need to clone primalSpring and wire manifest reader |
| AG-002 | primalSpring | No `primalspring` crate dependency | Cannot use `primalspring::composition`, `discover_by_capability()`, `checksums` for guideStone | **Open** — blocked on primalSpring clone |
| AG-003 | biomeOS | `health_method` inconsistency across deploy graphs | `airspring_deploy.toml` used `health.check`, `airspring_cell.toml` uses `health.liveness`, niche dispatches both but canonical is `health.liveness` | **Fixed** — aligned deploy to `health.liveness` |
| AG-004 | biomeOS | Capability naming drift across deploy surfaces | metalForge deploy used `ecology.et0.penman_monteith` while niche.rs uses `science.et0_fao56`; biomeOS must resolve both or one must converge | **Fixed** — aligned metalForge deploy to niche.rs canonical names |
| AG-005 | Squirrel | `inference.*` not exercised in science path | airspring_cell.toml includes Squirrel but no science code calls `inference.complete` or `inference.embed` | **Open** — waiting for neuralSpring WGSL inference evolution |
| AG-006 | coralReef | Sovereign shader compile not wired | `discover_shader_compiler()` hook exists but no active usage; all GPU dispatch through barraCuda direct | **Open** — coralReef integration is roadmap |
| AG-007 | ToadStool | `compute.dispatch` returns opaque results | airSpring `compute.offload` forwards raw JSON; no typed response contract for ecology workloads | **Open** — need wire standard for domain-specific dispatch results |
| AG-008 | NestGate | `data.open_meteo_weather` not a standard NestGate method | airSpring's `data.weather` handler calls NestGate with a non-standard method name; should use capability-based weather data routing | **Open** — needs ecosystem weather data standard |
| AG-009 | petalTongue | No direct IPC wiring from airSpring | Cell graph includes petalTongue but airspring_primal has no visualization dispatch; petalTongue integration is graph-level only | **Open** — low priority; petalTongue consumes via biomeOS SSE |
| AG-010 | barraCuda | `TensorSession` / `TensorContext` not available | Seasonal GPU pipeline blocked on persistent buffer pooling; documented in `evolution_gaps.rs` | **Open** — barraCuda roadmap item |
| AG-011 | barraCuda | Anderson coupling needs new WGSL shader | `science.anderson_coupling` runs CPU-only; no upstream shader exists | **Open** — Tier C in GPU promotion map |

---

## Resolved Gaps

| ID | Primal | Gap | Resolution | Date |
|----|--------|-----|------------|------|
| AG-003 | biomeOS | `health_method` inconsistency | Aligned metalForge deploy to `health.liveness` | 2026-04-27 |
| AG-004 | biomeOS | Capability naming drift | Converged metalForge deploy to niche.rs canonical names | 2026-04-27 |

---

## Provenance Drift (Internal)

These are not primal gaps but internal reconciliation items:

| Area | Rust Header Commit | JSON `_provenance` Commit | specs/README Commit | Status |
|------|-------------------|--------------------------|--------------------|---------| 
| Atlas | `e651409` | `fad2e1b` | `cb59873` | **Needs reconciliation** |
| Dual Kc | `3afc229` | `94cc51d` | — | **Needs reconciliation** — scripts also differ |
| FAO-56 | `94cc51d` | `94cc51d` | `94cc51d` | Consistent |

---

## guideStone Evolution Path

```
Current:  gS Level 0 (paths fixed, science validated, no guideStone binary)
Target:   gS Level 1 (guideStone scaffold, bare property checks)
Blocked:  AG-001, AG-002 (need primalSpring clone + dependency)
```

### Prerequisites for gS Level 1
1. Clone primalSpring beside springs/
2. Add `primalspring` path dependency (optional, feature-gated)
3. Read `downstream_manifest.toml` → validate against niche::CAPABILITIES
4. Create `airspring_guidestone` binary with Tier 1 (local) property checks
5. Validate P1–P5 properties without primals running (exit 2 = skip)

### Prerequisites for gS Level 2
6. Wire Tier 2 IPC checks (`check_skip()` when primals absent)
7. Validate science parity through `capability.call` vs direct Rust

### Prerequisites for gS Level 3+
8. Deploy NUCLEUS from plasmidBin
9. Run guideStone against live primals
10. Document remaining gaps → hand back

---

**This document is maintained by airSpring and consumed by primalSpring.**
**See also**: `primalSpring/docs/PRIMAL_GAPS.md` (ecosystem-wide gap registry)
