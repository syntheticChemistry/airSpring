# Primal Gaps — airSpring v0.10.0

**Date**: May 11, 2026 (post-interstadial evolution)
**Spring**: airSpring (ecology / agriculture)
**guideStone Level**: L2+ (IPC-wired, 46 capabilities, composition.status) → targeting 3+
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
| AG-001 | primalSpring | `downstream_manifest.toml` not read by airSpring | Cannot validate proto-nucleate composition; niche.rs capabilities defined independently | **In progress** — primalSpring at `springs/primalSpring/`, `airspring_guidestone` binary reads manifest directly (16/16 PASS) |
| AG-005 | Squirrel | `inference.*` not exercised in science path | airspring_cell.toml includes Squirrel but no science code calls `inference.complete` or `inference.embed` | **Open** — waiting for neuralSpring WGSL inference evolution |
| AG-006 | coralReef | Sovereign shader compile not wired | `discover_shader_compiler()` hook exists but no active usage; all GPU dispatch through barraCuda direct | **Open** — coralReef integration is roadmap |
| AG-007 | ToadStool | `compute.dispatch` returns opaque results | airSpring `compute.offload` forwards raw JSON; no typed response contract for ecology workloads | **Open** — need wire standard for domain-specific dispatch results |
| AG-008 | NestGate | `data.open_meteo_weather` not a standard NestGate method | airSpring's `data.weather` handler calls NestGate with a non-standard method name; should use capability-based weather data routing | **Open** — needs ecosystem weather data standard |
| AG-009 | petalTongue | No direct IPC wiring from airSpring | Cell graph includes petalTongue but airspring_primal has no visualization dispatch; petalTongue integration is graph-level only | **Open** — low priority; petalTongue consumes via biomeOS SSE |
| AG-010 | barraCuda | `TensorSession` / `TensorContext` not available | Seasonal GPU pipeline blocked on persistent buffer pooling; documented in `evolution_gaps.rs` | **Open** — barraCuda roadmap item |
| AG-011 | barraCuda | Anderson coupling needs new WGSL shader | `science.anderson_coupling` runs CPU-only; no upstream shader exists | **Open** — Tier C in GPU promotion map |
| AG-012 | toadStool | Live Science API not implemented | `toadstool.validate` JSON-RPC method (projectNUCLEUS `LIVE_SCIENCE_API.md`) not yet available — notebooks cannot call validation directly | **Open** — toadStool evolution item |
| AG-015 | barraCuda | barraCuda still mandatory path dep | Parity audit: should be `optional = true` with IPC-first for sovereign NUCLEUS deployment; blocks deployment where only primals are present | **Documented** — requires trait abstraction layer (MathBackend) to decouple science from barraCuda types; ecosystem-wide coordination needed |

---

## Resolved Gaps

| ID | Primal | Gap | Resolution | Date |
|----|--------|-----|------------|------|
| AG-002 | primalSpring | Path dep deprecated | Standalone manifest reader via `toml` crate — no primalspring crate dep needed | 2026-05-07 |
| AG-003 | biomeOS | `health_method` inconsistency | Aligned metalForge deploy to `health.liveness` | 2026-04-27 |
| AG-004 | biomeOS | Capability naming drift | Converged metalForge deploy to niche.rs canonical names | 2026-04-27 |
| AG-013 | projectNUCLEUS | Workload paths hardcoded to ironGate | Migrated to `${AIRSPRING_ROOT}` convention | 2026-05-07 |
| AG-014 | foundation | Thread 6 targets/workloads missing | 36 targets + 6 workloads created | 2026-05-07 |

---

## Provenance Drift (Internal)

These are not primal gaps but internal reconciliation items:

| Area | Rust Header Commit | JSON `_provenance` Commit | specs/README Commit | Status |
|------|-------------------|--------------------------|--------------------|---------| 
| Atlas | `fad2e1b` | `fad2e1b` | `fad2e1b` | **Consistent** (fixed Phase 5.18) |
| Dual Kc | `94cc51d` | `94cc51d` | `94cc51d` | **Consistent** (fixed Phase 5.18) |
| FAO-56 | `94cc51d` | `94cc51d` | `94cc51d` | Consistent |

---

## guideStone Evolution Path

```
Current:  gS Level 2+ (IPC-wired, **9 UniBin validation scenarios**, methods centralized)
Target:   gS Level 3+ (live NUCLEUS validation)
Next:     Deploy NUCLEUS from plasmidBin, validate Tier 3 experiments
```

### Prerequisites for gS Level 1 (DONE)
1. ~~Clone primalSpring beside springs/~~ — primalSpring at `springs/primalSpring/`
2. ~~Add primalspring path dependency~~ — **Deprecated**; standalone TOML reader
3. Read `downstream_manifest.toml` → validate against niche::CAPABILITIES — `airspring_guidestone` binary (16/16 PASS)
4. Create `airspring_guidestone` binary with Tier 1 (local) property checks — DONE
5. Validate P1–P5 properties without primals running (exit 2 = skip) — DONE

### Prerequisites for gS Level 2 (DONE)
6. ~~Wire Tier 2 IPC checks (`check_skip()` when primals absent)~~ — exp002 implements graceful skip (10/10 PASS)
7. ~~Validate science parity through `capability.call` vs direct Rust~~ — exp001 dispatches all 55 methods (55/55 PASS)

### Prerequisites for gS Level 3+
8. Add primalSpring as feature-gated dep (for `CompositionContext`)
9. Deploy NUCLEUS from plasmidBin, run guideStone against live primals
10. Create composition experiment crates (exp094 pattern replication)
11. Document remaining gaps → hand back

### Deep Debt Evolution (May 8 2026)
- [x] capability_registry.toml created (46 methods, sync test + cross-sync vs canonical **413**)
- [x] deny.toml promoted to workspace root (ecoBin v3.0, ring/openssl + aws-lc-sys banned)
- [x] `methods.rs` centralized constants module (46 methods, drift-proof)
- [x] **9 UniBin validation scenarios** (`validation/scenarios/`; exp001–exp003 absorbed plus expanded ScenarioRegistry coverage)
- [x] Test extraction: provenance.rs (747→496), rpc/mod.rs (650→341), seasonal_pipeline (738→539)
- [x] 3 compilation errors fixed (autobins, NestGateProvider→IPC, fhe_ntt cfg)
- [x] Missing docs resolved (DailyWeather, Station, YieldRecord, HttpResponse, DataError)
- [x] `/proc/*` paths gated behind `cfg(target_os = "linux")`
- [x] primalSpring feature-gated dep (guidestone feature)
- [x] CONTEXT.md reconciled with README.md (single source of truth, May 11)
- [x] composition.status handler wired (biomeOS v3.51 contract)
- [x] skunkBat added to niche deploy graph (9 nodes)
- [x] Zero `#[allow]` in production code (`#[expect]` with reason throughout)
- [x] benchmarks.rs refactored (810→148 + 607 bench_fns.rs, zero >800L files)
- [ ] barraCuda optional = true (ecosystem-wide, requires MathBackend trait)
- [ ] guidestone L3+ (deploy NUCLEUS from plasmidBin)

---

**This document is maintained by airSpring and consumed by primalSpring.**
**See also**: `primalSpring/docs/PRIMAL_GAPS.md` (ecosystem-wide gap registry)
**See also**: `primalSpring/docs/CROSS_SPRING_PARITY_SCORECARD.md` (parity dashboard)
