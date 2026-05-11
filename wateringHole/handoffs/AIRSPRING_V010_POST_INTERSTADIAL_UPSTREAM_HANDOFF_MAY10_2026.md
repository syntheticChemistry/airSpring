# airSpring v0.10.0 — Post-Interstadial Upstream Handoff

**Date**: May 10, 2026
**From**: airSpring (ecology / agriculture)
**To**: primalSpring, all primal teams, all spring teams
**guideStone Level**: **L4** (cross-atomic pipeline / provenance tier; targeting **L6** cross-spring pipeline with live NUCLEUS) (IPC-wired, 46 capabilities, composition.status + **`method.register`** + skunkBat IPC wired, **10** UniBin validation scenarios)

---

## What Was Done (May 10 Interstadial Pass)

### Composition & Security Wiring
- **`composition.status` handler** — airSpring now responds to `composition.status` JSON-RPC calls per biomeOS v3.51 contract, reporting `active_users`, `primal_health` (ratio across provenance trio, NestGate, ToadStool, skunkBat), and `resource_pressure`.
- **skunkBat integration** — `SKUNKBAT` constant added to `primal_names.rs`. Niche deploy graph expanded from 8 → 9 nodes with optional `skunkbat` node (provides `security.monitor`, `security.quarantine`, `security.audit`, `security.health`).
- **`composition.status`** registered in `capability_registry.toml` and `niche::CAPABILITIES`. **`method.register`** added as the additional capability (46 total) for biomeOS dynamic method registration.

### Cross-Sync Validation
- **`capability_cross_sync.rs`** — new integration test validates airSpring's shared-domain methods (`health.*`, `capability.*`, `compute.*`) align with primalSpring's canonical 413. Documents 9 "extending" methods (`provenance.*`, `primal.*`, `data.*`, `composition.*`, `method.*`) that are airSpring-local but tracked for upstream registration.

### Tier 4 barracuda rewiring (May 11)

- **`barracuda` optional dependency** — `optional = true` with **`local`** feature **default on**; the library compiles **without** the barraCuda source tree when building **`--no-default-features`**.
- **`gpu` module** — feature-gated; pure-Rust paths remain available when GPU stack is off.
- **`math.rs`** — dual-path dispatch: barraCuda-backed routes when enabled, **pure-Rust fallbacks** otherwise.
- **`ipc/barracuda_route.rs`** — IPC forwarding for barraCuda-backed operations when the in-tree crate is absent.

### IPC Wiring (May 11)
- **`method.register` IPC module** — `ipc/method_register.rs` sends `method.register` RPC to biomeOS at startup, registering all 46 niche capabilities dynamically. Dispatch handler processes inbound `method.register` calls.
- **skunkBat audit module** — `ipc/skunkbat.rs` emits `security.audit_log` events for certification, startup, and ad-hoc audit. Discovery via standard socket path (`/tmp/skunkbat.sock` or `SKUNKBAT_SOCKET`).
- **10 UniBin validation scenarios** — `validation/scenarios/` expanded (incl. **`s_tier4_math_parity`**): `fao56-et0`, `et0-methods`, `soil-physics`, `water-balance`, `atlas-pipeline`, `paper-chain` added alongside existing `local-science-parity`, `composition-parity`, `full-regression`. All wired to `airspring validate --scenario <id>`.
- **plasmidBin release binaries** — `airspring` (3.0M) and `airspring_primal` (2.4M) built, stripped, and deployed to `infra/plasmidBin/springs/`.
- **foundation seeded** — 36/36 thread06_ag targets validated, provenance manifest + sweetGrass braid published. All 6 workloads migrated to UniBin `airspring validate` pattern.

### Deep Debt Resolution
- **benchmarks.rs refactored** (810L → 197L + 634L `bench_fns.rs`) — zero files >800 lines remain.
- **`guidestone` feature drift fixed** — `certification/composition.rs` provides feature-gated `CompositionContext` integration. The feature now has a code anchor instead of being declared-but-unused.
- **Hardcoded API URLs evolved** — Open-Meteo (`OPEN_METEO_ARCHIVE_URL`) and USDA NASS (`NASS_API_BASE`) endpoints are now env-overridable for sovereign NestGate routing.
- **CONTEXT.md reconciled** with README.md as single source of truth (May 10). All stale dates and counts across 14+ documents aligned.
- **`#[allow]` → `#[expect]`** migration complete — zero `#[allow]` in production code. All 3 remaining instances migrated with explicit `reason`.

### Quality Gates (all green)
- `cargo build` — clean
- `cargo fmt --check` — clean
- `cargo clippy --workspace --all-targets` — zero warnings
- `cargo test --workspace --lib --tests` — 1,011 lib + 316 integration PASS
- `cargo test --test capability_cross_sync` — 3/3 PASS
- `cargo check --features guidestone` — clean

---

## Current State Snapshot

| Metric | Value |
|--------|-------|
| Lib tests | 1,011 |
| Integration tests | 316 |
| Forge tests | 62 |
| **Total tests** | **1,389** |
| Binaries | 93 |
| Capabilities | 46 |
| Deploy graphs | 4 (incl. skunkBat, 9-node niche) |
| Experiments | 90 (all PASS) |
| guideStone | **L4** (targeting **L6**; IPC-wired, **10** UniBin scenarios) |
| CPU speedup | 14.3× (24/24 parity) |
| Coverage | 90.56% |
| C dependencies | 0 |
| `#[allow()]` in production | 0 |
| `unsafe` in production | 0 |

---

## For Primal Teams

### Methods airSpring Extends Beyond Canonical 413

These 10 methods exist in airSpring's local `capability_registry.toml` but not in primalSpring's canonical registry. They are documented and tracked for upstream registration:

| Method | Domain | Description |
|--------|--------|-------------|
| `provenance.begin` | provenance | Start experiment session (rhizoCrypt DAG) |
| `provenance.record` | provenance | Record experiment step |
| `provenance.complete` | provenance | Complete experiment, seal provenance |
| `provenance.status` | provenance | Query session status |
| `primal.forward` | primal | Forward RPC to another primal |
| `primal.discover` | primal | Discover available primals |
| `data.cross_spring_weather` | data | Cross-spring weather data exchange |
| `data.weather` | data | Weather data via NestGate |
| `composition.status` | composition | biomeOS v3.51 health/status |
| `method.register` | method | biomeOS v3.51 dynamic method registration |

**Recommendation**: Register `composition.status`, `method.register`, and `provenance.*` in the canonical registry — these are ecosystem-wide patterns, not spring-specific.

### What Each Primal Team Should Know

| Primal | What airSpring Learned | Action |
|--------|------------------------|--------|
| **barraCuda** | 24 CPU benchmarks at parity, 21 GPU modules validated. **Tier 4 (2026-05-11):** `optional = true` + `local` default; `math.rs` dual-path; `ipc/barracuda_route.rs`; `--no-default-features` without barraCuda tree — **AG-015 resolved**. | Further trait polish for IPC-only deployments if desired. |
| **toadStool** | `compute.offload` works for ecology workloads. `compute.dispatch` returns opaque JSON (AG-007). `toadstool.validate` not yet available (AG-012). | Typed response contracts for domain-specific dispatch. Live Science API for notebook-driven validation. |
| **NestGate** | `data.weather` handler works. `data.open_meteo_weather` is non-standard (AG-008). airSpring URLs now env-overridable for NestGate routing. | Ecosystem weather data standard method name. |
| **coralReef** | `discover_shader_compiler()` hook exists but no active usage (AG-006). All GPU dispatch goes through barraCuda direct. | Low priority — sovereign shader compile when coralReef matures. |
| **Squirrel** | 10 MCP tools registered in primal dispatch. `inference.*` not exercised in science path (AG-005). | Wait for neuralSpring WGSL inference evolution. |
| **biomeOS** | `composition.status` and **`method.register`** wired (46-cap registry); socket discovery via `biomeos::discover_*` works well. | Further Neural API ergonomics as biomeOS evolves. |
| **skunkBat** | Added to niche deploy graph (order 3, optional). `SKUNKBAT` primal name constant. Security capabilities declared but not exercised in science path. | When Phase 3 ships, we get audit forwarding to rhizoCrypt DAG + sweetGrass braid. |
| **bearDog + songbird** | Sovereign TLS via Songbird working. Required nodes in niche deploy. | No action needed. |

### Active Gaps (for primalSpring gap registry)

| ID | Primal | Gap | Status |
|----|--------|-----|--------|
| AG-005 | Squirrel | `inference.*` not exercised in science path | Open |
| AG-006 | coralReef | Sovereign shader compile not wired | Open |
| AG-007 | ToadStool | `compute.dispatch` returns opaque results | Open |
| AG-008 | NestGate | Non-standard weather method name | Open |
| AG-009 | petalTongue | No direct IPC wiring | Open (low priority) |
| AG-010 | barraCuda | `TensorSession`/`TensorContext` not available | Open |
| AG-011 | barraCuda | Anderson coupling needs WGSL shader | Open |
| AG-012 | toadStool | Live Science API not implemented | Open |
| AG-015 | barraCuda | ~~Still mandatory path dep~~ | **Resolved** (Tier 4 optional + fallbacks, 2026-05-11) |

---

## For Spring Teams

### Patterns Worth Absorbing

1. **`capability_cross_sync.rs` test pattern** — validates shared-domain methods against primalSpring canonical 413. Distinguishes spring-local domains (exempt), aligned domains (must match), and extending domains (tracked). Other springs should replicate this.

2. **Env-overridable endpoints** — `OPEN_METEO_ARCHIVE_URL`, `NASS_API_BASE` allow sovereign routing without code changes. Any spring with hardcoded external URLs should adopt this pattern.

3. **`#[expect(reason)]` throughout** — zero `#[allow()]` in production. All lint suppressions have explicit reasons. `#[expect]` warns when the suppressed lint no longer fires (dead code cleanup for free).

4. **`composition.status` wiring** — per biomeOS v3.51 contract. Report `active_users`, `primal_health` (ratio of healthy optional primals), `resource_pressure`. Each spring should wire this.

5. **Feature-gated `CompositionContext`** — `certification/composition.rs` behind `guidestone` feature provides typed composition validation when primalSpring is available. Other springs can replicate for their L3+ certification.

6. **Smart refactoring** — split by semantic boundary (function bodies vs assembly), not just line count. `bench_fns.rs` (bodies) + `benchmarks.rs` (assembly) vs arbitrary 400/400 split.

### NUCLEUS Composition Patterns

airSpring's validated composition patterns for NUCLEUS deployment via biomeOS Neural API:

- **Registration**: `lifecycle.register` → domain `capability.register` → per-method registration with `operation_dependencies` and `cost_estimates`.
- **Discovery**: 5-tier escalation (Songbird → Neural API → UDS → biomeOS scan → env vars).
- **Provenance**: Optional trio (rhizoCrypt + loamSpine + sweetGrass) via `capability.call` on DAG domain, with graceful degradation when trio is absent.
- **Compute dispatch**: `compute.offload` → toadStool, with `IpcError::is_recoverable` for circuit-breaker resilience.
- **Transport**: `Transport` enum (Unix + TCP) for ecoBin-compliant platform-agnostic IPC.

### Graph node note

`ecology.experiment` appears on the niche deploy graph `airspring` node but is NOT in the 46-capability registration set. Upstream orchestrators should reconcile graph node capability lists with `niche.rs` / `capability_registry.toml` sources.

---

## What's Next for airSpring

1. ~~**Tier 4 barraCuda rewiring**~~ — **Done (2026-05-11):** `optional = true` with `local` (default on); `gpu` gated; `math.rs` dual-path; `ipc/barracuda_route.rs`; build `--no-default-features` without barraCuda source tree.
2. **guideStone L6 / live NUCLEUS** — deploy NUCLEUS from plasmidBin; certification engine now spans **L0–L6** (L5: `composition.status`, `method.register`, `compute.dispatch`; L6: deploy graphs, capability registry, scenario registry). Validate against live primals via `CompositionContext` and cross-spring pipeline.
3. **Phase 4.7 Penny Irrigation** — sovereign scheduling on consumer hardware
4. **Paper queue** — 5 pending papers awaiting field data (Dong lab 2026)

---

## Downstream Absorption (for projectNUCLEUS / foundation)

- **6 toadStool workloads** available in `foundation/workloads/thread06_ag/` and `projectNUCLEUS/workloads/airspring/`
- **36 foundation targets** for Thread 6 (Agricultural Science) validated
- **4 deploy graphs** ready for biomeOS orchestration
- airSpring IPC-composes cleanly with NestGate/storage and compute paths
- All 46 capabilities routable through biomeOS Neural API

---

*This handoff is maintained by airSpring and consumed by primalSpring.*
*See also: `docs/PRIMAL_GAPS.md`, `docs/PRIMAL_PROOF_IPC_MAPPING.md`, `capability_registry.toml`*
