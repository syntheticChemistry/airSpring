# airSpring Ecosystem Evolution Handoff — May 11, 2026

**From**: airSpring (ecology / agriculture)
**To**: primalSpring, all primal teams, all spring teams
**Date**: May 11, 2026
**Subject**: Lessons learned, composition patterns, upstream priorities, and next-round needs

---

## The Journey: Python → Rust (UniBin) → Primal (NUCLEUS Composition)

airSpring validates peer-reviewed agricultural and environmental science. It started
as Python baselines reproducing published equations (FAO-56, van Genuchten, Stewart,
SCS-CN, Green-Ampt, etc.), then cross-validated those against pure Rust implementations,
then evolved through GPU dispatch, mixed hardware, and finally primal composition via
NUCLEUS. This is the first spring to complete the full L0→L4 certification path and
implement Tier 4 IPC-first rewiring.

```
Phase 0   1,284 Python checks against digitized paper benchmarks
  ↓
Phase 1   1,011 Rust lib tests + 316 integration (14.3× CPU speedup, 24/24 parity)
  ↓
Phase 2   25 Tier A GPU modules, 21/21 CPU-GPU parity, 767+ WGSL shaders consumed
  ↓
Phase 3   Titan V live, AKD1000 NPU, metalForge mixed hardware (5 substrates)
  ↓
Phase 4   46 capabilities, 4 deploy graphs, JSON-RPC science via biomeOS Neural API
  ↓
Phase 5   UniBin eukaryotic evolution, certification L0–L6, 10 validation scenarios
  ↓
Tier 4   barracuda optional=true, pure-Rust fallbacks, IPC-first dispatch
```

**Key insight**: The Python→Rust→GPU→Primal progression is the pattern. Each spring
should follow this path. Python validates the science. Rust proves it. GPU accelerates it.
Primal composition makes it sovereign and composable.

---

## HIGH PRIORITY: NestGate Not Live

**NestGate is the single biggest blocker for the next evolution round.** airSpring has:

- `data.weather` handler ready for NestGate-mediated weather data
- `OPEN_METEO_ARCHIVE_URL` and `NASS_API_BASE` env-overridable for sovereign routing
- Provider architecture (`data/provider.rs`) with 3-tier discovery
- NCBI 16S pipeline designed (`whitePaper/baseCamp/ncbi_16s_coupling.md`)

**Without NestGate live**, airSpring hits Open-Meteo and USDA directly over HTTP.
This bypasses content-addressed caching, audit logging, and sovereign data governance.
Every spring with external data dependencies (weather, genomic, spectral) is blocked
on NestGate for production-grade sovereign data flows.

**Action for NestGate team**: Ship Unix socket IPC with weather + NCBI provider
capabilities. airSpring will be the first consumer for validation.

---

## What airSpring Learned About Composition

### Pattern 1: Capability Registration via `method.register`

airSpring registers all 46 capabilities with biomeOS at startup via `method.register`
IPC. This enables dynamic discovery — biomeOS knows what airSpring can do without
static configuration. Every spring and primal should implement `method.register`.

```
startup → discover biomeOS socket → method.register(46 capabilities) → ready
```

### Pattern 2: Dual-Path Dispatch (Tier 4)

Making `barraCuda` optional required:
1. `optional = true` in Cargo.toml with a `local` feature (default on)
2. Feature-gated imports: `#[cfg(feature = "local")]` on all `barracuda::` imports
3. `math.rs` with pure-Rust fallbacks for core primitives (`mean`, `pearson_r`, `std_dev`)
4. `ipc/barracuda_route.rs` for IPC forwarding when library is absent
5. Local shims for types (`Tolerance`, error variants)

**The result**: `cargo build --no-default-features` compiles without the barraCuda
source tree. Science correctness is preserved via pure-Rust paths. GPU dispatch
re-enables when the `local` feature is on.

**Pattern for other springs**: Any spring with a barraCuda dependency should follow
this model. The math must work without GPU. GPU is an accelerator, not a requirement.

### Pattern 3: 5-Tier Socket Discovery

```
1. Songbird relay (sovereign NAT traversal)
2. biomeOS Neural API (capability routing)
3. Unix domain socket (direct /tmp/{primal}.sock)
4. biomeOS orchestrator scan
5. Environment variable fallback (BIOMEOS_SOCKET, TOADSTOOL_SOCKET, etc.)
```

Every IPC call in airSpring follows this escalation. No hardcoded socket paths
in production code. No hardcoded primal names — all via `primal_names::*` constants.

### Pattern 4: Graceful Degradation

airSpring runs fully functional without ANY primals running. Every IPC call has:
- `IpcError::is_recoverable()` for circuit-breaker logic
- Timeout with meaningful fallback (pure-Rust compute, skip provenance, etc.)
- `check_skip()` in certification when primals absent (exit code 2 = skip, not fail)

**This is critical for development and testing.** A spring that fails when biomeOS
is down is useless for iteration.

### Pattern 5: `composition.status` (biomeOS v3.51)

airSpring responds to `composition.status` JSON-RPC calls reporting:
- `active_users` (count from recent science RPC calls)
- `primal_health` (ratio of healthy optional primals: provenance trio, NestGate, etc.)
- `resource_pressure` (memory/CPU heuristic)

Every spring should implement this. It gives biomeOS real-time health visibility
into the composition.

### Pattern 6: Deploy Graph + Niche Architecture

airSpring defines 4 TOML deploy graphs:
1. `airspring_eco_pipeline.toml` — weather → ET₀ → WB → yield
2. `airspring_provenance_pipeline.toml` — session → science → dehydrate → commit
3. `airspring_niche_deploy.toml` — full niche (9 nodes incl. skunkBat)
4. `cross_primal_soil_microbiome.toml` — airSpring θ(t) → wetSpring diversity

The niche self-knowledge module (`niche.rs`) declares all capabilities, deploy
graphs, and primal dependencies. The `capability_registry.toml` is the single
source of truth, CI-tested against `niche.rs` constants AND cross-synced against
primalSpring's canonical 413.

---

## NUCLEUS Deployment via Neural API from biomeOS

airSpring's validated deployment pattern for NUCLEUS via biomeOS:

### Registration Flow
```
lifecycle.register → domain capability.register → per-method registration
  with operation_dependencies and cost_estimates
```

### Compute Dispatch Flow
```
client → biomeOS (Neural API) → capability routing → airSpring primal
  → science compute (pure Rust or barraCuda GPU)
  → provenance (optional trio: rhizoCrypt DAG + loamSpine + sweetGrass braid)
  → response (JSON-RPC 2.0)
```

### Cross-Primal Pipeline Flow
```
biomeOS graph executor reads TOML deploy graph
  → topological sort (Kahn's algorithm, validated acyclic)
  → prerequisite checks (NestGate health, ToadStool health)
  → stage execution: fetch_weather → compute_et0 → water_balance → yield
  → each stage: capability.call → route to owning primal
  → provenance chain across stages
```

### What Works Today
- 46 JSON-RPC science methods via `airspring_primal` binary
- Tower + Node Atomic detection (BearDog + Songbird + ToadStool)
- 7 primals discovered in ecosystem
- Cross-primal forwarding (`primal.forward`, `primal.discover`)
- Provenance trio integration with graceful degradation
- `Transport` enum (Unix + TCP) for platform-agnostic IPC

### What Needs Live NUCLEUS (L5→L6)
- Live `composition.status` polling from biomeOS
- Live `method.register` handshake (currently validated against mock)
- Live `compute.dispatch` to toadStool with real GPU payloads
- Cross-spring pipeline with live NestGate weather data
- plasmidBin binary deployment → biomeOS starts airSpring from release binary

---

## For Each Primal Team

| Primal | What airSpring Learned | Priority Action |
|--------|------------------------|-----------------|
| **barraCuda** | 24 CPU benchmarks at parity, 21 GPU modules, Tier 4 done. `TensorSession` (AG-010) and Anderson WGSL (AG-011) still needed for advanced pipelines. | Trait polish for IPC-only deployments. |
| **toadStool** | `compute.offload` works. `compute.dispatch` returns opaque JSON (AG-007). Live Science API not available (AG-012). | Typed response contracts for science dispatch. `toadstool.validate` endpoint. |
| **NestGate** | **HIGH PRIORITY.** `data.weather` works but goes direct HTTP. Non-standard method name (AG-008). URLs env-overridable. | **Ship Unix socket IPC with weather + NCBI providers.** |
| **biomeOS** | `composition.status` + `method.register` wired. Socket discovery works well. | Neural API ergonomics as ecosystem grows. |
| **skunkBat** | Added to deploy graph. Security capabilities declared. Not exercised in science. | When Phase 3 ships: audit forwarding to rhizoCrypt DAG. |
| **coralReef** | Discovery hook exists. No active shader compilation usage (AG-006). | Low priority — sovereign compile when mature. |
| **Squirrel** | 10 MCP tools registered. `inference.*` not in science path (AG-005). | Await neuralSpring WGSL inference. |
| **sweetGrass** | Provenance braiding works via trio. BLAKE3 content hashes on all experiment data. | Continue trio pattern. |
| **bearDog + songbird** | Sovereign TLS working. Required nodes in niche deploy. Transport enum covers both. | No action needed. |

---

## For Spring Teams

### Patterns Worth Absorbing

1. **`capability_cross_sync.rs` test** — validates shared methods against canonical 413. Distinguishes spring-local (exempt), aligned (must match), extending (tracked). Every spring should replicate.

2. **Env-overridable endpoints** — `OPEN_METEO_ARCHIVE_URL`, `NASS_API_BASE` allow sovereign routing without code changes. Any spring with hardcoded URLs should adopt.

3. **`#[expect(reason)]` everywhere** — zero `#[allow()]` in production. Lint suppressions have explicit reasons. `#[expect]` warns when the suppression is no longer needed (free dead code cleanup).

4. **Feature-gated `CompositionContext`** — `certification/composition.rs` behind `guidestone` feature provides typed composition validation when primalSpring is available.

5. **UniBin pattern** — single binary with `certify`/`validate`/`serve`/`status`/`version` subcommands. Certification is a library module, not a standalone binary. Scenarios are a registry, not ad-hoc test files.

6. **Smart file refactoring** — split by semantic boundary (function bodies vs assembly), not arbitrary line count. `bench_fns.rs` (bodies) + `benchmarks.rs` (assembly).

### Methods airSpring Extends Beyond Canonical 413

10 methods in airSpring's `capability_registry.toml` not yet in primalSpring canonical:

| Method | Recommendation |
|--------|---------------|
| `composition.status` | **Register canonical** — ecosystem-wide biomeOS v3.51 pattern |
| `method.register` | **Register canonical** — ecosystem-wide dynamic registration |
| `provenance.begin/record/complete/status` | **Register canonical** — provenance is ecosystem-wide |
| `primal.forward` | **Register canonical** — cross-primal forwarding |
| `primal.discover` | **Register canonical** — runtime discovery |
| `data.cross_spring_weather` | Spring-specific, but weather exchange may be common |
| `data.weather` | Via NestGate — should align with NestGate method namespace |

---

## Next Round Requirements: Full Data and Compute Chains

The next evolution round needs to close the gap between validated science modules
and production sovereign compute. Specific needs:

### Data Chains (blocked on NestGate)
- **Weather pipeline**: Open-Meteo → NestGate → airSpring (content-addressed, cached)
- **Genomic pipeline**: NCBI SRA → NestGate → wetSpring/airSpring (16S FASTQ for soil microbiome)
- **Yield data**: USDA NASS → NestGate → airSpring (county-level crop yields)
- **Satellite**: NASA SMAP soil moisture, Sentinel-2 NDVI (future)

### Compute Chains (blocked on live NUCLEUS)
- **GPU dispatch**: airSpring → toadStool `compute.offload` → barraCuda GPU (live, not mock)
- **Cross-spring pipeline**: airSpring θ(t) → wetSpring diversity → neuralSpring prediction
- **LAN HPC**: Eastgate (Node+NPU) → Westgate (Heavy Nest, 76TB) → Southgate (RTX 3090) → Strandgate (dual EPYC, bioinformatics) → Northgate (RTX 5090, LLM)

### Certification Chains
- **L5 validation**: Live `composition.status` + `method.register` + `compute.dispatch` against running primals (not mocked)
- **L6 validation**: Cross-spring pipeline end-to-end with live deploy graphs and capability registry

### Foundation + projectNUCLEUS
- 36/36 thread06_ag targets validated
- 6 toadStool workloads published
- `foundation` seeded with FAO-56 ET₀ datasets + sweetGrass provenance braid
- Next: real field data (Dong lab 2026 growing season)

---

## Active Gaps for primalSpring Gap Registry

| ID | Primal | Gap | Priority |
|----|--------|-----|----------|
| AG-005 | Squirrel | `inference.*` not in science path | Low |
| AG-006 | coralReef | Sovereign shader compile not wired | Low |
| AG-007 | ToadStool | `compute.dispatch` opaque results | Medium |
| AG-008 | NestGate | Non-standard weather method name | **High** |
| AG-009 | petalTongue | No direct IPC wiring | Low |
| AG-010 | barraCuda | `TensorSession` not available | Medium |
| AG-011 | barraCuda | Anderson WGSL shader needed | Medium |
| AG-012 | toadStool | Live Science API not implemented | Medium |

**Resolved this round**: AG-015 (barraCuda mandatory path dep → Tier 4 optional + fallbacks)

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
| Deploy graphs | 4 |
| Experiments | 90 (all PASS) |
| guideStone | **L4** (targeting **L6**) |
| Validation scenarios | 10 (UniBin) |
| CPU speedup | 14.3× (24/24 parity) |
| CPU-GPU parity | 21/21 modules |
| Coverage | 90.56% |
| C dependencies | 0 |
| `#[allow()]` in production | 0 |
| `unsafe` in production | 0 |
| Clippy warnings | 0 |

---

*This handoff supersedes the May 10 post-interstadial handoff for ecosystem evolution context.*
*Technical inventory remains in `AIRSPRING_V010_POST_INTERSTADIAL_UPSTREAM_HANDOFF_MAY10_2026.md`.*
*See also: `docs/PRIMAL_GAPS.md`, `docs/PRIMAL_PROOF_IPC_MAPPING.md`, `capability_registry.toml`*
