# Primal Gaps — airSpring v0.10.0

**Date**: May 29, 2026 (Wave 60 Eukaryotic / Pre-Covalent — Forgejo periplasm, 474-method canonical, 38-repo manifest, plasmidBin-only, 46 niche live / 57 registered)
**Spring**: airSpring (ecology / agriculture)
**Gate Assignment**: **eastGate** (i9-12900, RTX 4070 + Akida NPU, 32GB DDR5) — co-residents: primalSpring (coord), neuralSpring, groundSpring
**guideStone Level**: **L4** (cross-atomic pipeline / provenance tier) → targeting **L5+** (NUCLEUS composition, live primals)
**License**: AGPL-3.0-or-later

---

## Purpose

This document tracks gaps discovered during airSpring's evolution from
validated Rust science toward primal composition and **L4–L6** guideStone
(certification layers L0–L6: structural through cross-spring pipeline). Each gap
is a missing capability, wire contract issue, or primal behavior that
blocks or complicates composition. Gaps are handed back to primalSpring
for ecosystem-wide refinement.

Format follows wetSpring/hotSpring `PRIMAL_GAPS.md` pattern.

---

## Active Gaps

| ID | Primal | Gap | Impact | Status |
|----|--------|-----|--------|--------|
| ~~AG-005~~ | Squirrel | ~~`inference.*` not exercised in science path~~ | Moved to Resolved | **RESOLVED** 2026-05-13 |
| AG-006 | coralReef | Sovereign shader compile not wired | `discover_shader_compiler()` hook exists but no active usage; all GPU dispatch through barraCuda direct | **Open** — coralReef stability items in Pass 12 (bind_stat timeout, FECS cold init, naga::Module ingest) |
| AG-007 | ToadStool | `compute.dispatch` returns opaque results | airSpring `compute.offload` forwards raw JSON; no typed response contract for ecology workloads | **Open** — need wire standard; toadStool Phase C (S245-S249) landed but Phase D pending |
| ~~AG-008~~ | NestGate | ~~`data.open_meteo_weather` non-standard method~~ | Moved to Resolved | **RESOLVED** 2026-05-13 |
| AG-009 | petalTongue | No direct IPC wiring from airSpring | Cell graph includes petalTongue but airspring_primal has no visualization dispatch; petalTongue integration is graph-level only | **Open** — low priority; petalTongue consumes via biomeOS SSE; Tier 3 convergence item |
| AG-010 | barraCuda | `TensorSession` / `TensorContext` not available | Seasonal GPU pipeline blocked on persistent buffer pooling; documented in `evolution_gaps.rs` | **Open** — barraCuda roadmap item |
| AG-011 | barraCuda | Anderson coupling needs new WGSL shader | `science.anderson_coupling` runs CPU-only; no upstream shader exists | **Open** — Tier C in GPU promotion map |
| AG-021 | toadStool / hardware | Akida AKD1000 not enabled | PCIe 07:00.0 present but BAR disabled, no kernel driver, no `/dev/akida*`. toadStool `neural_compute` workload type ready but 0 NPU devices. | **Open** — hardware driver gap; BrainChip firmware/driver installation needed |

---

## Resolved Gaps

| ID | Primal | Gap | Resolution | Date |
|----|--------|-----|------------|------|
| AG-001 | primalSpring | `downstream_manifest.toml` not read by airSpring | **Resolved:** `certification/bare.rs` reads manifest via `AIRSPRING_MANIFEST_PATH` / `ECOPRIMALS_ROOT` / relative path. Validates identity, fragments, dependencies, capabilities, health caps (16/16 PASS). Proto-nucleation gate met. | 2026-05-11 |
| AG-015 | barraCuda | barraCuda still mandatory path dep | **Tier 4 IPC-first (2026-05-11):** `optional = true` behind `local`; **`[features].default = []`** (was `["local", "testutil"]`); validation binaries **`required-features = ["local"]`**; `gpu` feature-gated; `math.rs` dual-path + `ipc/barracuda_route.rs`; default build without linking barraCuda | 2026-05-11 |
| AG-002 | primalSpring | Path dep deprecated | Standalone manifest reader via `toml` crate — no primalspring crate dep needed | 2026-05-07 |
| AG-003 | biomeOS | `health_method` inconsistency | Aligned metalForge deploy to `health.liveness` | 2026-04-27 |
| AG-004 | biomeOS | Capability naming drift | Converged metalForge deploy to niche.rs canonical names | 2026-04-27 |
| AG-013 | projectNUCLEUS | Workload paths hardcoded to ironGate | Migrated to `${AIRSPRING_ROOT}` convention | 2026-05-07 |
| AG-014 | foundation | Thread 6 targets/workloads missing | 36 targets + 6 workloads created | 2026-05-07 |
| AG-016 | airSpring | LTEE E3 (Dolgikh FLS2) | **COMPLETE:** `validate_ltee_fls2` binary — Python 12/12 + Rust 29/29 PASS (Langmuir/Hill/two-site binding, glycosylation Kd shift, soil-immune coupling) | 2026-05-12 |
| AG-017 | airSpring | `--format json` not available on validate | **COMPLETE:** `OutputFormat` enum + `harness_to_json()` — structured JSON output for Tier 2 projectNUCLEUS ingestion | 2026-05-12 |
| AG-018 | airSpring | GPU capability_registry.toml drift (7 methods) | **COMPLETE:** Makkink, Turc, Hamon, Blaney-Criddle, Green-Ampt, autocorrelation, ecology.autocorrelation corrected to `gpu_accelerated = true` | 2026-05-12 |
| AG-019 | airSpring | projectNUCLEUS workloads incomplete (1/6) | **COMPLETE:** 6 workload TOMLs created in `projectNUCLEUS/workloads/airspring/` (et0-validation, et0-methods, soil-physics, water-balance, atlas-pipeline, full-suite) | 2026-05-12 |
| AG-020 | foundation | Thread 4 expression missing | **COMPLETE:** `ENVIRONMENTAL_GENOMICS.md` authored — soil-immune coupling, Anderson QS, sentinel microbes, field science; FLS2 target added (13 total) | 2026-05-12 |
| AG-012 | toadStool | Live Science API not implemented | **RESOLVED:** `toadstool.validate` implemented upstream (S250) + wired in airSpring `ipc::toadstool_validate`; `precision.route` implemented + wired in `ipc::precision_route`; composition-parity scenario exercises both; Tier 2 unblocked | 2026-05-12 |
| AG-005 | Squirrel | `inference.*` not on science path | **RESOLVED:** `inference.embed`, `inference.complete`, `inference.models` wired through `dispatch_science` + `niche::CAPABILITIES` (49 caps) + `capability_registry.toml`; 7 dispatch tests + 8 IPC tests; soil sensor similarity search documented use case | 2026-05-13 |
| AG-008 | NestGate | `data.open_meteo_weather` non-standard method | **RESOLVED:** `data.weather` handler evolved to `capability.call` routing; typed CAS client wired (`ipc::nestgate_data` — `content.store`, `content.get`, `storage.status`; 8 TCP round-trip tests) | 2026-05-13 |

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
Current:  gS Level 4 (certification L0–L6 engine; 10 UniBin validation scenarios; Tier 4 IPC-first; 98 binaries; LTEE E3 DONE)
Target:   gS Level 5 (NUCLEUS composition — composition.status + method.register + compute.dispatch against live primals)
Next:     gS Level 6 (cross-spring pipeline — deploy graphs, capability registries, scenario registries)
```

### L5 Readiness Assessment (May 13, 2026)

airSpring has **all seven L5 RPC handlers wired and structurally tested** (1,057 lib tests):
- `composition.status` — wired (biomeOS v3.51 contract)
- `primal.announce` — Wave 17 single-call registration (57 capabilities); `method.register` legacy fallback
- `compute.dispatch` — wired (toadStool identity_f64 shader)
- `toadstool.validate` — wired via `ipc::toadstool_validate` (Tier 2 pre-flight)
- `precision.route` — wired via `ipc::precision_route` (Tier 2 precision advisory)
- `content.store` / `content.get` / `storage.status` — **NEW** wired via `ipc::nestgate_data` (NestGate CAS)
- `inference.embed` / `inference.complete` / `inference.models` — **NEW** wired via `ipc::squirrel_inference`

**Niche Atomic wiring complete**: All upstream primal IPC surfaces relevant to
airSpring's ecology niche are wired with typed clients and TCP round-trip tests.
AG-008 (NestGate non-standard method) RESOLVED — `data.weather` handler evolved
from hardcoded `data.open_meteo_weather` to standard `capability.call` routing.

**Remaining L5 blocker**: Live primals (biomeOS + toadStool at minimum).
Without running primals, the L5 certification probes print `SKIP`.
Structural L5 validation with TCP mock round-trip tests passes (1,057 lib tests).

### plasmidBin Deployment Readiness (May 13, 2026)

- `rust-toolchain.toml` includes `x86_64-unknown-linux-musl` target
- `cargo build --release --target x86_64-unknown-linux-musl --features local --bin airspring` produces a **3.3 MB static-pie** binary
- `airspring version` → `airspring 0.10.0 (UniBin)`
- `airspring validate --list` → 10 scenarios
- Binary is self-contained (statically linked, no glibc dependency)
- `infra/plasmidBin/manifest.toml` lists airSpring (org: syntheticChemistry)
- `sources.toml` excludes springs by design (primal-only harvest); spring binaries are staged manually or via CI

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
- [x] capability_registry.toml created (49 methods, sync test + cross-sync vs canonical **413**)
- [x] deny.toml promoted to workspace root (ecoBin v3.0, ring/openssl + aws-lc-sys banned)
- [x] `methods.rs` centralized constants module (49 methods, drift-proof)
- [x] **10 UniBin validation scenarios** (`validation/scenarios/`; exp001–exp003 absorbed plus expanded ScenarioRegistry coverage + **`s_tier4_math_parity`**)
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
- [x] barraCuda optional = true (Tier 4 IPC-first `default = []`, `local` opt-in, `math.rs` fallbacks, `ipc/barracuda_route.rs`, validation `required-features = ["local"]`)
- [x] LTEE E3 complete: `validate_ltee_fls2` (Python 12/12 + Rust 29/29 PASS, 94th binary)
- [x] `--format json` on UniBin `validate` subcommand (Tier 2 projectNUCLEUS ingestion)
- [x] GPU capability_registry.toml drift fix (7 methods corrected)
- [x] 6 projectNUCLEUS workload TOMLs (was 1)
- [x] Foundation Thread 4 expression authored (`ENVIRONMENTAL_GENOMICS.md`)
- [x] lithoSpore handoff README for LTEE E3 (`control/ltee_fls2_plant_immunity/README.md`)
- [x] `precision.route` now consumes all upstream fields (`requires_compiler`, `adapter` in addition to existing)
- [x] lithoSpore module packaging: `fetch_data.sh` + `tolerances.toml` added to LTEE E3
- [x] musl target (`x86_64-unknown-linux-musl`) in `rust-toolchain.toml`; static-pie 3.3 MB binary verified standalone
- [x] `ipc/barracuda_route.rs` hardcoded `/tmp/barracuda.sock` → `resolve_transport(BARRACUDA)` standard discovery
- [x] `bench_cpu_vs_python` Freundlich gap closed (25/25 parity, was 24)
- [x] `math.rs` unit tests (mean, pearson_r, std_dev — 6 tests, was 0)
- [x] `ipc/barracuda_route.rs` unit tests (2 tests, was 0)
- [x] `BARRACUDA_REQUIREMENTS.md` benchmark count drift fixed (25/25, was "18/18")
- [x] `ipc/nestgate_data.rs` — NestGate CAS typed client (`content.store`, `content.get`, `storage.status`; 8 tests)
- [x] `ipc/squirrel_inference.rs` — Squirrel inference typed client (`inference.embed`, `inference.complete`, `inference.models`; 8 tests)
- [x] `data.weather` handler evolved: `data.open_meteo_weather` → `capability.call` (AG-008 RESOLVED)
- [x] `methods.rs` — 6 new constants (NestGate CAS + Squirrel inference)
- [x] Clippy pedantic+nursery zero warnings: 9 `#[must_use]` attrs added (`barracuda_route`, `provenance`, `skunkbat` ×4, `math` ×3)
- [x] Last `/tmp/` hardcoded path eliminated: `data/provider.rs` `SongbirdTransport::discover` → biomeOS standard discovery
- [ ] guideStone L5 / live NUCLEUS validation (blocked on live biomeOS + toadStool — Pass 14)
- [ ] guideStone L6 / cross-spring pipeline (deploy graphs validated against live NUCLEUS)

### Wave 60 Eukaryotic Unicellular — After Action Report (May 29, 2026)

Per primalSpring Wave 60 — Forgejo periplasm, cascade sync, fresh-gate stress test:
- [x] **Cascade-pull executed**: 25/30 repos synced from Forgejo. 2 failed (wetSpring, healthSpring — merge conflicts). 3 skipped (not cloned).
- [x] **Missing repos cloned**: `primals/songBird`, `primals/nestGate` from Forgejo; `gardens/foundation` from GitHub (not on Forgejo).
- [x] **Dangling symlink found + fixed**: plasmidBin symlink pointed to `barracuda/target/release/` but workspace `target-dir` resolves to `ecoPrimals/target/release/`. Fixed to correct path.
- [x] **Binary rebuilt**: `target/` cleaned by co-tenant. Rebuilt `airspring_primal`, re-linked to plasmidBin.
- [x] **Forgejo round-trip verified**: Push + pull, HEAD == forgejo/main, SSH auth zero friction.
- [x] **Full AAR posted**: `AIRSPRING_WAVE60_AAR_EUKARYOTIC_DEPLOY_MAY29_2026.md` — 8 blocking issues, 6 enhancement proposals for cascade-pull + plasmidBin tooling.
- [ ] **cascade-pull --clone-missing**: Script can't clone repos, only pull existing. Fresh gate would stall.
- [ ] **Gate identity file**: Hostname `pop-os` doesn't match `east*` pattern. Need `.gate` config file.
- [ ] **Shared target dir isolation**: Co-tenant `cargo clean` wipes all binaries. Need copy-to-plasmidBin or per-crate target dirs.
- [ ] **skunkBat missing from CORE**: In `COMP_TOWER` but not in cascade-pull `CORE` profile.
- [ ] **gardens/foundation not on Forgejo**: `sporeGarden/foundation` returns 404 on Forgejo.

### Wave 50 Post-Primordial Absorption + Covalent HPC (May 25, 2026 PM)

Per primalSpring Wave 50 — post-primordial absorption + covalent HPC evolution:
- [x] **Post-primordial audit**: Zero `target/release/` primal hardcodes in tools/scripts/code. All stale PATH binaries removed (beardog, toadstool).
- [x] **Songbird mesh seeded**: `mesh.init` with `node_id=eastGate` + bootstrap peer `192.168.1.238:7700` (ironGate). ironGate TCP reachable.
- [x] **Cross-gate probed**: ironGate responds to `capability.list` and `discovery.peers` on TCP :7700. `capability.call` confirmed to route via biomeOS mesh dispatch, not direct Songbird HTTP.
- [x] **Akida neuromorphic explored**: BrainChip AKD1000 on PCIe 07:00.0, IOMMU group 21. toadStool supports `neural_compute` workload type.
- [ ] **AG-021 Akida driver**: AKD1000 BAR regions disabled, no kernel module loaded, no `/dev/akida*`. Need BrainChip driver/firmware to enable neuromorphic dispatch. **Upstream hardware gap.**
- [ ] **Cross-gate live peers**: `discovery.peers` returns 0 — ironGate not yet seeded with eastGate address. Bilateral seeding needed.
- [ ] **Cross-gate capability.call smoke**: Waiting for bilateral mesh seeding to test `science.et0_fao56` via biomeOS mesh dispatch on remote gate.

### Wave 49 Post-Primordial + Covalent Mesh (May 25, 2026)

Per primalSpring Wave 49 — cut primordial patterns, plasmidBin-only deployment:
- [x] **Primordial audit**: Zero primordial deploy patterns in airSpring code (`which`, `cargo install`, `target/release/`, `~/.local/bin/`). Stale `beardog` removed from `~/.cargo/bin/`. Stale `toadstool` in `/usr/local/bin/` flagged (needs sudo to remove).
- [x] **NUCLEUS 12/12**: All primals ALIVE from plasmidBin (petalTongue fixed via stale socket cleanup). plasmidBin auto-detect confirmed — no `NUCLEUS_BIN_DIR` env var needed.
- [x] **Federation LAN bind**: Songbird TCP `0.0.0.0:7700` (was `127.0.0.1:7700`). LAN IP 192.168.1.144:7700 verified.
- [x] **Discovery verified**: `discovery.peers` responds on UDS + localhost TCP + LAN TCP (0 peers — other gates offline)
- [ ] **toadstool /usr/local/bin/**: Stale binary needs sudo removal — **manual action**
- [ ] **loamSpine Tokio panic**: Known upstream — does not block mesh
- [ ] **Cross-gate live peers**: Waiting for ironGate/southGate/biomeGate to come online simultaneously

### Wave 48 Covalent Mesh — Sound Off (May 25, 2026)

Per primalSpring Wave 48 delta spring covalent mesh directive:
- [x] **Gate self-report**: CONTEXT.md `## Gate Deployment` updated — eastGate, i9-12900 / RTX 4070 / Akida NPU / 32GB DDR5, co-residents primalSpring (coord) + neuralSpring + groundSpring
- [x] **NUCLEUS + Songbird federation**: 11/12 primals ALIVE (petalTongue socket-only), Songbird TCP federation on port 7700 active, 7/10 BTSP handshake, 9 primals seeded in Songbird registry
- [x] **Cell deployed**: `cell_launcher.sh airspring start` — `airspring-nucleus01.sock`, family `nucleus01`, 46 capabilities, health ALIVE
- [x] **Discovery mesh verified**: `discovery.peers` operational on UDS + TCP (port 7700 `/jsonrpc`); 0 peers (other gates offline on current LAN segment)
- [x] **Handoff posted**: `AIRSPRING_WAVE48_COVALENT_MESH_MAY25_2026.md`
- [ ] **Cross-gate capability.call**: Waiting for ironGate/southGate/biomeGate to come online for end-to-end validation
- [ ] **Plasmodium status**: Need 3+ meshed gates for `biomeos plasmodium status`
- [ ] **toadStool S274 yield-to-owner**: GPU workload yield testing with co-tenant neuralSpring pending
- [ ] **Songbird federation TCP path**: Federation endpoint is `/jsonrpc` (not root `/`); upstream docs discrepancy

### Wave 46+ Post-Primordial Gate Deployment (May 23, 2026)

Per primalSpring Wave 46+ covalent gate deployment directive:
- [x] **Gate composition validator**: `validate_gate_composition` binary (Exp 094-AS) — probes all 10 primals in niche-airspring NUCLEUS + biomeOS + Neural API observatory + provenance trio + NestGate CAS + airSpring composition.status
- [x] **Proto-nucleate understood**: `downstream_manifest.toml` entry analyzed — 9 primals (beardog, songbird, skunkbat, toadstool, barracuda, coralreef, nestgate, rhizocrypt, loamspine, sweetgrass)
- [x] **Deployment pipeline mapped**: `fetch_primals.sh` → `nucleus_launcher.sh` → `validate_gate_composition` → `airspring validate`
- [x] **Co-tenant coordination**: primalSpring (coord) + neuralSpring share eastGate — NestGate CAS namespace + Akida NPU scheduling documented
- [x] **Gate deployment handoff**: First ecosystem-wide gate deployment handoff posted
- [x] **NUCLEUS deployed LIVE**: 12/12 primals ALIVE via `nucleus_launcher.sh` (zero TCP, UDS-only)
- [x] **Live validation run**: Exp 094-AS **23/32 PASS** — 4/4 capability routing, 8/10 primal health, NestGate CAS responding
- [ ] **Discovery convention gaps**: skunkBat socket naming, coralReef `coralreef-core-*` prefix, biomeOS `neural-api` vs `biomeos` — **upstream issues for plasmidBin/primalSpring**
- [ ] **Provenance trio env config**: Trio sockets alive but `is_available()` requires env vars, not socket scan — **airSpring fix needed**
- [ ] **guideStone L5**: Live primal proof in progress — 23/32 baseline established
- [x] **Cross-gate mesh**: Songbird federation deployed Wave 48 — `discovery.peers` operational, cross-gate `capability.call` ready via biomeOS v3.75

### Wave 46 Absorption (May 23, 2026)

Per primalSpring v0.9.27 Wave 46 — Ready for Absorption:
- [x] **Registry sync**: Cross-sync test updated from `>= 452` to `>= 458` (6 new `neural_api.*` methods)
- [x] **Doc sweep**: 11 files updated from stale 445-method count to 458 (445 was Wave 36 recount, never swept)
- [x] **NeuralBridge observatory module**: New `ipc/neural_bridge.rs` — `capability_call_instrumented`, `routing_weights`, `route_explain`, `utilization`, `weight_health`, `composition_patterns` (biomeOS v3.67+)
- [x] **composition.status observatory**: Handler now reports `observatory.neural_api_v3_67` health status
- [x] **BLAKE3 backfill**: All 62 benchmark JSONs now have `blake3` hash in `_provenance` block (FN-1 / SP-4 alignment)
- [x] **SP-4 sovereign publish**: `tools/publish_sporeprint.sh` — content.put pipeline to NestGate (base64 + BLAKE3, bearDog-signed)
- [x] **Degradation docs updated**: Neural API observatory degradation table added
- [ ] IonicContractRegistry: **Deferred** — not needed for core science; healthSpring is reference for cross-gate bonding
- [ ] Dark Forest gate scenario: **Deferred** — PENDING per DOWNSTREAM_PATTERN_GUIDE
- [ ] guideStone Tier 4 rewiring: **Deferred** — G column PENDING per scorecard

### Wave 20 PM — lithoSpore Audit Absorption (May 17, 2026)

Per primalSpring lithoSpore downstream audit + ecosystem evolution directive:
- [x] **Stability tier annotations**: All 57 capabilities annotated with `stability` in `capability_registry.toml` (53 stable, 4 evolving)
- [x] **Degradation behavior documented**: `docs/DEGRADATION_BEHAVIOR.md` — per-primal degradation table, trio partial completion states, degradation patterns
- [x] **Trio transaction semantics**: `ProvenanceCompletion` now reports `primals_reached` (which trio primals were successfully contacted); legacy pipeline correctly reports `"partial"` when braid creation fails (was incorrectly `"complete"`)
- [x] **Cross-tier parity validators**: 3 new `validate_*` binaries — `validate_autocorrelation`, `validate_gamma_cdf`, `validate_soil_moisture_topp` (closes parity gap for all methods with Python baselines)
- [x] **Cross-tier parity documented**: `docs/CROSS_TIER_PARITY.md` — per-method parity matrix (17 full, 7 Tier 2)
- [x] **Thread 4 expression**: confirmed present in foundation (`ENVIRONMENTAL_GENOMICS.md`, 12+1 targets). airSpring targets: FAO-56 ET₀ 36/36 validated, FLS2 soil-immune 29/29 validated, no-till Anderson pending field data
- [x] **Dead code audit**: `ipc/barracuda_route.rs` confirmed inactive — retained for future absorption, documented in degradation table

### Wave 20 Debt Resolution (May 17, 2026)

Per primalSpring audit — Wave 20 residual debt:
- [x] Test mock canonical `count`: `primal_dispatch.rs` capability.list mock now returns `"capabilities"` + `"count"` + `"primal"` (canonical envelope); assertion updated to validate `result["count"]`
- [x] `PRIMAL_GAPS` `--provenance-dir` status: marked as implemented (was stale "Remaining")
- [x] 6 missing ecology aliases registered: `ecology.et0_priestley_taylor`, `ecology.et0_makkink`, `ecology.et0_turc`, `ecology.et0_hamon`, `ecology.et0_blaney_criddle`, `ecology.timeseries` — capability count 51 → **57** (dispatch routing existed, but discovery was incomplete)
- [x] `unsafe` consolidation: `usda_nass.rs` and `provider.rs` test modules refactored — scattered `unsafe { env::set_var/remove_var }` consolidated into `testutil::EnvGuard` RAII guard; test modules no longer require `#[expect(unsafe_code)]`
- [x] Unfulfilled lint expectations: removed stale `#[expect(clippy::too_many_lines)]` from `eco/richards.rs` and `gpu/seasonal_pipeline/multi_field.rs`
- [x] Paper queue arithmetic: fixed "All 61" → "All 62", "All 41" → "All 62" in `PAPER_REVIEW_QUEUE.md`
- Zero deep debt remaining: 0 `todo!()`, 0 `unimplemented!()`, 0 `FIXME`, 0 `HACK`, 0 production mocks, 0 unsafe in production, 0 files >800 LOC

### Wave 20 Schema Standardization (May 16, 2026) — *historical; canonical registry is now 474 methods (Wave 60)*

Per primalSpring Wave 20 (445-method registry at the time, Schema Standardization + E2E Validation):
- `primal.list` constant added to `methods.rs` (biomeOS serves primal enumeration)
- `capability.list` canonical envelope: added top-level `"capabilities"` flat string array + `"count"` field (canonical subset per standard; enriched fields retained alongside)
- Registry sync: cross-sync test updated for 445-method canonical at Wave 20 (was 451; **now 458**, Wave 46+)
- 1,057 lib + 69 forge tests pass, 0 clippy warnings
- [x] `--provenance-dir` implemented: `airspring validate --provenance-dir <DIR>` writes `results.json` + `provenance.toml` (Thread 5+6 capture). E3 LTEE activation is the remaining step.

### Wave 17 Signal Adoption (May 16, 2026)

Per primalSpring Wave 17 (451-method registry, Neural API Signal Elevation):
- `primal.announce` adopted: `register_with_target()` now tries single-call `primal.announce` first, falls back to legacy 3-call (`lifecycle.register` + `capability.register` + `method.register`) for pre-v3.57 biomeOS
- `nest.store` signal: `record_experiment_step()` tries `nest.store` dispatch first (biomeOS manages content.put → dag.event.append → spine.seal graph), falls back to legacy `capability.call("dag", "append_event")`
- `nest.commit` signal: `complete_experiment()` tries `nest.commit` dispatch first (biomeOS manages dehydrate → commit → attribute), falls back to legacy 3-phase pipeline
- `primal.info` handler added: returns niche metadata for ecosystem introspection
- Dispatch table: `primal.announce` + `primal.info` added to `airspring_primal` binary
- Capability count: 49 → 51 methods (added `primal.announce`, `primal.info`); later 51 → 57 (Wave 20 Debt: 6 ecology aliases registered)
- Registry sync: cross-sync test updated for 451-method canonical (was 413)
- L5 certification: `validate_primal_announce` replaces `validate_method_register` (with fallback)
- 1,057 lib + 69 forge tests pass, 0 clippy warnings

### Tower Triple-First Evolution (May 14, 2026)

Per upstream plasmidBin manifest, Tower Atomic is now `bearDog + songBird + skunkBat` (was `bearDog + songBird`). All composition definitions updated:
- `metalForge/forge/src/nucleus.rs`: `AtomicKind::Tower` capabilities include `defense.audit`, descriptions include sentinel
- `s_composition_parity.rs`: skunkBat added to Tower health probe loop
- `validate_nucleus_graphs.rs`: Tower detection now requires skunkBat discovery
- Deploy graph comments, validation binaries, docs reconciled
- 1,057 lib + 69 forge tests pass, 0 clippy warnings

### Deep Debt Audit Results (May 13, 2026 — Sprint)

| Category | Finding |
|----------|---------|
| TODO/FIXME/HACK/XXX | **0** in production code |
| `unsafe` blocks (non-test) | **0** (`#![forbid(unsafe_code)]` enforced) |
| `unsafe fn` | **0** |
| Production mocks | **0** (all `Mock` types confined to `#[cfg(test)]`) |
| Files >800 LOC | **0** (largest: `validate_gpu_rewire_support.rs` at 775) |
| Hardcoded primal paths | **0** (last `/tmp/` in `provider.rs` eliminated this sprint) |
| `#[allow(` in production | **0** (all evolved to `#[expect()]` with reasons) |
| `todo!()` / `unimplemented!()` | **0** |
| `.unwrap()` in lib | **0** (only in `bin/` validation CLIs and test code) |
| Clippy pedantic+nursery | **0** warnings |
| External C deps | **0** (serde, clap, thiserror, toml, tracing — all pure Rust) |
| Edition | **2024** (Rust 1.92+) |

### Audit Questions — Answers

**Python baselines for barraCuda CPU parity:**
- **25/25** algorithms in `bench_cpu_vs_python` (ET₀ ×6, Soil ×5, Hydrology ×3, Crop ×4, Ecology ×5, Pipeline ×2)
- **1,284** Python control checks from `control/` scripts
- **Missing from 25-bench harness**: Turc, Hamon, SPI, MC ET₀, bootstrap/jackknife, kriging, autocorrelation, CN+GA coupled — these are validated elsewhere but not in the speed parity table

**Industry GPU benchmarks:**
- **Kokkos**: documented gap — Tier 1 Kokkos/Cabana performance reference not started (groundSpring V74 has 3.5×-2669× dispatch overhead data)
- **LAMMPS/SciPy/Galaxy**: referenced as methodology baselines, not as integrated GPU benchmark harnesses
- **GPU coverage is internal**: 21/21 CPU-GPU parity, 46/46 validate_gpu_math, 25 Tier A upstream ops

**Not implemented / tested:**
- ~~AG-005: Squirrel `inference.*`~~ **RESOLVED** — wired through `dispatch_science` (7 dispatch + 8 IPC tests)
- AG-006: coralReef sovereign shader compile not wired
- AG-007: `compute.dispatch` opaque JSON (no typed response contract)
- AG-010/011: TensorSession and Anderson WGSL shader
- L5/L6 certification: blocked on live primals
- Kokkos Tier 1 performance harness

**Unreviewed papers:** Papers #6, #7 (Tier 1, awaiting field data), #16 (Tier 3), #23/#24 (Tier 4/Future)

**Datasets to examine:** NOAA CDO, OpenWeatherMap (keyed access), Dong lab multi-sensor IoT + lysimeter (awaiting 2026), NCBI 16S (~50 GB 16S budget for metagenome pipeline)

---

**This document is maintained by airSpring and consumed by primalSpring.**
**See also**: `primalSpring/docs/PRIMAL_GAPS.md` (ecosystem-wide gap registry)
**See also**: `primalSpring/docs/CROSS_SPRING_PARITY_SCORECARD.md` (parity dashboard)
