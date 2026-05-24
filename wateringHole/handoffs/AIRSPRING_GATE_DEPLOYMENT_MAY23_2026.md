# airSpring — eastGate NUCLEUS Deployment

**Date**: May 23, 2026
**Spring**: airSpring v0.10.0 (ecology / agriculture)
**Gate**: eastGate (i9-12900, RTX 4070, Akida NPU)
**Co-tenant**: primalSpring (coord), neuralSpring
**Directive**: Wave 46+ Post-Primordial Covalent Gate Deployment

---

## Gate Assignment Confirmed

airSpring confirms **eastGate** as its assigned gate per the Delta Springs
deployment directive. Hardware characteristics are well-suited for airSpring's
workload profile:

- **i9-12900**: High-clock CPU for seasonal pipeline orchestration, kriging prep, and cross-primal dispatch
- **RTX 4070**: GPU-accelerated ET₀, soil physics, Richards Picard, seasonal batch workloads
- **Akida NPU**: Edge inference offload for soil-sensor similarity, neuralSpring co-tenant workloads

## NUCLEUS Composition (niche-airspring)

From `primalSpring/graphs/downstream/downstream_manifest.toml`:

| Layer | Primal | Role |
|-------|--------|------|
| Tower | BearDog | Crypto, BTSP |
| Tower | Songbird | Discovery, mesh, NAT |
| Tower | skunkBat | Defense |
| Node | ToadStool | Compute dispatch |
| Node | barraCuda | Tensor/GPU math |
| Node | coralReef | Shader compile |
| Nest | NestGate | Storage/CAS |
| Nest | rhizoCrypt | DAG |
| Nest | loamSpine | Ledger |
| Nest | sweetGrass | Attribution |

**Total**: 10 primals + biomeOS orchestrator

## IPC Readiness

All 13 NUCLEUS primals are wired in `primal_names.rs`:

| Primal | Socket Discovery | Typed Client | Health Probe | Active Usage |
|--------|-----------------|--------------|--------------|--------------|
| bearDog | env/UDS/probe | via `capability.forward` | `health` | crypto.hash forwarding |
| songBird | env/UDS/probe | `SongbirdTransport` | `health` | network fetch, DNS |
| skunkBat | env/UDS/probe | `ipc::skunkbat` | `health` | audit log emission |
| toadStool | env/UDS/probe | `ipc::toadstool_validate` | `health` | compute.dispatch |
| barraCuda | env/UDS/probe | `ipc::barracuda_route` | `health` | tensor forwarding |
| coralReef | env/UDS/probe | discovery hook | `health` | shader compile (pending) |
| nestGate | env/UDS/probe | `ipc::nestgate_data` | `storage.status` | CAS store/get |
| rhizoCrypt | provenance config | `ipc::provenance` | trio check | DAG sessions |
| loamSpine | provenance config | `ipc::provenance` | trio check | commit/ledger |
| sweetGrass | provenance config | `ipc::provenance` | trio check | attribution braids |
| biomeOS | env/UDS/probe | `NeuralBridge` | `health` | observatory + routing |
| Squirrel | env/UDS/probe | `ipc::squirrel_inference` | `health` | inference.embed |
| petalTongue | env/UDS/probe | graph-level only | `health` | AG-009 open |

## Validation Artifacts

### Gate Composition Validator (Exp 094-AS)

**Binary**: `validate_gate_composition`

Validates live NUCLEUS composition in 6 phases:

| Phase | Checks | Status |
|-------|--------|--------|
| 1. Primal Discovery + Health | Socket find + health probe for all 10 primals | Built |
| 2. Capability Domain Routing | stats.mean, compute.dispatch, storage.store, crypto.hash via `capability.call` | Built |
| 3. Provenance Trio | `is_available()` probe for rhizoCrypt + loamSpine + sweetGrass | Built |
| 4. NeuralBridge Observatory | `routing_weights()` + `weight_health()` (biomeOS v3.67+) | Built |
| 5. NestGate CAS | `storage.status` probe | Built |
| 6. airSpring Composition | `composition.status` + health ratio + observatory wiring | Built |

**Run**:
```sh
cargo run --features local --bin validate_gate_composition
```

### Existing Validators (Live-Primal Ready)

| Binary | Exp | What it validates |
|--------|-----|-------------------|
| `validate_nucleus_pipeline` | 063 | airSpring → cross-primal pipeline (7 phases) |
| `validate_gate_composition` | 094-AS | Full NUCLEUS composition health (6 phases) |
| `validate_neural_api` | 036 | Neural API round-trip parity |

## Deployment Flow

eastGate deployment is coordinated by primalSpring (gate owner). airSpring
follows the standard NUCLEUS bootstrap once eastGate hardware is live:

```sh
# 1. Fetch plasmidBin binaries (v2026.05.23, 13 primals) — on eastGate host
../../../springs/primalSpring/tools/fetch_primals.sh --all

# 2. Start NUCLEUS composition (primalSpring nucleus_launcher on eastGate)
../../../springs/primalSpring/tools/nucleus_launcher.sh --composition niche-airspring

# 3. Validate composition health on eastGate
cargo run --features local --bin validate_gate_composition

# 4. Start airspring_primal (family-scoped to eastGate NUCLEUS)
FAMILY_ID=<eastgate_family_id> cargo run --release --bin airspring_primal

# 5. Validate airSpring domain science against live primals
cargo run --features local --bin airspring -- validate

# 6. Run NUCLEUS pipeline validation
cargo run --features local --bin validate_nucleus_pipeline
```

**eastGate notes**: RTX 4070 is primary for barraCuda ecology dispatch; Akida
NPU is shared with neuralSpring — coordinate inference windows via toadStool
`max_guest_load` and primalSpring deploy schedule.

## Co-Tenant Coordination (primalSpring + neuralSpring)

eastGate hosts airSpring alongside primalSpring (NUCLEUS coordinator) and
neuralSpring (NPU inference). Cross-spring ecology pipelines that require
wetSpring 16S diversity run on **strandGate** (wetSpring co-tenant), not
eastGate:

```
NestGate ESearch → EFetch(FASTQ) → [wetSpring DADA2 on strandGate] → airSpring Shannon/Bray-Curtis
  → groundSpring uncertainty → NestGate store
```

### Coordination Points

| Resource | Convention |
|----------|-----------|
| NestGate CAS | Namespaced keys: `airspring/` prefix for ecology data, `neuralspring/` for inference artifacts |
| Provenance sessions | Separate DAG sessions; shared sweetGrass attribution braid per cross-gate pipeline run |
| GPU access | RTX 4070 primary (airSpring ET₀ + seasonal batch); neuralSpring defers heavy GPU when ecology batch active |
| Akida NPU | Shared with neuralSpring — schedule via primalSpring deploy graph; airSpring uses Squirrel embed for soil-sensor similarity |
| Socket directory | Shared family-scoped UDS: `/run/user/$(id -u)/biomeos/{primal}-{family}.sock` |
| toadStool dispatch | `max_guest_load` respected; primalSpring coord sets owner foreground priority |

## Current Status

| Item | Status |
|------|--------|
| Gate assignment | **Confirmed**: eastGate |
| IPC wiring | **Complete**: All 13 primals wired in `primal_names.rs` |
| Typed clients | **Complete**: 10/13 primals have typed IPC clients |
| Observatory | **Complete**: NeuralBridge module (v3.67+ routing weights, weight health) |
| BLAKE3 provenance | **Complete**: 62/62 benchmark JSONs hashed |
| Gate validator | **Complete**: `validate_gate_composition` (Exp 094-AS) |
| Deployment | **LIVE**: NUCLEUS launched via `nucleus_launcher.sh` — 12/12 primals ALIVE |
| Live validation | **23/32 PASS** — Exp 094-AS first live run against real NUCLEUS |
| guideStone L5 | **In progress**: Live primals running; discovery convention gaps remaining |

## Live Validation Results (May 23, 2026)

First live NUCLEUS run — **23/32 PASS, 9 FAIL**:

| Phase | Result | Detail |
|-------|--------|--------|
| Primal Discovery | 9/10 sockets, 8/10 healthy | skunkBat: socket not found (non-standard naming); coralReef: health probe failed (uses `coralreef-core-*` socket) |
| Capability Routing | **4/4 PASS** | stats.mean, compute.dispatch, storage.store, crypto.hash all routed via neural-api |
| Provenance Trio | FAIL | rhizoCrypt/loamSpine/sweetGrass sockets alive but `is_available()` requires env config |
| Observatory | FAIL | biomeOS neural-api alive but observatory methods not exposed in cleartext bootstrap |
| NestGate CAS | **PASS** | storage.status responding |
| airSpring Composition | SKIP | airspring_primal not started during this run |

### Discovery Convention Gaps (upstream issues for plasmidBin/primalSpring)

| Issue | Detail | Suggested Fix |
|-------|--------|---------------|
| skunkBat socket naming | No `skunkbat-{family}.sock` created; launcher doesn't start skunkBat separately | plasmidBin launcher: add skunkBat to Phase 1 Tower startup |
| coralReef socket naming | Uses `coralreef-core-{family}.sock` instead of `coralreef-{family}.sock` | Either coralReef normalize or airSpring discovery add `coralreef-core` prefix |
| biomeOS socket | `neural-api-{family}.sock` not discoverable as `biomeos` | airSpring discovery: add `neural-api` as biomeOS alias |
| Provenance trio config | `is_available()` checks env vars, not socket directory scan | airSpring: wire `ProvenanceConfig` from socket discovery, not env-only |
| Observatory methods | neural-api cleartext bootstrap doesn't expose `neural_api.routing_weights` | biomeOS: enable observatory in all modes, or airSpring: probe with BTSP |

## Gaps Discovered

| Gap | Impact | Owner |
|-----|--------|-------|
| skunkBat not started by launcher | Tower defense layer missing from NUCLEUS | plasmidBin / primalSpring |
| coralReef socket naming convention | `coralreef-core-*` vs `coralreef-*` prefix mismatch | coralReef upstream |
| AG-006: coralReef shader not wired | GPU shader dispatch uses barraCuda direct | coralReef + airSpring |
| AG-007: toadStool opaque dispatch | No typed ecology response contract | toadStool + airSpring |
| AG-009: petalTongue no direct IPC | Graph-level only | low priority |
| Provenance env-only discovery | Trio alive but not discovered via socket scan | airSpring |
| Cross-gate mesh | Covalent linking with ironGate/strandGate pending | biomeOS + songBird |

---

## Verification

```bash
cargo check --features local                         # zero errors
cargo clippy --features local -- -D warnings         # zero warnings
cargo test --lib --features local                     # 1061/1061 pass
FAMILY_ID=nucleus01 cargo run --features local --bin validate_gate_composition  # 23/32 PASS (live)
./tools/blake3_backfill.sh                           # 62/62 hashed
./tools/publish_sporeprint.sh --dry-run              # 2 files, 0 failed
```
