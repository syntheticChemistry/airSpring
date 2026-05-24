# airSpring — strandGate NUCLEUS Deployment

**Date**: May 23, 2026
**Spring**: airSpring v0.10.0 (ecology / agriculture)
**Gate**: strandGate (Dual EPYC 7452 64-core, 256GB ECC, RTX 3090 + RX 6950 XT)
**Co-tenant**: wetSpring
**Directive**: Wave 46+ Post-Primordial Covalent Gate Deployment

---

## Gate Assignment Confirmed

airSpring confirms **strandGate** as its assigned gate per the Delta Springs
deployment directive. Hardware characteristics are well-suited for airSpring's
workload profile:

- **128-thread EPYC**: 16S diversity pipeline (10 studies, ~500K seqs/study)
- **RTX 3090**: GPU-accelerated ET₀, soil physics, Anderson eigenvalue
- **RX 6950 XT**: Secondary GPU for mixed-hardware dispatch (metalForge)
- **256GB ECC**: Large dataset memory residence (NCBI bulk, metagenomics)

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

```sh
# 1. Fetch plasmidBin binaries (v2026.05.23, 13 primals)
../../../springs/primalSpring/tools/fetch_primals.sh --all

# 2. Start NUCLEUS composition
../../../springs/primalSpring/tools/nucleus_launcher.sh --composition niche-airspring

# 3. Validate composition health
cargo run --features local --bin validate_gate_composition

# 4. Start airspring_primal
FAMILY_ID=<gate_family_id> cargo run --release --bin airspring_primal

# 5. Validate airSpring domain science against live primals
cargo run --features local --bin airspring -- validate

# 6. Run NUCLEUS pipeline validation
cargo run --features local --bin validate_nucleus_pipeline
```

## Co-Tenant Coordination (wetSpring)

strandGate hosts both airSpring and wetSpring. The primary cross-spring
pipeline:

```
NestGate ESearch → EFetch(FASTQ) → [wetSpring DADA2] → airSpring Shannon/Bray-Curtis
  → groundSpring uncertainty → NestGate store
```

### Coordination Points

| Resource | Convention |
|----------|-----------|
| NestGate CAS | Namespaced keys: `airspring/` prefix for ecology data, `wetspring/` for metagenomics |
| Provenance sessions | Separate DAG sessions; shared sweetGrass attribution braid per pipeline run |
| GPU access | RTX 3090 primary (airSpring ET₀ + wetSpring), RX 6950 XT secondary (metalForge mixed dispatch) |
| Socket directory | Shared family-scoped UDS: `/run/user/$(id -u)/biomeos/{primal}-{family}.sock` |
| toadStool dispatch | `max_guest_load` respected; owner foreground priority |

## Current Status

| Item | Status |
|------|--------|
| Gate assignment | **Confirmed**: strandGate |
| IPC wiring | **Complete**: All 13 primals wired in `primal_names.rs` |
| Typed clients | **Complete**: 10/13 primals have typed IPC clients |
| Observatory | **Complete**: NeuralBridge module (v3.67+ routing weights, weight health) |
| BLAKE3 provenance | **Complete**: 62/62 benchmark JSONs hashed |
| Gate validator | **Complete**: `validate_gate_composition` (Exp 094-AS) |
| Deployment | **Blocked**: strandGate hardware ready, deploy order #3 |
| Live validation | **Blocked**: Waiting for NUCLEUS deployment on strandGate |
| guideStone L5 | **Blocked**: Requires live primals |

## Gaps Discovered

| Gap | Impact | Owner |
|-----|--------|-------|
| strandGate NUCLEUS not deployed | All live validation blocked | ops / projectNUCLEUS |
| AG-006: coralReef shader not wired | GPU shader dispatch uses barraCuda direct; no sovereign shader compile | coralReef + airSpring |
| AG-007: toadStool opaque dispatch | No typed ecology response contract for compute.dispatch | toadStool + airSpring |
| AG-009: petalTongue no direct IPC | Graph-level only; no visualization dispatch from airspring_primal | low priority |
| Cross-gate mesh | Covalent linking with ironGate/eastGate pending | biomeOS + songBird |

---

## Verification

```bash
cargo check --features local                         # zero errors
cargo clippy --features local -- -D warnings         # zero warnings
cargo test --lib --features local neural_bridge       # 4/4 pass
./tools/blake3_backfill.sh                           # 62/62 hashed
./tools/publish_sporeprint.sh --dry-run              # 2 files, 0 failed
```
