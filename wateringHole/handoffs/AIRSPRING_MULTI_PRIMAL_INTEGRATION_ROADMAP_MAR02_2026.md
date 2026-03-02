# airSpring Multi-Primal Integration Roadmap

**Date:** March 2, 2026
**Author:** airSpring v0.6.1 (post-ToadStool S79 sync)
**Status:** Planning — all prerequisite validation complete
**Scope:** 7-phase roadmap from local NUCLEUS bootstrap through LAN HPC expansion

---

## Context

With airSpring v0.6.1 fully synced to ToadStool S79 (25 Tier A GPU modules, 737
lib tests, 94.15% llvm-cov, ops 0-13 + GPU uncertainty stack), the codebase is
ready to move from single-primal development to multi-primal integration. This
document plans the phased incorporation of NestGate data providers, biomeOS NUCLEUS
orchestration, and cross-primal science pipelines — grounded in the real hardware
inventory documented in `whitePaper/gen3/about/HARDWARE.md`.

## Hardware Inventory

| Gate | Role | Key Hardware | airSpring Use |
|------|------|-------------|---------------|
| **Eastgate** | Node + NPU | i9-12900K, RTX 4070, AKD1000, 32 GB DDR5 | ET0, water balance, scheduling, NPU classifiers |
| **Westgate** | Nest | i7-4771, RTX 2070S, 76 TB ZFS, 32 GB DDR3 | Data gravity (weather archive, blob store) |
| **Southgate** | Node | Ryzen 5800X3D, RTX 3090, 128 GB DDR4 | Richards PDE at scale, kriging |
| **Strandgate** | Node | Dual EPYC 7452 (64 cores), RTX 3090 + RX 6950 XT, 256 GB ECC | Large-scale bioinformatics (16S, NCBI bulk) |
| **Northgate** | Node | i9-14900K, RTX 5090, 192 GB DDR5 | LLM-assisted analysis via Squirrel |
| **biomeGate** | Node | Threadripper 3970X, RTX 3090 + Titan V, 256 GB DDR4 | Semi-mobile heavy compute |

**Aggregate**: 130+ CPU cores, ~176 GB GPU VRAM, ~1.2 TB system RAM, ~105 TB storage,
4x AKD1000 NPU, 10G backbone (switch + NICs installed, cables pending).

---

## Data and Compute Hunger Analysis

### Workload Sizing

| Workload | Data Volume | CPU Time (Eastgate) | GPU Time (Eastgate) | Status |
|----------|-------------|---------------------|---------------------|--------|
| 100-station atlas (1 yr) | ~8 MB | 0.2 s | ~0.01 s | **Complete** |
| 100-station atlas (80 yr) | ~600 MB | ~3 s | ~0.1 s | Download ready |
| Richards PDE 80yr grid | ~600 MB | ~2.2 hrs | ~6 min | GPU wired |
| Kriging 100 stations | ~600 MB | ~8 hrs | ~20 min | GPU wired |
| NCBI 16S coupling | 20-50 GB | ~2 hrs | mixed | Providers validated |
| Full Michigan grid 10km | ~60 GB | TBD | TBD | Tier 2 |
| ERA5-Land Michigan | ~3 TB | days | hours | Tier 3 |
| Sentinel-2 NDVI Michigan | ~500 GB | TBD | TBD | Tier 3 |

### Verdict

Eastgate alone handles Phases 0-4 (everything through full local NUCLEUS). The NCBI
16S coupling (~50 GB data, ~2 hrs compute) is the hungriest single-node workload and
fits comfortably. ERA5-Land (3 TB) and Sentinel-2 NDVI (500 GB) are the workloads
that require Westgate's 76 TB ZFS for data gravity. Multi-node compute (Plasmodium)
becomes necessary at Tier 3 data scale or when running Richards PDE across the full
Michigan 10km grid (~60 GB, routed to Southgate's RTX 3090).

---

## Phased Roadmap

### Phase 0: Local NUCLEUS Bootstrap on Eastgate

**Goal**: Stand up Tower Atomic (BearDog + Songbird) on Eastgate.

**Tasks**:
- Build BearDog binary from `phase1/beardog/`, generate `.family.seed`
- Build Songbird binary from `phase1/songbird/`, verify UDP discovery + Ed25519 challenge-response
- Start Tower Atomic: `biomeos nucleus start --mode tower`
- Verify: `mesh.status`, `mesh.peers`, BearDog socket at `/run/user/1000/beardog-eastgate.sock`

**Data hunger**: Zero
**Compute hunger**: Negligible (two small daemons)
**Hardware purchases**: None
**Blocking on**: Nothing — all binaries buildable now

### Phase 1: NestGate Weather Provider (Node Atomic on Eastgate)

**Goal**: Replace airSpring's direct HTTP data acquisition with NestGate-mediated providers.

**Tasks**:
- Build NestGate binary from `phase1/nestgate/`
- Start Node Atomic: `biomeos nucleus start --mode node` (Tower + NestGate)
- NestGate already implements: `open_meteo_live_provider.rs`, `noaa_cdo_live_provider.rs`, `usda_nass_live_provider.rs`
- Wire airSpring's `HttpTransport::discover_transport()` to detect NestGate socket
- Validate: `capability.call("data.open_meteo_weather")` round-trip through NestGate
- Replace direct `ureq` calls in `barracuda/src/data/open_meteo.rs` with NestGate-mediated data flow
- Content-addressed caching: NestGate stores downloaded weather in blob store

**Data hunger**: Same 600 MB, now cached and content-addressed
**Compute hunger**: Negligible (NestGate is I/O-bound)
**Hardware purchases**: None
**Blocking on**: Phase 0 (Tower Atomic running)

**Validation**: Download 100-station 80yr atlas through NestGate, compare checksums with direct download.

### Phase 2: NestGate NCBI Provider (16S Coupling for baseCamp 06)

**Goal**: Wire the NCBI 16S soil microbiome pipeline through NestGate for `baseCamp/ncbi_16s_coupling.md`.

**Tasks**:
- NestGate already implements: `ncbi_live_provider.rs` with ESearch/ESummary/EFetch
  for nucleotide, protein, pubmed, sra, taxonomy databases
- Wire airSpring `data.ncbi_search` / `data.ncbi_fetch` capability calls
- Download 4-10 soil 16S studies (~20-50 GB FASTQ) via NestGate
- Build Anderson coupling pipeline:
  NestGate 16S data -> OTU table -> Shannon H' -> theta(t) -> S_e -> d_eff -> QS regime
- Cross-spring coordination:
  - wetSpring: 16S processing (DADA2, taxonomy, OTU table)
  - airSpring: soil moisture + Anderson coupling + GPU uncertainty
  - groundSpring: uncertainty propagation (sensor -> xi -> r)

**Data hunger**: 20-50 GB NCBI FASTQ + ~5 GB SILVA/RefSeq reference databases
**Compute hunger**: ~2 hrs on Eastgate (GPU Richards PDE + CPU 16S pipeline)
**Hardware purchases**: None
**Blocking on**: Phase 1 (NestGate running), `NCBI_EMAIL` env var

**Validation**: Reproduce `validate_ncbi_diversity` (63/63 checks) through NestGate pathway.

### Phase 3: ToadStool Compute Offload (Full Node Atomic)

**Goal**: Wire ToadStool GPU compute dispatch through NUCLEUS mesh.

**Tasks**:
- Build ToadStool binary from `phase1/toadstool/`
- Full Node Atomic: Tower (BearDog + Songbird) + ToadStool + NestGate
- Validate `compute.offload` for BarraCUDA GPU workloads through ToadStool socket
- Route AtlasStream GPU path through ToadStool dispatch rather than local device
- This enables future multi-node GPU routing (Phase 5)

**Data hunger**: Same as Phase 1-2
**Compute hunger**: Same workloads, routed through NUCLEUS mesh
**Hardware purchases**: None
**Blocking on**: Phase 1 (NestGate running)

**Validation**: AtlasStream 80yr atlas through ToadStool dispatch, compare results with local GPU path.

### Phase 4: biomeOS Orchestration (Full NUCLEUS on Eastgate)

**Goal**: Run the complete NUCLEUS graph on Eastgate before expanding to LAN.

**Tasks**:
- Start full NUCLEUS: `biomeos nucleus start --mode full`
- Validate `nucleus_complete.toml` graph (BearDog, Songbird, ToadStool, NestGate, Squirrel)
- Test `airspring_ecology_pipeline.toml` end-to-end through biomeOS
- Validate capability routing: `ecology.et0_fao56` through neural-api
- Validate cross-primal forwarding: airSpring -> ToadStool compute offload -> NestGate data
- Squirrel local inference for LLM-assisted analysis (optional, requires Squirrel binary)

**Data hunger**: Same as Phase 2
**Compute hunger**: Full NUCLEUS overhead, all on single node
**Hardware purchases**: None
**Blocking on**: Phase 3 (ToadStool running)

**Validation**: Full `airspring_ecology_pipeline.toml` graph execution,
`validate_nucleus` (29/29) and `validate_nucleus_pipeline` (28/28) through live NUCLEUS.

### Phase 5: LAN HPC (Plasmodium) - Multi-Gate Expansion

**Goal**: Connect gates via 10G backbone for distributed workloads.

**Prerequisites**: 10G Cat6a cables (~$50, pending purchase)

**Tasks**:
- Cable Eastgate, Westgate, Southgate, Strandgate to 10G switch
- Start NUCLEUS on each gate with appropriate mode:
  - Eastgate: Node + NPU (`--mode node`)
  - Westgate: Nest (`--mode nest`)
  - Southgate: Node (`--mode node`)
  - Strandgate: Node (`--mode node`)
  - Northgate: Node (`--mode node`)
- Validate Songbird mesh discovery across gates
- Validate BearDog covalent bonding (shared `.family.seed`)
- Route workloads by hardware capability:
  - ET0 batch (2.9M calcs): Eastgate GPU (0.01 sec, local)
  - Richards 80yr grid (29M sims): Southgate GPU (6 min, routed via `compute.offload`)
  - Kriging 100 stations: Southgate GPU (20 min, routed)
  - Weather data ingest: Westgate NestGate (data gravity, 76 TB ZFS)
  - Cross-spring theta(t)->Anderson: Strandgate (wetSpring + airSpring + 256 GB ECC)
  - LLM analysis: Northgate Squirrel (RTX 5090, 192 GB DDR5)

**Data hunger**: 3+ TB (ERA5-Land, Sentinel-2 NDVI at Tier 3)
**Compute hunger**: Distributed across 5+ gates
**Hardware purchases**: ~$50 Cat6a cables
**Blocking on**: Phase 4 (full local NUCLEUS stable), physical cabling

**Validation**: Richards PDE routed to Southgate, weather download routed to Westgate, results identical to local compute.

### Phase 6: Cross-Primal Science Extensions

**Goal**: Execute the science that motivated multi-primal integration.

**Tasks**:
- **baseCamp 06 live data**: Brandt farm real soil time series via NestGate -> NCBI -> Anderson coupling
  - wetSpring 16S processing on Strandgate (dual EPYC)
  - airSpring Richards PDE on Southgate (RTX 3090)
  - groundSpring uncertainty on Eastgate (local)
  - Results aggregated via biomeOS capability routing
- **baseCamp 09**: MinION + NPU field genomics (awaiting sequencer hardware)
  - MinION -> Strandgate for basecalling
  - AKD1000 on Eastgate/Strandgate for NPU classification
  - NestGate on Westgate for provenance storage
- **Satellite integration**: NASA SMAP soil moisture, Sentinel-2 NDVI
  - Tier 3 data through NestGate on Westgate (data gravity)
  - Processing distributed across Southgate + Strandgate
- **Full Michigan 10km grid**: Open-Meteo 1000 grid cells (~60 GB)
  - Download through Westgate NestGate
  - GPU compute distributed across Eastgate + Southgate

**Data hunger**: Variable (25 GB to 3+ TB depending on workload)
**Compute hunger**: Distributed, hours to days for large-scale runs
**Hardware purchases**: MinION nanopore sequencer (~$1000) for baseCamp 09
**Blocking on**: Phase 5 (LAN HPC operational)

---

## Execution Sequence

```
Phase 0: Tower Atomic on Eastgate            (0 cost, 0 data)
    │
Phase 1: NestGate Weather                    (0 cost, 600 MB cached)
    ├──────────────────────┐
Phase 2: NCBI 16S Pipeline │ Phase 3: ToadStool Compute
    │                      │ (0 cost, GPU routed)
    │  (0 cost, 20-50 GB)  │
    └──────────────────────┘
              │
Phase 4: Full NUCLEUS on Eastgate            (0 cost, all-in-one)
              │
Phase 5: LAN HPC (Plasmodium)               (~$50 cables, 3+ TB)
              │
Phase 6: Cross-Primal Science Extensions     (variable, science-driven)
```

Phases 0-4 are all single-Eastgate, zero hardware purchases, zero external
dependencies. Phase 5 requires only Cat6a cables. The LAN HPC is already
bought and assembled — it needs cabling and software orchestration, not hardware.

---

## NestGate Provider Inventory (Already Implemented)

These providers in `phase1/nestgate/code/crates/nestgate-core/src/data_sources/providers/live_providers/` are ready to wire:

| Provider | Module | JSON-RPC Method | airSpring Use |
|----------|--------|-----------------|---------------|
| NCBI | `ncbi_live_provider.rs` | `data.ncbi_search`, `data.ncbi_fetch` | 16S soil microbiome (baseCamp 06) |
| Open-Meteo | `open_meteo_live_provider.rs` | `data.open_meteo_weather` | Weather data for ET0/WB/atlas |
| NOAA CDO | `noaa_cdo_live_provider.rs` | `data.noaa_ghcnd` | Historical daily records |
| USDA NASS | `usda_nass_live_provider.rs` | `data.usda_nass` | Crop yield data |
| Ensembl | `ensembl_live_provider.rs` | `data.ensembl_*` | Genomic references |
| HuggingFace | `huggingface_live_provider.rs` | `data.huggingface_*` | ML model weights |

---

## biomeOS NUCLEUS Topology

```
                    biomeOS
                      │
         ┌────────────┼────────────┐
         │            │            │
    Tower Atomic  Node Atomic  Nest Atomic
    (BearDog +    (Tower +     (Tower +
     Songbird)     ToadStool)   NestGate)
         │            │            │
    Security +    Compute +    Storage +
    Discovery     GPU/NPU      Provenance
         │            │            │
         └────────────┼────────────┘
                      │
               NUCLEUS Complete
         (Tower + Node + Nest + Squirrel)
```

airSpring runs atop Node Atomic: `BearDog → Songbird → ToadStool → airSpring`.
Deployed via `airspring_deploy.toml` graph.

---

## Risk Assessment

| Risk | Mitigation | Phase |
|------|-----------|-------|
| BearDog/Songbird build failures | All primals are tested in their own CI; build from latest stable tag | 0 |
| NestGate provider API changes | Validate against NestGate's own test suite (11,200+ tests) | 1 |
| 10G cable delay | Phase 0-4 are fully functional on single Eastgate | 5 |
| NCBI rate limiting | `NCBI_API_KEY` env var for higher limits; NestGate caches responses | 2 |
| MinION hardware availability | baseCamp 09 is independent of other phases; defer until purchased | 6 |
| Cross-spring version drift | Pin ToadStool at S79, NestGate at 4.1.0-dev; use handoff protocol | All |

---

## Handoff Chain

| Document | Scope |
|----------|-------|
| `AIRSPRING_V061_TOADSTOOL_S79_SYNC_HANDOFF_MAR02_2026.md` | S79 GPU rewire (completed) |
| `AIRSPRING_V045_FULL_DISPATCH_BIOME_GRAPH_HANDOFF_MAR02_2026.md` | biomeOS graph + dispatch (completed) |
| **This document** | Multi-primal integration roadmap (planning) |

Next handoff will be created when Phase 0 (Tower Atomic on Eastgate) is validated.
