# airSpring wateringHole

**Updated**: May 8, 2026 | **Version**: v0.10.0
**Purpose**: Spring-local handoffs to barraCuda (math) / ToadStool (dispatch), biomeOS, and NUCLEUS ecosystem

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V0.10.0** | [AIRSPRING_V010_DOCS_CLEANUP_UPSTREAM_HANDOFF_MAY08_2026.md](handoffs/AIRSPRING_V010_DOCS_CLEANUP_UPSTREAM_HANDOFF_MAY08_2026.md) | 2026-05-08 | **Docs cleanup + upstream handoff** — 12 files reconciled (90 exp, 986 lib, 44 caps), primal wiring inventory, active gaps for 7 primal teams, composition patterns, archive pass |
| **V0.10.0** | [AIRSPRING_V010_DEEP_DEBT_EVOLUTION_HANDOFF_MAY08_2026.md](handoffs/AIRSPRING_V010_DEEP_DEBT_EVOLUTION_HANDOFF_MAY08_2026.md) | 2026-05-08 | **Deep debt evolution** — methods.rs (44 constants), 3 experiment crates (exp001-003), 3 files refactored, 3 compilation fixes, /proc gating, guideStone L2 |
| **V0.10.0** | [AIRSPRING_V010_PAPER_NOTEBOOKS_ECOSYSTEM_WIRING_HANDOFF_MAY07_2026.md](handoffs/AIRSPRING_V010_PAPER_NOTEBOOKS_ECOSYSTEM_WIRING_HANDOFF_MAY07_2026.md) | 2026-05-07 | **Paper notebooks + ecosystem wiring + parity audit** — 20 paper notebooks (first spring done), foundation thread06 (36 targets, 6 workloads), projectNUCLEUS expansion, capability_registry.toml (44 methods), deny.toml promoted, guideStone L1 scaffold |
| **V0.10.0** | [AIRSPRING_V010_DEEP_AUDIT_EXECUTION_HANDOFF_MAR24_2026.md](handoffs/AIRSPRING_V010_DEEP_AUDIT_EXECUTION_HANDOFF_MAR24_2026.md) | 2026-03-24 | **Deep audit execution** — cargo-deny 0.19 evolution, +43 tests (986 lib / 1,364 total), 90.56% coverage, `const assert` tolerance contracts, SPDX compliance, `blake3` cc wrapper, TCP mock IPC testing pattern |
| **≤V0.9.0** | *Archived* — see `handoffs/archive/` | | Superseded (fossil record) |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 767+ WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring/groundSpring) |
| `../specs/BIOMEOS_CAPABILITIES.md` | Ecology capability domain for biomeOS Neural API |
| `../specs/NUCLEUS_INTEGRATION.md` | NUCLEUS deployment: graphs, workloads, Neural API bridge |
| `../specs/GPU_PROMOTION_MAP.md` | GPU tier status: 24 Tier A + 2 Tier B + 2 Tier C, with blocker effort estimates |
| `../specs/TOLERANCE_REGISTRY.md` | 60 centralized `Tolerance` structs across 5 domain submodules (Rust + Python mirror) |
| `../graphs/airspring_eco_pipeline.toml` | biomeOS deployment graph: weather → ET₀ → WB → yield |
| `../graphs/airspring_provenance_pipeline.toml` | Provenance-tracked experiment: session → science → dehydrate → commit → attribute |
| `../graphs/airspring_niche_deploy.toml` | Full niche deployment: Tower + Trio + NestGate + ToadStool + airSpring |
| `../graphs/cross_primal_soil_microbiome.toml` | Cross-Spring pipeline: airSpring θ(t) → wetSpring diversity |
| `../barracuda/EVOLUTION_READINESS.md` | Tier A/B/C status, absorbed/stays-local, quality gates |
| `../metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66), post-absorption leaning status |

## Archive

| File | Scope |
|------|-------|
| `handoffs/archive/AIRSPRING_V010_DEEP_DEBT_EVOLUTION_HANDOFF_APR27_2026.md` | v0.10.0: Deep debt evolution — 60 centralized tolerances, registry alignment (superseded by May08 deep debt) |
| `handoffs/archive/AIRSPRING_V010_ECOSYSTEM_ABSORPTION_PRIMAL_SPRING_HANDOFF_MAR24_2026.md` | v0.10.0: ecosystem absorption — PRIMAL_REGISTRY, CONTRIBUTING/SECURITY, upstream contract pinning, GPU test_pool, deploy metadata, 1,321 tests (superseded by V0.10.0 Mar24 deep audit execution) |
| `handoffs/archive/AIRSPRING_V010_DEEP_AUDIT_EXECUTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR24_2026.md` | v0.10.0: deep audit execution — three-tier capability discovery, PRIMAL_NAME-derived RPC, provenance headers, forge alignment, doc reconciliation (superseded by V0.10.0 Mar24 ecosystem absorption) |
| `handoffs/archive/AIRSPRING_V010_DEEP_AUDIT_BARRACUDA_TOADSTOOL_HANDOFF_MAR23_2026.md` | v0.10.0: comprehensive deep audit — lint architecture, ecoBin `deny.toml`, CI symlink strategy, GPU evolution (superseded by V0.10.0 Mar24 execution) |
| `handoffs/archive/AIRSPRING_V010_PLATFORM_AGNOSTIC_IPC_HANDOFF_MAR23_2026.md` | v0.10.0: platform-agnostic IPC, `#[expect()]` sweep, doctest migration, Exp 062-087 (superseded by V0.10.0 Mar24 execution) |
| `handoffs/archive/HANDOFF_AIRSPRING_TO_BARRACUDA_TRANSPORT_ABSORPTION_MAR23_2026.md` | v0.10.0: Transport absorption, dependency health, tolerance architecture (superseded by V0.10.0 Mar24 execution) |
| `handoffs/archive/AIRSPRING_V084_DEEP_DEBT_EXECUTION_BARRACUDA_TOADSTOOL_HANDOFF_MAR16_2026.md` | v0.8.4: deep debt execution — primal binary refactored, primal_names, Python provenance, CI expanded (superseded by V085) |
| `handoffs/archive/AIRSPRING_V083_DEEP_DEBT_BARRACUDA_TOADSTOOL_HANDOFF_MAR16_2026.md` | v0.8.3: deep debt resolution — 19 findings, JSON-RPC protocol fix, forbid(unsafe_code), 58 tolerances (superseded by V084) |
| `handoffs/archive/AIRSPRING_V082_NICHE_ARCHITECTURE_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md` | v0.8.2: niche architecture, Edition 2024, deep code quality, barraCuda absorption (superseded by V083) |
| `handoffs/archive/AIRSPRING_V081_NEURALAPI_BARRACUDA_TOADSTOOL_HANDOFF_MAR15_2026.md` | v0.8.1: neuralAPI integration, barracuda usage analysis, evolution opportunities (superseded by V082) |
| `handoffs/archive/AIRSPRING_V080_BIOMEOS_COMPOSITION_HANDOFF_MAR15_2026.md` | v0.8.0: biomeOS composition, Provenance Trio, NestGateProvider, Cross-Spring Time Series (superseded by V082) |
| `handoffs/archive/AIRSPRING_V076_DEEP_DEBT_UPSTREAM_SYNC_HANDOFF_MAR14_2026.md` | v0.7.6: Deep debt resolution, barraCuda 0.3.5 sync (superseded by V082) |
| `handoffs/archive/AIRSPRING_V075_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR08_2026.md` | v0.7.5: CPU/GPU parity, 14 JSON-RPC, NUCLEUS mesh (superseded by V081) |
| `handoffs/archive/AIRSPRING_V074_STOCHASTIC_DROUGHT_TOADSTOOL_HANDOFF_MAR07_2026.md` | v0.7.4: MC ET₀, Bootstrap/Jackknife, SPI drought (superseded by V075) |
| `handoffs/archive/AIRSPRING_V073_MODERN_INTEGRATION_HANDOFF_MAR07_2026.md` | v0.7.3: PrecisionRoutingAdvice, provenance registry (superseded by V075) |
| `handoffs/archive/AIRSPRING_V072_UPSTREAM_LEAN_HANDOFF_MAR07_2026.md` | v0.7.2: Write→Absorb→Lean complete (superseded by V075) |
| `handoffs/archive/AIRSPRING_V071_DEEP_DEBT_NVK_TOADSTOOL_HANDOFF_MAR07_2026.md` | v0.7.1: Deep debt, NVK zero-output, provenance (superseded by V076) |
| `handoffs/archive/AIRSPRING_MULTI_PRIMAL_INTEGRATION_ROADMAP_MAR02_2026.md` | v0.6.1/v0.6.8: Multi-primal integration planning (superseded by V070 state) |
| `handoffs/archive/AIRSPRING_V069_BARRACUDA_ABSORPTION_HANDOFF_MAR05_2026.md` | v0.6.9: Superseded by V070 barraCuda 0.3.3 rewire |
| `handoffs/archive/AIRSPRING_V068_DEEP_DEBT_TOADSTOOL_HANDOFF_MAR04_2026.md` | v0.6.8: Superseded by V069 absorption handoff |
| `handoffs/archive/AIRSPRING_V053_TOADSTOOL_ABSORPTION_GUIDE_MAR02_2026.md` | v0.6.9: Superseded by V068 deep debt handoff (pre-round-2 state) |
| `handoffs/archive/AIRSPRING_V052_TOADSTOOL_S87_SYNC_HANDOFF_MAR02_2026.md` | v0.6.9: ToadStool S87 sync (superseded by S93 rewire) |
| `handoffs/archive/AIRSPRING_V051_LOCAL_GPU_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md` | v0.6.9: 6 local WGSL ops (now in V068 handoff) |
| `handoffs/archive/AIRSPRING_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md` | v0.6.9: Absorption recommendations (superseded by V068 handoff) |
| `handoffs/archive/AIRSPRING_V050_TOADSTOOL_EVOLUTION_HANDOFF_MAR02_2026.md` | v0.6.6: Full evolution handoff (14 contributed, 25 consumed, CPU→GPU→metalForge progression) |
| `handoffs/archive/AIRSPRING_V049_CROSS_SPRING_REWIRE_HANDOFF_MAR02_2026.md` | v0.6.6: Cross-spring rewire (BrentGpu VG inverse, RichardsGpu Picard, 68/68) |
| `handoffs/archive/AIRSPRING_V048_TOADSTOOL_S86_SYNC_HANDOFF_MAR02_2026.md` | v0.6.5: ToadStool S86 sync, 138/138 cross-spring, Tier B→A promotions |
| `handoffs/archive/AIRSPRING_V047_GPU_PIPELINE_EVOLUTION_HANDOFF_MAR02_2026.md` | v0.6.4: GPU multi-field pipeline (Exp 070-072), 13,000× speedup, pure GPU 46/46 |
| `handoffs/archive/AIRSPRING_V046_PAPER12_DEEP_AUDIT_HANDOFF_MAR02_2026.md` | v0.6.3: Paper 12 (Exp 066-069), deep debt audit, ToadStool S79 (124/124) |
| `handoffs/archive/AIRSPRING_V063_DEEP_DEBT_AUDIT_HANDOFF_MAR02_2026.md` | v0.6.3: deep debt audit, provenance, hardcoding elimination |
| `handoffs/archive/AIRSPRING_V062_NAUTILUS_BRAIN_DRIFT_INTEGRATION_MAR02_2026.md` | v0.6.2: Nautilus/AirSpringBrain, CytokineBrain, DriftMonitor |
| `handoffs/archive/AIRSPRING_V061_TOADSTOOL_S79_SYNC_HANDOFF_MAR02_2026.md` | v0.6.1: ToadStool S79 sync, 124/124 cross-spring benchmarks |
| `handoffs/archive/AIRSPRING_V045_*` through `AIRSPRING_V001_*` | Earlier evolution handoffs (fossil record) |
| `handoffs/archive/HANDOFF_AIRSPRING_TO_TOADSTOOL_FEB_16_2026.md` | Phase 3 GPU handoff (fossil record) |

## Convention

Handoff files follow: `AIRSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Direction: airSpring → barraCuda (math) + ToadStool (dispatch), biomeOS, NestGate, Songbird.
airSpring consumes barraCuda primitives and NUCLEUS services, and provides ecology
science capabilities; handoffs communicate what we learned, what we need, and what
we can contribute back. Cross-primal handoffs live in `ecoPrimals/wateringHole/handoffs/`.

Superseded handoffs move to `handoffs/archive/` (kept as fossil record).
