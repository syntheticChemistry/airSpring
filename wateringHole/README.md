# airSpring wateringHole

**Updated**: March 2, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda, biomeOS, and NUCLEUS ecosystem

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V053** | [AIRSPRING_V053_TOADSTOOL_ABSORPTION_GUIDE_MAR02_2026.md](handoffs/AIRSPRING_V053_TOADSTOOL_ABSORPTION_GUIDE_MAR02_2026.md) | 2026-03-02 | **current** — ToadStool absorption guide: 6 pending ops (14-19), cross-spring evolution intelligence, GPU driver insights, performance data, provenance map |
| V052 | [AIRSPRING_V052_TOADSTOOL_S87_SYNC_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_V052_TOADSTOOL_S87_SYNC_HANDOFF_MAR02_2026.md) | 2026-03-02 | ToadStool S87 sync (`2dc26792`), 846/846 tests pass, 0 clippy, revalidation complete |
| V051 | [AIRSPRING_V051_LOCAL_GPU_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_V051_LOCAL_GPU_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md) | 2026-03-02 | 6 local WGSL ops, proposed ops 14-19, NUCLEUS mesh routing |
| — | [AIRSPRING_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md) | 2026-03-02 | Absorption recommendations: 25 Tier A modules, patterns |
| — | [AIRSPRING_MULTI_PRIMAL_INTEGRATION_ROADMAP_MAR02_2026.md](handoffs/AIRSPRING_MULTI_PRIMAL_INTEGRATION_ROADMAP_MAR02_2026.md) | 2026-03-02 | Multi-primal integration: NUCLEUS, NestGate, Songbird, biomeOS |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 844+ WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring/groundSpring) — ToadStool S87 |
| `../specs/BIOMEOS_CAPABILITIES.md` | Ecology capability domain for biomeOS Neural API |
| `../specs/NUCLEUS_INTEGRATION.md` | NUCLEUS deployment: graphs, workloads, Neural API bridge |
| `../specs/GPU_PROMOTION_MAP.md` | GPU tier status: 25 Tier A + 6 GPU-local + Tier B + Tier C |
| `../graphs/airspring_eco_pipeline.toml` | biomeOS deployment graph: weather → ET₀ → WB → yield |
| `../graphs/cross_primal_soil_microbiome.toml` | Cross-Spring pipeline: airSpring θ(t) → wetSpring diversity |
| `../barracuda/EVOLUTION_READINESS.md` | Tier A/B/C status, absorbed/stays-local, quality gates |
| `../metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66), post-absorption leaning status |
| `../../wateringHole/SPRING_EVOLUTION_ISSUES.md` | **Shared** — Cross-primal issues for biomeOS and Spring teams |

## Archive

| File | Scope |
|------|-------|
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

Direction: airSpring → ToadStool, biomeOS, NestGate, BearDog, Songbird (unidirectional).
airSpring is a consumer of BarraCuda primitives and NUCLEUS services, and a provider
of ecology science capabilities; handoffs communicate what we learned, what we need,
and what we can contribute back.

Cross-primal issues go to `../../wateringHole/SPRING_EVOLUTION_ISSUES.md`
(the shared ecosystem issues tracker).

Superseded handoffs move to `handoffs/archive/` (kept as fossil record).
