# airSpring wateringHole

**Updated**: February 25, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda and cross-spring provenance

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V009** | [AIRSPRING_V009_EVOLUTION_HANDOFF_FEB25_2026.md](handoffs/AIRSPRING_V009_EVOLUTION_HANDOFF_FEB25_2026.md) | 2026-02-25 | Full evolution handoff: 758 shaders, 601 tests, cross-spring observations, updated absorption roadmap |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 758 WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring) |
| `../barracuda/EVOLUTION_READINESS.md` | Tier A/B/C status, absorbed/stays-local, quality gates |
| `../metalForge/ABSORPTION_MANIFEST.md` | 4 ready modules with signatures, tests, post-absorption rewiring |

## Archive

| File | Scope |
|------|-------|
| `handoffs/archive/AIRSPRING_V008_TOADSTOOL_SYNC_HANDOFF_FEB25_2026.md` | v0.4.2: ToadStool S62 sync, 585 tests, cross-spring provenance |
| `handoffs/archive/AIRSPRING_V007_LINT_MIGRATION_COVERAGE_HANDOFF_FEB25_2026.md` | v0.4.2: 555 tests, 97.58% coverage, lint migration, evolution readiness |
| `handoffs/archive/AIRSPRING_V006_DEEP_AUDIT_ABSORPTION_HANDOFF_FEB25_2026.md` | v0.4.2+: Deep audit pass 1, 96.84% coverage, testutil split |
| `handoffs/archive/AIRSPRING_V005_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md` | v0.4.2: Complete status, 14 primitives, 3 fixes contributed |
| `handoffs/archive/AIRSPRING_V004_TOADSTOOL_SYNC_HANDOFF_FEB25_2026.md` | v0.4.1: ToadStool S62 sync, multi-start NM |
| `handoffs/archive/AIRSPRING_V003_BARRACUDA_GPU_ABSORPTION_HANDOFF_FEB25_2026.md` | v0.4.0: Richards + isotherm GPU wiring |
| `handoffs/archive/AIRSPRING_V002_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md` | v0.3.10: Dual Kc, cover crops, deep debt |
| `handoffs/archive/AIRSPRING_V001_BARRACUDA_ABSORPTION_HANDOFF_FEB25_2026.md` | v0.3.8: Initial GPU absorption |
| `handoffs/archive/HANDOFF_AIRSPRING_TO_TOADSTOOL_FEB_16_2026.md` | Phase 3 GPU handoff (fossil record) |

## Convention

Handoff files follow: `AIRSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Direction: airSpring → ToadStool (unidirectional). airSpring is a consumer of
BarraCuda primitives; handoffs communicate what we learned, what we need, and
what we can contribute back.

Superseded handoffs move to `handoffs/archive/` (kept as fossil record).
