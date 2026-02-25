# airSpring wateringHole

**Updated**: February 25, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda and cross-spring provenance

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V005** | [AIRSPRING_V005_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md](handoffs/AIRSPRING_V005_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md) | 2026-02-25 | v0.4.2: Complete status, 14 primitives consumed, 3 fixes contributed, GPU integration tests, cross-spring benchmarks, P0/P1/P2 actionable items |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 608 WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring) |

## Archive

| File | Scope |
|------|-------|
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
