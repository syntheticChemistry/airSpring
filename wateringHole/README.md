# airSpring wateringHole

**Updated**: February 25, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda and cross-spring provenance

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| V002 | [AIRSPRING_V002_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md](handoffs/AIRSPRING_V002_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md) | 2026-02-25 | v0.3.10: dual Kc, cover crops, deep debt, GPU evolution |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 608 WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring) |

## Archive

| File | Purpose |
|------|---------|
| `handoffs/archive/AIRSPRING_V001_BARRACUDA_ABSORPTION_HANDOFF_FEB25_2026.md` | V001 handoff (v0.3.8, superseded by V002) |
| `handoffs/archive/HANDOFF_AIRSPRING_TO_TOADSTOOL_FEB_16_2026.md` | Original Phase 3 GPU handoff (superseded by V001, kept as fossil record) |

## Convention

Handoff files follow the pattern: `AIRSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Direction: airSpring → ToadStool (unidirectional). airSpring is a consumer of
BarraCuda primitives; handoffs communicate what we learned, what we need, and
what we can contribute back.

Superseded handoffs move to `handoffs/archive/` (kept as fossil record).
