# airSpring wateringHole

**Updated**: February 25, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda and cross-spring provenance

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| V001 | [AIRSPRING_V001_BARRACUDA_ABSORPTION_HANDOFF_FEB25_2026.md](handoffs/AIRSPRING_V001_BARRACUDA_ABSORPTION_HANDOFF_FEB25_2026.md) | 2026-02-25 | BarraCuda evolution + absorption targets |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 608 WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring) |
| `../HANDOFF_AIRSPRING_TO_TOADSTOOL_FEB_16_2026.md` | Original Phase 3 GPU handoff (historical, now superseded by V001) |

## Convention

Handoff files follow the pattern: `AIRSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Direction: airSpring → ToadStool (unidirectional). airSpring is a consumer of
BarraCuda primitives; handoffs communicate what we learned, what we need, and
what we can contribute back.

## Archive

Superseded handoffs move to `handoffs/archive/`.
