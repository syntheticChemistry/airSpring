# airSpring wateringHole

**Updated**: February 26, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda and cross-spring provenance

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V018** | [AIRSPRING_V018_ATLAS_SCALE_EVOLUTION_HANDOFF_FEB26_2026.md](handoffs/AIRSPRING_V018_ATLAS_SCALE_EVOLUTION_HANDOFF_FEB26_2026.md) | 2026-02-26 | **current** — 100-station Michigan Crop Water Atlas, 1302 atlas checks, scale validation of Tier A stack |
| **V017** | [AIRSPRING_V017_DEEP_AUDIT_EVOLUTION_HANDOFF_FEB26_2026.md](handoffs/AIRSPRING_V017_DEEP_AUDIT_EVOLUTION_HANDOFF_FEB26_2026.md) | 2026-02-26 | Deep audit: Clippy nursery, R-S66-001/003 wired, van_genuchten extracted, 97.45% coverage |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 774 WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring) |
| `../barracuda/EVOLUTION_READINESS.md` | Tier A/B/C status, absorbed/stays-local, quality gates |
| `../metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66), post-absorption leaning status |

## Archive

| File | Scope |
|------|-------|
| `handoffs/archive/AIRSPRING_V016_TOADSTOOL_S66_VALIDATION_HANDOFF_FEB26_2026.md` | v0.4.5: S66 validation complete — P0 resolved, 8 cross-spring tests, absorption candidates |
| `handoffs/archive/AIRSPRING_V015_TOADSTOOL_S66_SYNC_HANDOFF_FEB26_2026.md` | v0.4.5: S66 sync — all metalForge absorbed, evolution_gaps updated |
| `handoffs/archive/AIRSPRING_V014_TOADSTOOL_EXPERIMENT_BUILDOUT_HANDOFF_FEB26_2026.md` | v0.4.5: 3 new experiments, GPU promotion roadmap |
| `handoffs/archive/AIRSPRING_V013_TOADSTOOL_ABSORPTION_HANDOFF_FEB26_2026.md` | v0.4.5: Absorption candidates (all resolved by S66) |
| `handoffs/archive/AIRSPRING_V012_TOADSTOOL_S65_REWIRE_HANDOFF_FEB26_2026.md` | v0.4.4: S65 primitive rewiring: CN f64, brent+norm_ppf, 11 Tier A, 643 tests |
| `handoffs/archive/AIRSPRING_V011_FULL_REWIRE_ABSORPTION_HANDOFF_FEB26_2026.md` | v0.4.3: Full cross-spring rewiring: diversity, MC ET₀, stats re-exports, absorption roadmap |
| `handoffs/archive/AIRSPRING_V010_TOADSTOOL_SYNC_FEB26_2026.md` | v0.4.3: ToadStool S60–S65 sync, stats rewired, sovereign compiler regression, 582 tests |
| `handoffs/archive/AIRSPRING_V009_EVOLUTION_HANDOFF_FEB25_2026.md` | v0.4.2: Full evolution handoff, 758→774 shaders, 601 tests, 18 binaries |
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
