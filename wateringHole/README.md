# airSpring wateringHole

**Updated**: March 2, 2026
**Purpose**: Spring-local handoffs to ToadStool/BarraCuda, biomeOS, and NUCLEUS ecosystem

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V048** | [AIRSPRING_V048_TOADSTOOL_S86_SYNC_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_V048_TOADSTOOL_S86_SYNC_HANDOFF_MAR02_2026.md) | 2026-03-02 | **current** — v0.6.5: ToadStool S86 sync, 138/138 cross-spring, Tier B→A (`BatchedStatefulF64`, `BatchedNelderMeadGpu`), `BrentGpu`, `RichardsGpu`, `nautilus`, `lbfgs` |
| V047 | [AIRSPRING_V047_GPU_PIPELINE_EVOLUTION_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_V047_GPU_PIPELINE_EVOLUTION_HANDOFF_MAR02_2026.md) | 2026-03-02 | v0.6.4: GPU multi-field pipeline (Exp 070-072), 13,000× speedup, pure GPU 46/46, metalForge 66/66 |
| V046 | [AIRSPRING_V046_PAPER12_DEEP_AUDIT_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_V046_PAPER12_DEEP_AUDIT_HANDOFF_MAR02_2026.md) | 2026-03-02 | v0.6.3: Paper 12 (Exp 066-069), deep debt audit, ToadStool S79 (124/124) |
| — | [AIRSPRING_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md](handoffs/AIRSPRING_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md) | 2026-03-02 | Absorption recommendations: 25 Tier A modules, patterns, evolution opportunities |
| — | [AIRSPRING_MULTI_PRIMAL_INTEGRATION_ROADMAP_MAR02_2026.md](handoffs/AIRSPRING_MULTI_PRIMAL_INTEGRATION_ROADMAP_MAR02_2026.md) | 2026-03-02 | Multi-primal integration: NUCLEUS, NestGate, Songbird, biomeOS |
| V045 | [archive](handoffs/archive/) | 2026-03-02 | v0.6.0: full dispatch + biome graph (superseded by V046) |
| V041 | [archive](handoffs/archive/) | 2026-03-01 | v0.5.8: NUCLEUS primal, cross-primal pipeline |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `../specs/CROSS_SPRING_EVOLUTION.md` | 844+ WGSL shader provenance (hotSpring/wetSpring/neuralSpring/airSpring/groundSpring) — ToadStool S86 |
| `../specs/BIOMEOS_CAPABILITIES.md` | Ecology capability domain for biomeOS Neural API |
| `../specs/NUCLEUS_INTEGRATION.md` | NUCLEUS deployment: graphs, workloads, Neural API bridge |
| `../graphs/airspring_eco_pipeline.toml` | biomeOS deployment graph: weather → ET₀ → WB → yield |
| `../graphs/cross_primal_soil_microbiome.toml` | Cross-Spring pipeline: airSpring θ(t) → wetSpring diversity |
| `../barracuda/EVOLUTION_READINESS.md` | Tier A/B/C status, absorbed/stays-local, quality gates |
| `../metalForge/ABSORPTION_MANIFEST.md` | 6/6 modules absorbed upstream (S64+S66), post-absorption leaning status |
| `../../wateringHole/SPRING_EVOLUTION_ISSUES.md` | **Shared** — Cross-primal issues for biomeOS and Spring teams |

## Archive

| File | Scope |
|------|-------|
| `handoffs/archive/AIRSPRING_V063_DEEP_DEBT_AUDIT_HANDOFF_MAR02_2026.md` | v0.6.3: deep debt audit, provenance, hardcoding elimination (superseded by V046) |
| `handoffs/archive/AIRSPRING_V062_NAUTILUS_BRAIN_DRIFT_INTEGRATION_MAR02_2026.md` | v0.6.2: Nautilus/AirSpringBrain, CytokineBrain, DriftMonitor (superseded by V046) |
| `handoffs/archive/AIRSPRING_V061_TOADSTOOL_S79_SYNC_HANDOFF_MAR02_2026.md` | v0.6.1: ToadStool S79 sync, 124/124 cross-spring benchmarks (superseded by V046) |
| `handoffs/archive/AIRSPRING_V045_FULL_DISPATCH_BIOME_GRAPH_HANDOFF_MAR02_2026.md` | v0.6.0: full dispatch, biome graph, 30 capabilities (superseded by V046) |
| `handoffs/archive/AIRSPRING_V040_SCIENCE_EXTENSIONS_STREAMING_HANDOFF_MAR01_2026.md` | v0.5.7: science extensions (Exp 058), streaming pipeline, 21/21 CPU parity, technical debt |
| `handoffs/archive/AIRSPRING_V034_EXPERIMENT_BUILDOUT_DEBT_RESOLUTION_HANDOFF_FEB28_2026.md` | v0.5.3: 3 new experiments (049-051), deep technical debt resolution, 42+ named constants |
| `handoffs/archive/AIRSPRING_V032_TOADSTOOL_S68_SYNC_REVALIDATION_HANDOFF_FEB28_2026.md` | v0.5.2: ToadStool S68 full review, zero breaking changes, binary registration fix |
| `handoffs/archive/AIRSPRING_V031_GPU_MATH_PORTABILITY_METALFORGE_FIXES_HANDOFF_FEB28_2026.md` | v0.5.2: Exp 047 GPU math portability (13 modules, 46/46), metalForge fixes |
| `handoffs/archive/AIRSPRING_V030_EVOLUTION_ANDERSON_CPU_BENCHMARK_HANDOFF_FEB27_2026.md` | v0.5.1: Exp 045 Anderson coupling, 25.9× CPU benchmark |
| `handoffs/archive/AIRSPRING_V029_TOADSTOOL_S68_UNIVERSAL_PRECISION_SYNC_HANDOFF_FEB27_2026.md` | v0.5.0: ToadStool S68+ sync, universal precision architecture, evolution gaps |
| `handoffs/archive/AIRSPRING_V028_TOADSTOOL_ABSORPTION_TITAN_V_HANDOFF_FEB27_2026.md` | v0.5.0: ToadStool absorption, Titan V live learnings, batch scaling |
| `handoffs/archive/AIRSPRING_V027_GPU_PARITY_DISPATCH_HANDOFF_FEB27_2026.md` | v0.5.0: CPU↔GPU parity, metalForge dispatch, seasonal batch, Titan V + metalForge live hardware |
| `handoffs/archive/AIRSPRING_V026_ENSEMBLE_COUPLING_HANDOFF_FEB27_2026.md` | v0.4.15: ET₀ ensemble, pedotransfer-Richards coupling, bias correction |
| `handoffs/archive/AIRSPRING_V025_BIOMEOS_NEURAL_API_HANDOFF_FEB27_2026.md` | v0.4.13: biomeOS Neural API bridge, ecology capability domain |
| `handoffs/archive/AIRSPRING_V024_DEBT_RESOLUTION_BARRACUDA_ABSORPTION_HANDOFF_FEB27_2026.md` | v0.4.12: Debt resolution + barracuda absorption |
| `handoffs/archive/AIRSPRING_V023_COMPREHENSIVE_EVOLUTION_HANDOFF_FEB26_2026.md` | v0.4.11: Comprehensive evolution: 32 papers, NPU, metalForge mixed hardware |
| `handoffs/archive/AIRSPRING_V022_THORNTHWAITE_GDD_PEDOTRANSFER_HANDOFF_FEB26_2026.md` | v0.4.8: Thornthwaite ET₀, GDD, Saxton-Rawls pedotransfer (22 papers) |
| `handoffs/archive/AIRSPRING_V021_PT_INTERCOMPARISON_EVOLUTION_HANDOFF_FEB26_2026.md` | v0.4.7: Priestley-Taylor ET₀ + 3-method intercomparison (18 papers) |
| `handoffs/archive/AIRSPRING_V019_S68_UNIVERSAL_PRECISION_SYNC_HANDOFF_FEB26_2026.md` | v0.4.6: S68 universal f64 precision sync |
| `handoffs/archive/AIRSPRING_V020_CROSS_SPRING_EVOLUTION_HANDOFF_FEB26_2026.md` | v0.4.6: Cross-spring evolution: 14 primitives, 47 tests, 5-spring shader lineage |
| `handoffs/archive/AIRSPRING_V018_ATLAS_SCALE_EVOLUTION_HANDOFF_FEB26_2026.md` | v0.4.6: 100-station Michigan Crop Water Atlas, 1302 atlas checks, scale validation |
| `handoffs/archive/AIRSPRING_V017_DEEP_AUDIT_EVOLUTION_HANDOFF_FEB26_2026.md` | v0.4.6: Deep audit — Clippy nursery, R-S66 wired, van_genuchten extracted |
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

Direction: airSpring → ToadStool, biomeOS, NestGate, BearDog, Songbird (unidirectional).
airSpring is a consumer of BarraCuda primitives and NUCLEUS services, and a provider
of ecology science capabilities; handoffs communicate what we learned, what we need,
and what we can contribute back.

Cross-primal issues go to `../../wateringHole/SPRING_EVOLUTION_ISSUES.md`
(the shared ecosystem issues tracker).

Superseded handoffs move to `handoffs/archive/` (kept as fossil record).
