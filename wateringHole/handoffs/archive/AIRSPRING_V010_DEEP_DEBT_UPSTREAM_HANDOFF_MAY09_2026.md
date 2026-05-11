# airSpring V0.10.0 — Deep Debt Resolution + Upstream Handoff

**Date**: 2026-05-09
**From**: airSpring (ecology/agriculture spring)
**For**: primalSpring (audit), primals teams, sibling springs
**Commit**: `2a51bb8` (deep debt: zero warnings, zero failures, data module tracked)

---

## Summary

Two evolution passes completed since the interstadial eukaryotic wave:

1. **Deep debt resolution**: Removed dead features, fixed all test failures, tracked
   previously-gitignored source code, achieved zero clippy warnings.
2. **Documentation reconciliation**: All docs, counts, and dates synchronized across
   14 files (README, CHANGELOG, whitePaper, specs, experiments, baseCamp, sporeprint).

**Current state**: 1,008 lib tests, 0 failures, 0 clippy warnings, 93 binaries,
guideStone L2. UniBin `airspring` binary operational (certify/validate/serve/status/version).

---

## What Changed (Deep Debt)

| Change | Impact |
|--------|--------|
| Dead `standalone-http` feature removed | Feature was declared but `ureq` dep never existed — broken code paths eliminated |
| Unused `bytemuck` dependency removed | Zero imports in codebase — cleaner dep tree |
| `.gitignore` `data/` anchored to root | Was silently ignoring `barracuda/src/data/` (5 Rust source files) |
| 6 pre-existing test failures fixed | `open_meteo`/`usda_nass` provider constructors now use `try_new` |
| `NassProvider::from_env`/`from_file` → `try_new` | No panic on missing transport in production paths |
| Hardcoded primal names → `primal_names::*` | certification/bare.rs, airspring_guidestone.rs |
| `build_benchmarks()` refactored | 136→8 lines, split into 6 domain groups |
| All clippy warnings resolved | let-else, option_if_let_else, unexpected_cfgs, too_many_lines |
| Bare `#[allow]` → `#[expect(reason)]` | data test modules |
| `fossilRecord/experiments_prokaryotic_may2026/` | Missing directory now created with provenance README |
| `archive/scripts/test_nestgate_providers.py` | Orphaned script archived |

---

## Primal Use and Evolution Review

### Primals airSpring consumes

| Primal | How Used | Transport | Status |
|--------|----------|-----------|--------|
| **barraCuda** | Math primitives (stats, linalg, PDE, optimize), 767+ WGSL shaders | Compile-time (path dep) | **Healthy** — v0.3.7, wgpu 28 |
| **toadStool** | GPU compute dispatch, hardware discovery, AKD1000 NPU | JSON-RPC `compute.dispatch.*` | **Healthy** — akida-driver optional |
| **Songbird** | HTTP transport (Songbird → BearDog TLS), data acquisition | Unix socket `network.http_request` | **Healthy** — sole transport tier |
| **biomeOS** | Primal discovery, socket resolution, lifecycle registration | Unix socket `lifecycle.*`, `capability.*` | **Healthy** — runtime discovery |
| **NestGate** | Weather data routing (`data.open_meteo_weather`) | JSON-RPC `capability.call` | **Wired** — routes through NestGate |
| **rhizoCrypt** | DAG session management | JSON-RPC `dag.*` | **Wired** — provenance trio |
| **loamSpine** | Immutable ledger / commit | JSON-RPC `commit.*` | **Wired** — provenance trio |
| **sweetGrass** | Attribution / provenance braids | JSON-RPC `provenance.*` | **Wired** — provenance trio |
| **coralReef** | Shader compilation (sovereign) | JSON-RPC `shader.compile` | **Discovered** — not yet exercised |
| **Squirrel** | AI narration, 10 MCP tools | JSON-RPC `inference.*`, `tools/list`, `tools/call` | **Wired** — ecology tools discoverable |
| **petalTongue** | Visualization (3-tier discovery) | JSON-RPC `visualization.*` | **Discovered** — not yet exercised |
| **BearDog** | Crypto (Ed25519, encryption) via Songbird TLS delegation | Indirect via Songbird | **Indirect** |

### What airSpring provides back

- **44 IPC capabilities** (science, ecology, provenance, health, coordination)
- **Ecology domain expertise** for NUCLEUS composition graphs
- **3 upstream shader fixes** (TS-001 pow_f64, TS-003 acos, TS-004 reduce)
- **6 upstream shader contributions** (ops 14-19: SCS-CN, Stewart, Makkink, Turc, Hamon, Blaney-Criddle)
- **Validation patterns**: OrExit zero-panic, `#[expect(reason)]`, named tolerances, determinism contract

---

## Gaps for Upstream Primal Teams

### barraCuda

| Gap | Impact | Suggested Resolution |
|-----|--------|---------------------|
| `domain-fhe` feature referenced but undefined | cfg warning in bench_cross_spring_evolution | Define feature in barraCuda or remove reference |
| `bytemuck` not needed by airSpring | Was listed as dep for "GPU buffer I/O" but never imported | Confirm if downstream springs need it |

### toadStool

| Gap | Impact | Suggested Resolution |
|-----|--------|---------------------|
| Live compute dispatch requires running toadStool | Tier 2 validation blocked | Document minimal toadStool deploy for spring testing |

### Songbird

| Gap | Impact | Suggested Resolution |
|-----|--------|---------------------|
| No `standalone-http` fallback | Data providers return `Config` error without Songbird | Expected — Songbird is sovereign transport. Document "start Songbird first" |

### biomeOS

| Gap | Impact | Suggested Resolution |
|-----|--------|---------------------|
| `CompositionContext` not yet used in airSpring | guideStone L3+ blocked | airSpring needs primalSpring `CompositionContext` API for live validation |

### primalSpring

| Gap | Impact | Suggested Resolution |
|-----|--------|---------------------|
| `primalspring` crate unused in airSpring despite `guidestone` feature | Reserved for L3+ | Keep as optional dep, wire when `CompositionContext` API stabilizes |
| Path dep still in Cargo.toml | Requires monorepo layout | Fine for now — feature-gated and optional |

---

## Composition Patterns for NUCLEUS

### Deploy graphs (4 operational)

| Graph | File | What |
|-------|------|------|
| Eco pipeline | `graphs/airspring_eco_pipeline.toml` | weather → ET₀ → WB → yield |
| Provenance pipeline | `graphs/airspring_provenance_pipeline.toml` | session → science → dehydrate → commit → attribute |
| Niche deploy | `graphs/airspring_niche_deploy.toml` | Tower + Trio + NestGate + ToadStool + airSpring |
| Cross-primal soil microbiome | `graphs/cross_primal_soil_microbiome.toml` | airSpring θ(t) → wetSpring diversity |

### Neural API integration

airSpring exposes 10 MCP ecology tools discoverable by Squirrel:
`et0_compute`, `soil_moisture`, `water_balance`, `irrigation_schedule`,
`crop_coefficient`, `weather_fetch`, `yield_estimate`, `richards_solve`,
`diversity_index`, `drought_spi`.

These route through biomeOS `capability.call` for NUCLEUS deployment.

---

## Downstream Absorption Patterns

### For sibling springs

| Pattern | What airSpring demonstrated | Relevant for |
|---------|-----------------------------|--------------|
| **UniBin consolidation** | Single binary with clap subcommands absorbs separate binaries | All springs |
| **Certification organelle** | guidestone → library module (layered L0-L4) | All springs doing guideStone |
| **Scenario registry** | Experiment crates → `validation/scenarios/` with `ScenarioMeta` | All springs with experiments |
| **`try_new` over `new`** | Provider constructors return `Result` not panic | Data provider patterns |
| **Sovereign-only transport** | Remove `ureq`/HTTP fallback, Songbird-only | Springs with data acquisition |
| **`primal_names::*` constants** | Single source for primal name strings | All springs |
| **Domain-grouped benchmarks** | Split monolithic bench builder into domain fns | Springs with large benchmark suites |
| **`.gitignore` anchoring** | `data/` → `/data/` to avoid ignoring `src/data/` | All repos with `data/` patterns |

### For projectNUCLEUS / foundation

airSpring demonstrates the full Python → Rust → GPU → NUCLEUS validation pipeline.
The `whitePaper/baseCamp/` documents the journey per-faculty. Foundation `thread06_ag_targets.toml`
is validated via `exp003` / `s_foundation_targets.rs`.

### For sporeGarden

25 notebooks (20 paper + 5 sporePrint) provide publishable validation documentation.
`notebooks/papers/PAPER_NOTEBOOK_PATTERN.md` is the canonical template.

---

## Documentation Reconciliation

Files updated in this pass (counts unified to 1,008 lib / 93 binaries):

| File | What changed |
|------|-------------|
| `README.md` | Test counts, bin count, data/ description, footer date |
| `CHANGELOG.md` | New "Deep Debt Resolution" section |
| `whitePaper/README.md` | Date, Phase 1 counts |
| `whitePaper/STUDY.md` | Date (March→May), abstract counts |
| `whitePaper/METHODOLOGY.md` | Binary count |
| `whitePaper/baseCamp/README.md` | Date, status line counts |
| `experiments/README.md` | Status line, UniBin absorption note |
| `specs/README.md` | Date, Phase 1 counts |
| `sporeprint/validation-summary.md` | Date, test count, binary count |
| `fossilRecord/experiments_prokaryotic_may2026/README.md` | Created (was missing) |

---

## Archive Actions

| Item | Action |
|------|--------|
| `scripts/test_nestgate_providers.py` | Moved to `archive/scripts/` (unreferenced) |
| `experiments/exp001-003/` | Kept as reference, canonical is `validation/scenarios/` |
| `barracuda/src/bin/airspring_guidestone.rs` | Kept (transitional), superseded by `airspring certify` |
| `barracuda/src/bin/airspring_primal/` | Kept (transitional), superseded by `airspring serve` |

---

## Quality Gates

| Check | Status |
|-------|--------|
| `cargo fmt --check` | **Clean** |
| `cargo check --all-targets` | **Clean** (1 pre-existing domain-fhe cfg warning — resolved) |
| `cargo clippy --all-targets` | **0 warnings** |
| `cargo test --lib` | **1,008 passed, 0 failed** |
| `.gitignore` audit | **Fixed** — `barracuda/src/data/` now tracked |
| Documentation sync | **14 files reconciled** |

---

## Next Steps (airSpring-side)

1. **guideStone L3**: Wire `CompositionContext` for live NUCLEUS validation
2. **Remove transitional binaries**: `airspring_guidestone` + `airspring_primal` once CI/deploy migrated
3. **Live Tier 2 validation**: Deploy via `plasmidBin`, validate against running primals
4. **`barracuda/EVOLUTION_READINESS.md`**: Needs full refresh (March 2026 header, stale counts)
