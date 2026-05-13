# Primal Gaps — airSpring v0.10.0

**Date**: May 13, 2026 (Niche Atomic Convergence wave)
**Spring**: airSpring (ecology / agriculture)
**guideStone Level**: **L4** (cross-atomic pipeline / provenance tier) → targeting **L5+** (NUCLEUS composition, live primals)
**License**: AGPL-3.0-or-later

---

## Purpose

This document tracks gaps discovered during airSpring's evolution from
validated Rust science toward primal composition and **L4–L6** guideStone
(certification layers L0–L6: structural through cross-spring pipeline). Each gap
is a missing capability, wire contract issue, or primal behavior that
blocks or complicates composition. Gaps are handed back to primalSpring
for ecosystem-wide refinement.

Format follows wetSpring/hotSpring `PRIMAL_GAPS.md` pattern.

---

## Active Gaps

| ID | Primal | Gap | Impact | Status |
|----|--------|-----|--------|--------|
| ~~AG-005~~ | Squirrel | ~~`inference.*` not exercised in science path~~ | Moved to Resolved | **RESOLVED** 2026-05-13 |
| AG-006 | coralReef | Sovereign shader compile not wired | `discover_shader_compiler()` hook exists but no active usage; all GPU dispatch through barraCuda direct | **Open** — coralReef stability items in Pass 12 (bind_stat timeout, FECS cold init, naga::Module ingest) |
| AG-007 | ToadStool | `compute.dispatch` returns opaque results | airSpring `compute.offload` forwards raw JSON; no typed response contract for ecology workloads | **Open** — need wire standard; toadStool Phase C (S245-S249) landed but Phase D pending |
| ~~AG-008~~ | NestGate | ~~`data.open_meteo_weather` non-standard method~~ | Moved to Resolved | **RESOLVED** 2026-05-13 |
| AG-009 | petalTongue | No direct IPC wiring from airSpring | Cell graph includes petalTongue but airspring_primal has no visualization dispatch; petalTongue integration is graph-level only | **Open** — low priority; petalTongue consumes via biomeOS SSE; Tier 3 convergence item |
| AG-010 | barraCuda | `TensorSession` / `TensorContext` not available | Seasonal GPU pipeline blocked on persistent buffer pooling; documented in `evolution_gaps.rs` | **Open** — barraCuda roadmap item |
| AG-011 | barraCuda | Anderson coupling needs new WGSL shader | `science.anderson_coupling` runs CPU-only; no upstream shader exists | **Open** — Tier C in GPU promotion map |

---

## Resolved Gaps

| ID | Primal | Gap | Resolution | Date |
|----|--------|-----|------------|------|
| AG-001 | primalSpring | `downstream_manifest.toml` not read by airSpring | **Resolved:** `certification/bare.rs` reads manifest via `AIRSPRING_MANIFEST_PATH` / `ECOPRIMALS_ROOT` / relative path. Validates identity, fragments, dependencies, capabilities, health caps (16/16 PASS). Proto-nucleation gate met. | 2026-05-11 |
| AG-015 | barraCuda | barraCuda still mandatory path dep | **Tier 4 IPC-first (2026-05-11):** `optional = true` behind `local`; **`[features].default = []`** (was `["local", "testutil"]`); validation binaries **`required-features = ["local"]`**; `gpu` feature-gated; `math.rs` dual-path + `ipc/barracuda_route.rs`; default build without linking barraCuda | 2026-05-11 |
| AG-002 | primalSpring | Path dep deprecated | Standalone manifest reader via `toml` crate — no primalspring crate dep needed | 2026-05-07 |
| AG-003 | biomeOS | `health_method` inconsistency | Aligned metalForge deploy to `health.liveness` | 2026-04-27 |
| AG-004 | biomeOS | Capability naming drift | Converged metalForge deploy to niche.rs canonical names | 2026-04-27 |
| AG-013 | projectNUCLEUS | Workload paths hardcoded to ironGate | Migrated to `${AIRSPRING_ROOT}` convention | 2026-05-07 |
| AG-014 | foundation | Thread 6 targets/workloads missing | 36 targets + 6 workloads created | 2026-05-07 |
| AG-016 | airSpring | LTEE E3 (Dolgikh FLS2) | **COMPLETE:** `validate_ltee_fls2` binary — Python 12/12 + Rust 29/29 PASS (Langmuir/Hill/two-site binding, glycosylation Kd shift, soil-immune coupling) | 2026-05-12 |
| AG-017 | airSpring | `--format json` not available on validate | **COMPLETE:** `OutputFormat` enum + `harness_to_json()` — structured JSON output for Tier 2 projectNUCLEUS ingestion | 2026-05-12 |
| AG-018 | airSpring | GPU capability_registry.toml drift (7 methods) | **COMPLETE:** Makkink, Turc, Hamon, Blaney-Criddle, Green-Ampt, autocorrelation, ecology.autocorrelation corrected to `gpu_accelerated = true` | 2026-05-12 |
| AG-019 | airSpring | projectNUCLEUS workloads incomplete (1/6) | **COMPLETE:** 6 workload TOMLs created in `projectNUCLEUS/workloads/airspring/` (et0-validation, et0-methods, soil-physics, water-balance, atlas-pipeline, full-suite) | 2026-05-12 |
| AG-020 | foundation | Thread 4 expression missing | **COMPLETE:** `ENVIRONMENTAL_GENOMICS.md` authored — soil-immune coupling, Anderson QS, sentinel microbes, field science; FLS2 target added (13 total) | 2026-05-12 |
| AG-012 | toadStool | Live Science API not implemented | **RESOLVED:** `toadstool.validate` implemented upstream (S250) + wired in airSpring `ipc::toadstool_validate`; `precision.route` implemented + wired in `ipc::precision_route`; composition-parity scenario exercises both; Tier 2 unblocked | 2026-05-12 |
| AG-005 | Squirrel | `inference.*` not on science path | **RESOLVED:** `inference.embed`, `inference.complete`, `inference.models` wired through `dispatch_science` + `niche::CAPABILITIES` (49 caps) + `capability_registry.toml`; 7 dispatch tests + 8 IPC tests; soil sensor similarity search documented use case | 2026-05-13 |
| AG-008 | NestGate | `data.open_meteo_weather` non-standard method | **RESOLVED:** `data.weather` handler evolved to `capability.call` routing; typed CAS client wired (`ipc::nestgate_data` — `content.store`, `content.get`, `storage.status`; 8 TCP round-trip tests) | 2026-05-13 |

---

## Provenance Drift (Internal)

These are not primal gaps but internal reconciliation items:

| Area | Rust Header Commit | JSON `_provenance` Commit | specs/README Commit | Status |
|------|-------------------|--------------------------|--------------------|---------| 
| Atlas | `fad2e1b` | `fad2e1b` | `fad2e1b` | **Consistent** (fixed Phase 5.18) |
| Dual Kc | `94cc51d` | `94cc51d` | `94cc51d` | **Consistent** (fixed Phase 5.18) |
| FAO-56 | `94cc51d` | `94cc51d` | `94cc51d` | Consistent |

---

## guideStone Evolution Path

```
Current:  gS Level 4 (certification L0–L6 engine; 10 UniBin validation scenarios; Tier 4 IPC-first; 94 binaries; LTEE E3 DONE)
Target:   gS Level 5 (NUCLEUS composition — composition.status + method.register + compute.dispatch against live primals)
Next:     gS Level 6 (cross-spring pipeline — deploy graphs, capability registries, scenario registries)
```

### L5 Readiness Assessment (May 13, 2026)

airSpring has **all seven L5 RPC handlers wired and structurally tested** (1,057 lib tests):
- `composition.status` — wired (biomeOS v3.51 contract)
- `method.register` — wired (49 capabilities registered)
- `compute.dispatch` — wired (toadStool identity_f64 shader)
- `toadstool.validate` — wired via `ipc::toadstool_validate` (Tier 2 pre-flight)
- `precision.route` — wired via `ipc::precision_route` (Tier 2 precision advisory)
- `content.store` / `content.get` / `storage.status` — **NEW** wired via `ipc::nestgate_data` (NestGate CAS)
- `inference.embed` / `inference.complete` / `inference.models` — **NEW** wired via `ipc::squirrel_inference`

**Niche Atomic wiring complete**: All upstream primal IPC surfaces relevant to
airSpring's ecology niche are wired with typed clients and TCP round-trip tests.
AG-008 (NestGate non-standard method) RESOLVED — `data.weather` handler evolved
from hardcoded `data.open_meteo_weather` to standard `capability.call` routing.

**Remaining L5 blocker**: Live primals (biomeOS + toadStool at minimum).
Without running primals, the L5 certification probes print `SKIP`.
Structural L5 validation with TCP mock round-trip tests passes (1,057 lib tests).

### plasmidBin Deployment Readiness (May 13, 2026)

- `rust-toolchain.toml` includes `x86_64-unknown-linux-musl` target
- `cargo build --release --target x86_64-unknown-linux-musl --features local --bin airspring` produces a **3.3 MB static-pie** binary
- `airspring version` → `airspring 0.10.0 (UniBin)`
- `airspring validate --list` → 10 scenarios
- Binary is self-contained (statically linked, no glibc dependency)
- `infra/plasmidBin/manifest.toml` lists airSpring (org: syntheticChemistry)
- `sources.toml` excludes springs by design (primal-only harvest); spring binaries are staged manually or via CI

### Prerequisites for gS Level 1 (DONE)
1. ~~Clone primalSpring beside springs/~~ — primalSpring at `springs/primalSpring/`
2. ~~Add primalspring path dependency~~ — **Deprecated**; standalone TOML reader
3. Read `downstream_manifest.toml` → validate against niche::CAPABILITIES — `airspring_guidestone` binary (16/16 PASS)
4. Create `airspring_guidestone` binary with Tier 1 (local) property checks — DONE
5. Validate P1–P5 properties without primals running (exit 2 = skip) — DONE

### Prerequisites for gS Level 2 (DONE)
6. ~~Wire Tier 2 IPC checks (`check_skip()` when primals absent)~~ — exp002 implements graceful skip (10/10 PASS)
7. ~~Validate science parity through `capability.call` vs direct Rust~~ — exp001 dispatches all 55 methods (55/55 PASS)

### Prerequisites for gS Level 3+
8. Add primalSpring as feature-gated dep (for `CompositionContext`)
9. Deploy NUCLEUS from plasmidBin, run guideStone against live primals
10. Create composition experiment crates (exp094 pattern replication)
11. Document remaining gaps → hand back

### Deep Debt Evolution (May 8 2026)
- [x] capability_registry.toml created (49 methods, sync test + cross-sync vs canonical **413**)
- [x] deny.toml promoted to workspace root (ecoBin v3.0, ring/openssl + aws-lc-sys banned)
- [x] `methods.rs` centralized constants module (49 methods, drift-proof)
- [x] **10 UniBin validation scenarios** (`validation/scenarios/`; exp001–exp003 absorbed plus expanded ScenarioRegistry coverage + **`s_tier4_math_parity`**)
- [x] Test extraction: provenance.rs (747→496), rpc/mod.rs (650→341), seasonal_pipeline (738→539)
- [x] 3 compilation errors fixed (autobins, NestGateProvider→IPC, fhe_ntt cfg)
- [x] Missing docs resolved (DailyWeather, Station, YieldRecord, HttpResponse, DataError)
- [x] `/proc/*` paths gated behind `cfg(target_os = "linux")`
- [x] primalSpring feature-gated dep (guidestone feature)
- [x] CONTEXT.md reconciled with README.md (single source of truth, May 11)
- [x] composition.status handler wired (biomeOS v3.51 contract)
- [x] skunkBat added to niche deploy graph (9 nodes)
- [x] Zero `#[allow]` in production code (`#[expect]` with reason throughout)
- [x] benchmarks.rs refactored (810→148 + 607 bench_fns.rs, zero >800L files)
- [x] barraCuda optional = true (Tier 4 IPC-first `default = []`, `local` opt-in, `math.rs` fallbacks, `ipc/barracuda_route.rs`, validation `required-features = ["local"]`)
- [x] LTEE E3 complete: `validate_ltee_fls2` (Python 12/12 + Rust 29/29 PASS, 94th binary)
- [x] `--format json` on UniBin `validate` subcommand (Tier 2 projectNUCLEUS ingestion)
- [x] GPU capability_registry.toml drift fix (7 methods corrected)
- [x] 6 projectNUCLEUS workload TOMLs (was 1)
- [x] Foundation Thread 4 expression authored (`ENVIRONMENTAL_GENOMICS.md`)
- [x] lithoSpore handoff README for LTEE E3 (`control/ltee_fls2_plant_immunity/README.md`)
- [x] `precision.route` now consumes all upstream fields (`requires_compiler`, `adapter` in addition to existing)
- [x] lithoSpore module packaging: `fetch_data.sh` + `tolerances.toml` added to LTEE E3
- [x] musl target (`x86_64-unknown-linux-musl`) in `rust-toolchain.toml`; static-pie 3.3 MB binary verified standalone
- [x] `ipc/barracuda_route.rs` hardcoded `/tmp/barracuda.sock` → `resolve_transport(BARRACUDA)` standard discovery
- [x] `bench_cpu_vs_python` Freundlich gap closed (25/25 parity, was 24)
- [x] `math.rs` unit tests (mean, pearson_r, std_dev — 6 tests, was 0)
- [x] `ipc/barracuda_route.rs` unit tests (2 tests, was 0)
- [x] `BARRACUDA_REQUIREMENTS.md` benchmark count drift fixed (25/25, was "18/18")
- [x] `ipc/nestgate_data.rs` — NestGate CAS typed client (`content.store`, `content.get`, `storage.status`; 8 tests)
- [x] `ipc/squirrel_inference.rs` — Squirrel inference typed client (`inference.embed`, `inference.complete`, `inference.models`; 8 tests)
- [x] `data.weather` handler evolved: `data.open_meteo_weather` → `capability.call` (AG-008 RESOLVED)
- [x] `methods.rs` — 6 new constants (NestGate CAS + Squirrel inference)
- [x] Clippy pedantic+nursery zero warnings: 9 `#[must_use]` attrs added (`barracuda_route`, `provenance`, `skunkbat` ×4, `math` ×3)
- [x] Last `/tmp/` hardcoded path eliminated: `data/provider.rs` `SongbirdTransport::discover` → biomeOS standard discovery
- [ ] guideStone L5 / live NUCLEUS validation (blocked on live biomeOS + toadStool — Pass 14)
- [ ] guideStone L6 / cross-spring pipeline (deploy graphs validated against live NUCLEUS)

### Deep Debt Audit Results (May 13, 2026 — Sprint)

| Category | Finding |
|----------|---------|
| TODO/FIXME/HACK/XXX | **0** in production code |
| `unsafe` blocks (non-test) | **0** (`#![forbid(unsafe_code)]` enforced) |
| `unsafe fn` | **0** |
| Production mocks | **0** (all `Mock` types confined to `#[cfg(test)]`) |
| Files >800 LOC | **0** (largest: `validate_gpu_rewire_support.rs` at 775) |
| Hardcoded primal paths | **0** (last `/tmp/` in `provider.rs` eliminated this sprint) |
| `#[allow(` in production | **0** (all evolved to `#[expect()]` with reasons) |
| `todo!()` / `unimplemented!()` | **0** |
| `.unwrap()` in lib | **0** (only in `bin/` validation CLIs and test code) |
| Clippy pedantic+nursery | **0** warnings |
| External C deps | **0** (serde, clap, thiserror, toml, tracing — all pure Rust) |
| Edition | **2024** (Rust 1.92+) |

### Audit Questions — Answers

**Python baselines for barraCuda CPU parity:**
- **25/25** algorithms in `bench_cpu_vs_python` (ET₀ ×6, Soil ×5, Hydrology ×3, Crop ×4, Ecology ×5, Pipeline ×2)
- **1,284** Python control checks from `control/` scripts
- **Missing from 25-bench harness**: Turc, Hamon, SPI, MC ET₀, bootstrap/jackknife, kriging, autocorrelation, CN+GA coupled — these are validated elsewhere but not in the speed parity table

**Industry GPU benchmarks:**
- **Kokkos**: documented gap — Tier 1 Kokkos/Cabana performance reference not started (groundSpring V74 has 3.5×-2669× dispatch overhead data)
- **LAMMPS/SciPy/Galaxy**: referenced as methodology baselines, not as integrated GPU benchmark harnesses
- **GPU coverage is internal**: 21/21 CPU-GPU parity, 46/46 validate_gpu_math, 25 Tier A upstream ops

**Not implemented / tested:**
- ~~AG-005: Squirrel `inference.*`~~ **RESOLVED** — wired through `dispatch_science` (7 dispatch + 8 IPC tests)
- AG-006: coralReef sovereign shader compile not wired
- AG-007: `compute.dispatch` opaque JSON (no typed response contract)
- AG-010/011: TensorSession and Anderson WGSL shader
- L5/L6 certification: blocked on live primals
- Kokkos Tier 1 performance harness

**Unreviewed papers:** Papers #6, #7 (Tier 1, awaiting field data), #16 (Tier 3), #23/#24 (Tier 4/Future)

**Datasets to examine:** NOAA CDO, OpenWeatherMap (keyed access), Dong lab multi-sensor IoT + lysimeter (awaiting 2026), NCBI 16S (~50 GB 16S budget for metagenome pipeline)

---

**This document is maintained by airSpring and consumed by primalSpring.**
**See also**: `primalSpring/docs/PRIMAL_GAPS.md` (ecosystem-wide gap registry)
**See also**: `primalSpring/docs/CROSS_SPRING_PARITY_SCORECARD.md` (parity dashboard)
