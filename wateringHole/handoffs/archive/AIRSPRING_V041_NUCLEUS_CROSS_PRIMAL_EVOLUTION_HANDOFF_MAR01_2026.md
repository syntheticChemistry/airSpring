# airSpring → ToadStool / biomeOS / NUCLEUS Cross-Primal Evolution Handoff V041

**Date**: March 1, 2026
**From**: airSpring v0.5.8 (biomeGate) — 63 experiments, 72 binaries, 641 lib tests, 16 capabilities
**To**: ToadStool/BarraCuda, biomeOS, NestGate, BearDog, Songbird
**Supersedes**: V040 (archived)
**ToadStool Pin**: `1dd7e338` (S70+++)
**License**: AGPL-3.0-or-later

---

## Executive Summary

1. **airSpring is now a live biomeOS primal** — 16 capabilities, ecology domain, Unix socket JSON-RPC
2. **Cross-primal pipeline proven** — capability.call routing via neural-api, 28/28 PASS (Exp 063)
3. **5 new experiments** (059-063): atlas decade 80yr (102/102), NASS real yield (99/99), NCBI diversity (63/63), NUCLEUS integration (29/29), NUCLEUS cross-primal pipeline (28/28)
4. **capability_call node type wired** in biomeOS graph executor — fixes all science pipeline graphs
5. **Bug found + fixed**: `capability.register` expected `socket`, airSpring was sending `socket_path`
6. **72 binaries**, 63 experiments, zero unsafe, zero TODOs, zero clippy warnings

---

## Part 1: What Changed (v0.5.7 → v0.5.8)

### New Experiments

| Exp | Name | Checks | Key Finding |
|-----|------|:------:|-------------|
| 059 | Atlas 80yr Decade Analysis | 102/102 | Open-Meteo 1944-2024 decadal ET₀ + precipitation trends (pure Rust ureq) |
| 060 | NASS Real Yield Comparison | 99/99 | Stewart (1977) vs synthetic NASS corn/soy/wheat parity |
| 061 | Cross-Spring Diversity (NCBI 16S) | 63/63 | Shannon H', Pielou, Bray-Curtis, Anderson coupling |
| 062 | NUCLEUS Integration Validation | 29/29 | JSON-RPC science parity — 9 methods return identical values to direct Rust |
| 063 | NUCLEUS Cross-Primal Pipeline | 28/28 | ecology domain, capability.call routing, cross-primal forwarding, primal discovery |

### New Binaries

| Binary | Purpose |
|--------|---------|
| `airspring_primal` | biomeOS NUCLEUS primal (16 capabilities, Unix socket JSON-RPC) |
| `validate_nucleus` | Exp 062 — primal parity validation |
| `validate_nucleus_pipeline` | Exp 063 — cross-primal pipeline validation |
| `validate_atlas_decade` | Exp 059 — decadal Open-Meteo analysis |
| `validate_nass_real` | Exp 060 — NASS yield parity |
| `validate_ncbi_diversity` | Exp 061 — cross-spring diversity |

### airSpring Primal Capabilities (16)

```
science.et0_fao56          science.et0_hargreaves      science.et0_priestley_taylor
science.et0_makkink        science.et0_turc             science.et0_hamon
science.et0_blaney_criddle science.water_balance         science.yield_response
ecology.et0_fao56          ecology.water_balance         ecology.yield_response
ecology.full_pipeline      primal.forward                primal.discover
science.health
```

### Data Providers (Pure Rust)

| Provider | Crate | Protocol | Data |
|----------|-------|----------|------|
| `data::open_meteo` | ureq + serde_json | HTTPS REST | 80yr weather (1944-2024, 100 stations) |
| `data::usda_nass` | ureq + serde_json | HTTPS REST | County-level crop yields (corn/soy/wheat) |

---

## Part 2: biomeOS Integration Details

### What We Wired

```
airspring_primal (Unix socket)
  ├── lifecycle.register → neural-api
  ├── capability.register → neural-api (ecology domain + 16 capabilities)
  │   └── semantic_mappings: ecology.et0_fao56 → science.et0_fao56, etc.
  ├── dispatch (JSON-RPC 2.0)
  │   ├── science.* methods → direct calculation
  │   ├── ecology.* methods → alias to science.*
  │   ├── ecology.full_pipeline → ET₀ → water_balance → yield_response chain
  │   ├── primal.forward → proxy call to any discovered primal
  │   └── primal.discover → list all .sock files in socket directory
  └── health check (science.health → 16 capabilities)
```

### Bug Found: capability.register Field Name

**Symptom**: `capability.call ecology.et0_fao56` returned "Capability 'ecology' not registered" despite successful `lifecycle.register`.

**Root cause**: `biomeos-atomic-deploy/src/handlers/capability.rs` `CapabilityHandler::register` reads `params["socket"]`, but airSpring was sending `params["socket_path"]`. The lifecycle registration succeeded (it doesn't need the socket path), but capability routing failed silently.

**Fix**: Changed `airspring_primal.rs` to send `"socket"` instead of `"socket_path"`. Added `semantic_mappings` to the ecology domain registration for explicit method routing.

**Upstream implication**: Any new primal registering capabilities must use `"socket"` as the field name. Consider adding validation/error messages in `CapabilityHandler::register` when `"socket"` is missing.

### biomeOS Changes Made (upstream in phase2/biomeOS/)

| File | Change |
|------|--------|
| `config/capability_registry.toml` | Added `[domains.ecology]` (7 capability keywords) + `[translations.ecology]` (9 method translations) |
| `capability_domains.rs` | Added airSpring ecology domain to fallback array (10/10 tests pass) |
| `neural_executor.rs` | Wired `"capability_call"` node type — routes via neural-api, falls back to direct primal RPC |

### Graph: airspring_ecology_pipeline.toml

```
weather_data → et0_fao56 → et0_hargreaves → water_balance → yield_response → validation
        (health)  (capability_call)  (capability_call)    (capability_call)   (capability_call)
```

---

## Part 3: What Each Team Needs

### For ToadStool/BarraCuda Team

| # | Action | Priority | Context |
|---|--------|:--------:|---------|
| 1 | Absorb ops 5-8 into GPU-first dispatch | Medium | CPU fallback validated, GPU dispatch activates automatically when ToadStool absorbs |
| 2 | Profile `batched_elementwise_f64` at N=100K+ | Low | Multi-year regional grids need crossover point (AtlasStream) |
| 3 | Consider `ecology` op codes | Low | airSpring's science methods could map to dedicated ToadStool ops for zero-copy GPU pipelines |
| 4 | Streaming pipeline absorption | Medium | `GpuPipelined` backend pre-computes Kc, batches ET₀→WB→yield — candidate for ToadStool unidirectional streaming |

### For biomeOS Team

| # | Action | Priority | Context |
|---|--------|:--------:|---------|
| 1 | Add socket field validation in `CapabilityHandler::register` | High | Silent failure when wrong field name — cost us debugging time |
| 2 | Log capability.register success with socket path | Medium | Would have caught the socket_path/socket mismatch immediately |
| 3 | Consider capability.register bulk API | Low | airSpring registers 16 capabilities individually — one call with array would be cleaner |

### For NestGate Team

| # | Action | Priority | Context |
|---|--------|:--------:|---------|
| 1 | Unix socket NUCLEUS integration | High | NestGate is HTTP standalone; NUCLEUS Tower discovers via socket directory |
| 2 | Open-Meteo + NASS as NestGate providers | Medium | airSpring currently uses ureq directly; NestGate already has provider trait |
| 3 | Content-addressed blob storage for weather data | Low | 80yr × 100 stations = ~600MB; store once, provenance forever |

### For BearDog/Songbird

No action required. Cross-primal forwarding from airSpring to BearDog and Songbird works (tested via `primal.forward` in Exp 063).

---

## Part 4: Cross-Spring Shader Evolution Update

ToadStool S70+++ (774 WGSL shaders). No new shader requirements from airSpring in v0.5.8. All GPU work uses existing `batched_elementwise_f64`, `kriging_f64`, `fused_map_reduce_f64`, and pipeline primitives.

| Spring | Shaders Used by airSpring | Status |
|--------|--------------------------|--------|
| hotSpring | df64 core, pow/exp/log/trig | Stable — no changes |
| wetSpring | kriging_f64, fused_map_reduce, moving_window | Stable — diversity via CPU |
| neuralSpring | nelder_mead, multi_start | Stable |
| groundSpring | MC ET₀ uncertainty (xoshiro + Box-Muller) | Tier B wired |

---

## Part 5: Barracuda Evolution Summary

### Current Architecture (42,207 lines Rust)

```
barracuda/src/
├── eco/          20 modules — 8 ET₀ methods, water balance, dual Kc, Richards, soil, diversity, runoff, infiltration, yield
├── gpu/          17 modules — ToadStool bridge, 11 Tier A + 4 Tier B orchestrators, seasonal pipeline, atlas stream
├── data/         4 modules — provider trait, open_meteo, usda_nass, weather
├── npu/          2 modules — AKD1000 inference + streaming (feature-gated)
├── io/           1 module — CSV time series parser
├── testutil/     3 modules — generators, stats, bootstrap
├── validation/   — ValidationHarness (neuralSpring origin)
├── tolerances/   — centralized validation thresholds
├── error/        — AirSpringError unified type
└── bin/          68 source files (validate_*, bench_*, airspring_primal) — 72 Cargo.toml entries
```

### Quality Gates

| Gate | Status |
|------|--------|
| `cargo test --lib` | 641 passed, 0 failures |
| `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery` | 0 warnings |
| `cargo fmt --check` | Clean |
| `unsafe` blocks | **Zero** |
| `.unwrap()` in lib | **Zero** (binaries use for CLI args only) |
| `todo!()` / `FIXME` / `HACK` | **Zero** |
| Dead code (`#[allow(dead_code)]`) | **Zero** |

---

## Part 6: Activation Sequence

```bash
# Start NUCLEUS Tower
cd /home/eastgate/Development/ecoPrimals/phase2/biomeOS
cargo run -p biomeos-unibin --bin biomeos -- nucleus --mode tower --node-id eastgate

# Start airSpring primal
cd /home/eastgate/Development/ecoPrimals/airSpring/barracuda
cargo run --release --bin airspring_primal

# Validate NUCLEUS integration (standalone)
cargo run --release --bin validate_nucleus

# Validate cross-primal pipeline (requires NUCLEUS Tower + airSpring primal running)
cargo run --release --bin validate_nucleus_pipeline
```

---

Unidirectional handoff — no response expected.
airSpring is a consumer of BarraCuda primitives, a consumer of NUCLEUS services, and a provider of ecology science capabilities. Handoffs communicate what we learned, what we need, and what we contribute back.

*airSpring v0.5.8 — 63 experiments, 72 binaries, 641 lib tests, 16 capabilities, 28/28 cross-primal pipeline. AGPL-3.0-or-later.*
