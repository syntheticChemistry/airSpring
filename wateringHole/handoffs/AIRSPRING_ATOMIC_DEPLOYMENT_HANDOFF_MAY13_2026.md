# airSpring → Upstream Primal & Spring Teams Handoff

**Date**: May 13, 2026
**From**: airSpring (ecology / agriculture) — v0.10.0
**For**: primalSpring (coordination), upstream primal teams, delta spring teams
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring is at **zero code debt**, **1,057 lib tests** (1,435 total), **49 capabilities**, **zero clippy pedantic+nursery warnings**, and **zero unsafe in production**. Full deep debt audit complete. All structural L5 requirements met — blocked only on live primal deployment for certification. This handoff documents everything we learned about primal use, composition patterns, and evolution opportunities for upstream teams.

---

## Primal Consumption Map

airSpring consumes 10 of the 13 NUCLEUS primals. Here is what we use, how we use it, and what we learned.

### biomeOS (coordination)

| Method | Usage | Notes |
|--------|-------|-------|
| `method.register` | Startup — register all 49 capabilities | Works cleanly. Capability routing is the core composition pattern. |
| `capability.list` | Discovery — enumerate peer capabilities | Used for graceful degradation when primals are absent |
| `capability.call` | Runtime — route domain requests through NUCLEUS | Primary science dispatch path. Semantically routes `ecology.*` and `science.*` |
| Socket discovery | `BIOMEOS_SOCKET` env → XDG scan → `/run/biomeos/` | Three-tier pattern is robust. `primal_names::socket_filename()` helper is reusable. |

**Evolution opportunity**: `capability.call` routing should support batch payloads (array of operations) for pipeline composition — currently one-at-a-time.

### barraCuda (math primitives)

| Integration | Details |
|-------------|---------|
| Tier A GPU modules | **25** (all 20 upstream `BatchedElementwiseF64` + 5 dedicated) |
| CPU-GPU parity | **21/21** modules validated |
| WGSL shaders consumed | 6 families (from hotSpring, wetSpring, neuralSpring, groundSpring) |
| Upstream contributions | Richards PDE (S40), stats metrics (S64), 6 ops 14-19 absorbed |
| `precision.route` | Typed client consuming `requires_compiler`, `adapter`, precision advisory |
| Version | **0.4.0** (wgpu 28, DF64 transcendentals) |

**Key learning**: The Write → Absorb → Lean cycle works. All 6 local GPU ops were absorbed upstream. `local_dispatch` retired. `PrecisionRoutingAdvice` is excellent for per-silicon dispatch decisions.

**Evolution opportunity**: `compute.dispatch` response should be typed (AG-007 — currently opaque JSON). TensorSession for streamed GPU workloads (AG-010).

### toadStool (compute dispatch)

| Method | Usage |
|--------|-------|
| `toadstool.validate` | Workload pre-flight check (8 TCP round-trip tests) |
| `toadstool.list_workloads` | Enumerate available dispatch targets |
| `compute.dispatch` | GPU offload through NUCLEUS mesh |
| `compute.offload` | Alternative dispatch path for science methods |

**Key learning**: Pre-flight validation via `toadstool.validate` catches workload/hardware mismatches before dispatch. The validate → dispatch → result pattern is clean.

### NestGate (content-addressed storage)

| Method | Usage | Tests |
|--------|-------|:-----:|
| `content.store` | CAS blob store (experiment artifacts, weather cache) | 8 |
| `content.get` | Retrieve by BLAKE3 hash | 8 |
| `storage.status` | Health + capacity check | 8 |

**Key learning**: NestGate is a **storage** primal, not a data primal. It does not implement `data.*` methods. airSpring initially called `data.open_meteo_weather` on NestGate — this was wrong. The correct pattern is `capability.call` with `operation: "weather.daily"` for data routing. AG-008 was resolved by fixing this.

**Evolution opportunity**: NestGate should surface `content.store` deduplication stats in the response (currently returns hash + size + deduplicated flag, which is good).

### Squirrel (inference)

| Method | Usage | Tests |
|--------|-------|:-----:|
| `inference.embed` | Soil sensor similarity search (ecology use case) | 8+7 |
| `inference.complete` | Structured inference for experiment annotation | 8 |
| `inference.models` | Model discovery for capability probing | 8 |

**Key learning**: `inference.embed` is the primary Squirrel use case for airSpring — embedding soil sensor readings for similarity search across stations. The typed client returns `EmbedResult { embedding, model, dimensions }`. AG-005 was resolved by wiring these through `dispatch_science`.

**Evolution opportunity**: Squirrel `auto-discovery` should surface model capabilities (e.g., embedding dimension, supported input types) so springs can match models to domains.

### BearDog (health/crypto)

| Method | Usage |
|--------|-------|
| `health.liveness` | Health probe only |

**Key learning**: airSpring does not use BearDog for crypto payloads. Only `health.liveness` — correct per wire hygiene audit. ludoSpring found that BearDog uses base64 message encoding (not raw data) for `crypto.*` — we never hit this because we don't call `crypto.*`.

### Songbird (data transport)

| Method | Usage |
|--------|-------|
| Sovereign transport | All HTTP data access routes through Songbird IPC |

**Key learning**: Replacing `ureq` with Songbird IPC (v0.8.5) eliminated the last C dependency. `SongbirdTransport::discover` uses standard `biomeos::discover_primal_socket(SONGBIRD)` — no hardcoded paths.

### skunkBat (audit)

| Method | Usage |
|--------|-------|
| `security.audit_log` | Startup audit, certification events, experiment provenance |
| `security.audit_certification` | L0-L6 certification layer events |

**Key learning**: skunkBat routes audit via `security.audit_log` (not `defense.audit`). ludoSpring's wire correction is confirmed correct. airSpring emits audit events at startup (capability count) and during certification transitions.

### Provenance Trio (rhizoCrypt + loamSpine + sweetGrass)

| Pattern | Usage |
|---------|-------|
| `provenance.begin` → `provenance.record` → `provenance.complete` | Experiment session tracking |
| `provenance.status` | Trio availability check |

**Key learning**: The session model (begin → record → complete) maps cleanly to scientific experiments. `dehydrate → commit → attribute` is the internal trio handshake. Transport discovery uses `resolve_neural_api_transport` for the trio.

### Primals NOT Consumed

| Primal | Reason |
|--------|--------|
| coralReef | Shader compilation — airSpring consumes pre-compiled WGSL, doesn't compile shaders (AG-006 open) |
| petalTongue | 3-tier discovery wired but no visualization use case yet |

---

## Composition Patterns for NUCLEUS

### Pattern 1: Capability-Based Routing (primary)

```
client → biomeOS → capability.call("ecology.et0_fao56", params)
                      → discovers airSpring owns "ecology.et0_fao56"
                      → routes to airSpring socket
                      → airSpring computes ET₀
                      → returns result through biomeOS
```

This is the cleanest pattern. The client never needs to know which primal provides what.

### Pattern 2: Cross-Primal Forwarding

```
client → airSpring → primal.forward("science.diversity", params)
                       → airSpring resolves: wetSpring owns diversity
                       → forwards via biomeOS capability.call
                       → wetSpring computes
                       → result returns through chain
```

Used for cross-domain requests where airSpring is the entry point but another spring owns the computation.

### Pattern 3: Provenance-Tracked Science Pipeline

```
airSpring → provenance.begin(session_id)
          → science.et0_fao56(params) → provenance.record(step)
          → science.water_balance(params) → provenance.record(step)
          → provenance.complete(session_id)
          → skunkBat audit_log(session_summary)
```

Full experiment traceability from input to output, with cryptographic commitment via the trio.

### Pattern 4: Deploy Graph Execution

7 TOML deploy graphs define DAG pipelines:
- `airspring_eco_pipeline.toml` — weather → ET₀ → WB → yield
- `airspring_provenance_pipeline.toml` — session → science → dehydrate → commit → attribute
- `airspring_niche_deploy.toml` — Tower + Trio + NestGate + ToadStool + airSpring (9 nodes)
- `airspring_gpu_batch_deploy.toml` — barraCuda batched dispatch stages
- `airspring_sovereign_data_deploy.toml` — NestGate/Songbird data routing
- `airspring_uncertainty_deploy.toml` — bootstrap, jackknife, MC layers
- `cross_primal_soil_microbiome.toml` — airSpring θ(t) → wetSpring diversity

### Pattern 5: neuralAPI Deployment

biomeOS `neuralAPI` bridges NUCLEUS composition to external consumers:
- Structured metrics with `operation_dependencies` and `cost_estimates`
- Pathway Learner integration for adaptive routing
- `method.register` at startup seeds the neuralAPI routing table

---

## What We Learned for Upstream Evolution

### For primal teams

1. **Wire name hygiene matters**: airSpring initially called `data.open_meteo_weather` on NestGate — a method that doesn't exist. Every primal should publish its exact wire surface in a canonical spec.

2. **Three-tier discovery is robust**: `env override → named socket scan → capability probe` handles every deployment scenario. All springs should use this pattern.

3. **`#[must_use]` on IPC results**: Functions that return `Result<T>` from primal calls should always be `#[must_use]` — prevents silently ignoring errors.

4. **Graceful degradation is essential**: Every primal call must handle "peer not available" gracefully. airSpring skips NestGate/Squirrel probes in composition validation if those primals aren't running.

5. **Edition 2024 `env::set_var` requires `unsafe`**: Test cleanup code that calls `std::env::set_var()` must be wrapped in `unsafe` blocks in Edition 2024. This is the only `unsafe` in our codebase — all in `#[cfg(test)]`.

### For spring teams

1. **Write → Absorb → Lean works**: airSpring wrote 6 GPU ops locally, validated them, handed off to barraCuda, confirmed absorption, then deleted local code. Other springs should follow this cycle.

2. **Tier 4 IPC-first is the target**: `[features].default = []` means the crate compiles and tests without barraCuda source. Pure-Rust fallbacks in `math.rs` handle the IPC-only case. This enables `plasmidBin` deployment where only the binary ships.

3. **`methods.rs` centralized constants prevent drift**: 61 constants in one file, sync-tested against `capability_registry.toml`. No inline string literals for method names.

4. **60 named tolerances prevent magic numbers**: All numeric thresholds in `tolerances/` with doc comments explaining mathematical justification. Python mirror file keeps baselines aligned.

---

## Atomic Deployment Readiness

| Criterion | Status |
|-----------|--------|
| UniBin binary | **3.3 MB** static-pie (musl), plasmidBin harvestable |
| Validation scenarios | **10** (ScenarioRegistry, all passing) |
| `--format json` | Supported for Tier 2 ingestion |
| guideStone level | **L4** (structural L5, blocked on live primals) |
| `method.register` | Wired — 49 capabilities registered at startup |
| Health probes | `health.liveness` + `health.readiness` |
| Graceful degradation | All IPC calls handle peer-absent |
| Zero debt | Confirmed (see Deep Debt Sprint handoff) |

**Blocked on**: Live primal deployment for L5 certification. Per directive: holding on full NUCLEUS compositions until Tower (ludoSpring), Node (hotSpring), and Nest (healthSpring) confirm live atomic validation.

---

## Open Gaps (not debt — external dependencies)

| Gap | Owner | Status |
|-----|-------|--------|
| AG-006: coralReef shader compile | coralReef team | Open — `shader.compile.wgsl` not wired |
| AG-007: compute.dispatch typing | toadStool team | Open — response is opaque JSON |
| AG-010: TensorSession | barraCuda roadmap | Open — streamed GPU workloads |
| AG-011: Anderson WGSL shader | barraCuda Tier C | Open |
| Kokkos Tier 1 harness | barraCuda domain | Not started |
| LTEE E3 reproduction | airSpring | Queued — primary paper queue item |
| L5/L6 certification | Blocked on live primals | Structural complete |

---

## Metrics Snapshot

| Metric | Value |
|--------|-------|
| Lib tests | **1,057** |
| Total tests | **1,435** |
| Python baselines | **1,284** (60 papers) |
| CPU vs Python parity | **25/25** (14.3× geomean) |
| Capabilities | **49** |
| Method constants | **61** |
| IPC modules | **13** |
| Deploy graphs | **7** |
| Validation scenarios | **10** |
| Binaries | **94** |
| GPU modules | **25 Tier A** |
| WGSL shaders consumed | **767+** |
| Tolerances | **60** named |
| Edition | **2024** (Rust 1.92+) |
| Unsafe (non-test) | **0** |
| Clippy pedantic+nursery | **0** warnings |
| C dependencies | **0** |
| guideStone | **L4** (structural L5) |

---

**This document consumed by primalSpring and all upstream primal/spring teams.**
See also: `docs/PRIMAL_GAPS.md`, `wateringHole/handoffs/AIRSPRING_DEEP_DEBT_SPRINT_MAY13_2026.md`
