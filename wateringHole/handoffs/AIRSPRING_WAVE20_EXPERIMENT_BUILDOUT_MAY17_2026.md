# airSpring — Wave 20 Experiment Buildout + NUCLEUS Composition Handoff

**Date**: May 17, 2026
**From**: airSpring (v0.10.0)
**To**: primalSpring (L2 coordination), barraCuda, toadStool, coralReef, biomeOS teams, all delta springs
**Registry**: 452 methods (Wave 20, `primal.list` added)
**Capabilities**: 57 (science + ecology aliases + provenance + composition + coordination + inference)

---

## 1. What Changed (May 17, 2026)

### Experiment Buildout — All Green

| Suite | Result | Notes |
|-------|--------|-------|
| UniBin validation scenarios | **174/174 PASS** | Fixed sample/population std_dev; all 10 scenarios green |
| CPU-GPU parity | **37/37 PASS** | BatchedEt0, WB, diversity, tissue coupling |
| toadStool dispatch | **19/19 PASS** | 16 science methods + preflight + provenance |
| NUCLEUS graphs | **22/22 PASS** | Eco pipeline, soil microbiome, graph topology |
| Mixed pipeline | **66/66 PASS** | 7-stage GPU→NPU PCIe bypass, NUCLEUS mesh |
| Nucleus routing | **60/60 PASS** | 27 workloads, 20 absorbed, 4 NPU, 3 CPU |
| Lib tests | **1,057** | 0 failures |
| Forge tests | **69** | 7 new NUCLEUS composition tests |
| Clippy | **0 warnings** | pedantic + nursery, both crates |

### Wave 20 Debt Resolution

- **6 ecology aliases** registered: `ecology.et0_priestley_taylor`, `ecology.et0_makkink`, `ecology.et0_turc`, `ecology.et0_hamon`, `ecology.et0_blaney_criddle`, `ecology.timeseries` — capability count 51 → **57**
- **`capability.list` canonical envelope**: `{"capabilities": [...], "count": N, "primal": "airspring"}` per primalSpring schema standard
- **`unsafe` consolidation**: `testutil::EnvGuard` RAII pattern — one location instead of scattered blocks
- **`--provenance-dir`** implemented for Thread 5+6 capture

### Control Experiments Added

| Control | Script | Benchmark JSON | Checks |
|---------|--------|---------------|--------|
| `autocorrelation` | `autocorrelation_acf.py` | `benchmark_autocorrelation.json` | ACF on AR(1), white noise, constant data |
| `gamma_cdf` | `gamma_cdf_validation.py` | `benchmark_gamma_cdf.json` | Regularised incomplete gamma vs exponential/chi-squared |
| `soil_moisture_topp` | `soil_moisture_topp.py` | `benchmark_soil_moisture_topp.json` | Topp 1980 polynomial + inverse roundtrip |

**65 control scripts, 62 benchmark JSONs** across the full science domain.

---

## 2. NUCLEUS Atomic Composition — What We Learned

### Atomic Capability Counts (Current)

| Atomic | Capabilities | Components |
|--------|-------------|------------|
| **Tower** | 3: `crypto.tls`, `mesh.discovery`, `defense.audit` | BearDog + Songbird + SkunkBat |
| **Node** | 4: Tower + `compute.dispatch` | Tower + ToadStool |
| **Nest** | 4: Tower + `storage.provenance` | Tower + NestGate |

Key invariant: **Tower capabilities are a strict subset of both Node and Nest.** All atomics share the trust boundary; Node adds compute, Nest adds storage.

### Mixed Hardware Dispatch (metalForge)

Dispatch priority: **preferred substrate → GPU → NPU → Neural → CPU**

```
┌─────┐  PCIe P2P  ┌─────┐
│ NPU ├────────────►│ GPU │   ← bypass CPU roundtrip
└──┬──┘             └──┬──┘
   │ DMA               │ DMA
   ▼                   ▼
┌──────────────────────────┐
│        CPU memory        │   ← fallback only
└──────────────────────────┘
```

**Validated patterns:**
- NPU→GPU single-node pipeline: PCIe P2P bypass, 0 CPU roundtrips
- GPU→GPU same-device: zero-copy transfer
- Cross-node routing: sticky node preference, hop counting
- 27 eco workloads: **20 absorbed by barraCuda**, 4 NPU-native, 3 CPU-only
- All local WGSL shaders absorbed — zero `ShaderOrigin::Local` remaining

### biomeOS Graph Coordination

7 deploy graphs validated:
1. `airspring_eco_pipeline.toml` — weather → ET₀ → WB → yield (Tower prerequisite)
2. `airspring_provenance_pipeline.toml` — session → science → dehydrate → commit → attribute
3. `airspring_niche_deploy.toml` — full niche: Tower + Trio + NestGate + ToadStool + airSpring
4. `cross_primal_soil_microbiome.toml` — airSpring θ(t) → wetSpring diversity
5. `airspring_gpu_batch_deploy.toml` — GPU batch with Tower encryption tiers
6. `airspring_sovereign_data_deploy.toml` — NestGate-mediated sovereign data
7. `airspring_uncertainty_deploy.toml` — stochastic/UQ pipeline

---

## 3. Primal Consumption Map (airSpring's View)

| Primal | Methods Used | Status |
|--------|-------------|--------|
| **barraCuda** | `barracuda::ops::*` (20 GPU ops), `barracuda::stats::*` | **Direct crate dependency** (v0.4.0, wgpu 28) |
| **toadStool** | `toadstool.validate`, `toadstool.list_workloads`, `compute.dispatch.*` | **IPC client** — graceful absent |
| **bearDog** | `crypto.tls`, `mesh.discovery` (via Tower) | **Composition** — Tower atomic |
| **songBird** | `mesh.discovery`, sovereign transport | **Composition + data provider** |
| **skunkBat** | `defense.audit`, `security.audit_log` | **Composition** — Tower atomic |
| **nestGate** | `content.store`, `content.get`, `storage.status` | **IPC client** — CAS typed |
| **rhizoCrypt** | Provenance trio member | **Composition** — via signals |
| **loamSpine** | Provenance trio member | **Composition** — via signals |
| **sweetGrass** | Provenance trio member | **Composition** — via signals |
| **squirrel** | `inference.embed`, `inference.complete`, `inference.models` | **IPC client** — typed |
| **coralReef** | `coralreef.compile` (shader compile) | **Blocked** — AG-006 open |
| **petalTongue** | 3-tier discovery | **Low priority** — AG-009 |

### Wire Hygiene Learnings for Upstream

1. **bearDog** uses base64-encoded `message` field (not raw `data`) — parameter name matters for downstream parsing
2. **skunkBat** routes audit via `security.audit_log` (not `defense.audit`) — wire name differs from capability name
3. **`primal.announce`** single-call registration works cleanly with fallback for pre-v3.57 biomeOS
4. **`nest.store`** signal dispatch collapses `content.put → dag.event.append → spine.seal` — biomeOS manages the graph
5. **`nest.commit`** signal dispatch collapses `dehydrate → commit → attribute` — session finalization in one call

---

## 4. What Upstream Primals Should Know

### For barraCuda Team

- All 20 GPU ops (`BatchedElementwiseF64` ops 0–19) are validated at **f64 precision** against Python baselines (25/25 parity, 21/21 CPU-GPU parity modules)
- `BrentGpu` (VG inverse) and `RichardsGpu` (Picard iteration) are validated and absorbed
- **Suggestion**: Consider `autocorrelation_gpu` path — we have CPU path via `autocorrelation_cpu()` but GPU ACF would benefit time-series heavy workloads
- airSpring's `tolerance` framework (60 named tolerances across 5 domain submodules) could be upstreamed as a `barracuda::tolerances` module pattern

### For toadStool Team

- `toadstool.validate` preflight check is exercised in composition scenario + dedicated validator (19/19)
- `compute.dispatch.submit/result/capabilities` client is typed and feature-gated
- **Gap AG-007**: Typed `compute.dispatch` contract — current JSON params are untyped; a schema would prevent misrouted workloads
- All 27 metalForge eco workloads have been classified for compute dispatch routing

### For coralReef Team

- **AG-006** remains open: `coralreef.compile` IPC client exists but shader compilation wiring awaits coralReef evolution
- We can provide WGSL shader source strings for ecology-domain kernels when the compile API stabilizes

### For biomeOS Team

- `capability.list` canonical envelope adopted: `{"capabilities": [...], "count": N}`
- `primal.list` synced against (452-method registry, Wave 20)
- Signal dispatch (`nest.store`, `nest.commit`) working with graceful fallback
- 7 deploy graphs parse cleanly through the graph validation engine
- `NeuralBridge::discover()` with 4-tier socket resolution works reliably

---

## 5. For Other Delta Springs

### Composition Patterns We Validated

1. **Tower atomic prerequisite**: All pipelines require Tower (crypto + mesh + defense) before Node or Nest
2. **Signal dispatch with fallback**: `nest.store` / `nest.commit` with legacy `capability.call` fallback — adoptable pattern for all springs
3. **`primal.announce` single-call**: Replaces 3-call registration — all springs should adopt
4. **Mixed hardware routing**: metalForge dispatches across GPU/NPU/CPU/Neural with PCIe P2P bypass — same substrate/dispatch architecture works for any domain
5. **Control experiment triad**: Python script + benchmark JSON + provenance SHA-256 — reusable pattern for any validation domain

### Cross-Spring Data Exchange

- `cross_primal_soil_microbiome.toml` graph demonstrates airSpring→wetSpring data flow (soil moisture time series → microbial diversity analysis)
- Data exchange uses NestGate CAS for immutable artifact storage with BLAKE3 content addressing
- Any spring with provenance needs can adopt the same `nest.store` → `nest.commit` signal pattern

---

## 6. Open Gaps (airSpring-Specific)

| ID | Gap | Blocker | Priority |
|----|-----|---------|----------|
| AG-006 | coralReef shader compile IPC | coralReef API evolution | Medium |
| AG-007 | Typed `compute.dispatch` contract | toadStool schema | Low |
| AG-009 | petalTongue direct IPC | Low ROI | Low |
| AG-010 | TensorSession pooling | squirrel evolution | Low |
| AG-011 | Anderson GPU shader | barraCuda PDE evolution | Low |
| E3 LTEE | FLS2 plant immunity reproduction | Science execution | Medium |

**Zero deep debt remaining**: 0 `todo!()`, 0 `unimplemented!()`, 0 `FIXME`, 0 `HACK`, 0 production mocks, 0 unsafe in production, 0 files >800 LOC.

---

## 7. Metrics Summary

| Metric | Value |
|--------|-------|
| Version | v0.10.0 |
| Capabilities | 57 (science + ecology + provenance + composition + inference) |
| Lib tests | 1,057 |
| Forge tests | 69 |
| Total tests | 1,442 (barracuda + forge) |
| Python baselines | 1,284/1,284 |
| Binaries | 94 |
| Deploy graphs | 7 |
| UniBin scenarios | 10 (174/174 PASS) |
| Control scripts | 65 |
| Benchmark JSONs | 62 |
| Methods constants | 64 |
| Capabilities registry | 57 (synced vs 452 canonical) |
| Clippy | 0 warnings (pedantic + nursery) |
| Coverage | 90.56% (cargo llvm-cov) |
| Deep debt | 0 (all dimensions) |
| guideStone | L4 (targeting L6) |

---

*primalSpring will pull and review on next wave.*
