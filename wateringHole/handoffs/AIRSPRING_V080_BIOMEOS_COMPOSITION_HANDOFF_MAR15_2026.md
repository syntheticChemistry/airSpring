# airSpring V0.8.0 — biomeOS Composition Integration Handoff

> **Date**: March 15, 2026
> **From**: airSpring
> **To**: barraCuda, ToadStool, biomeOS, rhizoCrypt, loamSpine, sweetGrass, NestGate
> **Status**: 847 lib + 41 integration tests, 0 clippy warnings, `--all-features` compile

---

## Executive Summary

airSpring v0.8.0 integrates with the biomeOS primal ecosystem via capability-based
composition. The primal now exposes **41 capabilities** (up from 35) including provenance
lifecycle, cross-spring data exchange, and NestGate content-addressed caching.

All composition uses biomeOS `capability.call` routing — **zero compile-time coupling**
to external primals. Graceful degradation when any composition partner is unavailable.

---

## §1 Provenance Trio Integration

**File**: `barracuda/src/ipc/provenance.rs`

### Capability Routing

| Capability | Operation | Primal | Purpose |
|------------|-----------|--------|---------|
| `dag` | `create_session` | rhizoCrypt | Begin experiment DAG session |
| `dag` | `append_event` | rhizoCrypt | Record experiment step as DAG vertex |
| `dag` | `dehydrate` | rhizoCrypt | Content-addressed Merkle root |
| `commit` | `session` | loamSpine | Immutable ledger entry |
| `provenance` | `create_braid` | sweetGrass | W3C PROV-O attribution braid |

### Graceful Degradation

| Condition | Behavior | Experiment Status |
|-----------|----------|-------------------|
| Socket missing | Return `Ok` + `"unavailable"` | Succeeds |
| Dehydrate fails | Return `Ok` + `"unavailable"` | Succeeds |
| Commit fails | Return `Ok` + `"partial"` | Dehydration preserved |
| Braid fails | Return `Ok` + `"complete"` (empty braid_id) | Commit preserved |

### GPU Compute Provenance

`record_gpu_step()` tracks shader chain executions in the DAG:

```json
{
  "type": "gpu_compute",
  "shader": "fao56_et0_batch",
  "precision": "f64",
  "input_content_hash": "sha256:abc123",
  "output_summary": {"mean_et0_mm": 4.8},
  "backend": "barracuda_wgsl"
}
```

---

## §2 NestGate Content-Addressed Routing

**File**: `barracuda/src/data/provider.rs` (`NestGateProvider`)

Three-tier routing (following wetSpring `ncbi/nestgate/` pattern):

1. **NestGate cache** — `storage.get` via primal discovery, key = `airspring:weather:{lat}:{lon}:{start}:{end}`
2. **biomeOS capability** — `capability.call("ecology", "fetch_weather", ...)` via Neural API
3. **BiomeosProvider fallback** — direct primal discovery

Cached results stored in Cross-Spring Time Series v1 format, enabling offline
reproducibility and reducing redundant API calls across experiments.

---

## §3 Cross-Spring Time Series v1

**File**: `barracuda/src/data/provider.rs` (`WeatherResponse`)

### Format

```json
{
  "schema": "ecoPrimals/time-series/v1",
  "source": { "spring": "airspring", "provider": "nestgate_routed", "version": "0.8.0" },
  "variables": [
    { "variable": "temperature_max", "unit": "°C", "timestamps": [...], "values": [...] },
    { "variable": "temperature_min", "unit": "°C", "timestamps": [...], "values": [...] },
    { "variable": "precipitation", "unit": "mm", "timestamps": [...], "values": [...] }
  ],
  "metadata": { "count": 365, "has_solar_radiation": true, ... }
}
```

### API

- `WeatherResponse::to_cross_spring_v1(source)` — serialize to v1 format
- `WeatherResponse::from_cross_spring_v1(json)` — deserialize from v1 format
- Forward-compatible: unknown variables ignored during deserialization

---

## §4 Primal Binary Evolution

**File**: `barracuda/src/bin/airspring_primal.rs`

### New Capabilities (6)

| Method | Purpose | Composition |
|--------|---------|-------------|
| `provenance.begin` | Begin provenance-tracked experiment session | → rhizoCrypt |
| `provenance.record` | Record experiment step in DAG | → rhizoCrypt |
| `provenance.complete` | Complete: dehydrate → commit → attribute | → Trio |
| `provenance.status` | Check trio availability | → Trio |
| `capability.list` | List all capabilities + composition status | Self |
| `data.cross_spring_weather` | Fetch via NestGate, return v1 format | → NestGate |

### `capability.list` Response

```json
{
  "primal": "airspring",
  "version": "0.8.0",
  "domain": "ecology",
  "total": 41,
  "science": ["science.et0_fao56", ...],
  "infrastructure": ["provenance.begin", "capability.list", ...],
  "composition": {
    "provenance_trio": true,
    "nestgate": true,
    "toadstool": false
  }
}
```

---

## §5 biomeOS Deploy Graphs

### `airspring_provenance_pipeline.toml`

Provenance-tracked experiment pipeline:

```
check_airspring → begin_session → execute_science → record_step → complete_provenance → store_results
```

Parameterized by `EXPERIMENT_NAME`, `METHOD`, and `PARAMS`. Stores results in
NestGate with provenance metadata.

### `airspring_niche_deploy.toml`

Full niche deployment (8 ordered nodes):

```
BearDog → Songbird → rhizoCrypt → loamSpine → sweetGrass → NestGate → ToadStool → airSpring
```

Provenance trio and NestGate are `required = false` — airSpring starts regardless.
`by_capability = "ecology"` routes ecology.* calls to airSpring.

### `metalForge/deploy/airspring_deploy.toml` (updated)

Added rhizoCrypt, loamSpine, sweetGrass nodes (order 3-5). ToadStool → order 6,
airSpring → order 7. Provenance capabilities added to airSpring node.

---

## §6 Composition Opportunities for biomeOS

### Available Composition Points

1. **Science composition**: Any primal can call `ecology.*` capabilities via biomeOS
2. **Provenance composition**: Any graph can include `provenance.begin/record/complete` nodes
3. **Data composition**: `data.cross_spring_weather` returns canonical v1 format
4. **Health composition**: `health.check` and `capability.list` for orchestration decisions

### biomeOS Evolution Opportunities

- **Graph templates**: `airspring_provenance_pipeline.toml` is parameterized — biomeOS
  could evolve this into a generic `spring_provenance_experiment.toml` template
- **Capability aggregation**: `capability.list` reports composition status — biomeOS
  could use this for dynamic graph rewiring when primals come online/offline
- **Cross-spring pipelines**: `cross_primal_soil_microbiome.toml` demonstrates multi-spring
  composition — biomeOS could evolve graph merging for compound experiments

---

## §7 Quality Gates

| Gate | Status |
|------|--------|
| `cargo test --lib` | **847/847 PASS** |
| `cargo test --tests` | **41/41 PASS** |
| `cargo clippy --all-targets` | **0 warnings** |
| `cargo fmt --check` | **Clean** |
| `cargo doc --no-deps` | **Clean** |
| `--all-features` compile | **PASS** |

---

## §8 Next Steps

1. **E2E provenance test**: Start rhizoCrypt + loamSpine + sweetGrass locally, run
   `provenance.begin` → `science.et0_fao56` → `provenance.complete`, verify Merkle root
2. **NestGate cache test**: Start NestGate, run `data.cross_spring_weather`, verify
   cache hit on second call
3. **Graph execution**: `biomeos graph exec --graph graphs/airspring_provenance_pipeline.toml`
4. **Niche deployment**: `biomeos deploy --graph graphs/airspring_niche_deploy.toml`
