# airSpring — Wave 20 PM lithoSpore Audit Absorption

**Date**: May 17, 2026
**Source**: airSpring v0.10.0
**Audience**: primalSpring, lithoSpore, barraCuda, biomeOS teams
**Registry**: 452 methods, 57 capabilities, stability tiers annotated

---

## What Was Done

### 1. Stability Tier Annotations

All 57 capabilities in `capability_registry.toml` annotated with `stability`:
- **53 stable**: All `science.*`, `ecology.*`, `provenance.*`, `primal.*`,
  `capability.*`, `health.*`, `composition.*`, `method.*`, `data.*` methods
- **4 evolving**: `compute.offload` (toadStool pipeline), `inference.embed`,
  `inference.complete`, `inference.models` (Squirrel inference)

Ecology aliases use canonical stable names — no wire-name drift found.
Cross-sync test confirms all 57 capabilities match registry exactly.

### 2. Degradation Behavior Documentation

New `docs/DEGRADATION_BEHAVIOR.md`:

| Primal | On Unreachable | Science Gated? |
|--------|----------------|:--------------:|
| biomeOS (Neural API) | `status: "unavailable"`, empty `primals_reached` | No |
| rhizoCrypt (DAG) | Falls to legacy, `status: "unavailable"` | No |
| loamSpine (commit) | `status: "partial"`, only rhizoCrypt reached | No |
| sweetGrass (braid) | `status: "partial"`, DAG+spine but no braid | No |
| NestGate | `Err(NoPrimal)` — local/control data used | No |
| toadStool | `Err(NoPrimal)` — validation skipped | No |
| barraCuda (precision) | `Err(NoPrimal)` — conservative local f64 | No |
| Squirrel | `Err(NoPrimal)` — degradation JSON payload | No |
| skunkBat | `None` + warn log | No |

Design principle: science never gates behind provenance or infrastructure.

### 3. Trio Transaction Semantics Alignment

`ProvenanceCompletion` struct evolved per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md`:
- **New field**: `primals_reached: Vec<&'static str>` — reports which trio
  primals were successfully contacted during the pipeline
- **Bug fix**: Legacy pipeline incorrectly reported `status: "complete"` when
  `create_braid` failed (braid_id empty). Now correctly reports `"partial"`
  with `primals_reached: ["rhizoCrypt", "loamSpine"]`
- **`to_json()`** serialization includes `primals_reached` for downstream consumers
- Domain logic never panics on partial provenance

### 4. Cross-Tier Parity Validators

Three new `validate_*` binaries close the parity gap for methods that had
Python baselines but no dedicated Rust cross-tier validator:

| Binary | Method | Benchmark JSON | Checks |
|--------|--------|----------------|--------|
| `validate_autocorrelation` | `science.autocorrelation` | `benchmark_autocorrelation.json` | ACF parity, white noise, AR(1) decay, constant data |
| `validate_gamma_cdf` | `science.gamma_cdf` | `benchmark_gamma_cdf.json` | Exponential, chi-squared, boundary, SPI-typical, large-alpha |
| `validate_soil_moisture_topp` | `science.soil_moisture_topp` | `benchmark_soil_moisture_topp.json` | Forward equation, monotonicity, roundtrip, published values |

Total parity matrix: 17 methods with full 3-tier coverage (notebook + JSON +
Rust validator), 7 with Tier 2 (JSON + Rust validator, no notebook narrative).

### 5. Thread 4 Expression Status

Foundation `ENVIRONMENTAL_GENOMICS.md` confirmed present (12+1 targets):
- **Validated**: FAO-56 ET₀ 36/36, FLS2 soil-immune 29/29 Rust
- **Pending**: No-till Anderson coupling (awaits field data)
- **Coordination**: wetSpring owns sovereign 16S pipeline and metagenomics;
  airSpring contributes soil physics and soil-immune coupling

---

## For Upstream Teams

### barraCuda
- `validate_autocorrelation` exercises `autocorrelation_cpu` / `normalised_acf_cpu`
  against Python mean-centred sums (different formula from averaged lag products);
  both formulas validated — parity confirmed within documented semantics
- `validate_gamma_cdf` exercises `gamma_cdf` → `regularized_gamma_p` delegation

### biomeOS
- `primals_reached` field now in all provenance JSON payloads — downstream
  consumers (lithoSpore, projectNUCLEUS) can inspect which trio primals
  were contacted without parsing empty string fields

### lithoSpore
- airSpring `control/*/benchmark_*.json` format aligns with lithoSpore
  `validation/expected/*.json` pattern: JSON objects with named checks,
  tolerances, and provenance metadata
- airSpring provides `--provenance-dir` for Tier 3 capture when ready

---

## Metrics

| Metric | Value |
|--------|-------|
| Capabilities | 57 (53 stable, 4 evolving) |
| Registry sync | 452 methods |
| Lib tests | 1,057 |
| Forge tests | 69 |
| Clippy warnings | 0 |
| Validator binaries | 97 (was 94) |
| Cross-tier full parity | 17 methods |
| Cross-tier Tier 2 parity | 7 methods |
| Deep debt | 0 |
