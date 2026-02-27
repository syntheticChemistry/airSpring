# biomeOS / ToadStool — V025 Neural API Integration Handoff

**Date**: 2026-02-27
**From**: airSpring v0.4.13
**To**: biomeOS core team, ToadStool / BarraCuda core team
**License**: AGPL-3.0-or-later
**Covers**: biomeOS Neural API bridge, ecology capability domain, cross-primal pipeline graphs, metalForge Neural dispatch substrate
**Depends on**: V024 (debt resolution + barracuda absorption)

**airSpring**: 511 Rust lib tests + 934 validation + 1393 atlas checks, 41 barracuda + 1 forge binary, 885/885 Python, 0 clippy pedantic warnings, biomeOS Neural API bridge operational

---

## Executive Summary

airSpring now has a complete integration surface for biomeOS:

1. **metalForge Neural dispatch** — `SubstrateKind::Neural` added alongside GPU/NPU/CPU.
   The dispatch priority chain is GPU > NPU > Neural > CPU. Neural substrates are
   discovered by probing biomeOS's Neural API socket at runtime.

2. **Ecology capability domain** — 20+ capability translations proposed for biomeOS's
   `capability_registry.toml`, mapping `ecology.*` semantic names to airSpring methods
   (`eco.daily_et0`, `eco.water_balance_season`, `eco.yield_response`, etc.).

3. **Deployment graphs** — Two biomeOS-format TOML graphs:
   - `airspring_eco_pipeline.toml`: Weather → ET₀ → Water balance → Yield → Store
   - `cross_primal_soil_microbiome.toml`: airSpring soil moisture → wetSpring diversity

4. **Round-trip parity validated** — Exp 036 (29/29 PASS + 14/14 Python) confirms that
   JSON-RPC serialization through `capability.call` introduces zero numerical drift.

---

## Part 1: What biomeOS Should Absorb

### 1.1 Ecology Capability Domain

airSpring proposes a new domain in `capability_registry.toml`:

```toml
[domains.ecology]
provider = "airspring"
capabilities = ["ecology", "irrigation", "soil_moisture", "evapotranspiration", "crop_science"]

[translations.ecology]
"ecology.et0_pm" = { provider = "airspring", method = "eco.daily_et0" }
"ecology.et0_pt" = { provider = "airspring", method = "eco.priestley_taylor_et0" }
"ecology.et0_hargreaves" = { provider = "airspring", method = "eco.hargreaves_et0" }
"ecology.et0_thornthwaite" = { provider = "airspring", method = "eco.thornthwaite_monthly_et0" }
"ecology.et0_makkink" = { provider = "airspring", method = "eco.makkink_et0" }
"ecology.et0_turc" = { provider = "airspring", method = "eco.turc_et0" }
"ecology.et0_hamon" = { provider = "airspring", method = "eco.hamon_pet" }
"ecology.water_balance" = { provider = "airspring", method = "eco.water_balance_season" }
"ecology.richards_1d" = { provider = "airspring", method = "eco.richards_solve" }
"ecology.yield_response" = { provider = "airspring", method = "eco.yield_ratio_multistage" }
"ecology.gdd" = { provider = "airspring", method = "eco.gdd_accumulate" }
"ecology.fetch_weather" = { provider = "nestgate", method = "storage.retrieve", fallback_provider = "airspring" }
```

**Action**: Merge into `config/capability_registry.toml`. This gives biomeOS full
ecology/agriculture capability routing alongside the existing science (wetSpring)
and compute (ToadStool) domains.

### 1.2 Cross-Primal Pipeline Graph

`cross_primal_soil_microbiome.toml` demonstrates the first cross-Spring pipeline:

```
NestGate (weather) → airSpring (soil moisture) → wetSpring (diversity) → ToadStool (spectral GPU)
```

**Action**: Add to `graphs/` in biomeOS. This validates that the graph engine can
orchestrate sequential capability.call across different provider primals.

### 1.3 Neural API Client Pattern

airSpring's `metalForge/forge/src/neural.rs` implements a minimal synchronous
JSON-RPC client over Unix sockets with zero async dependencies:

- 4-tier socket discovery (env → XDG → /run → /tmp)
- `capability.call(domain, operation, args)` → parsed response
- `capability.discover(domain)` → provider listing
- UID detection via `/proc/self/status` (zero unsafe, no libc)
- Graceful degradation when biomeOS is absent

**Learning for biomeOS**: The `neural-api-client` crate in biomeOS uses tokio.
For Springs that don't need async, a synchronous alternative (like this pattern)
reduces the dependency footprint significantly. Consider shipping a
`neural-api-client-sync` crate.

---

## Part 2: What ToadStool / BarraCuda Should Absorb

### 2.1 ET₀ Method Portfolio (7 methods)

airSpring now validates 7 independent ET₀ methods with full Python↔Rust parity:

| Method | Data Needs | Rust Function | GPU Tier |
|--------|-----------|---------------|----------|
| Penman-Monteith (FAO-56) | Full weather station | `daily_et0()` | Tier A (`BatchedEt0`) |
| Priestley-Taylor (1972) | Radiation + temperature | `priestley_taylor_et0()` | Tier B |
| Hargreaves-Samani (1985) | Temperature only | `hargreaves_et0()` | Tier B |
| Thornthwaite (1948) | Monthly temperature | `thornthwaite_monthly_et0()` | Tier B |
| Makkink (1957) | Radiation + temperature | `makkink_et0()` | Tier B |
| Turc (1961) | Radiation + temperature + humidity | `turc_et0()` | Tier B |
| Hamon (1961) | Temperature + day length | `hamon_pet()` | Tier B |

All 7 are CPU-validated and ready for GPU promotion via `BatchedElementwise`.

### 2.2 Substrate Discovery Pattern

metalForge's `probe_neural()` function demonstrates how a substrate prober can
check for biomeOS at runtime and add a `Neural` substrate with inferred capabilities.
This pattern could be generalized: ToadStool could probe for biomeOS and expose
remote compute substrates discovered through the Neural API.

---

## Part 3: Cross-Primal Interactions

### 3.1 airSpring ↔ wetSpring

**Soil moisture → microbial diversity**: airSpring computes θ(t) (soil moisture
over time) via FAO-56 water balance. wetSpring's diversity analysis can use this
to model microbial community shifts under drought stress.

```
capability.call("ecology.water_balance", {...}) → θ_series
capability.call("science.diversity", { moisture_series: θ_series }) → Shannon, Simpson
capability.call("science.anderson", { diversity: results }) → spectral analysis
```

### 3.2 airSpring ↔ groundSpring

**Error propagation**: groundSpring's noise/measurement expertise can provide
Monte Carlo uncertainty bands for ET₀ predictions.

```
capability.call("ecology.et0_sensitivity", { inputs, perturbation: 0.05 })
capability.call("measurement.error_propagation", { et0_outputs, uncertainties })
```

### 3.3 All Springs → biomeOS Graphs

The graph format (`[[nodes]]` with `capability_call` operations) is validated
and ready for execution. airSpring's two graphs establish the pattern for
other Springs to define their own orchestrated pipelines.

---

## Quality

```
barracuda: cargo clippy --lib --bins --tests — 0 warnings (pedantic)
metalForge: cargo clippy --lib --bins --tests — 0 warnings (pedantic + nursery)
barracuda: cargo test — 511 lib + integration tests pass
metalForge: cargo test — 32 tests pass (31 unit + 1 doc)
Python controls: 885/885 pass
Rust validation binaries: 934/934 pass
Neural API parity: Exp 036 — 29/29 PASS (zero drift)
unsafe code: 0 (metalForge uses /proc/self/status instead of libc::getuid)
```

---

## Files Changed

| File | Change |
|------|--------|
| `metalForge/forge/src/neural.rs` | **New** — biomeOS Neural API bridge |
| `metalForge/forge/src/substrate.rs` | Added `SubstrateKind::Neural`, `Capability::NeuralApiRoute` |
| `metalForge/forge/src/dispatch.rs` | Neural tier in routing priority |
| `metalForge/forge/src/inventory.rs` | Neural substrate probing |
| `metalForge/forge/src/lib.rs` | Added `neural` module |
| `metalForge/forge/Cargo.toml` | Added `serde_json` dependency |
| `graphs/airspring_eco_pipeline.toml` | **New** — biomeOS deployment graph |
| `graphs/cross_primal_soil_microbiome.toml` | **New** — cross-Spring pipeline |
| `specs/BIOMEOS_CAPABILITIES.md` | **New** — ecology domain specification |
| `control/neural_api/benchmark_neural_api.json` | **New** — Exp 036 benchmark |
| `control/neural_api/neural_api_parity.py` | **New** — Python control (14/14) |
| `barracuda/src/bin/validate_neural_api.rs` | **New** — Rust validation (29/29) |
| `barracuda/Cargo.toml` | Added `validate_neural_api` binary |
| `specs/NUCLEUS_INTEGRATION.md` | Updated for Neural API bridge + graphs |
| `run_all_baselines.sh` | Added Exp 036 entries |
| `specs/PAPER_REVIEW_QUEUE.md` | Added Exp 036 |
| `experiments/README.md` | Added Exp 036 |
| `CONTROL_EXPERIMENT_STATUS.md` | Updated counts |
| `metalForge/README.md` | Updated for active dispatch layer |

---

## Next Steps

1. **biomeOS**: Merge ecology domain into `capability_registry.toml`
2. **biomeOS**: Deploy `airspring_eco_pipeline.toml` on Eastgate tower node
3. **biomeOS**: Test cross-primal graph with wetSpring
4. **ToadStool**: GPU promotion for Makkink/Turc/Hamon (Tier B → A)
5. **airSpring**: Full Neural API live test with running biomeOS NUCLEUS
6. **metalForge**: Evolve Neural dispatch to async (when Plasmodium routing is needed)
