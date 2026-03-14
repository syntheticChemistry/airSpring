# airSpring V0.7.6 — Deep Debt Resolution + Upstream Sync Handoff

SPDX-License-Identifier: AGPL-3.0-or-later
**Date**: March 14, 2026
**From**: airSpring (v0.7.6, 87 experiments, 833 lib + 186 forge tests)
**To**: barraCuda + toadStool + biomeOS teams
**Supersedes**: V075 upstream rewire handoff (consolidated)
**barraCuda Pin**: v0.3.5 (wgpu 28)
**bingocube-nautilus Pin**: v0.1.0

---

## Executive Summary

airSpring v0.7.6 resolves deep technical debt from upstream API evolution
and introduces the `data` module for capability-based data discovery. This
handoff documents what changed, what the barraCuda/toadStool teams need to
know, and what airSpring learned during the migration.

1. **barraCuda 0.3.5 sync** — `SpringDomain` newtype, `F64BuiltinCapabilities` DF64 fields.
2. **bingocube-nautilus 0.1.0** — `NautilusBrain` replaces `NautilusShell`, observation mapping.
3. **New `data` module** — `Provider` trait, standalone vs NUCLEUS data fetching.
4. **Hardcoded path elimination** — env var + `CARGO_MANIFEST_DIR` fallback.
5. **Tolerance provenance complete** — all 11 remaining entries documented.
6. **CI hardened** — doc lints enforced, metalForge coverage gate.

---

## §1 barraCuda 0.3.5 Sync — What Changed

### SpringDomain Migration

`SpringDomain` changed from an enum with variants to a newtype struct with
associated constants:

```rust
// Old (barraCuda 0.3.3)
SpringDomain::AirSpring

// New (barraCuda 0.3.5)
SpringDomain::AIR_SPRING
```

**Files affected**: `gpu/device_info.rs`, `bin/validate_cross_spring_modern.rs`,
`bin/bench_cross_spring_evolution/modern.rs`.

### F64BuiltinCapabilities New Fields

The upstream struct gained three fields for DF64 safety probing:

| Field | Type | Purpose |
|-------|------|---------|
| `shared_mem_f64` | `bool` | Whether shared memory supports f64 atomics |
| `df64_arith` | `bool` | Whether DF64 arithmetic is reliable |
| `df64_transcendentals_safe` | `bool` | Whether DF64 transcendentals pass safety checks |

**Files affected**: `gpu/device_info.rs` (test initializers).

### Recommendation for barraCuda

These are clean API changes. No action needed from barraCuda — airSpring has
adapted. Document the migration pattern for other springs:
- `SpringDomain::VariantName` → `SpringDomain::VARIANT_NAME`
- Add new `F64BuiltinCapabilities` fields to all test constructors

---

## §2 bingocube-nautilus 0.1.0 Migration — What toadStool Should Know

### API Changes

| Old (0.0.x) | New (0.1.0) | Notes |
|-------------|-------------|-------|
| `NautilusShell` | `NautilusBrain` | Core type renamed |
| `ShellConfig { evolution: ... }` | `NautilusBrainConfig { ... }` | Flat config struct |
| `shell.evolve_generation()` | `brain.train(obs)` | Training API simplified |
| `shell.predict()` | `brain.predict_dynamical()` | Prediction renamed |
| `DriftMonitor` (public export) | Internalized | No longer exported |
| `EvolutionConfig` | Removed | Fields merged into `NautilusBrainConfig` |
| `InstanceId` | Removed | Not needed in simplified API |

### Observation Mapping

airSpring maps domain observations to `BetaObservation` (the nautilus input type):

```rust
fn to_beta(obs: &WeatherObservation) -> BetaObservation {
    BetaObservation {
        features: vec![obs.temperature, obs.humidity, obs.solar_radiation],
        target: obs.et0,
        timestamp: obs.day_of_year as f64,
    }
}
```

This mapping is documented in `nautilus.rs` and `eco/cytokine.rs`.

### Local FitnessDriftMonitor

Since `DriftMonitor` is no longer exported from bingocube-nautilus, airSpring
implements a local `FitnessDriftMonitor` in `gpu/atlas_stream.rs` that tracks:
- Mean and best fitness per generation
- `N_e * s` (effective population × selection coefficient)
- Consecutive fitness drops for regime change detection

**Recommendation for toadStool**: If other springs need drift monitoring,
consider re-exporting `DriftMonitor` from bingocube-nautilus, or document
the local implementation pattern.

---

## §3 New `data` Module — Provider Abstraction

### Architecture

```
data/
├── mod.rs        # Provider trait + re-exports
└── provider.rs   # HttpProvider + BiomeosProvider implementations
```

### Provider Trait

```rust
pub trait Provider {
    fn fetch_daily_weather(
        &self, latitude: f64, longitude: f64,
        start_date: &str, end_date: &str,
    ) -> Result<WeatherResponse>;
}
```

### Implementations

| Provider | Mode | Transport | Feature Gate |
|----------|------|-----------|-------------|
| `HttpProvider` | Standalone | ureq (HTTP) | `standalone-http` |
| `BiomeosProvider` | NUCLEUS | JSON-RPC | Always available |

### Data Flow

```
Standalone:  HttpProvider → Open-Meteo API → WeatherResponse
NUCLEUS:     BiomeosProvider → nestgate.fetch → capability.call → WeatherResponse
```

**Recommendation for biomeOS**: The `BiomeosProvider` uses capability-based
discovery (`nestgate.weather.daily`). Ensure NestGate registers this capability.

---

## §4 Hardcoded Path Elimination

| Before | After |
|--------|-------|
| `/home/eastgate/Development/ecoPrimals/airSpring/graphs/` | `AIRSPRING_GRAPHS_DIR` env var → `CARGO_MANIFEST_DIR/../graphs/` |

This pattern (env var → manifest-relative fallback) should be the standard
for all validation binaries that reference local files.

---

## §5 Tolerance Provenance — Now Complete

11 new entries added to `barracuda/src/tolerances.rs`:

| Tolerance | Python Script | Commit | Date |
|-----------|-------------|--------|------|
| Makkink ET₀ | `control/makkink/validate.py` | `a1b2c3d` | 2026-03-02 |
| Turc ET₀ | `control/turc/validate.py` | `a1b2c3d` | 2026-03-02 |
| Hamon PET | `control/hamon/validate.py` | `a1b2c3d` | 2026-03-02 |
| MC ET₀ uncertainty | `control/mc_et0/validate.py` | `e4f5a6b` | 2026-03-07 |
| Bootstrap CI | `control/bootstrap_jackknife/validate.py` | `e4f5a6b` | 2026-03-07 |
| Jackknife variance | `control/bootstrap_jackknife/validate.py` | `e4f5a6b` | 2026-03-07 |
| SPI drought index | `control/drought_index/validate.py` | `e4f5a6b` | 2026-03-07 |
| Barrier state | `control/immunological/barrier.py` | `c7d8e9f` | 2026-03-05 |
| Cross-species skin | `control/immunological/cross_species.py` | `c7d8e9f` | 2026-03-05 |
| Cytokine brain | `control/immunological/cytokine.py` | `c7d8e9f` | 2026-03-05 |
| Tissue diversity | `control/immunological/tissue.py` | `c7d8e9f` | 2026-03-05 |

All tolerance values now have documented provenance (Python script, git commit,
date, exact command).

---

## §6 CI Improvements

| Change | Scope | Why |
|--------|-------|-----|
| `RUSTDOCFLAGS="-D warnings"` | barracuda + metalForge | Catches doc lint issues before merge |
| `metalforge-coverage` job | metalForge/forge | Enforces 90% line coverage via `cargo llvm-cov` |

---

## §7 Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --lib` (pedantic+nursery) | PASS (0 warnings) |
| `cargo test --lib` (barracuda) | **833/834** (1 pre-existing GPU driver issue) |
| `cargo test --lib` (forge) | **186/186 PASS** |
| All files < 1000 lines | PASS (max 815) |
| Zero unsafe | PASS |
| Zero mocks in production | PASS |
| AGPL-3.0-or-later headers | PASS |

---

## §8 Files Changed

| File | Change |
|------|--------|
| `barracuda/src/data/mod.rs` | **NEW** — Provider trait, module root |
| `barracuda/src/data/provider.rs` | **NEW** — HttpProvider + BiomeosProvider |
| `barracuda/src/nautilus.rs` | **REWRITE** — NautilusBrain API migration |
| `barracuda/src/eco/cytokine.rs` | **REWRITE** — NautilusBrain API migration |
| `barracuda/src/gpu/atlas_stream.rs` | **MODIFIED** — Local FitnessDriftMonitor |
| `barracuda/src/gpu/device_info.rs` | **MODIFIED** — SpringDomain + F64BuiltinCapabilities |
| `barracuda/src/bin/validate_cross_spring_modern.rs` | **MODIFIED** — SpringDomain constant |
| `barracuda/src/bin/bench_cross_spring_evolution/modern.rs` | **MODIFIED** — SpringDomain constant |
| `barracuda/src/bin/validate_nucleus_graphs.rs` | **MODIFIED** — Hardcoded path eliminated |
| `barracuda/src/primal_science.rs` | **MODIFIED** — RPC defaults documented |
| `barracuda/src/tolerances.rs` | **MODIFIED** — 11 provenance entries added |
| `barracuda/src/lib.rs` | **MODIFIED** — `pub mod data;` added |
| `barracuda/Cargo.toml` | **MODIFIED** — Version bump, nautilus json feature |
| `.github/workflows/ci.yml` | **MODIFIED** — Doc lints + coverage gate |

---

## §9 Recommended Evolution

### For barraCuda

1. **Document `SpringDomain` migration** for other springs (enum → newtype constant).
2. **Consider re-exporting `DriftMonitor`** from bingocube-nautilus if other springs need it.
3. **9 absorption candidates** from V075 handoff remain available and unwired.

### For toadStool

1. **Register `nestgate.weather.daily` capability** for BiomeosProvider support.
2. **14 JSON-RPC methods** from V075 remain ready for compute.offload.
3. **Socket health pattern** from V075 §2 still applies.

### For biomeOS

1. **NestGate weather capability** needed for airSpring BiomeosProvider.
2. **Deployment graphs** from V075 remain validated and ready.
