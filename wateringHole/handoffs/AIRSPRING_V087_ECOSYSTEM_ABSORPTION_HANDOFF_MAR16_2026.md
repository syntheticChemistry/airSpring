# airSpring V0.8.7 — Ecosystem Absorption Handoff

**Date**: March 16, 2026
**From**: airSpring niche
**To**: barraCuda, toadStool, ecosystem
**License**: AGPL-3.0-or-later

---

## Executive Summary

v0.8.7 absorbs cross-ecosystem patterns identified from reviewing all 6 springs
and 12 primals. Key changes: zero hardcoded primal names (new `BIOMEOS` constant,
swapped doc comments fixed), 4-format capability parsing (neuralSpring S156+),
Edition 2024 let-chains (collapsible-if), JSON-RPC proptest fuzz (7 properties),
and PRIMAL_REGISTRY updated from v0.7.6 to v0.8.7.

## Changes

### 1. Zero Hardcoded Primal Names

- Added `primal_names::BIOMEOS` constant
- Fixed swapped doc comments: TOADSTOOL was labelled "Security" and BEARDOG
  was labelled "Hardware" — corrected to match actual roles
- `discovery.rs` now uses `biomeos::discover_primal_socket(primal_names::BIOMEOS)`
  instead of iterating over `["biomeOS", "biomeos"]`

### 2. `parse_capabilities` — 4-Format Support

Extended `biomeos::parse_capabilities()` for all ecosystem capability response formats:

- **Format A**: Flat string array `["health", "compute.dispatch"]`
- **Format B**: Object array `[{"name": "health", "version": "1.0"}]`
- **Format C**: Nested wrapper `{"capabilities": ["health", ...]}`
  (neuralSpring S156+)
- **Format D**: Double-nested `{"capabilities": {"capabilities": [...]}}`
  (toadStool S155+)
- Also handles `{"result": [...]}` wrapper

Added 6 unit tests covering all formats plus empty/null edge cases.

### 3. Edition 2024 Let-Chains

Evolved nested `if let` patterns to Edition 2024 let-chains:

```rust
// Before:
if let Some(cached) = self.try_nestgate_cache(&key) {
    if let Ok(response) = WeatherResponse::from_cross_spring_v1(&cached) {

// After:
if let Some(cached) = self.try_nestgate_cache(&key)
    && let Ok(response) = WeatherResponse::from_cross_spring_v1(&cached)
```

### 4. `#[allow]` → `#[expect]` Complete

Last remaining `#[allow(dead_code)]` in production code (`dispatch.rs:10`)
evolved to `#[expect(dead_code, reason = "...")]`. Zero `#[allow()]` in
production code confirmed.

### 5. JSON-RPC Proptest Fuzz (petalTongue V166 Pattern)

Added 7 property-based tests to `tests/property_tests.rs`:

| Property | What It Tests |
|----------|---------------|
| `extract_rpc_error_never_panics` | Arbitrary JSON never panics `extract_rpc_error` |
| `parse_capabilities_never_panics` | Arbitrary JSON never panics `parse_capabilities` |
| `rpc_request_roundtrip` | `rpc::request` produces valid JSON-RPC 2.0 |
| `extract_rpc_error_correct_on_valid_error` | Correct (code, message) extraction |
| `parse_capabilities_flat_roundtrip` | String array round-trips through parse |
| `arbitrary_bytes_never_panic_rpc_parse` | Arbitrary bytes don't panic JSON parser |

### 6. Doc Quality

- Fixed doc backtick warnings (`NestGate`, `job_id`)
- PRIMAL_REGISTRY updated: v0.7.6 → v0.8.7 with full capability refresh

## Metrics

| Metric | v0.8.6 | v0.8.7 |
|--------|--------|--------|
| Library tests | 866 | **872** |
| Property tests | 15 | **22** |
| Hardcoded primal names | 2 locations | **0** |
| `#[allow()]` in production | 1 | **0** |
| `parse_capabilities` formats | 2 | **4** (+result wrapper) |
| Clippy collapsible-if warnings | 2 | **0** |

## Ecosystem Alignment

| Pattern | Source | airSpring Status |
|---------|--------|------------------|
| `#[expect(reason)]` | wetSpring V122 | Complete |
| `temp-env` for tests | neuralSpring S158 | N/A (dependency injection) |
| Dual-format capability | neuralSpring S156 | Extended to 4-format |
| Proptest fuzz | petalTongue V166 | 7 properties |
| `OrExit` trait | wetSpring V123 | Deferred (current explicit pattern is clear) |
| `compute_dispatch` | healthSpring V30 | Implemented in v0.8.6 |
| Zero C deps | ecosystem-wide | Maintained |

---
*ScyBorg Provenance Trio: AGPL-3.0-or-later (code) + ORC (game mechanics) + CC-BY-SA 4.0 (creative content)*
