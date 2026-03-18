# airSpring V0.9.0 → barraCuda / toadStool Evolution Handoff

**Date**: March 18, 2026
**From**: airSpring V0.9.0
**To**: barraCuda team, toadStool team
**License**: AGPL-3.0-or-later
**Covers**: V0.8.9 → V0.9.0 audit execution findings relevant to upstream evolution

---

## Executive Summary

- airSpring's `cast` module is a strong **upstream absorption candidate** for barraCuda — all springs need safe numeric cast helpers
- `DispatchOutcome<T>` and `ValidationSink` patterns are ready for upstream promotion (used across 4+ springs)
- Library-level `#![deny(cast_*)]` with module-level `#[expect()]` is the correct lint strategy — recommend adoption in barraCuda and all springs
- `soil_moisture` module refactoring pattern (672 LOC → 4 submodules, same public API) is a model for upstream large-file evolution
- Zero local WGSL shaders remain — Write→Absorb→Lean **complete** since V0.7.2
- 894 lib + 299 integration tests, zero clippy pedantic+nursery warnings

---

## Part 1: Upstream Absorption Candidates

### 1.1 `cast` Module → `barracuda::cast`

**Current state**: airSpring's `barracuda/src/cast.rs` provides 9 type-safe numeric cast helpers:

| Helper | From → To | Safety |
|--------|-----------|--------|
| `usize_f64(v)` | usize → f64 | Exact for N < 2^53 |
| `f64_usize(v)` | f64 → usize | Debug-panics on invalid |
| `usize_u32(v)` | usize → u32 | Debug-panics on overflow |
| `i32_f64(v)` | i32 → f64 | Always exact |
| `u32_f64(v)` | u32 → f64 | Always exact |
| `f64_u32(v)` | f64 → u32 | Debug-panics on invalid |
| `u32_usize(v)` | u32 → usize | Always exact |
| `u64_usize(v)` | u64 → usize | Debug-panics on 32-bit overflow |
| `u64_f64(v)` | u64 → f64 | Exact for N < 2^53 |

**Why upstream**: Every spring has the same problem — clippy pedantic flags `as` casts, and each spring reinvents the same helpers. neuralSpring has `safe_cast` (S162), healthSpring has `cast_helpers`. Centralizing in barraCuda eliminates duplication.

**barraCuda action**: Absorb `cast` module into `barracuda::cast`. Expose all 9 helpers. Springs lean on upstream.

### 1.2 `DispatchOutcome<T>` → `barracuda::dispatch`

airSpring's `DispatchOutcome<T>` enum handles JSON-RPC dispatch results:

```rust
pub enum DispatchOutcome<T> {
    Ok(T),
    MethodNotFound(String),
    InvalidParams { method: String, reason: String },
}
```

Used in: airSpring, wetSpring (V126+), healthSpring (V34+). Standard pattern for all primal dispatch.

**barraCuda action**: Consider absorbing into `barracuda::ipc` or a new `barracuda::dispatch` module.

### 1.3 `ValidationSink` Pattern

airSpring's validation binaries use a `ValidationSink` pattern for structured pass/fail reporting with tolerance tracking. Originally from neuralSpring's `ValidationHarness`.

**barraCuda action**: The `ValidationHarness` is already upstream. Verify airSpring's extensions (tolerance registry, benchmark JSON provenance) are reflected.

---

## Part 2: Lint Strategy Evolution

### Library-Level Cast Lint Denial

airSpring now uses a split lint strategy:

```rust
// lib.rs
#![deny(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

// cast module gets #[expect()] since it IS the centralization point
#[expect(clippy::cast_precision_loss, ...)]
pub mod cast;
```

Cargo.toml `[lints.clippy]` keeps `allow` for cast lints to cover 91 validation binaries. The `#![deny()]` in `lib.rs` overrides for library code only.

**Key learning**: Cargo.toml `[lints]` should be the single source of lint configuration. CLI flags like `-W clippy::pedantic` conflict with specific `allow` overrides. airSpring's CI now uses `cargo clippy --all-targets -- -D warnings` (no `-W` flags).

**barraCuda action**: Consider adopting this split strategy — strict cast checking in library code, pragmatic allows for test/example code.

---

## Part 3: Module Refactoring Pattern

### soil_moisture: 672 LOC → 4 Submodules

```
eco/soil_moisture.rs (672 LOC, monolithic)
    ↓
eco/soil_moisture/
├── mod.rs             # Re-exports only (21 LOC)
├── topp.rs            # Topp equation (127 LOC)
├── texture.rs         # USDA texture classification (179 LOC)
├── saxton_rawls.rs    # Pedotransfer functions (216 LOC)
└── water.rs           # PAW, SWD, irrigation trigger (90 LOC)
```

**Key principle**: Public API unchanged — all `use eco::soil_moisture::*` imports continue to work. Tests move with their code.

**barraCuda action**: Apply this pattern to any upstream modules exceeding 600 LOC. The split should follow domain responsibility, not just line count.

---

## Part 4: Discovery Evolution

### Three-Tier Discovery Pattern

All primal discovery functions now follow a three-tier resolution:

```rust
pub fn discover_visualization_primal() -> Option<PathBuf> {
    // Tier 1: Environment override
    if let Ok(path) = std::env::var("PETALTONGUE_SOCKET") { ... }
    // Tier 2: Named socket scan
    if let Some(path) = discover_primal_socket("petaltongue") { ... }
    // Tier 3: Capability probe
    discover_primal_by_capability("visualization")
}
```

airSpring now has discovery functions for: coralReef (shader), Squirrel (inference), petalTongue (visualization). The binary's compute/data primal discovery also uses three-tier.

**toadStool action**: Ensure toadStool's own discovery functions follow this pattern. The capability probe tier (scanning all sockets for `capability.list`) is the key for zero-configuration deployment.

---

## Part 5: Benchmark JSON Provenance

### Hardcoded Values → JSON with Provenance

airSpring migrated hardcoded validation constants from source code to benchmark JSON files:

```json
{
  "expected_kr": [1.0, 1.0, 0.6975, ...],
  "_kr_provenance": {
    "baseline_script": "control/dual_kc/cover_crop_dual_kc.py",
    "baseline_commit": "3afc229",
    "date": "2026-02-25",
    "tolerance": "RICHARDS_STEADY (1e-6)"
  }
}
```

**barraCuda action**: If barraCuda has any hardcoded test expected values, consider migrating them to JSON files with provenance metadata. This makes baselines reproducible and auditable.

---

## Part 6: Current barraCuda Touchpoints

airSpring v0.9.0 touches **73 barraCuda APIs** across 20+ source files:

| Category | APIs Used | Count |
|----------|-----------|-------|
| `ops::batched_elementwise_f64` | Ops 0-19 | 20 |
| `stats::*` | mean, percentile, bootstrap_ci, crop_coefficient, diversity | 12 |
| `linalg::*` | tridiagonal_solve, ridge_regression | 3 |
| `pde::richards` | GPU Richards solver | 1 |
| `optimize::*` | nelder_mead, multi_start, brent | 3 |
| `device::WgpuDevice` | GPU device management | 8 |
| `error::BarracudaError` | Error handling | 5 |
| `special::gamma` | ln_gamma for SPI | 1 |
| Other | shaders::provenance, nn::nautilus | 20 |

**Version**: barraCuda 0.3.5 (path dependency, wgpu 28)

---

## Part 7: Remaining Evolution Gaps

| Gap | Owner | Priority |
|-----|-------|----------|
| `TensorContext` for fused pipelines | barraCuda | Medium — seasonal pipeline still chains individual ops |
| `ComputeDispatch::df64()` | toadStool | Low — f64 dispatch works via existing pathway |
| `BatchedOdeRK45F64` | barraCuda | Medium — Richards could use adaptive stepping |
| Subgroup/wavefront detection | toadStool | Low — affects reduce performance only |
| Anderson coupling shader | barraCuda | Low — CPU path is fast enough |
| `UnidirectionalPipeline` for AtlasStream | barraCuda | Medium — atlas streaming is CPU-chained |

---

## Part 8: What We Learned

1. **CI lint config**: Never pass `-W clippy::pedantic` on the CLI if you also have `[lints.clippy]` in Cargo.toml — they conflict. Let Cargo.toml be the single source.

2. **Cast evolution**: Start with a cast module, then add `#![deny()]` in lib.rs. This is reversible and incremental. Binary code can evolve later.

3. **Module refactoring**: When splitting a file into submodules, keep `mod.rs` as pure re-exports. Don't put logic in `mod.rs`. Tests stay with their code.

4. **Discovery**: Three-tier is the right pattern. Hardcoded socket paths in deployment manifests are wrong — remove them, rely on biomeOS resolution.

5. **Validation provenance**: Hardcoded expected values in source code are a maintenance hazard. Move them to JSON with provenance metadata. The `_provenance` field pattern is human-readable and machine-parseable.

---

*airSpring V0.9.0, March 18, 2026. 894 lib + 299 integration tests, zero clippy warnings, zero unsafe code. AGPL-3.0-or-later.*
