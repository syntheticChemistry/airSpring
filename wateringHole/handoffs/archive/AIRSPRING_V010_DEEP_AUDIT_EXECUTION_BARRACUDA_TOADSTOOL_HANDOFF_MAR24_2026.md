# airSpring V0.10.0 → barraCuda / toadStool Deep Audit Execution Handoff

**Date:** 2026-03-24  
**From:** airSpring V0.10.0 (Phase 5.16)  
**To:** barraCuda / toadStool team  
**Supersedes:** (companion to `AIRSPRING_V010_DEEP_AUDIT_BARRACUDA_TOADSTOOL_HANDOFF_MAR23_2026.md`)  
**License:** AGPL-3.0-or-later

---

## Executive Summary

airSpring completed deep audit execution on March 24, resolving every actionable
finding from the March 23 comprehensive audit. This handoff documents patterns,
evolution opportunities, and absorption candidates relevant to barraCuda and
toadStool.

**Key outcomes:**

- Three-tier capability-based discovery (env override → named socket → capability probe)
- `normalize_method()` derived from `PRIMAL_NAME` — zero hardcoded primal strings
- 47/47 validation binaries standardized with `//! Provenance:` headers in the March 24 execution wave (full reconciled inventory: 90/90 — see §6)
- 8/8 benchmark JSON provenance structured (3 upgraded, 5 added)
- Forge barraCuda features aligned: `default-features = false, features = ["gpu"]`
- All doc counts reconciled: 938 lib + 316 integration + 62 forge = **1,316** total
- GPU tier reconciliation: `EVOLUTION_READINESS.md` ↔ `GPU_PROMOTION_MAP` aligned at 24/2/2

---

## 1. Capability-Based Discovery Pattern (Absorption Candidate)

### What We Evolved

airSpring’s `airspring_primal` binary now uses a three-tier discovery pattern,
documented in `barracuda/src/bin/airspring_primal/discovery.rs`:

1. **Environment override** — `AIRSPRING_COMPUTE_PRIMAL` / `AIRSPRING_DATA_PRIMAL`
   resolve a peer primal by name via `biomeos::discover_primal_socket` when the
   socket exists.
2. **Named socket scan** — well-known ecosystem primals (`toadStool` for compute,
   `nestgate` for storage) provide **defaults**, not hard compile-time coupling.
3. **Capability probe** — `biomeos::discover_primal_by_capability(domain)` scans
   `*.sock` files, calls JSON-RPC `capability.list` on each, and returns the first
   path whose capabilities match the domain prefix (e.g. `compute.*`, `storage`).

The capability probe is a **public** API on `biomeos::discovery`. Implementation
walks the biomeOS socket directory, skips non-`.sock` entries, parses capability
strings from the response, and matches `starts_with(domain)`.

### What barraCuda / toadStool Should Consider

This pattern should become ecosystem standard: springs discover peers by **what
they can do**, not only by **what they are named**. toadStool’s dispatch and
capability story already align with `capability.list`; a shared discovery helper
would stop N springs from copying the same scan loop.

**Proposed upstream API:**

```rust
pub fn discover_by_capability(domain: &str) -> Option<PathBuf>;
pub fn discover_by_capability_in(domain: &str, socket_dir: &Path) -> Option<PathBuf>;
```

airSpring already exposes `discover_primal_by_capability(domain)` with the
default socket root; an `_in` variant would complete the matrix next to
`discover_primal_socket_in` for tests and custom layouts.

---

## 2. PRIMAL_NAME–Derived Method Normalization

### What We Evolved

`normalize_method()` in `barracuda/src/rpc/mod.rs` derives the strip prefix from
`crate::PRIMAL_NAME` (see `lib.rs`: `pub const PRIMAL_NAME: &str = "airspring";`)
instead of hardcoding `"airspring."`. If a primal is renamed, the RPC layer
follows automatically.

### Pattern for barraCuda

```rust
pub fn normalize_method(method: &str) -> &str {
    method
        .strip_prefix(PRIMAL_NAME)
        .and_then(|rest| rest.strip_prefix('.'))
        .unwrap_or(method)
}
```

Every spring should use this pattern: the primal name constant is the **single
source of truth** for JSON-RPC method normalization as well as identity.

---

## 3. Provenance Standardization Patterns

### Validation Binary Pattern

Every validation binary now has a module-level header. **With Python baseline:**

```rust
//! Provenance:
//!   script = `control/path/to/baseline.py`
//!   commit = abc1234
//!   date   = 2026-MM-DD
//!   run    = `python3 control/path/to/baseline.py`
```

**Integration tests without Python baselines:**

```rust
//! Provenance: integration validation (no Python baseline)
```

### Benchmark JSON Pattern

Every benchmark JSON uses structured `_provenance` as the **first key**:

```json
{
    "_provenance": {
        "method": "...",
        "baseline_script": "control/.../script.py",
        "baseline_command": "python3 control/.../script.py",
        "baseline_commit": "abc1234",
        "reproduction_note": "Re-run baseline_command at baseline_commit to regenerate expected values",
        "python_version": "3.13",
        "created": "2026-MM-DD",
        "references": [...]
    }
}
```

### Recommendation for barraCuda

If `ValidationHarness` could enforce or generate provenance headers automatically
(perhaps via a builder or a small test-time verifier), springs would not maintain
dozens of near-duplicate `//!` blocks by hand.

---

## 4. Feature Minimization Pattern

### What We Evolved

metalForge `forge` now uses:

```toml
barracuda = { path = "../../../barraCuda/crates/barracuda", default-features = false, features = ["gpu"] }
```

Previously the dependency could pull default features and inflate the graph.
This matches the **explicit minimal set** philosophy used for airSpring’s own
barracuda crate.

### Recommendation

barraCuda should document which features each class of spring typically needs.
The `gpu` + `domain-pde` combination is airSpring’s **library** minimal set for
full PDE paths; forge aligns on `gpu` alone for its dependency slice. Other
springs should pick explicit features the same way.

---

## 5. GPU Evolution Status (24 Tier A, 2 Tier B, 2 Tier C)

All GPU modules are indexed in `specs/GPU_PROMOTION_MAP.md`. `EVOLUTION_READINESS.md`
cross-references the map and notes **nine** Tier B → Tier A promotions (sensor
calibration, Hargreaves, Kc climate, dual Kc, VG θ/K, Thornthwaite, GDD,
pedotransfer, seasonal), so narrative tables and the map stay aligned.

`barracuda/src/gpu/evolution_gaps.rs` continues to record shader/module mapping,
release notes, and “available but unwired” upstream primitives for future work.

### Remaining Gaps for barraCuda

| Gap | Description | Effort |
|-----|-------------|--------|
| `UnidirectionalPipeline` | Fused GPU seasonal / regional pipeline (stages 1–4 in one dispatch or streaming) to cut intermediate buffer churn | Medium |
| `ValidationSink` | Testable harness output trait (upstream from ludoSpring V23) for CI-stable JSON | Low |
| Tridiagonal batch | Direct use for PDE solvers beyond Richards where batch tridiagonal structure applies | Low |
| Adaptive ODE (RK45) | Soil moisture dynamics, biochar kinetics; upstream `BatchedOdeRK45F64` awaits domain wiring | Low |

Tier B in the current model still includes **seasonal pipeline** (fused vs
chained GPU) and **atlas stream** (`UnidirectionalPipeline`). Tier C examples
include workloads that still need new shaders or CPU-first orchestration—see the
promotion map for the live list.

---

## 6. Test and Quality Summary

| Metric | Value |
|--------|-------|
| Library tests | 938 |
| Integration tests | 316 |
| Forge tests | 62 |
| **Total** | **1,316** |
| Validation binaries | 91 |
| Python baselines | 1,284/1,284 |
| Line coverage | ~95.66% |
| Clippy warnings | 0 (pedantic + nursery) |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| `#[allow()]` in production | 0 |
| Hardcoded primal strings | 0 (all via `primal_names::*`) |
| Provenance coverage | 90/90 validation binaries + 59/59 JSON benchmarks |

---

## Reproduction

```bash
cd airSpring/barracuda
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cd ../metalForge/forge
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

---

## License

AGPL-3.0-or-later
