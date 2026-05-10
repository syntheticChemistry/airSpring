# airSpring V0.10.0 — Deep Audit Execution Handoff (March 24, 2026)

**From**: airSpring v0.10.0 → barraCuda, toadStool, coralReef, primalSpring, sibling Springs
**Date**: 2026-03-24
**Status**: 986 lib + 316 integration + 62 forge = 1,364 total tests, 91 binaries, 87 experiments, 90.56% line coverage, zero clippy (pedantic+nursery), zero unsafe, zero C deps, zero `#[allow()]`, cargo-deny 0.19 clean

---

## What Changed (This Session)

### Blocking Fixes

| Finding | Fix | Impact |
|---------|-----|--------|
| `cargo fmt` failed (3 files) | Applied rustfmt | CI gate green |
| clippy `must_use_candidate` on `all_tolerances()` | Added `#[must_use]` | Pedantic compliance |
| clippy `assertions_on_constants` in tolerance test | Evolved to `const { assert!(..) }` — compile-time contract | Rust 2024 idiom |
| cargo-deny 0.19 parse failures | Evolved `deny.toml`: removed `vulnerability` (now implicit), `unmaintained = "workspace"`, SPDX `AGPL-3.0-or-later`, `CC0-1.0` allowance, `blake3` cc wrapper, version-pinned path deps | Dependency audit green |
| Line coverage 89.86% < 90% target | +43 tests across 4 modules (json, provenance, dispatch, niche) | 90.56% — gate passes |

### Advisory Fixes

| Finding | Fix |
|---------|-----|
| `evolution_gaps.rs` header stale (v0.7.5) | Updated to v0.10.0 / barraCuda 0.3.7 HEAD `7a891dd` |
| Tier assignments stale | `richards_pde` → Tier A INTEGRATED, `isotherm_batch_fitting` → Tier A INTEGRATED |
| `tests/common/mod.rs` unfulfilled expects | `#[expect()]` → `#[allow()]` for shared test infra (lint fires per-binary) |
| README coverage claim 95.66% | Corrected to 90.56% with updated test counts |

### Coverage Expansion (89.86% → 90.56%)

| Module | Before | After | Tests Added |
|--------|--------|-------|-------------|
| `validation/json.rs` | 62.09% | 82.40% | 20 (parse, extract, checked/opt paths) |
| `ipc/provenance.rs` | 50.76% | 79.55% | 7 (TCP mock server, happy path begin/record/complete) |
| `ipc/compute_dispatch.rs` | 68.18% | 89.21% | 8 (error Display, From, RPC error, missing job_id) |
| `niche.rs` | 71.54% | 77.31% | 8 (capability inventory, domain validation, cost structure) |

---

## What We Learned (For Primal & Spring Teams)

### For barraCuda

1. **cargo-deny 0.19 breaking changes**: The `vulnerability` field in `[advisories]` was removed (all vulnerabilities now always error). `unmaintained` now accepts `"all"|"workspace"|"transitive"|"none"` — not `"warn"|"deny"`. SPDX requires full identifiers (`AGPL-3.0-or-later`, not `AGPL-3.0+`). All springs should audit their `deny.toml` files.

2. **`blake3` brings `cc` transitively**: barraCuda depends on `blake3` which uses `cc` for SIMD optimization. Springs that ban `cc` in `deny.toml` need `{ crate = "cc", wrappers = ["blake3"] }` to allow it through the wrapper chain. Consider whether blake3 can use `pure` feature to eliminate this.

3. **`CC0-1.0` license needed**: `hexf-parse` (via `naga` via `wgpu`) uses CC0-1.0. All springs using wgpu 28 need this in their license allow list.

4. **`const { assert!(..) }` for tolerance contracts**: When comparing const tolerance values in tests, Rust 2024 supports `const {}` blocks to make assertions compile-time. Catches drift at build time, not runtime. Pattern: `const { assert!(GPU_CPU_CROSS.abs_tol <= ET0_REFERENCE.abs_tol) }`.

5. **Version-pinned path deps eliminate wildcard bans**: Adding `version = "0.3.7"` alongside `path = "..."` satisfies cargo-deny's `wildcards = "deny"` while keeping local development ergonomics.

### For toadStool

1. **Cross-spring hardware validation live**: RTX 4070 + Titan V + AKD1000 + i9-12900K all validated through metalForge. 27 workloads route correctly via capability-based dispatch. NUCLEUS mesh routing with PCIe bypass confirmed.

2. **PrecisionRoutingAdvice wired**: All 20 `BatchedElementwiseF64` ops respect per-hardware precision routing (F64Native, F64NativeNoSharedMem, Df64Only, F32Only).

### For coralReef

1. **Zero local WGSL**: airSpring has completed the Write→Absorb→Lean cycle. All GPU shaders are upstream in barraCuda. No local `.wgsl` files. coralReef sovereign compilation targets only need to handle barraCuda's shader inventory.

### For sibling Springs

1. **`#[allow()]` vs `#[expect()]` for shared test helpers**: When `tests/common/mod.rs` is compiled into multiple test binaries, `#[expect(dead_code)]` triggers unfulfilled-lint-expectation errors in binaries that DO use the helper. Use `#[allow()]` with `reason` for shared test infra — `#[expect()]` is still correct for production code and single-binary tests.

2. **TCP mock server pattern for IPC testing**: The `ProvenanceConfig { transport_override: Some(Transport::Tcp(addr)) }` + `TcpListener` pattern exercises real JSON-RPC round-trips without live biomeOS. Reusable across springs for provenance trio, compute dispatch, and capability testing. Coverage went from 50% to 80% on provenance with 7 tests.

3. **Coverage target 90% is tight**: IPC modules that require live primals are the bottleneck. The DI pattern (config structs with transport override) is essential. Without it, `ipc/provenance.rs` would be stuck at ~50%.

---

## Absorption Candidates for Upstream

| Module | Destination | Description | Priority |
|--------|-------------|-------------|----------|
| `const assert` tolerance pattern | barraCuda tolerances | Compile-time tolerance contract verification | Medium |
| `deny.toml` template (0.19) | wateringHole guidance | Standardize cargo-deny 0.19 config across ecosystem | High |
| TCP mock server test pattern | wateringHole guidance | DI + TcpListener for IPC unit testing without live primals | Medium |
| `#[allow()]` vs `#[expect()]` guidance | wateringHole guidance | Document shared-test-infra exception to zero-allow rule | Low |

## Current State Summary

| Metric | Value |
|--------|-------|
| Lib tests | 986 |
| Integration tests | 316 |
| Forge tests | 62 |
| Total tests | 1,364 |
| Binaries | 91 |
| Experiments | 87 |
| Line coverage | 90.56% |
| Tier A GPU modules | 25 |
| Tolerances | 58 (5 submodules) |
| Capabilities | 45 |
| barraCuda version | 0.3.7 (wgpu 28) |
| cargo-deny | 0.19 clean |
| MSRV | 1.92 |
| Unsafe blocks | 0 |
| `#[allow()]` in production | 0 |
| C dependencies | 0 |

---

## Deferred Items

| Item | Blocker | ETA |
|------|---------|-----|
| `ValidationSink` trait | ludoSpring V30 → barraCuda absorption | When upstream absorbs |
| `TensorSession` fused pipelines | barraCuda 0.4.x | When available |
| `UnidirectionalPipeline` streaming | barraCuda streaming API | Medium-term |
| Anderson coupling GPU shader | New shader needed | Low priority |

---

## Verification

```bash
cd barracuda && cargo fmt --check                                              # clean
cd barracuda && cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery  # 0
cd barracuda && cargo test --lib                                               # 986 passed
cd barracuda && cargo deny check                                               # advisories ok, bans ok, licenses ok, sources ok
cd barracuda && cargo llvm-cov --lib --fail-under-lines 90                     # 90.56%
cd metalForge/forge && cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery  # 0
cd metalForge/forge && cargo deny check                                        # all ok
cd metalForge/forge && cargo test                                              # 62 passed
```

All green. Zero regressions.
