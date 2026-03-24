# airSpring V0.10.0 → barraCuda / toadStool Deep Audit & Evolution Handoff

**Date:** 2026-03-23
**From:** airSpring V0.10.0 (Phase 5.15)
**To:** barraCuda / toadStool team
**Supersedes:** HANDOFF_AIRSPRING_TO_BARRACUDA_TRANSPORT_ABSORPTION_MAR23_2026.md (companion, still active)
**License:** AGPL-3.0-or-later

---

## Executive Summary

airSpring completed a comprehensive deep audit and evolution pass targeting modern
idiomatic Rust, ecosystem compliance, and documentation fidelity. This handoff
documents findings, patterns, and evolution opportunities relevant to barraCuda
and toadStool.

**Key outcomes:**
- Zero `#[allow()]` in entire codebase (production + test) — all `#[expect(reason)]`
- `deny.toml` ecoBin enforcement on both workspace crates (14 C-dep bans)
- CI path dependency resolution via symlink strategy for GitHub Actions
- All documentation counts synchronized (947 lib + 306 integration + 61 forge)
- 24 Tier A + 2 Tier B GPU modules, Write→Absorb→Lean complete

---

## 1. Lint Architecture: #[allow()] → #[expect(reason)]

### What We Did

Eliminated every `#[allow()]` attribute across the entire codebase. Every lint
suppression now uses `#[expect(lint, reason = "...")]` which:
- Fails at compile time if the suppressed lint stops firing (prevents stale allows)
- Documents *why* the suppression exists via mandatory reason strings
- Aligns with Rust 2024 idioms (stable since MSRV 1.92)

### Specific Patterns

| Pattern | Count | Example |
|---------|-------|---------|
| `#[expect(clippy::unwrap_used)]` | ~60 | Test modules with assertion-style unwraps |
| `#[expect(clippy::float_cmp)]` | ~30 | Physical constant comparisons in tests |
| `#[expect(clippy::cast_precision_loss)]` | ~20 | Validation binaries with bounded fixture sizes |
| `#[expect(clippy::expect_used)]` | ~15 | Fail-fast validation binaries |
| `#[expect(clippy::too_many_arguments)]` | ~8 | PDE-style APIs (Richards, seasonal pipeline) |
| `#[expect(clippy::suboptimal_flops)]` | 1 | Synthetic weather generator (readable `a + b * sin(x)`) |

### Finding: Blanket Allows Hide Stale Suppressions

When converting `#![allow(clippy::unwrap_used, clippy::expect_used)]` to
`#![expect(...)]`, we discovered that several modules suppressed lints that
weren't actually triggered:
- `validate_nucleus_graphs.rs`: suppressed both `unwrap_used` and `expect_used`
  but used neither — attribute removed entirely
- `eco/dual_kc/tests.rs`, `gpu/stats.rs`: suppressed `unwrap_used` and
  `expect_used` alongside `float_cmp`, but only `float_cmp` fired — narrowed

**Recommendation for barraCuda:** Audit all `#[allow()]` attributes. Many are
stale from early development. `#[expect()]` will catch them automatically.

---

## 2. ecoBin Enforcement: deny.toml

### What We Did

Created `barracuda/deny.toml` (mirroring `metalForge/forge/deny.toml`) that
explicitly bans C-system dependencies:

**Banned crates (14):**
`openssl-sys`, `ring`, `native-tls`, `cmake`, `cc`, `bindgen`, `pkg-config`,
`vcpkg`, `libz-sys`, `zstd-sys`, `bzip2-sys`, `curl-sys`, `libssh2-sys`, `sysinfo`

**Wired into CI:** `cargo deny --config airSpring/barracuda/deny.toml check`

### Finding: blake3 Pulls cc as Build-Dep

`blake3` uses `cc` as a build dependency for optional SIMD acceleration. This is
infrastructure (not application code) and acceptable under ecoBin rules, but
`blake3` supports `features = ["pure"]` to disable C codegen entirely.

**Recommendation for barraCuda:** If ecosystem-wide ecoBin purity is a goal,
consider whether `blake3 = { features = ["pure"] }` should be the default, with
SIMD opt-in only where benchmarked.

---

## 3. CI Path Dependency Resolution

### Problem

`Cargo.toml` uses relative path dependencies (`../../barraCuda/crates/barracuda`)
that work locally but break in CI where repos checkout to different paths.

### Solution: Symlink Strategy

```yaml
- name: Checkout ecoPrimals (sparse)
  uses: actions/checkout@v4
  with:
    repository: syntheticChemistry/ecoPrimals
    path: ecoPrimals
    sparse-checkout: |
      barraCuda/crates/barracuda
      primalTools/bingoCube/nautilus

- name: Link path dependencies
  run: |
    ln -s "$GITHUB_WORKSPACE/ecoPrimals/barraCuda" "$GITHUB_WORKSPACE/barraCuda"
    ln -s "$GITHUB_WORKSPACE/ecoPrimals/primalTools" "$GITHUB_WORKSPACE/primalTools"
```

This preserves Cargo.toml's relative paths without modification. Applied to all
10 CI jobs.

**Recommendation for barraCuda CI:** If other springs need similar patterns,
the symlink approach is reusable. Document the expected workspace layout in
each spring's CI.

---

## 4. MSRV Documentation

airSpring's MSRV is 1.92 (ecosystem floor is 1.87). The divergence is
intentional — pervasive use of `#[expect()]` with `reason` requires stable
`lint_reasons` feature from Rust 1.92+.

**Recommendation:** If barraCuda adopts `#[expect()]` ecosystem-wide, consider
bumping the ecosystem MSRV floor to 1.92.

---

## 5. GPU Evolution Status

### Tier Summary

| Tier | Count | Status |
|------|-------|--------|
| **A** | 24 | GPU-first via `BatchedElementwiseF64` or dedicated shader |
| **B** | 2 | `seasonal_pipeline` (chained ops), `atlas_stream` (GPU streaming) |
| **C** | 2 | `anderson` (needs new shader), `et0_ensemble` (CPU logic) |

### Write → Absorb → Lean: Complete

All 20 `BatchedElementwiseF64` ops consumed by airSpring are upstream.
`local_dispatch` retired in v0.7.2. Zero local WGSL shaders remaining.

### Tier B Evolution Opportunities

1. **`seasonal_pipeline`**: Chains ops 0→7→1→yield. Currently dispatches each
   stage as separate GPU submission. A `TensorSession` fused pipeline (barraCuda
   0.4.x) would eliminate intermediate buffer copies for N≥1024 fields.

2. **`atlas_stream`**: 100-station regional pipeline. `UnidirectionalPipeline`
   (fire-and-forget GPU streaming) would avoid CPU round-trips per station.

### Available but Unwired barraCuda Primitives

| Primitive | Use Case in airSpring | Blocker |
|-----------|----------------------|---------|
| `BatchedOdeRK45F64` | Adaptive Dormand-Prince for dynamic soil | Needs Richards equation adapter |
| `TensorContext` | Pooled buffers for `SeasonalReducer` | Awaiting barraCuda 0.4.x |
| `NelderMeadGpu` | GPU Nelder-Mead for VG inverse | CPU path sufficient (small N) |
| `BFGS` | Gradient-based VG fitting | No gradient available |
| `unified_hardware` | Multi-device dispatch | Awaiting toadStool standardization |

---

## 6. Tolerance Architecture (Absorption Candidate)

58 named `Tolerance` structs across 4 domain submodules:

| Domain | Count | Examples |
|--------|-------|---------|
| Atmospheric | 15 | `et0_reference` (0.01 abs, 1e-3 rel), `mc_et0_propagation` (0.5 abs) |
| Soil | 19 | `water_balance_mass` (0.01 abs), `richards_steady_state` (0.001 abs) |
| GPU | 11 | `gpu_cpu_parity` (1e-5 abs), `gpu_regression` (1e-4 abs) |
| Instrument | 13 | `sensor_topp_vwc` (0.02 abs), `nass_yield_prediction` (5% rel) |

Each tolerance has `abs_tol`, `rel_tol`, and `justification` fields. Python
mirrors in `control/tolerances.py`.

**Recommendation:** If multiple springs use similar tolerance structures, a
`barracuda::validation::Tolerance` type could standardize the pattern.

---

## 7. Patterns for Upstream Absorption

### 7a. OrExit Trait

All 91 validation binaries use `OrExit` to convert `Result`/`Option` to
fail-fast `process::exit(1)` with structured `tracing::error!()`. No panic
backtraces in CI output.

```rust
trait OrExit<T> {
    fn or_exit(self, msg: &str) -> T;
}
```

Multiple springs have implemented this independently. A canonical
`barracuda::validation::OrExit` would eliminate duplication.

### 7b. ValidationSink Trait

`JsonSink` emits CI-parsable JSON from `ValidationHarness` checks:

```json
{"suite": "...", "passed": 42, "total": 42, "all_passed": true,
 "checks": [{"label": "...", "passed": true, "observed": 5.23, ...}]}
```

If barraCuda's `ValidationHarness` absorbed this, springs would get structured
CI output for free.

### 7c. ProvenanceConfig DI Pattern

Dependency injection for all IPC configuration:

```rust
pub struct ProvenanceConfig {
    pub transport_override: Option<Transport>,
    pub neural_api_socket: Option<PathBuf>,
    pub neural_api_address: Option<SocketAddr>,
    pub biomeos_socket_dir: Option<PathBuf>,
}
```

Production: `from_env()`. Tests: construct directly. Zero `set_var`, zero
`unsafe`, zero `#[serial]`.

### 7d. device_or_skip! Macro

```rust
macro_rules! device_or_skip {
    () => {
        match try_f64_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no f64 GPU"); return; }
        }
    };
}
```

Replicated across springs. Canonical version in `barracuda::device::test_pool`.

---

## 8. Dependency Health

| Dependency | Source | Status |
|-----------|--------|--------|
| `barracuda` | Path dep (ecoPrimals) | v0.3.7, `default-features = false` (gpu + domain-pde) |
| `bingocube-nautilus` | Path dep (primalTools) | Evolutionary reservoir computing |
| `wgpu 28` | Direct | Pure Rust surface, Vulkan backend |
| `bytemuck 1` | Direct | Zero-dep GPU buffer layout |
| `serde/serde_json` | Direct | JSON-RPC, validation I/O |
| `thiserror` | Direct | Typed error enums |
| `tracing` | Direct | Structured logging |
| `blake3` | Transitive (via barracuda) | Pulls `cc` build-dep (SIMD) |
| `libc` | Transitive (via tokio, rand) | Infrastructure |
| `akida-driver` | Optional (NPU feature) | Path dep to toadStool |

**No** `openssl`, `ring`, `cmake`, `bindgen`, or `pkg-config`. ecoBin compliant.

---

## 9. Cross-Spring Learnings

### From wetSpring V132
- Per-site `#[expect(clippy::cast_*)]` with reasons (adopted)
- `ValidationSink` trait for CI-parsable output (adopted)

### From neuralSpring V121
- Typed `PipelineError` enum with `thiserror` (adopted)

### From ludoSpring V29
- `default-features = false` on barraCuda dependency (adopted)
- `content_sha256` provenance in Python baseline generators (adopted)

### From groundSpring V121
- `deny.toml` convergence pattern (adopted)
- `#[allow]` → `#[expect(reason)]` evolution (completed)

---

## 10. Verified State

| Metric | Value |
|--------|-------|
| Lib tests | 947 |
| Integration + doc tests | 306 |
| Forge tests | 61 |
| Total | 1,314 |
| Validation binaries | 91 |
| Experiments | 87 |
| GPU tiers | 24 A + 2 B + 2 C |
| Tolerances | 58 named |
| Capabilities | 41 |
| MCP tools | 10 |
| Line coverage | ~95% (llvm-cov) |
| Clippy | Zero warnings (pedantic + nursery) |
| Unsafe | Zero (`#![forbid(unsafe_code)]`) |
| `#[allow()]` | Zero (entire codebase) |
| C dependencies | Zero in application code |
| Hardcoded primals | Zero in production code |
| Files > 1000 LOC | Zero |
| Ignored doctests | Zero |
| Platform | Unix + TCP (Transport enum) |
| MSRV | 1.92 (Rust 2024, `#[expect]` stable) |

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
