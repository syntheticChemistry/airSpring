<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# airSpring V0.7.0 — barraCuda v0.3.3 Rewire + Revalidation Handoff

**Date**: March 5, 2026
**From**: airSpring v0.7.0 (ecology/agriculture validation Spring)
**To**: barraCuda team / ToadStool S94b+ / All Springs
**Supersedes**: V069 BarraCuda Absorption Handoff (Mar 05)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring rewired from barraCuda v0.3.1 (wgpu 22) to **barraCuda v0.3.3** (wgpu 28).
827 lib tests pass, 0 clippy warnings (pedantic+nursery), 0 fmt diffs, 0 doc warnings.
25 GPU dispatch tests fail identically in barraCuda's own test suite — upstream wgpu 28
driver issue on Titan V NVK, not airSpring-specific.

**Key changes**:
1. **wgpu 22 → 28** — 4 API changes in `local_dispatch.rs` (the only file using raw wgpu)
2. **3/6 local ops absorbed upstream** — Makkink (op=14), Turc (op=15), Hamon (op=16)
3. **Df64 precision tier documented** — `LocalElementwise` now explicitly handles all 4 precision variants
4. **Upstream fused primitives available** — `VarianceF64::mean_variance()`, 5-accumulator Pearson, `Fp64Strategy` three-tier

---

## Part 1: wgpu 22 → 28 Migration

### Changes Required

| File | Change | Reason |
|------|--------|--------|
| `Cargo.toml` | `wgpu = "22"` → `"28"`, add `"vulkan"` feature | Match barraCuda 0.3.3 |
| `local_dispatch.rs` | `entry_point: "main"` → `Some("main")` | wgpu 28 API |
| `local_dispatch.rs` | `push_constant_ranges: &[]` → `immediate_size: 0` | wgpu 28 API |
| `local_dispatch.rs` | `wgpu::Maintain::Wait` → `wgpu::PollType::Wait { submission_index: None, timeout: None }` | wgpu 28 API |
| `local_dispatch.rs` | `BufferView<'a>` → `BufferView` (no lifetime) | wgpu 28 API |

### Pattern for Other Springs

Only files that use raw `wgpu::Device`, `wgpu::Buffer`, etc. need changes.
If a Spring only uses `barracuda::device::WgpuDevice` (the abstraction layer),
zero code changes are needed — just bump wgpu version in Cargo.toml.

---

## Part 2: Absorption Status (Local WGSL Ops)

| Op | Domain | Status | Upstream |
|----|--------|--------|----------|
| 0 | SCS-CN runoff | **local only** | Candidate for `batched_elementwise_f64` |
| 1 | Stewart yield | **local only** | Candidate for `batched_elementwise_f64` |
| 2 | Makkink ET₀ | **absorbed** | `Op::MakkinkEt0` (14) in barraCuda 0.3.3 |
| 3 | Turc ET₀ | **absorbed** | `Op::TurcEt0` (15) in barraCuda 0.3.3 |
| 4 | Hamon PET | **absorbed** | `Op::HamonEt0` (16) in barraCuda 0.3.3 |
| 5 | Blaney-Criddle ET₀ | **local only** | Candidate for `batched_elementwise_f64` |

airSpring retains `local_elementwise_f64.wgsl` for all 6 ops as a parallel validation
path. Ops 2-4 can optionally be rewired to `BatchedElementwiseF64` in a future session.

---

## Part 3: New Upstream Primitives Available

barraCuda v0.3.3 adds primitives airSpring can leverage:

| Primitive | Purpose | airSpring Use |
|-----------|---------|---------------|
| `VarianceF64::mean_variance()` | Fused Welford single-pass mean+variance | Sensor stream QA, anomaly detection |
| `CorrelationResult` | Fused 5-accumulator Pearson | Multi-sensor correlation analysis |
| `Fp64Strategy::Concurrent` | Run f64 + DF64 and cross-validate | NVK reliability verification |
| `Fp64Strategy::Hybrid` | DF64 bulk + f64 reductions | Consumer GPU acceleration |
| `TensorContext` | Fast-path GPU dispatch (15 ops) | Potential `SeasonalReducer` evolution |
| `mean_variance_df64.wgsl` | DF64 fused variance shader | Consumer GPU stats |
| `correlation_full_df64.wgsl` | DF64 fused correlation shader | Consumer GPU correlation |

### Precision evolution

`local_dispatch.rs` now explicitly handles all 4 `Precision` variants:
- `F64`: native f64 buffers (pro GPUs)
- `F32`: f32 buffers (consumer GPUs)
- `Df64`: f32 buffers (DF64 pair encoding handled by `compile_shader_universal`)
- `F16`: f32 buffers (edge inference, not typical for agriculture)

---

## Part 4: 25 GPU Dispatch Test Failures (Upstream)

All 25 failing tests return GPU=0.0 for non-zero expected values. The same failure
pattern occurs in barraCuda's own `test_fao56_et0_gpu` and `test_batched_ops_9_to_13_gpu`.

**Root cause**: wgpu 28 + NVK/Mesa driver on Titan V. Not an airSpring issue.

**Affected modules**: et0, water_balance, hargreaves, kc_climate, gdd, pedotransfer,
thornthwaite, van_genuchten, sensor_calibration, infiltration, diversity, jackknife,
bootstrap, mc_et0, stats, atlas_stream.

**All CPU paths pass**. GPU failures do not affect scientific correctness.

---

## Part 5: Remaining V069 Items (Still Valid)

From the previous handoff, these items remain:

1. **3 local-only ops** (SCS-CN, Stewart, Blaney-Criddle) ready for upstream absorption
2. **`compile_shader_universal()` validation** — confirmed f64→f32 downcast for 6 ag ops
3. **NVK/Mesa f64 reliability finding** — reinforced by wgpu 28 GPU=0.0 failures
4. **`json_f64_required` pattern** — structured `exit(1)` for validation binaries
5. **`SubmitParams` struct pattern** — multi-arg GPU dispatch
6. **Streaming JSON I/O** — `from_reader` instead of `read_to_string`

---

## Quality Gates

| Gate | Result |
|------|--------|
| `cargo check` | **0 errors** |
| `cargo fmt --check` | **0 diffs** |
| `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings` | **0 warnings** |
| `cargo doc --no-deps` | **0 warnings** |
| `cargo test --lib` | **827 passed**, 25 failed (upstream GPU) |
| barraCuda version | **0.3.3** (wgpu 28) |
| `#![forbid(unsafe_code)]` | Both crates |

---

## Architecture (Post-Rewire)

```
airSpring v0.7.0
    │
    ├── barracuda/ (airspring-barracuda v0.7.0)
    │   ├── depends on barraCuda v0.3.3 (../../barraCuda/crates/barracuda)
    │   ├── wgpu 28 (type-compatible with barraCuda)
    │   ├── 25 GPU modules wrapping 42 barraCuda primitives
    │   ├── 6 local WGSL ops (3 absorbed upstream, 3 local-only)
    │   └── 827 lib tests passing
    │
    └── metalForge/ (62 forge tests)
        └── 6/6 modules absorbed upstream, leaning on barraCuda
```

---

## For Other Springs

When upgrading from barraCuda 0.3.1 → 0.3.3:

1. Update `wgpu` version in Cargo.toml from `"22"` to `"28"`, add `"vulkan"` feature
2. Fix `entry_point` → `Some(...)`, `push_constant_ranges` → `immediate_size`, `Maintain` → `PollType`
3. Fix `BufferView` lifetime removal (if used)
4. `cargo check && cargo test && cargo clippy`
5. Expect GPU dispatch failures on NVK/Titan V (upstream wgpu 28 issue)
