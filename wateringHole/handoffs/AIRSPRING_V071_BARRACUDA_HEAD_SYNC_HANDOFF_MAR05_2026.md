<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# airSpring V0.7.1 — barraCuda HEAD Sync + metalForge wgpu 28

**Date**: March 5, 2026
**From**: airSpring v0.7.1 (ecology/agriculture validation Spring)
**To**: barraCuda team / ToadStool S94b+ / All Springs
**Supersedes**: V070 rewire handoff (barraCuda 0.3.3 release pin)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring synced to barraCuda HEAD (`15d3774`, 6 commits past v0.3.3 release).
metalForge forge migrated from wgpu 22 to wgpu 28, eliminating duplicate wgpu
compilation. Subgroup capability detection wired into `DevicePrecisionReport`.
New upstream features (TensorContext, DF64 ComputeDispatch, naga rewriter fix)
documented as future evolution paths. coralNAK documented in GPU promotion map.

All tests pass at previous baselines:
- 827 lib pass (25 GPU fail upstream, unchanged)
- 188 forge pass (2 GPU fail upstream)
- 24/24 CPU vs Python parity (21.0x geometric mean speedup)
- 0 clippy warnings (pedantic+nursery, both crates)

---

## Part 1: metalForge wgpu 22 -> 28 Migration

metalForge forge was the last airSpring crate on wgpu 22. This caused double
wgpu compilation (v22 + v28) in the workspace, wasting ~17s per clean build.

| File | Change |
|------|--------|
| `metalForge/forge/Cargo.toml` | `wgpu = "22"` -> `"28"`, add `"vulkan"` feature, add `pollster = "0.3"` |
| `metalForge/forge/src/probe.rs` | `Instance::new(desc)` -> `Instance::new(&desc)` |
| `metalForge/forge/src/probe.rs` | `enumerate_adapters(backends)` -> `pollster::block_on(enumerate_adapters(backends))` |

Only `probe.rs` uses wgpu directly (GPU adapter enumeration). All compute
dispatch goes through barraCuda.

---

## Part 2: Subgroup Capability Detection

`DevicePrecisionReport` in `gpu/device_info.rs` now reports subgroup
(warp/wavefront) sizes from `wgpu::AdapterInfo`:

```rust
pub subgroup_min_size: u32,  // NVIDIA: 32, AMD RDNA: 32/64, Intel: 8-32
pub subgroup_max_size: u32,  // Zero if adapter does not report
```

Probed from `device.adapter_info().subgroup_min_size` / `.subgroup_max_size`
(wgpu 28 `AdapterInfo` fields). Displayed in precision report output.

---

## Part 3: Upstream Features Documented (Future Evolution)

These barraCuda HEAD features are documented in `evolution_gaps.rs` and
`local_dispatch.rs` as future optimization paths:

| Feature | What It Enables | airSpring Status |
|---------|----------------|------------------|
| `TensorContext` | Pooled buffers, pipeline cache, batched submits | Documented; adoption requires `LocalElementwise` buffer rewrite |
| `ComputeDispatch::df64()` | DF64 shader path on Hybrid consumer GPUs | Documented; adoption changes precision behavior |
| Subgroup detection | Subgroup-optimized reduction shaders | Wired into `DevicePrecisionReport` |
| Naga rewriter fix | Compound assignments in DF64 rewrite | Transparent (upstream fix, no airSpring changes needed) |
| `chi_squared` GPU gate | CPU-only builds no longer break | Transparent (upstream fix) |

---

## Part 4: coralNAK Awareness

coralNAK (`ecoPrimals/coralNAK/`) documented in `specs/GPU_PROMOTION_MAP.md`
as part of the sovereign compute evolution roadmap:

```
barraCuda DF64 (complete) -> coralNAK (Phase 2, 183 tests) -> coralDriver -> coralGpu
```

coralNAK is a Rust NVIDIA shader compiler forked from Mesa's NAK. It will
fix f64 transcendental emission at the shader compiler level, replacing the
DF64 workaround with native f64 compilation.

---

## Part 5: Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** (both crates) |
| `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings` | **PASS** — 0 warnings |
| `cargo test --lib` | **827 pass**, 25 fail (upstream GPU wgpu 28 NVK) |
| `cargo test --test '*'` | **188 pass**, 2 fail (upstream GPU) |
| Cross-spring evolution | **11/11 pass** |
| CPU vs Python | **24/24 algorithms**, 21.0× geometric mean speedup |
| `#![forbid(unsafe_code)]` | **Both crates** |
| barraCuda source | HEAD post-0.3.3 (`15d3774`, wgpu 28) |
| metalForge wgpu | **28** (was 22) |
