# AIRSPRING V029 — ToadStool S68+ Universal Precision Sync

**Date**: February 27, 2026
**airSpring Version**: v0.5.0
**ToadStool PIN**: S68+ HEAD (`e96576ee`)
**Previous PIN**: S68 (`89356efa`)
**Direction**: airSpring → ToadStool / BarraCuda / biomeOS / metalForge teams

---

## Executive Summary

ToadStool underwent massive evolution from S42 through S68+ (180+ commits). The most
significant change is the **Universal Precision Architecture** (S67–S68): all 703 WGSL
shaders are now **f64 canonical** — written once in f64, compiled to any target precision
via `compile_shader_universal(source, precision, label)`. This eliminates the f32/f64
shader duplication that previously required maintaining separate codepaths.

airSpring has now synced to S68+ HEAD (`e96576ee`), removed stale S60–S65 regression
workarounds, and documented the full precision architecture in our GPU module docs
and evolution gaps. All 515 tests pass, 0 clippy warnings.

---

## Part 1: ToadStool S68+ Universal Precision Architecture

### The Doctrine: "Math is Universal — Precision is Silicon"

ToadStool S67 codified a fundamental principle: mathematical operations are
precision-independent. A GPU shader computing FAO-56 ET₀ is the same math whether
the hardware does it in f64, Df64 (double-float f32 pairs), f32, or f16. Only
the precision target changes.

### Dual-Layer Implementation

**Layer 1 — Source (op_preamble)**:
Abstract math operations (`op_add`, `op_mul`, `op_div`, etc.) injected at compile
time via `Precision::op_preamble()`. For F16/F32/F64 these inline to native ops.
For Df64 they route to `df64_add`, `df64_mul`, etc.

**Layer 2 — Compiler (df64_rewrite)**:
`sovereign/df64_rewrite.rs` uses naga to parse f64 WGSL, walk the IR for f64
binary/unary ops, and replace with DF64 bridge function calls. Existing f64
shaders run on consumer GPUs without any source changes.

### Precision Enum

```rust
pub enum Precision { F16, F32, F64, Df64 }
```

### Compilation Entry Points

| Method | Purpose |
|--------|---------|
| `compile_shader_universal(source, precision, label)` | One f64 source → any target |
| `compile_op_shader(source, precision, label)` | Inject `op_preamble` + compile |
| `compile_template(template, precision, label)` | Render `ShaderTemplate` + compile |
| `compile_shader_f64(source, label)` | f64 + driver patching + ILP + sovereign |
| `compile_shader_df64(source, label)` | DF64 preamble + ILP + sovereign |

### Fp64Strategy — Per-Device Precision

```rust
pub enum Fp64Strategy {
    Native,  // f64:f32 ratio ≤ 2.5 (Titan V, A100, GV100)
    Hybrid,  // f64:f32 ratio > 2.5 (RTX 4070, consumer: DF64 bulk + f64 reductions)
}
```

Selected automatically by `GpuDriverProfile::fp64_strategy()` from
`probe_f64_throughput_ratio(device)`.

### Shader Inventory (703 total)

| Category | Count | Notes |
|----------|-------|-------|
| f32 (LazyLock downcast from f64) | 497 (71%) | All from f64 canonical source |
| Native f64 | 182 (26%) | Scientific, lattice QCD, MD, PDE |
| Df64 (double-float f32 pair) | 19 (3%) | Consumer GPU f64-class work |
| Df64 infrastructure | 2 | `df64_core.wgsl` + `df64_transcendentals.wgsl` |

---

## Part 2: airSpring Changes in This Sync

### Stale Debt Removed

**`try_gpu` catch_unwind pattern** — removed from `gpu::et0` and `gpu::water_balance`
test modules. This was a workaround for the S60–S65 sovereign compiler bind-group
regression. S66 fixed it with explicit `BindGroupLayout` (R-S66-041), and we
validated live on Titan V (24/24 PASS). The catch_unwind was masking real failures.

### Architecture Docs Updated

- `gpu::mod.rs` — rewritten with universal precision architecture diagram
- `gpu::evolution_gaps.rs` — inventory updated to S68+ with 11 new Available
  capabilities (compile_shader_universal, Fp64Strategy, UnidirectionalPipeline,
  StatefulPipeline, MultiDevicePool, ShaderTemplate, compile_op_shader, probes)
- `gpu::mc_et0` — sovereign compiler regression marked RESOLVED (S66+)
- `EVOLUTION_READINESS.md` — S67/S68/S68+ milestones added, quality gates updated

### Test Counts

| Metric | Value |
|--------|-------|
| `cargo test --lib` | **515 passed**, 0 failed |
| clippy (pedantic + nursery) | **0 warnings** |
| Validation binaries | 47/47 + 4/4 = **51/51** |
| GPU live (Titan V) | **24/24 PASS** |
| metalForge live | **17/17 PASS** |

---

## Part 3: New Upstream Capabilities Available (Not Yet Wired)

These capabilities exist in ToadStool S68+ and are documented in our evolution gaps
as "Available" — we haven't needed them yet but they're ready when we do:

### Streaming & Stateful Pipelines

| Primitive | API | Use Case |
|-----------|-----|----------|
| `UnidirectionalPipeline` | `staging::unidirectional` | Fire-and-forget streaming: CPU writes, GPU computes, no round-trip. Perfect for continuous ET₀ over multi-year regional grids. |
| `StatefulPipeline` | `staging::stateful` | GPU-resident iterative solvers with minimal readback (convergence scalar only). Ideal for Richards PDE iterations. |
| `MultiDevicePool` | `multi_gpu` | Multi-GPU dispatch with load balancing and device requirements. Scaling to 2+ GPUs. |

### Precision-Aware Compilation

| Primitive | API | Use Case |
|-----------|-----|----------|
| `compile_shader_universal` | `WgpuDevice` method | One f64 source → any precision. airSpring currently uses `BatchedElementwiseF64` which handles precision internally — but custom shaders could use this directly. |
| `probe_f64_builtins` | `device::probe` | Cached probing of which f64 builtins (exp, log, sin, cos, sqrt, fma) are native. |
| `probe_f64_throughput_ratio` | `device::probe_throughput` | Cached f64:f32 FMA ratio → `F64Tier::{Native, Capable, Consumer, Throttled}`. |

### GPU Optimizers

| Primitive | API | Use Case |
|-----------|-----|----------|
| `NelderMeadGpu` | `optimize::nelder_mead_gpu` | GPU-resident Nelder-Mead for 5–50 parameter problems. Not cost-effective for our 2-param isotherms but useful for multi-adsorbate models. |
| `BatchedBisectionGpu` | `optimize::batched_bisection_gpu` | GPU-parallel root-finding for N independent problems. Ideal for inverse VG θ(h) across spatial grids. |

---

## Part 4: Recommendations for ToadStool Team

### 4.1 Shannon/Simpson Batch Ops

The `batched_elementwise_f64.wgsl` shader defines `SHANNON_BATCH` (op=2) and
`SIMPSON_BATCH` (op=3), but the Rust `Op` enum in `BatchedElementwiseF64` only
goes up to `Custom` (op=2). Exposing these in the Rust API would let airSpring
(and wetSpring) do batched diversity metrics on GPU without `FusedMapReduceF64`.

### 4.2 Hargreaves ET₀ Batch Op

airSpring has 7 validated ET₀ methods (FAO-56 PM, Hargreaves, Priestley-Taylor,
Thornthwaite, Makkink, Turc, Hamon). Hargreaves needs only `tmax`, `tmin`, `Ra`
(3 inputs vs FAO-56's 9) and is already validated. A `HARGREAVES_BATCH` (op=6)
in `batched_elementwise_f64.wgsl` would be a low-effort, high-impact addition.

### 4.3 Device-Lost Recovery Pattern

S68+ added `device-lost resilience`. airSpring's GPU tests currently skip when
no device is found (`try_device() → None`). If `WgpuDevice` can now recover
from device-lost, we could retry instead of skip — especially important for
long-running metalForge dispatch sequences.

---

## Part 5: Cross-Spring Sync Status

| Spring | ToadStool PIN | Key Evolution Since Our Last Review |
|--------|---------------|-------------------------------------|
| ToadStool | **S68+ (e96576ee)** | 703 shaders, universal precision, dual-layer, device-lost resilience |
| airSpring | **v0.5.0** | 44 experiments, 515 tests, Titan V live, metalForge live, S68+ synced |
| wetSpring | S68 (f0feb226) | 79 ToadStool primitives, NPU inference bridge proposed |
| neuralSpring | S68 (f0feb226) | `compile_shader_df64_streaming`, MLP/LSTM/ESN |
| hotSpring | — | Lattice QCD, MD transport, f64 math lineage (df64_core, df64_transcendentals) |
| groundSpring | S50–S62 | MC propagation, `batched_multinomial`, three-mode CI |

---

## Quality Certificate

```
airSpring v0.5.0 — ToadStool S68+ sync validation
  cargo test --lib          515 passed, 0 failed
  cargo clippy --all-targets  0 warnings (pedantic + nursery)
  GPU live (Titan V)        24/24 PASS
  metalForge live           17/17 PASS
  try_gpu catch_unwind      REMOVED (stale S60-S65 debt)
  evolution_gaps            Updated to S68+ inventory
  EVOLUTION_READINESS       S68+ PIN documented
```
