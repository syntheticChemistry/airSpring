# airSpring → wateringHole Handoff V037

**Date**: March 1, 2026
**From**: airSpring v0.5.6
**To**: ToadStool/BarraCuda team, ecoPrimals ecosystem
**Status**: 56 experiments, 1237 Python + 636 lib + 57 forge tests, 65 binaries
**Supersedes**: V036 (GPU Pipeline + Mixed Hardware + NUCLEUS)
**ToadStool HEAD**: `1dd7e338` (S70+++)
**License**: AGPL-3.0-or-later

---

## Executive Summary

- **Ops 5-8 GPU-first**: All four Tier B orchestrators (SensorCal, Hargreaves, KcClimate, DualKc) rewired from CPU fallback to GPU dispatch via `BatchedElementwiseF64` after ToadStool S70+ cross-spring absorption
- **Seasonal pipeline GPU Stages 1-2**: `SeasonalPipeline::gpu()` now dispatches both ET₀ (op=0) and Kc climate adjustment (op=7) to GPU; stages 3-4 remain CPU
- **636 lib tests** (up from 630): 6 new GPU parity tests for ops 5-8
- **78/78 pure GPU pipeline PASS** (Exp 055): Validated after rewiring
- **104/104 metalForge mixed-hardware PASS**: NPU→GPU PCIe bypass, NUCLEUS atomics
- **21.0× geometric mean speedup** over Python (18/18 CPU parity)
- **Zero clippy warnings** (pedantic, both crates)
- **ToadStool S70+ absorption validated**: `seasonal_pipeline.wgsl`, `brent_f64.wgsl`, `stats::hydrology`, `stats::jackknife/diversity/evolution`, `staging::pipeline`, DF64 ML primitives all confirmed available

---

## Part 1: What Changed (V036 → V037)

### 1.1 Ops 5-8 Rewired to GPU Dispatch

Previously, all four Tier B orchestrators had `TODO(toadstool)` comments and CPU fallback. ToadStool S70+ (`d3dae8bb`) absorbed ops 5-8 into `batched_elementwise_f64.wgsl` with full WGSL implementations.

airSpring rewiring:

| Op | File | Before | After |
|----|------|--------|-------|
| 5 (SensorCal) | `gpu/sensor_calibration.rs` | CPU fallback | `engine.execute(&packed, n, Op::SensorCalibration)` |
| 6 (Hargreaves) | `gpu/hargreaves.rs` | CPU fallback | `engine.execute(&packed, n, Op::HargreavesEt0)` |
| 7 (KcClimate) | `gpu/kc_climate.rs` | CPU fallback | `engine.execute(&packed, n, Op::KcClimateAdjust)` |
| 8 (DualKc) | `gpu/dual_kc.rs` | CPU fallback | `engine.execute(&packed, n, Op::DualKcKe)` + CPU state update |

Each orchestrator:
- Dispatches to GPU when a `WgpuDevice` is provided
- Falls back to CPU when constructed without a device
- Has new GPU parity tests (`compute_gpu_device_dispatch`, `compute_gpu_matches_cpu`)

### 1.2 Seasonal Pipeline GPU Stages 1-2

`SeasonalPipeline::gpu(device)` now initializes both:
- `BatchedEt0` for Stage 1 (ET₀, op=0)
- `BatchedKcClimate` for Stage 2 (Kc adjustment, op=7)

The `compute_kc_batch()` method dispatches to GPU or falls back to CPU, matching the existing `compute_et0_batch()` pattern.

### 1.3 Evolution Gaps Updated

- 4 entries promoted from Tier B → Tier A
- Gap count: 15 Tier A, 7 Tier B, 1 Tier C
- `ToadStool` HEAD reference updated to `1dd7e338` (S70+++)
- S68++–S70+++ changelog appended

### 1.4 Device Info Fix

`F64BuiltinCapabilities` struct gained `basic_f64` field in ToadStool S70++. airSpring test updated to include it.

---

## Part 2: What ToadStool Should Know

### 2.1 Remaining GPU Gaps (Stages 3-4)

The seasonal pipeline still runs stages 3-4 on CPU:
- **Stage 3 (Water Balance, op=1)**: Already has `BatchedWaterBalance::gpu_step()`, but the seasonal pipeline uses `WaterBalanceState::step()` which has sequential dependencies (de_prev → de_new). The fused `seasonal_pipeline.wgsl` shader handles this internally.
- **Stage 4 (Yield Response)**: Simple scalar computation, not worth GPU dispatch alone.

### 2.2 Fused Seasonal Pipeline Shader

ToadStool has `seasonal_pipeline.wgsl` which fuses ET₀ → Kc → WB → Stress in a single GPU dispatch. This eliminates per-stage round-trips entirely. airSpring is ready to wire to this when a Rust executor is added.

**Key**: The shader uses fixed Kc interpolation params (`kc_prev`, `kc_next`, `day_in_stage`, `stage_length`) rather than the FAO-56 growth stage schedule. airSpring's `stage_kc()` function maps growth fraction to Kc, which would need to be adapted.

### 2.3 Brent GPU Shader

ToadStool has `brent_f64.wgsl` with VG inverse (op=0) and Green-Ampt (op=1). airSpring's `eco::richards::inverse_van_genuchten_h()` currently uses CPU `optimize::brent`. A batched GPU path via this shader would accelerate Richards equation preprocessing.

**Note**: Line 49 of `brent_f64.wgsl` has invalid WGSL syntax (`(h: f64 - h + 1.0)` in `green_ampt_residual`).

### 2.4 StreamingPipeline = ChainedPipeline

The `ChainedPipeline` primitive requested in V036 already exists as `StreamingPipeline` (via `PipelineBuilder`). This chains GPU stages without CPU readback between them. airSpring's metalForge pipeline is separate — it handles cross-device routing (CPU/GPU/NPU), not single-GPU chaining.

### 2.5 New Capabilities Available

| ToadStool S70+ Addition | Relevance to airSpring |
|------------------------|----------------------|
| `stats::hydrology` | Already delegated (`hargreaves_et0_batch`, `fao56_et0`, `crop_coefficient`) |
| `stats::jackknife` | Available for jackknife uncertainty estimation on soil parameters |
| `stats::evolution` | Not currently used |
| `nn::simple_mlp` | Available for surrogate models (WDM, ESN readout) |
| `staging::pipeline` | Single-GPU chaining available for fused seasonal pipeline |
| `GpuDriverProfile::basic_f64` | New capability probe for f64 shader compilation |

---

## Part 3: Cumulative State (V037)

| Metric | V036 | V037 |
|--------|------|------|
| Lib tests | 630 | 636 |
| Forge tests | 57 | 57 |
| Python tests | 1237 | 1237 |
| Binaries | 65 | 65 |
| Tier A orchestrators | 11 | 15 |
| Tier B orchestrators | 4 | 0 (all promoted) |
| GPU pipeline checks | 78/78 | 78/78 |
| Mixed-hardware checks | 104/104 | 104/104 |
| Seasonal pipeline GPU stages | 1 | 2 |
| ToadStool HEAD | e96576ee (S68+) | 1dd7e338 (S70+++) |
| Rust vs Python speedup | 20.1× | 21.0× |

---

## Part 4: Recommended ToadStool Actions

1. **Add Rust executor for `seasonal_pipeline.wgsl`**: This would let airSpring wire to the fused shader for zero-round-trip seasonal simulation. Suggested API: `SeasonalPipelineBatch::execute(inputs, params) -> Vec<SeasonalOutput>`

2. **Fix `brent_f64.wgsl` line 49**: The Green-Ampt residual has invalid WGSL syntax. Once fixed, add a Rust executor for batched VG inverse and Green-Ampt root-finding.

3. **PCIe P2P buffer sharing**: `metalForge::pipeline::TransferPath::PciePeerToPeer` is modeled but needs real Vulkan external memory or ioctl for actual P2P transfers.

4. **biomeOS graph integration**: airSpring has TOML graph parsing (`metalForge::graph::GraphDef`). ToadStool could use `StreamingPipeline` stages as graph node executors.

---

## Part 5: Next Steps for airSpring

- Wire fused `seasonal_pipeline.wgsl` when Rust executor available (eliminate stages 3-4 CPU)
- Add batched GPU VG inverse via `brent_f64.wgsl` once the shader bug is fixed
- Explore `StreamingPipeline` for multi-year atlas stream (eliminate per-year round-trips)
- Real PCIe P2P validation on multi-GPU rigs
- Expand NUCLEUS mesh to multi-node deployment

---

*Handoff generated from airSpring v0.5.6 at ToadStool HEAD 1dd7e338 (S70+++).*
*Previous handoffs archived in `wateringHole/handoffs/archive/`.*
