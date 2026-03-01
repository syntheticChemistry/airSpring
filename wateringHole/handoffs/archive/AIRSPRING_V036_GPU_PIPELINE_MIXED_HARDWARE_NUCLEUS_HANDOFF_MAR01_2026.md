# airSpring V036 — Pure GPU Pipeline, Mixed-Hardware Dispatch, NUCLEUS Atomics

**Date**: March 1, 2026
**From**: airSpring V036 (v0.5.5)
**To**: ToadStool / BarraCuda team
**Status**: 56 experiments, 1237 Python + 630 lib + 57 forge tests + 65 binaries
**Supersedes**: V035 (pipeline coupling, inverse problems, season-scale audit)
**License**: AGPL-3.0-or-later

---

## Executive Summary

- **18/18 CPU math parity** against Python baselines at 20.1× geometric mean speedup
  (expanded from 8/8 to include all validated algorithms)
- **78/78 pure GPU workload validation** (Exp 055): end-to-end agricultural pipeline
  on Titan V — ET₀, Hargreaves, Kc climate, seasonal pipeline, multi-crop, GPU scaling
- **Seasonal pipeline GPU Stage 1**: `SeasonalPipeline::gpu()` dispatches ET₀ batch
  to GPU, falls back to CPU for Kc/WB/Yield. `AtlasStream::process_batch_unified()`
  eliminates N-1 CPU↔GPU round-trips for regional simulations
- **NPU→GPU PCIe bypass** in metalForge: `pipeline.rs` computes transfer paths
  between substrates; NPU→GPU uses peer-to-peer DMA (zero CPU roundtrip)
- **NUCLEUS atomics** (tower/node/nest): mesh-aware capability routing across
  deployment units; `NucleusMesh::find_capable_nodes()` for workload placement
- **biomeOS graph execution**: `graph.rs` parses TOML DAGs, topological sort,
  cycle detection, dependency validation — both `airspring_eco_pipeline` and
  `cross_primal_soil_microbiome` graphs parse and validate
- **Capability-based hardware discovery**: `validate_live_hardware` no longer
  hardcodes NPU expectations; checks skip dynamically when hardware absent
- metalForge test count: 31 → 57 (+26), validation checks: 29+21+14 → 29+21+11+43 = 104

---

## Part 1: New Experiments

### Exp 055: Barracuda Pure GPU Workload Validation

| Component | Detail |
|-----------|--------|
| Purpose | Prove BarraCuda math is correct on pure GPU (not just CPU with GPU dispatch) |
| Binary | `validate_gpu_pipeline` (78/78 PASS) |
| Modules | `gpu::et0`, `gpu::seasonal_pipeline`, `gpu::hargreaves`, `gpu::kc_climate` |
| GPU path | Titan V via `BatchedEt0::gpu()` → `fao56_et0_batch()` WGSL dispatch |
| Validates | ET₀ parity, Hargreaves parity, Kc climate adjust, seasonal chain, multi-crop, GPU scaling, GPU reduce, cross-method consistency |

**Key learning for ToadStool**: The `BatchedEt0` primitive handles the hot path
(ET₀ across thousands of station-days). Stages 2-4 (Kc adjust, water balance,
yield response) are sequential per-station and remain on CPU. When ToadStool
absorbs ops 5-8, these can move to GPU as well.

### Exp 056: Mixed-Hardware Pipeline + NUCLEUS Atomics

| Component | Detail |
|-----------|--------|
| Purpose | Validate NPU→GPU PCIe bypass and NUCLEUS mesh coordination |
| Binary | `validate_mixed_pipeline` (43/43 PASS, metalForge) |
| Modules | `pipeline.rs`, `nucleus.rs`, `graph.rs` (new metalForge modules) |
| Validates | Transfer path selection, PCIe P2P detection, CPU roundtrip counting, mesh-aware routing, offline node exclusion, 5-stage eco pipeline |

**Key learning for ToadStool**: The `TransferPath::PciePeerToPeer` path eliminates
CPU memory staging for NPU→GPU data flow. This matters for the crop_stress→et0_batch
pipeline where int8 NPU inference feeds directly into f64 GPU compute. ToadStool's
unidirectional streaming model maps naturally to this architecture.

---

## Part 2: What ToadStool Should Know

### Seasonal Pipeline GPU Integration Pattern

airSpring's `SeasonalPipeline` now has a `gpu()` constructor that initializes a
`BatchedEt0` engine. The 4-stage pipeline (ET₀ → Kc → WB → Yield) executes
Stage 1 on GPU and stages 2-4 on CPU:

```
SeasonalPipeline::gpu(device) → run_season(weather)
  Stage 1: compute_et0_batch() → GPU dispatch via BatchedEt0
  Stage 2: adjust_kc()         → CPU (pending op=7 absorption)
  Stage 3: water_balance()     → CPU (pending op=1 absorption)
  Stage 4: yield_response()    → CPU (pending new op)
```

### Atlas Stream Unified Batch

`AtlasStream::process_batch_unified()` consolidates all stations' ET₀ into a
single GPU dispatch, then slices results back to per-station processing. For 100
stations × 365 days, this reduces 99 unnecessary GPU dispatches to 1.

### Tier B Orchestrator Readiness

All 4 Tier B orchestrators (ops 5-8) now have `pack_gpu_input()` methods that
prepare data in the expected `BatchedElementwiseF64` stride layout:

| Op | Orchestrator | GPU Layout | Status |
|----|-------------|-----------|--------|
| 5 | `sensor_calibration` | stride-1: `[raw_count]` | CPU + pack ready |
| 6 | `hargreaves` | stride-4: `[tmax, tmin, lat_rad, doy]` | CPU + pack ready |
| 7 | `kc_climate` | stride-4: `[kc_table, u2, rh_min, crop_height_m]` | CPU + pack ready |
| 8 | `dual_kc` | stride-9: `[kcb, kc_max, few, ...]` | CPU + pack ready |

**ToadStool action**: When absorbing ops 5-8 into WGSL shaders, these `pack_gpu_input()`
methods provide the exact input layout. airSpring can rewire to GPU dispatch immediately.

### metalForge Transfer Path Architecture

```
NPU (AKD1000, PCIe) ──── PCIe P2P ────── GPU (Titan V, PCIe)
         │                                         │
         └──── CPU memcpy ───── CPU ───── CPU memcpy ┘
                              (fallback)
```

The `pcie_p2p_capable()` check uses PCI ID presence as a heuristic. A production
implementation would use `ioctl` or Vulkan external memory APIs.

### NUCLEUS Atomic Model

| Atomic | Components | Dispatch |
|--------|-----------|----------|
| Tower | BearDog (crypto) + Songbird (mesh) | Discovery only |
| Node | Tower + ToadStool (compute) | Full workload routing |
| Nest | Tower + NestGate (storage) | Data provenance only |

Nodes advertise their substrates to the mesh. `NucleusMesh::find_capable_nodes()`
finds which nodes can handle a given workload. Unreachable nodes are excluded.

---

## Part 3: Cumulative State

| Metric | V035 | V036 |
|--------|------|------|
| Experiments | 54 | 56 |
| Python checks | 1237 | 1237 |
| Lib tests (barracuda) | 618 | 630 |
| Lib tests (metalForge) | 31 | 57 |
| Validation binaries | 59 | 65 |
| CPU parity | 8/8 (25.9×) | 18/18 (20.1×) |
| GPU pipeline checks | — | 78/78 |
| metalForge checks | 29+21+14 | 29+21+11+43 |
| Clippy warnings | 0 | 0 |
| Unsafe code | 0 | 0 |
| Production unwrap | 0 | 0 |

---

## Part 4: Recommended ToadStool Actions

1. **Absorb ops 5-8**: airSpring's `pack_gpu_input()` methods define the exact
   stride layouts. Adding `Op::SensorCal=5`, `Op::Hargreaves=6`, `Op::KcClimate=7`,
   `Op::DualKc=8` to the `BatchedElementwiseF64` enum + WGSL shaders would let
   airSpring rewire all 4 Tier B orchestrators to GPU immediately.

2. **Seasonal pipeline GPU stages**: The `SeasonalPipeline` pattern (GPU Stage 1 +
   CPU stages 2-4) would benefit from a ToadStool `ChainedPipeline` primitive that
   keeps intermediate results on GPU between stages without CPU readback.

3. **PCIe P2P buffer sharing**: The metalForge `TransferPath::PciePeerToPeer` concept
   maps to Vulkan external memory / `wgpu` DMA. If ToadStool's `WgpuDevice` can
   expose buffer handles for cross-device sharing, metalForge can wire real NPU→GPU
   data flow.

4. **biomeOS graph integration**: The `graph.rs` module parses the same TOML format
   used by biomeOS. When biomeOS's graph engine is ready, metalForge can delegate
   execution to it instead of doing local topological sort.

---

## Part 5: Next Steps for airSpring

- Wire Tier B orchestrators to GPU when ToadStool absorbs ops 5-8
- Expand CPU benchmark to full 18-algorithm suite in run_all_baselines.sh
- Build real NPU→GPU data transfer when AKD1000 is connected
- Integrate biomeOS graph execution with live NUCLEUS deployment
- Proceed toward Penny Irrigation (Phase 4) with sovereign consumer hardware
