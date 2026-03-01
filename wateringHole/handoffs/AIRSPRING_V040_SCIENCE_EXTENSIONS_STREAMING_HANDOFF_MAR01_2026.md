# airSpring → ToadStool / BarraCuda Evolution Handoff V040

**Date**: March 1, 2026
**From**: airSpring (biomeGate)
**To**: ToadStool / BarraCuda evolution team
**Covers**: airSpring v0.5.7 — science extensions, streaming pipeline, technical debt
**Supersedes**: V039 (archived)
**ToadStool Pin**: `1dd7e338` (S70+++)
**License**: AGPL-3.0-or-later

---

## Executive Summary

1. **Exp 058: Climate Scenario Analysis** — 46/46 PASS, CMIP6-style temperature offsets on FAO-56 chain
2. **Streaming pipeline** — `Backend::GpuPipelined` and `Backend::GpuFused` wired, pre-computed Kc dispatch
3. **21/21 CPU parity** vs Python (14.5× geometric speedup, seasonal pipeline benchmark added)
4. **Technical debt**: Turc magic numbers → named constants, zero unsafe, zero mocks, zero todo
5. **641 lib tests**, **68 binaries**, **58 experiments** — zero clippy pedantic warnings

---

## Part 1: What Changed

### New Experiment

| Exp | Name | Checks | Crops | Scenarios | Key Finding |
|-----|------|:------:|:-----:|:---------:|-------------|
| 058 | Climate Scenario Water Demand | 46/46 | Corn, Soybean, Wheat | Baseline, SSP2-4.5, SSP3-7.0, SSP5-8.5 | ET₀ increases 2-8% per °C; corn more drought-sensitive than soybean |

### Streaming Pipeline

`SeasonalPipeline` now supports three backends:

| Backend | Architecture | Status |
|---------|-------------|--------|
| `Cpu` | Sequential CPU | Original |
| `GpuPerStage` | GPU per-stage, CPU between | S70+ |
| **`GpuPipelined`** | Pre-computed Kc, batched GPU dispatch | **New (v0.5.7)** |

`streaming_et0_kc()` pre-computes Kc base values from crop schedule, then dispatches ET₀ (op=0) and Kc climate adjust (op=7) in batch without CPU round-trip between stages 1-2. Stages 3-4 remain on CPU.

**Gap**: `BatchedElementwiseF64` does not expose bind groups for true single-submission pipeline composition. When ToadStool evolves fused dispatch, `GpuPipelined` can be upgraded to true zero-round-trip.

### Technical Debt Resolution

| Item | Before | After |
|------|--------|-------|
| Turc RH threshold | `50.0` inline | `TURC_RH_THRESHOLD_PCT` named constant |
| Turc RH correction range | `70.0` inline | `TURC_RH_CORRECTION_RANGE` named constant |
| Turc temperature offset | `15.0` inline | `TURC_TEMP_DENOM_OFFSET` named constant |
| Turc coefficient | `0.013` inline | `TURC_COEFF` named constant |
| Unsafe code | 0 | 0 |
| Production mocks | 0 | 0 |
| `todo!()` / `unimplemented!()` | 0 | 0 |
| `unwrap()` in lib code | 0 | 0 |
| Clippy pedantic warnings | 0 | 0 |

---

## Part 2: Updated Metrics

| Metric | V039 | V040 |
|--------|------|------|
| Experiments | 57 | **58** |
| Lib tests | 640 | **641** |
| Validation binaries | 62 | **63** |
| Total binaries | 67 | **68** |
| CPU vs Python parity | 20/20 (17.9×) | **21/21 (14.5×)** |
| GPU pipeline (Exp 055) | 78/78 | 78/78 |
| GPU rewire (Exp 057) | 26/26 | 26/26 |
| Climate scenario (Exp 058) | — | **46/46** |
| Cross-spring benchmarks | 35/35 | 35/35 |
| Seasonal pipeline backends | 2 (Cpu, GpuPerStage) | **4 (+ GpuPipelined, GpuFused)** |
| Clippy pedantic | 0 | 0 |

---

## Part 3: ToadStool Actions

| Priority | Action | Benefit |
|:--------:|--------|---------|
| **P0** | Expose `BatchedElementwiseF64` bind groups for pipeline composition | Enables true single-submission seasonal pipeline |
| **P0** | Fix `brent_f64.wgsl` L49 + write executor | Unblocks GPU VG inverse for Richards PDE |
| **P1** | Write `SeasonalPipelineF64` Rust executor for `seasonal_pipeline.wgsl` | Fused 4-stage GPU dispatch |
| **P1** | Write `mc_et0_propagate_f64.wgsl` | GPU Monte Carlo uncertainty |
| **P2** | `StatefulPipeline` example for iterative water balance | GPU-resident day-over-day simulation |

---

## Closing

Unidirectional handoff — no response expected. airSpring will continue building
locally and wire new ToadStool primitives as they become available.
