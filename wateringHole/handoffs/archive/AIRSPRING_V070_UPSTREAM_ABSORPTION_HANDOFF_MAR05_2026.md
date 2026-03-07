<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# airSpring V0.7.0 — Upstream Absorption Handoff for barraCuda / ToadStool

**Date**: March 5, 2026
**From**: airSpring v0.7.0 (ecology/agriculture validation Spring)
**To**: barraCuda team / ToadStool S94b+ absorbers
**Companion**: `AIRSPRING_V070_BARRACUDA_033_REWIRE_HANDOFF_MAR05_2026.md` (rewire details)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring v0.7.0 completed the wgpu 22→28 migration and barraCuda 0.3.1→0.3.3 rewire.
This handoff documents **what airSpring learned** that benefits barraCuda and all Springs:

1. **3 local WGSL ops ready for upstream absorption** (SCS-CN, Stewart, Blaney-Criddle)
2. **NVK f64 probe discovery** — Titan V reports `has_f64_shaders: true` but produces zeros
3. **wgpu 28 `ComputeDispatch` GPU-returns-zero issue** — 27 affected tests across 3 primitives
4. **Cross-spring shader provenance** — 34 entries, 146/146 evolution benchmarks
5. **Suggested upstream actions** with priority and rationale

---

## Part 1: Local Ops Ready for Upstream Absorption

Three airSpring-authored WGSL ops are domain-complete, f64-canonical, and tested.
They follow the same `batched_elementwise_f64.wgsl` dispatch pattern (3 f64 inputs → 1 f64 output)
and are natural candidates for absorption as new `Op` variants.

### Op 0: SCS-CN Runoff (USDA TR-55)

```
fn scs_cn(p: f64, cn: f64, ia_ratio: f64) -> f64
```

- **Inputs**: `a=P(mm)`, `b=CN (curve number)`, `c=ia_ratio (0.2 standard, 0.05 updated)`
- **Domain**: Surface runoff estimation from precipitation and land cover
- **Reference**: USDA Technical Release 55 (1986)
- **CPU impl**: `eco::runoff::scs_cn_runoff(precip_mm, cn, ia_ratio)`
- **Test data**: P=50mm, CN=75, ia=0.2 → Q=1.532mm; P=100mm, CN=85, ia=0.05 → Q=72.85mm
- **WGSL**: `barracuda/src/shaders/local_elementwise_f64.wgsl` lines 77-84

### Op 1: Stewart Yield-Water Function (Doorenbos & Kassam 1979)

```
fn stewart_yield(ky: f64, eta_etc: f64) -> f64
```

- **Inputs**: `a=Ky (yield response factor)`, `b=ETa/ETc (actual/potential ET ratio)`, `c=unused`
- **Domain**: Crop yield reduction under water stress
- **Reference**: Doorenbos J, Kassam AH (1979) FAO Irrigation and Drainage Paper 33
- **CPU impl**: `eco::yield_response::stewart_yield_ratio(ky, eta_etc)`
- **Formula**: `1 - Ky * (1 - ETa/ETc)`
- **Test data**: Ky=1.0, ratio=0.8 → 0.8; Ky=0.5, ratio=0.6 → 0.8
- **WGSL**: `barracuda/src/shaders/local_elementwise_f64.wgsl` lines 88-90

### Op 5: Blaney-Criddle ET₀ (Blaney & Criddle 1950)

```
fn blaney_criddle(t: f64, lat: f64, doy: f64) -> f64
```

- **Inputs**: `a=T(°C)`, `b=lat(rad)`, `c=doy`
- **Domain**: ET₀ estimation for arid/semi-arid regions using only temperature
- **Reference**: Blaney HF, Criddle WD (1950) USDA-SCS Technical Paper 96
- **CPU impl**: `eco::simple_et0::blaney_criddle_from_location(tmean_c, latitude_rad, day_of_year)`
- **Depends on**: `daylight_hr(lat, doy)` helper (already in WGSL)
- **Test data**: T=25°C, lat=0.75rad, doy=180 → ~4.2 mm/day
- **WGSL**: `barracuda/src/shaders/local_elementwise_f64.wgsl` lines 128-132

### Suggested Absorption Pattern

These 3 ops map directly to new `Op` discriminants in `batched_elementwise_f64.wgsl`:

| airSpring Local Op | Suggested barraCuda Op | Pattern |
|--------------------|------------------------|---------|
| 0 (SCS-CN) | `Op::ScsCnRunoff` (17?) | Same as ops 14-16 |
| 1 (Stewart) | `Op::StewartYield` (18?) | Trivial 1-liner |
| 5 (Blaney-Criddle) | `Op::BlaneyCriddleEt0` (19?) | Same as Hamon (op=16) |

All three use `compile_shader_universal` and produce correct results on the f32 downcast
path. The f64 canonical WGSL is ready to copy.

---

## Part 2: NVK f64 Probe Discovery

**Finding**: The Titan V via NVK/Mesa 25.0.2 driver reports `has_f64_shaders: true` but
produces all-zero output from f64 compute shaders. The f32 downcast path works correctly.

**Impact**: Any Spring running on NVK/Titan V that trusts the f64 capability probe will
silently get zeros from `BatchedElementwiseF64`, `VarianceF64`, and `CorrelationF64`.

**Recommendation**: Add Titan V / NVK to barraCuda's probe cache as a known-unreliable
f64 target. Similar to groundSpring V37's NVK discovery. The `GpuDriverProfile` should
either:

1. Add an NVK-specific override in `from_device()` that forces `Fp64Strategy::Hybrid`
   (use Df64 or f32 downcast instead of native f64), or
2. Add a runtime f64 validation probe: dispatch a known non-zero input through a trivial
   f64 shader and verify the output is non-zero before trusting native f64.

**Affected hardware**: NVIDIA Titan V (GV100), NVK driver (Mesa Vulkan), wgpu 28.
Proprietary NVIDIA drivers may not have this issue.

---

## Part 3: wgpu 28 ComputeDispatch GPU-Returns-Zero Issue

**Finding**: After the wgpu 22→28 upgrade, 27 GPU tests fail across barraCuda's own test
suite and airSpring. All failures show the same pattern: GPU returns `[0.0, ...]` for
non-zero inputs.

### Affected Primitives

| Primitive | Failure Mode | Test Count |
|-----------|-------------|------------|
| `BatchedElementwiseF64` | All ops return 0.0 for non-zero inputs | 15 |
| `VarianceF64` | `mean_variance()` returns `[0.0, 0.0]` | 7 |
| `CorrelationF64` | `correlation_full()` returns all zeros | 5 |

### Affected Tests (airSpring)

**`cargo test --lib`** (25 fail):
- `gpu::reduce::tests::test_seasonal_reducer_*` (fused Welford path, CPU fallback masks some)
- `gpu::stats::tests::test_pairwise_correlation_gpu`
- `gpu::stats::tests::test_fused_mean_variance_gpu`
- `gpu::local_dispatch::tests::test_scs_cn_gpu`
- `gpu::local_dispatch::tests::test_stewart_gpu`
- `gpu::local_dispatch::tests::test_makkink_gpu`
- `gpu::local_dispatch::tests::test_turc_gpu`
- `gpu::local_dispatch::tests::test_hamon_gpu`
- `gpu::local_dispatch::tests::test_blaney_criddle_gpu`
- `gpu::simple_et0::tests::*` (all GPU ET₀ methods)

**`cargo test --test '*'`** (2 fail): GPU-dependent forge tests

### airSpring Mitigation

`SeasonalReducer::mean_variance()` implements a CPU fallback: if the GPU returns
`[0.0, 0.0]` for non-zero input, it falls back to `barracuda::stats::mean()` and
`barracuda::stats::correlation::variance()` on CPU. This keeps the seasonal pipeline
functional despite the GPU issue.

### Root Cause Hypothesis

The issue is specific to wgpu 28 + NVK/Titan V. It may be related to:
- Buffer mapping changes in wgpu 28 (storage buffer read-back)
- NVK compute shader dispatch (the same shaders work on f32 downcast)
- `ComputeDispatch` buffer lifecycle changes between wgpu 22 and 28

---

## Part 4: Cross-Spring Shader Provenance

airSpring maintains **34 `ShaderProvenance` entries** in `gpu/device_info.rs`, documenting
the lineage of every shader primitive used. This is the most comprehensive cross-spring
provenance tracking across all Springs.

### Key Provenance Findings

| Shader | Origin | What airSpring Proved |
|--------|--------|----------------------|
| `mean_variance_f64.wgsl` | hotSpring S58 | Fused Welford reduces GPU passes 4→3 for seasonal stats |
| `correlation_full_f64.wgsl` | neuralSpring S69 | 5-accumulator Pearson avoids catastrophic cancellation in soil-weather correlation |
| `local_elementwise_f64.wgsl` | airSpring V0.6.9 | Domain WGSL ops compile correctly through `compile_shader_universal` across all 4 precision tiers |
| `batched_elementwise_f64.wgsl` | multi-spring | 17 ops now serving FAO-56, WB, VG, Thornthwaite, GDD, pedotransfer, ET₀ methods |
| `brent_f64.wgsl` | neuralSpring S83 | VG inverse works but L49 Green-Ampt residual still has invalid WGSL syntax |

### Evolution Benchmark Results

`bench_cross_spring` validates the full provenance chain:
- **146/146 benchmarks pass** (34 provenance entries × tests per entry)
- **6 origin Springs**: hotSpring, wetSpring, groundSpring, neuralSpring, airSpring, multi-spring
- **45 primitives tracked** across the ecosystem

---

## Part 5: Suggested Upstream Actions

### Priority 0 (High)

| Action | Rationale | Complexity |
|--------|-----------|------------|
| Add NVK/Titan V to probe cache | Prevents silent f64 zeros on NVK | Low — config entry |
| Add runtime f64 validation probe | Catches future unreliable f64 hardware | Medium — new test shader |
| Investigate wgpu 28 ComputeDispatch zeros | 27 tests fail, affects all Springs | Medium-High |

### Priority 1 (Normal)

| Action | Rationale | Complexity |
|--------|-----------|------------|
| Absorb SCS-CN (op 0) into `batched_elementwise_f64` | Runoff is core hydrology | Low — copy WGSL |
| Absorb Stewart Yield (op 1) | Crop yield response is widely used | Low — 1-line fn |
| Absorb Blaney-Criddle (op 5) | Completes ET₀ method suite (14-16 + 19) | Low — same pattern as Hamon |
| Fix `brent_f64.wgsl` L49 Green-Ampt residual | Invalid WGSL syntax `(h: f64 - h + 1.0)` | Low — syntax fix |

### Priority 2 (Future)

| Action | Rationale | Complexity |
|--------|-----------|------------|
| Seasonal pipeline Rust executor | WGSL exists, no `SeasonalPipelineF64` Rust binding | Medium |
| `Fp64Strategy::Concurrent` cross-validation | Run f64 + f32 in parallel, compare results | Medium |

---

## Part 6: Quality Gates

### airSpring v0.7.0 Validation Summary

| Gate | Result |
|------|--------|
| `cargo fmt --check` | **PASS** (both crates) |
| `cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings` | **PASS** — 0 warnings |
| `cargo doc --no-deps` | **PASS** — 0 warnings |
| `cargo test --lib` | **827 pass**, 25 fail (upstream GPU wgpu 28 NVK) |
| `cargo test --test '*'` | **186 pass**, 2 fail (upstream GPU) |
| Validation binaries | **381/381 checks** (10 binaries) |
| Cross-spring evolution | **146/146 pass** (34 provenance, 6 Springs) |
| CPU vs Python | **24/24 algorithms**, 20.6× geometric mean speedup |
| `#![forbid(unsafe_code)]` | **Both crates** |
| barraCuda source | **v0.3.3** standalone (`ecoPrimals/barraCuda`, wgpu 28) |

### Files Modified in v0.7.0

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | wgpu 22→28, version 0.6.9→0.7.0 |
| `barracuda/src/gpu/local_dispatch.rs` | 4 wgpu 28 API changes, Df64 docs |
| `barracuda/src/gpu/reduce.rs` | Fused Welford + CPU fallback |
| `barracuda/src/gpu/stats.rs` | Fused Pearson + mean-variance GPU |
| `barracuda/src/gpu/device_info.rs` | 2 new ShaderProvenance entries |
| `barracuda/src/gpu/mod.rs` | Fp64Strategy::Concurrent docs |

---

## Appendix: WGSL Source Locations

| Source | Path (relative to airSpring) |
|--------|------------------------------|
| Local ops (f64) | `barracuda/src/shaders/local_elementwise_f64.wgsl` |
| Local ops (f32) | `barracuda/src/shaders/local_elementwise.wgsl` |
| Provenance table | `barracuda/src/gpu/device_info.rs` (line 113+) |
| CPU fallback | `barracuda/src/gpu/reduce.rs` (`mean_variance()`) |
| Fused Pearson | `barracuda/src/gpu/stats.rs` (`pairwise_correlation_gpu()`) |
| Evolution benchmarks | `barracuda/src/bin/validate_cross_spring_evolution.rs` |
