# airSpring → ToadStool/BarraCuda: Absorption & Evolution Handoff

**Date**: March 2, 2026
**From**: airSpring v0.6.8 (Eastgate)
**To**: ToadStool / BarraCuda team
**ToadStool PIN**: S79 (`f97fc2ae`)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring has completed a deep debt audit and is at a stable quality plateau:
810 lib tests, 95.66% line coverage, zero clippy pedantic warnings, 69 experiments,
79 validation binaries, 25 Tier A GPU modules. This handoff documents what
airSpring has learned about BarraCuda usage, what patterns work, what could be
absorbed upstream, and what ToadStool evolution would most benefit airSpring's
next phase.

---

## 1. Current BarraCuda Usage Map

### Tier A — Integrated GPU Modules (25)

| airSpring Module | BarraCuda Primitive | Op/Shader | Status |
|-----------------|--------------------|----|---|
| `gpu::et0` | `batched_elementwise_f64` | op=0 | GPU-FIRST |
| `gpu::water_balance` | `batched_elementwise_f64` | op=1 | GPU-STEP |
| `gpu::sensor_calibration` | `batched_elementwise_f64` | op=5 | Integrated |
| `gpu::hargreaves` | `HargreavesBatchGpu` | dedicated | Integrated |
| `gpu::kc_climate` | `batched_elementwise_f64` | op=7 | Integrated |
| `gpu::dual_kc` | `batched_elementwise_f64` | op=8 | Integrated |
| `gpu::van_genuchten` | `batched_elementwise_f64` | op=9,10 | Integrated (S79) |
| `gpu::thornthwaite` | `batched_elementwise_f64` | op=11 | Integrated (S79) |
| `gpu::gdd` | `batched_elementwise_f64` | op=12 | Integrated (S79) |
| `gpu::pedotransfer` | `batched_elementwise_f64` | op=13 | Integrated (S79) |
| `gpu::jackknife` | `JackknifeMeanGpu` | dedicated | Integrated (S79) |
| `gpu::bootstrap` | `BootstrapMeanGpu` | dedicated | Integrated (S79) |
| `gpu::diversity` | `DiversityFusionGpu` | dedicated | Integrated (S79) |
| `gpu::kriging` | `kriging_f64::KrigingF64` | dedicated | Integrated |
| `gpu::reduce` | `fused_map_reduce_f64` | dedicated | GPU N≥1024 |
| `gpu::stream` | `moving_window_stats` | dedicated | Integrated |
| `gpu::richards` | `pde::richards::solve_richards` | dedicated | Integrated |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start` | dedicated | Integrated |
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` | dedicated | Integrated |
| `gpu::stats` | `linear_regression_f64` + `matrix_correlation_f64` | dedicated | Integrated |
| `gpu::seasonal_pipeline` | Chains ops 0→7→1→yield | fused | CPU + GpuPipelined + GpuFused |
| `gpu::atlas_stream` | `UnidirectionalPipeline` | streaming | CPU chained |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | CPU | Integrated |
| `eco::richards::inverse_vg_h` | `optimize::brent` | CPU | Integrated |
| `eco::diversity` | `stats::diversity` | CPU | Leaning |

### CPU-only Library Usage

| airSpring Module | BarraCuda Crate | Usage |
|-----------------|----------------|-------|
| `validation.rs` | `barracuda::validation::ValidationHarness` | All validation binaries |
| `tolerances.rs` | Re-exports from `barracuda::tolerances` | Shared named tolerances |
| `testutil/` | `barracuda::stats`, `barracuda::testutil` | Bootstrap CI, Pearson, Spearman |
| `nautilus.rs` | `bingocube_nautilus` | Evolutionary reservoir (AirSpringBrain) |
| `rpc.rs` | (standalone) | JSON-RPC 2.0 |
| `biomeos.rs` | (standalone) | Socket discovery, primal registration |

---

## 2. What airSpring Learned (Patterns for Upstream)

### Pattern: ValidationHarness for Binaries

All 79 airSpring binaries now use `ValidationHarness` for structured pass/fail:

```rust
let mut harness = ValidationHarness::new("validate_richards");
harness.check("VG retention sand θ_s", (theta - 0.43).abs() < 1e-6);
harness.check("Mass balance", mass_err.abs() < tolerances::MASS_BALANCE_ABS);
harness.finish(); // exit 0 if all pass, exit 1 with summary if any fail
```

This replaced 48+ raw `process::exit(1)` calls across 2 binaries alone. Every
check gets a name, and the summary output shows exactly which checks failed.

**Recommendation**: Consider promoting `ValidationHarness` to a first-class
barracuda primitive (it's already in `barracuda::validation` from neuralSpring).

### Pattern: Named Tolerances with Provenance

airSpring's `tolerances.rs` re-exports from barracuda and adds domain-specific:

```rust
pub const ET0_REL: f64 = 1e-3;      // FAO-56 tables ±0.01 mm/day
pub const MASS_BALANCE_ABS: f64 = 0.1; // Water balance closure ≤0.1 mm
pub const VG_THETA_ABS: f64 = 1e-6;  // Van Genuchten retention precision
```

Each tolerance is justified by its physical or paper provenance.

**Recommendation**: Establish a shared `barracuda::tolerances` module with
cross-spring named constants. airSpring has ~30, wetSpring has ~86.

### Pattern: Benchmark JSON Provenance

All benchmark JSONs now include:

```json
{
  "_provenance": {
    "script": "control/fao56/penman_monteith.py",
    "command": "python3 control/fao56/penman_monteith.py",
    "baseline_commit": "d3ecdc8",
    "date": "2026-03-02"
  },
  "_tolerance_justification": "..."
}
```

**Recommendation**: Document this as a standard for all Springs' benchmark data.

### Pattern: Sovereignty in Primal Code

airSpring's `airspring_primal.rs` was hardcoding external primal names
("toadstool", "nestgate") in error messages. Fixed to:

```rust
const PRIMAL_NAME: &str = "airspring";
// External primals referenced by capability, not name:
// "compute primal" instead of "toadstool"
// "data primal" instead of "nestgate"
```

**Recommendation**: Lint or CI check for hardcoded primal name strings in
all Springs' primal binaries.

---

## 3. What ToadStool Should Absorb

### Tier 1 — Direct Absorption (airSpring → upstream barracuda)

| Module | What | Why |
|--------|------|-----|
| `eco::tissue` | Tissue diversity profiling (Pielou→W, regime classification) | Generalizes diversity beyond ecology — immunological, microbial |
| `eco::cytokine` | CytokineBrain pattern (Nautilus reservoir for regime prediction) | Extends NautilusBrain pattern to new domains |
| Named tolerances | `tolerances.rs` domain constants | Shared ground truth across Springs |

### Tier 2 — Pattern Absorption (airSpring patterns → barracuda conventions)

| Pattern | What | Why |
|---------|------|-----|
| ValidationHarness everywhere | Standardized binary pass/fail | Already in barracuda; promote usage |
| Provenance JSON | `_provenance` + `_tolerance_justification` in benchmark data | Scientific reproducibility |
| `let...else` for GPU results | `let Ok(result) = gpu_call else { return cpu_fallback(); }` | Idiomatic fallback pattern |

### Tier 3 — Evolution Opportunities

| Opportunity | Description | Impact |
|-------------|-------------|--------|
| `BatchedStatefulF64` | Stateful step dispatch (water balance needs carry-forward state) | Would promote `gpu::water_balance` from Tier B to Tier A |
| `SeasonalPipelineF64` | Fused multi-op GPU pipeline (ET₀→Kc→WB→Stress in single dispatch) | Eliminates GPU round-trips for airSpring seasonal simulation |
| `ComputeDispatch` migration | airSpring's 14 ops → `ComputeDispatch` pattern | Aligns with S80's migration (95/250 done) |
| Batch Nelder-Mead GPU | N parallel optimizations via batched simplex shaders | S80 delivered this — airSpring should wire it for isotherm fitting |

---

## 4. What airSpring Needs Next from ToadStool

### Priority 1: ComputeDispatch alignment

S80 migrated 19 more ops to `ComputeDispatch`. airSpring uses ops 0-13 via
the older `batched_elementwise_f64` API. When `ComputeDispatch` reaches these
ops, airSpring should migrate for consistency.

### Priority 2: Batch Nelder-Mead

S80 delivered batched Nelder-Mead. airSpring's `gpu::isotherm` uses single-start
NM + multi-start wrapper. Migrating to batched NM would parallelize isotherm
fitting across soil samples (currently ~36.6K fits/sec CPU).

### Priority 3: BatchedEncoder for fused pipelines

S80's `BatchedEncoder` (fused multi-op GPU pipeline) could replace airSpring's
`SeasonalPipeline` manual op chaining, eliminating CPU round-trips between
ET₀ → Kc → WB → Stress stages.

### Priority 4: GpuDriverProfile workarounds

S80 added Taylor-series sin/cos preamble for NVK driver issues. airSpring
runs on Titan V via NVK/Mesa — these workarounds directly benefit us.

---

## 5. airSpring Quality State for Absorption Review

| Metric | Value |
|--------|-------|
| `cargo test --lib` | 810 passed, 0 failed |
| `cargo llvm-cov` | 95.66% lines, 96.33% functions |
| `cargo clippy --pedantic` | 0 warnings |
| `cargo fmt --check` | Clean |
| `cargo doc --no-deps` | Clean |
| `cargo-deny check` | Clean (AGPL-3.0-or-later) |
| `#![forbid(unsafe_code)]` | Yes |
| Files > 1000 lines | 0 |
| `unwrap()` in binaries | 0 (all `.expect("context")`) |
| `process::exit(1)` in binaries | 0 (all `ValidationHarness` or `Result`) |
| Mocks in production | 0 |
| Hardcoded primal names | 0 |
| TODOs / FIXMEs | 0 |

---

## 6. Cross-Spring Learnings for ToadStool Evolution

### Observation: f64 precision matters for soil physics

Van Genuchten θ(h) and K(h) involve `pow(x, n)` where n ranges 1.1–2.7 and h
ranges 0–15,000 cm. The f64-canonical shader architecture (S67-S68) is essential.
Df64 provides sufficient precision for consumer GPUs, but f32 would fail
catastrophically on clay soils (n≈1.09, K spans 12 orders of magnitude).

### Observation: domain-specific ops plateau quickly

airSpring needed 14 `batched_elementwise_f64` ops total. Each op is 5-15 lines
of math. The ops are domain-specific but trivial once the dispatch machinery exists.
This suggests a stable steady state for new Springs — they'll need ~10-20 ops each,
not hundreds.

### Observation: GPU uncertainty stack is transformative

The jackknife/bootstrap/diversity GPU shaders (S71+S79) transformed airSpring's
ability to quantify prediction confidence. Previously CPU-only at ~10ms per
jackknife; now GPU-capable at ~700µs. This enables uncertainty bands on every
prediction in the atlas pipeline.

### Observation: Nautilus (bingoCube) integrates cleanly

airSpring's `AirSpringBrain` (3-head Nautilus reservoir for ET₀/soil/crop
prediction) and `MonitoredAtlasStream` (DriftMonitor for regime change detection)
integrated cleanly via the bingocube-nautilus library. S80's absorption of
Nautilus into barracuda proper means future Springs won't need the external
dependency path.

---

## 7. Handoff Chain

| Document | Scope |
|----------|-------|
| `AIRSPRING_V061_TOADSTOOL_S79_SYNC_HANDOFF_MAR02_2026.md` | S79 GPU rewire |
| `AIRSPRING_V062_NAUTILUS_BRAIN_DRIFT_INTEGRATION_MAR02_2026.md` | Nautilus + DriftMonitor |
| `AIRSPRING_V063_DEEP_DEBT_AUDIT_HANDOFF_MAR02_2026.md` | Quality hardening |
| **This document** | ToadStool absorption recommendations |

---

AGPL-3.0-or-later
