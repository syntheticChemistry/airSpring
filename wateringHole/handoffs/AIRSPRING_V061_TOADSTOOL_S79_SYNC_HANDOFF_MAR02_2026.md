# airSpring V0.6.1 — ToadStool S79 Complete Rewiring Handoff

**Date**: March 2, 2026
**From**: airSpring (Eastgate)
**To**: ToadStool / ecoPrimals ecosystem
**ToadStool HEAD**: S79

---

## Summary

airSpring v0.6.1 completes a full rewiring to modern ToadStool S79 APIs. All 14
`batched_elementwise_f64` ops (0-13) are wired with GPU orchestrators. The GPU
uncertainty stack (jackknife, bootstrap, diversity fusion) is wired via dedicated
science shaders. `pollster` is eliminated in favour of ToadStool's `test_pool`
pattern. Cross-spring shader provenance is tracked for every GPU path.

## Cross-Spring Shader Evolution Map

| Spring | Domain | Contributions → airSpring |
|--------|--------|--------------------------|
| **hotSpring** | Nuclear/precision physics | `math_f64.wgsl` (pow, exp, log, sin, cos, acos), `df64_core.wgsl` (FMA double-float), `df64_transcendentals.wgsl`, `crank_nicolson_f64.wgsl`, `norm_ppf.wgsl` |
| **wetSpring** | Bio/environmental | `kriging_f64.wgsl`, `fused_map_reduce_f64.wgsl`, `moving_window_stats.wgsl`, `diversity_fusion_f64.wgsl` (Shannon/Simpson/evenness GPU), diversity CPU indices |
| **neuralSpring** | ML/optimization | `nelder_mead.wgsl`, `ValidationHarness`, `stats_f64` (OLS regression, correlation), batch orchestrator pattern |
| **airSpring** | Precision agriculture | ops 0-1, 5-8 domain equations, Richards PDE, regression, hydrology, Blaney-Criddle, SCS-CN, Green-Ampt |
| **groundSpring** | Uncertainty/stats | `jackknife_mean_f64.wgsl`, `bootstrap_mean_f64.wgsl`, `mc_et0_propagate_f64.wgsl`, MC methodology |

### Notable Cross-Spring Evolution Paths

1. **hotSpring precision → airSpring VG curves**: `pow_f64`/`exp_f64` from lattice QCD (S54) enables Van Genuchten retention curves (op=9,10) on GPU with f64 precision
2. **wetSpring diversity → airSpring soil microbiome**: Shannon/Simpson/Bray-Curtis indices (S28) → GPU fusion shader (S70) → airSpring cover crop and 16S microbiome analysis
3. **neuralSpring optimizers → airSpring isotherm fitting**: Nelder-Mead/BFGS (S52) → batch isotherm fitting for biochar Langmuir/Freundlich models
4. **groundSpring uncertainty → airSpring yield CI**: Bootstrap/jackknife methodology → GPU shaders (S71) → airSpring ET₀ and yield prediction uncertainty bands
5. **All springs → universal precision**: f64 canonical WGSL (S68) means every airSpring shader runs at native f64 on Titan V, Df64 on consumer GPUs, f32 fallback — no code changes needed

## New GPU Orchestrators

### Ops 9-13 (S79 — `batched_elementwise_f64`)

| Op | Module | Stride | Description | CPU Provenance |
|----|--------|--------|-------------|----------------|
| 9 | `gpu::van_genuchten` | 5 | VG θ(h) retention curve | `eco::van_genuchten::theta()` |
| 10 | `gpu::van_genuchten` | 7 | VG K(h) hydraulic conductivity | `eco::van_genuchten::conductivity()` |
| 11 | `gpu::thornthwaite` | 5 | Thornthwaite monthly ET₀ | `eco::thornthwaite::thornthwaite_monthly_et0()` |
| 12 | `gpu::gdd` | 1+aux | Growing Degree Days | `eco::crop::gdd_avg()` |
| 13 | `gpu::pedotransfer` | 7 | Horner polynomial evaluation | `eco::soil_moisture::saxton_rawls()` |

### Science Shaders (S71 — dedicated GPU dispatch)

| Module | Shader | Origin | Description |
|--------|--------|--------|-------------|
| `gpu::jackknife` | `jackknife_mean_f64.wgsl` | groundSpring→S71 | Leave-one-out variance estimation |
| `gpu::bootstrap` | `bootstrap_mean_f64.wgsl` | groundSpring→S71 | Non-parametric bootstrap CI |
| `gpu::diversity` | `diversity_fusion_f64.wgsl` | wetSpring→S70 | Shannon+Simpson+evenness in one dispatch |

## Modernization

### pollster → test_pool Migration

All 13+ files using `pollster::block_on` migrated to `barracuda::device::test_pool::tokio_block_on`. The `pollster` crate is removed from Cargo.toml. This aligns with ToadStool S74's removal of pollster from barracuda.

### Provenance Tracking

`device_info::PROVENANCE` extended from 17 to 22 entries, covering all GPU modules with cross-spring lineage:
- Origin spring (who created it)
- Scientific domain
- Which springs evolved it further
- How airSpring uses it

## ToadStool Evolution Absorbed (S71→S79)

| Session | Key Changes | airSpring Impact |
|---------|-------------|------------------|
| S78 | libc→rustix, AFIT migration, wildcard narrowing | ecoBin pure-Rust path closer |
| S79 | ESN v2 MultiHeadEsn, spectral extensions, ops 9-13 | New GPU ops for ecology |

## Tier Counts (v0.6.1 Complete)

| Tier | Count | Delta from v0.5.9 |
|------|-------|-------------------|
| A (Integrated) | 25 | +8 (ops 9-13, jackknife, bootstrap, diversity) |
| B (Needs wiring) | 3+6 orchestrators | -8 |
| C (Needs primitive) | 1 | unchanged |

## Validation Status

| Gate | Status | Value |
|------|--------|-------|
| `cargo fmt --check` | PASS | both crates |
| `cargo clippy -D warnings -W clippy::pedantic` | PASS | both crates |
| `cargo doc --no-deps` | PASS | both crates |
| `cargo test --lib` | PASS | 737 tests |
| `cargo llvm-cov --lib --fail-under-lines 90` | PASS | 94.15% |
| `cargo check` (forge) | PASS | — |
| Cross-spring evolution benchmark | PASS | **110/110** |

### Benchmark Sections (all PASS)

| Section | Checks | Timing |
|---------|--------|--------|
| hotSpring Precision | 8 | ~31µs |
| wetSpring Bio | 13 | ~71µs |
| neuralSpring Optimizers | 8 | ~230µs |
| airSpring Rewired | 4 | ~32µs |
| groundSpring Uncertainty | 8 | ~10ms |
| Tridiagonal Solver | 3 | ~22ms |
| S71 Upstream Evolution | 9 | ~357µs |
| **S79: Ops 9-13** | **33** | ~19µs |
| **S79: GPU Uncertainty** | **24** | ~686µs |

## Files Changed

### New files
- `barracuda/src/gpu/van_genuchten.rs` (274 lines)
- `barracuda/src/gpu/thornthwaite.rs` (196 lines)
- `barracuda/src/gpu/gdd.rs` (133 lines)
- `barracuda/src/gpu/pedotransfer.rs` (~200 lines)
- `barracuda/src/gpu/jackknife.rs` (~200 lines)
- `barracuda/src/gpu/bootstrap.rs` (~200 lines)
- `barracuda/src/gpu/diversity.rs` (~200 lines)

### Updated files
- `barracuda/src/gpu/mod.rs` — 7 new modules + doc table
- `barracuda/src/gpu/evolution_gaps.rs` — S78-S79 docs, 4 new Tier A entries
- `barracuda/src/gpu/device_info.rs` — 5 new provenance entries, pollster→test_pool
- `barracuda/src/bin/bench_cross_spring_evolution.rs` — 2 new benchmark sections (S79)
- `barracuda/Cargo.toml` — v0.6.0→v0.6.1, pollster removed
- 13 GPU modules — pollster→test_pool migration
- 2 validation binaries — pollster→test_pool migration

## Next Evolution

- Wire `HargreavesBatchGpu` (dedicated science shader, computes Ra internally)
- Wire `SeasonalPipelineF64` (fused ET₀→Kc→WB→Stress in single dispatch)
- Wire `McEt0PropagateGpu` (GPU Monte Carlo uncertainty propagation)
- Evaluate `ComputeDispatch` migration for airSpring GPU modules
- Wire `DiversityFusionGpu` for multi-sample GPU diversity profiling

---

AGPL-3.0-or-later
