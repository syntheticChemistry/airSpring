# airSpring V045 — Full Dispatch + biomeOS Graph + ToadStool Absorption Handoff

**Date**: March 2, 2026
**From**: airSpring (ecology sciences primal)
**To**: ToadStool/BarraCuda team, biomeOS, NUCLEUS ecosystem
**airSpring Version**: 0.6.0
**ToadStool Pin**: S71 (HEAD `8dc01a37`)
**License**: AGPL-3.0-or-later
**Supersedes**: V044 (ToadStool Nautilus Evolution)

---

## Executive Summary

- **30 science capabilities** registered in NUCLEUS (up from 16): 26 ecology, 2 science, 1 compute, 1 data
- **Exp 064: Full Dispatch** (51/51 PASS) — CPU/GPU parity validated across all 21 science domains, 5 GPU dispatch domains, batch scaling to 10K, ToadStool absorption audit (15 Tier A), mixed-backend seasonal pipeline
- **Exp 065: biomeOS Graph** (35/35 PASS) — `airspring_deploy.toml` graph topology validated, 30-capability registry, 182-day offline ecology pipeline (mass balance 6.4e-13 mm), GPU seasonal parity
- **673 lib tests**, 58 forge tests, 75 binaries, 53/53 cross-spring benchmarks — all green
- **Deployment graph**: `metalForge/deploy/airspring_deploy.toml` — BearDog → Songbird → ToadStool → airSpring
- **Songbird transport tier**: `HttpTransport` trait enables sovereign HTTPS via BearDog crypto delegation

---

## Part 1: What airSpring Needs from ToadStool

### Absorbed Primitives (15 Tier A — working, validated)

| airSpring GPU Module | BarraCuda Primitive | Op | Status |
|---------------------|--------------------|----|--------|
| `gpu::et0` | `BatchedElementwiseF64` | 0 | GPU-FIRST, CPU fallback |
| `gpu::water_balance` | `BatchedElementwiseF64` | 1 | GPU-STEP |
| `gpu::sensor_calibration` | `BatchedElementwiseF64` | 5 | GPU-FIRST (S70+) |
| `gpu::hargreaves` | `BatchedElementwiseF64` | 6 | GPU-FIRST (S70+) |
| `gpu::kc_climate` | `BatchedElementwiseF64` | 7 | GPU-FIRST (S70+) |
| `gpu::dual_kc` | `BatchedElementwiseF64` | 8 | GPU-FIRST (S70+) |
| `gpu::kriging` | `KrigingF64` | — | Integrated |
| `gpu::reduce` | `FusedMapReduceF64` | — | GPU N≥1024 |
| `gpu::stream` | `MovingWindowStats` | — | Wired |
| `gpu::richards` | `pde::richards` | — | Wired |
| `gpu::isotherm` | `optimize::nelder_mead` | — | Wired |
| `eco::correction` | `linalg::ridge` | — | Wired |
| `eco::richards` | `optimize::brent` | — | Wired |
| `eco::diversity` | `stats::diversity` | — | Leaning |
| `validation` | `ValidationHarness` | — | Leaning |

### Remaining Gaps (Tier B — wired, pending full shader)

| Gap | What's Needed | Priority |
|-----|---------------|----------|
| `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` — Monte Carlo ET₀ uncertainty | Medium |
| `gpu::seasonal_pipeline` | `BatchedStatefulF64` — multi-day state carry on GPU | Low |
| `gpu::atlas_stream` | `UnidirectionalPipeline` integration | Low |

### What airSpring Gave Back

| Contribution | ToadStool Issue | Status |
|-------------|----------------|--------|
| Richards PDE solver | S40 absorption | Absorbed |
| Stats metrics re-exports | S64 absorption | Absorbed |
| All metalForge modules | S64 + S66 absorption | Absorbed (6/6) |
| `pow_f64` fractional exponent fix | TS-001 | Resolved |
| `BatchedElementwiseF64` op buffer fix | TS-002 | Resolved |
| `acos` precision boundary fix | TS-003 | Resolved |
| Reduce buffer N≥1024 fix | TS-004 | Resolved |

---

## Part 2: Dispatch Experiment Results (Exp 064)

### CPU Science Baseline (21 methods)

All 21 validated methods produce correct outputs: FAO-56 PM ET₀ (5.97 mm/day),
Hargreaves, Priestley-Taylor, Makkink, Turc, Hamon, Blaney-Criddle, water balance
depletion, yield ratio (Stewart), SCS-CN runoff, Green-Ampt infiltration, Topp TDR,
Saxton-Rawls pedotransfer, GDD, Shannon/Simpson diversity, Bray-Curtis, Anderson
coupling (d_eff=2.10), Thornthwaite monthly ET, Richards PDE convergence, sensor
calibration (SoilWatch 10).

### GPU Dispatch Parity (5 domains)

| Domain | CPU Value | GPU Value | Δ | Tolerance |
|--------|-----------|-----------|---|-----------|
| FAO-56 ET₀ | 5.497 mm/d | 5.497 mm/d | 4.7e-4 | 0.01 |
| Hargreaves | varies | varies | 4.3e-3 | 0.01 |
| Sensor cal | 0.0495 | 0.0495 | 6.9e-18 | 1e-10 |
| Water balance Dr | 43.4 mm | 43.4 mm | 0.0 | 1.0 |
| Reduce mean (N=2048) | 10.235 | 10.235 | 0.0 | 0.1 |

### Batch Scaling

| N | Time (ms) | Identical to N=1 |
|---|-----------|-----------------|
| 10 | 0.001 | yes |
| 100 | 0.01 | yes |
| 1,000 | 0.11 | yes |
| 10,000 | 0.97 | yes |

### ToadStool Absorption Audit

- 15 Tier A (integrated), 7 Tier B (wired), 1 Tier C (pending)
- 95.7% primitive coverage (22/23 gaps have ToadStool primitives)
- All key domains tracked: ET₀, water balance, Richards, kriging, reduce

---

## Part 3: biomeOS Deployment Graph

`metalForge/deploy/airspring_deploy.toml` defines:

```
BearDog (crypto) → Songbird (network) → ToadStool (compute) → airSpring (ecology)
```

30 capabilities registered across 4 domains:
- `ecology.*` (26): et0 (9 methods), water_balance (2), dual_kc, crop (3), soil (3), hydrology (3), diversity (3), anderson, richards
- `science.*` (2): health, version
- `compute.offload` (1): GPU workload routing to ToadStool
- `data.weather` (1): routing through NestGate cache

---

## Part 4: What ToadStool Should Absorb Next

### Priority 1: `mc_et0_propagate_f64.wgsl`

Monte Carlo ET₀ uncertainty propagation. airSpring has the CPU path validated;
the GPU shader exists in groundSpring's lineage (Box-Muller + xoshiro). This
would complete the uncertainty budget for precision irrigation scheduling.

### Priority 2: `BatchedStatefulF64`

Multi-day state carry for `SeasonalPipeline::GpuFused`. Currently CPU handles
inter-day state; a stateful GPU primitive would eliminate the CPU round-trip
for season simulations. This is the single biggest performance opportunity.

### Priority 3: Upstream `fao56_et0` batch

ToadStool S71 has scalar `fao56_et0`. Promoting this to a batched dispatch
(like `HargreavesBatchGpu`) would let airSpring lean entirely on upstream
for ET₀ — the most-called science method.

---

## Part 5: Quality State

| Check | Result |
|-------|--------|
| `cargo test` (barracuda) | 673 passed, 0 failures |
| `cargo test` (forge) | 58 passed, 0 failures |
| `cargo clippy --all` | 0 warnings (pedantic + nursery) |
| `cargo fmt --check` | Clean |
| `validate_dispatch_experiment` | 51/51 PASS |
| `validate_biome_graph` | 35/35 PASS |
| `bench_cross_spring_evolution` | 53/53 PASS |

### Reproduction

```bash
cd airSpring/barracuda
cargo build --release
./target/release/validate_dispatch_experiment  # 51/51
./target/release/validate_biome_graph          # 35/35
cargo test                                     # 673/673
cd ../metalForge/forge && cargo test           # 58/58
```

---

Unidirectional handoff — no response expected.
