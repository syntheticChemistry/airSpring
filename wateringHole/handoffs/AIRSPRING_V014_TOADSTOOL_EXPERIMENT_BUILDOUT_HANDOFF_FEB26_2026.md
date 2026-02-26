# airSpring → ToadStool Handoff V014: Experiment Buildout + Evolution Learnings

**Date**: February 26, 2026
**From**: airSpring v0.4.5 — 16 experiments, 474 Python + 719 Rust checks, 21 binaries
**To**: ToadStool / BarraCuda core team
**ToadStool PIN**: `17932267` (S65 — 774 WGSL shaders)
**Supersedes**: V013 (absorption candidates) — V013 remains active for pending absorption items

---

## Summary

airSpring v0.4.5 adds 3 new paper reproductions, each with full Python→Rust CPU
pipeline. These demonstrate the composability of existing BarraCuda primitives for
real agricultural science workloads and identify what's needed for GPU promotion.

---

## New Experiments (v0.4.5)

### Exp 014: Irrigation Scheduling Optimization

**Paper**: Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148
**Pipeline**: `daily_et0` → `kc_schedule` → `water_balance::simulate_season` → `yield_response::yield_ratio_single` → WUE

| Strategy | Irrigation (mm) | Yield (Ya/Ym) | Stress Days | WUE (kg/ha/mm) |
|----------|:-:|:-:|:-:|:-:|
| Rainfed | 0 | 0.411 | 85 | 20.5 |
| MAD 50% | 350 | 1.000 | 0 | 20.3 |
| MAD 60% | 325 | 0.970 | 33 | 20.6 |
| MAD 70% | 225 | 0.835 | 75 | 21.6 |
| Growth-stage | 250 | 0.909 | 43 | **22.3** |

**ToadStool relevance**: This is the "Penny Irrigation" core pipeline. For GPU
promotion, needs `BatchedWaterBalance` (op=1) chained with `BatchedElementwise`
yield computation (op=new). The growth-stage strategy's conditional trigger
(`d ∈ [70,120] && dr > 0.55×TAW`) requires shader-side branching or pre-computed
mask arrays.

**Checks**: Python 25/25, Rust 28/28. Mass balance closure < 1e-13 mm.

### Exp 016: Lysimeter ET Direct Measurement

**Paper**: Dong & Hansen (2023) Smart Ag Tech 4:100147
**Pipeline**: load cell mass → temperature compensation → quality filter → ET (mm)

- Mass-to-ET: 1 kg / 1 m² = 1 mm (standard lysimeter simplification)
- Temp compensation: M_corr = M_raw − α(T − T_ref)/1000, α = 2.5 g/°C
- Calibration R² = 0.9999 (known mass additions)
- Diurnal pattern: sinusoidal, night ≈ 0, peak at solar noon

**ToadStool relevance**: Lightweight module — pure arithmetic. Good candidate for
`BatchedElementwiseF64` with a new op code for mass→ET conversion with thermal
correction. Alternatively absorb into `barracuda::eco` as a utility.

**Checks**: Python 26/26, Rust 25/25.

### Exp 017: ET₀ Sensitivity Analysis

**Paper**: Gong et al. (2006) Ag Water Mgmt 86:57-63 (methodology)
**Pipeline**: `daily_et0` × 2 perturbations × 6 variables × 3 climatic zones

| Variable | Uccle |S| | Phoenix |S| | Manaus |S| |
|----------|:-:|:-:|:-:|
| Wind u₂ | 0.144 | **1.912** | 0.253 |
| Solar Rs | 0.107 | 0.116 | 0.162 |
| T_max | 0.081 | 0.098 | 0.058 |
| T_min | 0.039 | 0.017 | 0.032 |
| RH_min | 0.020 | 0.005 | 0.009 |
| RH_max | 0.011 | 0.002 | 0.006 |

**ToadStool relevance**: Demonstrates embarrassingly parallel ET₀ perturbation —
ideal for `BatchedEt0` at scale. Each OAT perturbation is an independent ET₀
computation. A 6-variable × 2-direction × N-site batch = 12N GPU dispatches,
perfectly suited for existing `batched_elementwise_f64.wgsl` with op=0.

**Checks**: Python 23/23, Rust 23/23.

---

## BarraCuda Primitive Usage Report (v0.4.5)

### Actively Used Primitives (11 Tier A)

| Primitive | airSpring Usage | First Wired |
|-----------|----------------|:-:|
| `ops::batched_elementwise_f64` | ET₀, water balance, sensitivity batch | v0.3.0 |
| `ops::kriging_f64` | Soil moisture spatial interpolation | v0.3.6 |
| `ops::fused_map_reduce_f64` | Seasonal statistics | v0.3.6 |
| `ops::moving_window_stats` | IoT stream smoothing | v0.3.6 |
| `pde::richards::solve_richards` | Unsaturated flow, CW2D | v0.4.0 |
| `optimize::nelder_mead` | Isotherm fitting | v0.4.1 |
| `optimize::multi_start_nelder_mead` | Multi-start isotherm | v0.4.1 |
| `optimize::brent` | VG θ→h inversion | v0.4.4 |
| `stats::normal::norm_ppf` | MC ET₀ parametric CI | v0.4.4 |
| `linalg::ridge::ridge_regression` | Sensor correction | v0.3.10 |
| `validation::ValidationHarness` | All validation binaries | v0.3.6 |

### Absorption Candidates (from V013, still pending)

| metalForge Module | Tests | Target barracuda Module |
|-------------------|:-----:|------------------------|
| `forge::regression` | 11 | `barracuda::linalg` or `stats::regression` |
| `forge::hydrology` | 9 | `barracuda::eco::hydrology` (TAW/RAW/Ks) |
| `forge::moving_window_f64` | 12 | Already overlaps `ops::moving_window_stats` |
| `forge::isotherm` | 14 | `barracuda::eco::isotherm` |

### New Absorption Candidates (from v0.4.5 experiments)

| Module | Functions | Rationale |
|--------|-----------|-----------|
| Lysimeter ET | `mass_to_et_mm`, `compensate_temperature`, `is_valid_reading` | Pure arithmetic, good for `eco` utility or new `BatchedElementwise` op |
| Scheduling trigger | MAD-based + growth-stage irrigation triggers | Common agricultural pattern, could be a `scheduling` module |
| Sensitivity OAT | `oat_sensitivity`, `full_sensitivity_analysis` | Reusable for any model, not just ET₀ |

---

## Cross-Spring Learnings for ToadStool

### 1. Composability of Existing Primitives

Exp 014 demonstrates that `daily_et0` → `water_balance` → `yield_response` chains
together cleanly using only existing primitives. No new math was needed — just
plumbing. This validates the "modular pipeline" architecture.

### 2. Conditional Irrigation on GPU

Growth-stage scheduling requires `if day ∈ range && depletion > threshold` logic.
For GPU, two approaches:
- **Pre-computed mask array**: CPU generates a `[bool; N]` mask, GPU applies it
- **Shader-side conditional**: Add a `schedule_mask` uniform buffer to the water
  balance shader

Recommendation: Pre-computed mask is simpler and keeps shaders stateless.

### 3. Sensitivity Analysis = Embarrassingly Parallel ET₀

OAT sensitivity is 12 × N independent ET₀ computations. This is exactly the
workload `BatchedElementwiseF64` was designed for. For a 1000-site analysis with
6 variables × 2 directions = 12,000 GPU dispatches, which ToadStool's unidirectional
streaming massively reduces round-trips for.

### 4. Lysimeter as Ground Truth for ET₀ Calibration

The lysimeter module provides direct ET measurement (mass change) that can calibrate
equation-based ET₀ (Penman-Monteith). This is the "control experiment for the
control experiment" — measuring ET directly rather than computing it.

---

## P0 Blocker (unchanged from V013)

`BatchedElementwiseF64` GPU dispatch panics with:
```
dispatch_workgroups: Dispatch dimension 0 (X) is 0. At least one dimension must be > 0
```

Root cause: `ceil(n / workgroup_size)` returns 0 when `n == 0` or integer division
rounds down. This is a ToadStool dispatch issue, not airSpring-specific.

**Impact**: All GPU-FIRST dispatches fall back to CPU. CPU benchmarks validate the
math is correct; GPU validation blocked until this is resolved.

---

## Quality Gates (v0.4.5)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | **Clean** |
| `cargo clippy --workspace -- -D warnings` | **0 warnings** |
| `cargo test --workspace` | **464 lib + 126 integration + 53 forge** = 643 cargo tests |
| Validation binaries | **21 binaries**, all PASS |
| Python controls | **474/474** PASS (16 experiments) |
| Cross-validation | **75/75** match (tol=1e-5) |
| Coverage | **96.81%** lines (lib) |
| `unsafe` blocks | **0** |
| `unwrap()` in lib | **0** |

---

## Open Items

| ID | Item | Priority | Status |
|----|------|:--------:|--------|
| N1 | `BatchedElementwiseF64` GPU dispatch panic (P0) | P0 | **Blocked** |
| N2 | Named VG soil type constants (from V013) | P2 | Open |
| N3 | `spearman_correlation` re-export (from V013) | P3 | Open |
| N4 | `forge::regression` → `barracuda::linalg` absorption | P2 | Open |
| N5 | `forge::hydrology` → `barracuda::eco` absorption | P2 | Open |
| N6 | Lysimeter `mass_to_et` utility for `barracuda::eco` | P3 | New |
| N7 | OAT sensitivity utility for `barracuda::stats` | P3 | New |
| N8 | Scheduling irrigation trigger API for `barracuda::eco` | P3 | New |

---

*airSpring v0.4.5 — 16 experiments, 474/474 Python, 719 Rust checks, 21 binaries.
Pure Rust + BarraCuda. AGPL-3.0-or-later.*
