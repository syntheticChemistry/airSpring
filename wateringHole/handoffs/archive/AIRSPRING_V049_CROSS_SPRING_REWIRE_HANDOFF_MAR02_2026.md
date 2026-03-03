# airSpring V049 — Cross-Spring Rewire Handoff

**Date**: March 2, 2026
**From**: airSpring v0.6.6 → ToadStool/BarraCuda team
**ToadStool PIN**: S86 HEAD (`2fee1969`)
**Status**: 68/68 Exp 073 PASS | 815 lib tests | 83 binaries | 73 experiments

---

## Executive Summary

airSpring has completed full rewiring to modern ToadStool S86 GPU primitives,
replacing local CPU-only implementations with upstream GPU-accelerated solvers.
This handoff documents what was wired, validated, and benchmarked — and maps the
cross-spring shader provenance showing how each spring's contributions flow
through the ecosystem.

## What Was Rewired

### 1. BrentGpu VG Inverse (gpu::van_genuchten)

- **Before**: CPU-only `inverse_van_genuchten_h` (bisection in `eco::van_genuchten`)
- **After**: `BatchedVanGenuchten::compute_inverse_gpu` → `BrentGpu::solve_vg_inverse`
- **Shader**: `brent_f64.wgsl` (hotSpring precision: `pow_f64`, `exp_f64`)
- **Validation**: 8 reference points, max error < 5 cm GPU↔CPU, batch scaling to 10K
- **Throughput**: CPU 1.0M/s, GPU 153K/s (dispatch-bound at small N; GPU wins at scale)

### 2. RichardsGpu Picard Solver (gpu::richards)

- **Before**: CPU-only `solve_upstream` (Crank-Nicolson via `pde::richards`)
- **After**: `BatchedRichards::solve_gpu` → `RichardsGpu::solve`
- **Shader**: `richards_picard_f64.wgsl` — 3 GPU passes per Picard iteration:
  1. `compute_hydraulics` (K, C, θ per node — parallel)
  2. `assemble_tridiag` (Crank-Nicolson system — parallel)
  3. `thomas_solve` (Thomas tridiagonal — sequential)
- **Cross-spring math**: hotSpring precision (`pow_f64`, `exp_f64`) + neuralSpring
  tridiagonal solver (`cyclic_reduction_f64.wgsl`)
- **Validation**: Sand, Silt Loam, Clay — all θ in physical range, time steps complete
- **Timing**: CPU 117µs vs GPU 463ms (small grid; GPU amortises at large N)

### 3. StatefulPipeline + BatchedStatefulF64

- **StatefulPipeline<WaterBalanceState>** (S80): passthrough validated, ready for stages
- **BatchedStatefulF64** (S83): type available, GPU-resident state carry for pipelines
- Confirms day-over-day water balance can chain via upstream abstractions

### 4. Cross-Spring Provenance (Exp 073)

Full validation of each spring's contribution to the airSpring pipeline:

| Spring | Validated Primitive | Check |
|--------|-------------------|-------|
| hotSpring | `erf(1)`, `Γ(5)`, `anderson_4d(L=3)` | ✓ |
| wetSpring | `Shannon(uniform(5))` | ✓ |
| neuralSpring | `brent(√2)`, `lbfgs_numerical(Rosenbrock)` | ✓ |
| groundSpring | `bootstrap_ci(mean, N=5)` | ✓ |
| airSpring | `daily_et0(FAO-56 PM)` | ✓ |

### 5. Hydrology CPU↔GPU Parity

Validated 9 upstream hydrology functions against local airSpring implementations:
`fao56_et0`, `hargreaves_et0`, `soil_water_balance`, `crop_coefficient`,
`thornthwaite_et0`, `makkink_et0`, `turc_et0`, `hamon_et0`, plus local PM ET₀.

## Cross-Spring Shader Provenance

```text
hotSpring ──── precision math (df64, pow_f64, exp_f64, erf, gamma)
    │              └── Used by: ALL springs, ALL shaders
    ├── Lanczos eigensolve ── wetSpring (Anderson QS), airSpring (Anderson coupling)
    ├── CrankNicolson1D f64 ── airSpring (Richards PDE linearised baseline)
    └── anderson_4d (S83) ── wetSpring (QS), airSpring (soil disorder)

wetSpring ──── bio diversity (Shannon, Simpson, Bray-Curtis, Hill)
    │              └── Used by: airSpring (soil biodiversity, Paper 12 tissue)
    ├── kriging_f64 ── airSpring (soil moisture interpolation)
    └── moving_window_f64 ── airSpring (IoT stream smoothing)

neuralSpring ── Nelder-Mead, BFGS, BatchedBisection
    │              └── Used by: airSpring (isotherm fitting, VG calibration)
    ├── ValidationHarness ── ALL springs, ALL validation binaries
    └── BatchedNelderMeadGpu (S80) ── airSpring (batch isotherm)

airSpring ──── FAO-56 ET₀ (op=0), WB (op=1), ops 5-13
    │              └── Used by: ToadStool (seasonal pipeline shader)
    ├── StatefulPipeline (S80) ── day-over-day water balance
    ├── BatchedStatefulF64 (S83) ── GPU-resident state carry
    ├── BrentGpu (S83) ── VG inverse θ→h on GPU
    ├── RichardsGpu (S83) ── GPU Picard solver
    └── SeasonalPipelineF64 ── fused ET₀→Kc→WB→stress

groundSpring ── MC uncertainty propagation
    ├── bootstrap/jackknife GPU ── airSpring (uncertainty stack)
    └── batched_multinomial ── airSpring (stochastic soil sampling)
```

## What airSpring Still Maintains Locally

- `eco::evapotranspiration` (local FAO-56 PM, Priestley-Taylor, 8 methods)
- `eco::richards` (CPU implicit Euler baseline for cross-validation)
- `eco::van_genuchten` (CPU forward/inverse for reference)
- `eco::water_balance`, `eco::dual_kc`, `eco::crop` (domain structs)
- `eco::cytokine`, `eco::tissue`, `nautilus` (bio/reservoir computing)
- `gpu::seasonal_pipeline` (multi-field orchestrator)
- `metalForge/` (cross-system dispatch: GPU→NPU→CPU)

## Next Evolution Targets

1. **Wire `BatchedStatefulF64` into `SeasonalPipeline`** — GPU-resident day-over-day
   state (soil moisture, snow) without CPU round-trip per time step.
2. **Fused GPU Richards → Water Balance** — chain `RichardsGpu` output directly
   into `BatchedWaterBalance` via `BatchedEncoder` single-submit.
3. **`SeasonalPipelineF64` adoption** — move from per-stage dispatch to the fused
   upstream seasonal GPU pipeline.
4. **Multi-GPU scaling** — use upstream `multi_gpu` module for field-parallel execution.

## Metrics Summary

| Metric | Value |
|--------|-------|
| airSpring version | 0.6.6 |
| ToadStool PIN | S86 HEAD |
| Lib tests | 815 |
| Binaries | 83 |
| Experiments | 73 |
| Exp 073 checks | 68/68 PASS |
| GPU primitives wired | BrentGpu, RichardsGpu |
| Springs validated | 5/5 (hot, wet, neural, ground, air) |
