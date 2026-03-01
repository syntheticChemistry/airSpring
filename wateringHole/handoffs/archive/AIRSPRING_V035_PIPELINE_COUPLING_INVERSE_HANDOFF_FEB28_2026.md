# airSpring V035 — Pipeline Coupling, Inverse Problems, and Season-Scale Audit

**Date**: February 28, 2026
**From**: airSpring V035 (v0.5.4)
**To**: ToadStool / BarraCuda team
**Status**: 54 experiments, 1237 Python + 618 lib tests + 59 binaries + 30/30 cross-spring benchmarks
**Supersedes**: V034 (experiment buildout + deep debt resolution)
**License**: AGPL-3.0-or-later

---

## Executive Summary

- **3 new experiments** coupling validated modules into integrated pipelines:
  Coupled Runoff-Infiltration (Exp 052), VG Inverse Estimation (Exp 053),
  Full-Season Water Budget Audit (Exp 054)
- **410 new validation checks** (292+84+34) across Python and Rust
- **3 new validation binaries**: `validate_coupled_runoff`, `validate_vg_inverse`,
  `validate_season_wb`
- **Module coupling validated**: SCS-CN + Green-Ampt compose with mass conservation;
  VG inverse via `barracuda::optimize::brent` converges for all practical Se;
  full FAO-56 chain (weather → ET₀ → Kc → WB → yield) audited end-to-end
- **Zero clippy warnings**, zero `unsafe`, zero production `unwrap()`, clippy pedantic clean
- Binary count: 56 → 59. Experiment count: 51 → 54.

---

## Part 1: New Experiments

### Exp 052: SCS-CN + Green-Ampt Coupled Rainfall Partitioning

**Purpose**: Validate that `eco::runoff` (SCS-CN) and `eco::infiltration` (Green-Ampt)
compose correctly for event-scale rainfall partitioning: P → Q (runoff) → net rain →
F (infiltration) → surface storage, with mass balance closure.

| Component | Detail |
|-----------|--------|
| Python | 292/292 PASS — 48 storm×soil×land-use, 80 conservation sweep, 4 monotonicity, 160 sensitivity |
| Rust | `validate_coupled_runoff` — 292/292 PASS, mass balance ≤ 1e-8 mm |
| Modules | `eco::runoff::scs_cn_runoff_standard` + `eco::infiltration::cumulative_infiltration` |
| Data | USDA NEH-4 CN tables + Rawls (1983) soil parameters (open literature) |
| GPU path | `BatchedElementwise` (CN op + GA op), batch-composable |

**Key learning for ToadStool**: The coupled partition is pure math — two function
calls and three `min`/`max` operations. Mass conservation is inherent (P = Q + F + S).
A fused `rainfall_partition(P, CN, GA_params, duration)` primitive would be a
natural addition to `barracuda::stats::hydrology`.

### Exp 053: Van Genuchten Inverse Parameter Estimation

**Purpose**: Validate that `barracuda::optimize::brent` can solve the inverse VG
problem: given θ_target, find h such that θ(h) = θ_target. Tests forward VG θ(h),
Mualem K(h), monotonicity, and θ→h→θ round-trip for 7 USDA textures.

| Component | Detail |
|-----------|--------|
| Python | 84/84 PASS — forward, conductivity, round-trip, monotonicity (7 soils × 5 Se fractions) |
| Rust | `validate_vg_inverse` — 84/84 PASS, Brent convergence for all practical Se |
| Modules | `eco::van_genuchten::{van_genuchten_theta, van_genuchten_k, inverse_van_genuchten_h}` |
| Data | Carsel & Parrish (1988) Table 1 — 7 USDA soil textures (open literature) |
| GPU path | `BatchedOptimize` (Tier B, Brent root-finding) |

**Key learning for ToadStool**: Fine-textured soils (clay, n=1.09) have extremely
long VG tails — θ(h) approaches θ_r only at astronomically negative heads (< -10^7 cm).
The Python baseline explicitly marks Se fractions unreachable within practical search
bounds. A batched Brent on GPU would accelerate soil parameter inversion at field scale
(thousands of grid cells).

### Exp 054: Full-Season Irrigation Water Budget Audit

**Purpose**: Audit the complete FAO-56 pipeline end-to-end: synthetic deterministic
weather → PM ET₀ → trapezoidal Kc schedule → daily water balance → Stewart yield,
for 4 crops (corn, soybean, winter_wheat, alfalfa) over a full growing season.

| Component | Detail |
|-----------|--------|
| Python | 34/34 PASS — mass balance, ETa ≤ ETc, yield ∈ [0,1], cross-crop ordering |
| Rust | `validate_season_wb` — 34/34 PASS, mass balance to 0.1 mm |
| Modules | `eco::evapotranspiration::daily_et0` + `eco::water_balance` + `eco::yield_response` |
| Data | FAO-56 Ch 2-8 equations + Stewart (1977) Ky values (open literature) |
| GPU path | `SeasonalPipeline` (chains ET₀→Kc→WB→Yield, zero round-trip) |

**Key learning for ToadStool**: The mass balance equation is
`Σ(P + Irr) - Σ(ETa + DP) + Dr_final = 0`. Getting the sign of final root-zone
depletion correct is critical — it represents stored water, not a loss. The seasonal
pipeline already has a `SeasonalPipeline` GPU orchestrator; this experiment validates
the CPU path that the GPU will replicate.

---

## Part 2: What ToadStool Should Know

### Module Coupling Pattern

All three experiments follow the same pattern: compose existing validated primitives
into a higher-order computation, then verify that the composition preserves physical
invariants (mass conservation, monotonicity, bounds). This is the pattern for GPU
pipeline evolution:

```
Individual primitives (validated in isolation)
     │
     ▼
Composed pipelines (validated for coupling invariants)
     │
     ▼
GPU dispatch (same math, batched)
     │
     ▼
ToadStool absorption (primitive → shader)
```

### Brent Root-Finding as a General GPU Primitive

Exp 053 uses `barracuda::optimize::brent` for VG inversion. The same pattern applies
to any monotonic function inversion — soil hydraulic properties, isotherm equilibria,
thermodynamic state equations. A **batched Brent GPU shader** (`brent_f64.wgsl`) would
serve all Springs:

- **airSpring**: VG inverse θ→h, Green-Ampt ponding time
- **hotSpring**: EOS inverse (density→pressure)
- **wetSpring**: Dose-response IC50, growth rate inversion
- **groundSpring**: Spectral eigenvalue bracketing

### Absorption Candidates (V035)

| Function | Current Location | Suggested Upstream | Priority |
|----------|-----------------|-------------------|----------|
| `rainfall_partition(P, CN, GA, dur)` | airSpring Exp 052 | `barracuda::stats::hydrology` | Medium |
| Batched Brent GPU | `barracuda::optimize::brent` (CPU) | `brent_f64.wgsl` | High |
| Season pipeline GPU | `gpu::seasonal_pipeline` (CPU chain) | `seasonal_pipeline.wgsl` | High |
| Trapezoidal Kc schedule | `eco::water_balance` inline | `barracuda::stats::hydrology::kc_schedule` | Low |

### Cross-Spring Relevance

These experiments strengthen baseCamp papers:

| Experiment | baseCamp Paper | Contribution |
|-----------|---------------|-------------|
| Exp 052 (coupled runoff) | 06 (No-Till Anderson) | Event-scale hydrology for tillage impact modeling |
| Exp 053 (VG inverse) | 03 (BioAg Microbiome) | Soil hydraulic characterization for rhizosphere geometry |
| Exp 054 (season WB) | 08 (NPU Ag IoT) | End-to-end pipeline the NPU will accelerate |

---

## Part 3: Cumulative State

| Metric | V034 | V035 |
|--------|------|------|
| Experiments | 51 | **54** |
| Python checks | 1237 | **1237** (new checks in new scripts) |
| Rust lib tests | 618 | **618** |
| Validation binaries | 56 | **59** (+3) |
| New validation checks | — | **410** (292+84+34) |
| Cross-spring benchmarks | 30/30 | **30/30** |
| ET₀ methods | 8 | **8** |
| Clippy warnings | 0 | **0** |
| Unsafe blocks | 0 | **0** |
| Named constants | 42+ | **42+** |

---

## Part 4: Recommended ToadStool Actions

1. **Batched Brent GPU shader** (`brent_f64.wgsl`) — Exp 053 demonstrates the need.
   VG inversion at field scale (10K+ grid cells) is a natural GPU workload. The
   function signature: `brent_f64(f, a, b, tol) -> f64` maps cleanly to WGSL.

2. **Absorb `rainfall_partition`** into `barracuda::stats::hydrology` — pure math,
   composes `scs_cn_runoff` + `cumulative_infiltration` + min/max. Widely applicable.

3. **Seasonal pipeline GPU promotion** — Exp 054 validates the CPU chain. The existing
   `SeasonalPipeline` orchestrator already defines the dispatch structure. Wiring to
   ToadStool shaders would give end-to-end GPU: weather → ET₀ → Kc → WB → yield in
   a single GPU dispatch with zero CPU round-trips.

4. **VG fine-texture handling** — ToadStool's precision evolution (df64) is relevant
   here. Clay VG curves with n ≈ 1.09 require extreme head ranges; df64 precision
   in the Brent shader would extend the convergence domain.

---

## Part 5: Next Steps for airSpring

- CPU benchmarks for Exp 052-054 (Rust vs Python speedup measurement)
- Wire coupled runoff into `water_balance::RunoffModel` as `RunoffModel::ScsCn`
- Candidate experiments: multi-station coupled hydrology, VG pedotransfer coupling
- Continue GPU path: CPU → GPU parity for coupled modules via batched dispatch
- Prepare for GPU validation: Exp 054 pipeline → `SeasonalPipeline` GPU dispatch
