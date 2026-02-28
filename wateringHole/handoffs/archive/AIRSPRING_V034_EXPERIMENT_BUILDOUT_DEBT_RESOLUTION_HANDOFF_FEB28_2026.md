# airSpring V034 — Experiment Buildout + Deep Technical Debt Resolution

**Date**: February 28, 2026
**From**: airSpring V034 (v0.5.2)
**To**: ToadStool / BarraCuda team
**Status**: 51 experiments, 1237 Python + 618 lib tests + 56 binaries + 30/30 cross-spring benchmarks
**Supersedes**: V033 (cross-spring rewiring + benchmark evolution)
**License**: AGPL-3.0-or-later

---

## Executive Summary

- **3 new experiments** built through full pipeline (Python → Rust CPU → GPU path ready):
  Blaney-Criddle PET (Exp 049), SCS Curve Number runoff (Exp 050), Green-Ampt infiltration (Exp 051)
- **93 new Python checks** (18+38+37), **29 new Rust lib tests**, **3 new validation binaries**,
  **5 new cross-spring benchmarks** (25→30), **3 new provenance entries** (13→16)
- **Deep technical debt resolution**: 30+ named constants extracted from FAO-56/solar/Thornthwaite
  magic numbers, all `#[allow(dead_code)]` resolved, `#[allow(unreachable_code)]` fixed,
  unsafe casts evolved to `TryFrom`/`from_ne_bytes`, GPU adapter hardcoding → capability-based
- **Zero clippy warnings**, zero `unsafe`, zero production `unwrap()`, zero mocks in production
- **8th ET₀ method** (Blaney-Criddle) completes the temperature-only PET portfolio

---

## Part 1: New Experiments

### Exp 049: Blaney-Criddle (1950) Temperature PET

**Paper**: Blaney HF, Criddle WD (1950) USDA-SCS Tech Paper 96.

The simplest widely-used PET method — requires only temperature and daylight fraction.
Historically dominant in western US irrigation districts.

| Component | Detail |
|-----------|--------|
| Python | 18/18 PASS — analytical, daylight, monotonicity, cross-method |
| Rust | `eco::evapotranspiration::blaney_criddle_et0()`, `blaney_criddle_p()`, `blaney_criddle_from_location()` |
| Tests | 5 unit tests |
| Binary | `validate_blaney_criddle` PASS |
| Data | FAO-24 Table 18, USDA-SCS (open literature) |
| GPU path | `BatchedElementwise` Tier B, op=BC |

**toadStool action**: Consider absorbing `blaney_criddle_et0(tmean, p) -> f64` into
`barracuda::stats::hydrology` alongside `hargreaves_et0` — same pattern, different method.

### Exp 050: SCS Curve Number Runoff (USDA 1972)

**Paper**: USDA-SCS (1972) NEH-4; USDA-SCS (1986) TR-55.

Industry-standard rainfall-runoff estimation: Q = (P - Ia)² / (P - Ia + S).
Complements airSpring's existing water balance by providing physics-based runoff
estimation (replacing the current `RunoffModel::SimpleThreshold`).

| Component | Detail |
|-----------|--------|
| Python | 38/38 PASS — CN table, AMC adjustment, monotonicity, edge cases |
| Rust | New `eco::runoff` module — `scs_cn_runoff()`, `potential_retention()`, `amc_cn_dry/wet()`, `LandUse`/`SoilGroup` enums |
| Tests | 12 unit tests |
| Binary | `validate_scs_cn` PASS |
| Data | USDA NEH-4 / TR-55 (public domain) |
| GPU path | `BatchedElementwise` Tier B, op=CN |

**toadStool action**: `scs_cn_runoff(precip, cn, ia_ratio) -> f64` is a strong absorption
candidate for `barracuda::stats::hydrology` — pure math, widely used across hydrology.

### Exp 051: Green-Ampt (1911) Infiltration

**Paper**: Green WH, Ampt GA (1911) J Agricultural Science 4(1):1-24.
Parameters: Rawls WJ et al. (1983) J Hydraul Eng 109(1):62-70.

Physics-based infiltration model complementing Richards equation. Solves the implicit
Green-Ampt equation via Newton-Raphson iteration. Includes 7 named soil constants
from Rawls et al. (1983) and ponding time estimation under constant rainfall.

| Component | Detail |
|-----------|--------|
| Python | 37/37 PASS — 7 soils, Newton-Raphson, ponding, monotonicity |
| Rust | New `eco::infiltration` module — `cumulative_infiltration()`, `infiltration_rate()`, `ponding_time()`, `GreenAmptParams` |
| Tests | 12 unit tests |
| Binary | `validate_green_ampt` PASS |
| Data | Rawls et al. (1983) Table 1 (open literature) |
| GPU path | `BatchedElementwise` Tier B, op=GA (Newton-Raphson on GPU needs iterative dispatch) |

**toadStool action**: The Newton-Raphson solver is a candidate for `barracuda::optimize::newton`
or a batched implicit equation solver. The 7 soil parameters (`GreenAmptParams`) could
be a companion to `pde::richards::SoilParams`.

---

## Part 2: Deep Technical Debt Resolution

### Constants Extraction (30+ named `const`)

Inline FAO-56 magic numbers extracted to named constants with paper references:

| Module | Constants | Example |
|--------|-----------|---------|
| `eco::evapotranspiration` | 21 | `MAGNUS_A/B/C`, `HARGREAVES_COEFF`, `MJ_TO_MM`, `BC_TEMP_COEFF` |
| `eco::solar` | 9 | `SOLAR_CONSTANT_MJ`, `STEFAN_BOLTZMANN`, `CLEAR_SKY_BASE` |
| `eco::thornthwaite` | 12 | `EXPONENT_C0-C3`, `WILLMOTT_A/B/C`, `HIGH_TEMP_THRESHOLD` |

Doc comments retain the original numeric values for paper traceability.

### Dead Code Resolution

| File | Before | After |
|------|--------|-------|
| `gpu::hargreaves` | `#[allow(dead_code)]` on `gpu_engine` | `pub const fn gpu_engine()` accessor |
| `gpu::sensor_calibration` | same | same pattern |
| `gpu::kc_climate` | same | same pattern |
| `gpu::dual_kc` | same | same pattern |
| `validate_long_term_wb` | `#[allow(dead_code)]` on `SeasonResult` | Unused fields prefixed `_` |

### Cast Hygiene

| File | Before | After |
|------|--------|-------|
| `eco::richards` | `as usize` on float ceil | `u64::try_from()` safe conversion |
| `npu.rs` | `as u64` on u128 | `u64::try_from().unwrap_or(u64::MAX)` |
| `npu.rs` | `as i8` on u8 | `i8::from_ne_bytes([b])` byte reinterpretation |

### Capability-Based Discovery

- **Removed**: `std::env::set_var("BARRACUDA_GPU_ADAPTER", "titan")` in `validate_gpu_live.rs`
- **Added**: Runtime discovery via `BARRACUDA_GPU_ADAPTER` env var with fallback
- Pattern: read env → if empty, use `WgpuDevice` runtime discovery → print capabilities

### Other Fixes

- `validate_atlas_stream.rs`: removed `#[allow(unreachable_code)]` and dead return after `h.finish()`
- All 4 GPU engine accessors: `pub const fn` + backtick-quoted `ToadStool` in doc comments

---

## Part 3: What ToadStool Should Know

1. **airSpring now has 8 ET₀ methods**: PM, Priestley-Taylor, Hargreaves, Thornthwaite,
   Makkink, Turc, Hamon, Blaney-Criddle. All validated against papers with open data.
   The ensemble consensus (Exp 037) weights all methods — more methods = better consensus.

2. **SCS-CN is the industry standard for runoff**: It maps directly to USDA soil groups
   and land use categories. The `LandUse`/`SoilGroup` enum + CN table pattern is the same
   pattern as `pde::richards::SoilParams` — consider a `stats::hydrology::CurveNumber` type.

3. **Green-Ampt complements Richards**: Where Richards solves the full PDE, Green-Ampt
   gives a fast analytical approximation. The implicit equation requires Newton-Raphson,
   which could benefit from a batched GPU solver.

4. **Named constants improve precision safety**: By extracting `MAGNUS_A/B/C` etc., any
   future precision evolution (f64 → df64) only needs to change the constant definition.

5. **The `gpu_engine()` accessor pattern** is now consistent across all 4 Tier B GPU
   modules. When ToadStool wires a new `BatchedElementwise` op, the engine is ready.

6. **Zero clippy warnings at pedantic+nursery level** — this is maintained across all
   51 experiments, 618 lib tests, and 56 binaries.

7. **Capability-based GPU discovery** is now the standard pattern. `validate_gpu_live`
   reads `BARRACUDA_GPU_ADAPTER` from env with fallback to runtime discovery, matching
   the pattern used by ToadStool's `WgpuDevice::new_f64_capable()`.

---

## Part 4: Cumulative State

| Metric | V033 | V034 |
|--------|------|------|
| Experiments | 48 | **51** |
| Python checks | 1144 | **1237** (+93) |
| Rust lib tests | 589 | **618** (+29) |
| Validation binaries | 53 | **56** (+3) |
| Cross-spring benchmarks | 25/25 | **30/30** (+5) |
| Provenance entries | 13 | **16** (+3) |
| ET₀ methods | 7 | **8** |
| Clippy warnings | 0 | **0** |
| Unsafe blocks | 0 | **0** |
| Production unwrap() | 0 | **0** |
| Named constants | ~5 | **42+** |
| #[allow(dead_code)] | 5 | **0** |

---

## Part 5: Recommended ToadStool Actions

1. **Absorb `blaney_criddle_et0`** into `barracuda::stats::hydrology` — same module as
   `hargreaves_et0`, completing the temperature-based ET₀ portfolio.

2. **Absorb `scs_cn_runoff`** into `barracuda::stats::hydrology` — pure math, widely
   used, natural companion to water balance primitives.

3. **Consider `GreenAmptParams`** as a companion type to `pde::richards::SoilParams` —
   the 7 Rawls (1983) soil types are standard reference parameters.

4. **Batched Newton-Raphson** GPU primitive — Green-Ampt's implicit equation is a
   natural test case for a `batched_newton_gpu.wgsl` shader.

5. **Named constant extraction pattern** — the `MAGNUS_A/B/C` pattern is worth adopting
   in ToadStool's own math modules for precision-evolution safety.

---

## Part 6: Next Steps for airSpring

- Wire SCS-CN into `water_balance::RunoffModel` as `RunoffModel::ScsCn { cn, ia_ratio }`
- Build CPU benchmark for Blaney-Criddle, SCS-CN, Green-Ampt (Rust vs Python speedup)
- Candidate experiments from queue: more cross-spring coupling, field data when available
- Continue `BarraCuda` GPU path: CPU → GPU parity for new modules via `BatchedElementwise`
