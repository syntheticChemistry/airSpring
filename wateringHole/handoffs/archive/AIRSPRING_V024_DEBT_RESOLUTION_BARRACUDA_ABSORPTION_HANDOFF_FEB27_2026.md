# toadStool / barracuda — V024 Debt Resolution + Barracuda Absorption Handoff

**Date**: 2026-02-27
**From**: airSpring v0.4.12
**To**: ToadStool / BarraCuda core team
**ToadStool pin**: S68 HEAD (`89356efa`)
**License**: AGPL-3.0-or-later
**Covers**: v0.4.12 (clippy pedantic enforcement, tolerance centralization, CI coverage gate, error type evolution, capability-based NPU discovery, primal self-knowledge docs)
**Supersedes**: V023 (comprehensive evolution)

**airSpring**: 499 Rust lib tests + 853 validation + 1393 atlas checks, 37 barracuda + 1 forge binary, 808/808 Python, 0 clippy pedantic warnings, 97.06% coverage, CI coverage gate at 80%

---

## Executive Summary

- **Deep debt resolution complete**: 156 clippy warnings eliminated, pedantic + nursery enforced across all 37 barracuda binaries + lib + integration tests + forge crate. Zero warnings at `-D warnings`.
- **Tolerance centralization**: 20 new domain-specific `Tolerance` constants (using `barracuda::tolerances::Tolerance` S52 struct) — every validation threshold now has `name`, `abs_tol`, `rel_tol`, and citation-backed `justification`. 10 binaries migrated from local constants.
- **Error type evolution**: `gpu::richards` migrated from `Result<_, String>` to `crate::error::Result<_>` (`AirSpringError::Barracuda`). Zero `String`-typed errors remain in library code.
- **CI hardened**: `cargo-llvm-cov` coverage gate (80% minimum), pedantic clippy for both barracuda and forge, Python baseline runner.
- **Primal self-knowledge**: All module docs evolved to describe capabilities consumed from `barracuda::` rather than naming sibling Springs. Runtime discovery for NPU (`/dev/akida*` scan) and weather data (`LONG_TERM_WB_CACHE` env override).
- **6 baseline commits pinned**: hargreaves, thornthwaite, gdd, pedotransfer, diversity, ameriflux benchmarks pinned to `fad2e1b` (was `"pending"`).

---

## Part 1: What BarraCuda Should Absorb from This Session

### 1.1 Tolerance Registry Pattern

airSpring's `tolerances.rs` (21 constants) demonstrates a domain-specific tolerance registry built on `barracuda::tolerances::Tolerance`. Each constant carries:

```rust
pub const WATER_BALANCE_MASS: Tolerance = Tolerance {
    name: "water_balance_mass_balance",
    abs_tol: 1e-10,
    rel_tol: 0.0,
    justification: "FAO-56 Ch 8 conservation of mass — total in = total out",
};
```

**Absorption target**: Consider shipping a `tolerance_registry!` macro or builder in `barracuda::tolerances` that Springs can use to declare domain registries with automatic meta-tests. airSpring's pattern (meta-test that iterates all constants, verifies non-empty justification, checks rel_tol > 0 for comparison tolerances) is reusable.

### 1.2 Centralized Cast Allows

airSpring allows `cast_precision_loss`, `cast_possible_truncation`, and `cast_sign_loss` in both `lib.rs` and all binaries. The justification: ecological domain uses f64 exclusively, input sizes are bounded (≤15,300 station-days), and `usize → f64` is lossless for N < 2^53.

**Learning for BarraCuda**: If barracuda ships pedantic lints in `Cargo.toml`, Springs consuming barracuda may need these allows. Consider documenting the recommended allow set for f64-heavy scientific consumers.

### 1.3 Error Type Consolidation

`gpu::richards` was the last module using `Result<_, String>`. Migrated to `AirSpringError::Barracuda(String)` with explicit `map_err`. Pattern:

```rust
fn solve_batch_cpu(/* ... */) -> crate::error::Result<Vec<Vec<f64>>> {
    upstream_solve(/* ... */).map_err(|e| AirSpringError::Barracuda(e))
}
```

**Absorption target**: If `barracuda::pde::richards::solve_richards` returned `Result<_, BarracudaError>` instead of `Result<_, String>`, Springs wouldn't need to map. This applies to any barracuda function still returning `String` errors.

---

## Part 2: BarraCuda Evolution Review — How airSpring Uses BarraCuda

### 2.1 Active Delegations (11 Tier A, all wired)

| airSpring Module | BarraCuda Primitive | Usage Pattern |
|-----------------|--------------------|----|
| `gpu::et0` | `ops::batched_elementwise_f64` (op=0) | GPU-FIRST — 12.7M ops/s CPU |
| `gpu::water_balance` | `ops::batched_elementwise_f64` (op=1) | GPU-STEP per field |
| `gpu::kriging` | `ops::kriging_f64::KrigingF64` | 100-station atlas spatial interp |
| `gpu::reduce` | `ops::fused_map_reduce_f64` | Seasonal aggregation, N≥1024 GPU |
| `gpu::stream` | `ops::moving_window_stats` | IoT 24-hour sliding window |
| `gpu::richards` | `pde::richards::solve_richards` | Crank-Nicolson + Thomas |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start` | LHS global search |
| `gpu::mc_et0` | `stats::normal::norm_ppf` | Parametric CI for MC ET₀ |
| `eco::correction` | `linalg::ridge::ridge_regression` | Sensor calibration |
| `eco::richards` | `optimize::brent` | VG inversion θ→h |
| `eco::diversity` | `stats::diversity` | Shannon, Simpson, Chao1, Bray-Curtis |

### 2.2 Pending Delegations (Tier B, ready to wire)

| Need | Primitive | Effort | Papers Benefiting |
|------|-----------|:------:|:-----------------:|
| Dual Kc batch | `batched_elementwise_f64` (op=8) | Low | 006, 008 |
| Thornthwaite batch | `batched_elementwise_f64` (op=TH) | Low | 021 |
| Hargreaves batch | `batched_elementwise_f64` (op=HG) | Low | 031 |
| GDD scan/accumulate | `wgsl_scan_f64` (prefix sum) | Low | 022 |
| Pedotransfer batch | `batched_elementwise_f64` (new) | Low | 023 |
| Multi-crop pipeline | New `BatchedCropPipeline` | Medium | 027 |
| Forecast MC loop | Parallel scheduling loop | Medium | 025 |
| NPU dispatch trait | `NpuDispatch` | Medium | 028-029b |

### 2.3 What Stays Local (domain consumer, not compute primitive)

| Module | Reason |
|--------|--------|
| `eco::evapotranspiration` | 23+ FAO-56 functions — domain consumer |
| `eco::dual_kc` | FAO-56 Ch 7/11 domain logic |
| `eco::crop` | FAO-56 Table 12 crop database |
| `eco::soil_moisture` | Saxton-Rawls pedotransfer |
| `eco::sensor_calibration` | SoilWatch 10 specific |
| `eco::yield_response` | Stewart 1977 yield model |
| `eco::water_balance` | FAO-56 Ch 8 water balance |
| `io::csv_ts` | IoT CSV streaming parser |
| `npu` | akida-driver wrapper (may evolve to `barracuda::npu`) |

---

## Part 3: Cross-Spring Learnings for BarraCuda Evolution

### 3.1 NPU Convergence (multi-spring)

Three springs (airSpring, wetSpring, groundSpring) are independently proposing NPU integration:

| Spring | Use Case | Hardware | int8 Viable? |
|--------|----------|----------|:------------:|
| **airSpring** | Crop stress / irrigation / anomaly | AKD1000 | Yes |
| **wetSpring** | PFAS screening / microbiome | AKD1000 | Yes |
| **groundSpring** | MC uncertainty classification | Proposed | Yes |

All three share the pattern: `discover → load → infer → cpu_fallback`. This strongly suggests `barracuda::npu::NpuDispatch` should be a P0 upstream primitive.

### 3.2 Error Handling Standard (from groundSpring V10)

wateringHole standard: `if let Ok` + always-compiled CPU fallback.

```rust
let result = if let Ok(device) = Device::discover() {
    device.compute(&input).unwrap_or_else(|_| cpu_fallback(&input))
} else {
    cpu_fallback(&input)
};
```

This pattern is now used consistently in airSpring's NPU, GPU, and Richards modules.

### 3.3 Primal Self-Knowledge Pattern

airSpring evolved all module-level docs from naming sibling Springs to describing barracuda capabilities:

**Before**: "Uses hotSpring precision math for pow_f64"
**After**: "Uses barracuda::ops for batched f64 elementwise dispatch"

This decouples documentation from the Spring ecosystem graph. A module describes what capabilities it consumes from `barracuda::`, not which Spring wrote the implementation.

### 3.4 From airSpring → Upstream (already absorbed, S40-S66)

| Contribution | Upstream Module | Session |
|-------------|----------------|---------|
| Richards PDE | `pde::richards` | S40 |
| Stats metrics (RMSE, MBE, NSE, IA, R²) | `stats::metrics` | S64 |
| Regression fitting | `stats::regression` | S66 |
| Hydrology (Hargreaves, Kc, WB) | `stats::hydrology` | S66 |
| Moving window f64 | `stats::moving_window_f64` | S66 |
| `pow_f64` fractional fix | TS-001 | S54 |
| `acos` precision boundary | TS-003 | S54 |
| Reduce buffer N≥1024 | TS-004 | S54 |

### 3.5 Items Tracked for Future Absorption

| Primitive | Source | When | airSpring Impact |
|-----------|--------|------|------------------|
| `barracuda::npu::NpuDispatch` | Multi-spring | Pending S69+ | Replace local `npu.rs` |
| `barracuda::nn::SimpleMLP` | neuralSpring V24 | Pending S69+ | Crop regime surrogates |
| `batched_multinomial.wgsl` | groundSpring V10 | Pending | GPU rarefaction for diversity |
| Power-budget dispatch | wetSpring V61 | Design phase | Edge deployment routing |
| `barracuda::pde::richards` VG lookup | This handoff (P0 item 3) | Pending | 12 USDA texture standard table |

---

## Part 4: Open Data + Controls Audit

### All 32 Papers Use Open Data

| Data Source | Papers Using It | Access |
|-------------|:--------------:|--------|
| FAO-56 tables/equations (open literature) | 20 | Free |
| Open-Meteo ERA5 (80+ yr, global) | 8 | Free, no key |
| Published paper tables/equations | 6 | Free |
| Carsel & Parrish 1988 (USDA textures) | 3 | Free |
| AmeriFlux (eddy covariance) | 1 | Free, registration |
| BrainChip AKD1000 (live hardware) | 3 | Hardware |

Zero institutional access. Zero proprietary data. Every experiment reproducible from `git clone` + public APIs.

### Three-Tier Control Matrix

| Tier | Tool | Coverage |
|------|------|----------|
| Python control | `control/*/` scripts (30 scripts) | 808/808 checks |
| BarraCuda CPU | `validate_*` binaries (37 barracuda + 1 forge) | 853 + 1393 atlas |
| BarraCuda GPU/NPU | `gpu::*` orchestrators (11 Tier A) + NPU (3 exp) | 11 wired + 95 NPU checks |

### Compute Pipeline Coverage

| Paper | Python | Rust CPU | GPU | NPU | metalForge |
|:-----:|:------:|:--------:|:---:|:---:|:----------:|
| 001-005 | 206/206 | 102/102 | GPU-FIRST | — | All modules |
| 006-008 | 117/117 | 116/116 | Tier B | — | hydro/VG |
| 009-011 | 164/164 | 162/162 | WIRED | — | VG/isotherm |
| 012-017 | 167/167 | 162/162 | BatchedEt0 | — | hydro/metrics |
| 018 | cross-val | 1393/1393 | All scale | — | All modules |
| 019-023 | 215/215 | 222/222 | Tier B | — | evapotrans/soil |
| 024-027 | 141/141 | 140/140 | BatchedWB | — | hydro/yield |
| 028-029b | — | 95/95 | — | **AKD1000** | dispatch |
| 030-032 | 73/73 | 73/73 | BatchedEt0 | — | metrics/diversity |

---

## Part 5: Recommended ToadStool Actions (Updated)

### P0 — High Priority

1. **`NpuDispatch` trait**: Generic NPU interface. airSpring (Exp 028-029b), wetSpring (V61), and groundSpring (V10) all propose this. airSpring's `npu.rs` is the reference implementation with discover/load/infer/batch.

2. **`BatchedCropPipeline` shader**: Compose ET₀ → Kc → WB → yield as a single GPU dispatch. Validated for 5 crops × 100 stations = 500 work units. The "Penny Irrigation" kernel.

3. **Carsel & Parrish (1988) VG lookup**: Ship 12 USDA texture VG parameters (θr, θs, α, n, Ks) as a standard table in `barracuda::pde::richards`. Every Spring doing soil physics needs this.

4. **`Result<_, BarracudaError>` for `pde::richards`**: Replace `Result<_, String>` returns. Springs currently map String errors into their own error types.

### P1 — Medium Priority

5. **`tolerance_registry!` macro**: Builder for domain-specific tolerance registries with automatic meta-tests (justification check, rel_tol validation).

6. **`BatchedForecastLoop`**: Monte Carlo scheduling with stochastic weather noise. N realizations per field → confidence intervals on irrigation decisions.

7. **Thornthwaite + Hargreaves + GDD batch ops**: Low-effort additions completing the ET₀ method family.

### P2 — Low Priority

8. **Pedantic lint guidance**: Document recommended `#![allow]` set for f64-heavy scientific consumers using `barracuda` with pedantic clippy.

---

## Part 6: Quality Gates (v0.4.12)

| Check | Status |
|-------|--------|
| `cargo clippy --pedantic` | **0 warnings** (lib + 37 bins + forge) |
| `cargo test --lib` | **499 passed** |
| `cargo test` (all) | **643 total** |
| `cargo fmt --check` | **Clean** |
| `cargo doc` | **0 warnings** |
| `cargo llvm-cov --lib` | **97.06%** line coverage |
| Forge tests | **26 passed** |
| Python baselines | **808/808 PASS** |
| Unsafe blocks | **0** |
| `.unwrap()` in lib | **0** |
| TODO/FIXME in lib | **0** |

## Test Verification

```bash
cd barracuda
cargo test --lib                    # 499 passed
cargo clippy -- -D warnings -W clippy::pedantic  # 0 warnings
cargo fmt --check                   # clean
cargo llvm-cov --lib --summary-only # 97.06% lines

cd ../metalForge/forge
cargo test                          # 26 passed
cargo clippy -- -D warnings -W clippy::pedantic  # 0 warnings
```

---

## Handoff Checklist

- [x] Debt resolution documented (clippy pedantic, tolerance centralization, error evolution)
- [x] Full barracuda delegation inventory (11 Tier A + 8 Tier B + stays-local)
- [x] Cross-spring NPU convergence documented (3 springs proposing `NpuDispatch`)
- [x] Open data audit complete (all 32 papers, zero institutional access)
- [x] Three-tier control matrix verified (Python → CPU → GPU/NPU)
- [x] Recommended ToadStool actions prioritized (P0/P1/P2)
- [x] CI pipeline hardened (coverage gate, pedantic clippy)
- [x] V023 superseded (all V023 items remain valid, this extends)
- [x] All quality gates passing
