# airSpring V023 — Comprehensive Evolution Handoff for ToadStool/BarraCuda Absorption

**Date**: 2026-02-27 (updated from Feb 26 with S68 HEAD sync + cross-spring review)
**From**: airSpring v0.4.11
**To**: ToadStool / BarraCuda team
**ToadStool pin**: S68 HEAD (`89356efa` — synced from `f0feb226`, +1 CPU feature-gate fix)
**License**: AGPL-3.0-or-later
**Covers**: v0.4.9–v0.4.11 (NASS yield, forecast scheduling, SCAN moisture, multicrop budget, NPU trilogy, AmeriFlux ET, Hargreaves, diversity, metalForge mixed hardware)
**Supersedes**: V022 (Thornthwaite + GDD + pedotransfer)

**airSpring**: 499 Rust lib tests + 853 validation + 1393 atlas checks, 37 barracuda + 1 forge binary, 808/808 Python, 0 clippy errors, 97.06% coverage

---

## Executive Summary

- **32 experiments validated** (up from 22 at V022): full Python → Rust CPU → GPU pipeline for precision agriculture, environmental systems, and NPU edge inference.
- **NPU breakthrough** (Exp 028/029/029b): BrainChip AKD1000 live — crop stress, irrigation decision, and anomaly classifiers at ~48µs. LOCOMOS power budget proves 5W solar-powered edge is feasible (2.53 Wh/day).
- **metalForge evolved**: from absorption staging to mixed hardware dispatch — CPU+GPU+NPU substrate discovery, capability routing, 14 eco workloads.
- **9 new experiments** since V022: NASS yield (024), forecast scheduling (025), SCAN moisture (026), multicrop budget (027), NPU trilogy (028/029/029b), AmeriFlux ET (030), Hargreaves (031), diversity (032).
- **All 32 papers use open data** — zero institutional access, zero proprietary sensors. Every paper reproducible from `git clone` + public APIs.
- **toadStool actions**: NPU dispatch primitives, batched forecast loop, SCAN-scale Richards, multi-crop parallel pipeline, AmeriFlux-scale ET₀ batch.

---

## Part 1: What BarraCuda Should Absorb

### Priority 1: Batched Multi-Crop Pipeline (from Exp 027)

The multi-crop water budget exercises the full FAO-56 pipeline as 5 independent work units (corn, soybean, potato, winter wheat, dry bean) — each with distinct Kc curves, root depths, and yield sensitivity (Ky).

```
Per crop:  ET₀ → Kc(stage) → ETc → water_balance(θ) → Ks → ETa → yield_ratio
```

**Absorption target**: A `BatchedCropPipeline` that dispatches N crop×field pairs in parallel. Each work item is: `(weather[T], crop_params, soil_params) → (ETa[T], yield_ratio, irrigation_events)`.

**Binding layout**:
- @group(0) @binding(0): weather buffer (N_days × 6 floats: tmax, tmin, rh, wind, rs, rain)
- @group(0) @binding(1): crop params (Kc_ini, Kc_mid, Kc_end, stages[4], Ky, root_depth)
- @group(0) @binding(2): soil params (θ_fc, θ_wp, θ_s, Ksat, TAW, REW)
- @group(0) @binding(3): output (ETa_total, yield_ratio, n_irrigations, water_used)

**Why**: This is the "Penny Irrigation" kernel — the thing that runs on a farmer's $600 GPU. The pipeline is already validated for 5 crops × 100 stations = 500 work units. GPU dispatch would make it real-time for 1000+ fields.

### Priority 2: NPU Dispatch Primitive (from Exp 028-029b)

airSpring's `npu.rs` module wraps `akida-driver` for BrainChip AKD1000. The pattern:

```rust
let npu = NpuHandle::discover()?;       // find AKD1000 on PCIe
npu.load_model(weights)?;               // upload int8 weights
let result = npu.infer(&features)?;     // DMA round-trip ~48µs
```

**What BarraCuda should provide**: A generic `NpuDispatch` trait matching the GPU `GpuDispatch` pattern:

```rust
pub trait NpuDispatch {
    fn discover() -> Result<Self, BarracudaError>;
    fn load(&mut self, model: &[u8]) -> Result<(), BarracudaError>;
    fn infer(&self, input: &[f32]) -> Result<Vec<f32>, BarracudaError>;
    fn infer_batch(&self, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>, BarracudaError>;
}
```

**Key findings from airSpring NPU experiments**:
- int8 quantization: <0.01 error on [0,1] range, <3mm on [0,300] range
- Inference latency: ~48µs mean, P99 <70µs (single classification)
- Streaming throughput: 20,545 Hz (single-feature) to 21,023 Hz (multi-sensor)
- Weight hot-swap: 23.5µs per crop model load
- Power: 0.0009% of active cycle energy — negligible vs sensors
- **LOCOMOS deployment**: Pi Zero 2 W + AKD1000 = 2.53 Wh/day total, 5W solar = 8× surplus

### Priority 3: Forecast Scheduling Loop (from Exp 025)

Forecast-driven irrigation adds stochastic noise to the scheduling pipeline:

```
forecast_et0 = perfect_et0 + Normal(0, σ_noise)
forecast_rain = perfect_rain × LogNormal(μ, σ)
```

**Absorption target**: A `BatchedForecastLoop` that runs N Monte Carlo realizations of the scheduling pipeline per field, producing uncertainty-aware irrigation decisions.

**Key metric from Exp 025**: Forecast scheduling achieves 90-95% of perfect-knowledge yield while using real-world noisy weather data. Mass balance conserved under all noise levels.

### Priority 4: Richards at SCAN Scale (from Exp 026)

USDA SCAN provides in-situ soil moisture profiles for 3 Michigan soil textures (sand, silt loam, clay). The validation confirms our Richards 1D solver produces θ profiles within SCAN-published seasonal ranges.

**Absorption target**: `BatchedRichards` already exists (Tier A, wired). The new learning is that Carsel & Parrish (1988) VG parameters should be shipped as a standard lookup in `barracuda::pde::richards` — 12 USDA textures with θr, θs, α, n, Ks.

### Priority 5: AmeriFlux-Scale ET Validation (from Exp 030)

AmeriFlux eddy covariance provides direct ET measurements (energy balance closure) — the gold standard. Our FAO-56 PM matches flux tower ET within documented RMSE bounds.

**Absorption target**: No new primitive needed. But the validation pattern (flux tower data → ET₀ comparison → energy balance check) should be available as a `ValidationHarness` extension for any Spring validating atmospheric fluxes.

---

## Part 2: Error Handling and Patterns

### NPU Error Handling (wateringHole standard)

Following the wateringHole `if let Ok` + always-compiled CPU fallback pattern:

```rust
let result = if let Ok(npu) = NpuHandle::discover() {
    npu.infer(&features).unwrap_or_else(|_| cpu_classify(&features))
} else {
    cpu_classify(&features)
};
```

- NPU is always optional (`--features npu`)
- CPU fallback is always compiled
- Feature gate prevents akida-driver from being a hard dependency
- Discovery failure → silent CPU fallback (no panic, no error log)

### Forecast Loop Error Handling

Stochastic noise can produce negative ET₀ or rain. The validated pattern:

```rust
let et0 = (perfect_et0 + noise).max(0.0);  // clamp negative
let rain = (perfect_rain * factor).max(0.0); // clamp negative
```

Mass balance is verified after every simulation step — if ΔBalance > 1e-10 mm, the step is flagged.

---

## Part 3: Complete Delegation Inventory

### Active Delegations (airSpring → BarraCuda, all wired)

| airSpring Module | BarraCuda Primitive | Status |
|-----------------|--------------------|----|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **Integrated** |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **Wired** |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | **Wired** |
| `gpu::isotherm::fit_*_nm/global` | `optimize::nelder_mead` + `multi_start` | **Wired** |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **Wired** |
| `eco::diversity` | `stats::diversity` | **Leaning** (S64) |
| `gpu::mc_et0::parametric_ci` | `stats::normal::norm_ppf` | **Wired** |
| `eco::richards::inverse_van_genuchten_h` | `optimize::brent` | **Wired** |

### Pending Delegations (ready to wire)

| Need | BarraCuda Primitive | Effort |
|------|---------------------|:------:|
| Dual Kc batch (Ke) | `batched_elementwise_f64` (op=8) | Low |
| VG θ/K batch | `batched_elementwise_f64` (new op) | Low |
| Thornthwaite monthly ET₀ batch | `batched_elementwise_f64` (op=TH) | Low |
| GDD scan/accumulate | `wgsl_scan_f64` (prefix sum) | Low |
| Pedotransfer per-sample | `batched_elementwise_f64` (new op) | Low |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | Low |
| Multi-crop pipeline | New `BatchedCropPipeline` | Medium |
| Forecast Monte Carlo | Parallel scheduling loop | Medium |
| NPU dispatch | `NpuDispatch` trait | Medium |

### Not in BarraCuda (stays local in airSpring)

| Module | Reason |
|--------|--------|
| `eco::dual_kc` | FAO-56 Ch 7/11 domain logic — too specialized |
| `eco::sensor_calibration` | SoilWatch 10 specific — domain consumer |
| `eco::crop` | FAO-56 Table 12 crop database, GDD, kc_from_gdd — domain data |
| `eco::evapotranspiration` | 23+ functions — domain consumer, delegates math to BarraCuda |
| `eco::soil_moisture` | Saxton-Rawls pedotransfer — domain consumer |
| `io::csv_ts` | airSpring-specific IoT CSV parser |
| `npu` (partial) | akida-driver wrapper — may evolve to BarraCuda `NpuDispatch` |

---

## Part 4: Cross-Spring Learnings for BarraCuda Evolution

### From hotSpring
- `pow_f64` fractional exponent fix (TS-001) — critical for VG retention curves
- `norm_ppf` (Moro 1995) — MC ET₀ parametric confidence intervals
- `df64_transcendentals` — available for future VG precision improvement

### From wetSpring
- `kriging_f64` — spatial interpolation for 100-station atlas
- `fused_map_reduce` — seasonal aggregation (TS-004 N≥1024 fix)
- `moving_window_stats` — IoT sensor stream smoothing (24-hour window)
- `ridge_regression` — sensor calibration correction fitting
- `diversity` — agroecosystem Shannon/Simpson/Bray-Curtis (S64 absorption)

### From neuralSpring
- `nelder_mead` + `multi_start` — isotherm fitting (global search with LHS)
- `ValidationHarness` — used by all 37 validation binaries
- `brent` — VG pressure head inversion (θ→h, guaranteed convergence)

### From groundSpring
- `mc_et0_propagate_f64.wgsl` — Monte Carlo ET₀ uncertainty propagation
- Humidity dominates ET₀ uncertainty at 66% (Exp 003) — propagates into Anderson coupling

### From airSpring → upstream (already absorbed)
- Richards PDE → `pde::richards` (S40)
- Stats metrics (RMSE, MBE, NSE, IA, R²) → `stats::metrics` (S64)
- Regression fitting → `stats::regression` (S66)
- Hydrology (Hargreaves, Kc, WB) → `stats::hydrology` (S66)
- Moving window f64 → `stats::moving_window_f64` (S66)
- `pow_f64` fractional fix → TS-001 (S54)
- `acos` precision boundary → TS-003 (S54)
- Reduce buffer N≥1024 → TS-004 (S54)

---

## Part 5: NPU Learnings for BarraCuda Evolution

### AKD1000 Integration Architecture

```
eco:: domain modules (CPU, f64 validated)
    │
    ▼
npu:: wrapper (int8 quantization, feature encoding)
    │
    ▼
akida-driver (PCIe DMA, model loading)
    │
    ▼
BrainChip AKD1000 (80 NPs, 10 MB SRAM, 0.5 GB/s)
```

### Key NPU Findings

| Metric | Value | Implication for BarraCuda |
|--------|-------|---------------------------|
| int8 quantization error | <0.01 on [0,1] | Classification tasks tolerate quantization |
| f64→int8 round-trip | <3mm on [0,300] | Soil moisture classification is NPU-viable |
| Single inference | ~48µs (mean) | 1000× cheaper than cloud round-trip |
| Streaming throughput | 21,023 Hz | Handles 1-min cadence with 99.9% margin |
| Weight hot-swap | 23.5µs | Multi-crop switching is near-free |
| Daily energy (15-min) | 2.53 Wh | 18650 battery-powered field deployment |
| NPU share of energy | 0.0009% | NPU is effectively free in power budget |
| Solar surplus (5W panel) | 8× daily need | Year-round solar-only is feasible |

### LOCOMOS Deployment Model

Dong's LOCOMOS (LOw-COst MOnitoring System) + AKD1000:
- Pi Zero 2 W ($15) + AKD1000 ($99) + soil sensor ($20) + solar ($30)
- Total BoM: ~$165/node (vs $500-$5000 for commercial systems)
- Edge-sovereign: no cloud dependency, no recurring costs
- NPU enables classification without WiFi: stress/normal/anomaly decisions in-field

### Three NPU Workload Classes

| Class | Example | int8 Viable? | Latency |
|-------|---------|:------------:|---------|
| Classification | Crop stress (4 features → 2 classes) | **Yes** | 48µs |
| Streaming | Soil moisture time series (500 steps) | **Yes** | 48µs/step |
| Ensemble | 10-model consensus voting | **Yes** | 480µs total |

---

## Part 6: metalForge Mixed Hardware Dispatch

### Forge Architecture (v0.4.11)

```
forge/src/
├── lib.rs           # Crate root, feature gates
├── substrate.rs     # Runtime hardware discovery (CPU, GPU, NPU)
├── dispatch.rs      # Capability-based routing: GPU > NPU > CPU
├── probe.rs         # Hardware capability querying
├── workloads.rs     # 14 eco workload definitions + classification
├── inventory.rs     # Live device inventory
└── bin/
    └── validate_dispatch_routing.rs  # 21/21 dispatch routing checks
```

### Dispatch Priority

```
if workload.needs_f64_precision() → GPU (RTX 4070, ToadStool shaders)
if workload.is_classification() → NPU (AKD1000, int8)
else → CPU (i9-12900K, Rust --release)
```

### Workload Classification

| Workload | Substrate | Rationale |
|----------|-----------|-----------|
| ET₀ batch (PM) | GPU | f64 precision, embarrassingly parallel |
| Water balance | GPU | f64 + sequential per-field, parallel across fields |
| Richards PDE | GPU | f64, Crank-Nicolson + Thomas algorithm |
| Isotherm fitting | GPU | Nelder-Mead optimization, f64 |
| Kriging | GPU | Matrix operations, f64 |
| Stream smoothing | GPU | Moving window, f64→f32→f64 bridge |
| Reduce (seasonal) | GPU | fused_map_reduce, f64 |
| Ridge regression | GPU | Linear algebra, f64 |
| MC ET₀ uncertainty | GPU | Monte Carlo, f64 |
| Crop stress classify | NPU | int8, 4 features → 2 classes |
| Irrigation decision | NPU | int8, 6 features → 3 classes |
| Sensor anomaly | NPU | int8, 3 features → 2 classes |
| CSV parsing | CPU | I/O-bound, streaming |
| Benchmark JSON | CPU | Compile-time embedding |

---

## Part 7: Three-Tier Control Matrix

Every completed paper has been validated through three tiers:

| Tier | Tool | Purpose | Coverage |
|------|------|---------|----------|
| Python control | `control/*/` scripts | Reproduce paper results | 808/808 |
| BarraCuda CPU | `validate_*` binaries | Cross-validate Rust impl | 853 + 1393 atlas |
| BarraCuda GPU | `gpu::*` orchestrators | Verify GPU dispatch | 11 Tier A wired |

### Per-Paper Status

| Paper (Exp) | Python | Rust CPU | GPU Path | NPU Path | metalForge |
|:-----------:|:------:|:--------:|:--------:|:--------:|:----------:|
| 001 (ET₀ PM) | 64/64 | 31/31 | GPU-FIRST | — | metrics |
| 002 (Soil) | 36/36 | 26/26 | fit_ridge | — | regression |
| 003 (IoT) | 24/24 | 11/11 | StreamSmoother | — | moving_window |
| 004 (WB) | 18/18 | 13/13 | GPU-STEP | — | hydrology |
| 005 (Real) | R²=0.97 | 23/23 | All Tier A | — | All modules |
| 006 (Dual Kc) | 63/63 | 61/61 | Tier B | — | hydrology |
| 007 (Regional) | 61/61 | 61/61 | BatchedEt0 | — | metrics |
| 008 (Cover) | 40/40 | 40/40 | Tier B | — | hydrology |
| 009 (Richards) | 14/14 | 15/15 | WIRED | — | VG absorbed |
| 010 (Biochar) | 14/14 | 14/14 | WIRED | — | isotherm |
| 011 (60yr WB) | 10/10 | 11/11 | BatchedEt0+WB | — | hydrology |
| 012 (Yield) | 32/32 | 32/32 | Tier B | — | yield |
| 013 (CW2D) | 24/24 | 24/24 | BatchedRichards | — | VG |
| 014 (Sched) | 25/25 | 28/28 | BatchedWB+Et0 | — | hydrology |
| 016 (Lysim) | 26/26 | 25/25 | BatchedEt0 | — | metrics |
| 017 (Sens) | 23/23 | 23/23 | BatchedEt0 | — | metrics |
| 018 (Atlas) | cross-val | 1393/1393 | All scale | — | All modules |
| 019 (PT) | 32/32 | 32/32 | Tier B op=PT | — | evapotrans |
| 020 (Intercomp) | 36/36 | 36/36 | All methods | — | evapotrans |
| 021 (Thorn) | 23/23 | 50/50 | Tier B op=TH | — | evapotrans |
| 022 (GDD) | 33/33 | 26/26 | GDD scan | — | crop |
| 023 (Pedotr) | 70/70 | 58/58 | Tier B | — | soil |
| 024 (NASS) | 41/41 | 40/40 | BatchedWB+yield | — | yield+WB |
| 025 (Forecast) | 19/19 | 19/19 | BatchedWB | — | WB+forecast |
| 026 (SCAN) | 34/34 | 34/34 | BatchedRichards | — | VG (SCAN) |
| 027 (Multicrop) | 47/47 | 47/47 | Batch pipeline | — | hydro+yield |
| 028 (NPU) | — | 35+21 | — | **AKD1000** | **dispatch** |
| 029 (Funky) | — | 32/32 | — | **AKD1000** | streaming |
| 029b (Hi-cad) | — | 28/28 | — | **AKD1000** | fusion |
| 030 (Ameriflux) | 27/27 | 27/27 | BatchedEt0 | — | metrics |
| 031 (Hargreaves) | 24/24 | 24/24 | Tier B op=HG | — | evapotrans |
| 032 (Diversity) | 22/22 | 22/22 | DiversityGpu | — | diversity |

---

## Part 8: Recommended ToadStool Actions

### P0 — High Priority

1. **`NpuDispatch` trait**: Generic NPU interface matching `GpuDispatch` pattern. airSpring's `npu.rs` is the reference implementation. Other Springs (neuralSpring for protein folding edge inference, wetSpring for field-deployed PFAS screening) would benefit.

2. **`BatchedCropPipeline` shader**: Compose ET₀ → Kc → WB → yield as a single GPU dispatch. This is the "Penny Irrigation" kernel. airSpring has validated the pipeline for 5 crops × 100 stations.

3. **Carsel & Parrish (1988) VG lookup**: Ship 12 USDA texture VG parameters as a standard table in `barracuda::pde::richards`. Every Spring doing soil physics needs this.

### P1 — Medium Priority

4. **`BatchedForecastLoop`**: Monte Carlo scheduling with stochastic weather noise. N realizations per field, producing confidence intervals on irrigation decisions.

5. **Thornthwaite batch op**: Monthly ET₀ batch (reduce → map pattern). Low effort, completes the ET₀ method family for data-sparse deployments.

6. **GDD scan primitive**: Prefix sum for thermal time accumulation. Already available as `wgsl_scan_f64` — just needs a domain wrapper.

### P2 — Low Priority

7. **Pedotransfer batch**: Saxton-Rawls per-sample regression. Each sample is independent → embarrassingly parallel.

8. **Hargreaves batch op**: Temperature-only ET₀ for regions without wind/humidity data.

---

## Handoff Checklist

- [x] All 32 experiments documented with open data sources
- [x] Three-tier control matrix complete (Python → CPU → GPU/NPU)
- [x] NPU architecture and findings documented
- [x] metalForge dispatch routing validated (21/21 checks)
- [x] Cross-spring provenance documented for all delegations
- [x] Absorption targets prioritized (P0/P1/P2)
- [x] Error handling follows wateringHole standard (`if let Ok` + CPU fallback)
- [x] All quality gates passing (0 clippy, 0 unsafe, 97.06% coverage)
- [x] V022 superseded (all V022 items remain valid, this extends)

---

## Test Verification

```bash
cd barracuda
cargo test --lib                    # 499 passed
cargo clippy -- -D warnings         # 0 warnings
cargo fmt --check                   # clean
cargo llvm-cov --lib --summary-only # 97.06% lines

cd ../metalForge/forge
cargo test                          # 26 passed
```

## File Locations

| File | Purpose |
|------|---------|
| `barracuda/src/npu.rs` | AKD1000 NPU module (feature-gated) |
| `barracuda/src/eco/` | 14 domain modules |
| `barracuda/src/gpu/` | 11 Tier A GPU orchestrators |
| `barracuda/src/bin/validate_*.rs` | 33 validation binaries |
| `metalForge/forge/src/` | Mixed hardware dispatch crate |
| `control/*/` | 30 Python control scripts (808/808) |
| `specs/BARRACUDA_REQUIREMENTS.md` | Kernel requirements and tiers |
| `specs/CROSS_SPRING_EVOLUTION.md` | 774 WGSL shader provenance |
| `barracuda/EVOLUTION_READINESS.md` | Full Tier A/B/C inventory |

---

## Addendum: S68 HEAD Sync + Cross-Spring Review (Feb 27)

### ToadStool Sync

Synced from `f0feb226` → `89356efa` (1 commit: CPU feature-gate fix).
- `wgsl_hessian_column()`, `WGSL_HISTOGRAM`, `WGSL_BOOTSTRAP_MEAN_F64` now gated with `#[cfg(feature = "gpu")]`
- Reported by wetSpring V57 revalidation; no impact on airSpring (we use default features)
- **Revalidation**: 499/499 tests, 0 clippy, 97.06% coverage — all green
- **No new sessions beyond S68** — ToadStool is stable at this HEAD

### Sibling Spring Handoff Review

Reviewed 4 sibling handoffs to identify cross-spring learnings:

**wetSpring V61 (Feb 27)**: 79 ToadStool primitives, 39/39 three-tier controls. Proposes `barracuda::npu` (NPU inference bridge) and power-budget-aware metalForge dispatch. airSpring's NPU work (Exp 028-029b) directly aligns — we would be the reference implementation for agricultural IoT NPU dispatch.

**neuralSpring V24 (Feb 27)**: df64 core streaming for AlphaFold2 WGSL shaders. Proposes `compile_shader_df64_streaming` as first-class API and `barracuda::nn` module (MLP, LSTM, ESN classifiers). airSpring could use `barracuda::nn` for crop regime surrogates replacing our current FC classifiers.

**groundSpring V10 (Feb 25)**: Definitive handoff with wateringHole error-handling standard (`if let Ok` + always-compiled CPU fallback). Proposes `batched_multinomial.wgsl` for rarefaction and `mc_et0_propagate` for Monte Carlo ET₀ uncertainty. airSpring already uses `mc_et0_propagate` via CPU mirror; GPU path would be direct upgrade.

**ToadStool S61-63 (Feb 25)**: Sovereign compiler (naga-IR → SPIR-V passthrough), cyclic reduction for n≥2048, maximin LHS O(n). No API changes; improvements are automatic.

### Convergence Points

Three springs (airSpring, wetSpring, groundSpring) are all proposing NPU integration patterns to ToadStool. The convergence suggests `barracuda::npu` is high priority for the toadstool team:

| Spring | NPU Use Case | Hardware | Quantization |
|--------|-------------|----------|:------------:|
| **airSpring** | Crop stress, irrigation, anomaly | AKD1000 | int8 |
| **wetSpring** | PFAS screening, microbiome | AKD1000 | int8 |
| **groundSpring** | MC uncertainty classification | Proposed | int8 |
| **neuralSpring** | Protein structure inference | Proposed | int8 |

### Items Tracked for Future Absorption

| Primitive | Source | When Available | airSpring Impact |
|-----------|--------|---------------|------------------|
| `barracuda::npu::NpuDispatch` | Multi-spring proposal | Pending S69+ | Replace local `npu.rs` with upstream |
| `barracuda::nn::SimpleMLP` | neuralSpring V24 | Pending S69+ | Crop regime surrogates |
| `compile_shader_df64_streaming` | neuralSpring V24 | Pending S69+ | Simplify df64 GPU paths |
| `batched_multinomial.wgsl` | groundSpring V10 | Pending absorption | GPU rarefaction for diversity |
| Power-budget dispatch | wetSpring V61 | Design phase | Edge deployment routing |
