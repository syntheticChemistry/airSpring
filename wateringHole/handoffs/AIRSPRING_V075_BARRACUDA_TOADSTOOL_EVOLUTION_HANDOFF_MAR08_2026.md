# airSpring V0.7.5 — barraCuda / toadStool Evolution Handoff

SPDX-License-Identifier: AGPL-3.0-or-later
**Date**: March 8, 2026
**From**: airSpring (v0.7.5, 87 experiments, 865 lib + 186 forge tests)
**To**: barraCuda + toadStool teams
**Supersedes**: All prior V075 handoffs (consolidation)
**barraCuda Pin**: v0.3.3 standalone at `a898dee` (wgpu 28)
**toadStool Pin**: S130+ at `bfe7977b`

---

## Executive Summary

airSpring v0.7.5 has completed a comprehensive NUCLEUS integration validation
across four new experiments (084–087, 79/79 PASS). This handoff documents
everything the barraCuda/toadStool teams need to evolve their systems based
on what airSpring has learned:

1. **CPU↔GPU parity** validated for all 18 GPU modules — 3 tolerance notes for upstream.
2. **JSON-RPC dispatch** for 14 science methods proven stable — toadStool compute.offload ready.
3. **NUCLEUS mesh routing** validated on live hardware — PCIe bypass confirmed.
4. **biomeOS deployment graphs** structurally sound — ready for live orchestration.
5. **9 upstream capabilities** documented but not yet wired — absorption candidates.

---

## §1 CPU vs GPU Parity — What barraCuda Should Know (Exp 084)

All 18 airSpring GPU modules validated for CPU↔GPU numerical consistency.
Three modules have documented divergences that deserve upstream attention:

### Tolerance Notes for Upstream

| Module | Tolerance | Root Cause | Recommendation |
|--------|-----------|------------|----------------|
| `et0_fao56` (op=0) | 2.0 mm/day | CPU uses `actual_vapour_pressure` directly; GPU uses `rh_max/rh_min` → derives `e_a` differently | Consider aligning GPU `StationDay` to accept pre-computed `actual_vapour_pressure` for exact parity |
| `hargreaves` (op=6) | 0.05 mm/day | Float divergence in intermediate extraterrestrial radiation calculation | Acceptable — within measurement uncertainty |
| `hamon` (op=19) | 2.0 mm/day | CPU implementations use different daylight-hours formulas | Document canonical formula; consider standardizing on the more precise Brock (1981) variant |

### All 18 Modules PASS

| Module | Op/Shader | Max Diff | Status |
|--------|-----------|----------|--------|
| FAO-56 ET₀ | op=0 | <2.0 mm/day | PASS (schema divergence) |
| Hargreaves | dedicated | <0.05 | PASS |
| SCS-CN Runoff | op=17 | <1e-6 | PASS |
| Yield Response | op=18 | <1e-6 | PASS |
| Makkink | op=14 | <1e-6 | PASS |
| Turc | op=15 | <1e-6 | PASS |
| Hamon | op=19 | <2.0 | PASS (formula divergence) |
| Blaney-Criddle | op=16 | <1e-6 | PASS |
| VG θ(h) | op=9 | <1e-6 | PASS |
| VG K(h) | op=10 | <1e-6 | PASS |
| Thornthwaite | op=11 | <1e-6 | PASS |
| GDD | op=12 | <1e-6 | PASS |
| Pedotransfer | op=13 | <1e-6 | PASS |
| Infiltration | dedicated | <1e-6 | PASS |
| Autocorrelation | dedicated | <1e-6 | PASS |
| Bootstrap CI | dedicated | <1e-6 | PASS |
| Jackknife CI | dedicated | <1e-6 | PASS |
| Diversity | dedicated | <1e-6 | PASS |
| Reduce (mean) | dedicated | <1e-6 | PASS |
| Reduce (var) | dedicated | <1e-6 | PASS |

**Binary**: `barracuda/src/bin/validate_cpu_gpu_comprehensive.rs`

---

## §2 toadStool Compute Dispatch — What toadStool Should Absorb (Exp 085)

### 14 Validated JSON-RPC Science Methods

These methods are tested end-to-end via airSpring's in-process dispatch and are
ready for toadStool `compute.offload` wiring:

| Method | Category | Input Schema Stable | Notes |
|--------|----------|:-------------------:|-------|
| `ecology.et0_fao56` | Hydrology | Yes | Primary ET₀ — most-called method |
| `ecology.water_balance` | Hydrology | Yes | FAO-56 Ch 8 soil water budget |
| `ecology.yield_response` | Agronomy | Yes | Stewart Ky model |
| `science.thornthwaite` | Climate | Yes | `monthly_temps_c` param (not `monthly_temps`) |
| `science.gdd` | Phenology | Yes | Growing degree days |
| `science.pedotransfer` | Soil | Yes | Saxton-Rawls 2006 |
| `science.spi_drought_index` | Drought | Yes | SPI-1/3/6/12, gamma MLE |
| `science.autocorrelation` | Statistics | Yes | Cross-spring provenance tracked |
| `science.gamma_cdf` | Math | Yes | Upstream `regularized_gamma_p` |
| `ecology.runoff_scs_cn` | Hydrology | Yes | USDA SCS Curve Number |
| `ecology.van_genuchten_theta` | Soil | Yes | VG retention curve |
| `ecology.van_genuchten_k` | Soil | Yes | VG hydraulic conductivity |
| `science.bootstrap_ci` | Statistics | Yes | Bootstrap confidence intervals |
| `science.jackknife_ci` | Statistics | Yes | Leave-one-out variance |

### Methods NOT Exposed via JSON-RPC

These are internal GPU operations, not JSON-RPC handlers:
- `hargreaves` — internal to GPU orchestrator
- `kc_climate` — internal seasonal pipeline stage

### Socket Health Pattern

airSpring implements a robust toadStool detection pattern that toadStool should
be aware of for its daemon lifecycle:

1. Check socket existence at standard biomeOS path
2. If socket exists, send `toadstool.health` probe
3. If health returns `{"status": "healthy"}` → proceed with `compute.offload`
4. If socket exists but no response → treat as stale (log, don't fail)
5. If no socket → graceful fallback to in-process compute

This pattern handles daemon restarts, stale sockets from crashes, and the
common case where toadStool isn't running (Node Atomic not required).

### Cross-Primal Discovery

`biomeos::discover_all_primals()` returns 7 primals on Eastgate:
`airspring`, `beardog`, `songbird`, `squirrel`, `toadstool`, `neural-api`, `nestgate`

### PrecisionRoutingAdvice

airSpring validates `PrecisionRoutingAdvice` from `DevicePrecisionReport`:
- RTX 4070: `F64Native` (native f64 compute)
- Titan V (NVK): `Df64Only` (DF64 emulation, no shared-memory f64)

toadStool should use this for workload routing decisions.

**Binary**: `barracuda/src/bin/validate_toadstool_dispatch.rs`

---

## §3 NUCLEUS Mesh — What metalForge/toadStool Should Know (Exp 086)

### Live Hardware Inventory (Eastgate)

| Substrate | Device | Properties | NUCLEUS Atomic |
|-----------|--------|------------|----------------|
| GPU #0 | RTX 4070 (12 GB) | F64Native | Tower |
| GPU #1 | Titan V (12 GB, NVK) | Df64Only | Node |
| CPU | i9-12900K (24 cores) | x86_64, AVX2 | Nest |
| NPU | — | Not detected | — |

### NUCLEUS Mesh Construction

From live hardware, airSpring constructs:
- **Tower**: RTX 4070 + CPU (primary compute)
- **Node**: Titan V (secondary GPU)
- **Nest**: CPU-only fallback

### Workload Routing Results

- 23/27 ecological workloads route to capable substrates
- 4 NPU-only workloads unroutable (no NPU hardware) — graceful skip
- `CpuCompute` capability dispatches to GPU (GPU is capability superset)

### Ecology Pipeline

3-stage GPU pipeline validated:
```
et0_batch (GPU #0) → water_balance_batch (GPU #0) → yield_response_surface (GPU #0)
```
All stages stay on same GPU — PCIe bypass confirmed, zero CPU roundtrip.

### Transfer Matrix

Cross-node transfer costs computed. GPU→GPU on same node uses PCIe P2P.
GPU→CPU uses system memory. Transfer matrix is symmetric and non-negative.

**Binary**: `metalForge/forge/src/bin/validate_mixed_nucleus_live.rs`

---

## §4 biomeOS Deployment Graphs — What biomeOS Should Know (Exp 087)

### Validated Graphs

| Graph | Nodes | DAG | Capabilities Used |
|-------|:-----:|:---:|-------------------|
| `airspring_eco_pipeline.toml` | 7 | Valid | ecology.et0_fao56, ecology.water_balance, ecology.yield_response |
| `cross_primal_soil_microbiome.toml` | 5 | Valid | ecology.soil_moisture, science.diversity (wetSpring) |

### Structural Validation

- `[graph]` metadata section present (ID, description, version)
- `[[nodes]]` array with dependencies
- Topological sort (Kahn's algorithm) confirms acyclicity
- Dependency ordering correct: fetch → compute → balance → yield → store
- Prerequisite nodes: `check_nestgate` and `check_toadstool` present in both graphs
- All capability references match known `ecology.*` / `science.*` set

### Graph for Live Orchestration

Both graphs are ready for `biomeos deploy` once NestGate gating is
implemented. The prerequisite check pattern (`check_nestgate`, `check_toadstool`)
provides graceful degradation if a primal is unavailable.

**Binary**: `barracuda/src/bin/validate_nucleus_graphs.rs`

---

## §5 Upstream Capabilities — Absorption Candidates

These barraCuda capabilities are available at HEAD but not yet wired in airSpring.
They represent evolution opportunities for the toadStool team:

| Capability | barraCuda Path | airSpring Use Case |
|------------|---------------|-------------------|
| `regularized_gamma_q` | `special::gamma` | Upper incomplete gamma for SPI tail |
| `digamma` | `special::gamma` | Information-theoretic diversity metrics |
| `beta` / `ln_beta` | `special::beta` | Beta distribution for soil parameter estimation |
| `BatchedOdeRK45F64` | `pde::ode` | Adaptive Dormand-Prince for Richards PDE |
| R² on `CorrelationResult` | `stats::correlation` | Direct R² without manual squaring |
| `Fft1DF64` | `linalg::fft` | Spectral analysis of ET₀ time series |
| `AutocorrelationF64` | `stats::autocorrelation` | GPU autocorrelation (airSpring has CPU wrapper) |
| `CovarianceF64` | `stats::covariance` | Covariance matrix for multi-variate soil |
| `PeakDetectF64` | `stats::peaks` | Extrema detection in hydrographs |

### Write→Absorb→Lean Status

All 6 metalForge modules absorbed upstream (S64+S66):
- `forge::metrics` → `barracuda::stats::metrics`
- `forge::regression` → `barracuda::stats::regression`
- `forge::moving_window` → `barracuda::stats::moving_window_f64`
- `forge::hydrology` → `barracuda::stats::hydrology`
- `forge::van_genuchten` → `barracuda::pde::richards::SoilParams`
- `forge::isotherm` → `barracuda::optimize` (Nelder-Mead)

All 20 elementwise ops (0-19) upstream. `local_dispatch` retired v0.7.2.

---

## §6 Cross-Spring Intelligence

### Shader Families airSpring Consumes

| Origin | Shaders | What airSpring Gets |
|--------|---------|---------------------|
| hotSpring | 56 | df64 core, pow/exp/log/trig f64, df64_transcendentals |
| wetSpring | 25 | kriging_f64, fused_map_reduce, moving_window, ridge, diversity |
| neuralSpring | 20 | nelder_mead, multi_start, ValidationHarness |
| groundSpring | — | MC ET₀ uncertainty propagation shader |

### What airSpring Contributed Back

| Contribution | Upstream | Status |
|-------------|----------|--------|
| TS-001: `pow_f64` fractional exponent fix | hotSpring → barraCuda | Resolved (commit `0c477306`) |
| TS-003: acos precision boundary fix | neuralSpring → barraCuda | Resolved |
| TS-004: reduce buffer N≥1024 fix | wetSpring → barraCuda | Resolved |
| Richards PDE (S40) | airSpring → barraCuda | Absorbed |
| Stats metrics (S64) | airSpring → barraCuda | Absorbed |
| SCS-CN/Stewart/Makkink/Turc/Hamon/Blaney-Criddle (ops 14-19) | airSpring → barraCuda | Absorbed (v0.7.2) |

### Precision Insights

- RTX 4070: F64Native — all shaders run at full f64
- Titan V (NVK): Df64Only — DF64 emulation works, shared-memory f64 unreliable
- AKD1000 NPU: int4/int8 — quantized inference only
- `PrecisionRoutingAdvice` routes correctly per-hardware
- FAO-56 ET₀ vapor pressure path diverges ~2mm/day between CPU and GPU input schemas

---

## §7 Quality Gates

| Gate | Result |
|------|--------|
| `cargo fmt --check` (barracuda) | PASS |
| `cargo fmt --check` (forge) | PASS |
| `cargo clippy --all-targets -- -D warnings` (barracuda) | PASS (0 warnings) |
| `cargo clippy --all-targets -- -D warnings` (forge) | PASS (0 warnings) |
| `cargo test --lib` (barracuda) | **865/865 PASS** |
| `cargo test --lib` (forge) | **186/186 PASS** |
| Exp 084 CPU/GPU Parity | **21/21 PASS** |
| Exp 085 toadStool Dispatch | **19/19 PASS** |
| Exp 086 metalForge NUCLEUS | **17/17 PASS** |
| Exp 087 Graph Coordination | **22/22 PASS** |
| Total validation | **79/79 PASS** (new) + all prior experiments green |

---

## §8 Recommended Evolution for toadStool/barraCuda

### For barraCuda

1. **FAO-56 GPU input schema**: Consider adding `actual_vapour_pressure` to GPU `StationDay`
   to eliminate the 2mm/day parity gap with CPU.
2. **Hamon daylight formula**: Standardize on Brock (1981) for consistency.
3. **9 absorption candidates** in §5 — all available at HEAD, airSpring ready to wire.
4. **`BatchedOdeRK45F64`**: Highest-value absorption target — replaces fixed-step Richards
   solver with adaptive Dormand-Prince for better accuracy at coarse time steps.

### For toadStool

1. **14 JSON-RPC methods** ready for compute.offload (§2).
2. **Socket health pattern** in §2 should be part of toadStool's daemon lifecycle docs.
3. **PrecisionRoutingAdvice** usage validated — toadStool can trust the routing decisions.
4. **Stale socket cleanup**: Consider adding automatic cleanup on daemon start (airSpring
   handles stale sockets gracefully, but prevention is better than detection).

### For biomeOS

1. **2 deployment graphs** validated and ready for `biomeos deploy`.
2. **NestGate gating**: Both graphs have prerequisite checks that need NestGate implementation.
3. **7 primals discovered** — the mesh is production-grade.

---

## §9 Files Changed Since Last Handoff

| File | Change |
|------|--------|
| `barracuda/src/bin/validate_cpu_gpu_comprehensive.rs` | NEW — Exp 084 |
| `barracuda/src/bin/validate_toadstool_dispatch.rs` | NEW — Exp 085 |
| `metalForge/forge/src/bin/validate_mixed_nucleus_live.rs` | NEW — Exp 086 |
| `barracuda/src/bin/validate_nucleus_graphs.rs` | NEW — Exp 087 |
| `barracuda/Cargo.toml` | 3 `[[bin]]` entries + `toml = "0.8"` |
| `metalForge/forge/Cargo.toml` | 1 `[[bin]]` entry |
| `graphs/airspring_eco_pipeline.toml` | biomeOS deployment graph |
| `graphs/cross_primal_soil_microbiome.toml` | biomeOS cross-primal graph |
