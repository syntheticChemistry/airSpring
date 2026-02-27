# metalForge — airSpring Cross-System Compute Dispatch

> **ACTIVE**: Cross-system dispatch (CPU + GPU + NPU + biomeOS Neural API).
> Original 6 domain modules were absorbed into `barracuda` (ToadStool S64–S66).
> The forge crate is now the **dispatch + integration layer**: substrate
> discovery, capability-based routing, and biomeOS Neural API bridge.

**Date**: February 27, 2026
**Crate**: `airspring-forge` v0.1.0 (dispatch layer: 32 tests + 1 binary)
**License**: AGPL-3.0-or-later

---

## Philosophy

metalForge serves two purposes:

1. **Upstream staging**: Domain primitives validated locally, then handed off
   to barracuda/ToadStool for absorption (Phase 1 — complete).
2. **Cross-system dispatch**: Runtime substrate discovery and capability-based
   routing across CPU, GPU, NPU, and biomeOS Neural API (Phase 2 — active).

```
Write locally → Validate → Hand off → Absorb → Lean on upstream → Dispatch across systems
```

**Phase 1 Status: COMPLETE.** All 6 domain modules absorbed upstream.
**Phase 2 Status: ACTIVE.** Substrate probe, dispatch routing, and Neural API bridge operational.

## What's Here

### `forge/` — Rust crate (`airspring-forge`)

Active dispatch layer + absorbed domain modules. 32/32 tests pass, zero clippy warnings.

#### Active Modules (dispatch + integration)

| Module | Purpose | Status |
|--------|---------|--------|
| `substrate` | Substrate abstraction (GPU, NPU, CPU, Neural) | **Active** |
| `probe` | Runtime hardware discovery (wgpu, procfs, devfs) | **Active** |
| `inventory` | Unified device inventory (all substrate types) | **Active** |
| `dispatch` | Capability-based routing (GPU > NPU > Neural > CPU) | **Active** |
| `workloads` | airSpring workload definitions (ET₀, WB, Richards, NPU) | **Active** |
| `neural` | biomeOS Neural API bridge (`capability.call` over Unix socket) | **Experimental** |

#### Absorbed Modules (fossil record, validated upstream)

| Module | Functions | Absorbed Into | When |
|--------|-----------|--------------|------|
| `metrics` | `rmse`, `mbe`, `nash_sutcliffe`, `index_of_agreement`, `coefficient_of_determination` | `barracuda::stats::metrics` | **S64** |
| `regression` | `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `fit_all` + `FitResult::predict()` | `barracuda::stats::regression` | **S66** (R-S66-001) |
| `moving_window_f64` | `moving_window_stats` (mean, variance, min, max) | `barracuda::stats::moving_window_f64` | **S66** (R-S66-003) |
| `hydrology` | `hargreaves_et0`, `hargreaves_et0_batch`, `crop_coefficient`, `soil_water_balance` | `barracuda::stats::hydrology` | **S66** (R-S66-002) |
| `van_genuchten` | VG retention, conductivity, capacity | `barracuda::pde::richards::SoilParams` | **S40+S66** |
| `isotherm` | Langmuir/Freundlich linearized fits | `barracuda::eco::isotherm` (via NM) | **S64** |

### Provenance

All implementations are validated against published benchmarks:

- **Metrics**: Dong et al. (2020) soil sensor calibration (36/36), FAO-56
  real data pipeline (918 station-days, R²=0.967), 65/65 Python-Rust
  cross-validation
- **Regression**: Dong et al. (2020) four-model correction suite, validated
  against scipy `curve_fit` outputs.  `FitResult::predict()` follows the
  `RidgeResult::predict()` pattern from `barracuda::linalg::ridge`
- **Moving window f64**: CPU f64 complement to upstream f32 GPU path
  (wetSpring S28+ `moving_window.wgsl`)
- **Hydrology**: Hargreaves & Samani (1985), FAO-56 (Allen et al. 1998),
  918 station-days, cross-validated with Python ETo library

## biomeOS Neural API Integration (Exp 036)

The `neural` module provides a minimal JSON-RPC 2.0 client that talks to
biomeOS's Neural API over Unix sockets. Zero external async dependencies —
uses `std::os::unix::net::UnixStream` for synchronous communication.

- **Discovery**: 4-tier socket resolution (env → XDG → /run → /tmp)
- **Interface**: `capability.call(domain, operation, args)` → JSON response
- **Substrate**: `SubstrateKind::Neural` sits between NPU and CPU in dispatch priority
- **Parity**: Exp 036 validates JSON round-trip introduces zero numerical drift (29/29 PASS)

See `specs/BIOMEOS_CAPABILITIES.md` for the full ecology capability registry.

## Relationship to hotSpring's metalForge

| | hotSpring metalForge | airSpring metalForge |
|--|----------------------|----------------------|
| **Focus** | Hardware characterization, substrate discovery, capability dispatch | Cross-system dispatch + biomeOS Neural API bridge |
| **Upstream target** | `barracuda::device::unified` | `barracuda::stats::*`, Neural API |
| **Crate** | `hotspring-forge` | `airspring-forge` |
| **Dependencies** | barracuda, wgpu, tokio | wgpu, serde_json |
| **Modules** | substrate, probe, inventory, dispatch, bridge | substrate, probe, inventory, dispatch, workloads, neural |
| **Tests** | Hardware probing, bridge seam | 32 tests (31 unit + 1 doc), dispatch + neural |

## Cross-Spring Absorption Candidates

These airSpring patterns may benefit other springs:

| Pattern | Used by | Potential |
|---------|---------|-----------|
| RMSE / MBE / NSE / IA | airSpring, groundSpring | Universal validation metrics |
| Analytical curve fitting + `predict()` | airSpring (soil calibration) | Any domain with empirical fits |
| CPU f64 moving window | airSpring (IoT sensor streams) | Complement to GPU f32 path |
| Hargreaves ET₀ + Kc interpolation | airSpring (precision agriculture) | Environmental modeling |
| Soil water balance | airSpring (irrigation scheduling) | Hydrology domains |
| `ValidationRunner` pattern | airSpring | Shared validation harness |
| `len_f64()` utility | All springs | Already in barracuda candidates |

## Quality

```
cargo fmt   — clean
cargo clippy --all-targets — zero warnings (pedantic + nursery)
cargo test  — 32/32 pass (7 dispatch + 5 neural + 5 probe + 5 substrate + 5 workloads + 2 inventory + 1 doc + 2 bin)
unsafe code — 0 (uid discovery via /proc/self/status, not libc::getuid)
```
