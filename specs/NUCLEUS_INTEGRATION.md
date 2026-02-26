# airSpring NUCLEUS Integration

**Date:** February 26, 2026
**Status:** Planning — deployment graph exists, workload definitions needed
**Gate:** Eastgate (i9-12900K, RTX 4070, 32GB DDR5, 2TB NVMe, BrainChip Akida NPU)

---

## Architecture

```
airSpring (Spring — validation consumer)
    │
    ├── control/ (Python baselines, download scripts)
    │       │
    │       └── NestGate providers (Open-Meteo, NOAA, USDA NASS)
    │               download → ZFS store with provenance
    │
    ├── barracuda/ (Rust validation, 608 cargo tests)
    │       │
    │       ├── eco:: modules (CPU validated, 464 lib tests)
    │       │
    │       └── gpu:: modules (ToadStool bridge, 11 Tier A)
    │               │
    │               └── capability.call("compute", ...) → ToadStool
    │
    └── metalForge/ (upstream absorption staging, 64 tests)
```

## Deployment Topology

### Step 1: Local NUCLEUS on Eastgate

Use the existing `nucleus_complete.toml` deployment graph:

```bash
biomeos deploy graphs/nucleus_complete.toml \
    --env FAMILY_ID=$(biomeos family-id) \
    --env NODE_ID=eastgate \
    --env XDG_RUNTIME_DIR=/run/user/1000
```

This starts: BearDog → Songbird → ToadStool → NestGate → Squirrel

### Step 2: Register airSpring Data Providers

NestGate data providers (created in `nestgate-core/src/data_sources/providers/live_providers/`):

| Provider | File | API Key | Data |
|----------|------|---------|------|
| Open-Meteo | `open_meteo_live_provider.rs` | None (free) | 80yr ERA5 weather |
| NOAA CDO | `noaa_cdo_live_provider.rs` | `NOAA_CDO_TOKEN` | GHCND observations |
| USDA NASS | `usda_nass_live_provider.rs` | `USDA_NASS_API_KEY` | County crop yields |

### Step 3: airSpring Workload Definitions

airSpring workloads map to ToadStool compute capabilities:

| Workload | ToadStool Op | GPU Shader | Scale |
|----------|-------------|-----------|-------|
| Batched ET₀ | `batched_elementwise_f64` (op=0) | `batched_et0.wgsl` | 12.7M/s |
| Batched Water Balance | `batched_elementwise_f64` (op=1) | `batched_wb.wgsl` | ~1M/s |
| Richards PDE | `pde::richards::solve_richards` | `crank_nicolson.wgsl` | 3,620/s |
| Kriging Interpolation | `kriging_f64::KrigingF64` | `kriging_f64.wgsl` | GPU batched |
| Isotherm Fitting | `optimize::nelder_mead` | `nelder_mead.wgsl` | 175K/s |
| Stream Smoothing | `moving_window_stats` | `moving_window.wgsl` | GPU batched |
| Seasonal Reduce | `fused_map_reduce_f64` | `fused_map_reduce.wgsl` | GPU N>=1024 |
| MC ET₀ Uncertainty | `norm_ppf` + `batched_elementwise` | `mc_et0.wgsl` | GPU batched |

### Step 4: Workload Submission Pattern

```rust
// airSpring submits compute workloads to local NUCLEUS via capability.call
use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64;

// Current: direct GPU dispatch (works without NUCLEUS)
let device = WgpuDevice::new()?;
let results = batched_elementwise_f64(&device, &inputs, op)?;

// Future: NUCLEUS-routed (routes to best available GPU in Plasmodium)
// capability.call("compute", "batched_elementwise_f64", { inputs, op })
```

airSpring's current GPU dispatch works without NUCLEUS. NUCLEUS adds:
- Workload routing across gates (Plasmodium)
- Data provenance via NestGate
- Discovery of available compute via Songbird
- Encrypted transport via BearDog

---

## LAN HPC (Plasmodium) — Step 5

When 10G cabling connects gates:

| Gate | Role | Hardware | airSpring Use |
|------|------|----------|---------------|
| **Eastgate** | Node + NPU | RTX 4070 + Akida | ET₀, water balance, scheduling |
| **Westgate** | Nest | 76TB ZFS + RTX 2070S | Data gravity (weather archive) |
| **Southgate** | Node | RTX 3090 + 128GB | Richards PDE at scale, kriging |
| **Strandgate** | Node | Dual EPYC + RTX 3090 | Large-scale bioinformatics |
| **Northgate** | Node | RTX 5090 + 192GB | LLM-assisted analysis |

Workload routing:
- ET₀ batch (2.9M calcs): Eastgate GPU (0.01 sec, local)
- Richards 80yr grid (29M sims): Southgate GPU (6 min, routed)
- Kriging 100 stations: Southgate GPU (20 min, routed)
- Weather data ingest: Westgate NestGate (data gravity)
- Cross-spring θ(t)→Anderson: Strandgate (wetSpring + airSpring)

---

## Prerequisites

| Prerequisite | Status | Blocker |
|-------------|--------|---------|
| biomeOS `nucleus_complete.toml` graph | Ready | None |
| BearDog binary | Built | None |
| Songbird binary | Built | None |
| ToadStool binary | Built | None |
| NestGate binary | Built | None |
| `.family.seed` on Eastgate | Needed | Generate once |
| NestGate weather providers | Created | Compile + test |
| 10G cables (LAN HPC) | Pending | Physical cabling |
| ZFS pool on Westgate | Existing | None (76TB online) |

---

## Environment Variables

```bash
# Required for NUCLEUS
export FAMILY_ID=$(biomeos family-id)
export NODE_ID=eastgate
export XDG_RUNTIME_DIR=/run/user/1000

# Optional for data providers
export NOAA_CDO_TOKEN=<token>           # NOAA CDO API
export USDA_NASS_API_KEY=<key>          # USDA NASS Quick Stats
export OPENWEATHERMAP_API_KEY=<key>     # OpenWeatherMap forecast (in testing-secrets/)
```

Open-Meteo requires no API key. USDA NASS registration is instant and free.

---

## Testing Strategy

1. **Without NUCLEUS**: `cargo test` + `cargo run --bin validate_*` (current, works now)
2. **With local NUCLEUS**: Same tests, but data flows through NestGate
3. **With Plasmodium**: Same tests, but GPU workloads route to best available gate
4. **Validation invariant**: All 608 Rust tests + 1354 atlas checks must pass regardless of deployment mode

The compute results are deterministic. NUCLEUS changes *where* the compute runs,
not *what* it computes. The 75/75 cross-validation match (tol=1e-5) is the
invariant across all deployment modes.
