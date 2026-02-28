# airSpring NUCLEUS Integration

**Date:** February 28, 2026
**Status:** Experimental — deployment graph + capability spec + Neural API bridge operational
**Gate:** Eastgate (i9-12900K, RTX 4070, 32GB DDR5, 2TB NVMe, BrainChip Akida NPU)
**Invariant:** 584 Rust lib tests + 46/46 GPU math + 75/75 cross-validation + 1393 atlas must pass regardless of deployment mode

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
    ├── barracuda/ (Rust validation, 511 lib tests, 42 binaries)
    │       │
    │       ├── eco:: modules (CPU validated, 14 domain modules)
    │       │
    │       ├── gpu:: modules (ToadStool bridge, 11 Tier A)
    │       │       │
    │       │       └── capability.call("compute", ...) → ToadStool
    │       │
    │       └── npu:: module (AKD1000 edge inference, feature-gated)
    │
    ├── metalForge/ (mixed hardware dispatch: CPU+GPU+NPU+Neural, 32 tests + 1 binary)
    │       │
    │       └── neural:: module (biomeOS capability.call over Unix socket)
    │
    └── graphs/ (biomeOS deployment graphs)
            ├── airspring_eco_pipeline.toml (ecological compute pipeline)
            └── cross_primal_soil_microbiome.toml (airSpring × wetSpring)
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

// Experimental: NUCLEUS-routed via metalForge Neural dispatch
// metalForge::neural::NeuralBridge discovers biomeOS at runtime
// and routes via capability.call to ToadStool through the Neural API
use airspring_forge::neural::NeuralBridge;
if let Some(bridge) = NeuralBridge::discover() {
    let result = bridge.capability_call("compute", "batched_elementwise_f64", &args)?;
}
```

airSpring's current GPU dispatch works without NUCLEUS. NUCLEUS adds:
- Workload routing across gates (Plasmodium)
- Data provenance via NestGate
- Discovery of available compute via Songbird
- Encrypted transport via BearDog

### Step 4b: Ecology Capabilities (Exp 036)

airSpring registers as an **ecology** domain provider (see `specs/BIOMEOS_CAPABILITIES.md`):

```toml
[domains.ecology]
provider = "airspring"
capabilities = ["ecology", "irrigation", "soil_moisture", "evapotranspiration", "crop_science"]
```

20+ capability translations map semantic names to airSpring methods:
`ecology.et0_pm`, `ecology.water_balance`, `ecology.yield_response`, etc.

Cross-primal interactions via biomeOS:
- airSpring → wetSpring: `ecology.water_balance(θ)` → `science.diversity(moisture_series)`
- groundSpring → airSpring: `measurement.error_propagation` → `ecology.et0_sensitivity`
- wetSpring → airSpring: `science.soil_microbiome_coupling` → `ecology.water_balance`

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
3. **With Neural API**: `validate_neural_api` confirms JSON round-trip parity (29/29 PASS)
4. **With Plasmodium**: Same tests, but GPU workloads route to best available gate
5. **Validation invariant**: All 584 Rust lib tests + 46/46 GPU math portability + 1393 atlas checks must pass regardless of deployment mode

The compute results are deterministic. NUCLEUS changes *where* the compute runs,
not *what* it computes. The 75/75 cross-validation match (tol=1e-5) is the
invariant across all deployment modes.

---

## Quick Start: Eastgate Tower Deployment

### Phase A — Atomic Tower (BearDog + Songbird)

```bash
# 1. Build biomeOS
cd /home/eastgate/Development/ecoPrimals/phase2/biomeOS
cargo build --workspace --release

# 2. Create family seed (one-time)
head -c 32 /dev/urandom > .family.seed
chmod 600 .family.seed

# 3. Start tower
export NODE_ID=eastgate
export XDG_RUNTIME_DIR=/run/user/$(id -u)
biomeos nucleus --mode tower --node-id eastgate

# 4. Verify: airSpring tests still pass (math is deployment-invariant)
cd /home/eastgate/Development/ecoPrimals/airSpring/barracuda
cargo test --release
cargo run --release --bin validate_gpu_math
```

### Phase B — Node (add ToadStool compute)

```bash
biomeos nucleus --mode node --node-id eastgate
```

airSpring GPU workloads (18 eco-domain) route through ToadStool instead of direct
`wgpu`. The `metalForge::neural::NeuralBridge` discovers the socket automatically:

```
$XDG_RUNTIME_DIR/biomeos/neural-api-<family_id>.sock
```

### Phase C — Nest (add NestGate storage)

```bash
# Eastgate (local compute + storage)
biomeos nucleus --mode nest --node-id eastgate

# Later: Westgate (76TB ZFS data gravity)
biomeos nucleus --mode nest --node-id westgate
```

NestGate providers download and store with content-addressed provenance (BLAKE3):
- Open-Meteo: 80yr ERA5, no API key
- NOAA CDO: GHCND observations, `NOAA_CDO_TOKEN`
- USDA NASS: County crop yields, `USDA_NASS_API_KEY`
- NCBI: 16S + metagenomes, no key for E-utilities

### Phase D — Plasmodium (LAN HPC, after 10G cabling)

See "LAN HPC (Plasmodium) — Step 5" above.
