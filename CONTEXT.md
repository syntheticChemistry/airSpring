# Context — airSpring

**Doc sync:** 2026-05-12 (v0.10.0; eukaryotic UniBin; post-interstadial; all CI gates green).

## What This Is

airSpring is a pure Rust ecological and agricultural science validation spring.
It reproduces published results from FAO-56 (irrigation), HYDRUS (soil physics),
USDA/NASS (crop yield), and field sensor research — then validates that Rust
implementations match Python baselines to documented tolerances. It is part of
the ecoPrimals sovereign computing ecosystem: a collection of self-contained
binaries that coordinate via JSON-RPC 2.0 over Unix sockets and TCP, with zero
compile-time coupling between components.

## Role in the Ecosystem

airSpring validates *systems* — agricultural fields, soil-plant-atmosphere
continua, irrigation networks. Where hotSpring validates nuclear physics and
wetSpring validates analytical/life science, airSpring covers precision
agriculture, environmental hydrology, and land-water-energy interactions.
Its validated Rust modules feed into GPU acceleration via barraCuda shaders
and mixed-hardware dispatch via metalForge (CPU + GPU + NPU).

## Architecture

- **Eukaryotic UniBin** — single `airspring` binary with `certify`, `validate`,
  `serve`, `status`, `version` subcommands. Pre-extinction experiment crates
  fossilized in `fossilRecord/`.
- **Certification organelle** — `certification/` library module (**L0–L6** layered
  guidestone validation: bare → discovery → health → capability parity → cross-atomic
  pipeline → NUCLEUS composition → cross-spring pipeline; absorbed from standalone binary).
- **Scenario registry** — `validation/scenarios/` modules (**10 UniBin validation
  scenarios**, absorbed from composition experiments + expanded registry coverage).

## Technical Facts

- **Language:** 100% Rust, zero C dependencies
- **Architecture:** Two workspace crates (`airspring-barracuda` library + `airspring-forge` dispatch)
- **Communication:** JSON-RPC 2.0 over Unix sockets + TCP (biomeOS capability routing, Songbird sovereign transport)
- **License:** AGPL-3.0-or-later
- **Lib tests:** 1,011 (barracuda, `cargo test --features local,testutil --lib`)
- **Integration + doc tests:** 316 (barracuda)
- **Barracuda total:** 1,327 (1,011 lib + 316 integration/doc)
- **Forge tests:** 62 (metalForge)
- **Grand total:** 1,389 (both crates)
- **Binaries:** 94 (85 validation, 4 bench, 3 operational, 1 UniBin, 1 guidestone)
- **Proptest invariants:** 7 (SVP, delta, Hargreaves, TAW, RAW, Ks)
- **Line coverage:** 90.56% (cargo llvm-cov --lib --fail-under-lines 90)
- **MSRV:** 1.92
- **Edition:** 2024
- **Crate count:** 2 workspace crates
- **GPU backend:** barraCuda 0.3.13 (wgpu 28, Vulkan, DeviceCapabilities API)
- **Experiments:** 90 (all PASS)
- **Capabilities:** 46 (science + ecology + provenance + composition + coordination + health)
- **Deploy graphs:** 7 (eco + provenance + niche + cross-primal + GPU batch + sovereign data + uncertainty)
- **GuideStone level:** L4 (targeting L6 with live NUCLEUS; **10 UniBin validation scenarios**)
- **Tier 4 IPC-first:** `[features].default = []` (was `["local", "testutil"]`). Opt in with **`--features local`** for in-tree barraCuda + GPU; **`testutil`** extends `local` for dev helpers. All validation binaries use **`required-features = ["local"]`**. `gpu` is feature-gated; `math.rs` dual-path dispatch; `ipc/barracuda_route.rs` IPC forwarding. Default feature set builds without linking barraCuda.
- **deny.toml:** workspace-root, `aws-lc-sys` + `aws-lc-rs` banned

## Key Capabilities (JSON-RPC methods)

46 methods registered in `capability_registry.toml`:

- **Evapotranspiration (7):** `science.et0_fao56`, `science.et0_hargreaves`,
  `science.et0_priestley_taylor`, `science.et0_makkink`, `science.et0_turc`,
  `science.et0_hamon`, `science.et0_blaney_criddle`
- **Water/yield (2):** `science.water_balance`, `science.yield_response`
- **Soil physics (5):** `science.richards_1d`, `science.scs_cn_runoff`,
  `science.green_ampt_infiltration`, `science.soil_moisture_topp`,
  `science.pedotransfer_saxton_rawls`
- **Crop/irrigation (3):** `science.dual_kc`, `science.sensor_calibration`, `science.gdd`
- **Biodiversity (2):** `science.shannon_diversity`, `science.bray_curtis`
- **Geophysics (1):** `science.anderson_coupling`
- **Monthly ET/drought (4):** `science.thornthwaite`, `science.spi_drought_index`,
  `science.autocorrelation`, `science.gamma_cdf`
- **Time series (1):** `science.timeseries`
- **Ecology aliases (7):** `ecology.et0_fao56`, `ecology.et0_hargreaves`,
  `ecology.water_balance`, `ecology.yield_response`, `ecology.full_pipeline`,
  `ecology.spi_drought_index`, `ecology.autocorrelation`
- **Provenance (4):** `provenance.begin`, `provenance.record`,
  `provenance.complete`, `provenance.status`
- **Composition (1):** `composition.status`
- **Infrastructure (7):** `health.liveness`, `health.readiness`,
  `capability.list`, `method.register`, `compute.offload`, `data.weather`, `data.cross_spring_weather`
- **Cross-primal (2):** `primal.forward`, `primal.discover`

## What This Does NOT Do

- Does not compile GPU shaders (that is coralReef)
- Does not manage hardware discovery or process orchestration (that is toadStool)
- Does not handle cryptography, networking, or storage (those are BearDog, Songbird, NestGate)
- Does not provide ML training — uses bingoCube/nautilus for evolutionary reservoir computing only

## Primal Dependencies (runtime, zero compile-time coupling)

| Primal | Role | Discovery |
|--------|------|-----------|
| **barraCuda** | GPU math primitives (ops 0-19, PDE, optimize, stats) | `barracuda` crate (optional path dep; Tier 4 IPC-first: enable `local`; pure-Rust fallbacks when absent) |
| **biomeOS** | Orchestration, socket resolution, primal discovery | `biomeos::find_socket()` |
| **toadStool** | Hardware discovery, compute dispatch | `compute.offload` IPC |
| **bearDog** | TLS, key management | Sovereign TLS for transport |
| **songbird** | NAT traversal, sovereign HTTP relay | `SongbirdTransport` IPC |
| **nestGate** | Data routing, weather data provider | `data.weather` IPC |
| **coralReef** | Shader compilation (WGSL → SPIR-V) | Discovery only |
| **squirrel** | AI coordination | Discovery only |
| **sweetGrass** | Provenance braiding | `provenance.*` IPC |
| **rhizoCrypt** | DAG storage | Via provenance trio |
| **loamSpine** | Configuration | Via biomeOS |

## Related Repositories

- [wateringHole](https://github.com/ecoPrimals/wateringHole) — ecosystem standards and registry
- [barraCuda](https://github.com/ecoPrimals/barraCuda) — GPU math library (767+ WGSL shaders)
- [toadStool](https://github.com/ecoPrimals/toadStool) — hardware discovery and compute orchestration
- [bingoCube](https://github.com/ecoPrimals/primalTools/bingoCube) — evolutionary reservoir computing
- [projectNUCLEUS](https://github.com/sporeGarden/projectNUCLEUS) — deployable NUCLEUS infrastructure
- [foundation](https://github.com/sporeGarden/foundation) — scientific knowledge layer

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
