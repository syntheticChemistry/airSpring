# Context — airSpring

**Doc sync:** 2026-03-24 (v0.10.0; post–deep-audit execution; all CI gates green).

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

## Technical Facts

- **Language:** 100% Rust, zero C dependencies
- **Architecture:** Two workspace crates (`airspring-barracuda` library + `airspring-forge` dispatch)
- **Communication:** JSON-RPC 2.0 over Unix sockets + TCP (biomeOS capability routing, platform-agnostic Transport)
- **License:** AGPL-3.0-or-later
- **Lib tests:** 986 (barracuda, `cargo test --lib`)
- **Integration + doc tests:** 316 (barracuda)
- **Barracuda total:** 1,302 (986 lib + 316 integration/doc)
- **Forge tests:** 62 (metalForge)
- **Grand total:** 1,364 (both crates)
- **Binaries:** 91 (84 validation, 4 bench, 3 operational)
- **Proptest invariants:** 7 (SVP, delta, Hargreaves, TAW, RAW, Ks)
- **Line coverage:** 90.56% (cargo llvm-cov --lib --fail-under-lines 90)
- **MSRV:** 1.92
- **Edition:** 2024
- **Crate count:** 2 workspace crates
- **GPU backend:** barraCuda 0.3.7 (wgpu 28, Vulkan, DeviceCapabilities API)
- **Experiments:** 87 (all PASS)

## Key Capabilities (JSON-RPC methods)

- `eco.daily_et0` — FAO-56 Penman-Monteith reference evapotranspiration
- `eco.et0_multi_method` — 8-method ET₀ ensemble (PM, Hargreaves, Priestley-Taylor, Makkink, Turc, Hamon, Blaney-Criddle, Thornthwaite)
- `eco.water_balance_season` — Full-season field water budget
- `eco.richards_1d` — Unsaturated flow (Richards equation)
- `eco.soil_calibration` — Dielectric sensor VWC calibration
- `eco.crop_coefficient` — FAO-56 Kc with climate adjustment
- `eco.diversity_indices` — Shannon, Simpson, Bray-Curtis biodiversity
- `eco.drought_index` — Standardized Precipitation Index (SPI)

## What This Does NOT Do

- Does not compile GPU shaders (that is coralReef)
- Does not manage hardware discovery or process orchestration (that is toadStool)
- Does not handle cryptography, networking, or storage (those are BearDog, Songbird, NestGate)
- Does not provide ML training — uses bingoCube/nautilus for evolutionary reservoir computing only

## Related Repositories

- [wateringHole](https://github.com/ecoPrimals/wateringHole) — ecosystem standards and registry
- [barraCuda](https://github.com/ecoPrimals/barraCuda) — GPU math library (800+ WGSL shaders)
- [toadStool](https://github.com/ecoPrimals/toadStool) — hardware discovery and compute orchestration
- [bingoCube](https://github.com/ecoPrimals/primalTools/bingoCube) — evolutionary reservoir computing

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
