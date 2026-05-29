+++
title = "airSpring Validation Summary"
description = "Precision agriculture & irrigation — 1,442 Rust tests, 90 experiments, 10 UniBin validation scenarios, 14.3× speedup, 57 registered / 46 live capabilities, eastGate operational, zero failures"
date = 2026-05-25

[taxonomies]
primals = ["barracuda", "toadstool", "biomeos", "nestgate", "squirrel", "coralreef", "petaltongue", "beardog", "songbird"]
springs = ["airspring", "hotspring", "wetspring", "neuralspring", "groundspring"]
+++

## Status

- **1,442 Rust tests** passing (1,057 lib + 316 integration + 69 forge), 0 failed
- **1,284 Python baseline checks** (60 papers reproduced)
- **90 experiments** across 12 categories + 3 composition crates (exp001 local parity, exp002 composition parity, exp003 foundation targets)
- **14.3× geometric mean** Rust-vs-Python speedup (25/25 algorithms, 21/21 CPU-GPU parity)
- **57 registered / 46 live IPC capabilities** (science + 13 ecology aliases + provenance + composition + coordination + inference)
- **64 centralized method constants** in `methods.rs` (drift-proof, single source of truth)
- **98 validation binaries** (all zero-panic, OrExit pattern, UniBin consolidation)
- **90.56% line coverage** (gated at 90%)
- **60 named tolerances** in 5 submodules (Rust + Python mirror, zero inline magic numbers)
- **25 Tier A GPU modules** (20 upstream batched ops, local_dispatch retired)
- **Zero C dependencies**, zero unsafe, zero `#[allow()]`, Edition 2024
- **guideStone Level 4** (targeting L6 with live NUCLEUS; IPC-wired, **10 UniBin validation scenarios**)
- **57-method `niche::CAPABILITIES`** (sync-tested vs 474-method canonical cross-sync, Wave 60, stability tiers annotated)
- **36 foundation targets** + **6 toadStool workloads** (thread06_ag)
- **deny.toml** promoted to workspace root (ecoBin v3.0, ring/openssl banned)
- **3 largest files refactored** (provenance 747→496, rpc 650→341, seasonal_pipeline 738→539)

## Key Validation Binaries

- `validate_et0` — FAO-56 Penman-Monteith ET₀ (8 methods)
- `validate_atlas` — Michigan Crop Water Atlas (100 stations × 80 years, 1354/1354)
- `validate_dual_kc` — FAO-56 Ch 7 dual Kc with cover crops
- `bench_cpu_vs_python` — 25-algorithm Rust vs Python benchmark (14.3×)
- `validate_gpu_rewire_benchmark` — cross-spring GPU shader parity
- `validate_biome_graph` — biomeOS deploy graph topology (35/35)
- `validate_dispatch_experiment` — CPU/GPU/batch parity (51/51)
- `bench_cross_spring_evolution` — 146/146 cross-spring checks
- `validate_cross_spring_provenance` — 5-spring shader provenance (32/32)
- `airspring_primal` — NUCLEUS primal binary (57 capabilities, JSON-RPC 2.0, `primal.announce` Wave 17)

## Notebooks (25)

### sporePrint Summary (5)

| # | Notebook | Focus |
|---|----------|-------|
| 01 | Composition Validation | 57 capabilities, deploy graphs, primal composition, gaps |
| 02 | Benchmark Comparison | Python vs Rust vs GPU timing, 14.3× speedup, GPU tiers |
| 03 | Ecosystem Evidence | 90 experiments, 60 tolerances, quality gates, provenance |
| 04 | Cross-Spring Connections | barraCuda integration, shader families, primal consumption |
| 05 | Domain Deep Dive | Michigan Atlas, seasonal pipeline, Penny Irrigation vision |

### Paper Baseline Notebooks (20)

| # | Notebook | Citation |
|---|----------|----------|
| 001 | FAO-56 Penman-Monteith ET₀ | Allen et al. 1998 |
| 002 | Soil Sensor Calibration | Dong et al. 2020 |
| 004 | FAO-56 Water Balance | Allen et al. 1998 Ch 8 |
| 006 | Richards Equation (VG-Mualem) | Richards 1931, van Genuchten 1980 |
| 007 | Biochar P Adsorption | Kumari et al. 2025 |
| 008 | Yield Response (Stewart) | Stewart et al. 1977 |
| 009 | Dual Crop Coefficient | Allen et al. 1998 Ch 7 |
| 017 | ET₀ Sensitivity Analysis | Gong et al. 2006 |
| 018 | Michigan Crop Water Atlas | Open-Meteo ERA5 |
| 019 | Priestley-Taylor ET₀ | Priestley & Taylor 1972 |
| 021 | Thornthwaite ET₀ | Thornthwaite 1948 |
| 023 | Saxton-Rawls PTFs | Saxton & Rawls 2006 |
| 031 | Hargreaves-Samani ET₀ | Hargreaves & Samani 1985 |
| 033 | Makkink ET₀ | Makkink 1957 |
| 034 | Turc ET₀ | Turc 1961 |
| 035 | Hamon PET | Hamon 1961 |
| 049 | Blaney-Criddle PET | Blaney & Criddle 1950 |
| 050 | SCS Curve Number | USDA 1972 |
| 051 | Green-Ampt Infiltration | Green & Ampt 1911 |
| 081 | SPI Drought Index | McKee et al. 1993 |

## Workload TOMLs (6)

| Workload | Domain |
|----------|--------|
| `airspring-et0-fao56` | FAO-56 PM 75/75 cross-validated |
| `airspring-et0-methods` | 8 ET₀ methods suite |
| `airspring-water-balance` | Ch 8 + dual Kc + yield |
| `airspring-soil-physics` | Richards + GA + SCS-CN + PTF |
| `airspring-atlas-pipeline` | 100 stations, 80 years |
| `airspring-full-suite` | All 90 experiments |

Available in both `foundation/workloads/thread06_ag/` and `projectNUCLEUS/workloads/airspring/`.

## See Also

- [Spring Catalog](https://primals.eco/architecture/spring-catalog-status-science-and-evolution/) on primals.eco
- [Lab Notebooks](https://primals.eco/lab/notebooks/) for rendered notebook views
- [baseCamp Papers](https://primals.eco/science/) (Dong lab, FAO-56, Richards, Stewart)
