+++
title = "airSpring Validation Summary"
description = "Precision agriculture & irrigation — 1,364 Rust tests, 87 experiments, 14.3× speedup, 44/44 capabilities routable, 5 notebooks"
date = 2026-05-07

[taxonomies]
primals = ["barracuda", "toadstool", "biomeos", "nestgate", "squirrel", "coralreef", "petaltongue", "beardog", "songbird"]
springs = ["airspring", "hotspring", "wetspring", "neuralspring", "groundspring"]
+++

## Status

- **1,364 Rust tests** passing (986 lib + 316 integration + 62 forge), 0 failed
- **1,284 Python baseline checks** (60 papers reproduced)
- **87 experiments** across 12 categories (evapotranspiration → NUCLEUS mesh)
- **14.3× geometric mean** Rust-vs-Python speedup (24/24 algorithms, 21/21 CPU-GPU parity)
- **44/44 IPC capabilities** routable (science + ecology + provenance + coordination)
- **91 validation binaries** (all zero-panic, OrExit pattern)
- **90.56% line coverage** (gated at 90%)
- **60 named tolerances** in 5 submodules (Rust + Python mirror, zero inline magic numbers)
- **25 Tier A GPU modules** (20 upstream batched ops, local_dispatch retired)
- **Zero C dependencies**, zero unsafe, zero `#[allow()]`, Edition 2024
- **guideStone Level 0** → targeting Level 1 (primalSpring dependency next)

## Key Validation Binaries

- `validate_et0` — FAO-56 Penman-Monteith ET₀ (8 methods)
- `validate_atlas` — Michigan Crop Water Atlas (100 stations × 80 years, 1354/1354)
- `validate_dual_kc` — FAO-56 Ch 7 dual Kc with cover crops
- `bench_cpu_vs_python` — 24-algorithm Rust vs Python benchmark (14.3×)
- `validate_gpu_rewire_benchmark` — cross-spring GPU shader parity
- `validate_biome_graph` — biomeOS deploy graph topology (35/35)
- `validate_dispatch_experiment` — CPU/GPU/batch parity (51/51)
- `bench_cross_spring_evolution` — 146/146 cross-spring checks
- `validate_cross_spring_provenance` — 5-spring shader provenance (32/32)
- `airspring_primal` — NUCLEUS primal binary (44 capabilities, JSON-RPC 2.0)

## Notebooks (5)

| # | Notebook | Focus |
|---|----------|-------|
| 01 | Composition Validation | 44 capabilities, deploy graphs, primal composition, gaps |
| 02 | Benchmark Comparison | Python vs Rust vs GPU timing, 14.3× speedup, GPU tiers |
| 03 | Ecosystem Evidence | 87 experiments, 60 tolerances, quality gates, provenance |
| 04 | Cross-Spring Connections | barraCuda integration, shader families, primal consumption |
| 05 | Domain Deep Dive | Michigan Atlas, seasonal pipeline, Penny Irrigation vision |

## Workload TOMLs

Not yet created — contribute to `projectNUCLEUS/workloads/airspring/`.

## See Also

- [Spring Catalog](https://primals.eco/architecture/spring-catalog-status-science-and-evolution/) on primals.eco
- [Lab Notebooks](https://primals.eco/lab/notebooks/) for rendered notebook views
- [baseCamp Papers](https://primals.eco/science/) (Dong lab, FAO-56, Richards, Stewart)
