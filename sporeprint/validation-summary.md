+++
title = "airSpring Validation Summary"
description = "Precision agriculture and irrigation — 57 papers reproduced, R²=0.97 on open data, 13,000x speedup at atlas scale"
date = 2026-05-06

[taxonomies]
primals = ["barracuda", "toadstool", "biomeos"]
springs = ["airspring", "hotspring", "wetspring", "neuralspring", "groundspring"]
+++

## Status

- **57 papers reproduced** with full provenance
- **FAO-56 ET0** matches Python to 1e-5 across 75 cross-validated values
- **R²=0.97** on 100 Michigan stations (15,300 station-days) using open data
- **19.8x** geometric mean Rust speedup, **13,000x** at atlas scale
- NUCLEUS primal with 30 science capabilities

## Key Validation Binaries

<!-- TODO: Update with actual binary names from target/release/ -->
- `validate_fao56_et0` — FAO-56 reference evapotranspiration
- `validate_crop_coefficients` — Kc calculation pipeline
- `validate_water_balance` — soil moisture tracking
- `validate_real_data` — Michigan station cross-validation

## Workload TOMLs

Skeleton available in `projectNUCLEUS/workloads/airspring/`.

## See Also

- [airSpring Science Hub](https://primals.eco/lab/springs/airspring/) on primals.eco
- [baseCamp Papers 03, 06, 08, 12](https://primals.eco/science/)
