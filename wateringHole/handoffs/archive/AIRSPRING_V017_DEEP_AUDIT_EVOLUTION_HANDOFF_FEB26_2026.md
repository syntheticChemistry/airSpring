# airSpring → ToadStool Handoff V017: Deep Audit & Evolution

**Date**: February 26, 2026
**From**: airSpring v0.4.6 — 16 experiments, 474 Python + 662 Rust tests + 1302 atlas checks, 22 binaries
**To**: ToadStool / BarraCuda core team
**ToadStool PIN**: `f0feb226` (S68 — universal f64, ValidationHarness tracing, LazyLock shader constants)
**Supersedes**: V016 (S66 validation — both retained for context)

---

## Executive Summary

airSpring v0.4.6 — deep audit & evolution session. All prior metrics maintained
plus improvements: Clippy nursery enforcement (0 warnings), R-S66-001/003 wiring,
eco::van_genuchten extraction, 11 doc-tests in metalForge, 8 Python baselines with
provenance, LOG_DOMAIN_GUARD documented, coverage 96.81% → 97.45%.

---

## Part 1: What We Did

- **Clippy nursery enforcement** — 0 warnings across both crates (pedantic + nursery)
- **R-S66-001 WIRED**: `eco::correction` delegates to `barracuda::stats::regression`
- **R-S66-003 WIRED**: `gpu::stream::smooth_cpu` delegates to `barracuda::stats::moving_window_stats_f64`
- **Smart refactor**: `eco::van_genuchten` extracted (930→800 + 150 lines)
- **11 doc-tests added** to metalForge
- **8 Python baselines** got provenance (commit `cb59873`)
- **LOG_DOMAIN_GUARD** documented
- **Coverage improved**: 96.81% → 97.45%
- **validate_atlas** (Exp 018): Michigan Crop Water Atlas, 100 stations, 1302/1302 checks

---

## Part 2: What We Need from ToadStool/BarraCuda

- Tier B GPU evolution priorities (dual Kc batch, VG batch, sensor calibration)
- Any new primitives that could benefit airSpring
- Feedback on R-S66-001/003 wiring quality

---

## Part 3: Absorption Candidates

Items from `eco::` that could be absorbed upstream:

| Function | Module | Notes |
|----------|--------|-------|
| `psychrometric_constant` | `eco::evapotranspiration` | Simple formula, broadly useful |
| `atmospheric_pressure` | `eco::evapotranspiration` | Altitude-based |
| `saturation_vapour_pressure` | `eco::evapotranspiration` | Tetens formula |
| `topp_equation` | `eco::soil_moisture` | Universal soil sensor calibration |
| `total_available_water` / `stress_coefficient` | `eco::water_balance` | FAO-56 |
| `yield_ratio_single` | `eco::yield_response` | Stewart 1977 |

---

## Part 4: Cross-Spring Learnings

- **mul_add transformations** improve both precision and performance
- **nursery lints** catch real issues (redundant_clone, needless_collect, option_if_let_else)
- **Smart refactoring** by responsibility > arbitrary line splits
- **LOG_DOMAIN_GUARD** values are domain-specific — don't blindly unify
- **serde_json** is pure Rust — no sovereignty concern
- **const fn** should be used wherever f64 operations allow it

---

## Part 5: Quality Gates (v0.4.6)

| Check | Status |
|-------|--------|
| Lib tests | **464** |
| Integration tests | **134** |
| Forge tests | **64** |
| **cargo test total** | **662** |
| Validation binaries | **22/22** PASS |
| Python checks | **474/474** PASS |
| Cross-validation | **75/75** match |
| Line coverage | **97.45%** (llvm-cov) |
| Clippy | **0 warnings** (pedantic + nursery) |
| `unsafe` blocks | **Zero** |
| Files > 1000 lines | **Zero** |
| AGPL-3.0 SPDX | Every .rs file |

---

## Part 6: Evolution Readiness Update

### Tier A (11 integrated)

ET₀, water_balance, kriging, reduce, stream, richards, isotherm, ridge, norm_ppf, brent, diversity

### Tier B (11 ready)

dual_kc batch, sensor_calibration, hargreaves, kc_climate, nonlinear, richards_full, tridiagonal, rk45, isotherm_batch

### Tier C (1 new)

HTTP/JSON client

---

## Open Items

None critical. All P0/P1 items resolved.

---

*airSpring v0.4.6 → ToadStool S68 (`f0feb226`). Deep audit complete, 662 Rust tests + 1302 atlas checks,
474 Python checks, 22 binaries, 97.45% coverage. Next: Tier B GPU wiring.*
