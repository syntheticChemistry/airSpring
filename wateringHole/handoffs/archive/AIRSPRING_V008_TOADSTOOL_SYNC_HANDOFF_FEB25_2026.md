# airSpring V008: ToadStool S62 Sync + Absorption Roadmap

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — v0.4.2, 585 total tests, 97.55% coverage)
**To**: ToadStool/BarraCuda core team
**ToadStool PIN**: `02207c4a` (HEAD — S62+, 170 commits since `0c477306`, 4,224+ tests)
**License**: AGPL-3.0-or-later
**Supersedes**: V007 (archived)

---

## Executive Summary

airSpring has reviewed ToadStool's S42–S62 evolution (170 commits, 46 cross-spring
absorptions, 4,224+ core tests, 758 WGSL shaders). Key findings:

- **All 4 airSpring TS issues confirmed resolved** (S54): TS-001 `pow_f64`,
  TS-003 `acos` precision, TS-004 reduce buffer, TS-002 Rust orchestrator
- **Zero breaking changes**: airSpring v0.4.2 compiles and passes all 585 tests
  against ToadStool HEAD without modification
- **16/16 validation binaries PASS** (341/341 checks) against current upstream
- **75/75 cross-validation MATCH** (Python↔Rust, tol=1e-5) — no drift
- **Rewired to modern BarraCuda**: `barracuda::tolerances` (S52) wired for
  21 domain-specific validated constants; cross-spring provenance documented
  in all 7 GPU modules; 18 cross-spring evolution integration tests added
- **4 metalForge modules still pending absorption** (metrics, regression,
  moving_window_f64, hydrology — 42 tests, pure arithmetic)
- Upstream capabilities cataloged: `solve_f64_cpu`, `OdeSystem`,
  `GpuSessionBuilder`, `dot()`, `provenance` tags

---

## Part 1: ToadStool S42–S62 Impact on airSpring

### Confirmed Resolutions

| Issue | Session | Resolution | airSpring Impact |
|-------|---------|------------|-----------------|
| TS-001: `pow_f64` fractional exponents | S54 (H-011) | `round()` + tolerance in `batched_elementwise_f64.wgsl` | VG retention θ(h) now correct |
| TS-002: Rust orchestrator | S54 (L-011) | Already present in `ops::batched_elementwise_f64` | No action needed |
| TS-003: `acos_simple` precision | S54 (H-012) | Replaced with `acos_f64` from `math_f64.wgsl` | Radiation calculations correct |
| TS-004: `FusedMapReduceF64` N≥1024 | S54 (H-013) | Separate `partials_buffer` for pass 2 | Seasonal stats GPU dispatch works |

### API Stability

airSpring uses 14 BarraCuda primitives. All APIs remained stable across 170
commits — zero code changes required in airSpring. This validates the
"consume upstream, lean on stability" pattern.

### New Capabilities Available (Not Yet Wired)

| Capability | Session | Potential airSpring Use |
|-----------|---------|------------------------|
| `FusedMapReduceF64::dot(a, b)` | S51 | GPU dot product for statistical inner products |
| `barracuda::tolerances` | S52 | Centralized tolerance registry (we use JSON `_tolerance_justification`) |
| `barracuda::provenance` | S52 | 12 `ProvenanceTag` consts for cross-spring origin tracking |
| `solve_f64_cpu()` | S51 | Gaussian elimination (we use Thomas algorithm for tridiagonal) |
| `GpuSessionBuilder` | S52 | Pre-warmed GPU sessions for benchmarks |
| `OdeSystem` + `BatchedOdeRK4` | S51 | Generic ODE solver (future: dynamic soil-plant models) |
| `crank_nicolson` f32 | S46+ | PDE solver (**needs f64 for Richards** — P1 ask) |

---

## Part 2: Revalidation Results

Full revalidation against ToadStool HEAD `02207c4a`:

```
cargo fmt --check              → clean
cargo clippy -- -D warnings    → 0 warnings (pedantic)
cargo test                     → 585 total (417 lib + 115 integration + 53 forge), 0 failures
cargo llvm-cov --lib           → 97.55% line coverage
cargo doc --no-deps            → 0 warnings
validation binaries            → 16/16 PASS (341/341 checks)
cross-validation               → 75/75 MATCH (tol=1e-5)
Python baselines               → 344/344 PASS
bench_airspring_gpu            → 8 benchmarks pass (ET₀, reduce, stream, kriging,
                                 ridge, Richards, isotherm, VG θ(h))
```

No drift detected. CPU and GPU paths produce identical results to pre-sync state.

---

## Part 3: Pending Absorption — 4 metalForge Modules (P1)

These 4 modules were offered in V006/V007 and remain ready:

| Module | Target | Tests | Status |
|--------|--------|:-----:|--------|
| `forge::metrics` | `barracuda::stats::metrics` | 11 | **Ready** — RMSE, MBE, NSE, IA, R² |
| `forge::regression` | `barracuda::stats::regression` | 11 | **Ready** — linear, quadratic, exponential, logarithmic |
| `forge::moving_window_f64` | `barracuda::ops::moving_window_stats_f64` | 7 | **Ready** — f64 CPU moving window |
| `forge::hydrology` | `barracuda::ops::hydrology` | 13 | **Ready** — Hargreaves ET₀, crop Kc, soil water balance |

**Total: 42 tests, pure arithmetic, zero dependencies beyond std.**

Full signatures, validation provenance, and post-absorption rewiring plan are in
`metalForge/ABSORPTION_MANIFEST.md`.

### Why These Matter to Every Spring

- **metrics** (RMSE, MBE, NSE, IA, R²): Universal model evaluation. hotSpring uses
  chi², wetSpring uses IA/NSE, airSpring uses all five. Currently each Spring
  implements its own — these should be shared.
- **regression** (4 fit models): Sensor correction is cross-domain — any IoT pipeline
  needs linear/exponential/logarithmic fitting.
- **hydrology** (Hargreaves ET₀, crop Kc): Climate-driven agriculture primitives
  useful for any environmental Spring.

---

## Part 4: Updated Action Items for ToadStool/BarraCuda

### P0 — Blocking

*None. airSpring is not blocked.*

### P1 — High Value (unchanged from V007, still pending)

| # | Item | Since | Justification |
|---|------|-------|---------------|
| 1 | **Absorb 4 metalForge modules** | V006 | 42 tests, pure arithmetic, every Spring benefits |
| 2 | **`crank_nicolson_f64`** | V007 | f32 exists (S46+); Richards PDE requires f64 |
| 3 | **Named constants in `pde::richards`** | V007 | 8 constants from airSpring for cross-Spring consistency |
| 4 | **Preallocation in `pde::richards`** | V007 | Picard buffers outside loops |

### P1 — New Recommendations (from S42–S62 review)

| # | Item | Justification |
|---|------|---------------|
| 5 | **Re-export `spearman_correlation` from `stats/mod.rs`** | Function exists in `correlation.rs` but isn't in the pub use block — airSpring accesses it via `stats::correlation::spearman_correlation` |
| 6 | **`[lints.clippy]` in barracuda Cargo.toml** | Modern Rust pattern (stable 1.74+). airSpring migrated; consistency across ecosystem |

### P2 — Nice to Have

| # | Item | Justification |
|---|------|---------------|
| 7 | Batch PDE dispatch | `pde::richards::solve_batch_gpu` for M soil columns |
| 8 | `fma_f64` WGSL instruction | `mul_add()` CPU parity for numerical consistency |
| 9 | Dual Kc GPU shader | `batched_elementwise_f64` op=8 for multi-field crop coefficients |

### P3 — Research

| # | Item | Justification |
|---|------|---------------|
| 10 | `unified_hardware` integration | metalForge → `HardwareDiscovery` + `ComputeScheduler` |
| 11 | Surrogate learning | Richards PDE → neural surrogate for real-time irrigation |
| 12 | `OdeSystem` for soil-plant | Dynamic root water uptake via generic ODE (S51) |

---

## Part 5: Cross-Spring Evolution Acknowledgment

airSpring acknowledges and benefits from these S42–S62 cross-spring absorptions:

| Absorption | Source | Benefit to airSpring |
|-----------|--------|---------------------|
| CG GPU-resident shaders (S51) | hotSpring | GPU infrastructure maturity |
| `solve_f64_cpu()` (S51) | hotSpring | CPU fallback for small linear systems |
| `OdeSystem` trait (S51) | wetSpring | Future dynamic soil models |
| `ValidationHarness` (S59) | neuralSpring | All 16 airSpring binaries use upstream |
| Tolerance registry (S52) | neuralSpring | Pattern for centralized justification |
| Provenance tags (S52) | wateringHole | Cross-spring origin tracking |
| `FusedMapReduceF64::dot()` (S51) | wetSpring | GPU dot product convenience |
| `GpuSessionBuilder` (S52) | wetSpring | Pre-warmed sessions for benchmarks |

---

## Part 6: Artifacts

| Document | Location |
|----------|----------|
| This handoff | `wateringHole/handoffs/AIRSPRING_V008_TOADSTOOL_SYNC_HANDOFF_FEB25_2026.md` |
| Previous handoff | `wateringHole/handoffs/archive/AIRSPRING_V007_*.md` |
| Evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| Absorption manifest | `metalForge/ABSORPTION_MANIFEST.md` |
| Evolution gaps (code) | `barracuda/src/gpu/evolution_gaps.rs` |
| ToadStool absorption tracker | `phase1/toadstool/ABSORPTION_TRACKER.md` |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001 | 2026-02-25 | Initial GPU handoff (v0.3.8) |
| V002 | 2026-02-25 | Dual Kc, cover crops, deep debt cleanup (v0.3.10) |
| V003 | 2026-02-25 | Richards + isotherm GPU wiring (v0.4.0) |
| V004 | 2026-02-25 | ToadStool S62 sync, multi-start NM (v0.4.1) |
| V005 | 2026-02-25 | Complete status, GPU integration tests (v0.4.2) |
| V006 | 2026-02-25 | Deep audit pass 1, 96.84% coverage (v0.4.2+) |
| V007 | 2026-02-25 | Lint migration, 97.58% coverage, 555 tests (v0.4.2, archived) |
| **V008** | **2026-02-25** | **ToadStool S62 sync: revalidated, 0 breaking changes, 4 modules still pending absorption** |

---

*End of V008 handoff. Direction: airSpring → ToadStool (unidirectional).
Supersedes V007 (archived). All 585 tests pass against ToadStool HEAD `02207c4a`.
Next handoff: V009 after metalForge absorption completes or `crank_nicolson_f64` lands.*
