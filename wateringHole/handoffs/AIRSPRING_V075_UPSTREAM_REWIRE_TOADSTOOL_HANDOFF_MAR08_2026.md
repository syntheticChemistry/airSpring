# airSpring V0.7.5 — Upstream Rewire Handoff

SPDX-License-Identifier: AGPL-3.0-or-later
**Date**: March 8, 2026
**From**: airSpring
**To**: barraCuda / toadStool teams
**Supersedes**: AIRSPRING_V074_STOCHASTIC_DROUGHT_TOADSTOOL_HANDOFF_MAR07_2026.md

---

## Executive Summary

- Synced to barraCuda HEAD (`a898dee`), toadStool S130+, coralReef Phase 10.
- **Write→Absorb→Lean applied to `eco::drought_index`**: Removed 55 lines of local
  `regularized_gamma_p`/`gamma_series`/`gamma_cf` — delegated to upstream
  `barracuda::special::gamma::regularized_gamma_p`. Validation: 20/20, 854/854 lib.
- Documented 9 new upstream capabilities available for future wiring.
- Zero API breakage. Zero clippy warnings. All tests green.

---

## §1 What Changed

### `eco::drought_index` Lean

| Before (v0.7.4) | After (v0.7.5) |
|------------------|----------------|
| Local `regularized_gamma_p` (series + CF, 55 lines) | `barracuda::special::gamma::regularized_gamma_p` (upstream) |
| Local `gamma_series` (~15 lines) | Removed — upstream handles internally |
| Local `gamma_cf` (~25 lines) | Removed — upstream handles internally |
| `barracuda::special::gamma::ln_gamma` (already upstream) | Unchanged |
| `barracuda::stats::normal::norm_ppf` (already upstream) | Unchanged |

The local implementation used the same Numerical Recipes series/continued-fraction
algorithm. The upstream version adds proper error handling via `Result<f64>`.

### Upstream Pin

| Component | Version | Commit |
|-----------|---------|--------|
| barraCuda | 0.3.3 (unreleased HEAD) | `a898dee` |
| toadStool | S130+ | `bfe7977b` |
| coralReef | Phase 10 Iteration 10 | `d29a734` |

---

## §2 New Upstream Capabilities (Available, Not Yet Wired)

These were added to barraCuda since our last sync and are now available to airSpring:

| Capability | Module | Potential Use |
|------------|--------|---------------|
| `regularized_gamma_q(a, x)` | `special::gamma` | Complement of gamma CDF (survival function) |
| `digamma(x)` | `special::gamma` + GPU shader | Fisher information for gamma MLE refinement |
| `beta(a, b)` / `ln_beta(a, b)` | `special::gamma` + GPU shader | Beta distribution fitting (soil texture) |
| `lower_incomplete_gamma(a, x)` | `special::gamma` | Raw incomplete gamma (no regularization) |
| `upper_incomplete_gamma(a, x)` | `special::gamma` | Complement incomplete gamma |
| `BatchedOdeRK45F64` | `ops::rk45_adaptive` | Adaptive Richards solver on GPU |
| `mean_variance_to_buffer()` | `ops::variance_f64_wgsl` | Zero-readback fused Welford for chained pipelines |
| `AutocorrelationF64` | `ops::autocorrelation_f64_wgsl` | Temporal autocorrelation for ET₀ time series |
| R² on `CorrelationResult` | `ops::correlation_f64_wgsl` | GPU-side R² without separate computation |

### Priority Wiring Candidates

1. **`BatchedOdeRK45F64`** — direct upgrade for `gpu::richards` (adaptive step, better accuracy)
2. **`mean_variance_to_buffer()`** — enables zero-readback seasonal pipeline
3. **`AutocorrelationF64`** — enables GPU-accelerated ET₀ temporal analysis
4. **`digamma`** — enables iterative gamma MLE refinement for more accurate SPI

---

## §3 toadStool Absorption Status

toadStool S130+ has airSpring V071 absorbed. Key gaps vs current V075:

| Feature | V071 Status | V075 Status | toadStool Action |
|---------|-------------|-------------|------------------|
| Experiments | 72 | 81 | Update tracker |
| Lib tests | ~850 | 854 | Update tracker |
| Binaries | ~86 | 89 | Update tracker |
| `eco::drought_index` | N/A | New module | Note as new consumer of `special::gamma` |
| `local_dispatch` | Active (3 ops) | **Retired (v0.7.2)** | Remove from absorption map |
| `local_elementwise_f64.wgsl` | Active | **Retired (v0.7.2)** | Remove from coralReef corpus |
| `PrecisionRoutingAdvice` | Documented | **Wired (v0.7.3)** | Confirmed operational |
| Upstream provenance | N/A | **Wired (v0.7.3)** | Confirmed queryable |

### coralReef Note

coralReef Phase 10 still has `local_elementwise_f64.wgsl` in its WGSL corpus. This
shader was retired in airSpring v0.7.2 when all 6 ops were absorbed into upstream
`BatchedElementwiseF64`. Suggest updating the corpus reference to point to
`batched_elementwise_f64.wgsl` from barraCuda instead.

---

## §4 Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | PASS (0 warnings) |
| `cargo test --lib` | **854/854 PASS** |
| `validate_drought_index` | **20/20 PASS** (with upstream `regularized_gamma_p`) |

---

## §5 Verification

```bash
cd airSpring/barracuda

# Confirm upstream sync
cargo check    # builds against barraCuda HEAD a898dee

# Quality gates
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --lib    # 854 tests

# Drought index (exercises upstream special::gamma)
cargo run --release --bin validate_drought_index
```
