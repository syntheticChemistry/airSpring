# airSpring V0.7.4 — Stochastic Methods & Drought Index Handoff

SPDX-License-Identifier: AGPL-3.0-or-later
**Date**: March 7, 2026
**From**: airSpring
**To**: barraCuda / toadStool teams
**Supersedes**: AIRSPRING_V073_MODERN_INTEGRATION_HANDOFF_MAR07_2026.md

---

## Executive Summary

- Three new experiments (079-081) add stochastic uncertainty quantification and drought
  analysis to airSpring, filling the last major gap in the Python→Rust CPU pipeline
  before GPU promotion of these methods.
- **Exp 079** (MC ET₀ Uncertainty Propagation): validates `gpu::mc_et0::mc_et0_cpu` against
  a deterministic Python baseline (Lehmer LCG + Box-Muller). 26/26 Rust checks.
- **Exp 080** (Bootstrap & Jackknife CI): validates `gpu::bootstrap::GpuBootstrap::cpu()`
  and `gpu::jackknife::GpuJackknife::cpu()` for seasonal ET₀ confidence intervals. 20/20.
- **Exp 081** (Standardized Precipitation Index): new `eco::drought_index` Rust module
  with gamma MLE, regularized incomplete gamma, and multi-scale SPI (1/3/6/12). 20/20.
- All three are GPU-promotable: embarrassingly parallel per-sample (MC), per-replicate
  (bootstrap), or per-grid-cell (SPI).

---

## §1 New Experiments

### Exp 079: Monte Carlo ET₀ Uncertainty Propagation

| Item | Value |
|------|-------|
| Python checks | 47/47 |
| Rust checks | 26/26 (`validate_mc_et0`) |
| Rust module | `gpu::mc_et0::mc_et0_cpu` |
| Tolerance | `MC_ET0_PROPAGATION` (abs=0.15, rel=0.08) |
| Benchmark | `control/mc_et0/benchmark_mc_et0.json` |
| Key insight | ET₀ std dev ~0.2-0.5 mm/day for 3-5% CV inputs |

**GPU promotion path**: `mc_et0_propagate_f64.wgsl` shader already exists in barraCuda.
The CPU path validates the statistical properties. GPU promotion would parallelize the
N=2000+ MC samples across workgroups — each sample is independent.

### Exp 080: Bootstrap & Jackknife CI for Seasonal ET₀

| Item | Value |
|------|-------|
| Python checks | 20/20 |
| Rust checks | 20/20 (`validate_bootstrap_jackknife`) |
| Rust modules | `gpu::bootstrap::GpuBootstrap::cpu()`, `gpu::jackknife::GpuJackknife::cpu()` |
| Benchmark | `control/bootstrap_jackknife/benchmark_bootstrap_jackknife.json` |
| Key insight | Bootstrap CI ≈ analytical SE/√n for large n; jackknife variance matches leave-one-out theory |

**GPU promotion path**: `BootstrapMeanGpu` and `JackknifeMeanGpu` shaders exist.
Each bootstrap replicate is independent (B=1000 parallel), each jackknife sample
is independent (n parallel). Already wired in `gpu::bootstrap` and `gpu::jackknife`.

### Exp 081: Standardized Precipitation Index (SPI)

| Item | Value |
|------|-------|
| Python checks | 20/20 |
| Rust checks | 20/20 (`validate_drought_index`) |
| New Rust module | `eco::drought_index` |
| Benchmark | `control/drought_index/benchmark_drought_index.json` |
| Dependencies | `barracuda::special::gamma::ln_gamma` |
| Key insight | Multi-scale SPI reveals different drought signals at 1/3/6/12 month scales |

**New module contents**:
- `DroughtClass` — WMO 7-class drought classification (extremely wet → extremely dry)
- `GammaParams` — shape/scale pair for gamma distribution
- `gamma_mle_fit` — maximum likelihood estimation via Thom (1958) method
- `regularized_gamma_p` — incomplete gamma via series expansion + continued fraction
- `gamma_cdf` — cumulative distribution via regularized gamma
- `compute_spi` — full SPI pipeline (rolling sum → gamma fit → normal quantile)

**GPU promotion path**: SPI is embarrassingly parallel per grid cell. Each cell's
rolling precipitation window is independent. The gamma MLE involves iterative
Newton-Raphson (convergent in ~10 iterations) and the inverse normal CDF is a
rational approximation — both are GPU-friendly. A `BatchedElementwiseF64` op or
dedicated SPI shader could process continental-scale drought maps.

---

## §2 BarraCuda Usage Summary (v0.7.4)

### Primitives Consumed

| barraCuda Module | airSpring Consumer | Notes |
|------------------|--------------------|-------|
| `ops::batched_elementwise_f64` | 20 ops (0-19) | All upstream, `local_dispatch` retired |
| `special::gamma::ln_gamma` | `eco::drought_index` | **New** — SPI gamma fitting |
| `stats::diversity` | `eco::diversity`, `gpu::diversity` | Shannon/Simpson/Pielou |
| `stats::pearson`, `spearman` | benchmarks, cross-validation | Statistical validation |
| `stats::bootstrap_ci` | `gpu::bootstrap` | Bootstrap confidence intervals |
| `linalg::ridge` | `eco::correction` | Ridge regression fitting |
| `optimize::brent` | VG inverse θ→h | Root-finding |
| `optimize::nelder_mead` | Isotherm fitting | Multi-start optimization |
| `pde::richards` | `gpu::richards` | Unsaturated flow PDE |
| `shaders::provenance` | `gpu::device_info` | Cross-spring shader tracking |
| `validation::ValidationHarness` | All 89 binaries | Structured pass/fail |

### Evolution Metrics

| Metric | v0.7.3 | v0.7.4 |
|--------|--------|--------|
| Experiments | 78 | **81** |
| Python checks | 1237 | **1284** |
| Lib tests | 848 | **854** |
| Validation binaries | 86 | **89** |
| `eco::` modules | 21 | **22** (+drought_index) |
| Named tolerances | 13-tier | 13-tier (+MC_ET0_PROPAGATION) |

---

## §3 toadStool Action Items

### For barraCuda team

1. **`special::gamma::ln_gamma`**: airSpring now depends on this for SPI. Any breaking
   changes to the return type (currently `Result<f64, BarracudaError>`) or the function
   signature will affect `eco::drought_index`.

2. **GPU SPI shader opportunity**: A dedicated `spi_drought_f64.wgsl` or
   `BatchedElementwiseF64` op for SPI would enable continental-scale drought mapping
   on GPU. The math is: rolling mean → gamma MLE (Newton-Raphson, ~10 iter) → gamma CDF
   (series/CF, ~100 terms) → inverse normal (rational approx). All f64-safe.

3. **MC ET₀ validation complete**: The `mc_et0_propagate_f64.wgsl` shader's CPU fallback
   is now validated against deterministic Python baselines. Ready for GPU dispatch
   confidence.

### For toadStool team

1. **Three new GPU promotion candidates**:
   - MC ET₀ (per-sample parallel, N=2000+)
   - Bootstrap CI (per-replicate parallel, B=1000)
   - SPI drought (per-cell parallel, continental grid)

2. **Tolerance `MC_ET0_PROPAGATION`**: Added to the 13-tier architecture. abs_tol=0.15,
   rel_tol=0.08 — wider than deterministic tolerances because stochastic algorithms
   with different RNG implementations will not produce identical distributions, only
   statistically equivalent ones.

3. **NaN handling convention**: Python `float('nan')` is serialized as JSON `null`
   (not NaN literal) for `serde_json` compatibility. SPI produces NaN for insufficient
   data windows — these are `null` in benchmark JSON and `f64::NAN` in Rust.

---

## §4 Cross-Spring Relevance

| Spring | Relevance |
|--------|-----------|
| **groundSpring** | MC ET₀ (Exp 079) validates the same uncertainty propagation infrastructure used by groundSpring's Monte Carlo methods. Bootstrap/Jackknife (Exp 080) mirrors groundSpring's jackknife variance estimation. |
| **wetSpring** | SPI drought index (Exp 081) connects to wetSpring's ecosystem monitoring — drought stress affects microbial community composition (baseCamp Paper 06). |
| **hotSpring** | Gamma function implementation uses the same `ln_gamma` special function infrastructure shared across all springs. |

---

## §5 Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings` | PASS (0 warnings) |
| `cargo test --lib` | **854/854 PASS** |
| `cargo test --test determinism` | **5/5 PASS** |
| `validate_mc_et0` | **26/26 PASS** |
| `validate_bootstrap_jackknife` | **20/20 PASS** |
| `validate_drought_index` | **20/20 PASS** |

---

## §6 Verification

```bash
# Run all three new validation binaries
cargo run --release --bin validate_mc_et0
cargo run --release --bin validate_bootstrap_jackknife
cargo run --release --bin validate_drought_index

# Full test suite
cargo test --lib          # 854 tests
cargo fmt --check         # formatting
cargo clippy --all-targets -- -D warnings  # linting
```
