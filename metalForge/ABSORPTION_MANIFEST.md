# Absorption Manifest — airSpring → barracuda

**Date**: February 25, 2026 (updated v0.4.0)
**Source**: `metalForge/forge/` (airspring-forge v0.2.0)
**Target**: `barracuda` (ToadStool crate)
**Absorption Status**: 2/6 absorbed upstream (van_genuchten → pde::richards, isotherm → optimize), 4 pending

---

## Absorption Process (Write → Absorb → Lean)

Following the pattern established by hotSpring:

1. **Write**: Implement primitives in pure Rust inside `metalForge/forge/`
2. **Validate**: Test against published benchmarks and Python baselines
3. **Hand off**: Document in `ABSORPTION_MANIFEST.md` with signatures, tests, provenance
4. **Absorb**: ToadStool copies into `barracuda::stats::*` or `barracuda::ops::*`
5. **Lean**: airSpring rewires to `use barracuda::*`, deletes local code
6. **Verify**: Run `validate_all` to confirm suites still pass

---

## Ready for Absorption

### 1. Statistical Agreement Metrics → `barracuda::stats::metrics`

| Function | Signature | Reference |
|----------|-----------|-----------|
| `rmse` | `fn(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError>` | Standard |
| `mbe` | `fn(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError>` | Standard |
| `nash_sutcliffe` | `fn(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError>` | Nash & Sutcliffe (1970) |
| `index_of_agreement` | `fn(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError>` | Willmott (1981) |
| `coefficient_of_determination` | `fn(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError>` | Standard (alias of NSE) |

**Validation**: Dong et al. (2020) *Agriculture* 10(12):598 — 36/36 checks.
918 station-days real data.  75/75 Python-Rust cross-validation.

**Dependencies**: None (pure arithmetic on `&[f64]`).

**Tests**: 11 unit tests covering perfect match, known values, zero bias,
mean predictor, constant bias, R²=NSE identity, length mismatch, and empty input.

**BarraCuda integration**: Follows the `stats::correlation` pattern — standalone
functions on `&[f64]`, re-exported from `stats/mod.rs`.

### 2. Analytical Regression → `barracuda::stats::regression`

| Function | Model | Method | Min points |
|----------|-------|--------|:----------:|
| `fit_linear` | y = a·x + b | 2×2 normal equations | 2 |
| `fit_quadratic` | y = a·x² + b·x + c | 3×3 Cramer's rule | 3 |
| `fit_exponential` | y = a·exp(b·x) | Log-linearized LS | 2 (y > 0) |
| `fit_logarithmic` | y = a·ln(x) + b | Linearized LS | 2 (x > 0) |
| `fit_all` | All four | Convenience wrapper | — |

**Result type**: `FitResult { model, params, r_squared, rmse }` with
`predict(&[f64]) -> Vec<Option<f64>>` and `predict_one(f64) -> Option<f64>` —
returns `None` for unknown model types instead of `0.0`.

**Validation**: Dong et al. (2020) sensor correction equations.  Perfect-fit
tests for all models.  Edge cases: insufficient points, singular systems,
negative domains.

**Dependencies**: None (pure arithmetic).

**Tests**: 11 unit tests (including predict/predict_one coverage).

### 3. CPU f64 Moving Window Statistics → `barracuda::ops::moving_window_stats_f64`

| Function | Signature | Output |
|----------|-----------|--------|
| `moving_window_stats` | `fn(data: &[f64], window_size: usize) -> Option<MovingWindowResultF64>` | mean, variance, min, max |

**Rationale**: Upstream `moving_window_stats` (wetSpring S28+) operates in f32
on GPU.  Agricultural sensor data requires f64 precision for sub-degree
temperature and sub-percent soil moisture readings.

**Dependencies**: None (pure arithmetic on `&[f64]`).

**Tests**: 7 unit tests (constant signal, ramp, variance, edge cases, diurnal
smoothing).

### 4. Hydrology Primitives → `barracuda::ops::hydrology`

| Function | Signature | Reference |
|----------|-----------|-----------|
| `hargreaves_et0` | `fn(ra: f64, t_max: f64, t_min: f64) -> Option<f64>` | Hargreaves & Samani (1985) |
| `hargreaves_et0_batch` | `fn(ra: &[f64], t_max: &[f64], t_min: &[f64]) -> Option<Vec<f64>>` | Batched convenience |
| `crop_coefficient` | `fn(kc_prev: f64, kc_next: f64, day: u32, stage_len: u32) -> f64` | FAO-56 Ch. 6 |
| `soil_water_balance` | `fn(theta: f64, precip: f64, irrig: f64, et_c: f64, fc: f64) -> f64` | FAO-56 Ch. 8 |

**Validation**: FAO-56 (Allen et al. 1998), 918 station-days, Python ETo
cross-validation.

**Dependencies**: None (pure arithmetic).

**Tests**: 13 unit tests covering typical values, edge cases, FAO-56 example,
batch operations, crop growth stages, saturation, and dry-down.

---

## Post-Absorption Rewiring (in airSpring)

Once barracuda absorbs these modules, airSpring will:

| airSpring location | Current code | Rewire to |
|--------------------|-------------|-----------|
| `barracuda/src/testutil/stats.rs` | `rmse()`, `mbe()`, `index_of_agreement()`, `nash_sutcliffe()` | `barracuda::stats::metrics::*` |
| `barracuda/src/eco/correction.rs` | `fit_linear()`, `fit_quadratic()`, `fit_exponential()`, `fit_logarithmic()` | `barracuda::stats::regression::*` |
| `barracuda/src/gpu/stream.rs` | `smooth_cpu()` (CPU fallback) | `barracuda::ops::moving_window_stats_f64::*` |
| `barracuda/src/eco/evapotranspiration.rs` | `hargreaves_et0()` inline | `barracuda::ops::hydrology::*` |

### 5. Van Genuchten Soil Hydraulics → `barracuda::pde::richards` (**ABSORBED**)

| Function | Signature | Reference |
|----------|-----------|-----------|
| `theta` | `fn(h, theta_r, theta_s, alpha, n) -> f64` | van Genuchten (1980) |
| `conductivity` | `fn(h, ks, theta_r, theta_s, alpha, n) -> f64` | Mualem (1976) |
| `capacity` | `fn(h, theta_r, theta_s, alpha, n) -> f64` | dθ/dh |

**Status**: ABSORBED — `barracuda::pde::richards::SoilParams` provides `theta()`,
`conductivity()`, `capacity()`. airSpring wires via `gpu::richards::to_barracuda_params`.

**Validation**: Carsel & Parrish (1988), HYDRUS baseline. Python ↔ Rust cross-validation.

**Dependencies**: None (pure arithmetic).

**Tests**: 5 unit tests (saturation, dry, conductivity, capacity).

### 6. Isotherm Models → `barracuda::optimize` (fitting via Nelder-Mead)

| Function | Signature | Reference |
|----------|-----------|-----------|
| `langmuir` | `fn(ce, qmax, kl) -> f64` | Langmuir (1918) |
| `freundlich` | `fn(ce, kf, n_inv) -> f64` | Freundlich (1906) |
| `separation_factor` | `fn(kl, c0) -> f64` | RL favorability |
| `fit_langmuir` | `fn(ce, qe) -> Option<IsothermFit>` | Linearized LS |
| `fit_freundlich` | `fn(ce, qe) -> Option<IsothermFit>` | Log-linearized + grid |

**Status**: WIRED — `gpu::isotherm::fit_langmuir_nm()` uses `barracuda::optimize::nelder_mead`
for nonlinear refinement. Linearized fits serve as initial guesses.

**Validation**: Kumari, Dong & Safferman (2025), wood and sugar beet biochar datasets.

**Dependencies**: None (pure arithmetic).

**Tests**: 5 unit tests (langmuir, freundlich, separation factor, synthetic fits).

---

## Post-Absorption Rewiring (in airSpring) — Updated v0.4.0

| airSpring location | Current code | Rewire to |
|--------------------|-------------|-----------|
| `barracuda/src/testutil/stats.rs` | `rmse()`, `mbe()`, etc. | `barracuda::stats::metrics::*` |
| `barracuda/src/eco/correction.rs` | `fit_linear()`, etc. | `barracuda::stats::regression::*` |
| `barracuda/src/gpu/stream.rs` | `smooth_cpu()` | `barracuda::ops::moving_window_stats_f64::*` |
| `barracuda/src/eco/evapotranspiration.rs` | `hargreaves_et0()` | `barracuda::ops::hydrology::*` |
| `barracuda/src/eco/richards.rs` | `van_genuchten_theta()`, etc. | `barracuda::pde::richards::SoilParams::*` |
| `barracuda/src/eco/isotherm.rs` | `fit_langmuir()`, etc. | `barracuda::optimize::nelder_mead` |

---

## Future Absorption Candidates (Not Yet Extracted)

| Candidate | Current location | Upstream fit | Blocker |
|-----------|-----------------|--------------|---------|
| `ValidationRunner` | `barracuda/src/validation.rs` | `barracuda::validation` | Needs abstraction from airSpring-specific JSON |
| `bootstrap_rmse` pattern | `barracuda/src/testutil/bootstrap.rs` | `barracuda::stats::bootstrap` extension | Already wraps `barracuda::stats::bootstrap_ci` |
| `len_f64` utility | `forge/src/lib.rs` | `barracuda::util` | Trivial — could be a PR |
| Batched sensor calibration op | `eco::sensor_calibration` | `batched_elementwise_f64.wgsl` op=5 | Needs WGSL shader work |

---

## Absorption Procedure

1. **Review**: ToadStool team reviews forge source modules
2. **Integrate**: Copy into `barracuda/src/stats/` and `barracuda/src/ops/`
3. **Register**: Add `pub mod metrics;` etc. to `stats/mod.rs` and `ops/mod.rs`
4. **Test**: Import forge tests alongside barracuda's test suite
5. **Release**: Bump barracuda version
6. **Lean**: airSpring replaces local implementations with `barracuda::*` imports
7. **Archive**: metalForge code marked as absorbed (keep for provenance)

---

## Quality

```
cargo fmt   — clean
cargo clippy --all-targets — zero warnings (pedantic)
cargo test  — 53/53 pass
```
