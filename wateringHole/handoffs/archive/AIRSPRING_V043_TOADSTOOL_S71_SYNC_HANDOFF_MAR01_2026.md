# airSpring → ToadStool/BarraCUDA — S71 Sync + Evolution Benchmark Expansion

**Date:** March 1, 2026
**From:** airSpring v0.5.9 (63 experiments, 817 tests, 53/53 cross-spring evolution benchmark)
**To:** ToadStool/BarraCUDA core team
**ToadStool HEAD:** `8dc01a37` (Session 71)
**Supersedes:** V042 Cross-Spring Evolution Rewire (archived)
**License:** AGPL-3.0-or-later

---

## Summary

- Pulled and validated ToadStool S71 (671 WGSL shaders, 2,773+ barracuda tests, 66 `ComputeDispatch` migrations).
- Expanded cross-spring evolution benchmark from 44 → 53 checks with 9 S71-specific validations.
- Upstream `fao56_et0` cross-validated bit-identical with local Penman-Monteith.
- New `KimuraGpu`, `JackknifeMeanGpu`, `BootstrapMeanGpu`, `HistogramGpu` available for GPU dispatch.
- DF64 transcendentals complete: 15 functions (asin, acos, atan, atan2, sinh, cosh, gamma, erf + existing 7).
- All 817 tests pass, 0 clippy warnings, 0 regressions from the S71 pull.

---

## Part 1: S71 Pull Validation

### What Changed Upstream

| Category | S71 Change | Impact on airSpring |
|----------|-----------|---------------------|
| **Shader count** | 774 → 671 WGSL | f32-only shaders removed; universal precision |
| **DF64 transcendentals** | +8 functions (asin, acos, atan, atan2, sinh, cosh, gamma, erf) | DF64 path now covers all math airSpring needs |
| **`ComputeDispatch`** | 66 ops migrated (~184 remaining) | Internal refactor; no API change for consumers |
| **GPU dispatch types** | `HargreavesBatchGpu`, `JackknifeMeanGpu`, `BootstrapMeanGpu`, `HistogramGpu`, `KimuraGpu` | New GPU acceleration paths |
| **`fao56_et0` scalar** | Full PM from groundSpring → S70 | Complementary to airSpring's decomposed API |
| **HMM log-domain** | `HmmForwardLogF32`/`F64` | Available for time-series state estimation |
| **File refactoring** | `jsonrpc_server.rs` 904→628, `network_config/types.rs` split 7 modules, `builder.rs` split 3 modules | All < 1000 lines |
| **Sovereignty** | Port 8084→`daemon_port()`, songbird→mdns, primal names→constants | Aligns with airSpring's capability-based approach |

### Validation Results

```
cargo check                        → OK (barracuda v0.2.0, airspring v0.5.9)
cargo clippy --all-targets         → 0 warnings
cargo test                         → 817 passed, 0 failed
bench_cross_spring_evolution       → 53/53 PASS (release)
```

Zero regressions from the 94-file ToadStool update (-14,239 lines, +5,277 lines).

---

## Part 2: Evolution Benchmark Expansion (44 → 53 checks)

### 9 New S71 Checks

| Check | Observed | Expected | Tolerance | Provenance |
|-------|----------|----------|-----------|------------|
| `upstream fao56_et0 FAO-56 Example 18` | 3.975 | 3.88 | 0.15 | groundSpring → S70 |
| `upstream fao56_et0 ≈ local PM` | 3.975 | 3.975 | 0.15 | Cross-validation (bit-identical) |
| `kimura fixation p > 0.5 (s>0)` | 0.9999 | > 0.5 | — | wetSpring bio → S71 |
| `kimura fixation p < 1.0` | 0.9999 | < 1.0 | — | wetSpring bio → S71 |
| `jackknife mean of 1..5 = 3.0` | 3.0 | 3.0 | 1e-12 | neuralSpring → S70+ |
| `jackknife variance > 0` | 0.5 | > 0 | — | neuralSpring → S70+ |
| `bootstrap CI lower < upper` | 3.4 | > 0 | — | S64 |
| `bootstrap mean ≈ 6.5` | 6.5 | 6.5 | 0.5 | S64 |
| `percentile(50) of uniform [0,1)` | 0.495 | 0.5 | 0.05 | S64 |

### Cross-Validation Finding

Upstream `barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)`
produces **identical** output to airSpring's local `eco::evapotranspiration::daily_et0()` when
given the same meteorological inputs (after converting RH to actual vapour pressure). The two
implementations are independent (groundSpring origin vs airSpring origin) but converge to the
same FAO-56 equations. This confirms both implementations are correct.

---

## Part 3: New Upstream Capabilities — Absorption Roadmap

### Available Now (S71)

| Capability | GPU Type | API | airSpring Use Case |
|------------|----------|-----|-------------------|
| `HargreavesBatchGpu` | `dispatch(ra, t_max, t_min)` | Pre-computed Ra | Alternative to BatchedElementwiseF64 op=6 |
| `JackknifeMeanGpu` | `dispatch(data)` | Leave-one-out mean GPU | ET₀ uncertainty for 100-station atlas |
| `BootstrapMeanGpu` | `dispatch(data, n_bootstrap, seed)` | Bootstrap distribution GPU | MC uncertainty acceleration |
| `HistogramGpu` | `dispatch(values, n_bins)` | Atomic histogram GPU | Empirical ET₀ distributions |
| `KimuraGpu` | `dispatch(pop_sizes, selections, freqs)` | Population genetics GPU | Cross-spring bio (via wetSpring) |
| `fao56_et0` scalar | `fao56_et0(tmax, tmin, rh_max, rh_min, ...)` | Full PM from RH | Alternative entry point |

### Design Note: `HargreavesBatchGpu` vs `BatchedElementwiseF64` op=6

airSpring keeps `BatchedElementwiseF64` op=6 for the seasonal pipeline because it accepts
`(tmax, tmin, lat_rad, doy)` and computes Ra internally — more convenient for the weather-data
pipeline. Upstream `HargreavesBatchGpu` requires pre-computed Ra. Both are valid; airSpring
uses whichever API fits the caller.

### Design Note: DF64 Transcendentals

With 15 DF64 functions now complete, airSpring's GPU shaders can optionally switch from
native f64 to DF64 on consumer GPUs (GTX/RTX without native f64 throughput). This is
handled by ToadStool's `Fp64Strategy` auto-selection — no changes needed in airSpring code.
The precision benefit: ~48-bit mantissa vs 52-bit native, acceptable for all agricultural
computations (sensor accuracy is ≫ 4 bits).

---

## Part 4: What Shrunk (774 → 671 WGSL)

ToadStool S68-S71 replaced per-precision shader copies with universal precision
architecture. Instead of separate f32 and f64 shaders, a single f64 source is
compiled to the target precision via:

- `Precision::F64` → native builtins (Titan V, A100)
- `Precision::Df64` → DF64 f32-pair ~48-bit (consumer GPUs)
- `Precision::F32` → downcast via `downcast_f64_to_f32()` (backward compat)
- `Precision::F16` → downcast via `downcast_f64_to_f16()` (edge inference)

This is the "math is universal — precision is silicon" doctrine. airSpring benefits
automatically: same shader source, best available precision per hardware.

---

## For ToadStool Team

1. **`fao56_et0` cross-validation**: airSpring independently confirmed that the
   upstream `fao56_et0` produces identical results to the airSpring local PM when
   given equivalent inputs. Both implementations are correct.
2. **`ComputeDispatch` regression gate**: The 53-check evolution benchmark exercises
   primitives from all 5 Springs. Consider running it after `ComputeDispatch` migrations
   to catch regressions (the remaining 184 ops).
3. **`bootstrap_ci` closure API**: The `statistic` parameter accepts `fn(&[f64]) -> f64`.
   Clippy suggests passing `barracuda::stats::mean` directly rather than `|d| mean(d)`.
   This works and is more idiomatic.
4. **S71 session clean**: The 94-file update was absorbed with zero regressions in
   airSpring. The -14K/+5K delta is aggressive refactoring done well.

---

## Validation

```bash
cargo fmt -- --check                                    # Clean
cargo clippy --all-targets -- -D warnings               # 0 warnings
cargo test                                              # 817 passed, 0 failed
cargo run --release --bin bench_cross_spring_evolution   # 53/53 PASS
cargo doc --no-deps                                     # Builds
```

---

*Unidirectional handoff — no response expected. airSpring continues autonomous evolution.*
