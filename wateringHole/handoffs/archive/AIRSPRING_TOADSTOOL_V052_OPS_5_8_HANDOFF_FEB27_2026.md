# airSpring → ToadStool V0.5.2 Handoff: Ops 5–8 Shader Absorption

**Date**: February 27, 2026
**From**: airSpring v0.5.2 (584 lib tests, 73/73 atlas stream PASS)
**To**: ToadStool S68+ / BarraCUDA team
**License**: AGPL-3.0-or-later
**Previous**: airSpring v0.5.0 (ops 0–1 wired, Tier A integrated)

---

## Part 1: Executive Summary

airSpring has wired 4 new GPU orchestrators (ops 5–8) with CPU fallback that
automatically activates GPU dispatch when ToadStool absorbs the corresponding
shader cases. The orchestrators are fully tested (584 lib tests), validated
on real 80-year Open-Meteo ERA5 data (12 Michigan stations, 4,800 crop-year
results, mass balance ~2e-13 mm), and pass `clippy::pedantic` with zero warnings.

**What ToadStool needs to do**: Add 4 `case` blocks to
`batched_elementwise_f64.wgsl` and 4 enum variants to
`batched_elementwise_f64.rs`. No new shaders needed — these reuse the
existing `BatchedElementwiseF64` infrastructure.

**Estimated effort**: Low — each op is 5–15 lines of WGSL math.

---

## Part 2: Op Specifications

### Op 5: SoilWatch 10 Sensor Calibration

**airSpring orchestrator**: `gpu::sensor_calibration::BatchedSensorCal`
**Stride**: 1 (single input per batch element)
**Input**: `[raw_count]`
**Output**: Volumetric water content (cm³/cm³)

**WGSL math** (Horner's method, FMA-friendly):
```wgsl
case 5u: {
    let raw = input[base + 0u];
    // Dong et al. (2024) SoilWatch 10 polynomial calibration
    output[batch_idx] = fma_f64(
        fma_f64(
            fma_f64(f64(2e-13), raw, f64(-4e-9)),
            raw, f64(4e-5)
        ),
        raw, f64(-0.0677)
    );
}
```

**Rust reference**: `eco::sensor_calibration::soilwatch10_vwc()`
**Validation**: 9 lib tests in `gpu::sensor_calibration`

### Op 6: Hargreaves-Samani ET₀

**airSpring orchestrator**: `gpu::hargreaves::BatchedHargreaves`
**Stride**: 4
**Input**: `[tmax, tmin, lat_rad, doy]`
**Output**: ET₀ (mm/day)

**WGSL math**:
```wgsl
case 6u: {
    let tmax = input[base + 0u];
    let tmin = input[base + 1u];
    let lat_rad = input[base + 2u];
    let doy_f = input[base + 3u];

    // Extraterrestrial radiation Ra (MJ/m²/day) — FAO-56 Eq. 21-25
    let dr = f64(1.0) + f64(0.033) * cos_f64(f64(6.283185307) * doy_f / f64(365.0));
    let delta = f64(0.4093) * sin_f64(f64(6.283185307) * doy_f / f64(365.0) - f64(1.405));
    let ws = acos_f64(f64(-1.0) * tan_f64(lat_rad) * tan_f64(delta));
    let ra_mj = f64(37.586) * dr * (
        ws * sin_f64(lat_rad) * sin_f64(delta) +
        cos_f64(lat_rad) * cos_f64(delta) * sin_f64(ws)
    );
    // Convert MJ/m²/day → mm/day equivalent
    let ra_mm = ra_mj * f64(0.408);

    // Hargreaves-Samani (1985) — FAO-56 Eq. 52
    let tmean = (tmax + tmin) * f64(0.5);
    let td = max(tmax - tmin, f64(0.0));
    output[batch_idx] = max(f64(0.0023) * (tmean + f64(17.8)) * sqrt_f64(td) * ra_mm, f64(0.0));
}
```

**Rust reference**: `eco::evapotranspiration::hargreaves_et0()` + `eco::solar::extraterrestrial_radiation()`
**Validation**: 11 lib tests in `gpu::hargreaves`

### Op 7: Kc Climate Adjustment (FAO-56 Eq. 62)

**airSpring orchestrator**: `gpu::kc_climate::BatchedKcClimate`
**Stride**: 4
**Input**: `[kc_table, u2, rh_min, crop_height_m]`
**Output**: Adjusted Kc (dimensionless)

**WGSL math**:
```wgsl
case 7u: {
    let kc_table = input[base + 0u];
    let u2 = input[base + 1u];
    let rh_min = input[base + 2u];
    let h = input[base + 3u];

    // FAO-56 Eq. 62: Kc adjustment for non-standard climate
    let adj = fma_f64(f64(0.04), u2 - f64(2.0), f64(-0.004) * (rh_min - f64(45.0)))
            * pow_f64(h / f64(3.0), f64(0.3));
    output[batch_idx] = max(kc_table + adj, f64(0.0));
}
```

**Rust reference**: `eco::crop::adjust_kc_for_climate()`
**Validation**: 8 lib tests in `gpu::kc_climate`

### Op 8: Dual Kc Evaporation Layer (Ke Step)

**airSpring orchestrator**: `gpu::dual_kc::BatchedDualKc` (with `with_gpu()` + `step_gpu()`)
**Stride**: 9
**Input**: `[kcb, kc_max, few, mulch_factor, de_prev, rew, tew, p_eff, et0]`
**Output**: Ke (evaporation coefficient, dimensionless)

**WGSL math**:
```wgsl
case 8u: {
    let kcb = input[base + 0u];
    let kc_max = input[base + 1u];
    let few = input[base + 2u];
    let mulch = input[base + 3u];
    let de_prev = input[base + 4u];
    let rew = input[base + 5u];
    let tew = input[base + 6u];
    let p_eff = input[base + 7u];
    let et0 = input[base + 8u];

    // FAO-56 dual Kc: soil evaporation reduction coefficient
    var kr = f64(1.0);
    if de_prev > rew {
        kr = max((tew - de_prev) / max(tew - rew, f64(0.001)), f64(0.0));
    }
    let ke_full = kr * (kc_max - kcb);
    let ke_limit = max(few * (kc_max - kcb), f64(0.0));
    let ke_base = min(ke_full, ke_limit);
    let ke = ke_base * mulch;

    // Update depletion: De = De_prev - P_eff + (Ke * ET₀)
    let de_new = clamp(de_prev - p_eff + ke * et0, f64(0.0), tew);
    // Pack: output is Ke (the main result); De_new via aux buffer or second pass
    output[batch_idx] = ke;
}
```

**Rust reference**: `eco::dual_kc::step()` (partial — Ke component only)
**Validation**: 2 GPU-specific tests in `gpu::dual_kc`

---

## Part 3: Rust Orchestrator Changes

Add enum variants to `BatchedElementwiseF64`:

```rust
#[repr(u32)]
pub enum Op {
    Fao56Et0 = 0,
    WaterBalance = 1,
    Custom = 2,
    ShannonBatch = 3,
    SimpsonBatch = 4,
    SensorCalibration = 5,  // NEW: SoilWatch 10 VWC
    HargreavesEt0 = 6,      // NEW: Temperature-only ET₀
    KcClimateAdjust = 7,    // NEW: FAO-56 Eq. 62
    DualKcKe = 8,           // NEW: Evaporation layer
}
```

Each op needs a `stride()` entry:
- Op 5: stride 1
- Op 6: stride 4
- Op 7: stride 4
- Op 8: stride 9

---

## Part 4: airSpring Auto-Activation

All 4 orchestrators are designed for zero-effort GPU activation:

```rust
// In gpu::sensor_calibration::BatchedSensorCal::compute_gpu():
// Currently falls back to compute_cpu_batch().
// When ToadStool absorbs op=5, replace fallback with:
//   self.gpu_engine.as_ref().unwrap().execute(&flat_data)
```

The Rust orchestrators already pack input buffers with the correct stride and
unpack GPU output. Only the `execute()` call is gated on ToadStool absorption.

---

## Part 5: Validation Commands

```bash
# All lib tests (584 pass)
cd airSpring/barracuda && cargo test --lib

# Atlas stream on real 80yr data (73/73 PASS, 12 stations, 4800 results)
cargo run --release --bin validate_atlas_stream

# Pure GPU validation (16/16 PASS)
cargo run --release --bin validate_pure_gpu

# Clippy pedantic (0 warnings)
cargo clippy --lib
```

---

## Part 6: Evolution After Absorption

Once ops 5–8 are absorbed:

1. **SeasonalPipeline** upgrades from CPU-chained to GPU per-stage (each stage dispatches independently)
2. **AtlasStream** upgrades to `UnidirectionalPipeline` for fire-and-forget streaming
3. **MC ET₀** GPU path activates via `mc_et0_propagate_f64.wgsl`
4. **metalForge** routes all 18 workloads to actual GPU hardware (currently 4 are `ShaderOrigin::Local`)

**Supersedes**: No previous airSpring→ToadStool handoff (first formal handoff).
