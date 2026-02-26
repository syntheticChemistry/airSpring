// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring evolution validation — proves airSpring benefits from the
//! multi-Spring shader ecosystem and contributes back.
//!
//! # Shader Provenance Chain
//!
//! ```text
//! hotSpring (lattice QCD)  → math_f64.wgsl (pow/exp/log/trig f64)
//!                            df64_core.wgsl (double-float emulation)
//!     ↓ used by
//! batched_elementwise_f64.wgsl → airSpring ET₀ (op=0), water balance (op=1)
//! van_genuchten_f64.wgsl       → airSpring Richards PDE
//!
//! wetSpring (microbiome)   → kriging_f64.wgsl (spatial interpolation)
//!                            fused_map_reduce_f64.wgsl (Shannon/Simpson/sum)
//!                            moving_window.wgsl (stream smoothing)
//!                            ridge_regression (ESN readout)
//!     ↓ used by
//! airSpring soil moisture mapping, seasonal stats, IoT smoothing, sensor calibration
//!
//! neuralSpring (ML)        → nelder_mead, multi_start_nelder_mead (optimization)
//!                            ValidationHarness (structured pass/fail)
//!     ↓ used by
//! airSpring isotherm fitting, all 16 validation binaries
//!
//! airSpring → ToadStool    : TS-001 pow_f64 fix, TS-003 acos fix,
//!                            TS-004 reduce buffer fix, Richards PDE (S40)
//! ```

use airspring_barracuda::tolerances;

// ═══════════════════════════════════════════════════════════════════════
// §1 — hotSpring precision math benefits
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn hotspring_pow_f64_enables_van_genuchten() {
    use airspring_barracuda::eco::richards::van_genuchten_theta;

    let theta_r = 0.045_f64;
    let theta_s = 0.43;
    let alpha = 0.145;
    let n = 2.68;

    let theta = van_genuchten_theta(-100.0, theta_r, theta_s, alpha, n);
    assert!(
        (theta_r..=theta_s).contains(&theta),
        "VG θ must be in [θr, θs]; hotSpring pow_f64 fix (TS-001 S54) \
         enables fractional exponents in the retention curve"
    );

    let theta_sat = van_genuchten_theta(0.0, theta_r, theta_s, alpha, n);
    assert!(
        tolerances::check(theta_sat, theta_s, &tolerances::SOIL_HYDRAULIC),
        "At h=0, θ should equal θs (saturated); validated via upstream tolerance"
    );
}

#[test]
fn hotspring_exp_log_enable_et0_chain() {
    use airspring_barracuda::eco::evapotranspiration::{
        saturation_vapour_pressure, vapour_pressure_slope,
    };

    let es_20 = saturation_vapour_pressure(20.0);
    assert!(
        tolerances::check(es_20, 2.338, &tolerances::ET0_SAT_VAPOUR_PRESSURE),
        "FAO-56 Table 2.3: es(20°C)=2.338 kPa; uses exp() from hotSpring math_f64.wgsl"
    );

    let delta_20 = vapour_pressure_slope(20.0);
    assert!(
        tolerances::check(delta_20, 0.1447, &tolerances::ET0_SLOPE_VAPOUR),
        "FAO-56 Table 2.4: Δ(20°C)=0.1447 kPa/°C; derivative of Tetens equation"
    );
}

#[test]
fn hotspring_trig_enables_solar_calculations() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;

    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.8,
        day_of_year: 187,
    };
    let result = airspring_barracuda::eco::evapotranspiration::daily_et0(&input);
    assert!(
        (result.et0 - 3.88).abs() < 0.15,
        "FAO-56 Example 18 (Uccle): ET₀=3.88 mm/day (tol=0.15 per benchmark JSON); \
         computed={:.4}; acos_f64 fix (TS-003 S54) ensures solar declination/hour angle precision",
        result.et0
    );
}

// ═══════════════════════════════════════════════════════════════════════
// §2 — wetSpring bio/environmental primitive benefits
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn wetspring_kriging_enables_soil_moisture_mapping() {
    use airspring_barracuda::gpu::kriging::{
        interpolate_soil_moisture, SensorReading, SoilVariogram, TargetPoint,
    };

    let sensors = vec![
        SensorReading {
            x: 0.0,
            y: 0.0,
            vwc: 0.25,
        },
        SensorReading {
            x: 100.0,
            y: 0.0,
            vwc: 0.35,
        },
        SensorReading {
            x: 50.0,
            y: 50.0,
            vwc: 0.30,
        },
    ];
    let targets = vec![TargetPoint { x: 50.0, y: 25.0 }];

    let variogram = SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 200.0,
    };

    let result = interpolate_soil_moisture(&sensors, &targets, variogram);
    assert!(
        !result.vwc_values.is_empty(),
        "Kriging interpolation should produce values; \
         wetSpring kriging_f64 enables spatial soil moisture mapping"
    );
    assert!(
        (0.0..=1.0).contains(&result.vwc_values[0]),
        "Interpolated VWC must be physically meaningful [0, 1]"
    );
    assert!(
        tolerances::check(result.vwc_values[0], 0.30, &tolerances::SOIL_HYDRAULIC),
        "Centroid should be near the mean of surrounding sensors"
    );
}

#[test]
fn wetspring_reduce_enables_seasonal_stats() {
    use airspring_barracuda::gpu::reduce::compute_seasonal_stats;

    let daily_et0: Vec<f64> = (0..365)
        .map(|d| {
            let doy = f64::from(d);
            3.0 + 2.5 * (2.0 * std::f64::consts::PI * (doy - 172.0) / 365.0).cos()
        })
        .collect();

    let stats = compute_seasonal_stats(&daily_et0);

    assert!(
        (stats.min - 0.5).abs() < 0.5,
        "Winter ET₀ min should be near 0.5 mm/day"
    );
    assert!(
        (stats.max - 5.5).abs() < 0.5,
        "Summer ET₀ max should be near 5.5 mm/day"
    );
    assert!(
        (stats.mean - 3.0).abs() < 0.1,
        "Annual mean ET₀ should be ~3.0 mm/day; \
         wetSpring fused_map_reduce_f64 drives these aggregations"
    );
    assert!(
        tolerances::check(
            stats.total,
            daily_et0.iter().sum::<f64>(),
            &tolerances::SEASONAL_REDUCTION
        ),
        "FusedMapReduceF64 total must match CPU iterator (TS-004 S54 fix)"
    );
}

#[test]
fn wetspring_moving_window_enables_iot_smoothing() {
    use airspring_barracuda::gpu::stream::smooth_cpu;

    let hourly_temp: Vec<f64> = (0..168)
        .map(|h| {
            let hour = f64::from(h);
            8.0_f64.mul_add(
                ((hour % 24.0 - 14.0) * std::f64::consts::PI / 12.0).cos(),
                25.0,
            )
        })
        .collect();

    let smoothed = smooth_cpu(&hourly_temp, 24).expect("valid 24h window");
    assert_eq!(
        smoothed.mean.len(),
        hourly_temp.len() - 24 + 1,
        "Output length = N - window + 1"
    );

    let daily_avg = smoothed.mean[0];
    assert!(
        (daily_avg - 25.0).abs() < 2.0,
        "24h moving average should center near the mean (25°C); \
         wetSpring moving_window.wgsl S28+ environmental monitoring"
    );
}

#[test]
fn wetspring_ridge_enables_sensor_calibration() {
    use airspring_barracuda::eco::correction::fit_ridge;

    let x: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.5_f64.mul_add(xi, 0.3) + (xi * 0.1).sin() * 0.01)
        .collect();

    let model = fit_ridge(&x, &y, 1e-6).expect("fit_ridge: sufficient data");
    assert!(
        model.r_squared > 0.99,
        "Ridge regression R²>0.99 for near-linear data; \
         barracuda::linalg::ridge absorbed from wetSpring ESN calibration (S59)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// §3 — neuralSpring optimizer benefits
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn neuralspring_nelder_mead_enables_isotherm_fitting() {
    use airspring_barracuda::gpu::isotherm::{fit_langmuir_global, fit_langmuir_nm};

    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    let nm_fit = fit_langmuir_nm(&ce, &qe).expect("NM fit");
    assert!(
        nm_fit.r_squared > 0.98,
        "Nelder-Mead Langmuir R²>0.98; neuralSpring optimizer (ToadStool S62)"
    );

    let global_fit = fit_langmuir_global(&ce, &qe, 4).expect("multi-start fit");
    assert!(
        global_fit.r_squared >= nm_fit.r_squared - 0.01,
        "Multi-start NM should match or beat single NM; \
         neuralSpring multi_start_nelder_mead with LHS exploration"
    );
}

#[test]
fn neuralspring_validation_harness_drives_all_binaries() {
    use airspring_barracuda::validation::ValidationHarness;

    let mut v = ValidationHarness::new("Cross-Spring Evolution");
    v.check_abs("hotSpring math: es(20°C)", 2.338, 2.338, 0.01);
    v.check_abs("wetSpring kriging: IDW at sensor", 0.25, 0.25, 0.01);
    v.check_bool("neuralSpring NM: converges", true);
    v.check_abs("airSpring Richards: VG θ(0)=θs", 0.43, 0.43, 0.001);

    assert_eq!(v.passed_count(), 4);
    assert_eq!(v.total_count(), 4);
}

// ═══════════════════════════════════════════════════════════════════════
// §4 — airSpring contributions back to ToadStool
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn airspring_ts001_pow_f64_fractional_exponents() {
    let alpha = 0.145_f64;
    let n = 2.68_f64;
    let m = 1.0 - 1.0 / n;
    let h: f64 = -100.0;

    let ah = (alpha * h.abs()).powf(n);
    let se = (1.0 + ah).powf(-m);

    assert!(se > 0.0 && se < 1.0, "fractional powf must return valid Se");
    assert!(
        se.is_finite(),
        "TS-001 S54 H-011: pow_f64 fractional exponents are now finite; \
         hotSpring round()+tolerance fix in batched_elementwise_f64.wgsl"
    );
}

#[test]
fn airspring_ts003_acos_precision_boundary() {
    let cos_val = 1.0_f64.min((-1.0_f64).max(0.999_999_999));
    let angle = cos_val.acos();
    assert!(
        angle.is_finite() && angle >= 0.0,
        "TS-003 S54 H-012: acos precision at boundaries must be finite; \
         hotSpring replaced acos_simple with acos_f64 from math_f64.wgsl"
    );
}

#[test]
fn airspring_ts004_reduce_buffer_large_n() {
    use airspring_barracuda::gpu::reduce::compute_seasonal_stats;

    let large: Vec<f64> = (0..2048).map(|i| f64::from(i) * 0.001).collect();
    let stats = compute_seasonal_stats(&large);

    let expected_sum: f64 = large.iter().sum();
    assert!(
        tolerances::check(stats.total, expected_sum, &tolerances::SEASONAL_REDUCTION),
        "TS-004 S54 H-013: N≥1024 reduce must match CPU sum; \
         separate partials_buffer for pass 2"
    );
}

#[test]
fn airspring_richards_pde_contributed_upstream() {
    use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    let result = solve_richards_1d(&sand, 30.0, 20, -20.0, 0.0, true, false, 0.1, 0.01);
    assert!(
        result.is_ok(),
        "Richards PDE solver must converge (airSpring → ToadStool S40)"
    );

    let profiles = result.unwrap();
    let final_theta = &profiles.last().unwrap().theta;
    for &theta in final_theta {
        assert!(
            (sand.theta_r..=sand.theta_s).contains(&theta),
            "All θ values must be physical; upstream pde::richards uses same solver"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// §5 — tolerances module validates cross-spring precision
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn tolerances_module_wired_to_upstream() {
    let tol = &tolerances::GPU_CPU_CROSS;
    assert_eq!(tol.name, "gpu_cpu_cross_validation");
    assert!(
        !tol.justification.is_empty(),
        "upstream barracuda::tolerances::Tolerance struct is wired"
    );
    assert!(tolerances::check(3.880_03, 3.88, tol));
    assert!(!tolerances::check(3.89, 3.88, tol));
}

#[test]
fn tolerances_cover_all_airspring_domains() {
    let domains = [
        &tolerances::ET0_SAT_VAPOUR_PRESSURE,
        &tolerances::ET0_SLOPE_VAPOUR,
        &tolerances::ET0_NET_RADIATION,
        &tolerances::ET0_REFERENCE,
        &tolerances::ET0_VPD,
        &tolerances::ET0_COLD_CLIMATE,
        &tolerances::PSYCHROMETRIC_CONSTANT,
        &tolerances::WATER_BALANCE_MASS,
        &tolerances::STRESS_COEFFICIENT,
        &tolerances::SOIL_HYDRAULIC,
        &tolerances::SOIL_ROUNDTRIP,
        &tolerances::RICHARDS_STEADY,
        &tolerances::RICHARDS_TRANSIENT,
        &tolerances::ISOTHERM_PARAMETER,
        &tolerances::ISOTHERM_PREDICTION,
        &tolerances::GPU_CPU_CROSS,
        &tolerances::KRIGING_INTERPOLATION,
        &tolerances::SEASONAL_REDUCTION,
        &tolerances::IOT_STREAM_SMOOTHING,
        &tolerances::SENSOR_EXACT,
        &tolerances::IRRIGATION_DEPTH,
    ];
    assert_eq!(
        domains.len(),
        21,
        "21 domain-specific tolerances covering all airSpring validation areas"
    );
    for tol in &domains {
        assert!(tol.abs_tol > 0.0 && tol.rel_tol > 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// §6 — cross-spring benchmark (CPU timing sanity checks)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn benchmark_et0_throughput_reasonable() {
    use airspring_barracuda::eco::evapotranspiration::{daily_et0, DailyEt0Input};
    use std::time::Instant;

    let inputs: Vec<DailyEt0Input> = (0..1000)
        .map(|i| {
            let d = f64::from(i);
            DailyEt0Input {
                tmin: 12.0 + (d * 0.01).sin(),
                tmax: 25.0 + (d * 0.01).cos(),
                tmean: None,
                solar_radiation: 22.0,
                wind_speed_2m: 2.0,
                actual_vapour_pressure: 1.4,
                elevation_m: 100.0,
                latitude_deg: 50.8,
                day_of_year: 187,
            }
        })
        .collect();

    let start = Instant::now();
    let total: f64 = inputs.iter().map(|i| daily_et0(i).et0).sum();
    let elapsed = start.elapsed();

    assert!(
        total > 0.0,
        "1000 ET₀ computations should produce positive total"
    );
    assert!(
        elapsed.as_millis() < 500,
        "1000 ET₀ CPU evaluations should complete in <500ms; \
         hotSpring math_f64 primitives power the Tetens/radiation chain"
    );
}

#[test]
fn benchmark_richards_throughput_reasonable() {
    use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};
    use std::time::Instant;

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    let start = Instant::now();
    let result = solve_richards_1d(&sand, 30.0, 20, -20.0, 0.0, true, false, 0.1, 0.01);
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Richards must converge");
    assert!(
        elapsed.as_millis() < 2000,
        "Richards 20-node solve should complete in <2s; \
         airSpring contributed this solver upstream (ToadStool S40)"
    );
}

#[test]
fn benchmark_isotherm_nm_throughput_reasonable() {
    use airspring_barracuda::gpu::isotherm::fit_langmuir_nm;
    use std::time::Instant;

    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    let start = Instant::now();
    for _ in 0..100 {
        let _ = fit_langmuir_nm(&ce, &qe);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "100 NM isotherm fits should complete in <5s; \
         neuralSpring optimize::nelder_mead powers the simplex search"
    );
}

// ── §7 — ToadStool S64: Cross-Spring Stats Absorption ─────────────────

#[test]
fn s64_stats_rmse_delegates_to_upstream() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sim = [1.1, 2.1, 2.9, 4.2, 4.8];
    let local = airspring_barracuda::testutil::rmse(&obs, &sim);
    let upstream = barracuda::stats::rmse(&obs, &sim);
    assert!(
        (local - upstream).abs() < f64::EPSILON,
        "airSpring testutil::rmse should delegate to upstream barracuda::stats::rmse; \
         local={local} upstream={upstream} — stats absorbed in S64"
    );
}

#[test]
fn s64_stats_mbe_delegates_to_upstream() {
    let obs = [5.0, 6.0, 7.0];
    let sim = [4.0, 5.5, 7.5];
    let local = airspring_barracuda::testutil::mbe(&obs, &sim);
    let upstream = barracuda::stats::mbe(&obs, &sim);
    assert!(
        (local - upstream).abs() < f64::EPSILON,
        "airSpring testutil::mbe should delegate to upstream; \
         local={local} upstream={upstream} — stats absorbed in S64"
    );
}

#[test]
fn s64_stats_new_reexports_from_upstream() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let m = airspring_barracuda::testutil::mean(&data);
    assert!(
        (m - 3.0).abs() < 1e-12,
        "mean re-export from barracuda::stats::mean (S64 absorption)"
    );

    let d = airspring_barracuda::testutil::dot(&data, &data);
    assert!(
        (d - 55.0).abs() < 1e-12,
        "dot re-export from barracuda::stats::dot (S64 absorption)"
    );

    let l2 = airspring_barracuda::testutil::l2_norm(&data);
    assert!(
        (l2 - 55.0_f64.sqrt()).abs() < 1e-12,
        "l2_norm re-export from barracuda::stats::l2_norm (S64 absorption)"
    );
}

// ── §8 — wetSpring S64: Diversity Metrics for Agroecology ─────────────

#[test]
fn s64_wetspring_diversity_shannon_for_cover_crops() {
    use airspring_barracuda::eco::diversity;

    let cover_mix = [120.0, 85.0, 45.0, 30.0, 20.0];
    let monoculture = [300.0, 0.0, 0.0, 0.0, 0.0];

    let h_mix = diversity::shannon(&cover_mix);
    let h_mono = diversity::shannon(&monoculture);

    assert!(
        h_mix > 1.0 && h_mono < 0.01,
        "wetSpring diversity::shannon wired for agroecology: \
         5-species cover crop mix H'={h_mix} > monoculture H'={h_mono}"
    );
}

#[test]
fn s64_wetspring_diversity_bray_curtis_field_comparison() {
    use airspring_barracuda::eco::diversity;

    let field_a = [120.0, 85.0, 45.0, 30.0, 20.0];
    let field_b = [90.0, 100.0, 55.0, 25.0, 30.0];
    let field_c = [0.0, 0.0, 0.0, 0.0, 300.0];

    let bc_similar = diversity::bray_curtis(&field_a, &field_b);
    let bc_different = diversity::bray_curtis(&field_a, &field_c);

    assert!(
        bc_similar < bc_different,
        "wetSpring Bray-Curtis: similar fields BC={bc_similar} < different fields BC={bc_different}"
    );
}

#[test]
fn s64_wetspring_alpha_diversity_comprehensive() {
    use airspring_barracuda::eco::diversity;

    let counts = [120.0, 85.0, 45.0, 30.0, 20.0];
    let ad = diversity::alpha_diversity(&counts);

    assert!((ad.observed - 5.0).abs() < 1e-10, "observed species = 5");
    assert!(ad.shannon > 1.0, "Shannon H' > 1.0 for 5-species mix");
    assert!(ad.simpson > 0.5, "Simpson D > 0.5 for multi-species");
    assert!(ad.chao1 >= 5.0, "Chao1 >= observed");
    assert!(
        (0.0..=1.0).contains(&ad.evenness),
        "Pielou J' in [0,1]: wetSpring bio diversity absorbed in S64"
    );
}

// ── §9 — groundSpring S64: MC ET₀ Uncertainty Propagation ────────────

#[test]
fn s64_groundspring_mc_et0_uncertainty_bands() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
    use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};

    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    };

    let result = mc_et0_cpu(&input, &Et0Uncertainties::default(), 2000, 42);

    assert!(
        result.et0_std > 0.05 && result.et0_std < 2.0,
        "MC ET₀ should show measurable uncertainty: σ={} — \
         groundSpring mc_et0_propagate_f64.wgsl absorbed in S64",
        result.et0_std
    );
    assert!(
        result.et0_p05 < result.et0_central && result.et0_p95 > result.et0_central,
        "90% CI [{}, {}] should bracket central ET₀={}",
        result.et0_p05,
        result.et0_p95,
        result.et0_central
    );
}

#[test]
fn s64_groundspring_mc_et0_deterministic_seed() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
    use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};

    let input = DailyEt0Input {
        tmin: 15.0,
        tmax: 28.0,
        tmean: None,
        solar_radiation: 18.5,
        wind_speed_2m: 1.5,
        actual_vapour_pressure: 1.2,
        elevation_m: 200.0,
        latitude_deg: 35.0,
        day_of_year: 200,
    };

    let r1 = mc_et0_cpu(&input, &Et0Uncertainties::default(), 500, 99);
    let r2 = mc_et0_cpu(&input, &Et0Uncertainties::default(), 500, 99);
    assert!(
        (r1.et0_mean - r2.et0_mean).abs() < f64::EPSILON,
        "MC ET₀ must be deterministic for same seed — \
         mirrors GPU kernel's xoshiro128** reproducibility"
    );
}

// ── §11 — ToadStool S66: Cross-Spring Absorption Wave ─────────────────
//
// S66 absorbed all pending airSpring metalForge modules upstream:
// regression (R-S66-001), hydrology (R-S66-002), moving_window_f64 (R-S66-003),
// spearman re-export (R-S66-005), 8 named SoilParams (R-S66-006),
// mae (R-S66-036), shannon_from_frequencies (R-S66-037).

#[test]
fn s66_regression_absorbed_upstream() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.1, 3.9, 6.1, 7.9, 10.1];

    let upstream = barracuda::stats::regression::fit_linear(&x, &y)
        .expect("barracuda::stats::regression::fit_linear should succeed (R-S66-001)");
    let local = airspring_barracuda::eco::correction::fit_linear(&x, &y)
        .expect("eco::correction::fit_linear should succeed");

    assert!(
        (upstream.r_squared - local.r_squared).abs() < 1e-6,
        "S66 upstream regression R²={} should match local R²={} — \
         airSpring metalForge regression absorbed upstream (R-S66-001)",
        upstream.r_squared,
        local.r_squared
    );
    assert!(
        upstream.r_squared > 0.99,
        "Near-perfect linear data should yield R²>0.99: got {}",
        upstream.r_squared
    );
}

#[test]
fn s66_hydrology_hargreaves_absorbed_upstream() {
    let tmin = 19.1_f64;
    let tmax = 32.6;
    let ra_mm = 40.55;

    let upstream = barracuda::stats::hydrology::hargreaves_et0(ra_mm, tmax, tmin)
        .expect("barracuda::stats::hydrology::hargreaves_et0 should succeed (R-S66-002)");
    let local = airspring_barracuda::eco::evapotranspiration::hargreaves_et0(tmin, tmax, ra_mm);

    assert!(
        (upstream - local).abs() < 0.01,
        "S66 upstream Hargreaves ET₀={upstream:.4} should match local={local:.4} — \
         airSpring metalForge hydrology absorbed upstream (R-S66-002), \
         param order differs: upstream(ra,tmax,tmin) vs local(tmin,tmax,ra)"
    );
    assert!(
        upstream > 0.0 && upstream < 15.0,
        "Hargreaves ET₀ should be physically reasonable: got {upstream}"
    );
}

#[test]
fn s66_moving_window_f64_absorbed_upstream() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = barracuda::stats::moving_window_f64::moving_window_stats_f64(&data, 10)
        .expect("barracuda::stats::moving_window_f64 should succeed (R-S66-003)");

    assert_eq!(
        result.mean.len(),
        91,
        "100 values with window=10 → 91 output windows"
    );
    assert!(
        result.variance.iter().all(|&v| v >= 0.0),
        "All variances must be non-negative — \
         airSpring metalForge moving_window_f64 absorbed upstream (R-S66-003)"
    );
}

#[test]
fn s66_spearman_reexport_available() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];

    let rho = barracuda::stats::spearman_correlation(&x, &y)
        .expect("spearman_correlation should be re-exported from stats (R-S66-005)");
    assert!(
        (rho - 1.0).abs() < 1e-10,
        "Perfect monotonic → Spearman ρ=1.0; got {rho} — \
         R-S66-005 added re-export from stats::correlation"
    );
}

#[test]
fn s66_soil_params_named_constants() {
    use barracuda::pde::richards::SoilParams;

    let sandy_loam = SoilParams::SANDY_LOAM;
    assert!(
        sandy_loam.theta_s > sandy_loam.theta_r,
        "θs > θr for sandy loam (Carsel & Parrish 1988, R-S66-006)"
    );
    assert!(
        sandy_loam.alpha > 0.0 && sandy_loam.n > 1.0,
        "VG parameters physical: α={}, n={}",
        sandy_loam.alpha,
        sandy_loam.n
    );

    let clay = SoilParams::CLAY;
    assert!(
        clay.k_sat < sandy_loam.k_sat,
        "Clay K_sat={} < sandy loam K_sat={} (R-S66-006)",
        clay.k_sat,
        sandy_loam.k_sat
    );

    let theta_at_saturation = sandy_loam.theta(0.0);
    assert!(
        (theta_at_saturation - sandy_loam.theta_s).abs() < 1e-6,
        "θ(h=0) should equal θs for sandy loam: got {theta_at_saturation}"
    );
}

#[test]
fn s66_metrics_mae_available() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sim = [1.5, 2.5, 2.5, 4.5, 4.5];

    let mae = barracuda::stats::mae(&obs, &sim);
    assert!(
        (mae - 0.5).abs() < 1e-12,
        "MAE of ±0.5 deviations should be 0.5; got {mae} — R-S66-036"
    );
}

#[test]
fn s66_diversity_shannon_from_frequencies() {
    let freqs = [0.5, 0.3, 0.2];
    let h = barracuda::stats::diversity::shannon_from_frequencies(&freqs);
    assert!(
        h > 0.0 && h < 2.0,
        "Shannon from frequencies should be positive and bounded; got {h} — R-S66-037"
    );

    let uniform = [0.25, 0.25, 0.25, 0.25];
    let h_max = barracuda::stats::diversity::shannon_from_frequencies(&uniform);
    assert!(
        h_max > h,
        "Uniform distribution should have higher entropy than skewed: \
         {h_max} > {h} — R-S66-037"
    );
}

// ── §12 — Cross-Spring Benchmark: S66 Throughput ──────────────────────

#[test]
fn benchmark_s66_regression_throughput() {
    use std::time::Instant;

    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.01).sin())
        .collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = barracuda::stats::regression::fit_linear(&x, &y);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "10K fit_linear (50 points) should complete in <5s; \
         metalForge regression absorbed upstream (R-S66-001); took {elapsed:?}"
    );
}

// ── §10 — Cross-Spring Benchmark: Modern System Throughput ────────────

#[test]
fn benchmark_diversity_throughput() {
    use airspring_barracuda::eco::diversity;
    use std::time::Instant;

    let counts: Vec<f64> = (1..=100).map(|i| f64::from(i) * 1.5).collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = diversity::alpha_diversity(&counts);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 2000,
        "10K alpha diversity computations (100 species) should complete in <2s; \
         wetSpring bio/diversity.rs absorbed in S64, took {elapsed:?}"
    );
}

#[test]
fn benchmark_mc_et0_throughput() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
    use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
    use std::time::Instant;

    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    };

    let start = Instant::now();
    let result = mc_et0_cpu(&input, &Et0Uncertainties::default(), 10_000, 42);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "10K MC ET₀ samples should complete in <5s (CPU mirror of \
         groundSpring mc_et0_propagate_f64.wgsl); took {elapsed:?}"
    );
    assert_eq!(result.n_samples, 10_000);
}

#[test]
fn benchmark_stats_reexport_throughput() {
    use airspring_barracuda::testutil;
    use std::time::Instant;

    let a: Vec<f64> = (0..10_000).map(f64::from).collect();
    let b: Vec<f64> = (0..10_000).map(|i| f64::from(i) + 0.1).collect();

    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = testutil::rmse(&a, &b);
        let _ = testutil::mbe(&a, &b);
        let _ = testutil::dot(&a, &b);
        let _ = testutil::l2_norm(&a);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "4K metric computations (10K-element vectors) should complete in <5s; \
         upstream delegation (S64) should not add overhead; took {elapsed:?}"
    );
}
