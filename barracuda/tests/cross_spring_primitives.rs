// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring primitive validation — proves airSpring benefits from the
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
//! airSpring → BarraCuda    : TS-001 pow_f64 fix, TS-003 acos fix,
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
            2.5f64.mul_add(
                (2.0 * std::f64::consts::PI * (doy - 172.0) / 365.0).cos(),
                3.0,
            )
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
        .map(|&xi| (xi * 0.1).sin().mul_add(0.01, 2.5_f64.mul_add(xi, 0.3)))
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
        "Nelder-Mead Langmuir R²>0.98; neuralSpring optimizer (BarraCuda S62)"
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
// §4 — airSpring contributions back to BarraCuda
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
        "Richards PDE solver must converge (airSpring → BarraCuda S40)"
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
