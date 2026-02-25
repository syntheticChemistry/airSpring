// SPDX-License-Identifier: AGPL-3.0-or-later
//! Eco-science integration tests for airSpring `BarraCuda`.
//!
//! Tests cross-module interactions within the `eco` domain:
//! FAO-56 ET₀, water balance, soil properties, crop coefficients,
//! wind conversion, sunshine radiation, Hargreaves cross-checks,
//! sensor calibration pipelines, and correction models.

use airspring_barracuda::eco::{
    correction,
    crop::CropType,
    evapotranspiration::{self as et, DailyEt0Input},
    sensor_calibration as sc,
    soil_moisture::{self as sm, SoilTexture},
    water_balance::{self as wb, DailyInput, RunoffModel, WaterBalanceState},
};
// ── Cross-module integration ─────────────────────────────────────────

#[test]
fn test_et0_drives_water_balance() {
    let et_input = DailyEt0Input {
        tmin: 18.0,
        tmax: 30.0,
        tmean: None,
        solar_radiation: 20.0,
        wind_speed_2m: 2.0,
        actual_vapour_pressure: 1.5,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: 180,
    };
    let et_result = et::daily_et0(&et_input);
    assert!(et_result.et0 > 0.0);

    let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    for _ in 0..10 {
        let wb_input = DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: et_result.et0,
            kc: 1.0,
        };
        state.step(&wb_input);
    }

    assert!(state.depletion > 0.0);
    assert!(state.current_theta() < 0.33);
}

#[test]
fn test_soil_texture_into_water_balance() {
    let props = SoilTexture::SiltLoam.hydraulic_properties();
    let state = WaterBalanceState::new(props.field_capacity, props.wilting_point, 600.0, 0.5);

    let paw = sm::plant_available_water(props.field_capacity, props.wilting_point, 600.0);
    assert!((state.taw - paw).abs() < f64::EPSILON);
}

// ── Determinism tests ────────────────────────────────────────────────

#[test]
fn test_et0_deterministic() {
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

    let r1 = et::daily_et0(&input);
    let r2 = et::daily_et0(&input);
    assert!((r1.et0 - r2.et0).abs() < f64::EPSILON);
    assert!((r1.rn - r2.rn).abs() < f64::EPSILON);
    assert!((r1.vpd - r2.vpd).abs() < f64::EPSILON);
}

#[test]
fn test_water_balance_deterministic() {
    let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let inputs: Vec<DailyInput> = (0..60)
        .map(|day| DailyInput {
            precipitation: if day % 5 == 0 { 20.0 } else { 0.0 },
            irrigation: if day % 12 == 0 { 30.0 } else { 0.0 },
            et0: 4.5,
            kc: 1.0,
        })
        .collect();

    let (final1, out1) = airspring_barracuda::eco::water_balance::simulate_season(&state, &inputs);
    let (final2, out2) = airspring_barracuda::eco::water_balance::simulate_season(&state, &inputs);

    assert!((final1.depletion - final2.depletion).abs() < f64::EPSILON);
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((a.depletion - b.depletion).abs() < f64::EPSILON);
        assert!((a.actual_et - b.actual_et).abs() < f64::EPSILON);
        assert!((a.deep_percolation - b.deep_percolation).abs() < f64::EPSILON);
    }
}

#[test]
fn test_topp_inverse_deterministic() {
    for &theta in &[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] {
        let e1 = sm::inverse_topp(theta);
        let e2 = sm::inverse_topp(theta);
        assert!((e1 - e2).abs() < f64::EPSILON);
    }
}

// ── Boundary / edge cases ────────────────────────────────────────────

#[test]
fn test_et0_arctic_conditions() {
    let input = DailyEt0Input {
        tmin: -20.0,
        tmax: -10.0,
        tmean: None,
        solar_radiation: 2.0,
        wind_speed_2m: 3.0,
        actual_vapour_pressure: 0.1,
        elevation_m: 50.0,
        latitude_deg: 65.0,
        day_of_year: 355,
    };
    let result = et::daily_et0(&input);
    assert!(result.et0 >= 0.0);
}

#[test]
fn test_et0_tropical_conditions() {
    let input = DailyEt0Input {
        tmin: 24.0,
        tmax: 34.0,
        tmean: None,
        solar_radiation: 25.0,
        wind_speed_2m: 1.5,
        actual_vapour_pressure: 2.5,
        elevation_m: 10.0,
        latitude_deg: 0.0,
        day_of_year: 80,
    };
    let result = et::daily_et0(&input);
    assert!(
        result.et0 > 3.0 && result.et0 < 10.0,
        "Tropical ET₀: {}",
        result.et0
    );
}

#[test]
fn test_water_balance_saturation_overflow() {
    let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let output = state.step(&DailyInput {
        precipitation: 0.0,
        irrigation: 1000.0,
        et0: 5.0,
        kc: 1.0,
    });
    assert!(
        (state.depletion).abs() < f64::EPSILON,
        "Depletion should be 0"
    );
    assert!(output.deep_percolation > 900.0, "Excess should drain");
}

#[test]
fn test_runoff_model_configurable() {
    let state_default = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let state_runoff = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5).with_runoff_model(
        RunoffModel::SimpleThreshold {
            threshold_mm: 20.0,
            fraction: 0.2,
        },
    );

    let heavy_rain = DailyInput {
        precipitation: 50.0,
        irrigation: 0.0,
        et0: 3.0,
        kc: 1.0,
    };

    let (_, out_default) = airspring_barracuda::eco::water_balance::simulate_season(
        &state_default,
        std::slice::from_ref(&heavy_rain),
    );
    let (_, out_runoff) =
        airspring_barracuda::eco::water_balance::simulate_season(&state_runoff, &[heavy_rain]);

    assert!((out_default[0].runoff).abs() < f64::EPSILON);
    assert!((out_runoff[0].runoff - 6.0).abs() < 0.01);
}

// ── Wind speed conversion ───────────────────────────────────────────

#[test]
fn test_wind_speed_at_2m_from_10m() {
    let u10 = 3.0;
    let u2 = et::wind_speed_at_2m(u10, 10.0);
    let expected = 3.0 * 0.748;
    assert!(
        (u2 - expected).abs() < 0.02,
        "u₂ from 10m: {u2}, expected ~{expected:.3}",
    );
}

#[test]
fn test_wind_speed_at_2m_identity() {
    let u2 = et::wind_speed_at_2m(5.0, 2.0);
    assert!((u2 - 5.0).abs() < 0.1, "u₂ at 2m should be ~5.0: {u2}");
}

#[test]
fn test_wind_speed_conversion_into_et0() {
    let u10 = 3.5;
    let u2 = et::wind_speed_at_2m(u10, 10.0);

    let input = DailyEt0Input {
        tmin: 18.0,
        tmax: 30.0,
        tmean: None,
        solar_radiation: 20.0,
        wind_speed_2m: u2,
        actual_vapour_pressure: 1.5,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: 180,
    };
    let result = et::daily_et0(&input);
    assert!(result.et0 > 0.0, "ET₀ with converted wind: {}", result.et0);
}

// ── Crop coefficient → water balance pipeline ───────────────────────

#[test]
fn test_crop_kc_drives_water_balance() {
    let crop = CropType::Corn.coefficients();
    let soil = SoilTexture::SandyLoam.hydraulic_properties();

    let mut state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        crop.root_depth_m * 1000.0,
        crop.depletion_fraction,
    );

    for _ in 0..30 {
        state.step(&DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: crop.kc_mid,
        });
    }

    assert!(state.depletion > 0.0, "Should be depleted");
    assert!(
        state.current_theta() < soil.field_capacity,
        "θ should be below FC"
    );
}

#[test]
fn test_tomato_vs_corn_water_demand() {
    let tomato = CropType::Tomato.coefficients();
    let corn = CropType::Corn.coefficients();
    let soil = SoilTexture::Loam.hydraulic_properties();

    let mut tom_state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        tomato.root_depth_m * 1000.0,
        tomato.depletion_fraction,
    );
    let mut corn_state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        corn.root_depth_m * 1000.0,
        corn.depletion_fraction,
    );

    for _ in 0..20 {
        tom_state.step(&DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: tomato.kc_mid,
        });
        corn_state.step(&DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: corn.kc_mid,
        });
    }

    let tom_frac = tom_state.depletion / tom_state.taw;
    let corn_frac = corn_state.depletion / corn_state.taw;
    assert!(
        tom_frac > corn_frac,
        "Tomato should deplete faster: {tom_frac:.3} vs corn {corn_frac:.3}"
    );
}

// ── Hargreaves cross-check ──────────────────────────────────────────

#[test]
fn test_hargreaves_vs_penman_monteith_same_order() {
    let doy: u32 = 180;
    let lat_rad = 42.0_f64.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, doy);
    let ra_mm = ra / 2.45;

    let harg_et0 = et::hargreaves_et0(18.0, 32.0, ra_mm);

    let pm_input = DailyEt0Input {
        tmin: 18.0,
        tmax: 32.0,
        tmean: None,
        solar_radiation: 22.0,
        wind_speed_2m: 2.0,
        actual_vapour_pressure: 1.5,
        elevation_m: 200.0,
        latitude_deg: 42.0,
        day_of_year: doy,
    };
    let pm_et0 = et::daily_et0(&pm_input).et0;

    assert!(harg_et0 > 1.0 && harg_et0 < 12.0, "Hargreaves: {harg_et0}");
    assert!(pm_et0 > 1.0 && pm_et0 < 12.0, "PM: {pm_et0}");

    let ratio = harg_et0 / pm_et0;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "Harg/PM ratio: {ratio:.2} ({harg_et0:.2} vs {pm_et0:.2})"
    );
}

// ── Sunshine-based radiation integration ─────────────────────────────

#[test]
fn test_sunshine_radiation_into_et0() {
    let lat_rad = 50.80_f64.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, 187);
    let n_hours = et::daylight_hours(lat_rad, 187);
    let rs = et::solar_radiation_from_sunshine(9.25, n_hours, ra);

    assert!(rs > 10.0 && rs < 30.0, "Rs from sunshine: {rs} MJ/m²/day");

    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: rs,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    };
    let result = et::daily_et0(&input);
    assert!(result.et0 > 0.0 && result.et0 < 8.0, "ET₀: {}", result.et0);
}

// ── Low-level FAO-56 PM + standalone water balance functions ────────

#[test]
fn test_lowlevel_pm_matches_daily_et0() {
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
    let result = et::daily_et0(&input);

    let pm = et::fao56_penman_monteith(
        result.rn,
        result.g,
        16.9,
        input.wind_speed_2m,
        result.vpd,
        result.delta,
        result.gamma,
    );
    assert!(
        (pm - result.et0).abs() < 1e-10,
        "Low-level PM={pm} vs daily_et0={}",
        result.et0
    );
}

#[test]
fn test_standalone_taw_raw_match_state() {
    let fc = 0.33;
    let wp = 0.13;
    let root_depth_mm = 600.0;
    let p = 0.5;

    let taw = wb::total_available_water(fc, wp, root_depth_mm);
    let raw = wb::readily_available_water(taw, p);

    let state = WaterBalanceState::new(fc, wp, root_depth_mm, p);
    assert!((taw - state.taw).abs() < f64::EPSILON, "TAW mismatch");
    assert!((raw - state.raw).abs() < f64::EPSILON, "RAW mismatch");
}

#[test]
fn test_standalone_stress_coefficient() {
    let taw = wb::total_available_water(0.30, 0.10, 500.0);
    let raw = wb::readily_available_water(taw, 0.5);

    assert!((wb::stress_coefficient(0.0, taw, raw) - 1.0).abs() < f64::EPSILON);
    assert!((wb::stress_coefficient(raw, taw, raw) - 1.0).abs() < f64::EPSILON);
    assert!((wb::stress_coefficient(taw, taw, raw)).abs() < f64::EPSILON);

    let mid = f64::midpoint(taw, raw);
    assert!(
        (wb::stress_coefficient(mid, taw, raw) - 0.5).abs() < 0.01,
        "Ks at midpoint"
    );
}

#[test]
fn test_standalone_daily_step() {
    let taw = wb::total_available_water(0.30, 0.10, 500.0);
    let (new_dr, actual_et, dp) = wb::daily_water_balance_step(20.0, 5.0, 0.0, 4.0, 1.0, 1.0, taw);

    assert!((new_dr - 19.0).abs() < 1e-10, "Dr_new={new_dr}");
    assert!((actual_et - 4.0).abs() < 1e-10, "ETa={actual_et}");
    assert!(dp.abs() < 1e-10, "DP should be 0");
}

// ── Sensor calibration integration ──────────────────────────────────

#[test]
fn test_soilwatch10_drives_irrigation_recommendation() {
    let raw_count = 15_000.0;
    let vwc = sc::soilwatch10_vwc(raw_count);

    assert!(vwc > 0.0 && vwc < 0.5, "VWC({raw_count}) = {vwc}");

    let ir = sc::irrigation_recommendation(0.12, vwc, 30.0);
    if vwc < 0.12 {
        assert!(ir > 0.0, "Should need irrigation: IR = {ir}");
    } else {
        assert!(ir.abs() < f64::EPSILON, "Should not need irrigation");
    }
}

#[test]
fn test_soilwatch10_multi_layer_integration() {
    let raw_counts = [12_000.0, 18_000.0, 22_000.0];
    let depths = [30.0, 30.0, 30.0];
    let field_capacities = [0.12, 0.15, 0.18];

    let layers: Vec<sc::SoilLayer> = raw_counts
        .iter()
        .zip(depths.iter())
        .zip(field_capacities.iter())
        .map(|((&rc, &d), &fc)| sc::SoilLayer {
            field_capacity: fc,
            current_vwc: sc::soilwatch10_vwc(rc),
            depth_cm: d,
        })
        .collect();

    let total_ir = sc::multi_layer_irrigation(&layers);
    assert!(total_ir >= 0.0, "Total IR must be non-negative: {total_ir}");
}

// ── Correction model integration tests ──────────────────────────────

#[test]
fn test_correction_fit_all_models() {
    let factory: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.02).collect();
    let measured: Vec<f64> = factory.iter().map(|&x| 1.1f64.mul_add(x, 0.02)).collect();

    let models = correction::fit_correction_equations(&factory, &measured);
    assert!(models.len() >= 2, "Should fit at least 2 models");

    let linear = models
        .iter()
        .find(|m| m.model_type == correction::ModelType::Linear)
        .unwrap();
    assert!(linear.r_squared > 0.99, "Linear R²={}", linear.r_squared);
}

#[test]
fn test_correction_evaluate_roundtrip() {
    let x: Vec<f64> = (1..=10).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0f64.mul_add(xi, 0.5)).collect();

    let model = correction::fit_linear(&x, &y).unwrap();
    for (&xi, &yi) in x.iter().zip(&y) {
        let predicted = correction::evaluate(&model, xi);
        assert!(
            (predicted - yi).abs() < 1e-6,
            "Evaluate({xi})={predicted} vs {yi}"
        );
    }
}

#[test]
fn test_correction_models_soil_calibration_pipeline() {
    let factory_vwc: Vec<f64> = (0..15).map(|i| f64::from(i).mul_add(0.02, 0.05)).collect();
    let true_vwc: Vec<f64> = factory_vwc
        .iter()
        .map(|&x| 0.85f64.mul_add(x, 0.03))
        .collect();

    let models = correction::fit_correction_equations(&factory_vwc, &true_vwc);
    assert!(!models.is_empty(), "Should fit at least one model");

    let best = models
        .iter()
        .max_by(|a, b| a.r_squared.partial_cmp(&b.r_squared).unwrap())
        .unwrap();

    let new_factory = 0.20;
    let corrected = correction::evaluate(best, new_factory);
    let expected = 0.85f64.mul_add(new_factory, 0.03);
    assert!(
        (corrected - expected).abs() < 0.01,
        "Corrected={corrected} vs expected={expected}"
    );
}

// ── Soil texture exhaustive coverage ─────────────────────────────────

#[test]
fn test_all_soil_textures_have_valid_properties() {
    let all_textures = [
        SoilTexture::Sand,
        SoilTexture::LoamySand,
        SoilTexture::SandyLoam,
        SoilTexture::Loam,
        SoilTexture::SiltLoam,
        SoilTexture::Silt,
        SoilTexture::SandyClayLoam,
        SoilTexture::ClayLoam,
        SoilTexture::SiltyClayLoam,
        SoilTexture::SandyClay,
        SoilTexture::SiltyClay,
        SoilTexture::Clay,
    ];

    for texture in &all_textures {
        let props = texture.hydraulic_properties();

        // FC must be positive and < 1.0
        assert!(
            props.field_capacity > 0.0 && props.field_capacity < 1.0,
            "{texture:?} FC={} out of range",
            props.field_capacity
        );

        // WP must be positive, less than FC
        assert!(
            props.wilting_point > 0.0 && props.wilting_point < props.field_capacity,
            "{texture:?} WP={} must be < FC={}",
            props.wilting_point,
            props.field_capacity
        );

        // Ksat must be positive
        assert!(
            props.ksat_mm_hr > 0.0,
            "{texture:?} Ksat={} must be positive",
            props.ksat_mm_hr
        );

        // Porosity must be > FC
        assert!(
            props.porosity > props.field_capacity,
            "{texture:?} porosity={} must be > FC={}",
            props.porosity,
            props.field_capacity
        );

        // PAW must be positive
        let paw = sm::plant_available_water(props.field_capacity, props.wilting_point, 500.0);
        assert!(paw > 0.0, "{texture:?} PAW={paw} must be positive");
    }
}

#[test]
fn test_soil_texture_ordering_ksat() {
    // Ksat should decrease from Sand → Clay (coarser = more permeable).
    let sand = SoilTexture::Sand.hydraulic_properties().ksat_mm_hr;
    let clay = SoilTexture::Clay.hydraulic_properties().ksat_mm_hr;
    assert!(
        sand > clay,
        "Sand Ksat ({sand}) should be > Clay Ksat ({clay})"
    );
}

// ── Additional crop/soil/correction edge cases ───────────────────────

#[test]
fn test_crop_coefficients_all_10_crops() {
    let crops = [
        CropType::Corn,
        CropType::Soybean,
        CropType::WinterWheat,
        CropType::Alfalfa,
        CropType::Tomato,
        CropType::Potato,
        CropType::SugarBeet,
        CropType::DryBean,
        CropType::Blueberry,
        CropType::Turfgrass,
    ];
    for crop in &crops {
        let kc = crop.coefficients();
        assert!(kc.kc_ini > 0.0, "{crop:?} kc_ini must be positive");
        assert!(kc.kc_ini < kc.kc_mid, "{crop:?} kc_ini < kc_mid");
        assert!(kc.kc_mid > 0.0, "{crop:?} kc_mid must be positive");
    }
}

#[test]
fn test_all_soil_textures_hydraulic_properties() {
    let textures = [
        SoilTexture::Sand,
        SoilTexture::LoamySand,
        SoilTexture::SandyLoam,
        SoilTexture::Loam,
        SoilTexture::SiltLoam,
        SoilTexture::Silt,
        SoilTexture::Clay,
    ];
    for tex in &textures {
        let props = tex.hydraulic_properties();
        assert!(
            props.field_capacity > props.wilting_point,
            "{tex:?} FC > WP"
        );
        assert!(props.field_capacity > 0.0, "{tex:?} FC > 0");
        assert!(props.wilting_point >= 0.0, "{tex:?} WP >= 0");
        assert!(props.ksat_mm_hr > 0.0, "{tex:?} Ksat > 0");
    }
}

#[test]
fn test_correction_quadratic_fit() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 0.5 * xi * xi + 2.0 * xi + 1.0).collect();
    let result = correction::fit_quadratic(&x, &y);
    assert!(result.is_some(), "quadratic fit should succeed");
    let fit = result.unwrap();
    assert!(
        fit.r_squared > 0.99,
        "perfect quadratic data R² should be ~1.0"
    );
}

#[test]
fn test_correction_exponential_fit() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (0.5_f64 * xi).exp()).collect();
    let result = correction::fit_exponential(&x, &y);
    assert!(result.is_some(), "exponential fit should succeed");
}

#[test]
fn test_correction_logarithmic_fit() {
    let x: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi.ln() + 1.0).collect();
    let result = correction::fit_logarithmic(&x, &y);
    assert!(result.is_some(), "logarithmic fit should succeed");
    let fit = result.unwrap();
    assert!(fit.r_squared > 0.99, "perfect log data R² should be ~1.0");
}

#[test]
fn test_sensor_calibration_boundary_raw_counts() {
    // Test boundary raw count values
    let vwc = sc::soilwatch10_vwc(0.0);
    assert!(vwc.is_finite(), "zero raw count should produce finite VWC");
    let vwc_high = sc::soilwatch10_vwc(10000.0);
    assert!(
        vwc_high.is_finite(),
        "high raw count should produce finite VWC"
    );
}
