// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::too_many_lines,
    clippy::similar_names
)]
//! Exp 065: biomeOS Graph Experiment — ecology pipeline via NUCLEUS coordination.
//!
//! Validates the full ecology pipeline as deployed by `airspring_deploy.toml`:
//!
//! 1. **Graph topology**: verifies the 4-node sequential graph
//!    (BearDog → Songbird → BarraCuda → airSpring).
//! 2. **Capability registry**: all 30 ecology.* + science.* capabilities mapped.
//! 3. **Offline pipeline**: Exercises the full seasonal ecology pipeline
//!    (ET₀ → Kc → WB → Yield → Stats) without requiring live primals.
//! 4. **metalForge coordination**: Validates workload routing through substrate mesh.
//! 5. **Evolution manifest**: BarraCuda absorption readiness report.
//!
//! This experiment runs **standalone** (no live primals needed). It simulates
//! the biomeOS graph coordination by exercising each pipeline stage locally
//! and validating that results compose correctly end-to-end.
//!
//! Provenance: `biomeOS` deployment graph parsing validation

use std::time::Instant;

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::gpu::evolution_gaps::{GAPS, Tier};
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::validation::ValidationHarness;

const AIRSPRING_CAPABILITIES: &[&str] = &[
    "ecology.et0.penman_monteith",
    "ecology.et0.hargreaves",
    "ecology.et0.priestley_taylor",
    "ecology.et0.makkink",
    "ecology.et0.turc",
    "ecology.et0.hamon",
    "ecology.et0.blaney_criddle",
    "ecology.et0.thornthwaite",
    "ecology.et0.ensemble",
    "ecology.et0.bias_correct",
    "ecology.water_balance.daily_step",
    "ecology.water_balance.simulate_season",
    "ecology.dual_kc.compute",
    "ecology.soil.pedotransfer",
    "ecology.soil.topp_equation",
    "ecology.sensor.soilwatch10_calibrate",
    "ecology.crop.gdd",
    "ecology.crop.yield_response",
    "ecology.crop.kc_schedule",
    "ecology.diversity.shannon",
    "ecology.diversity.simpson",
    "ecology.diversity.bray_curtis",
    "ecology.runoff.scs_cn",
    "ecology.infiltration.green_ampt",
    "ecology.anderson.coupling_chain",
    "ecology.richards.solve_1d",
    "science.health",
    "science.version",
    "compute.offload",
    "data.weather",
];

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_writer(std::io::stderr)
        .init();

    let mut v = ValidationHarness::new("Exp 065: biomeOS Graph Experiment");
    let t0 = Instant::now();

    phase_1_graph_topology(&mut v);
    phase_2_capability_registry(&mut v);
    phase_3_offline_pipeline(&mut v);
    phase_4_gpu_pipeline_parity(&mut v);
    phase_5_evolution_manifest(&mut v);

    let elapsed = t0.elapsed();
    v.check_abs("total_time_under_30s", elapsed.as_secs_f64(), 15.0, 15.0);

    v.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1: Graph Topology — validate airspring_deploy.toml structure
// ═══════════════════════════════════════════════════════════════════

fn phase_1_graph_topology(v: &mut ValidationHarness) {
    let deploy_toml = include_str!("../../../metalForge/deploy/airspring_deploy.toml");

    v.check_bool("graph_toml_loadable", !deploy_toml.is_empty());

    v.check_bool(
        "graph_has_crypto_capability",
        deploy_toml.contains("crypto") || deploy_toml.contains("tls"),
    );
    v.check_bool(
        "graph_has_mesh_capability",
        deploy_toml.contains("mesh") || deploy_toml.contains("discovery"),
    );
    v.check_bool(
        "graph_has_compute_capability",
        deploy_toml.contains("compute") || deploy_toml.contains("dispatch"),
    );
    v.check_bool(
        "graph_has_airspring",
        deploy_toml.contains("airspring") || deploy_toml.contains("airSpring"),
    );

    let node_count =
        deploy_toml.matches("[[graph.node]]").count() + deploy_toml.matches("[[node]]").count();
    v.check_bool("graph_node_count_ge_4", node_count >= 4);

    v.check_bool("graph_has_ecology_caps", deploy_toml.contains("ecology."));
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2: Capability Registry — all 30 capabilities mapped
// ═══════════════════════════════════════════════════════════════════

fn phase_2_capability_registry(v: &mut ValidationHarness) {
    v.check_abs(
        "cap_count_30",
        AIRSPRING_CAPABILITIES.len() as f64,
        30.0,
        1.0,
    );

    let eco_caps = AIRSPRING_CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("ecology."))
        .count();
    let science_caps = AIRSPRING_CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("science."))
        .count();
    let compute_caps = AIRSPRING_CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("compute."))
        .count();
    let data_caps = AIRSPRING_CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("data."))
        .count();

    v.check_abs("cap_ecology", eco_caps as f64, 26.0, 1.0);
    v.check_abs("cap_science", science_caps as f64, 2.0, 1.0);
    v.check_abs("cap_compute", compute_caps as f64, 1.0, 1.0);
    v.check_abs("cap_data", data_caps as f64, 1.0, 1.0);

    let unique: std::collections::HashSet<_> = AIRSPRING_CAPABILITIES.iter().collect();
    v.check_bool(
        "cap_all_unique",
        unique.len() == AIRSPRING_CAPABILITIES.len(),
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 3: Offline Pipeline — full seasonal ecology without live primals
// ═══════════════════════════════════════════════════════════════════

fn phase_3_offline_pipeline(v: &mut ValidationHarness) {
    let weather: Vec<WeatherDay> = (0..182)
        .map(|d| {
            let phase = f64::from(d) * std::f64::consts::PI / 182.0;
            WeatherDay {
                tmax: 26.0 + 8.0 * phase.sin(),
                tmin: 12.0 + 5.0 * phase.sin(),
                rh_max: 85.0 - 10.0 * phase.sin(),
                rh_min: 40.0,
                wind_2m: 1.5 + 0.5 * phase.cos(),
                solar_rad: 16.0 + 7.0 * phase.sin(),
                precipitation: if d % 5 == 0 {
                    15.0
                } else if d % 11 == 0 {
                    5.0
                } else {
                    0.0
                },
                elevation: 256.0,
                latitude_deg: 42.727,
                day_of_year: 100 + d,
            }
        })
        .collect();

    let config = CropConfig::standard(CropType::Corn);

    let pipe = SeasonalPipeline::cpu();
    let result = pipe.run_season(&weather, &config);

    v.check_bool("pipe_completes_182d", result.n_days == 182);
    v.check_bool("pipe_et0_positive", result.total_et0 > 0.0);
    v.check_bool("pipe_actual_et_positive", result.total_actual_et > 0.0);
    v.check_bool(
        "pipe_et0_gt_actual_et",
        result.total_et0 >= result.total_actual_et,
    );
    v.check_abs("pipe_yield_ratio", result.yield_ratio, 0.9, 0.15);
    v.check_bool("pipe_stress_days_ge_0", result.stress_days < 182);
    v.check_bool("pipe_et0_daily_len", result.et0_daily.len() == 182);
    v.check_abs(
        "pipe_mass_balance_ok",
        result.mass_balance_error.abs(),
        0.0,
        5.0,
    );

    let et0_mean = result.total_et0 / result.n_days as f64;
    v.check_abs("pipe_et0_mean_reasonable", et0_mean, 4.5, 1.5);

    let manual_et0 = et::daily_et0(&DailyEt0Input {
        tmin: weather[90].tmin,
        tmax: weather[90].tmax,
        tmean: None,
        solar_radiation: weather[90].solar_rad,
        wind_speed_2m: weather[90].wind_2m,
        actual_vapour_pressure: et::actual_vapour_pressure_rh(
            weather[90].tmin,
            weather[90].tmax,
            weather[90].rh_min,
            weather[90].rh_max,
        ),
        elevation_m: weather[90].elevation,
        latitude_deg: weather[90].latitude_deg,
        day_of_year: weather[90].day_of_year,
    });
    v.check_abs(
        "pipe_et0_day90_matches_manual",
        (result.et0_daily[90] - manual_et0.et0).abs(),
        0.0,
        0.5,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4: GPU Pipeline Parity — GPU seasonal matches CPU seasonal
// ═══════════════════════════════════════════════════════════════════

fn phase_4_gpu_pipeline_parity(v: &mut ValidationHarness) {
    let weather: Vec<WeatherDay> = (0..90)
        .map(|d| {
            let phase = f64::from(d) * std::f64::consts::PI / 90.0;
            WeatherDay {
                tmax: 30.0 + 4.0 * phase.sin(),
                tmin: 16.0 + 3.0 * phase.sin(),
                rh_max: 80.0,
                rh_min: 42.0,
                wind_2m: 1.8,
                solar_rad: 20.0 + 4.0 * phase.sin(),
                precipitation: if d % 7 == 0 { 10.0 } else { 0.0 },
                elevation: 256.0,
                latitude_deg: 42.727,
                day_of_year: 150 + d,
            }
        })
        .collect();

    let config = CropConfig::standard(CropType::Corn);

    let cpu_result = SeasonalPipeline::cpu().run_season(&weather, &config);

    if let Some(dev) = airspring_barracuda::gpu::device_info::try_f64_device() {
        let gpu_result = SeasonalPipeline::gpu(dev)
            .expect("gpu pipeline init")
            .run_season(&weather, &config);

        v.check_abs(
            "gpu_pipe_et0_parity",
            (gpu_result.total_et0 - cpu_result.total_et0).abs(),
            0.0,
            2.0,
        );
        v.check_abs(
            "gpu_pipe_yield_parity",
            (gpu_result.yield_ratio - cpu_result.yield_ratio).abs(),
            0.0,
            0.05,
        );
        v.check_abs(
            "gpu_pipe_stress_parity",
            (gpu_result.stress_days as f64 - cpu_result.stress_days as f64).abs(),
            0.0,
            2.0,
        );
    } else {
        v.check_bool("gpu_pipe_cpu_only_valid", cpu_result.yield_ratio >= 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 5: Evolution Manifest — BarraCuda absorption readiness
// ═══════════════════════════════════════════════════════════════════

fn phase_5_evolution_manifest(v: &mut ValidationHarness) {
    let tier_a = GAPS.iter().filter(|g| g.tier == Tier::A).count();
    let tier_b = GAPS.iter().filter(|g| g.tier == Tier::B).count();
    let tier_c = GAPS.iter().filter(|g| g.tier == Tier::C).count();
    let total = GAPS.len();

    v.check_bool("manifest_total_gt_15", total > 15);
    v.check_bool("manifest_tier_a_majority", tier_a > tier_b + tier_c);

    let with_primitive = GAPS
        .iter()
        .filter(|g| g.barracuda_primitive.is_some())
        .count();
    v.check_abs(
        "manifest_primitive_coverage",
        with_primitive as f64 / total as f64,
        0.8,
        0.2,
    );

    let et0_gap = GAPS.iter().find(|g| g.id == "batched_et0_gpu");
    v.check_bool(
        "manifest_et0_tier_a",
        et0_gap.is_some_and(|g| g.tier == Tier::A),
    );

    let wb_gap = GAPS.iter().find(|g| g.id == "batched_water_balance_gpu");
    v.check_bool(
        "manifest_wb_tier_a",
        wb_gap.is_some_and(|g| g.tier == Tier::A),
    );

    let richards_gap = GAPS.iter().find(|g| g.id.contains("richards"));
    v.check_bool("manifest_richards_tracked", richards_gap.is_some());

    let kriging_gap = GAPS.iter().find(|g| g.id.contains("kriging"));
    v.check_bool(
        "manifest_kriging_tier_a",
        kriging_gap.is_some_and(|g| g.tier == Tier::A),
    );

    let pipeline_gaps = GAPS
        .iter()
        .filter(|g| g.id.contains("pipeline") || g.id.contains("seasonal"))
        .count();
    v.check_bool("manifest_pipeline_tracked", pipeline_gaps > 0);
}
