// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark entry assembly — maps bench function bodies from [`bench_fns`]
//! into domain-grouped `BenchEntry` vectors for the harness.

use super::bench_fns;
use super::{BenchEntry, BenchFn};

macro_rules! entry {
    ($id:literal, $title:literal, $n:expr, $f:expr) => {
        ($id, $title, $n, Box::new($f) as BenchFn)
    };
}

fn et0_benchmarks() -> Vec<BenchEntry> {
    vec![
        entry!(
            "fao56_et0",
            "FAO-56 PM ET₀",
            10_000,
            bench_fns::bench_fao56_et0
        ),
        entry!(
            "thornthwaite",
            "Thornthwaite PET",
            10_000,
            bench_fns::bench_thornthwaite
        ),
        entry!(
            "hargreaves",
            "Hargreaves-Samani",
            10_000,
            bench_fns::bench_hargreaves
        ),
        entry!(
            "priestley_taylor",
            "Priestley-Taylor ET₀",
            10_000,
            bench_fns::bench_priestley_taylor
        ),
        entry!(
            "makkink_et0",
            "Makkink ET₀",
            100_000,
            bench_fns::bench_makkink_et0
        ),
        entry!(
            "blaney_criddle",
            "Blaney-Criddle ET₀",
            100_000,
            bench_fns::bench_blaney_criddle
        ),
    ]
}

fn soil_benchmarks() -> Vec<BenchEntry> {
    vec![
        entry!(
            "van_genuchten",
            "Van Genuchten θ(h)",
            100_000,
            bench_fns::bench_van_genuchten
        ),
        entry!(
            "saxton_rawls",
            "Saxton-Rawls Pedotransfer",
            100_000,
            bench_fns::bench_saxton_rawls
        ),
        entry!(
            "richards_1d",
            "Richards 1D (20 nodes)",
            1_000,
            bench_fns::bench_richards_1d
        ),
        entry!(
            "langmuir_fit",
            "Langmuir Isotherm Fit",
            10_000,
            bench_fns::bench_langmuir_fit
        ),
    ]
}

fn hydrology_benchmarks() -> Vec<BenchEntry> {
    vec![
        entry!(
            "water_balance_step",
            "Water Balance Step",
            10_000,
            bench_fns::bench_water_balance_step
        ),
        entry!(
            "scs_cn_runoff",
            "SCS-CN Runoff",
            100_000,
            bench_fns::bench_scs_cn_runoff
        ),
        entry!(
            "green_ampt",
            "Green-Ampt Infiltration",
            100_000,
            bench_fns::bench_green_ampt
        ),
    ]
}

fn crop_benchmarks() -> Vec<BenchEntry> {
    vec![
        entry!(
            "dual_kc_step",
            "Dual Kc (7-day sim)",
            10_000,
            bench_fns::bench_dual_kc_step
        ),
        entry!(
            "yield_response",
            "Stewart Yield Response",
            100_000,
            bench_fns::bench_yield_response
        ),
        entry!(
            "sensor_cal",
            "SensorCal VWC (op=5)",
            100_000,
            bench_fns::bench_sensor_cal
        ),
        entry!(
            "kc_climate_adjust",
            "Kc Climate Adj (op=7)",
            100_000,
            bench_fns::bench_kc_climate_adjust
        ),
    ]
}

fn ecology_benchmarks() -> Vec<BenchEntry> {
    vec![
        entry!(
            "shannon_diversity",
            "Shannon Diversity",
            10_000,
            bench_fns::bench_shannon_diversity
        ),
        entry!(
            "anderson_coupling",
            "Anderson Coupling",
            100_000,
            bench_fns::bench_anderson_coupling
        ),
        entry!(
            "tissue_w",
            "Tissue Anderson W (P12)",
            100_000,
            bench_fns::bench_tissue_w
        ),
        entry!(
            "barrier_d_eff",
            "Barrier d_eff (P12)",
            100_000,
            bench_fns::bench_barrier_d_eff
        ),
        entry!(
            "anderson_regime",
            "Anderson Regime (P12)",
            100_000,
            bench_fns::bench_anderson_regime
        ),
    ]
}

fn pipeline_benchmarks() -> Vec<BenchEntry> {
    vec![
        entry!(
            "season_simulation",
            "Season Sim (153d)",
            1_000,
            bench_fns::bench_season_simulation
        ),
        entry!(
            "seasonal_pipeline",
            "Seasonal Pipeline (153d)",
            1_000,
            bench_fns::bench_seasonal_pipeline
        ),
    ]
}

pub fn build_benchmarks() -> Vec<BenchEntry> {
    let mut all = Vec::with_capacity(24);
    all.extend(et0_benchmarks());
    all.extend(soil_benchmarks());
    all.extend(hydrology_benchmarks());
    all.extend(crop_benchmarks());
    all.extend(ecology_benchmarks());
    all.extend(pipeline_benchmarks());
    all
}
