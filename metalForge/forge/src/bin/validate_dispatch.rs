// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
//! Exp 041: metalForge Mixed-Hardware Dispatch Validation.
//!
//! Tests actual Rust `dispatch::route()` logic with synthetic inventories.
//! Validates GPU > NPU > Neural > CPU priority, capability matching, and fallback.
//!
//! Benchmark: `control/metalforge_dispatch/benchmark_metalforge_dispatch.json`
//! Baseline: `control/metalforge_dispatch/metalforge_dispatch.py` (14/14 PASS)

use airspring_forge::dispatch::{self, Reason};
use airspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use airspring_forge::workloads;
use barracuda::validation::ValidationHarness;

const BENCHMARK_JSON: &str =
    include_str!("../../../../control/metalforge_dispatch/benchmark_metalforge_dispatch.json");

fn make_sub(kind: SubstrateKind, name: &str, caps: Vec<Capability>) -> Substrate {
    Substrate {
        kind,
        identity: Identity::named(name),
        properties: Properties::default(),
        capabilities: caps,
    }
}

fn full_inventory() -> Vec<Substrate> {
    vec![
        make_sub(
            SubstrateKind::Gpu,
            "Titan V GPU",
            vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::ShaderDispatch,
                Capability::ScalarReduce,
                Capability::TimestampQuery,
            ],
        ),
        make_sub(
            SubstrateKind::Npu,
            "AKD1000 NPU",
            vec![
                Capability::F32Compute,
                Capability::QuantizedInference { bits: 8 },
                Capability::BatchInference { max_batch: 8 },
            ],
        ),
        make_sub(
            SubstrateKind::Neural,
            "biomeOS Neural API",
            vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::NeuralApiRoute,
            ],
        ),
        make_sub(
            SubstrateKind::Cpu,
            "AMD Ryzen 7",
            vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::CpuCompute,
                Capability::SimdVector,
            ],
        ),
    ]
}

fn check_workload_routing(inv: &[Substrate], v: &mut ValidationHarness) {
    println!("── GPU Workload Routing ──");
    let gpu_wls = [
        workloads::et0_batch(),
        workloads::water_balance_batch(),
        workloads::richards_pde(),
        workloads::yield_response_surface(),
        workloads::monte_carlo_uq(),
        workloads::isotherm_batch(),
        workloads::hargreaves_et0_batch(),
        workloads::kc_climate_batch(),
        workloads::sensor_calibration_batch(),
        workloads::seasonal_pipeline(),
    ];
    for ew in &gpu_wls {
        let r = dispatch::route(&ew.workload, inv);
        v.check_bool(
            &format!("{} → GPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
        );
    }

    println!("\n── NPU Workload Routing ──");
    let npu_wls = [
        workloads::crop_stress_classifier(),
        workloads::irrigation_decision(),
        workloads::sensor_anomaly(),
    ];
    for ew in &npu_wls {
        let r = dispatch::route(&ew.workload, inv);
        v.check_bool(
            &format!("{} → NPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
        );
    }

    println!("\n── CPU Workload Routing ──");
    let ew = workloads::weather_ingest();
    let r = dispatch::route(&ew.workload, inv);
    v.check_bool(
        &format!("{} → CPU", ew.workload.name),
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
    );
}

fn check_priority_and_fallback(inv: &[Substrate], v: &mut ValidationHarness) {
    println!("\n── Priority Chain ──");
    let r = dispatch::route(&workloads::et0_batch().workload, inv);
    v.check_bool(
        "GPU preferred over Neural for F64 workloads",
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
    );

    println!("\n── Fallback Behavior ──");
    let cpu_only_inv = vec![make_sub(
        SubstrateKind::Cpu,
        "CPU only",
        vec![
            Capability::F64Compute,
            Capability::CpuCompute,
            Capability::ShaderDispatch,
        ],
    )];
    let r = dispatch::route(&workloads::et0_batch().workload, &cpu_only_inv);
    v.check_bool(
        "ET₀ batch falls back to CPU when no GPU",
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
    );

    let gpu_cpu_only = vec![
        make_sub(
            SubstrateKind::Gpu,
            "GPU",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        make_sub(
            SubstrateKind::Cpu,
            "CPU",
            vec![Capability::F64Compute, Capability::CpuCompute],
        ),
    ];
    let r = dispatch::route(&workloads::crop_stress_classifier().workload, &gpu_cpu_only);
    v.check_bool("NPU workload fails when no NPU available", r.is_none());
}

fn check_reasons_and_inventory(inv: &[Substrate], v: &mut ValidationHarness) {
    println!("\n── Dispatch Reason ──");
    let r =
        dispatch::route(&workloads::crop_stress_classifier().workload, inv).expect("should route");
    v.check_bool(
        "NPU workload reports Preferred reason",
        r.reason == Reason::Preferred,
    );

    let r = dispatch::route(&workloads::et0_batch().workload, inv).expect("should route");
    v.check_bool(
        "GPU workload reports BestAvailable reason",
        r.reason == Reason::BestAvailable,
    );

    println!("\n── Inventory Completeness ──");
    let all = workloads::all_workloads();
    v.check_bool(
        &format!("{} workloads in catalog", all.len()),
        all.len() == 18,
    );

    let all_route = all
        .iter()
        .all(|ew| dispatch::route(&ew.workload, inv).is_some());
    v.check_bool("All 18 workloads route in full inventory", all_route);

    let (absorbed, local, npu_native, cpu_only) = workloads::origin_summary();
    v.check_bool(
        &format!(
            "9 absorbed + 4 local + 3 NPU + 2 CPU = {}",
            absorbed + local + npu_native + cpu_only
        ),
        absorbed == 9 && local == 4 && npu_native == 3 && cpu_only == 2,
    );
}

fn check_cross_system_routing(inv: &[Substrate], v: &mut ValidationHarness) {
    println!("\n── Cross-System Pipeline ──");
    let seasonal = dispatch::route(&workloads::seasonal_pipeline().workload, inv)
        .expect("seasonal_pipeline should route");
    let stress = dispatch::route(&workloads::crop_stress_classifier().workload, inv)
        .expect("crop_stress_classifier should route");
    let ingest = dispatch::route(&workloads::weather_ingest().workload, inv)
        .expect("weather_ingest should route");

    let sub_seasonal = seasonal.substrate.kind;
    let sub_stress = stress.substrate.kind;
    let sub_ingest = ingest.substrate.kind;

    v.check_bool(
        "seasonal_pipeline → GPU",
        sub_seasonal == SubstrateKind::Gpu,
    );
    v.check_bool(
        "crop_stress_classifier → NPU",
        sub_stress == SubstrateKind::Npu,
    );
    v.check_bool("weather_ingest → CPU", sub_ingest == SubstrateKind::Cpu);

    let all_different =
        sub_seasonal != sub_stress && sub_stress != sub_ingest && sub_seasonal != sub_ingest;
    v.check_bool("pipeline routes to 3 different substrates", all_different);

    let kinds: [SubstrateKind; 3] = [sub_seasonal, sub_stress, sub_ingest];
    let covers_all = kinds.contains(&SubstrateKind::Gpu)
        && kinds.contains(&SubstrateKind::Npu)
        && kinds.contains(&SubstrateKind::Cpu);
    v.check_bool("pipeline covers GPU + NPU + CPU", covers_all);
}

fn check_benchmark_expectations(v: &mut ValidationHarness) {
    let json: serde_json::Value =
        serde_json::from_str(BENCHMARK_JSON).expect("benchmark JSON must parse");

    println!("\n── Benchmark Provenance ──");

    let test_cases = json
        .get("validation_checks")
        .and_then(|vc| vc.get("workload_routing"))
        .and_then(|wr| wr.get("test_cases"))
        .and_then(|tc| tc.as_array());
    let test_cases_len = test_cases.map_or(0, Vec::len);

    v.check_bool(
        &format!("workload_routing test cases >= 7 ({test_cases_len})"),
        test_cases_len >= 7,
    );

    let expected_count = json
        .get("validation_checks")
        .and_then(|vc| vc.get("inventory_completeness"))
        .and_then(|ic| ic.get("expected_count"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    v.check_bool(
        &format!("inventory_completeness expected_count == 18 ({expected_count})"),
        expected_count == 18,
    );
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Exp 041: metalForge Mixed-Hardware Dispatch");
    println!("═══════════════════════════════════════════════════════════\n");

    let inv = full_inventory();
    let mut v = ValidationHarness::new("metalForge Dispatch");

    check_workload_routing(&inv, &mut v);
    check_priority_and_fallback(&inv, &mut v);
    check_reasons_and_inventory(&inv, &mut v);
    check_cross_system_routing(&inv, &mut v);
    check_benchmark_expectations(&mut v);

    v.finish();
}
