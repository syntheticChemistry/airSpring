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

fn check(name: &str, ok: bool, pass: &mut u32, fail: &mut u32) {
    if ok {
        *pass += 1;
        println!("  [PASS] {name}");
    } else {
        *fail += 1;
        println!("  [FAIL] {name}");
    }
}

fn check_workload_routing(inv: &[Substrate], pass: &mut u32, fail: &mut u32) {
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
        check(
            &format!("{} → GPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
            pass,
            fail,
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
        check(
            &format!("{} → NPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
            pass,
            fail,
        );
    }

    println!("\n── CPU Workload Routing ──");
    let ew = workloads::weather_ingest();
    let r = dispatch::route(&ew.workload, inv);
    check(
        &format!("{} → CPU", ew.workload.name),
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
        pass,
        fail,
    );
}

fn check_priority_and_fallback(inv: &[Substrate], pass: &mut u32, fail: &mut u32) {
    println!("\n── Priority Chain ──");
    let r = dispatch::route(&workloads::et0_batch().workload, inv);
    check(
        "GPU preferred over Neural for F64 workloads",
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
        pass,
        fail,
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
    check(
        "ET₀ batch falls back to CPU when no GPU",
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
        pass,
        fail,
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
    check(
        "NPU workload fails when no NPU available",
        r.is_none(),
        pass,
        fail,
    );
}

fn check_reasons_and_inventory(inv: &[Substrate], pass: &mut u32, fail: &mut u32) {
    println!("\n── Dispatch Reason ──");
    let r =
        dispatch::route(&workloads::crop_stress_classifier().workload, inv).expect("should route");
    check(
        "NPU workload reports Preferred reason",
        r.reason == Reason::Preferred,
        pass,
        fail,
    );

    let r = dispatch::route(&workloads::et0_batch().workload, inv).expect("should route");
    check(
        "GPU workload reports BestAvailable reason",
        r.reason == Reason::BestAvailable,
        pass,
        fail,
    );

    println!("\n── Inventory Completeness ──");
    let all = workloads::all_workloads();
    check(
        &format!("{} workloads in catalog", all.len()),
        all.len() == 18,
        pass,
        fail,
    );

    let all_route = all
        .iter()
        .all(|ew| dispatch::route(&ew.workload, inv).is_some());
    check(
        "All 18 workloads route in full inventory",
        all_route,
        pass,
        fail,
    );

    let (absorbed, local, npu_native, cpu_only) = workloads::origin_summary();
    check(
        &format!(
            "9 absorbed + 4 local + 3 NPU + 2 CPU = {}",
            absorbed + local + npu_native + cpu_only
        ),
        absorbed == 9 && local == 4 && npu_native == 3 && cpu_only == 2,
        pass,
        fail,
    );
}

fn check_cross_system_routing(inv: &[Substrate], pass: &mut u32, fail: &mut u32) {
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

    check(
        "seasonal_pipeline → GPU",
        sub_seasonal == SubstrateKind::Gpu,
        pass,
        fail,
    );
    check(
        "crop_stress_classifier → NPU",
        sub_stress == SubstrateKind::Npu,
        pass,
        fail,
    );
    check(
        "weather_ingest → CPU",
        sub_ingest == SubstrateKind::Cpu,
        pass,
        fail,
    );

    let all_different =
        sub_seasonal != sub_stress && sub_stress != sub_ingest && sub_seasonal != sub_ingest;
    check(
        "pipeline routes to 3 different substrates",
        all_different,
        pass,
        fail,
    );

    let kinds: [SubstrateKind; 3] = [sub_seasonal, sub_stress, sub_ingest];
    let covers_all = kinds.contains(&SubstrateKind::Gpu)
        && kinds.contains(&SubstrateKind::Npu)
        && kinds.contains(&SubstrateKind::Cpu);
    check(
        "pipeline covers GPU + NPU + CPU",
        covers_all,
        pass,
        fail,
    );
}

fn check_benchmark_expectations(pass: &mut u32, fail: &mut u32) {
    let json: serde_json::Value =
        serde_json::from_str(BENCHMARK_JSON).expect("benchmark JSON must parse");

    println!("\n── Benchmark Provenance ──");

    let test_cases = json
        .get("validation_checks")
        .and_then(|vc| vc.get("workload_routing"))
        .and_then(|wr| wr.get("test_cases"))
        .and_then(|tc| tc.as_array());
    let test_cases_len = test_cases.map_or(0, Vec::len);

    check(
        &format!("workload_routing test cases >= 7 ({test_cases_len})"),
        test_cases_len >= 7,
        pass,
        fail,
    );

    let expected_count = json
        .get("validation_checks")
        .and_then(|vc| vc.get("inventory_completeness"))
        .and_then(|ic| ic.get("expected_count"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    check(
        &format!("inventory_completeness expected_count == 18 ({expected_count})"),
        expected_count == 18,
        pass,
        fail,
    );
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Exp 041: metalForge Mixed-Hardware Dispatch");
    println!("═══════════════════════════════════════════════════════════\n");

    let inv = full_inventory();
    let mut pass = 0_u32;
    let mut fail = 0_u32;

    check_workload_routing(&inv, &mut pass, &mut fail);
    check_priority_and_fallback(&inv, &mut pass, &mut fail);
    check_reasons_and_inventory(&inv, &mut pass, &mut fail);
    check_cross_system_routing(&inv, &mut pass, &mut fail);
    check_benchmark_expectations(&mut pass, &mut fail);

    let total = pass + fail;
    println!("\n=== metalForge Dispatch: {pass}/{total} PASS, {fail} FAIL ===");
    if fail > 0 {
        std::process::exit(1);
    }
}
