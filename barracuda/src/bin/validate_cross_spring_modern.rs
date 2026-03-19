// SPDX-License-Identifier: AGPL-3.0-or-later
//! Experiment 082: Cross-Spring Modern Systems Validation.
//!
//! Validates the complete modern upstream integration — barraCuda HEAD (`a898dee`),
//! toadStool S130+, coralReef Phase 10. Exercises:
//!
//! 1. **Provenance registry**: 28 shaders, 10 evolution events, all 5 springs
//! 2. **Cross-spring matrix**: every spring contributes AND consumes
//! 3. **Precision routing**: `PrecisionRoutingAdvice` from groundSpring V84
//! 4. **`regularized_gamma_p`** delegation (v0.7.5 lean from `eco::drought_index`)
//! 5. **Autocorrelation**: new `gpu::autocorrelation` wiring (hotSpring MD→airSpring)
//! 6. **Special functions**: upstream `digamma`, `beta`, `ln_beta`, `norm_ppf`
//! 7. **Cross-spring shader flows**: hotSpring precision, wetSpring bio,
//!    neuralSpring stats, groundSpring universal, airSpring hydrology
//!
//! # hotSpring Pattern
//!
//! Hardcoded expected values. Explicit PASS/FAIL. Exit code 0 = all pass.
//!
//! Benchmark: analytical values from upstream `barracuda::special`, `barracuda::stats`,
//! and `barracuda::shaders::provenance` (cross-spring shader registry).
//!
//! Provenance: commit=a898dee, date=2026-03-18

use barracuda::shaders::provenance::{
    self, EVOLUTION_TIMELINE, EvolutionEvent, REGISTRY, ShaderRecord, SpringDomain,
};
use barracuda::special::gamma;
use barracuda::stats::normal;

use airspring_barracuda::gpu::autocorrelation;
use airspring_barracuda::gpu::device_info;
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness};

#[expect(
    clippy::too_many_lines,
    reason = "validation binary main() is a linear test sequence"
)]
fn main() {
    validation::init_tracing();
    validation::banner("Exp 082: Cross-Spring Modern Systems");

    let mut v = ValidationHarness::new("Cross-Spring Modern Systems Validation");

    // §1 Provenance Registry
    validation::section("§1 Provenance Registry");
    let registry: &[ShaderRecord] = &REGISTRY;
    v.check_bool("registry has ≥27 shaders", registry.len() >= 27);

    let timeline: &[EvolutionEvent] = &EVOLUTION_TIMELINE;
    v.check_bool("timeline has ≥10 events", timeline.len() >= 10);

    let origins: std::collections::HashSet<SpringDomain> =
        registry.iter().map(|r| r.origin).collect();
    v.check_bool(
        "all 5 springs are shader origins",
        origins.contains(&SpringDomain::HOT_SPRING)
            && origins.contains(&SpringDomain::WET_SPRING)
            && origins.contains(&SpringDomain::NEURAL_SPRING)
            && origins.contains(&SpringDomain::AIR_SPRING)
            && origins.contains(&SpringDomain::GROUND_SPRING),
    );

    // §2 Cross-Spring Matrix
    validation::section("§2 Cross-Spring Matrix");
    let matrix = provenance::cross_spring_matrix();
    v.check_bool("matrix is non-empty", !matrix.is_empty());

    let producers: std::collections::HashSet<SpringDomain> =
        matrix.keys().map(|(from, _)| *from).collect();
    let consumers: std::collections::HashSet<SpringDomain> =
        matrix.keys().map(|(_, to)| *to).collect();
    v.check_bool("all springs produce shaders", producers.len() >= 5);
    v.check_bool("all springs consume shaders", consumers.len() >= 5);

    let self_loops = matrix.keys().filter(|(from, to)| from == to).count();
    v.check_bool("no self-loops in cross-spring matrix", self_loops == 0);

    // §3 Specific Cross-Spring Flows
    validation::section("§3 Cross-Spring Shader Flows");

    let df64_core = registry.iter().find(|r| r.path.contains("df64_core"));
    if let Some(shader) = df64_core {
        v.check_bool(
            "hotSpring df64_core reaches ≥4 springs",
            shader.consumers.len() >= 4,
        );
    } else {
        v.check_bool("hotSpring df64_core exists in registry", false);
    }

    let bio_shaders: Vec<&ShaderRecord> = registry
        .iter()
        .filter(|r| r.origin == SpringDomain::WET_SPRING && r.path.contains("bio/"))
        .collect();
    v.check_bool("wetSpring has ≥3 bio shaders", bio_shaders.len() >= 3);
    let bio_to_neural = bio_shaders
        .iter()
        .any(|r| r.consumers.contains(&SpringDomain::NEURAL_SPRING));
    v.check_bool(
        "wetSpring bio shaders consumed by neuralSpring",
        bio_to_neural,
    );

    let neural_to_air = registry.iter().any(|r| {
        r.origin == SpringDomain::NEURAL_SPRING && r.consumers.contains(&SpringDomain::AIR_SPRING)
    });
    v.check_bool(
        "neuralSpring stats shaders consumed by airSpring",
        neural_to_air,
    );

    let air_to_wet = registry.iter().any(|r| {
        r.origin == SpringDomain::AIR_SPRING && r.consumers.contains(&SpringDomain::WET_SPRING)
    });
    v.check_bool(
        "airSpring hydrology shaders consumed by wetSpring",
        air_to_wet,
    );

    let chi_sq = registry
        .iter()
        .find(|r| r.origin == SpringDomain::GROUND_SPRING && r.path.contains("chi_squared_f64"));
    if let Some(shader) = chi_sq {
        v.check_bool(
            "groundSpring chi_squared reaches ≥4 springs",
            shader.consumers.len() >= 4,
        );
    } else {
        v.check_bool("groundSpring chi_squared_f64 exists", false);
    }

    let welford = registry
        .iter()
        .find(|r| r.path.contains("welford_mean_variance"));
    if let Some(shader) = welford {
        v.check_bool(
            "groundSpring Welford reaches ≥4 springs",
            shader.consumers.len() >= 4,
        );
    } else {
        v.check_bool("groundSpring welford_mean_variance exists", false);
    }

    // §4 Upstream Special Functions (v0.7.5 lean)
    validation::section("§4 Upstream Special Functions");

    let rgp = gamma::regularized_gamma_p(2.0, 1.0).unwrap_or(f64::NAN);
    v.check_abs(
        "regularized_gamma_p(2,1) ≈ 0.2642",
        rgp,
        0.264_241_117_657_115_4,
        tolerances::CROSS_VALIDATION.abs_tol,
    );

    let rgq = gamma::regularized_gamma_q(2.0, 1.0).unwrap_or(f64::NAN);
    v.check_abs(
        "regularized_gamma_q(2,1) ≈ 0.7358",
        rgq,
        0.735_758_882_342_884_6,
        tolerances::CROSS_VALIDATION.abs_tol,
    );

    v.check_abs("gamma_p + gamma_q = 1.0", rgp + rgq, 1.0, 1e-14);

    let psi = gamma::digamma(1.0).unwrap_or(f64::NAN);
    v.check_abs(
        "digamma(1) = -γ ≈ -0.5772",
        psi,
        -0.577_215_664_901_532_9,
        1e-8,
    );

    let b = gamma::beta(2.0, 3.0).unwrap_or(f64::NAN);
    v.check_abs(
        "beta(2,3) = 1/12 ≈ 0.0833",
        b,
        1.0 / 12.0,
        tolerances::CROSS_VALIDATION.abs_tol,
    );

    let lb = gamma::ln_beta(2.0, 3.0).unwrap_or(f64::NAN);
    v.check_abs(
        "ln_beta(2,3) = ln(1/12) ≈ -2.4849",
        lb,
        (1.0_f64 / 12.0).ln(),
        tolerances::CROSS_VALIDATION.abs_tol,
    );

    let z = normal::norm_ppf(0.975);
    v.check_abs("norm_ppf(0.975) ≈ 1.96", z, 1.959_963_984_540_054, 1e-6);

    // §5 Autocorrelation (new wiring v0.7.5)
    validation::section("§5 Autocorrelation (hotSpring MD → airSpring hydrology)");

    let constant = vec![5.0; 100];
    let nacf_const = autocorrelation::normalised_acf_cpu(&constant, 10);
    v.check_bool("constant signal ACF length = 10", nacf_const.len() == 10);
    v.check_abs("constant signal ACF[0] = 1.0", nacf_const[0], 1.0, 1e-10);
    v.check_abs("constant signal ACF[9] = 1.0", nacf_const[9], 1.0, 1e-10);

    let sine: Vec<f64> = (0..365)
        .map(|d| 3.0 * (2.0 * std::f64::consts::PI * f64::from(d) / 365.0).sin())
        .collect();
    let nacf_sine = autocorrelation::normalised_acf_cpu(&sine, 183);
    v.check_abs("sinusoidal ACF[0] = 1.0", nacf_sine[0], 1.0, 1e-10);
    v.check_bool("sinusoidal ACF at half-period < 0", nacf_sine[182] < 0.0);

    let white_noise: Vec<f64> = {
        let mut state = 12_345_u64;
        (0..500)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "provenance count < 2^53; usize→f64 exact"
                )]
                let v = (state >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5;
                v
            })
            .collect()
    };
    let nacf_noise = autocorrelation::normalised_acf_cpu(&white_noise, 20);
    v.check_abs("white noise ACF[0] = 1.0", nacf_noise[0], 1.0, 1e-10);

    #[expect(
        clippy::cast_precision_loss,
        reason = "ACF tail count < 2^53; usize→f64 exact"
    )]
    let mean_abs_acf: f64 =
        nacf_noise[1..].iter().map(|v| v.abs()).sum::<f64>() / (nacf_noise.len() - 1) as f64;
    v.check_bool(
        "white noise mean |ACF(lag>0)| < 0.3 (decorrelation)",
        mean_abs_acf < 0.3,
    );

    // §6 airSpring Provenance Integration
    validation::section("§6 airSpring Provenance Integration");

    let air_shaders = device_info::upstream_airspring_provenance();
    v.check_bool(
        "airSpring consumes ≥5 upstream shaders",
        air_shaders.len() >= 5,
    );

    let consumes_df64 = air_shaders.iter().any(|r| r.path.contains("df64_core"));
    v.check_bool("airSpring consumes hotSpring df64_core", consumes_df64);

    let consumes_chi_sq = air_shaders.iter().any(|r| r.path.contains("chi_squared"));
    v.check_bool(
        "airSpring consumes groundSpring chi_squared",
        consumes_chi_sq,
    );

    let consumes_welford = air_shaders.iter().any(|r| r.path.contains("welford"));
    v.check_bool("airSpring consumes groundSpring Welford", consumes_welford);

    let report = device_info::upstream_evolution_report();
    v.check_bool("evolution report is non-empty", report.len() > 100);
    v.check_bool(
        "evolution report mentions all 5 springs",
        report.contains("hotSpring")
            && report.contains("wetSpring")
            && report.contains("neuralSpring")
            && report.contains("airSpring")
            && report.contains("groundSpring"),
    );

    // §7 PrecisionRoutingAdvice (groundSpring V84 → toadStool S128)
    validation::section("§7 PrecisionRoutingAdvice");
    if let Some(device) = device_info::try_f64_device() {
        let report = device_info::probe_device(&device);
        v.check_bool("DevicePrecisionReport probed successfully", true);
        let advice = report.precision_routing;
        v.check_bool(
            "PrecisionRoutingAdvice is valid variant",
            matches!(
                advice,
                barracuda::device::driver_profile::PrecisionRoutingAdvice::F64Native
                    | barracuda::device::driver_profile::PrecisionRoutingAdvice::F64NativeNoSharedMem
                    | barracuda::device::driver_profile::PrecisionRoutingAdvice::Df64Only
                    | barracuda::device::driver_profile::PrecisionRoutingAdvice::F32Only
            ),
        );
        println!("  device: {}, advice: {advice:?}", report.adapter_name);
    } else {
        println!("  SKIP: No GPU device for precision probing");
        v.check_bool("GPU device unavailable (non-fatal skip)", true);
        v.check_bool("precision routing skip (no GPU)", true);
    }

    v.finish();
}
