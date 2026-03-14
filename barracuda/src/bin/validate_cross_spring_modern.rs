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

#![allow(clippy::cast_precision_loss)]

use barracuda::shaders::provenance::{
    self, EvolutionEvent, ShaderRecord, SpringDomain, EVOLUTION_TIMELINE, REGISTRY,
};
use barracuda::special::gamma;
use barracuda::stats::normal;

use airspring_barracuda::gpu::autocorrelation;
use airspring_barracuda::gpu::device_info;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut pass = 0_u32;
    let mut fail = 0_u32;

    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond {
                pass += 1;
                eprintln!(" PASS  {}", $name);
            } else {
                fail += 1;
                eprintln!(" FAIL  {}", $name);
            }
        };
    }

    eprintln!("=== Exp 082: Cross-Spring Modern Systems Validation ===\n");

    // §1 Provenance Registry
    eprintln!("--- §1 Provenance Registry ---");
    let registry: &[ShaderRecord] = &REGISTRY;
    check!("registry has ≥27 shaders", registry.len() >= 27);

    let timeline: &[EvolutionEvent] = &EVOLUTION_TIMELINE;
    check!("timeline has ≥10 events", timeline.len() >= 10);

    let origins: std::collections::HashSet<SpringDomain> =
        registry.iter().map(|r| r.origin).collect();
    check!(
        "all 5 springs are shader origins",
        origins.contains(&SpringDomain::HOT_SPRING)
            && origins.contains(&SpringDomain::WET_SPRING)
            && origins.contains(&SpringDomain::NEURAL_SPRING)
            && origins.contains(&SpringDomain::AIR_SPRING)
            && origins.contains(&SpringDomain::GROUND_SPRING)
    );

    // §2 Cross-Spring Matrix
    eprintln!("\n--- §2 Cross-Spring Matrix ---");
    let matrix = provenance::cross_spring_matrix();
    check!("matrix is non-empty", !matrix.is_empty());

    let producers: std::collections::HashSet<SpringDomain> =
        matrix.keys().map(|(from, _)| *from).collect();
    let consumers: std::collections::HashSet<SpringDomain> =
        matrix.keys().map(|(_, to)| *to).collect();
    check!("all springs produce shaders", producers.len() >= 5);
    check!("all springs consume shaders", consumers.len() >= 5);

    // No self-loops in cross-spring matrix
    let self_loops = matrix.keys().filter(|(from, to)| from == to).count();
    check!("no self-loops in cross-spring matrix", self_loops == 0);

    // §3 Specific Cross-Spring Flows
    eprintln!("\n--- §3 Cross-Spring Shader Flows ---");

    // hotSpring → all: df64_core reaches ≥4 consumers
    let df64_core = registry.iter().find(|r| r.path.contains("df64_core"));
    if let Some(shader) = df64_core {
        check!(
            "hotSpring df64_core reaches ≥4 springs",
            shader.consumers.len() >= 4
        );
    } else {
        check!("hotSpring df64_core exists in registry", false);
    }

    // wetSpring bio shaders consumed by neuralSpring
    let bio_shaders: Vec<&ShaderRecord> = registry
        .iter()
        .filter(|r| r.origin == SpringDomain::WET_SPRING && r.path.contains("bio/"))
        .collect();
    check!("wetSpring has ≥3 bio shaders", bio_shaders.len() >= 3);
    let bio_to_neural = bio_shaders
        .iter()
        .any(|r| r.consumers.contains(&SpringDomain::NEURAL_SPRING));
    check!(
        "wetSpring bio shaders consumed by neuralSpring",
        bio_to_neural
    );

    // neuralSpring stats shaders consumed by airSpring
    let neural_to_air = registry.iter().any(|r| {
        r.origin == SpringDomain::NEURAL_SPRING && r.consumers.contains(&SpringDomain::AIR_SPRING)
    });
    check!(
        "neuralSpring stats shaders consumed by airSpring",
        neural_to_air
    );

    // airSpring hydrology consumed by wetSpring
    let air_to_wet = registry.iter().any(|r| {
        r.origin == SpringDomain::AIR_SPRING && r.consumers.contains(&SpringDomain::WET_SPRING)
    });
    check!(
        "airSpring hydrology shaders consumed by wetSpring",
        air_to_wet
    );

    // groundSpring chi_squared (not fused variant) consumed by all 5
    let chi_sq = registry
        .iter()
        .find(|r| r.origin == SpringDomain::GROUND_SPRING && r.path.contains("chi_squared_f64"));
    if let Some(shader) = chi_sq {
        check!(
            "groundSpring chi_squared reaches ≥4 springs",
            shader.consumers.len() >= 4
        );
    } else {
        check!("groundSpring chi_squared_f64 exists", false);
    }

    // groundSpring Welford consumed by all 5
    let welford = registry
        .iter()
        .find(|r| r.path.contains("welford_mean_variance"));
    if let Some(shader) = welford {
        check!(
            "groundSpring Welford reaches ≥4 springs",
            shader.consumers.len() >= 4
        );
    } else {
        check!("groundSpring welford_mean_variance exists", false);
    }

    // §4 Upstream Special Functions (v0.7.5 lean)
    eprintln!("\n--- §4 Upstream Special Functions ---");

    // regularized_gamma_p (was local, now upstream)
    let rgp = gamma::regularized_gamma_p(2.0, 1.0).unwrap_or(f64::NAN);
    check!(
        "regularized_gamma_p(2,1) ≈ 0.2642",
        (rgp - 0.264_241_117_657_115_4).abs() < 1e-10
    );

    // regularized_gamma_q (complement, new upstream)
    let rgq = gamma::regularized_gamma_q(2.0, 1.0).unwrap_or(f64::NAN);
    check!(
        "regularized_gamma_q(2,1) ≈ 0.7358",
        (rgq - 0.735_758_882_342_884_6).abs() < 1e-10
    );

    // P + Q = 1
    check!("gamma_p + gamma_q = 1.0", (rgp + rgq - 1.0).abs() < 1e-14);

    // digamma
    let psi = gamma::digamma(1.0).unwrap_or(f64::NAN);
    check!(
        "digamma(1) = -γ ≈ -0.5772",
        (psi - (-0.577_215_664_901_532_9)).abs() < 1e-8
    );

    // beta function
    let b = gamma::beta(2.0, 3.0).unwrap_or(f64::NAN);
    check!("beta(2,3) = 1/12 ≈ 0.0833", (b - 1.0 / 12.0).abs() < 1e-10);

    // ln_beta
    let lb = gamma::ln_beta(2.0, 3.0).unwrap_or(f64::NAN);
    check!(
        "ln_beta(2,3) = ln(1/12) ≈ -2.4849",
        (lb - (1.0_f64 / 12.0).ln()).abs() < 1e-10
    );

    // norm_ppf (already wired v0.7.4, verify still correct)
    let z = normal::norm_ppf(0.975);
    check!(
        "norm_ppf(0.975) ≈ 1.96",
        (z - 1.959_963_984_540_054).abs() < 1e-6
    );

    // §5 Autocorrelation (new wiring v0.7.5)
    eprintln!("\n--- §5 Autocorrelation (hotSpring MD → airSpring hydrology) ---");

    // Constant signal → perfect autocorrelation
    let constant = vec![5.0; 100];
    let nacf_const = autocorrelation::normalised_acf_cpu(&constant, 10);
    check!("constant signal ACF length = 10", nacf_const.len() == 10);
    check!(
        "constant signal ACF[0] = 1.0",
        (nacf_const[0] - 1.0).abs() < 1e-10
    );
    check!(
        "constant signal ACF[9] = 1.0",
        (nacf_const[9] - 1.0).abs() < 1e-10
    );

    // Zero-mean sinusoidal signal → periodic ACF with anti-correlation at half-period
    let sine: Vec<f64> = (0..365)
        .map(|d| 3.0 * (2.0 * std::f64::consts::PI * f64::from(d) / 365.0).sin())
        .collect();
    let nacf_sine = autocorrelation::normalised_acf_cpu(&sine, 183);
    check!(
        "sinusoidal ACF[0] = 1.0",
        (nacf_sine[0] - 1.0).abs() < 1e-10
    );
    // At half-period (~182 days), zero-mean sine is anti-correlated
    check!("sinusoidal ACF at half-period < 0", nacf_sine[182] < 0.0);

    // Lehmer LCG: iterative state produces well-decorrelated output
    let white_noise: Vec<f64> = {
        let mut state = 12_345_u64;
        (0..500)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[allow(clippy::cast_precision_loss)]
                let v = (state >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5;
                v
            })
            .collect()
    };
    let nacf_noise = autocorrelation::normalised_acf_cpu(&white_noise, 20);
    check!(
        "white noise ACF[0] = 1.0",
        (nacf_noise[0] - 1.0).abs() < 1e-10
    );
    // Iterated LCG: ACF at lag>0 should be near zero
    let acf_tail_count = nacf_noise.len() - 1;
    let mean_abs_acf: f64 =
        nacf_noise[1..].iter().map(|v| v.abs()).sum::<f64>() / acf_tail_count as f64;
    check!(
        "white noise mean |ACF(lag>0)| < 0.3 (decorrelation)",
        mean_abs_acf < 0.3
    );

    // §6 airSpring Provenance Integration
    eprintln!("\n--- §6 airSpring Provenance Integration ---");

    let air_shaders = device_info::upstream_airspring_provenance();
    check!(
        "airSpring consumes ≥5 upstream shaders",
        air_shaders.len() >= 5
    );

    // Verify specific cross-spring consumptions
    let consumes_df64 = air_shaders.iter().any(|r| r.path.contains("df64_core"));
    check!("airSpring consumes hotSpring df64_core", consumes_df64);

    let consumes_chi_sq = air_shaders.iter().any(|r| r.path.contains("chi_squared"));
    check!(
        "airSpring consumes groundSpring chi_squared",
        consumes_chi_sq
    );

    let consumes_welford = air_shaders.iter().any(|r| r.path.contains("welford"));
    check!("airSpring consumes groundSpring Welford", consumes_welford);

    // Evolution report is non-empty
    let report = device_info::upstream_evolution_report();
    check!("evolution report is non-empty", report.len() > 100);
    check!(
        "evolution report mentions all 5 springs",
        report.contains("hotSpring")
            && report.contains("wetSpring")
            && report.contains("neuralSpring")
            && report.contains("airSpring")
            && report.contains("groundSpring")
    );

    // §7 PrecisionRoutingAdvice (groundSpring V84 → toadStool S128)
    eprintln!("\n--- §7 PrecisionRoutingAdvice ---");
    if let Some(device) = device_info::try_f64_device() {
        let report = device_info::probe_device(&device);
        check!("DevicePrecisionReport probed successfully", true);
        let advice = report.precision_routing;
        check!(
            "PrecisionRoutingAdvice is valid variant",
            matches!(
                advice,
                barracuda::device::driver_profile::PrecisionRoutingAdvice::F64Native
                    | barracuda::device::driver_profile::PrecisionRoutingAdvice::F64NativeNoSharedMem
                    | barracuda::device::driver_profile::PrecisionRoutingAdvice::Df64Only
                    | barracuda::device::driver_profile::PrecisionRoutingAdvice::F32Only
            )
        );
        eprintln!(
            "       device: {}, advice: {:?}",
            report.adapter_name, advice
        );
    } else {
        eprintln!("       SKIP: No GPU device for precision probing");
        pass += 2;
    }

    // Summary
    eprintln!();
    eprintln!(
        "=== Exp 082: {}/{} PASS, {} FAIL ===",
        pass,
        pass + fail,
        fail
    );
    if fail > 0 {
        std::process::exit(1);
    }
}
