// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paper 12: Immunological Anderson — tissue diversity + `CytokineBrain`.

use airspring_barracuda::eco::cytokine::{CytokineBrain, CytokineBrainConfig, CytokineObservation};
use airspring_barracuda::eco::tissue::{
    AndersonRegime, CellTypeAbundance, SkinCompartment, analyze_tissue_disorder,
    barrier_disruption_d_eff,
};
use airspring_barracuda::gpu::diversity::GpuDiversity;
use barracuda::validation::ValidationHarness;
use std::time::Instant;

pub fn bench_paper12_immunological(v: &mut ValidationHarness) {
    println!("\n── Paper 12: Immunological Anderson ──");
    let t0 = Instant::now();

    let engine = GpuDiversity::cpu();

    bench_tissue_disorder(v, &engine);
    bench_barrier_disruption(v);
    bench_cytokine_brain(v);

    println!("  Paper 12 immunological: {:.1?}", t0.elapsed());
}

fn bench_tissue_disorder(v: &mut ValidationHarness, engine: &GpuDiversity) {
    let epidermis_cells: Vec<CellTypeAbundance> = [85.0, 5.0, 8.0, 2.0]
        .iter()
        .enumerate()
        .map(|(i, &a)| CellTypeAbundance {
            cell_type: format!("type_{i}"),
            abundance: a,
        })
        .collect();

    let result = analyze_tissue_disorder(&epidermis_cells, SkinCompartment::Epidermis, engine)
        .expect("tissue analysis should succeed");

    v.check_lower(
        "Paper 12: epidermis Shannon > 0 [Pielou→W mapping]",
        result.diversity.shannon,
        0.0,
    );
    v.check_lower(
        "Paper 12: epidermis evenness < 1.0 (keratinocyte dominated)",
        1.0 - result.diversity.evenness,
        0.0,
    );
    v.check_lower(
        "Paper 12: epidermis W > 0 (non-uniform cell types)",
        result.w_effective,
        0.0,
    );

    let dermis_cells: Vec<CellTypeAbundance> = [20.0, 15.0, 12.0, 10.0, 8.0, 5.0, 10.0, 8.0, 12.0]
        .iter()
        .enumerate()
        .map(|(i, &a)| CellTypeAbundance {
            cell_type: format!("type_{i}"),
            abundance: a,
        })
        .collect();

    let dermis_result =
        analyze_tissue_disorder(&dermis_cells, SkinCompartment::PapillaryDermis, engine)
            .expect("dermis analysis should succeed");

    v.check_lower(
        "Paper 12: dermis evenness > epidermis (diverse cell pop)",
        dermis_result.diversity.evenness - result.diversity.evenness,
        0.0,
    );
}

fn bench_barrier_disruption(v: &mut ValidationHarness) {
    let d_intact = barrier_disruption_d_eff(0.0);
    let d_breached = barrier_disruption_d_eff(1.0);
    v.check_abs(
        "Paper 12: intact barrier d_eff = 2.0 (2D epidermis)",
        d_intact,
        2.0,
        1e-10,
    );
    v.check_abs(
        "Paper 12: full breach d_eff = 3.0 (dimensional promotion)",
        d_breached,
        3.0,
        1e-10,
    );

    v.check_abs(
        "Paper 12: Epidermis d=2 [Anderson prediction]",
        SkinCompartment::Epidermis.effective_dimension_intact(),
        2.0,
        1e-10,
    );
    v.check_abs(
        "Paper 12: PapillaryDermis d=3 [Anderson prediction]",
        SkinCompartment::PapillaryDermis.effective_dimension_intact(),
        3.0,
        1e-10,
    );
}

fn bench_cytokine_brain(v: &mut ValidationHarness) {
    let config = CytokineBrainConfig::default();
    let mut brain = CytokineBrain::new(config, "bench-paper12");
    for i in 0..10 {
        let fi = f64::from(i);
        brain.observe(CytokineObservation {
            time_hours: fi * 6.0,
            il31_level: fi.mul_add(20.0, 100.0),
            il4_level: 50.0,
            il13_level: 40.0,
            pruritus_score: fi.mul_add(0.5, 3.0),
            tewl: 25.0,
            pielou_evenness: 0.7,
            signal_extent_observed: fi.mul_add(0.05, 0.3),
            w_observed: 0.4,
            barrier_integrity_observed: fi.mul_add(-0.03, 0.8),
        });
    }
    let mse = brain.train();
    v.check_bool("Paper 12: CytokineBrain trains successfully", mse.is_some());
    v.check_bool(
        "Paper 12: CytokineBrain is_trained after train()",
        brain.is_trained(),
    );

    if let Some(pred) = brain.predict(&CytokineObservation {
        time_hours: 48.0,
        il31_level: 200.0,
        il4_level: 50.0,
        il13_level: 40.0,
        pruritus_score: 5.0,
        tewl: 30.0,
        pielou_evenness: 0.7,
        signal_extent_observed: 0.0,
        w_observed: 0.0,
        barrier_integrity_observed: 0.0,
    }) {
        v.check_bool(
            "Paper 12: prediction signal_extent in [0,1]",
            (0.0..=1.0).contains(&pred.signal_extent),
        );
        v.check_bool(
            "Paper 12: prediction barrier in [0,1]",
            (0.0..=1.0).contains(&pred.barrier_integrity),
        );

        let regime = pred.anderson_regime();
        v.check_bool(
            "Paper 12: regime classification valid",
            matches!(
                regime,
                AndersonRegime::Extended | AndersonRegime::Localized | AndersonRegime::Critical
            ),
        );
    }

    let json = brain.export_json().expect("export should succeed");
    v.check_bool("Paper 12: shell export non-empty", !json.is_empty());
}
