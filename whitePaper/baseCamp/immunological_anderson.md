# airSpring Contributions to Paper 12: Immunological Anderson

**Updated**: March 2, 2026 (v0.6.3)
**Parent**: `gen3/baseCamp/12_immunological_anderson.md`
**Status**: All 4 experiments validated (Exp 066-069) — tissue diversity, CytokineBrain, barrier state, cross-species

---

## airSpring Role in Paper 12

Paper 12 extends Anderson localization from microbial QS (Papers 01, 05, 06)
to immunological cytokine signaling. airSpring contributes three specific
computational capabilities:

### 1. Cell-Type Diversity as Disorder W (`GpuDiversity`)

The Anderson disorder parameter W maps to cell-type heterogeneity in tissue.
airSpring's `gpu::diversity::GpuDiversity` computes Shannon, Simpson, and
Pielou evenness on GPU — the same metrics that quantify W for skin cell
populations (keratinocytes, Th2, mast cells, eosinophils, neurons).

| Metric | Anderson Meaning | Computation |
|--------|-----------------|-------------|
| Shannon H' | Total disorder strength | `GpuDiversity::compute_alpha()` → `DiversityMetrics::shannon` |
| Pielou J' | Disorder uniformity (0=one type dominates, 1=all equal) | `GpuDiversity::compute_alpha()` → `DiversityMetrics::pielou_evenness` |
| Simpson D | Probability two random cells are same type | `GpuDiversity::compute_alpha()` → `DiversityMetrics::simpson` |

W ∝ (1 - Pielou J'): high evenness = low effective disorder (all cell types
equally represented = periodic), low evenness = high disorder (one type dominates
= heterogeneous).

### 2. Regime Change Detection (`DriftMonitor` / `CytokineBrain`)

The `AirSpringBrain` pattern (Nautilus evolutionary reservoir) directly
extends to cytokine time series. A `CytokineBrain` with 3 heads:

| Head | Input Features | Target | Anderson Meaning |
|------|---------------|--------|-----------------|
| IL-31 propagation | IL-31 serum level, tissue depth, barrier integrity | Signal extent (localized vs propagating) | Anderson transport: ξ/L ratio |
| Cell diversity | Single-cell transcriptomics → cell-type abundances | Pielou W | Disorder parameter |
| Barrier state | TEWL, filaggrin expression, scratch score | Effective dimension d_eff | 2D (intact) vs 3D (breached) |

The `DriftMonitor` from `bingocube-nautilus` detects when `N_e * s` drops below
threshold — in the immunological context, this flags AD flare onset (regime
transition from localized to propagating cytokine state).

### 3. Dimensional Promotion/Collapse Duality (Paper 06 ↔ Paper 12)

airSpring owns Paper 06 (no-till Anderson) and provides the computational
framework for the dimensional promotion/collapse duality:

```
Paper 06 (soil):  Tillage = d collapse (3D→2D) → QS FAILS    → ecosystem loss
Paper 12 (skin):  Scratch = d promotion (2D→3D) → cytokine PROPAGATES → AD amplification
```

Same `coupling_chain` and `coupling_series` functions from
`validate_ncbi_16s_coupling.rs` apply to cytokine propagation through tissue
channels. The GPU van Genuchten θ(h)/K(h) (ops 9-10) models fluid transport
in porous media — skin ECM is a biological porous medium where cytokine
diffusion follows analogous physics.

---

## airSpring Experiments (Validated)

| Exp | Description | Module | Checks | Status |
|-----|-------------|--------|:------:|:------:|
| Exp 066 | Tissue cell-type diversity profiling: Pielou W from single-cell data across healthy/AD/chronic skin states | `eco::tissue` | 30+30 | **Complete** |
| Exp 067 | Cytokine regime detection: CytokineBrain trained on IL-31 time series, DriftMonitor flags flare onset | `eco::cytokine` + `nautilus` | 14+28 | **Complete** |
| Exp 068 | Barrier state model: VG θ(h)/K(h) applied to skin barrier integrity as cytokine transport analogue | `eco::van_genuchten` + `eco::tissue` | 16+16 | **Complete** |
| Exp 069 | Cross-species skin comparison: canine/human/feline Anderson predictions, One Health bridge | `eco::diversity` + `eco::tissue` | 19+20 | **Complete** |

---

## Infrastructure (v0.6.3)

| Module | Paper 12 Application | Status |
|--------|---------------------|--------|
| `gpu::diversity` (GpuDiversity) | Pielou evenness → tissue W | Integrated (v0.6.1) |
| `nautilus` (AirSpringBrain) | CytokineBrain pattern for AD flare prediction | Integrated (v0.6.2) |
| `gpu::atlas_stream` (MonitoredAtlasStream) | DriftMonitor pattern for cytokine regime changes | Integrated (v0.6.2) |
| `gpu::van_genuchten` (BatchedVanGenuchten) | Barrier permeability model | Integrated (v0.6.1) |
| `validate_ncbi_16s_coupling` | Anderson coupling chain → cytokine propagation chain | Existing (v0.5.9) |
| `eco::diversity` | CPU Shannon/Simpson/Chao1/Bray-Curtis | Existing (v0.5.6) |

---

## Gonzales Data Ingestion Targets

Published data from Gonzales catalog that airSpring can directly process:

| Paper | Data | airSpring Pipeline |
|-------|------|-------------------|
| G3 (2016) | Pruritus scores at 1, 6, 11, 16 hr (oclacitinib vs steroids) | CytokineBrain time series → DriftMonitor regime transition |
| G4 (2021) | Lokivetmab duration curves (0.125/0.5/2.0 mg/kg × 14/28/42 days) | CytokineBrain pharmacokinetic decay as signal extinction |
| G2 (2014) | JAK1 IC50 curves (10-249 nM range) | Dose-response modeling (IC50 as effective W reduction) |
| G6 (2014) | IL-31 target cell distribution (immune/skin/neural) | GpuDiversity: 3-compartment cell-type heterogeneity |

---

## Connection to airSpring Core Mission

Paper 12 extends airSpring's domain from precision agriculture to precision
medicine, using identical computational infrastructure:

| Agricultural airSpring | Immunological Extension |
|----------------------|------------------------|
| Soil moisture θ(h) | Barrier permeability |
| ET₀ (evapotranspiration) | Cytokine flux |
| Pielou evenness (soil microbiome) | Pielou evenness (tissue cell types) |
| DriftMonitor (drought onset) | DriftMonitor (AD flare onset) |
| Crop stress factor | Tissue inflammation score |
| Weather → AirSpringBrain | Cytokine panel → CytokineBrain |
| 80-year climate atlas | Treatment response time series |

The Anderson framework is domain-agnostic — airSpring's precision and
diversity infrastructure applies wherever signals propagate through
disordered biological media.
