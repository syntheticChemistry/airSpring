# Cross-Spring Soil-Microbiome Pipeline

**Date**: March 17, 2026
**Status**: Architecture defined — prerequisite NestGate NCBI wiring pending
**Papers**: 06 (no-till Anderson), 03 (precision microbiome), 04 (sentinels),
16 (anaerobic-aerobic QS)
**Springs**: airSpring (soil physics) x wetSpring (16S/Anderson) x groundSpring
(uncertainty) x neuralSpring (ML) x NestGate (data)

---

## The Pipeline

Four baseCamp papers share a common cross-spring data flow:

```
         NestGate                    wetSpring
    ┌──────────────┐           ┌──────────────────┐
    │ NCBI ESearch  │           │ 16S DADA2        │
    │ NCBI EFetch   │──FASTQ──→│ OTU table        │
    │ (SRA/Protein) │           │ Shannon H'       │
    └──────────────┘           │ Bray-Curtis      │
                                └────────┬─────────┘
                                         │ diversity
                                         ▼
                                    airSpring
                               ┌──────────────────┐
                               │ θ(t) FAO-56 WB   │
                               │ S_e (saturation)  │
                               │ d_eff (pore conn) │
                               │ Anderson coupling │
                               │ QS regime class   │
                               └────────┬──────────┘
                                        │ W, r(t)
                               ┌────────┼────────┐
                               ▼        ▼        ▼
                          groundSpring  neuralSpring  NestGate
                          ┌──────────┐ ┌──────────┐ ┌─────────┐
                          │ jackknife │ │ LSTM r(t)│ │ store   │
                          │ bootstrap │ │ ESN QS   │ │ results │
                          │ spectral  │ │ transfer │ │ (BLAKE3)│
                          └──────────┘ └──────────┘ └─────────┘
```

---

## What Each Spring Contributes

### airSpring (this spring)

| Module | Contribution | Status |
|--------|-------------|--------|
| `eco/evapotranspiration/` | FAO-56 PM ET0 (5 sub-modules, v0.8.9) | Validated |
| `eco/water_balance.rs` | Field-scale daily water budget | Validated |
| `eco/richards.rs` | 1D Richards vadose zone flow | Validated |
| `eco/soil_moisture.rs` | Topp/Saxton-Rawls pedotransfer | Validated |
| `eco/tissue.rs` | Anderson coupling: theta → S_e → d_eff → QS | Validated (Exp 045) |
| `gpu/kriging.rs` | Spatial interpolation for field-scale mapping | Validated |
| `gpu/diversity.rs` | Shannon H', Bray-Curtis on GPU | Validated |
| `data/` | NestGateProvider 3-tier routing | Implemented |

### wetSpring (16S pipeline + Anderson QS)

| Capability | Contribution |
|-----------|-------------|
| `bio::dada2` | FASTQ → OTU table (sovereign Rust pipeline) |
| `bio::taxonomy` | OTU → species classification |
| `bio::anderson` | Anderson eigenvalue, level spacing ratio r |
| `bio::diversity` | Shannon, Simpson, Pielou, Bray-Curtis |
| QS gene HMM | Hidden Markov Model for QS gene detection |

### groundSpring (uncertainty)

| Capability | Contribution |
|-----------|-------------|
| `uncertainty::jackknife` | LOO variance for diversity statistics |
| `uncertainty::bootstrap` | Bootstrap CI for Anderson W |
| `uncertainty::spectral` | Spectral theory for eigenvalue validation |
| `uncertainty::sensitivity` | OAT sensitivity for theta → W propagation |

### neuralSpring (ML)

| Capability | Contribution |
|-----------|-------------|
| `ml::lstm` | LSTM time series for r(t) prediction from soil parameters |
| `ml::esn` | ESN reservoir for QS regime classification |
| `ml::transfer` | Transfer learning: Michigan → other climates |

---

## Per-Paper Extensions

### Paper 06: No-Till Anderson

**Hypothesis**: Tillage = dimensional collapse of QS-active 3D pore network.
No-till preserves pore connectivity → higher d_eff → Anderson-GOE regime → QS active.

**Extension data**:

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| No-till 16S | NCBI SRA (~105K entries) | 25-55 GB FASTQ | Free (API key) |
| OSU Triplett-Van Doren | soilfertility.osu.edu | Published | Free |
| Zuber 2016 | NCBI BioProject | ~5 GB FASTQ | Free |
| Liang 2015 | NCBI BioProject | ~5 GB FASTQ | Free |
| Open-Meteo (Ohio) | Open-Meteo ERA5 | ~100 MB | Free |
| USDA Web Soil Survey | USDA WSS | <1 MB | Free |

**New experiments**:
- Exp 088: Real LTER 16S → OTU → Shannon → theta → Anderson coupling
- Exp 089: Brandt farm r(t) prediction via LSTM
- Exp 090: Kriging spatial interpolation of W at field scale
- Exp 091: EMP Atlas 30K Anderson-QS profiling

### Paper 16: Anaerobic-Aerobic QS Phase Transition

**Hypothesis**: Oxygen boundary triggers QS gene expression reprogramming
(FNR/ArcAB/Rex) → Anderson disorder parameter W undergoes phase transition.

**Extension data**:

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| Digester 16S | NCBI BioProject (ADREC) | ~5 GB | Free |
| FNR/ArcAB/Rex QS genes | NCBI Protein | <1 GB | Free |
| Wang/Liao 2020 time series | Published | <100 MB | Free |

**New experiments**:
- Exp 092: Soil aerobic/anaerobic zonation from Richards PDE theta(t)
- Exp 093: Pore connectivity → O2 gradient → W phase transition
- Exp 094: QS gene profiling for FNR/ArcAB/Rex regulon

### Paper 03: Precision Microbiome for Tree Crops

**Hypothesis**: Soil pore geometry guides inoculant design;
rhizosphere W approximates 6.7 (extended QS regime).

**Extension data**:

| Dataset | Source | Size | Access |
|---------|--------|------|--------|
| Bulgarelli 2012 | Published | <1 MB | Free |
| KBS LTAR | Michigan State | <100 MB | Free |
| FAO-56 Kc (tree crops) | Published tables | <5 KB | Free |

**New experiments**:
- Exp 095: Orchard theta(t) for rhizosphere d_eff
- Exp 096: Irrigation scheduling for inoculant establishment
- Exp 097: Monod kinetics for rhizosphere microbial dynamics

### Paper 12: Immunological Anderson (One Health Bridge)

**Extension**: Connect soil microbiome diversity (airSpring) to gut
microbiome diversity (healthSpring) via Anderson disorder W.

**New experiments**:
- Exp 098: One Health bridge — soil W vs gut W correlation
- Exp 099: Barrier state (VG analogue) for skin compartments

---

## Data Budget Summary

```
Paper 06 (no-till):          ~30-60 GB (NCBI FASTQ + weather)
Paper 16 (anaerobic):         ~5-10 GB (NCBI digester + protein)
Paper 03 (microbiome):         ~1 GB (published + KBS)
Paper 12 (immunological):     ~1 GB (Gonzales + NCBI protein)
────────────────────────────────────
Total:                       ~37-72 GB
Storage:  Eastgate NVMe (2 TB) — fits easily
Cold:     westGate ZFS (76 TB) — for Tier 3 satellite data
```

---

## Deployment Graph

The existing `cross_primal_soil_microbiome.toml` graph coordinates:
```
airSpring theta(t) → wetSpring diversity → spectral analysis → NestGate store
```

Extend with:
- NestGate NCBI fetch node (data acquisition)
- groundSpring uncertainty node (jackknife/bootstrap)
- neuralSpring prediction node (LSTM r(t))
- Provenance trio (rhizoCrypt session + sweetGrass attribution + loamSpine record)

---

## Prerequisites

1. Local NUCLEUS on Eastgate (see `nucleus_local_deployment.md`)
2. NestGate NCBI provider wired and validated
3. wetSpring 16S pipeline accessible via capability discovery
4. Cross-spring time series exchange format (`ecoPrimals/time-series/v1`)
