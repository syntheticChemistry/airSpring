# NCBI 16S + Soil Moisture Coupling

**Date:** February 28, 2026
**Status:** Complete — Exp 048 done (14+29 checks)
**Extension of:** baseCamp Sub-thesis 06 (No-Till Anderson)
**Cross-Spring:** airSpring (θ(t), ET₀, Anderson coupling) × wetSpring (16S pipeline, Anderson QS) × groundSpring (uncertainty, rare biosphere) × NestGate (NCBI data acquisition)

---

## Motivation

baseCamp 06 validates the Anderson localization framework for no-till soil
health using published data (Islam 2014, Brandt farm metrics, OSU 60-year
experiment). This experiment couples **real NCBI 16S rRNA metagenome data**
from agricultural soil studies to airSpring's moisture-driven Anderson QS
model, in a quantitative pipeline from weather → soil moisture →
pore geometry → microbial community state.

This is the first experiment that exercises the full cross-primal pipeline:

```
NestGate (NCBI EFetch)
    → wetSpring (16S pipeline: FASTQ → OTU table → Shannon H′)
    → airSpring (Anderson coupling: H′ → W, θ(t) → d_eff(t) → QS regime)
    → groundSpring (uncertainty budget: sensor → ξ → r)
```

---

## Data Sources

### NCBI SRA — Soil 16S Studies

Provider test (2026-02-28) confirmed:
- **104,967** SRA entries for soil 16S agricultural metagenomes
- **155,510** nucleotide entries for soil 16S rRNA sequences (1200-1600 bp)
- Taxonomy lookup operational (Bacillus subtilis TaxID 1423)

Target studies for download:

| Study | NCBI Accession | Soil Type | Tillage | Region | Est. Size |
|-------|---------------|-----------|---------|--------|-----------|
| Zuber et al. (2016) — no-till vs conv | PRJNA (search) | Silt loam | NT vs CT × 31yr | Illinois | ~2-5 GB |
| Liang et al. (2015) — conservation tillage | PRJNA (search) | Multiple | NT vs CT | China | ~2-5 GB |
| Wang et al. (2025) — tillage microbiome | PRJNA (search) | Agricultural | Multiple | Global | ~2-5 GB |
| OSU Triplett-Van Doren (if SRA exists) | PRJNA (search) | Hoytville/Wooster | NT vs CT × 60yr | Ohio | ~2-5 GB |

**Data budget**: ~20-50 GB total for 4-10 studies.

### Open-Meteo ERA5 — Site Weather Reconstruction

Already validated (115 CSVs, 80yr Michigan). For the NCBI coupling, reconstruct
weather at the study sites (Illinois, Ohio, etc.) using Open-Meteo's global
coverage.

### USDA Web Soil Survey — Soil Properties

Van Genuchten parameters (θ_r, θ_s, α, n, K_s) for the study sites.
Already integrated into airSpring via Saxton-Rawls pedotransfer (Exp 023).

---

## Pipeline Design

### Step 1: Data Acquisition (NestGate)

```python
# Search for soil 16S studies with tillage metadata
ncbi_search("sra", "soil 16S agricultural tillage")
# Fetch SRA metadata (study design, sample count, sequencing platform)
ncbi_summary(ids)
# Download FASTQ files (via SRA Toolkit or EFetch)
ncbi_fetch(ids, rettype="fastq")
```

### Step 2: 16S Processing (wetSpring)

wetSpring's sovereign Rust pipeline processes FASTQ → OTU table:

1. Quality filtering (`barracuda::bio::fastq`)
2. Chimera removal
3. OTU clustering (97% identity)
4. Taxonomy assignment (SILVA/RefSeq, via NestGate content-addressed store)
5. Diversity metrics: Shannon H′, Simpson D, Bray-Curtis β

### Step 3: Moisture Reconstruction (airSpring)

For each study site:

1. Fetch 80yr Open-Meteo ERA5 weather data
2. Get soil properties from USDA Web Soil Survey
3. Compute ET₀ (FAO-56 PM, 7-method ensemble)
4. Run water balance → θ(t) daily time series
5. Richards PDE for vertical soil moisture profile

### Step 4: Anderson Coupling (airSpring × wetSpring)

Use the validated `eco::anderson` module (Exp 045, 55+95 checks):

```
θ(t) → S_e(t) → pore_connectivity(t) → z(t) → d_eff(t) → QS_regime(t)
```

Overlay 16S diversity from Step 2:

```
Shannon H′ → W (disorder parameter calibration)
d_eff(t) from Step 4 + W from 16S → Anderson r(t) → QS regime prediction
```

### Step 5: Validation

Compare predicted QS regime (from physics) against observed microbial
community function (from 16S data):

- No-till plots → predicted QS-active (d_eff ≈ 3, W < W_c)
- Conventional till → predicted QS-suppressed (d_eff ≈ 2, W irrelevant)
- Cover crop → predicted W modulation within QS-active regime

---

## Compute Requirements

| Component | Hardware | Time Estimate |
|-----------|----------|---------------|
| NCBI download (4-10 studies) | Eastgate (I/O bound) | 1-4 hours |
| 16S pipeline per study | Eastgate CPU | ~30 min |
| ET₀ + water balance per site | Eastgate GPU | ~5 sec |
| Richards PDE per site × 365 days | Eastgate GPU | ~6 min |
| Anderson coupling (full grid) | Eastgate CPU | ~1 min |
| **Total per study** | | **~35 min compute + download** |

All workloads fit on Eastgate (i9-12900K, RTX 4070, 32GB DDR5, 2TB NVMe).
Strandgate (dual EPYC + RTX 3090) available for bioinformatics scale-up.

---

## Storage Requirements

| Data | Size | Location |
|------|------|----------|
| NCBI FASTQ files (4-10 studies) | 20-50 GB | Eastgate NVMe |
| SILVA/RefSeq reference DB | ~5 GB | NestGate (BLAKE3 provenance) |
| Open-Meteo weather reconstruction | ~600 MB | Existing (`data/open_meteo/`) |
| OTU tables + diversity results | ~100 MB | airSpring results |
| **Total** | **~25-55 GB** | Eastgate 2TB NVMe (ample) |

---

## Success Criteria

1. **Data pipeline operational**: NCBI → NestGate → 16S OTU table (at least 1 study)
2. **Diversity computed**: Shannon H′ for no-till vs conventional samples
3. **Moisture reconstructed**: θ(t) for the study site from Open-Meteo
4. **Anderson coupling exercised**: d_eff(t) + W → QS regime for real data
5. **Prediction validated**: No-till = QS-active, conventional till = QS-suppressed
6. **groundSpring uncertainty**: Error bars on the QS regime prediction

---

## Existing Validated Components

| Component | Experiment | Checks | Status |
|-----------|-----------|:------:|--------|
| FAO-56 PM ET₀ | Exp 001 | 64/64 | PASS |
| Water balance | Exp 004 | 18/18 | PASS |
| Richards PDE | Exp 009 | 14+15 | PASS |
| Saxton-Rawls pedotransfer | Exp 023 | 70+58 | PASS |
| Anderson coupling | Exp 045 | 55+95 | PASS |
| NCBI 16S + Anderson coupling | Exp 048 | 14+29 | PASS |
| GPU math portability | Exp 047 | 46/46 | PASS |
| NestGate NCBI provider | Provider test | 23/23 | PASS |
| Open-Meteo download | 115 CSVs | Atlas 1498 | PASS |

---

## Dependencies

- **wetSpring 16S pipeline**: FASTQ → OTU. Currently sovereign Rust (Exp184-185).
- **NestGate SRA download**: `ncbi_live_provider.rs` is architecturally ready but
  HTTP client is stubbed. Use Python `scripts/ncbi_16s_fetch_sample.py` + SRA toolkit as interim.
- **SILVA reference database**: Downloaded once via NestGate, content-addressed.
- **No institutional access required**: All NCBI data is open via E-utilities.
