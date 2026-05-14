# Local NUCLEUS Deployment on Eastgate

**Date**: May 11, 2026 (updated from March 17)
**Status**: Steps 0-2 complete, Step 3 next — prerequisite for all baseCamp extensions
**Hardware**: Eastgate (i9-12900K, RTX 4070, AKD1000, 64 GB DDR5-5101, 2 TB NVMe)

---

## Why Local NUCLEUS First

Every baseCamp extension — NCBI 16S coupling (Paper 06), Penny Irrigation
(Paper 08), CytokineBrain evolution (Paper 12), soil O2 zonation (Paper 16)
— depends on cross-primal coordination. Local NUCLEUS on Eastgate provides
the minimum viable orchestration layer without requiring LAN HPC.

**Current state** (Steps 0-2 complete):

| Step | Status | What |
|------|--------|------|
| 0 | DONE | metalForge cross-system routing (GPU+NPU+CPU, 27 workloads) |
| 1 | DONE | airSpring NUCLEUS primal (49 caps, 28/28 cross-primal pipeline) |
| 2 | DONE | Local NUCLEUS Tower on Eastgate (7 primals discovered) |
| 3 | NEXT | NestGate weather provider (replace direct HTTP) |
| 4 | NEXT | NestGate NCBI 16S (baseCamp 06 extension) |
| 5 | LATER | ToadStool compute offload through NUCLEUS mesh |
| 6 | LATER | Full NUCLEUS on Eastgate |
| 7 | LATER | LAN HPC (Plasmodium across gates via 10G backbone) |

---

## Step 3: NestGate Weather Provider

Replace `SongbirdHttpProvider` (direct HTTP to Open-Meteo) with
`NestGateProvider` (3-tier routing with content-addressed caching).

**What NestGate gives us**:
- Content-addressed caching via blob store (BLAKE3 hash)
- Deduplication across experiments (same station/date → same blob)
- Offline operation (cached data serves without network)
- Unified `capability.call("data.fetch_daily_weather", ...)` routing

**NestGate already supports**: Open-Meteo, NOAA CDO, USDA NASS, NCBI,
Ensembl, HuggingFace — all via `NCBILiveProvider`, `OpenMeteoLiveProvider`, etc.

**airSpring already has**: `NestGateProvider` with 3-tier routing in
`barracuda/src/data/`. The provider is implemented but not yet wired to
a live NestGate instance.

**Deployment**:
```
biomeos nucleus start --mode nest --node-id eastgate
# This starts: BearDog + Songbird + NestGate
# Then start airSpring as a primal in the same NUCLEUS
```

**Validation**: Run `validate_nucleus` (29/29) and `validate_nucleus_pipeline`
(28/28) with NestGate routing enabled.

---

## Step 4: NestGate NCBI 16S Pipeline

**Prerequisite for**: Paper 06 (no-till Anderson), Paper 03 (precision
microbiome), Paper 04 (sentinels), Paper 16 (anaerobic-aerobic QS).

**Pipeline**:
```
NestGate ESearch("16S soil no-till", db=sra)
  → NestGate EFetch(accessions, format=fastq)
    → [wetSpring 16S pipeline: DADA2 → OTU table]
      → airSpring: Shannon H' + Bray-Curtis
        → airSpring: θ(t) → S_e → d_eff → QS regime
          → groundSpring: uncertainty (jackknife/bootstrap)
            → NestGate: store results (content-addressed)
```

**Data volumes**: NCBI SRA has ~105K no-till 16S entries. Target studies:
- Zuber et al. 2016 (Ohio tillage factorial)
- Liang et al. 2015 (long-term tillage, China)
- Wang et al. 2025 (pore-scale communities)
- OSU Triplett-Van Doren (60yr tilled vs no-till)

FASTQ size: ~25-55 GB for 4-10 studies. Fits on Eastgate NVMe (2 TB).

---

## Step 5: ToadStool Compute Offload

Route GPU workloads through `compute.offload`:
```
capability.call("compute.offload", {
  "workload": "batched_et0",
  "params": { "stations": 100, "years": 80 },
  "priority": "normal"
})
```

ToadStool routes to best available GPU (RTX 4070 on Eastgate, or
Titan V on biomeGate when LAN HPC is active).

---

## Step 6: Full NUCLEUS on Eastgate

Deploy all atomics:
```
biomeos nucleus start --mode full --node-id eastgate
```

This gives: Tower (BearDog + Songbird + SkunkBat) + Node (ToadStool) + Nest
(NestGate) + Squirrel (AI) + Provenance Trio (rhizoCrypt + loamSpine
+ sweetGrass) + airSpring (ecology).

Use `airspring_niche_deploy.toml` deployment graph for the full stack.

---

## Step 7: LAN HPC (Plasmodium)

After local NUCLEUS is stable, extend to LAN:

| Gate | NUCLEUS Role | Workload | GPU |
|------|-------------|----------|-----|
| eastGate | Node + NPU | airSpring ecology, NPU inference | RTX 4070 + AKD1000 |
| strandGate | Heavy Node | 16S pipeline, bioinformatics | RTX 3090 (dual EPYC 256 GB) |
| biomeGate | HBM2 Node | Anderson eigenvalue, heavy GPU | 2x Titan V + 2x MI50 |
| westGate | Heavy Nest | 76 TB ZFS cold storage | RTX 2070 Super |
| northGate | AI Node | Squirrel inference, LLM | RTX 5090 (192 GB DDR5) |

Connected via 10G backbone (switch acquired, NICs installed, cables pending).
Plasmodium forms when 2+ gates share `.family.seed`.

---

## Compute Budget for Extensions

| Workload | Scale | GPU Time | Gate |
|----------|-------|----------|------|
| ET0 100 stations, 80yr | 2.9M | 0.01s | Eastgate |
| Water balance 100 stations, 80yr | 2.9M | 0.1s | Eastgate |
| Richards 1D, 1000 grids, 80yr | 29M | 6 min | Eastgate |
| Kriging 100 stations/timestep | O(100^3)x29K | 20 min | Eastgate |
| Full MI grid 80yr (ET0+WB+yield) | ~1B | 5s | Eastgate |
| 16S diversity (10 studies) | ~500K seqs/study | hours | strandGate |
| Anderson eigenvalue (30K EMP) | 30K matrices | ~1 hr | biomeGate |
| LSTM r(t) training | ~1M samples | ~10 min | any |

Compute is not the bottleneck. Data download and cross-spring pipeline
integration are the limiting factors.
