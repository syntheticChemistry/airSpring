# airSpring v0.10.0 — Paper Notebooks + Ecosystem Wiring Handoff

**Date**: May 7, 2026
**From**: airSpring
**For**: primalSpring, all spring teams, projectNUCLEUS, foundation
**Version**: v0.10.0 (deep debt resolved, paper notebooks done)
**License**: AGPL-3.0-or-later

## Summary

airSpring is the **first spring with publishable paper notebooks** — 20 core
papers converted to Jupyter notebooks with LaTeX equations, frozen benchmark
data, matplotlib visualizations, and validation against Rust binaries. This
handoff documents the paper notebook pattern for other springs and the
ecosystem wiring that connects airSpring to foundation thread06 and
projectNUCLEUS.

## What We Shipped

### 1. Paper Baseline Notebooks (20/20 Batch 1)

All in `notebooks/papers/`, following `PAPER_NOTEBOOK_PATTERN.md`:

| # | Notebook | Domain | Citation |
|---|----------|--------|----------|
| 001 | FAO-56 Penman-Monteith ET₀ | ET₀ | Allen et al. 1998 |
| 002 | Soil Sensor Calibration | Sensors | Dong et al. 2020 |
| 004 | FAO-56 Water Balance | Irrigation | Allen et al. 1998 Ch 8 |
| 006 | Richards Equation (VG-Mualem) | Soil Physics | Richards 1931, van Genuchten 1980 |
| 007 | Biochar P Adsorption | Soil Chemistry | Kumari et al. 2025 |
| 008 | Yield Response (Stewart) | Crop Science | Stewart et al. 1977 |
| 009 | Dual Crop Coefficient | Irrigation | Allen et al. 1998 Ch 7 |
| 017 | ET₀ Sensitivity Analysis | Analysis | Gong et al. 2006 |
| 018 | Michigan Crop Water Atlas | Integration | Open-Meteo ERA5 |
| 019 | Priestley-Taylor ET₀ | ET₀ | Priestley & Taylor 1972 |
| 021 | Thornthwaite ET₀ | ET₀ | Thornthwaite 1948 |
| 023 | Saxton-Rawls PTFs | Soil Physics | Saxton & Rawls 2006 |
| 031 | Hargreaves-Samani ET₀ | ET₀ | Hargreaves & Samani 1985 |
| 033 | Makkink ET₀ | ET₀ | Makkink 1957 |
| 034 | Turc ET₀ | ET₀ | Turc 1961 |
| 035 | Hamon PET | ET₀ | Hamon 1961 |
| 049 | Blaney-Criddle PET | ET₀ | Blaney & Criddle 1950 |
| 050 | SCS Curve Number | Hydrology | USDA 1972 |
| 051 | Green-Ampt Infiltration | Hydrology | Green & Ampt 1911 |
| 081 | SPI Drought Index | Climatology | McKee et al. 1993 |

Each notebook follows a 7-cell pattern:
1. **Title + Citation** (markdown) — paper reference, DOI, context
2. **Theory** (markdown) — LaTeX equations directly from the paper
3. **Setup** (code) — imports, frozen benchmark JSON loading
4. **Implementation** (code) — pure Python reproducing the paper's math
5. **Validation** (code) — comparison against benchmark values with tolerances
6. **Visualization** (code) — matplotlib plots (color palette: #2ecc71/#e74c3c/#3498db)
7. **Provenance + Summary** (markdown) — benchmark commit, tolerance table, primals.eco links

### 2. sporePrint Notebooks (5)

Previously shipped (May 7) — composition validation, benchmarks, ecosystem
evidence, cross-spring connections, domain deep dive. All use frozen JSON
from `experiments/results/`.

### 3. Foundation Thread 06 Wiring

Created `gardens/foundation/data/targets/thread06_ag_targets.toml`:
- **36 validation targets** across 16 papers
- Scalar expected values with tolerances, provenance commits, source citations
- All `spring = "airSpring"`, `validated = false` (awaiting guideStone L3+ execution)

Created `gardens/foundation/workloads/thread06_ag/` with 6 toadStool-dispatchable workloads:
- `airspring-et0-fao56.toml` — 75/75 PM cross-validated
- `airspring-et0-methods.toml` — 8 ET₀ methods
- `airspring-water-balance.toml` — Ch 8 + dual Kc + yield
- `airspring-soil-physics.toml` — Richards + GA + SCS-CN + PTF
- `airspring-atlas-pipeline.toml` — 100 stations, 80 years
- `airspring-full-suite.toml` — all 87 experiments

Updated `gardens/foundation/expressions/MEASUREMENT_SCIENCE.md` with
airSpring agricultural science section (20 papers, primal composition,
sediment contribution, Penny Irrigation vision).

### 4. projectNUCLEUS Workload Expansion

Fixed `gardens/sporeGarden/workloads/airspring/airspring-et0-validation.toml`:
- Migrated from hardcoded `/home/irongate/` paths to `${AIRSPRING_ROOT}`
- Added `isolation_level = "process"`

Created 5 new workload TOMLs matching foundation workloads:
- `airspring-et0-methods.toml`
- `airspring-water-balance.toml`
- `airspring-soil-physics.toml`
- `airspring-atlas-pipeline.toml`
- `airspring-full-suite.toml`

### 5. guideStone Level 0 → 1

Created `barracuda/src/bin/airspring_guidestone.rs`:
- Standalone binary — reads `downstream_manifest.toml` directly (TOML parsing)
- No `primalspring` crate dependency (path deps deprecated)
- Cross-checks manifest `validation_capabilities` against `niche::CAPABILITIES`
- Reports alignment, drift, coverage percentage
- Tier 1 local property checks only (exit 2 = skip when primals absent)
- Unblocks AG-001/AG-002

### 6. PRIMAL_GAPS Updated (14 gaps)

| Gap | Status | What Changed |
|-----|--------|-------------|
| AG-001 | In progress | primalSpring local, guidestone binary reads manifest |
| AG-002 | Resolved | Path deps deprecated; standalone TOML approach |
| AG-012 | New | Live Science API not yet available from toadStool |
| AG-013 | Fixed | Workload paths migrated to `${AIRSPRING_ROOT}` |
| AG-014 | Fixed | Foundation thread06 targets + workloads created |

## What Other Springs Need to Know

### Paper Notebook Pattern (Replicable)

The pattern in `notebooks/papers/PAPER_NOTEBOOK_PATTERN.md` is designed for
any spring to follow. Key conventions:

- Load frozen benchmark JSON via `pathlib.Path` — no live compute needed
- LaTeX equations directly from papers (not approximations)
- matplotlib with consistent color palette
- Validation cells with pass/fail assertions using tolerances from Rust code
- Provenance summary linking to primals.eco

**For hotSpring**: Nuclear binding energies → notebook per AME2020 comparison
**For wetSpring**: 16S pipeline, PFAS identification → notebook per validation domain
**For groundSpring**: Measurement science → notebook per experiment group
**For neuralSpring**: Inference validation → notebook per model benchmark
**For ludoSpring**: Game theory → notebook per algorithm proof

### Foundation Integration Pattern

To wire your spring into foundation:
1. Create `data/targets/threadNN_xxx_targets.toml` following `thread06_ag_targets.toml`
2. Create `workloads/threadNN_xxx/` with toadStool-dispatchable TOMLs
3. Update your thread's expression document in `expressions/`
4. Use `${YOUR_SPRING_ROOT}` for portable paths (not absolute)

### guideStone Standalone Pattern

Path dependencies to primalSpring are deprecated. To build your guideStone:
1. Parse `downstream_manifest.toml` directly with the `toml` crate
2. Cross-check against your `niche.rs` CAPABILITIES
3. Tier 1: local property checks (no IPC needed)
4. Exit codes: 0 = pass, 1 = fail, 2 = skip (primals absent)

## Live Science API Alignment

From `projectNUCLEUS/specs/LIVE_SCIENCE_API.md`:

| Method | Status | Notes |
|--------|--------|-------|
| `toadstool.validate` | Not yet available | Needed for Tier 2 notebook elevation |
| `toadstool.list_workloads` | Not yet available | Would allow notebooks to discover available validations |
| `toadstool.submit_workload` | Not yet available | Would allow notebooks to trigger validation runs |

airSpring is at **Notebook Elevation Tier 0/1** (frozen data + visualization).
Tier 2 (JSON-RPC validation APIs) blocked on Live Science API implementation.

## Notebook Elevation Path

From `projectNUCLEUS/specs/NOTEBOOK_ELEVATION.md`:

| Tier | Description | airSpring Status |
|------|-------------|-----------------|
| 0 | CLI validation binaries | Done (91 binaries) |
| 1 | Notebook visualization (frozen data) | Done (25 notebooks) |
| 2 | JSON-RPC validation APIs | Blocked (AG-012) |
| 3 | Live compute dashboard | Roadmap |

## airSpring Current State

| Metric | Value |
|--------|-------|
| Rust tests | 1,364 (986 lib + 316 integration + 62 forge) |
| Python baseline checks | 1,284 |
| Experiments | 87 |
| Validation binaries | 91 |
| Notebooks | 25 (20 paper + 5 sporePrint) |
| Tolerances | 60 (5 submodules, Rust + Python mirror) |
| IPC capabilities | 44/44 routable |
| Coverage | 90.56% |
| guideStone Level | 1 (scaffold) |
| Foundation targets | 36 |
| Workloads | 6 (foundation) + 6 (projectNUCLEUS) |

## For primalSpring

1. **Review** `docs/PRIMAL_GAPS.md` — 14 gaps, 4 resolved, 10 open
2. **Absorb** paper notebook pattern for ecosystem-wide adoption
3. **Wire** `airspring_guidestone` validation capabilities in deployment matrix
4. **Note** AG-012: toadStool Live Science API spec exists but isn't implemented

## For projectNUCLEUS

1. **6 new workloads** in `workloads/airspring/` ready for toadStool dispatch
2. **Hardcoded paths eliminated** — all use `${AIRSPRING_ROOT}`
3. **ABG workload**: `airspring-full-suite.toml` exercises all 87 experiments

## For foundation

1. **36 validation targets** in `data/targets/thread06_ag_targets.toml`
2. **6 workloads** in `workloads/thread06_ag/`
3. **Expression updated** with airSpring agricultural science section
4. **Thread 6 is the most notebook-complete thread** in the ecosystem

---

*airSpring v0.10.0 — first spring with publishable paper notebooks.
20 papers, 25 total notebooks, foundation thread06 wired, guideStone L1.
AGPL-3.0-or-later.*
