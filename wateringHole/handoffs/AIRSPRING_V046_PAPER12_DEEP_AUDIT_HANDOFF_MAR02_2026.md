# airSpring V046 — Paper 12 + Deep Audit + ToadStool S79 Evolution Handoff

**Date**: March 2, 2026
**From**: airSpring v0.6.3
**To**: ToadStool / BarraCuda team, biomeOS
**Supersedes**: V045 (full dispatch + biome graph), V061-V063 (S79 sync, nautilus, debt)
**ToadStool PIN**: S79 (`f97fc2ae` — 844 WGSL shaders, ops 0-13, GPU uncertainty stack)
**License**: AGPL-3.0-or-later
**Covers**: v0.6.0 → v0.6.3

---

## Executive Summary

- **69 experiments**, 1237/1237 Python, 810 lib tests (95.58% llvm-cov), 79 binaries
- **Paper 12**: Immunological Anderson — tissue diversity, CytokineBrain, barrier
  state, cross-species (Exp 066-069) extending Anderson localization to medicine
- **Deep audit**: all 79 binaries have provenance, all 47 tolerances have provenance
  table, zero hardcoded `/tmp/`, dependency evolution documented (ureq→Songbird)
- **ToadStool S79**: 124/124 cross-spring benchmarks, ops 0-13 validated, GPU
  uncertainty stack (jackknife/bootstrap/diversity) integrated
- **Quality**: `clippy --pedantic` clean, `#![forbid(unsafe_code)]`, zero mocks,
  zero `todo!()`, all files < 1000 lines, AGPL-3.0-or-later everywhere

---

## Part 1: New Experiments (v0.6.1–v0.6.3)

| Exp | Name | Checks | What it proves |
|:---:|------|:------:|----------------|
| 064 | Full Dispatch Experiment | 51/51 | CPU/GPU parity across all domains |
| 065 | biomeOS Graph Experiment | 35/35 | Offline ecology pipeline, deployment graph |
| 066 | Tissue Diversity (Paper 12) | 30+30 | Shannon→Pielou→Anderson W for skin |
| 067 | CytokineBrain (Paper 12) | 14+28 | Nautilus reservoir for AD flare prediction |
| 068 | Barrier State Model (Paper 12) | 16+16 | VG θ(h)/K(h) analogy for skin barrier |
| 069 | Cross-Species Skin (Paper 12) | 19+20 | Canine/human/feline One Health bridge |

### Paper 12: Anderson Localization → Immunological Tissue

The Anderson localization framework validated in Exp 045 (soil θ → S_e → d_eff →
QS regime) was extended to immunological tissue analysis:

- **Effective disorder W**: Pielou evenness → Anderson W = (1−J) × ln(S) where
  S = species richness. W > W_c(d) triggers delocalization = inflammation.
- **CytokineBrain**: 3-head Nautilus evolutionary reservoir predicting AD flare
  probability from cytokine cocktail inputs (IL-4, IL-13, IL-31, IFN-γ, TNF-α).
  Brain trained via `NautilusShell::evolve()` + `DriftMonitor` for N_e·s boundary
  regime change detection.
- **Barrier integrity**: VG retention θ(h) applied to transepidermal water loss.
  Barrier disruption modeled as d_eff = 2 + breach_fraction.
- **Cross-species**: Anderson W comparison between canine atopic dermatitis, human
  psoriasis, and feline eosinophilic granuloma. Shannon diversity serves as
  sufficient universal measure across organisms.

**Relevance to ToadStool**: The Anderson W and CytokineBrain patterns are
domain-agnostic and could benefit other Springs (wetSpring for microbiome analysis,
groundSpring for lattice disorder).

---

## Part 2: Deep Audit Results

### Validation Provenance (was largest gap, now resolved)

| Category | Before | After |
|----------|--------|-------|
| Binaries with full provenance (script/commit/date) | 10/79 | **79/79** |
| Tolerances with provenance table | 0 | **47/47** (19 domain entries) |
| Benchmark JSONs with `_provenance` | ~40 | All |

### Quality Gates (v0.6.3)

| Gate | Status |
|------|--------|
| `cargo fmt --check` | PASS (both crates) |
| `cargo clippy --workspace -- -D warnings -W clippy::pedantic` | PASS — 0 warnings |
| `cargo doc --no-deps` | PASS (both crates) |
| `cargo test --lib` | **810 passed**, 0 failed |
| `cargo llvm-cov` | **95.58% line** / 96.33% function |
| `cargo deny check` | PASS |
| `#![forbid(unsafe_code)]` | Both crates |
| Files > 1000 lines | 0 (max: 935, bench binary) |
| Mocks in production | 0 |
| TODOs / FIXMEs / HACK | 0 |
| Hardcoded `/tmp/` in production | 0 (fixed) |
| `unwrap()` in library code | 0 (all in `#[cfg(test)]`) |
| SPDX AGPL-3.0-or-later headers | All `.rs` files |

### Dependency Evolution

| Dep | C code? | Evolution |
|-----|---------|-----------|
| barracuda (path) | wgpu (vulkan) | **Core** — stays |
| bingocube-nautilus (path) | None | **Core** — pure Rust |
| serde + serde_json | None | Pure Rust, stays |
| tracing-subscriber | None | Pure Rust, stays |
| **ureq** | **ring (C/asm via rustls)** | **Evolve**: already abstracted behind `discover_transport()`; Songbird sovereign TLS replaces ureq when Tower Atomic running |

---

## Part 3: BarraCuda Usage (25 Tier A + 3 Pipeline)

airSpring exercises the widest range of `batched_elementwise_f64` ops (0-13)
of any Spring, plus dedicated shaders, optimization, PDE, and statistics.

| Op | Domain | Status |
|:--:|--------|--------|
| 0 | FAO-56 PM ET₀ | GPU-FIRST |
| 1 | Water balance step | GPU-STEP |
| 5 | Sensor calibration | Integrated |
| 6 | Hargreaves ET₀ | Integrated (dedicated shader) |
| 7 | Kc climate adjustment | Integrated |
| 8 | Dual Kc evaporation | Integrated |
| 9 | VG θ(h) retention | Integrated (S79) |
| 10 | VG K(h) conductivity | Integrated (S79) |
| 11 | Thornthwaite heat index | Integrated (S79) |
| 12 | GDD accumulation | Integrated (S79) |
| 13 | Pedotransfer Saxton-Rawls | Integrated (S79) |
| — | Kriging (KrigingF64) | Integrated |
| — | Reduction (FusedMapReduceF64) | GPU N≥1024 |
| — | Stream smoothing (MovingWindowStats) | Integrated |
| — | Richards PDE (solve_richards) | Integrated |
| — | Isotherm (nelder_mead + multi_start) | Integrated |
| — | MC ET₀ uncertainty | Integrated |
| — | Jackknife (S79) | Integrated |
| — | Bootstrap (S79) | Integrated |
| — | Diversity fusion (S79) | Integrated |
| — | OLS regression | Integrated |
| — | Correlation matrix | Integrated |
| — | Seasonal pipeline (chained) | CPU + GpuPipelined |
| — | Atlas stream (80yr) | CPU chained |

---

## Part 4: Absorption Recommendations for ToadStool

### Tier 1 — Direct Absorption

| Module | Lines | What | Why absorb |
|--------|:-----:|------|-----------|
| Tissue diversity (Anderson W) | ~320 | Pielou→W effective disorder | Domain-agnostic disorder metric |
| CytokineBrain pattern | ~530 | Nautilus reservoir for regime prediction | Reusable brain pattern |
| Named tolerances (47) | ~630 | Physical/numerical constants | Shared ground truth |

### Tier 2 — Pattern Promotion

| Pattern | Description |
|---------|-------------|
| Provenance JSON | `_provenance` block in all benchmark data |
| Provenance in binaries | `script=`, `commit=`, `date=`, `Run:` in doc headers |
| Capability-based discovery | No hardcoded primal names; env vars + runtime scan |
| Benchmark JSON embedding | `include_str!()` at compile time, not runtime file access |

### Tier 3 — Evolution Opportunities (from airSpring learnings)

| Opportunity | Impact |
|-------------|--------|
| `ComputeDispatch` migration for ops 0-13 | Align with S80's dispatch migration |
| Batch Nelder-Mead GPU | Parallelize isotherm fitting (~36K→millions fits/sec) |
| `BatchedEncoder` fused pipelines | Eliminate CPU round-trips in seasonal pipeline |
| `BatchedStatefulF64` | Carry-forward state for water balance GPU step |

---

## Part 5: Cross-Spring Learnings

1. **f64 precision is non-negotiable for soil physics** — Van Genuchten K(h) spans
   12 orders of magnitude for clay soils. f32 fails catastrophically.

2. **Domain ops plateau at ~15** — airSpring needed 14 `batched_elementwise_f64` ops.
   Each is 5-15 lines of math. Suggests new Springs will need ~10-20 ops, not hundreds.

3. **GPU uncertainty stack transforms confidence quantification** — Jackknife/bootstrap
   at ~700µs GPU vs ~10ms CPU enables uncertainty bands on every atlas prediction.

4. **Nautilus integrates cleanly** — AirSpringBrain (3-head ET₀/soil/crop) and
   CytokineBrain (3-head cytokine regime) both used NautilusShell → DriftMonitor →
   EdgeSeeder without modification. Pattern is domain-agnostic.

5. **Anderson localization is universal** — soil disorder (d_eff from Bethe lattice),
   immunological disorder (W from Shannon/Pielou), and microbial disorder (wetSpring
   OTU diversity) all use the same W vs W_c(d) phase boundary. ToadStool could
   provide `anderson_regime_classify(W, d, W_c_table)` as a shared primitive.

---

## Part 6: Paper Queue Status

69 experiments, 54+ completed paper reproductions. All use open data and systems.
Compute pipeline per paper: Python control → BarraCuda CPU → BarraCuda GPU →
metalForge mixed hardware.

Remaining paper queue targets (from `specs/PAPER_REVIEW_QUEUE.md`):
- Agrivoltaics PAR (photovoltaic shading impact on crop ET₀)
- Extended lysimeter validation with real-time data
- Additional climate scenario analysis with CMIP6 projections
- Exp 018 Phase 2: 100 stations × 80yr, decade trends, kriging interpolation

---

## Handoff Chain

| Document | Scope |
|----------|-------|
| V045 | Full dispatch + biome graph (v0.6.0) — **superseded** |
| V061 | ToadStool S79 sync — **superseded** |
| V062 | Nautilus/AirSpringBrain — **superseded** |
| V063 | Deep debt audit — **superseded** |
| **V046** (this) | Paper 12 + deep audit + S79 evolution (v0.6.3) — **current** |
| `AIRSPRING_TOADSTOOL_ABSORPTION_HANDOFF_MAR02_2026.md` | Detailed absorption recommendations |

---

AGPL-3.0-or-later
