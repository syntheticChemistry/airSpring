# airSpring Specifications

**Last Updated**: February 25, 2026
**Status**: Phase 0-3 complete — 400/400 Python + 601 Rust tests + 75/75 cross-validation + 8 GPU orchestrators + 69x CPU speedup
**Domain**: Precision agriculture, ET₀, soil moisture, irrigation scheduling

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 0 (Python) | 400/400 PASS — 13 experiments (FAO-56, soil, IoT, WB, dual Kc, cover crops, regional ET₀, Richards, biochar, 60yr WB, yield, CW2D) |
| Phase 0+ (Real data) | 918 station-days, R²=0.967 across 6 Michigan stations |
| Phase 1 (Rust) | 601 tests — 18 binaries, 433 lib + 115 integration + 53 forge |
| Phase 1.5 (CPU benchmark) | Rust 69x faster than Python (geometric mean, 20x–502x) |
| Phase 2 (Cross-validation) | 75/75 Python↔Rust match within 1e-5 (includes Richards + isotherm) |
| Phase 3 (GPU) | 8 orchestrators wired, 20 evolution gaps (8A+11B+1C) |
| Faculty | Dong (BAE, MSU — new lab 2026) |
| Handoff | V009 in `wateringHole/handoffs/` |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Active | Papers to review/reproduce with controls audit |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements + compute pipeline |
| [CROSS_SPRING_EVOLUTION.md](CROSS_SPRING_EVOLUTION.md) | Active | Cross-spring shader provenance and evolution story |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | Detailed experiment logs and check counts |
| CHANGELOG.md | `../` | Evolution history (Keep a Changelog format) |
| experiments/README.md | `../experiments/` | Experiment index (13 completed) |
| whitePaper/baseCamp/README.md | `../whitePaper/baseCamp/` | Per-faculty research briefings |
| whitePaper/STUDY.md | `../whitePaper/` | Full study results |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Multi-phase validation protocol |
| wateringHole/README.md | `../wateringHole/` | ToadStool handoff hub |

---

## Baseline Commit Lineage

Python baselines were generated across two commits as experiments expanded:

| Commit | Phase | Benchmarks | Date |
|--------|-------|-----------|------|
| `94cc51d` | Phase 1 | FAO-56, Dong 2020, Dong 2024, water balance, dual Kc, cover crop Kc | 2026-02-16 — 2026-02-25 |
| `3afc229` | Phase 2 | Richards equation, biochar isotherms, 60-year water balance | 2026-02-25 |

Each benchmark JSON embeds its provenance (script, commit, command, date).
Re-run `scripts/run_all_baselines.sh` at the respective commits to verify.

---

## Scope

### airSpring IS:
- **ET₀ validation** — FAO-56 Penman-Monteith reproduction against paper examples
- **Soil sensor calibration** — Dong 2020 factory-to-field correction
- **IoT irrigation pipeline** — Dong 2024 smart scheduling demonstration
- **Real data pipeline** — Open-Meteo ERA5 + NOAA CDO + OpenWeatherMap
- **Rust evolution proof** — Python→Rust cross-validated to 1e-5
- **GPU integration proof** — 8 orchestrators wired to ToadStool/BarraCuda (incl. Richards PDE + isotherm NM)

### airSpring IS NOT:
- Machine learning / neural surrogates (neuralSpring provides those)
- Sensor noise characterization (groundSpring Exp 001/003)
- Microbial soil health (wetSpring)
- GPU computation engine (ToadStool/BarraCuda — we *consume* it, GPU-FIRST)

---

## Reading Order

**New to airSpring** (15 min):
1. This README (3 min)
2. `../whitePaper/README.md` — overview and key results (7 min)
3. PAPER_REVIEW_QUEUE.md — what's next (5 min)

**Deep dive** (1 hour):
`../whitePaper/STUDY.md` → `../CONTROL_EXPERIMENT_STATUS.md` → BARRACUDA_REQUIREMENTS.md

**Cross-spring evolution** (15 min):
CROSS_SPRING_EVOLUTION.md → `../wateringHole/handoffs/AIRSPRING_V009_*.md`

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All airSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using airSpring code, must publish source under the same license.
