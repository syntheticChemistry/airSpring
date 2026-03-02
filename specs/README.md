# airSpring Specifications

**Last Updated**: March 1, 2026
**Status**: Phase 0-4 complete — 1237/1237 Python + 641 lib + 57 forge tests + 73/73 atlas stream + 75/75 cross-validation + 11 Tier A + 4 Tier B GPU orchestrators + seasonal pipeline + AKD1000 NPU live + 14.5× CPU speedup (21/21 parity) + metalForge 18 workloads 29/29 + coupled runoff-infiltration (292/292) + VG inverse (84/84) + full-season WB (34/34) + NUCLEUS primal (16 capabilities, 28/28 cross-primal) + ecology domain + 63 experiments + 72 binaries (v0.6.0)
**Domain**: Precision agriculture, ET₀, soil moisture, irrigation scheduling, Anderson coupling

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 0 (Python) | 1237/1237 PASS — 54 experiments (FAO-56, soil, IoT, WB, dual Kc, cover crops, regional ET₀, Richards, biochar, 60yr WB, yield, CW2D, scheduling, lysimeter, sensitivity, Priestley-Taylor, 3-method intercomparison, Thornthwaite, GDD, pedotransfer, AmeriFlux, Hargreaves, diversity, multi-crop, NPU eco, forecast, SCAN moisture, NASS yield, Anderson coupling, Blaney-Criddle, SCS-CN, Green-Ampt, coupled runoff-infiltration, VG inverse, full-season WB) |
| Phase 0+ (Real data) | 15,300 station-days, R²=0.967 across 100 Michigan stations |
| Phase 1 (Rust) | 641 lib + 57 forge tests — 72 binaries (67 barracuda + 5 forge) |
| Phase 1.5 (CPU benchmark) | Rust 14.5× faster than Python (geometric mean, 21/21 parity) |
| Phase 2 (Cross-validation) | 75/75 Python↔Rust match within 1e-5; 690 crop-station yield pairs within 0.01 |
| Phase 2.5 (Tier B GPU) | 4 Tier B orchestrators wired (ops 5-8), seasonal pipeline, atlas stream |
| Phase 3 (GPU) | 11 Tier A + 4 Tier B modules wired, cross-spring S68 fully rewired |
| Phase 3.8 (Cross-system) | metalForge 18 workloads, 29/29 cross-system routing (GPU+NPU+CPU) |
| Faculty | Dong (BAE, MSU — new lab 2026) |
| Phase 3.9 (NUCLEUS primal) | 29/29 biomeOS primal parity, 9 science capabilities |
| Phase 4.0 (Cross-primal) | 28/28 PASS — ecology domain, capability.call, cross-primal forwarding |
| Handoff | V041 (NUCLEUS cross-primal evolution) in `wateringHole/handoffs/` |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Active | Papers to review/reproduce with controls audit |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements + compute pipeline |
| [CROSS_SPRING_EVOLUTION.md](CROSS_SPRING_EVOLUTION.md) | Active | Cross-spring shader provenance and evolution story |
| [ATLAS_STATION_LIST.md](ATLAS_STATION_LIST.md) | Planning | Michigan 100-station expansion for crop water atlas |
| [NUCLEUS_INTEGRATION.md](NUCLEUS_INTEGRATION.md) | **Complete** | NUCLEUS deployment — primal registered, ecology domain, 28/28 pipeline |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | Detailed experiment logs and check counts |
| CHANGELOG.md | `../` | Evolution history (Keep a Changelog format) |
| experiments/README.md | `../experiments/` | Experiment index (54 completed) |
| whitePaper/baseCamp/README.md | `../whitePaper/baseCamp/` | Per-faculty research briefings |
| whitePaper/STUDY.md | `../whitePaper/` | Full study results |
| whitePaper/METHODOLOGY.md | `../whitePaper/` | Multi-phase validation protocol |
| wateringHole/README.md | `../wateringHole/` | ToadStool handoff hub |

---

## Baseline Commit Lineage

Python baselines were generated across five commits as experiments expanded:

| Commit | Phase | Benchmarks | Date |
|--------|-------|-----------|------|
| `94cc51d` | Phase 1 | FAO-56, Dong 2020, Dong 2024, water balance, dual Kc, cover crop Kc | 2026-02-16 — 2026-02-25 |
| `3afc229` | Phase 2 | Richards equation, biochar isotherms, 60-year water balance | 2026-02-25 |
| `5684b1e` | Phase 2+ | Scheduling, sensitivity, lysimeter | 2026-02-26 |
| `af1eb97` | Phase 2+ | Yield response, CW2D Richards | 2026-02-26 |
| `cb59873` | Phase 0+ | Atlas (100 stations), compute_et0_real_data, simulate_real_data, regional_et0_intercomparison | 2026-02-26 |
| `9a84ae5` | Phase 2+ | Priestley-Taylor ET₀, 3-method intercomparison | 2026-02-26 |

Each benchmark JSON embeds its provenance (script, commit, command, date).
Re-run `run_all_baselines.sh` at the respective commits to verify.

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
CROSS_SPRING_EVOLUTION.md → `../wateringHole/handoffs/` (V038 active)

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All airSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using airSpring code, must publish source under the same license.
