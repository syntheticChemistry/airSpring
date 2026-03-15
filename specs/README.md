# airSpring Specifications

**Last Updated**: March 15, 2026
**Status**: Phase 0–5 complete — 1284/1284 Python + 851 lib + 280 integration + 61 forge tests + 381/381 validation + 146/146 evolution + 33/33 cross-validation + 94 binaries + ops 0-19 upstream (`BatchedElementwiseF64`) + `PrecisionRoutingAdvice` + upstream provenance registry + barraCuda 0.3.5 (wgpu 28) + 14.3× CPU speedup (24/24 algorithms, 21/21 CPU-GPU parity modules) + metalForge 66/66 + niche adapter (41 capabilities) + 87 experiments (v0.8.2). Edition 2024, deep code quality complete. Exp 084-087: 79/79 PASS (CPU/GPU parity, toadStool dispatch, metalForge NUCLEUS, graph coordination)
**Domain**: Precision agriculture, ET₀, soil moisture, irrigation scheduling, Anderson coupling

---

## Quick Status

| Metric | Value |
|--------|-------|
| Phase 0 (Python) | 1284/1284 PASS — 57 papers reproduced (FAO-56, soil, IoT, WB, dual Kc, cover crops, regional ET₀, Richards, biochar, 60yr WB, yield, CW2D, scheduling, lysimeter, sensitivity, Priestley-Taylor, 3-method intercomparison, Thornthwaite, GDD, pedotransfer, AmeriFlux, Hargreaves, diversity, multi-crop, NPU eco, forecast, SCAN moisture, NASS yield, Anderson coupling, Blaney-Criddle, SCS-CN, Green-Ampt, coupled runoff-infiltration, VG inverse, full-season WB) |
| Phase 0+ (Real data) | 15,300 station-days, R²=0.967 across 100 Michigan stations |
| Phase 1 (Rust) | 851 lib + 280 integration + 61 forge tests — 94 binaries (88 barracuda + 6 forge) |
| Phase 1.5 (CPU benchmark) | 14.3× geometric mean speedup (24/24 algorithms, 21/21 CPU-GPU parity), 13,000× atlas-scale |
| Phase 2 (Cross-validation) | 75/75 Python↔Rust match within 1e-5; 690 crop-station yield pairs within 0.01 |
| Phase 2.5 (Tier B→A GPU) | 4 Tier B→A promotions (ops 5-8), `BatchedStatefulF64`, `BatchedNelderMeadGpu` |
| Phase 2.6 (Seasonal pipeline) | GPU Stages 1-3 (ET₀ + Kc + WB), multi-field `gpu_step()`, streaming |
| Phase 2.7 (GPU streaming) | 57/57 PASS — M fields × N days, 6.8M field-days/s (Exp 070) |
| Phase 3 (GPU dispatch) | 25 Tier A + 6 GPU-local modules wired, `BrentGpu` VG inverse, `RichardsGpu` Picard |
| Phase 3.1 (Pure GPU) | 46/46 PASS — all 4 stages on GPU, 19.7× dispatch reduction (Exp 072) |
| Phase 3.2 (Cross-spring rewire) | 68/68 PASS — 5/5 springs validated (Exp 073) |
| Phase 3.5 (NPU edge) | AKD1000 live, 95/95 NPU checks |
| Phase 3.8 (Cross-system) | metalForge 27 workloads, 66/66 cross-system (GPU→NPU→CPU) + Exp 076 NUCLEUS routing (60/60) |
| Phase 3.9 (Niche adapter) | 41 capabilities, ecology domain in biomeOS registry |
| Phase 4.0 (Cross-primal) | 28/28 PASS — capability.call routing, cross-primal forwarding |
| Faculty | Dong (BAE, MSU — new lab 2026) |
| Handoff | V082 in `wateringHole/handoffs/` |
| ToadStool | S147+ — barraCuda v0.3.5 standalone (wgpu 28) |

---

## Specifications

### Validation & Reproduction

| Spec | Status | Description |
|------|--------|-------------|
| [PAPER_REVIEW_QUEUE.md](PAPER_REVIEW_QUEUE.md) | Active | Papers to review/reproduce with controls audit |
| [BARRACUDA_REQUIREMENTS.md](BARRACUDA_REQUIREMENTS.md) | Active | GPU kernel requirements + compute pipeline |
| [CROSS_SPRING_EVOLUTION.md](CROSS_SPRING_EVOLUTION.md) | Active | Cross-spring shader provenance and evolution story (S87) |
| [ATLAS_STATION_LIST.md](ATLAS_STATION_LIST.md) | Planning | Michigan 100-station expansion for crop water atlas |
| [NUCLEUS_INTEGRATION.md](NUCLEUS_INTEGRATION.md) | **Complete** | NUCLEUS deployment — primal registered, ecology domain, 28/28 pipeline |
| [BIOMEOS_CAPABILITIES.md](BIOMEOS_CAPABILITIES.md) | **Complete** | Ecology capability domain for biomeOS Neural API |

### Existing Documentation (in parent directories)

| Document | Location | Description |
|----------|----------|-------------|
| CONTROL_EXPERIMENT_STATUS.md | `../` | Detailed experiment logs and check counts |
| CHANGELOG.md | `../` | Evolution history (Keep a Changelog format) |
| experiments/README.md | `../experiments/` | Experiment index (87 completed) |
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
- **Rust evolution proof** — Python→Rust cross-validated to 1e-5, 13,000× faster
- **GPU integration proof** — 25 Tier A orchestrators wired to ToadStool/BarraCuda (BrentGpu, RichardsGpu, StatefulPipeline, SeasonalPipelineF64)
- **Cross-spring provenance** — 5/5 springs contribute shaders to airSpring pipeline

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
CROSS_SPRING_EVOLUTION.md → `../wateringHole/handoffs/` (V082 active)

---

## License

**AGPL-3.0** — GNU Affero General Public License v3.0

All airSpring code, data, and documentation are aggressively open science. See `../LICENSE` for full text. Any derivative work, including network-accessible services using airSpring code, must publish source under the same license.
