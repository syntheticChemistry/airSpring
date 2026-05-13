# airSpring Tier 2 Convergence Wave Handoff — May 13, 2026

**From**: airSpring
**To**: primalSpring (coordination), lithoSpore (LTEE module), plasmidBin (binary harvest)
**Date**: May 13, 2026
**Trigger**: primalSpring "ecoPrimals Delta Spring Evolution — Tier 2 Convergence Wave"

---

## Audit Response Summary

All 5 priority items from the convergence wave audit are addressed:

| # | Item | Status |
|---|------|--------|
| 1 | Wire `toadstool.validate` | **DONE** (May 12) — `ipc::toadstool_validate`, 8 TCP round-trip tests, composition-parity scenario |
| 2 | Wire `barracuda.precision.route` | **DONE** (May 12+13) — `ipc::precision_route`, 8 TCP tests. **May 13**: now consumes `requires_compiler` and `adapter` fields |
| 3 | LTEE handoff for lithoSpore | **DONE** — `control/ltee_fls2_plant_immunity/` packaged as lithoSpore module candidate |
| 4 | Validate plasmidBin deployment | **DONE** — musl static-pie binary (3.3 MB), `version` + `validate --list` verified |
| 5 | Surface gaps upstream | **DONE** — `docs/PRIMAL_GAPS.md` refreshed, this handoff |

---

## 1. Tier 2 IPC Wiring

### `toadstool.validate` (wired May 12)

- **Module**: `barracuda/src/ipc/toadstool_validate.rs`
- **Method constant**: `methods::TOADSTOOL_VALIDATE` = `"toadstool.validate"`
- **Params sent**: `{ "workload_path": string, "dry_run": bool }`
- **Result consumed**: `valid`, `gpu_available`, `precision_tier`, `estimated_dispatch_time_ms`, `warnings`, `required_capabilities`
- **Alignment**: Matches toadStool `handler/workload.rs` (S250) exactly
- **Tests**: 8 (TCP round-trip, RPC error, parse variants, display, Error trait, env keys)
- **Integration**: `s_composition_parity` scenario calls `toadstool.validate` with graceful skip when toadStool absent

### `barracuda.precision.route` (wired May 12, extended May 13)

- **Module**: `barracuda/src/ipc/precision_route.rs`
- **Method constant**: `methods::PRECISION_ROUTE` = `"precision.route"`
- **Params sent**: `{ "domain": string }`
- **Result consumed**: `recommended_tier`, `fma_safe`, `needs_sovereign_compile`, `requires_compiler`, `hardware_hint`, `adapter`, `rationale`
- **Alignment**: Matches barraCuda `ipc/methods/precision.rs` — all returned fields now consumed
- **Tests**: 8 (TCP round-trip with adapter field, RPC error, parse variants, display, Error trait, env keys)
- **Integration**: `s_composition_parity` scenario calls `precision.route` with graceful skip

### Wire Contract Note

The audit's example params (`workload.name/requirements`, `operation/tolerance`) describe a simplified contract. The actual upstream implementations use `workload_path`/`dry_run` (toadStool) and `domain` (barraCuda). airSpring follows the **actual primal implementations**, not the audit examples. No mismatch — the audit descriptions are conceptual summaries.

---

## 2. LTEE E3 — lithoSpore Module Candidate

**Module name**: `ltee-immunity`
**Directory**: `control/ltee_fls2_plant_immunity/`

### lithoSpore Package Layout

```
fetch_data.sh                    # generates reference data from Python baseline
ltee_fls2_plant_immunity.py      # run_baseline.py (deterministic, seed=20250511)
validate_ltee_fls2               # run_validation binary (29/29 PASS)
tolerances.toml                  # named tolerance bounds (binding, glycosylation, coupling)
expected_values.json             # ground truth (Kd, models, tolerances)
benchmark_ltee_fls2.json         # full benchmark (Python-generated reference)
README.md                        # documentation
```

### Validation Results

- Python baseline: **12/12 PASS** (3 binding models × 4 check classes)
- Rust reproduction: **29/29 PASS** (benchmark structure, model fits, Rust binding, glycosylation shift, soil-immune coupling)
- Python↔Rust parity: 1e-10 tolerance on coupling model, 1e-12 on binding calculations
- Unique contribution: **soil-immune coupling model** (moisture×temperature×microbial activity→flagellin exposure)

---

## 3. plasmidBin Deployment

```
Binary:    airspring 0.10.0 (UniBin)
Target:    x86_64-unknown-linux-musl
Linkage:   static-pie (ELF 64-bit, statically linked)
Size:      3.3 MB (not stripped)
Build:     cargo build --release --target x86_64-unknown-linux-musl --features local --bin airspring
Verify:    airspring version → "airspring 0.10.0 (UniBin)"
           airspring validate --list → 10 scenarios
```

- `rust-toolchain.toml` now includes `x86_64-unknown-linux-musl` target
- `infra/plasmidBin/manifest.toml` already lists airSpring
- `sources.toml` excludes springs by design (primal-only harvest)
- Ready for manual staging or CI harvest

---

## 4. Gaps Surfaced

### Active (upstream blockers)

| ID | Primal | Gap | Impact |
|----|--------|-----|--------|
| AG-005 | Squirrel | `inference.*` not exercised in science path | Blocked on neuralSpring WGSL inference |
| AG-006 | coralReef | Sovereign shader compile not wired | coralReef stability items in Pass 12 |
| AG-007 | toadStool | `compute.dispatch` opaque results | Need typed response for ecology workloads |
| AG-008 | NestGate | Non-standard weather method name | Ecosystem-level standardization needed |
| AG-009 | petalTongue | No direct IPC wiring | Low priority, Tier 3 convergence item |
| AG-010 | barraCuda | `TensorSession`/`TensorContext` not available | Seasonal GPU pipeline blocked |
| AG-011 | barraCuda | Anderson coupling needs WGSL shader | Tier C in GPU promotion map |

### L5 Blockers

- Live biomeOS + toadStool required for L5 certification (L5 probes print `SKIP` without running primals)
- All structural L5 validation passes via TCP mock round-trip (1,027 lib tests)

---

## 5. Metrics Snapshot

| Metric | Value |
|--------|-------|
| Lib tests | 1,035 |
| Integration tests | 316 |
| Forge tests | 62 |
| **Total tests** | **1,413** |
| Binaries | 94 |
| Validation scenarios | 10 |
| Method constants | 49 |
| Capabilities registered | 46 |
| Deploy graphs | 7 + skunkBat |
| Clippy warnings | 0 |
| Python baselines | 1,284/1,284 PASS |
| guideStone level | **L4** (targeting L5+) |

---

**Next**: L5 requires live primals. airSpring is structurally L5-ready. Awaiting biomeOS + toadStool deployment for live certification.
