# airSpring V0.10.0 — Ecosystem Absorption Handoff (March 24, 2026)

**From**: airSpring v0.10.0 → barraCuda, toadStool, primalSpring, sibling Springs
**Date**: 2026-03-24
**Status**: 943 lib + 316 integration + 62 forge = 1,321 total tests, 91 binaries, 87 experiments, zero clippy (pedantic+nursery), zero unsafe, zero C deps, zero `#[allow()]`

---

## What Changed

### PRIMAL_REGISTRY.md (P0)
- airSpring entry updated: v0.7.6 → v0.10.0 (1,321 tests, 91 binaries, three-tier capability discovery)
- barraCuda: v0.3.5 → v0.3.7; hotSpring → v0.6.32; groundSpring → V122; neuralSpring → S174; wetSpring → V135
- Spring versions table date bumped to March 24, 2026

### CONTRIBUTING.md + SECURITY.md (P1 — neuralSpring S174 pattern)
- `CONTRIBUTING.md`: prerequisites (Rust 1.92+, Edition 2024), quality table (clippy pedantic+nursery, fmt, doc, deny, forbid unsafe, expect-not-allow, 1000 LOC, SPDX), tolerance policy, validation binary pattern, barraCuda evolution lifecycle, IPC coordination (self-knowledge only, three-tier discovery), commit conventions
- `SECURITY.md`: supported versions, security model, vulnerability reporting, data provenance

### Upstream Contract Pinning (P1 — neuralSpring S174 pattern)
- `tolerances/mod.rs`: `all_tolerances()` → `const fn` returning `&'static [&'static Tolerance]` (58 entries)
- New `upstream_contract_tests` module: positive thresholds, no duplicates, justification presence, GPU/CPU parity tighter than science, registry count ≥58
- Pattern adopted from neuralSpring S174 for verifying upstream tolerance constants haven't drifted

### GPU Test Resilience (P2 — barraCuda `test_pool` pattern)
- `tests/common/mod.rs`: `try_create_device()` delegates to `barracuda::device::test_pool::get_test_device_if_gpu_available()` — retry with exponential backoff + device health checks, instead of per-test device creation
- Three `#[allow()]` → `#[expect()]` with reason strings (Rust 2024 complete in test infrastructure)

### Deploy Graph Metadata (P2 — primalSpring v0.7.0 pattern)
- `[graph.metadata]` added to all 4 deploy graphs with tailored `capabilities_required` per graph
- Fields: `spring`, `version`, `domain`, `license`, `updated`, `capabilities_required`
- Enables biomeOS to programmatically inspect graph requirements without parsing nodes

### Debt Cleanup
- `error.rs` + `ipc/resilience.rs` tests: hardcoded `"nestgate"`/`"toadstool"` → `primal_names::NESTGATE`/`primal_names::TOADSTOOL`
- `validate_cross_spring_provenance.rs`: doc backtick formatting for clippy `doc_markdown`
- `validate_npu_funky_eco.rs`: `#[expect(clippy::cast_sign_loss)]` with mathematical justification
- Version banner corrected: v0.7.3 → v0.10.0
- All docs reconciled: 943 lib tests, 1,321 total, 45 capabilities, March 24 2026

---

## What We Learned (Relevant to Team Evolution)

### For barraCuda
1. **`all_tolerances() const fn` pattern**: Returning `&'static [&'static Tolerance]` enables compile-time tolerance inventory. Consider making `barracuda::tolerances` provide a similar registry function for springs to verify against.
2. **Contract pinning**: Springs should verify upstream tolerance constants haven't silently changed. The `upstream_contract_tests` module is a lightweight pattern (~30 lines) that catches drift.
3. **`test_pool` vs per-test device creation**: The pooled device with retry+backoff is significantly more resilient on shared CI machines. All springs should migrate.

### For toadStool
1. **Deploy graph metadata**: `[graph.metadata]` makes graphs self-describing. biomeOS can filter graphs by domain, spring, or required capabilities without parsing the full DAG.
2. **`capabilities_required` array**: Enables pre-flight capability checks before graph execution — fail fast if a required capability isn't available.

### For primalSpring
1. **Graph metadata adoption**: primalSpring's v0.7.0 pattern works well. The `capabilities_required` field is the key enabler for programmatic graph selection.
2. **`primal_names` in test code**: Even test helpers should use centralized constants rather than string literals. Prevents silent drift when primal names change.

### For sibling Springs
1. **CONTRIBUTING.md + SECURITY.md**: Simple, high-value files. The neuralSpring S174 pattern is worth adopting across all springs.
2. **Upstream contract pinning**: All springs consuming barraCuda tolerances should add similar invariant tests.
3. **`#[expect()]` in test infrastructure**: Shared test helpers that aren't used by every test crate trigger `dead_code`/`unused_macros` warnings. `#[expect(lint, reason = "...")]` is the Rust 2024 solution.

---

## Absorption Candidates for Upstream

| Module | Destination | Description | Priority |
|--------|-------------|-------------|----------|
| `all_tolerances() const fn` | barraCuda | Registry pattern for tolerance inventory | Medium |
| Graph metadata schema | biomeOS spec | Standardize `[graph.metadata]` across ecosystem | Medium |
| Contract pinning test pattern | wateringHole guidance | Document as ecosystem standard | Low |
| `CONTRIBUTING.md` template | wateringHole | Standard template for all springs | Low |

---

## Deferred Items

| Item | Blocker | ETA |
|------|---------|-----|
| `ValidationSink` trait | ludoSpring V30 → barraCuda absorption | When upstream absorbs |
| `TensorSession` fused pipelines | barraCuda 0.4.x | When available |
| Feature-matrix CI | wetSpring V135 pattern stabilization | Medium-term |

---

## Verification

```bash
cd barracuda && cargo test --lib --all-features
# 943 passed; 0 failed

cd barracuda && cargo clippy --workspace --all-features -- -D warnings
# 0 warnings

cd barracuda && cargo test --tests --all-features
# 316+ passed; 0 failed

cd metalForge/forge && cargo test
# 62 passed; 0 failed
```

All green. Zero regressions.
