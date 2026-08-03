# airSpring — Wave 46 Absorption Handoff

**Date**: May 23, 2026
**Spring**: airSpring v0.10.0 (ecology / agriculture)
**Upstream**: primalSpring v0.9.27 (Wave 46)
**Gap Closed**: Waves 21–46 (26-wave delta)

---

## Summary

airSpring absorbed primalSpring Wave 46 infrastructure, closing the 26-wave
gap from the Wave 20 lithoSpore absorption. Key deliverables:

1. **458-method registry sync** — cross-sync test and 11 doc files updated
2. **NeuralBridge observatory** — new IPC module for biomeOS v3.67+ adaptive routing
3. **BLAKE3 provenance** — all 62 benchmark JSONs now carry `blake3` hashes
4. **SP-4 sovereign publish** — NestGate content.put pipeline adopted

---

## Changes by Category

### Registry & Documentation

| Item | Detail |
|------|--------|
| Cross-sync test | `>= 452` → `>= 458` in `barracuda/tests/capability_cross_sync.rs` |
| Doc sweep | 11 `.md` files: 445 → 458 method count (stale since Wave 36 recount) |
| PRIMAL_GAPS | Wave 46 section added; date updated |
| DEGRADATION_BEHAVIOR | Neural API observatory table added |
| CHANGELOG | Wave 46 entry |

### NeuralBridge Observatory (NEW)

**File**: `barracuda/src/ipc/neural_bridge.rs`

Public API surface:

| Function | biomeOS Version | Purpose |
|----------|----------------|---------|
| `capability_call_instrumented()` | v3.67+ | Dispatch + BridgeOutcome for adaptive routing |
| `routing_weights()` | v3.67+ | Adaptive routing weight snapshot |
| `route_explain(method)` | v3.67+ | Routing decision explanation |
| `utilization()` | v3.67+ | Real-time utilization metrics |
| `weight_health()` | v3.70+ | Convergence diagnostics |
| `composition_patterns()` | v3.67+ | Named composition patterns |

Integrated into `composition.status` handler — reports `observatory.neural_api_v3_67`
health status. All observatory methods degrade gracefully (NoPrimal error).

### BLAKE3 Provenance Backfill

- **62 benchmark JSONs** now have `blake3` field in `_provenance`
- **New tool**: `tools/blake3_backfill.sh` — idempotent, re-runnable
- Aligns with FN-1 (projectFOUNDATION Thread 10) and SP-4 NestGate CAS

### SP-4 Sovereign Publish

- **New tool**: `tools/publish_sporeprint.sh` — content.put to NestGate
- Base64 encoding + BLAKE3 hash + `source: "airspring"` metadata
- Dry-run mode verified (2 files: README.md + validation-summary.md)
- Requires NestGate UDS + bearDog content.* scope (Wave 108+)

### Clippy & Tests

- Zero clippy warnings (`-D warnings`)
- 4 new NeuralBridge unit tests (all passing)
- Cross-sync test passes against 458-method canonical registry

---

## Remaining Gaps (Deferred)

| Gap | Priority | Notes |
|-----|----------|-------|
| IonicContractRegistry | P3 | healthSpring is reference; not needed for core science |
| Dark Forest gate scenario | P2 | PENDING per DOWNSTREAM_PATTERN_GUIDE |
| Tier 4 guidestone rewiring | P2 | G column PENDING per scorecard — all springs |
| Live Neural API scenarios (S47–S49) | P1 | Blocked on live biomeOS v3.67+ availability |
| Cross-tier L3 parity (WS-9) | P1 | Ongoing science depth expansion |

---

## Verification

```bash
cargo check --features local                  # zero errors
cargo clippy --features local -- -D warnings  # zero warnings
cargo test --lib --features local neural_bridge  # 4/4 pass
./tools/publish_sporeprint.sh --dry-run       # 2 files, 0 failed
./tools/blake3_backfill.sh                    # 62/62 hashed
```
