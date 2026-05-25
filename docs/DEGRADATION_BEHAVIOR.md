# airSpring Degradation Behavior

> When an upstream primal is unreachable, what does the consumer see?

**Date**: May 25, 2026 (Wave 50 Covalent HPC)
**Context**: lithoSpore R1 — primalSpring documented `CompositionContext`
degradation; each spring documents its own. Updated for `NeuralBridge`
observatory (v3.67+).

---

## Design Principle

Science never gates behind provenance or infrastructure availability.
`has_capability()` before `call()`. Provenance is enrichment, not a
prerequisite. Domain logic always succeeds or fails on its own terms.

---

## Per-Primal Degradation Table

| Primal | Module | On Unreachable | Science Gated? |
|--------|--------|----------------|:--------------:|
| biomeOS (Neural API) | `ipc/provenance.rs` | `ProvenanceCompletion { status: "unavailable", primals_reached: [] }` | **No** |
| rhizoCrypt (DAG) | `ipc/provenance.rs` | Legacy: `status: "unavailable"`, signal: falls to legacy path | **No** |
| loamSpine (commit) | `ipc/provenance.rs` | Legacy: `status: "partial"`, `primals_reached: ["rhizoCrypt"]` | **No** |
| sweetGrass (braid) | `ipc/provenance.rs` | `status: "partial"`, `primals_reached` omits `"sweetGrass"` | **No** |
| NestGate (CAS) | `ipc/nestgate_data.rs` | `Result::Err(NoPrimal \| Ipc)` — callers handle per-context | **No** — niche science uses local/control data |
| toadStool (validate) | `ipc/toadstool_validate.rs` | `Result::Err(NoPrimal)` — scenarios `check_skip` | **No** |
| toadStool (dispatch) | `ipc/compute_dispatch.rs` | `Result::Err(NoComputePrimal)` — dispatch-only path | **No** |
| barraCuda (precision) | `ipc/precision_route.rs` | `Result::Err(NoPrimal)` — falls back to conservative local f64 | **No** |
| Squirrel (inference) | `ipc/squirrel_inference.rs` | `Result::Err(NoPrimal)` — `primal_science` returns degradation JSON | **No** |
| skunkBat (audit) | `ipc/skunkbat.rs` | `Option::None` + `warn!` log | **No** |
| biomeOS (method.register) | `ipc/method_register.rs` | `Option::None` + `warn!` log | **No** |
| barraCuda (forward) | `ipc/barracuda_route.rs` | `Option::None` — module inactive, no production callers | **No** |

---

## Provenance Trio Partial Completion States

Per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md`, trio commits are **not atomic**.
airSpring reports partial state via `primals_reached` in `ProvenanceCompletion`:

| State | `status` | `primals_reached` | Meaning |
|-------|----------|-------------------|---------|
| Full trio | `"complete"` | `["rhizoCrypt", "loamSpine", "sweetGrass"]` | DAG + spine + braid |
| DAG + spine, no braid | `"partial"` | `["rhizoCrypt", "loamSpine"]` | Attribution without permanence |
| DAG only, no commit | `"partial"` | `["rhizoCrypt"]` | Merkle root exists, no ledger entry |
| No transport | `"unavailable"` | `[]` | Neural API not discoverable |
| Signal path success | `"complete"` or `"partial"` | `["rhizoCrypt", "loamSpine", "sweetGrass"]` | Signal reports braid status |

Domain logic **never panics** on partial provenance. Handlers always return
valid JSON with the `provenance` field indicating the pipeline's reach.

---

## Degradation Patterns

### 1. Capability-gated dispatch

```
if composition.has_capability("nest.store") {
    ctx.dispatch("nest.store", params);   // enrichment
}
// science always runs regardless
```

### 2. Try-or-skip IPC

```
let validation = toadstool_validate(&workload);
match validation {
    Ok(report) => use_report(report),
    Err(NoPrimal) => skip_validation(),  // proceed without pre-flight
}
```

### 3. Degradation JSON envelope

Squirrel inference returns structured degradation payloads that callers
can distinguish from real results:

```json
{
  "degraded": true,
  "reason": "inference primal unreachable",
  "fallback": "local_embedding_unavailable"
}
```

---

## Neural API Observatory (v3.67+)

| Method | On Unreachable | Impact |
|--------|----------------|--------|
| `neural_api.routing_weights` | `Err(NoPrimal)` | No routing weight visibility |
| `neural_api.route_explain` | `Err(NoPrimal)` | No routing decision explanation |
| `neural_api.utilization` | `Err(NoPrimal)` | No utilization metrics |
| `neural_api.weight_health` | `Err(NoPrimal)` | No convergence diagnostics |
| `capability_call_instrumented` | `(Err, BridgeOutcome { success: false })` | Fallback to direct dispatch |

All observatory methods are informational — science dispatch never depends
on observatory availability. `composition.status` handler reports
`observatory.neural_api_v3_67: false` when biomeOS is unavailable.

## Inactive Modules

`ipc/barracuda_route.rs` — generic barraCuda forwarder with no production
call sites. Retained for future absorption; returns `None` on failure.
