# airSpring V0.7.3 — Modern Upstream Integration Handoff

SPDX-License-Identifier: AGPL-3.0-or-later
**Date**: March 7, 2026
**From**: airSpring
**Supersedes**: AIRSPRING_V072_UPSTREAM_LEAN_HANDOFF_MAR07_2026.md

---

## Summary

airSpring v0.7.3 completes the modern upstream integration cycle, wiring three
new barraCuda capabilities into the airSpring GPU layer:

1. **`PrecisionRoutingAdvice`** (toadStool S128) — per-hardware precision dispatch
2. **Upstream provenance registry** (`barracuda::shaders::provenance`) — programmatic
   cross-spring shader evolution tracking
3. **Cross-spring evolution benchmark** — exercises all 20 ops with provenance

This builds on v0.7.2's Write→Absorb→Lean completion (ops 14-19 absorbed upstream).

---

## New Wiring

### PrecisionRoutingAdvice

`DevicePrecisionReport` now includes a `precision_routing` field from
`GpuDriverProfile::precision_routing()`:

| Variant | Meaning | Dispatch Recommendation |
|---------|---------|------------------------|
| `F64Native` | Full native f64 everywhere | All 20 ops at native f64, workgroup reductions OK |
| `F64NativeNoSharedMem` | Native f64 compute, shared-mem broken | BatchedElementwiseF64 OK, reductions → DF64/scalar |
| `Df64Only` | No reliable native f64 | All ops via `compile_shader_universal → Df64` (~48-bit) |
| `F32Only` | No f64 support at all | Edge/inference only, not suitable for science pipelines |

This captures the shared-memory reliability axis that `Fp64Strategy` alone does not.
airSpring GPU modules can use this to route workloads at dispatch time.

### Upstream Provenance Registry

Three new functions in `gpu::device_info`:

- `upstream_airspring_provenance()` — shaders consumed by airSpring from upstream
- `upstream_evolution_report()` — full markdown cross-spring evolution report
- `upstream_cross_spring_matrix()` — `(from, to)` → shader count dependency map

These query `barracuda::shaders::provenance` which tracks 27 shaders with origin,
consumers, categories, creation dates, and absorption dates.

### Cross-Spring Evolution Benchmark

`bench_cross_spring_evolution/modern.rs` validates the full modern pipeline:

1. Prints upstream provenance registry with per-shader details
2. Prints cross-spring dependency matrix
3. Reports `PrecisionRoutingAdvice` for the current GPU
4. Benchmarks ops 14-19 (Makkink, Turc, Hamon, SCS-CN, Stewart, Blaney-Criddle)
5. Validates CPU↔GPU parity for each op
6. Batch scaling test (100 → 100K elements)

---

## Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets` | PASS (0 errors) |
| `cargo doc --no-deps` | PASS |
| `cargo test --lib` | **848/848 PASS** |

---

## Cross-Spring Shader Provenance (Ecosystem State)

### Dependency Matrix (shader count)

| From \ To | hotSpring | wetSpring | neuralSpring | airSpring | groundSpring |
|-----------|-----------|-----------|--------------|-----------|--------------|
| **hotSpring** | — | ✓ | ✓ | ✓ | ✓ |
| **wetSpring** | · | — | ✓ | ✓ | ✓ |
| **neuralSpring** | ✓ | ✓ | — | ✓ | ✓ |
| **airSpring** | · | ✓ | · | — | · |
| **groundSpring** | · | ✓ | ✓ | ✓ | — |

### Key Cross-Spring Flows

- **hotSpring → all springs**: `df64_core`, `math_f64`, precision routing, DF64 transcendentals
- **wetSpring → neuralSpring, airSpring**: Shannon/Simpson diversity, kriging, HMM bio shaders
- **neuralSpring → all springs**: `compile_shader_universal`, Welford mean variance, batch IPR
- **airSpring → wetSpring**: Hargreaves ET₀, seasonal pipeline, moving_window_f64
- **groundSpring → airSpring, wetSpring**: MC propagation, bootstrap, Anderson Lyapunov

---

## Ecosystem Sync Status

| System | Version | Status |
|--------|---------|--------|
| **barraCuda** | 0.3.3+ (unreleased → 0.3.4) | Fully synced — 20 ops, provenance registry, PrecisionRoutingAdvice |
| **ToadStool** | S130 | Synced — shader.compile.* proxy, toadstool.provenance, CoralReefClient |
| **coralReef** | Phase 10 | Aware — 14/27 cross-spring shaders compile; airSpring not direct consumer |

---

## Future Opportunities

| Capability | Source | Benefit |
|------------|--------|---------|
| `mean_variance_to_buffer()` | barraCuda 0.3.4 | Zero-readback Welford for chained GPU pipelines |
| `GpuView<T>` | barraCuda 0.3.4 | Persistent buffer API (80×–600× reduction improvement) |
| `BatchedOdeRK45F64` | barraCuda 0.3.4 | Adaptive RK45 on GPU for dynamic soil models |
| `toadstool.provenance` | ToadStool S130 | JSON-RPC cross-spring introspection at runtime |
| `shader.compile.wgsl` | ToadStool S130 | Sovereign compilation proxy (coralReef via ToadStool) |

---

## Notes for Ecosystem Teams

### For barraCuda team

- airSpring now queries `shaders::provenance` registry — any breaking changes to the
  `ShaderRecord`, `SpringDomain`, or query functions will affect us.
- `PrecisionRoutingAdvice` is wired; any new variants should be handled gracefully
  (default to `Df64Only` for unknown).

### For ToadStool team

- airSpring references `toadstool.provenance` in documentation but does not call it
  at runtime yet. When biomeOS integration begins, we will use JSON-RPC for runtime
  cross-spring introspection.

### For coralReef team

- `local_elementwise_f64.wgsl` in the coralReef test corpus was retired in v0.7.2.
  Suggest pointing to `batched_elementwise_f64.wgsl` from barraCuda instead.
