# airSpring — Niche Atomic Convergence Handoff (May 13, 2026)

**Version**: v0.10.0 | **guideStone**: L4 (structural L5 complete)
**Upstream pull**: All 13 primals pulled; barraCuda, toadStool, coralReef had new evolution
**License**: AGPL-3.0-or-later

---

## Audit Response

Responding to **Delta Spring Evolution — Upstream Clear, Niche Atomic Convergence (May 13, 2026)** from primalSpring.

airSpring is a **cross-atomic validator**. Per audit directive, we hold on full NUCLEUS compositions until Tower (ludoSpring), Node (hotSpring), and Nest (healthSpring) confirm live atomic validation. Focus: **deepen niche** — LTEE E3, NestGate/Squirrel wiring, gS L5+.

---

## Completed This Wave

### 1. NestGate CAS Typed Client (AG-008 RESOLVED)

- **`ipc::nestgate_data`** — `content.store`, `content.get`, `storage.status`
- NestGate is a **storage primal** — it does not implement `data.*` methods
- The `data.weather` handler was forwarding `data.open_meteo_weather` to NestGate — a method that **does not exist** on NestGate's wire surface
- **Fix**: evolved to `capability.call` routing with `operation: "weather.daily"`
- 8 TCP round-trip tests, standard transport discovery (`NESTGATE_SOCKET` / `NESTGATE_ADDRESS` / biomeOS)

### 2. Squirrel Inference Typed Client (AG-005 IPC Wired)

- **`ipc::squirrel_inference`** — `inference.embed`, `inference.complete`, `inference.models`
- Domain uses: embed soil sensor profiles for similarity search, structured JSON crop parameter suggestions, model discovery prelude for experiment dispatch
- 8 TCP round-trip tests, standard transport discovery (`SQUIRREL_SOCKET` / `SQUIRREL_ADDRESS` / biomeOS)
- **AG-005 partially resolved**: IPC typed client wired; science path (`ecology.experiment`) does not yet call inference methods — awaiting cross-atomic composition clearance

### 3. Composition-Parity Scenario Extended

- NestGate `storage.status` probe added (Tier 2 skip-if-absent)
- Squirrel `inference.models` probe added (Tier 2 skip-if-absent)
- Now 4 Tier 2 primal probes: toadStool + barraCuda + NestGate + Squirrel

### 4. Method Constants

6 new constants in `methods.rs` (61 total):
- `CONTENT_STORE`, `CONTENT_GET`, `STORAGE_STATUS` (NestGate)
- `INFERENCE_EMBED`, `INFERENCE_COMPLETE`, `INFERENCE_MODELS` (Squirrel)

### 5. Tier 2 Re-verification

After upstream pull (barraCuda, toadStool, coralReef all updated):
- `toadstool_validate`: 9/9 PASS
- `precision_route`: 8/8 PASS (includes `requires_compiler` + `adapter` fields)
- Zero clippy warnings, zero regressions

---

## Metrics Snapshot

| Metric | Value |
|--------|-------|
| Lib tests | **1,051** (was 1,035) |
| Total Rust tests | **1,429** (1,051 lib + 316 integration + 62 forge) |
| IPC modules | **13** (`ipc/mod.rs` submodules) |
| Method constants | **61** (`methods.rs`) |
| Capabilities registered | **46** (niche + infrastructure) |
| CPU vs Python parity | **25/25** |
| guideStone Level | **L4** (structural L5 complete — blocked on live primals only) |
| Clippy warnings | **0** |

---

## Active Gaps

| ID | Primal | Status | Summary |
|----|--------|--------|---------|
| AG-005 | Squirrel | **Partial** | IPC typed client wired; science path integration pending |
| AG-006 | coralReef | Open | Sovereign shader compile not wired |
| AG-007 | toadStool | Open | `compute.dispatch` returns opaque results |
| AG-009 | petalTongue | Open | No direct IPC wiring (low priority, Tier 3) |
| AG-010 | barraCuda | Open | TensorSession/TensorContext not available |
| AG-011 | barraCuda | Open | Anderson coupling needs WGSL shader |

---

## Holding Per Audit Directive

**Not expanding** to full NUCLEUS compositions until:
- ludoSpring confirms Tower atomic live validation
- hotSpring confirms Node atomic live validation
- healthSpring confirms Nest atomic live validation

airSpring's next composition tier (Tower+Nest cross-atomic for weather→storage→science pipeline) awaits these upstream confirmations.

---

**This handoff consumed by primalSpring.** See also: `docs/PRIMAL_GAPS.md`
