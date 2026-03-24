# Contributing to airSpring

**License**: AGPL-3.0-or-later (scyBorg trio: AGPL + ORC + CC-BY-SA)

airSpring is part of the [ecoPrimals](https://github.com/ecoPrimals) ecosystem.
Contributions that strengthen validation fidelity, improve barraCuda absorption,
or advance GPU evolution are welcome.

## Prerequisites

- **Rust 1.92+** (edition 2024)
- **barraCuda** checkout at `../../barraCuda` (path dependency)
- GPU with Vulkan support (optional — CI validates on llvmpipe fallback)
- Python 3.13+ with NumPy (for baseline regeneration)

## Quality Standards

Every change must maintain these invariants:

| Standard | Requirement |
|----------|-------------|
| `cargo clippy --workspace --all-features` | Zero warnings (pedantic + nursery) |
| `cargo fmt --check` | Zero formatting drift |
| `cargo doc --workspace --no-deps` | Zero warnings (`-D warnings`) |
| `cargo test --workspace --lib` | All pass |
| `cargo llvm-cov --workspace --lib` | ≥ 90% line coverage |
| `cargo deny check` | Zero advisories, license, or source violations |
| `#![forbid(unsafe_code)]` | Zero unsafe in application code |
| `#[allow()]` | Zero in production — use `#[expect(lint, reason = "...")]` |
| File size | All `.rs` files under 1,000 lines |
| SPDX | Every `.rs` file starts with `// SPDX-License-Identifier: AGPL-3.0-or-later` |

## Tolerance Policy

All numeric thresholds live in `barracuda/src/tolerances/`. No ad-hoc magic numbers in
validation binaries. Each tolerance constant must have:

- A descriptive name
- A doc comment explaining the mathematical justification
- Registration in the appropriate domain submodule (`evapotranspiration.rs`, `hydrology.rs`, `gpu.rs`, `numerics.rs`)

## Validation Binaries

Follow the hotSpring pattern:

1. Use `ValidationHarness` for all checks
2. Include `//! Provenance:` header with script, commit, date, run command
3. Exit 0 on all-pass, exit 1 on any failure
4. Reference tolerances from `barracuda/src/tolerances/`
5. Include structured `_provenance` in benchmark JSON

## barraCuda Evolution

When adding GPU compute:

1. Check if barraCuda already provides the primitive (`barracuda::ops`, `stats`, `linalg`, `pde`)
2. If yes: delegate, don't duplicate
3. If no: implement locally in `gpu/`, validate, then hand off via wateringHole
4. Track in `specs/GPU_PROMOTION_MAP.md` and `barracuda/EVOLUTION_READINESS.md`

## IPC / Primal Coordination

Primal code has **self-knowledge only**. Never hardcode another primal's name,
socket path, or capability set. Discover at runtime via:

- Three-tier pattern: env override → named socket scan → capability probe
- `discover_primal_by_capability(domain)` for pure capability discovery
- Graceful degradation when peers are unavailable

## Commit Conventions

- Keep a Changelog format in `CHANGELOG.md`
- Handoffs go to `wateringHole/handoffs/` with naming convention:
  `AIRSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

## Running the Full Suite

```bash
cd barracuda && cargo test --lib              # 938 lib tests
cd barracuda && cargo test --tests            # 316 integration tests
cd metalForge/forge && cargo test             # 62 forge tests
cd barracuda && cargo llvm-cov --lib          # coverage (≥90%)
cd barracuda && cargo clippy -- -D warnings   # zero warnings
cd barracuda && cargo deny check              # dependency audit
```
