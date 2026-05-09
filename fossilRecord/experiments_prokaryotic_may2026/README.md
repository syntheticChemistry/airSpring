# Fossil Record: Prokaryotic Experiment Binaries

**What**: Standalone experiment crates (`exp001_local_science_parity`,
`exp002_composition_parity`, `exp003_foundation_target_validation`) from the
prokaryotic era of separate validation binaries.

**When**: Pre-interstadial (before May 2026 eukaryotic evolution).

**Why fossilized**: These experiment crates are absorbed into the UniBin as
`validation/scenarios/` modules (`s_local_science_parity`, `s_composition_parity`,
`s_foundation_targets`). The standalone crates are no longer the primary
validation path — the UniBin's `airspring validate` command replaces them.

**What supersedes**:
- `exp001` → `barracuda/src/validation/scenarios/s_local_science_parity.rs`
- `exp002` → `barracuda/src/validation/scenarios/s_composition_parity.rs`
- `exp003` → `barracuda/src/validation/scenarios/s_foundation_targets.rs`

**Original locations**:
- `experiments/exp001_local_science_parity/`
- `experiments/exp002_composition_parity/`
- `experiments/exp003_foundation_target_validation/`

The standalone crates remain in `experiments/` as build artifacts for CI
backward compatibility. They share the same validation logic as the absorbed
scenarios but execute as independent binaries rather than through the
UniBin scenario runner.

**Provenance**: airSpring v0.10.0, interstadial transition, May 2026.
