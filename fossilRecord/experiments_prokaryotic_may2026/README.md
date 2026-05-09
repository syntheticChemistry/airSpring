# Fossil: Prokaryotic Experiment Crates (May 2026)

**Fossilized**: 2026-05-09
**Superseded by**: `barracuda/src/validation/scenarios/` (UniBin eukaryotic)

## What was absorbed

| Crate | Scenarios | UniBin scenario |
|-------|-----------|-----------------|
| `experiments/exp001_local_science_parity/` | 55/55 local science dispatch | `s_local_science_parity.rs` |
| `experiments/exp002_composition_parity/` | 10/10 NUCLEUS niche parity | `s_composition_parity.rs` |
| `experiments/exp003_foundation_target_validation/` | 4/4 foundation thread06 targets | `s_foundation_targets.rs` |

## Why fossilized

The interstadial eukaryotic evolution wave (primalSpring v0.9.25) directed
springs to absorb standalone experiment crates into the UniBin's
`validation/scenarios/` module. This eliminates separate Cargo workspaces,
unifies the validation registry, and enables `airspring validate --tier rust`
to run all scenarios from a single binary.

## Originals

The standalone crate sources remain in `experiments/exp00{1,2,3}_*/` as
reference. They compile independently but are no longer the canonical runners.
The canonical path is `airspring validate`.
