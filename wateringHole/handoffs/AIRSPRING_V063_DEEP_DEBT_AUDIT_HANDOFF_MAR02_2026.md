# airSpring v0.6.3 — Deep Debt Audit & Quality Hardening

**Date**: March 2, 2026
**From**: airSpring (Eastgate)
**To**: ecoPrimals ecosystem
**ToadStool PIN**: S79 (`f97fc2ae`)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring v0.6.3 completes a comprehensive deep debt audit across the entire codebase.
810 lib tests pass at 95.66% line coverage. All clippy pedantic warnings resolved.
All benchmark JSONs have provenance metadata. All validation binaries standardized to
`ValidationHarness`. All `unwrap()` calls in binaries replaced with `.expect("context")`.
Magic numbers extracted into named constants. Sovereignty violations corrected. CI
pipeline updated with correct binary names and full validation coverage. `cargo-deny`
configuration corrected for SPDX compliance.

## What Changed

### 1. Clippy Pedantic (21 errors → 0)

Resolved across 11 files:
- `suboptimal_flops`: `#[allow]` in test modules (readability over micro-optimization)
- `manual_range_contains`: replaced with `.contains()`
- `manual_let_else` / `single_match_else`: refactored to `let...else` pattern
- `unused_must_use`: wrapped with `let _ =`
- `redundant_clone`: removed unnecessary `.clone()` in property tests
- `doc_markdown`: backtick-wrapped identifiers (BarraCuda, etc.)
- `float_cmp`: replaced `!=` with `.abs() > f64::EPSILON`
- `cast_precision_loss` / `cast_possible_truncation`: file-level `#[allow]` in test files
- `double_must_use`: removed redundant `#[must_use]` from `Result`-returning functions
- `implicit_clone`: `.to_string()` → `.clone()` when type is `&String`
- `uninlined_format_args`: `format!("{}", var)` → `format!("{var}")`

### 2. Validation Binary Standardization

- `validate_richards.rs` and `validate_biochar.rs`: fully refactored from ~24 `process::exit(1)` calls each to `ValidationHarness`
- `validate_cw2d.rs`: `process::exit(1)` → `.unwrap_or_else(|| panic!(...))` with context
- 10+ binaries: all bare `unwrap()` → `.expect("descriptive context")`

### 3. Magic Number Extraction

- `validate_gpu_rewire_benchmark.rs`: 15+ magic numbers → named constants with doc comments
- `validate_cpu_gpu_parity.rs`: domain constants extracted, doc_markdown fixed

### 4. Data Provenance & Tolerance Justification

All 7 benchmark JSONs now have `_provenance` (script, command, baseline_commit, date)
and `_tolerance_justification` metadata:
- `barrier_skin`, `cross_species_skin`, `cytokine_brain`, `tissue_diversity` (Paper 12)
- `metalforge_dispatch`, `cpu_gpu_parity`, `hargreaves`

### 5. Sovereignty Hardening

- `airspring_primal.rs`: hardcoded external primal names ("toadstool", "nestgate") →
  capability-based descriptions ("compute primal", "data primal")
- `const PRIMAL_NAME: &str = "airspring"` for self-references
- `main()` refactored to `run() -> Result<(), String>` (no more `process::exit(1)`)

### 6. Test Coverage Expansion

810 tests (up from ~750), 95.66% line coverage (up from 94.27%):
- `rpc.rs`: +11 tests (JSON-RPC construction, send error handling, response parsing)
- `biomeos.rs`: +10 tests (socket discovery, capability parsing, error paths)
- Fixed flaky RPC mock tests (server must read request before writing response)

### 7. CI Pipeline Fixes

- Fixed binary name mismatches: `validate_soil_moisture` → `validate_soil`,
  `validate_coupled_ri` → `validate_coupled_runoff`
- Added 4 Paper 12 Python baselines to CI
- Added `cargo doc --no-deps` to metalForge quality job
- Added 4 missing `[[bin]]` entries to Cargo.toml: `validate_barrier_skin`,
  `validate_cross_species`, `validate_cytokine`, `validate_tissue`

### 8. cargo-deny Configuration

- Fixed invalid SPDX: `"AGPL-3.0+"` → `"AGPL-3.0-or-later"`, `"AGPL-3.0"` → `"AGPL-3.0-only"`
- Fixed deprecated field: `vulnerability = "deny"` → removed, `unmaintained = "workspace"`

## Quality Gates

| Gate | Status |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --all-targets -- -D warnings -W clippy::pedantic` | PASS (0 warnings) |
| `cargo doc --no-deps` | PASS (both crates) |
| `cargo test --lib` | PASS (810 tests) |
| `cargo llvm-cov --lib --fail-under-lines 90` | PASS (95.66% lines) |
| `cargo-deny check` | PASS (SPDX clean) |
| All validation binaries (CPU) | PASS |
| Cross-spring evolution benchmark | 124/124 PASS |

## Files Changed

### Updated
- `.github/workflows/ci.yml` — binary name fixes, Paper 12 baselines, metalForge doc check, validate job
- `barracuda/Cargo.toml` — 4 new `[[bin]]` entries
- `barracuda/deny.toml` — SPDX + field corrections
- `barracuda/src/eco/cytokine.rs` — clippy fixes in test module
- `barracuda/src/nautilus.rs` — clippy fixes in test module
- `barracuda/src/gpu/bootstrap.rs` — `let...else` refactor
- `barracuda/src/gpu/atlas_stream.rs` — `unused_must_use` fix, clone cleanup
- `barracuda/src/gpu/pedotransfer.rs` — test readability fix
- `barracuda/src/gpu/thornthwaite.rs` — test `#[allow]`
- `barracuda/src/eco/crop.rs` — removed redundant `#[must_use]`
- `barracuda/src/eco/richards.rs` — removed redundant `#[must_use]`, justified allows
- `barracuda/src/eco/tissue.rs` — removed redundant `#[must_use]`, justified allows
- `barracuda/src/eco/yield_response.rs` — removed redundant `#[must_use]`
- `barracuda/src/io/csv_ts.rs` — `implicit_clone` fix
- `barracuda/src/biomeos.rs` — +10 tests, `uninlined_format_args` fix
- `barracuda/src/rpc.rs` — +11 tests, mock server fix, `uninlined_format_args` fix
- `barracuda/src/bin/airspring_primal.rs` — sovereignty hardening
- `barracuda/src/bin/validate_richards.rs` — `ValidationHarness` migration
- `barracuda/src/bin/validate_biochar.rs` — `ValidationHarness` migration
- `barracuda/src/bin/validate_cw2d.rs` — error handling fix
- `barracuda/src/bin/validate_gpu_rewire_benchmark.rs` — magic numbers → constants
- `barracuda/src/bin/validate_cpu_gpu_parity.rs` — magic numbers → constants
- `barracuda/src/bin/bench_cross_spring_evolution.rs` — `unwrap()` → `.expect()`
- `barracuda/src/bin/bench_cross_spring/main.rs` — `unwrap()` → `.expect()`
- `barracuda/tests/property_tests.rs` — doc_markdown, float_cmp, redundant_clone fixes
- `barracuda/tests/cross_spring_absorption.rs` — cast lint allows
- `control/*/benchmark_*.json` — provenance + tolerance justification (7 files)

## Cross-Spring Impact

- **ToadStool**: No changes needed. airSpring S79 pin stable.
- **All Springs**: Deep debt patterns (ValidationHarness, provenance JSON, named constants) available as reference for other springs' audits.
- **biomeOS**: airSpring primal sovereignty-compliant (no hardcoded external primal names).

---

AGPL-3.0-or-later
