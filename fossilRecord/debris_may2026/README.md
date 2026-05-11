# Debris Archive — May 2026

Orphaned artifacts cleaned during May 11, 2026 documentation reconciliation pass.

| File | Origin | Reason |
|------|--------|--------|
| `benchmark_regional_et0.json` | `control/regional_et0/` | Unreferenced — `validate_regional_et0.rs` discovers stations from filesystem, does not use this JSON fixture. Kept as fossil record. |

## Known Provenance Gap

`control/ncbi_diversity/ncbi_diversity_analysis.py` — referenced in `provenance.rs` and `benchmark_ncbi_diversity.json` but missing from tree. Original script was at commit `88d07c0` (Feb 28, 2026). The benchmark JSON retains all expected values so the Rust validator (`validate_ncbi_diversity`) works correctly. Restore from git history if the Python baseline needs re-running.
