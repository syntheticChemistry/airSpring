# archive/ — Historical Artifacts

Superseded scripts and data preserved as fossil record. These are not used
by the active pipeline but document the evolution of data sourcing.

| File | Original Purpose | Superseded By | When |
|------|-----------------|---------------|------|
| `scripts/download_noaa.py` | NOAA CDO daily weather data (requires API token) | `scripts/download_open_meteo.py` (free, no API key) | v0.4.x |
| `scripts/test_nestgate_providers.py` | NestGate provider integration smoke test | NestGate IPC via `data/provider.rs` (Songbird transport) | v0.8.5 |
| `scripts/ncbi_16s_search.py` | NCBI BioProject search for tillage 16S studies | `whitePaper/baseCamp/ncbi_16s_coupling.md` pipeline design + `ncbi_16s_fetch_sample.py` | v0.6.x |
| `scripts/bench_compare.py` | Rust vs Python speedup comparison (parsed `items/s` format) | `bench_cpu_vs_python` binary (end-to-end, Markdown tables, 24 algorithms) | v0.10.0 |
| `scripts/bench_python_baselines.py` | Standalone Python baseline throughput runs | `control/bench_python_timing.py` invoked by `bench_cpu_vs_python` | v0.10.0 |
| `scripts/bench_comparison.json` | Cached comparison results from old pipeline | Live output from `cargo run --release --bin bench_cpu_vs_python` | v0.10.0 |
| `scripts/bench_python_results.json` | Cached Python throughput results | Live output from `control/bench_python_timing.py` | v0.10.0 |
