# Paper Notebook Pattern — airSpring

Publishable-grade Jupyter notebooks reproducing peer-reviewed science.
Each notebook validates one paper's methods against frozen benchmark data,
with equations, visualizations, and provenance for the full lineage chain.

## Evolution Path

```
Tier 1 (static)  — Load frozen benchmark JSON, display results
Tier 2 (live)    — Execute Python compute cells, generate dynamically
Tier 3 (primal)  — Call science.* via JSON-RPC, compare Python vs Rust vs GPU
```

## Cell Structure

Every notebook follows this sequence:

1. **Title + Citation** (markdown)
   - Paper title, authors, journal, year, DOI/URL
   - Experiment number (e.g., "airSpring Experiment 001")
   - One-paragraph abstract: what we validate and why
   - "For other springs" adaptation note

2. **Theory** (markdown)
   - Key equations in LaTeX
   - Physical context and parameter definitions
   - Reference to specific paper sections (e.g., "FAO-56 Eq. 6")

3. **Setup** (code)
   - Imports: `json`, `math`, `numpy`, `matplotlib`, `pathlib`
   - Load benchmark JSON from `../../control/<topic>/benchmark_*.json`
   - Print provenance metadata

4. **Implementation** (code cells, one per logical block)
   - Python functions from `control/<topic>/<script>.py`
   - Inline equation references in docstrings
   - Exact numerical logic preserved (outputs must match benchmark)

5. **Validation** (code)
   - Compute values using the implementation
   - Compare to benchmark expected values
   - Print pass/fail with named tolerances
   - Summary table of all checks

6. **Visualization** (code)
   - matplotlib charts: computed vs expected, error distribution
   - Color palette: `#2ecc71` (pass), `#e74c3c` (fail), `#3498db` (info)
   - Chart titles include key numbers

7. **Provenance + Summary** (markdown)
   - Results table (metric, expected, computed, tolerance, status)
   - Links: Rust binary (`validate_*`), primal capability (`science.*`)
   - Benchmark JSON path and commit
   - "Future: Tier 2" note about primal IPC wiring
   - License: AGPL-3.0-or-later

## Data Loading Pattern

```python
import json
from pathlib import Path

CONTROL = Path('..') / '..' / 'control'

def load_benchmark(topic, filename):
    with open(CONTROL / topic / filename) as f:
        return json.load(f)

data = load_benchmark('fao56', 'benchmark_fao56.json')
provenance = data.get('_provenance', {})
```

## Visualization Standards

- Use `matplotlib` (renders to static PNG, works in CI)
- Color palette: `#2ecc71` (pass/ok), `#e74c3c` (fail), `#3498db` (info)
- Always include chart titles with key numbers
- Save figures to `/tmp/airspring_paper_<NNN>_<chart>.png`

## Naming Convention

```
NNN-short-descriptive-name.ipynb
```

where NNN is the experiment number (zero-padded to 3 digits).

## CI Execution

All cells must execute cleanly:

```bash
jupyter nbconvert --execute --to notebook notebooks/papers/*.ipynb
```
