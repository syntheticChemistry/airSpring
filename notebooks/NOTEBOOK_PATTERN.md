# Public Notebook Pattern — airSpring

How to create public-facing notebooks for airSpring. Adapted from the
primalSpring/wetSpring exemplar pattern.

## Directory Convention

```
airSpring/
  notebooks/
    NOTEBOOK_PATTERN.md          <- this file
    01-composition-validation.ipynb   <- primal composition & capability validation
    02-benchmark-comparison.ipynb     <- Python vs Rust vs GPU performance
    03-ecosystem-evidence.ipynb       <- 87 experiments, tolerances, provenance
    04-cross-spring-connections.ipynb <- barraCuda integration, shader evolution
    05-domain-deep-dive.ipynb         <- Michigan Atlas, seasonal pipeline, Penny vision
```

## Cell Structure

Every notebook follows the same structure:

1. **Title cell** (markdown): Title, one-paragraph context, data sources, "for other springs" adaptation note
2. **Imports + data loading** (code): Load from `../experiments/results/*.json`
3. **Domain-specific cells** (code + markdown): Visualization and analysis
4. **Summary cell** (markdown): Validation table, provenance note, links to primals.eco

## Data Loading Pattern

```python
import json
from pathlib import Path

RESULTS = Path('..') / 'experiments' / 'results'

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

data = load('composition_validation.json')
```

Notebooks load **frozen data** (committed JSON artifacts), not live API responses.
This means they work without primals running.

## Frozen Data for airSpring

| File | Contents |
|------|----------|
| `composition_validation.json` | 44 capabilities, deploy graphs, primal composition, gaps |
| `test_suite_report.json` | Module-level test counts (1,364 total), coverage, quality gates |
| `experiment_catalog.json` | All 87 experiments categorized by focus area |
| `security_convergence.json` | Safety lints, cargo-deny, IPC security, CI gates |
| `cross_spring_matrix.json` | barraCuda integration, shader families, primal consumption |
| `benchmark_timing.json` | 24-algorithm Rust vs Python timing, GPU tiers, atlas scale |

## Visualization Standards

- Use `matplotlib` (available everywhere, renders to static PNG)
- Color palette: `#2ecc71` (pass/ok), `#e74c3c` (fail), `#3498db` (info)
- Always include chart titles with key numbers

## Adapting for Your Spring

1. Copy this directory structure
2. Replace data paths with your `experiments/results/` JSONs
3. Update the narrative for your domain
4. Keep the cell structure (title -> load -> analyze -> summary)
5. All cells must execute cleanly in CI (`jupyter nbconvert --execute`)
