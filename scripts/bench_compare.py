#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate Rust vs Python speedup comparison from benchmark results.

Usage:
    1. cargo run --release --bin bench_cpu_vs_python  (captures Rust numbers)
    2. python3 scripts/bench_python_baselines.py      (captures Python numbers)
    3. python3 scripts/bench_compare.py               (generates report)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_python_results():
    path = ROOT / "scripts" / "bench_python_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run bench_python_baselines.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def run_rust_benchmark():
    """Run Rust benchmark and parse throughput from stdout."""
    print("Running Rust benchmark (--release)...\n")
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "bench_cpu_vs_python"],
        capture_output=True, text=True, cwd=ROOT / "barracuda",
    )
    if result.returncode != 0:
        print(f"ERROR: Rust benchmark failed:\n{result.stderr}")
        sys.exit(1)

    rust_results = {}
    for line in result.stdout.splitlines():
        m = re.match(
            r"\s+(.+?)\s+(\d+)\s+items\s+\S+/iter\s+([\d.]+)\s+items/s",
            line,
        )
        if m:
            label = m.group(1).strip()
            throughput = float(m.group(3))
            rust_results[label] = throughput
    return rust_results


def match_label(py_label, rust_labels):
    """Find the best Rust match for a Python benchmark label."""
    if py_label in rust_labels:
        return py_label
    normalized = py_label.replace("(scalar Python)", "").strip()
    for rl in rust_labels:
        if normalized in rl or rl in normalized:
            return rl
    base = re.sub(r"\(.*?\)", "", py_label).strip()
    for rl in rust_labels:
        rl_base = re.sub(r"\(.*?\)", "", rl).strip()
        if base == rl_base:
            n_py = re.search(r"\((\d+)", py_label)
            n_rs = re.search(r"\((\d+)", rl)
            if n_py and n_rs and n_py.group(1) == n_rs.group(1):
                return rl
    return None


def main():
    py_results = load_python_results()
    rust_map = run_rust_benchmark()

    print("═══════════════════════════════════════════════════════════════════════════")
    print("  airSpring Benchmark Report — Rust CPU vs Python (interpreted)")
    print("═══════════════════════════════════════════════════════════════════════════\n")
    print(f"  {'Computation':<44} {'Python':>12} {'Rust':>12} {'Speedup':>10}")
    print(f"  {'':─<44} {'':─>12} {'':─>12} {'':─>10}")

    comparisons = []
    for pr in py_results:
        label = pr["label"]
        py_tp = pr["throughput"]
        rl = match_label(label, rust_map)
        if rl:
            rs_tp = rust_map[rl]
            speedup = rs_tp / py_tp if py_tp > 0 else float("inf")
            comparisons.append({
                "label": label,
                "python_throughput": py_tp,
                "rust_throughput": rs_tp,
                "speedup": speedup,
            })
            print(
                f"  {label:<44} {py_tp:>10.0f}/s {rs_tp:>10.0f}/s {speedup:>8.0f}x"
            )
        else:
            print(f"  {label:<44} {py_tp:>10.0f}/s {'—':>12} {'—':>10}")

    if comparisons:
        speedups = [c["speedup"] for c in comparisons]
        geo_mean = 1.0
        for s in speedups:
            geo_mean *= s
        geo_mean = geo_mean ** (1.0 / len(speedups))
        min_s = min(speedups)
        max_s = max(speedups)
        print(f"\n  {'Geometric mean speedup':<44} {'':>12} {'':>12} {geo_mean:>8.0f}x")
        print(f"  {'Range':<44} {'':>12} {'':>12} {min_s:.0f}x – {max_s:.0f}x")

    print("\n  Notes:")
    print("  - Rust compiled with --release (LTO, native CPU)")
    print("  - Python 3.x CPython scalar loops (no numpy vectorization)")
    print("  - Same algorithms, same precision (f64)")
    print("  - Richards uses scipy.integrate.solve_ivp (Python) vs hand-coded Euler (Rust)")
    print()

    report_path = ROOT / "scripts" / "bench_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"Full comparison saved to {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
