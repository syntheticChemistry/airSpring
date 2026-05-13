#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# lithoSpore fetch script for LTEE E3 — FLS2 Plant Immunity
#
# Generates the reference data files from the Python baseline.
# All data is synthetic (deterministic RNG seed 20250511), so no
# external download is needed — the baseline script produces both
# expected_values.json and benchmark_ltee_fls2.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== LTEE E3 fetch: generating reference data ==="

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 required" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/ltee_fls2_plant_immunity.py"

for f in expected_values.json benchmark_ltee_fls2.json; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "ERROR: ${f} not generated" >&2
        exit 1
    fi
done

if command -v b3sum &>/dev/null; then
    echo "--- BLAKE3 hashes ---"
    b3sum "${SCRIPT_DIR}/expected_values.json"
    b3sum "${SCRIPT_DIR}/benchmark_ltee_fls2.json"
elif command -v sha256sum &>/dev/null; then
    echo "--- SHA-256 hashes (b3sum not installed) ---"
    sha256sum "${SCRIPT_DIR}/expected_values.json"
    sha256sum "${SCRIPT_DIR}/benchmark_ltee_fls2.json"
fi

echo "=== LTEE E3 fetch: DONE ==="
