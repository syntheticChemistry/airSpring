#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# BLAKE3 backfill for benchmark JSON provenance (FN-1 / SP-4 alignment).
#
# Adds "blake3" field to _provenance blocks in control/*/benchmark_*.json.
# Idempotent: re-running updates existing blake3 hashes to current content.
#
# Usage: ./tools/blake3_backfill.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

count=0
updated=0

for f in control/*/benchmark_*.json; do
    [[ -f "$f" ]] || continue
    count=$((count + 1))

    hash=$(b3sum --no-names "$f")

    if python3 -c "
import json, sys
with open('$f') as fh:
    data = json.load(fh)
if '_provenance' not in data:
    data['_provenance'] = {}
if data['_provenance'].get('blake3') == '$hash':
    sys.exit(1)
data['_provenance']['blake3'] = '$hash'
with open('$f', 'w') as fh:
    json.dump(data, fh, indent=2)
    fh.write('\n')
" 2>/dev/null; then
        updated=$((updated + 1))
        if $DRY_RUN; then
            echo "[DRY] $f -> $hash"
        else
            echo "[OK]  $f -> $hash"
        fi
    fi
done

echo ""
echo "Scanned $count benchmarks, updated $updated."
