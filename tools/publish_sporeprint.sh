#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# SP-4: Sovereign publish pipeline — push sporePrint content to NestGate
# via content.put (bearDog-signed session, BTSP-encrypted).
#
# Adapted from primalSpring/tools/publish_sporeprint.sh (Wave 37).
#
# Usage:
#   ./tools/publish_sporeprint.sh                    # publish all summaries
#   ./tools/publish_sporeprint.sh --dry-run          # show what would be published
#   ./tools/publish_sporeprint.sh validation-summary  # publish one file
#
# Requires:
#   - NestGate running on UDS (default: /tmp/nestgate.sock)
#   - bearDog session with content.* scope (Wave 108+)
#   - b3sum for BLAKE3 hashing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOREPRINT_DIR="${SPOREPRINT_DIR:-$ROOT/sporeprint}"
NESTGATE_SOCK="${NESTGATE_SOCK:-/tmp/nestgate.sock}"
DRY_RUN=0
TARGET=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [file-stem]"
            echo ""
            echo "Publishes sporeprint/ content to NestGate via content.put."
            echo "If file-stem is given, publishes only that file."
            exit 0
            ;;
        *) TARGET="$arg" ;;
    esac
done

rpc_call() {
    local method="$1"
    local params="$2"
    local payload
    payload=$(printf '{"jsonrpc":"2.0","method":"%s","params":%s,"id":1}' "$method" "$params")

    if [ ! -S "$NESTGATE_SOCK" ]; then
        echo "ERROR: NestGate socket not found at $NESTGATE_SOCK" >&2
        echo "  Start NestGate or set NESTGATE_SOCK" >&2
        return 1
    fi

    echo "$payload" | socat - UNIX-CONNECT:"$NESTGATE_SOCK" 2>/dev/null
}

publish_file() {
    local filepath="$1"
    local filename
    filename=$(basename "$filepath")
    local content_type="text/markdown"

    if [[ "$filename" == *.json ]]; then
        content_type="application/json"
    fi

    local content
    content=$(base64 -w0 < "$filepath")
    local blake3
    blake3=$(b3sum --no-names "$filepath")
    local size
    size=$(stat -c%s "$filepath")

    echo "  Publishing: $filename ($size bytes, $content_type)"
    echo "  Hash: $blake3"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] Would call content.put"
        return 0
    fi

    local params
    params=$(printf '{"path":"sporeprint/airspring/%s","content_base64":"%s","content_type":"%s","blake3":"%s","metadata":{"source":"airspring","pipeline":"SP-4","version":"%s"}}' \
        "$filename" "$content" "$content_type" "$blake3" "0.10.0")

    local result
    result=$(rpc_call "content.put" "$params" 2>&1) || {
        echo "  WARNING: content.put failed (NestGate may not be running)" >&2
        echo "  Result: $result" >&2
        return 1
    }

    if echo "$result" | grep -q '"error"'; then
        echo "  WARNING: content.put returned error" >&2
        echo "  $result" >&2
        return 1
    fi

    echo "  OK: $filename published"
}

echo "━━━ SP-4: airSpring Sovereign Publish Pipeline ━━━"
echo "Source: $SPOREPRINT_DIR"
echo "Target: $NESTGATE_SOCK"
[ "$DRY_RUN" -eq 1 ] && echo "Mode: DRY RUN"
echo ""

if [ ! -d "$SPOREPRINT_DIR" ]; then
    echo "ERROR: sporeprint directory not found at $SPOREPRINT_DIR" >&2
    exit 1
fi

published=0
failed=0

if [ -n "$TARGET" ]; then
    found=$(find "$SPOREPRINT_DIR" -name "${TARGET}*" -type f 2>/dev/null | head -1)
    if [ -z "$found" ]; then
        echo "ERROR: No file matching '$TARGET' in $SPOREPRINT_DIR" >&2
        exit 1
    fi
    if publish_file "$found"; then
        published=$((published + 1))
    else
        failed=$((failed + 1))
    fi
else
    for f in "$SPOREPRINT_DIR"/*.md "$SPOREPRINT_DIR"/*.json; do
        [ -f "$f" ] || continue
        if publish_file "$f"; then
            published=$((published + 1))
        else
            failed=$((failed + 1))
        fi
    done
fi

echo ""
echo "━━━ Summary ━━━"
echo "Published: $published"
echo "Failed:    $failed"

if [ "$failed" -gt 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    echo ""
    echo "Some files failed to publish. Ensure NestGate is running"
    echo "with a bearDog session that includes content.* scope."
    exit 1
fi
