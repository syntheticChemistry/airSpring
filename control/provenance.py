# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark JSON provenance: content SHA-256 for drift detection (ludoSpring V29 pattern).

Hash covers JSON-serialized benchmark payload excluding ``_provenance``, so the digest
is stable while metadata can list generation time and interpreter version.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


def content_sha256(data: dict[str, Any]) -> str:
    """SHA-256 of benchmark content (excluding _provenance) for drift detection."""
    content = {k: v for k, v in data.items() if k != "_provenance"}
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def attach_provenance(
    data: dict[str, Any],
    merge: Mapping[str, Any] | None = None,
) -> None:
    """Set ``data['_provenance']`` with ``content_sha256``, ``python_version``, ``generation_date``.

    Removes any existing ``_provenance`` before hashing. If ``merge`` is omitted, prior
    ``_provenance`` is preserved when it is a dict; non-dict values are stored under
    ``legacy_provenance``.
    """
    existing = data.pop("_provenance", None)
    if merge is None:
        if isinstance(existing, dict):
            merge = dict(existing)
        elif existing is not None:
            merge = {"legacy_provenance": existing}
        else:
            merge = {}
    else:
        merge = dict(merge)
    sha = content_sha256(data)
    data["_provenance"] = {
        **merge,
        "content_sha256": sha,
        "python_version": sys.version,
        "generation_date": datetime.now(timezone.utc).isoformat(),
    }
