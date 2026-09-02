"""Fail-closed record-ID subsets for validation evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from openprotein.provenance import sha256_file


def load_record_id_allowlist(path: str) -> Tuple[List[str], Set[str], Dict[str, Any]]:
    """Load a nonempty, unique ID allowlist and bind its exact bytes."""

    allowlist_path = Path(path).resolve()
    if not allowlist_path.is_file():
        raise FileNotFoundError(f"record-ID allowlist is missing: {allowlist_path}")
    record_ids = [line.strip() for line in allowlist_path.read_text(encoding="utf-8").splitlines()]
    if not record_ids or any(not record_id for record_id in record_ids):
        raise ValueError("record-ID allowlist is empty or contains blank IDs")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record-ID allowlist contains duplicate IDs")
    return record_ids, set(record_ids), {
        "path": str(allowlist_path),
        "bytes": allowlist_path.stat().st_size,
        "sha256": sha256_file(allowlist_path),
        "records": len(record_ids),
    }


def verify_selected_record_ids(
    selected_ids: Iterable[str],
    expected_ids: List[str],
) -> Dict[str, Any]:
    """Require every allowlisted ID exactly once and bind evaluation order."""

    selected = list(selected_ids)
    if len(set(selected)) != len(selected):
        raise RuntimeError("evaluation selected duplicate record IDs")
    expected_set = set(expected_ids)
    selected_set = set(selected)
    missing = sorted(expected_set - selected_set)
    unexpected = sorted(selected_set - expected_set)
    if missing or unexpected or len(selected) != len(expected_ids):
        raise RuntimeError(
            "evaluation rows do not exactly match the record-ID allowlist: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"selected={len(selected)}, expected={len(expected_ids)}"
        )
    payload = "\n".join(selected).encode("utf-8")
    return {
        "records": len(selected),
        "evaluation_order_sha256": hashlib.sha256(payload).hexdigest(),
        "set_match": True,
    }
