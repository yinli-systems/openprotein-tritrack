from pathlib import Path

import pytest

from openprotein.evaluation_subset import (
    load_record_id_allowlist,
    verify_selected_record_ids,
)


def test_record_id_allowlist_binds_bytes_and_accepts_different_evaluation_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    expected, record_ids, receipt = load_record_id_allowlist(str(path))
    selected = verify_selected_record_ids(["b", "a", "c"], expected)
    assert record_ids == {"a", "b", "c"}
    assert receipt["records"] == 3
    assert selected["set_match"] is True


def test_record_id_allowlist_rejects_duplicates_and_missing_rows(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_record_id_allowlist(str(path))
    with pytest.raises(RuntimeError, match="exactly match"):
        verify_selected_record_ids(["a"], ["a", "b"])
