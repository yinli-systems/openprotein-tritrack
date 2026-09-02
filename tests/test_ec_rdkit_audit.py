import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from openprotein.ec_rdkit_audit import audit_ec_rdkit_cohort


def _nested(split: str, records: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record_id, pdb in records.items():
            archive.writestr(f"{split}/{record_id}_1.pdb", pdb)
    return payload.getvalue()


def test_rdkit_audit_publishes_valid_ids_and_cutoff_counts(tmp_path: Path) -> None:
    source = tmp_path / "ec.zip"
    cutoffs = (
        "PDB-chain,<30%,<40%,<50%,<70%,<95%\n"
        "TEST-A,1,1,1,1,1\nTEST-B,0,0,1,1,1\nSTALE-A,1,1,1,1,1\n"
    )
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        root = "EnzymeCommission"
        archive.writestr(f"{root}/nrPDB-EC_test.csv", cutoffs)
        archive.writestr(f"{root}/train.zip", _nested("train", {"TRAIN-A": b"ok"}))
        archive.writestr(f"{root}/valid.zip", _nested("valid", {"VALID-A": b"bad"}))
        archive.writestr(
            f"{root}/test.zip",
            _nested("test", {"TEST-A": b"ok", "TEST-B": b"bad", "TEST-C": b"ok"}),
        )
    definition = tmp_path / "source.json"
    definition.write_text(
        json.dumps(
            {
                "bytes": source.stat().st_size,
                "md5": hashlib.md5(source.read_bytes()).hexdigest(),  # noqa: S324
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    receipt = audit_ec_rdkit_cohort(
        str(source),
        str(definition),
        str(output),
        expected_rdkit_version="2022.09.5",
        record_parser=lambda payload: None if payload == b"ok" else "synthetic_rejection",
        rdkit_version_override="2022.09.5",
    )

    assert receipt["records"] == {
        "train": {"total": 1, "valid": 1, "invalid": 0},
        "valid": {"total": 1, "valid": 0, "invalid": 1},
        "test": {"total": 3, "valid": 2, "invalid": 1},
    }
    assert receipt["test_identity_cutoff_counts"] == {
        "below_30_percent": 1,
        "below_40_percent": 1,
        "below_50_percent": 1,
        "below_70_percent": 1,
        "below_95_percent": 1,
    }
    assert receipt["invalid_record_ids"] == {
        "train": [],
        "valid": ["VALID-A"],
        "test": ["TEST-B"],
    }
    assert receipt["rejection_ids_by_reason"] == {
        "train": {},
        "valid": {"synthetic_rejection": ["VALID-A"]},
        "test": {"synthetic_rejection": ["TEST-B"]},
    }
    assert receipt["proteingym_rows_opened"] is False
    with (output / "rdkit-valid-records.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["TRAIN-A", "TEST-A", "TEST-C"]
    assert rows[-1]["below_95_percent"] == ""
