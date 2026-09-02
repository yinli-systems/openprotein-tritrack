"""Audit the official TorchProtein EC archive with a pinned RDKit parser."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import string
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from openprotein.ec_benchmark import (
    _copy_member,
    _md5_file,
    _member_by_suffix,
    _parse_test_cutoffs,
)
from openprotein.provenance import sha256_file

RecordParser = Callable[[bytes], Optional[str]]  # noqa: UP045
ProgressCallback = Callable[[str, int, int, int], None]


def _load_rdkit_parser() -> tuple[RecordParser, str]:
    try:
        import rdkit
        from rdkit import Chem
    except ImportError as error:
        raise RuntimeError("RDKit is required for the EC parser audit") from error

    supported_metadata_codes = set(" " + string.ascii_letters + string.digits)

    def parse(payload: bytes) -> str | None:
        molecule = Chem.MolFromPDBBlock(
            payload.decode("latin-1"),
            sanitize=True,
            removeHs=True,
            flavor=0,
            proximityBonding=True,
        )
        if molecule is None:
            return "rdkit_molecule_parse_failed"

        # TorchDrug 0.2.1 performs these checks in Protein.from_molecule and
        # skips the sample when that method returns None.
        last_residue = None
        for atom in molecule.GetAtoms():
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info is None:
                return "torchdrug_missing_pdb_residue_info"
            canonical_residue = (
                pdb_info.GetResidueNumber(),
                pdb_info.GetInsertionCode(),
                pdb_info.GetResidueName().strip(),
            )
            if canonical_residue == last_residue:
                continue
            last_residue = canonical_residue
            if pdb_info.GetInsertionCode() not in supported_metadata_codes:
                return "torchdrug_unsupported_insertion_code"
            if pdb_info.GetChainId() not in supported_metadata_codes:
                return "torchdrug_unsupported_chain_id"
        return None

    return parse, rdkit.__version__


def _iter_pdb_members(nested_zip: Path, split: str):
    with zipfile.ZipFile(nested_zip) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{split}/") and name.endswith(".pdb")
        )
        if not members:
            raise RuntimeError(f"EC nested ZIP contains no {split} PDB members")
        for member in members:
            record_id = Path(member).name.split("_", 1)[0]
            yield record_id, archive.read(member)


def audit_ec_rdkit_cohort(
    source_zip_path: str,
    source_definition_path: str,
    output_directory: str,
    expected_rdkit_version: str | None = None,
    record_parser: RecordParser | None = None,
    rdkit_version_override: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Publish parse-valid IDs without changing or hiding the raw benchmark view."""

    source = Path(source_zip_path).resolve()
    definition_path = Path(source_definition_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EC RDKit audit: {output}")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if source.stat().st_size != int(definition["bytes"]):
        raise RuntimeError("EC source byte count differs from its definition")
    observed_md5 = _md5_file(source)
    if observed_md5 != definition["md5"]:
        raise RuntimeError("EC source MD5 differs from its definition")
    observed_sha256 = sha256_file(source)
    if definition.get("sha256") not in {None, observed_sha256}:
        raise RuntimeError("EC source SHA-256 differs from its definition")

    if record_parser is None:
        parser, rdkit_version = _load_rdkit_parser()
    else:
        parser = record_parser
        rdkit_version = rdkit_version_override or "test-double"
    if expected_rdkit_version is not None and rdkit_version != expected_rdkit_version:
        raise RuntimeError(
            f"RDKit version mismatch: expected {expected_rdkit_version}, got {rdkit_version}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    nested_paths: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(source) as outer:
            cutoff_member = _member_by_suffix(outer, "EnzymeCommission/nrPDB-EC_test.csv")
            test_cutoffs = _parse_test_cutoffs(outer.read(cutoff_member).decode("utf-8"))
            for split in ("train", "valid", "test"):
                member = _member_by_suffix(outer, f"EnzymeCommission/{split}.zip")
                nested = staging / f"{split}.zip"
                _copy_member(outer, member, nested)
                nested_paths[split] = nested

        valid_path = staging / "rdkit-valid-records.csv"
        fields = [
            "id",
            "split",
            "below_30_percent",
            "below_40_percent",
            "below_50_percent",
            "below_70_percent",
            "below_95_percent",
        ]
        records: dict[str, dict[str, int]] = {}
        invalid_ids: dict[str, list[str]] = {}
        rejection_ids: dict[str, dict[str, list[str]]] = {}
        valid_test_ids = set()
        with valid_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            seen_ids = set()
            for split in ("train", "valid", "test"):
                total = 0
                valid = 0
                invalid_ids[split] = []
                rejection_ids[split] = {}
                for record_id, payload in _iter_pdb_members(nested_paths[split], split):
                    if record_id in seen_ids:
                        raise RuntimeError(f"EC benchmark ID crosses splits: {record_id}")
                    seen_ids.add(record_id)
                    total += 1
                    rejection_reason = parser(payload)
                    if rejection_reason is not None:
                        invalid_ids[split].append(record_id)
                        rejection_ids[split].setdefault(rejection_reason, []).append(record_id)
                        if progress_callback is not None and total % 1000 == 0:
                            progress_callback(split, total, valid, total - valid)
                        continue
                    valid += 1
                    flags = test_cutoffs.get(record_id)
                    writer.writerow(
                        {
                            "id": record_id,
                            "split": split,
                            **{
                                field: "" if flags is None else flags[index]
                                for index, field in enumerate(fields[2:])
                            },
                        }
                    )
                    if split == "test":
                        valid_test_ids.add(record_id)
                    if progress_callback is not None and total % 1000 == 0:
                        progress_callback(split, total, valid, total - valid)
                records[split] = {
                    "total": total,
                    "valid": valid,
                    "invalid": total - valid,
                }
                if progress_callback is not None:
                    progress_callback(split, total, valid, total - valid)

        for nested in nested_paths.values():
            nested.unlink()
        cutoff_counts = {
            field: sum(
                test_cutoffs[record_id][index] == 1
                for record_id in valid_test_ids & set(test_cutoffs)
            )
            for index, field in enumerate(fields[2:])
        }
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "evidence_level": "PUBLIC_BENCHMARK_RDKIT_PARSE_AUDIT",
            "test_data_accessed": True,
            "proteingym_rows_opened": False,
            "benchmark": "TorchProtein EnzymeCommission",
            "source_definition": {
                "path": str(definition_path),
                "sha256": sha256_file(definition_path),
            },
            "source": {
                **definition,
                "path": str(source),
                "observed_md5": observed_md5,
                "observed_sha256": observed_sha256,
            },
            "parser": {
                "library": "RDKit",
                "version": rdkit_version,
                "entrypoint": (
                    "Chem.MolFromPDBBlock followed by TorchDrug 0.2.1 "
                    "Protein.from_molecule metadata acceptance checks"
                ),
                "arguments": {
                    "sanitize": True,
                    "removeHs": True,
                    "flavor": 0,
                    "proximityBonding": True,
                },
                "python": platform.python_version(),
                "python_executable": sys.executable,
            },
            "records": records,
            "invalid_record_ids": invalid_ids,
            "rejection_ids_by_reason": rejection_ids,
            "test_identity_cutoff_counts": cutoff_counts,
            "identity_cutoff_alignment": {
                "valid_test_records_without_published_cutoff": sorted(
                    valid_test_ids - set(test_cutoffs)
                ),
                "published_cutoff_records_not_parse_valid": sorted(
                    set(test_cutoffs) - valid_test_ids
                ),
            },
            "valid_records": {
                "path": valid_path.name,
                "bytes": valid_path.stat().st_size,
                "sha256": sha256_file(valid_path),
            },
            "comparability_limitations": [
                "Parse validity is RDKit-version-sensitive.",
                (
                    "The TorchDrug acceptance checks are source-equivalent, not an import "
                    "of TorchDrug."
                ),
                "This audit records dataset eligibility only; it does not evaluate a model.",
            ],
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
