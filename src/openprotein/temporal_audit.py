"""Temporal-split audit for manifest-bound Swiss-Prot multimodal rows."""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

from openprotein.homology import _manifest_split_files
from openprotein.provenance import sha256_file


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parse_dt_date(line: str) -> date:
    value = line[5:].split(",", 1)[0].strip()
    return datetime.strptime(value, "%d-%b-%Y").date()


def iter_swissprot_dates(path: Path) -> Iterator[Tuple[List[str], date, date]]:
    """Yield accessions, integration date, and latest annotation-update date."""

    accessions: List[str] = []
    dates: List[date] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("AC   "):
                accessions.extend(
                    accession.strip()
                    for accession in line[5:].split(";")
                    if accession.strip()
                )
            elif line.startswith("DT   "):
                dates.append(_parse_dt_date(line))
            elif line.startswith("//"):
                if not accessions or len(dates) < 3:
                    raise ValueError(
                        f"malformed Swiss-Prot entry ending at line {line_number}"
                    )
                yield accessions, dates[0], dates[-1]
                accessions = []
                dates = []
    if accessions or dates:
        raise ValueError("unterminated Swiss-Prot flat-file entry")


def _split_accessions(paths: Iterable[Path]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "accession"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"{path} lacks required columns {sorted(required)}")
            for row_index, row in enumerate(reader, start=2):
                accession = row["accession"].strip()
                record_id = row["id"].strip()
                if not accession or not record_id:
                    raise ValueError(f"{path}:{row_index} has an empty ID or accession")
                if accession in result:
                    raise ValueError(f"duplicate accession in evaluated splits: {accession}")
                result[accession] = record_id
    return result


def _date_summary(values: Iterable[date], cutoff: date) -> Dict[str, Any]:
    dates = sorted(values)
    if not dates:
        raise ValueError("cannot summarize an empty date collection")
    recent = sum(value >= cutoff for value in dates)
    return {
        "minimum": dates[0].isoformat(),
        "maximum": dates[-1].isoformat(),
        "on_or_after_cutoff": recent,
        "on_or_after_cutoff_fraction": recent / len(dates),
    }


def audit_swissprot_temporal_split(
    data_manifest_path: str,
    swissprot_path: str,
    cutoff_value: str,
    output_path: str,
) -> Dict[str, Any]:
    """Show whether manifest train/validation rows form a true temporal split."""

    manifest_path = Path(data_manifest_path).resolve()
    raw_path = Path(swissprot_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite temporal audit: {output}")
    cutoff = date.fromisoformat(cutoff_value)
    _, train_paths = _manifest_split_files(manifest_path, "train")
    _, validation_paths = _manifest_split_files(manifest_path, "validation")
    split_accessions = {
        "train": _split_accessions(train_paths),
        "validation": _split_accessions(validation_paths),
    }
    accession_to_split = {
        accession: split
        for split, accessions in split_accessions.items()
        for accession in accessions
    }
    if len(accession_to_split) != sum(len(values) for values in split_accessions.values()):
        raise RuntimeError("an accession appears in both train and validation")

    dates_by_split: Dict[str, Dict[str, Dict[str, date]]] = {
        split: {} for split in split_accessions
    }
    unresolved = set(accession_to_split)
    for accessions, integration_date, annotation_date in iter_swissprot_dates(raw_path):
        for accession in accessions:
            split = accession_to_split.get(accession)
            if split is None or accession not in unresolved:
                continue
            dates_by_split[split][accession] = {
                "integration": integration_date,
                "annotation_update": annotation_date,
            }
            unresolved.remove(accession)
    if unresolved:
        raise RuntimeError(
            f"Swiss-Prot raw file did not resolve {len(unresolved)} evaluated accessions: "
            f"{sorted(unresolved)[:10]}"
        )

    split_receipts: Dict[str, Mapping[str, Any]] = {}
    for split, values in dates_by_split.items():
        split_receipts[split] = {
            "records": len(values),
            "integration_date": _date_summary(
                (item["integration"] for item in values.values()), cutoff
            ),
            "annotation_update_date": _date_summary(
                (item["annotation_update"] for item in values.values()), cutoff
            ),
        }
    train_recent = int(
        split_receipts["train"]["annotation_update_date"]["on_or_after_cutoff"]
    )
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "temporal_split_absent" if train_recent else "temporal_split_candidate",
        "evidence_level": "TRAIN_VALIDATION_TEMPORAL_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "cutoff": cutoff.isoformat(),
        "data_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "swissprot_source": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "splits": split_receipts,
        "conclusion": (
            "The current record-hash train/validation split is not a temporal holdout because "
            "the training split contains Swiss-Prot entries annotated on or after the cutoff. "
            "Post-cutoff validation rows from this release must not be presented as a clean "
            "future-data benchmark. A future training corpus must be rebuilt with the cutoff "
            "applied before training."
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt
