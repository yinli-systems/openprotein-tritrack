"""Verified Swiss-Prot flat-file parsing and annotation indexing."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, TextIO

from openprotein.provenance import sha256_file


_EVIDENCE = re.compile(r"ECO:\d+")
_EVIDENCE_BLOCK = re.compile(r"\s*\{ECO:[^}]+\}")
_SEQUENCE = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


class _DigestingRawReader(io.RawIOBase):
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()  # noqa: S324 - upstream publishes an MD5 receipt
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        chunk = self.handle.read(len(buffer))
        size = len(chunk)
        if size:
            buffer[:size] = chunk
            self.sha256.update(chunk)
            self.md5.update(chunk)
            self.bytes_read += size
        return size


def _clean_annotation(parts: List[str]) -> str:
    text = " ".join(part.strip() for part in parts if part.strip())
    text = _EVIDENCE_BLOCK.sub("", text)
    return " ".join(text.split())


def iter_swissprot(handle: TextIO) -> Iterator[Dict[str, Any]]:
    """Yield canonical sequence and curated text from UniProt flat records."""

    entry: Dict[str, Any] = {}
    descriptions: List[str] = []
    organisms: List[str] = []
    comments: List[str] = []
    keywords: List[str] = []
    go_terms: List[str] = []
    evidence_codes: set[str] = set()
    sequence_parts: List[str] = []
    in_sequence = False

    def reset() -> None:
        nonlocal entry, descriptions, organisms, comments, keywords, go_terms
        nonlocal evidence_codes, sequence_parts, in_sequence
        entry = {}
        descriptions = []
        organisms = []
        comments = []
        keywords = []
        go_terms = []
        evidence_codes = set()
        sequence_parts = []
        in_sequence = False

    def finish() -> Optional[Dict[str, Any]]:
        if not entry:
            return None
        required = {"entry_name", "accession"}
        missing = required - set(entry)
        if missing:
            raise ValueError(f"Swiss-Prot entry is missing fields: {sorted(missing)}")
        sequence = "".join(sequence_parts).upper()
        if not sequence or not set(sequence) <= _SEQUENCE:
            raise ValueError(f"invalid canonical sequence for {entry['entry_name']}")
        sections = []
        description = _clean_annotation(descriptions)
        organism = _clean_annotation(organisms)
        comment = _clean_annotation(comments)
        keyword = _clean_annotation(keywords)
        if description:
            sections.append(f"Protein: {description}")
        if organism:
            sections.append(f"Organism: {organism}")
        if comment:
            sections.append(f"Annotation: {comment}")
        if go_terms:
            sections.append(f"Gene Ontology: {'; '.join(go_terms)}")
        if keyword:
            sections.append(f"Keywords: {keyword}")
        text = " ".join(sections)
        return {
            "entry_name": entry["entry_name"],
            "accession": entry["accession"],
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "evidence_codes": sorted(evidence_codes),
        }

    reset()
    for line_number, line in enumerate(handle, start=1):
        if line.startswith("//"):
            record = finish()
            if record is not None:
                yield record
            reset()
            continue
        if in_sequence:
            sequence_parts.append("".join(character for character in line if character.isalpha()))
            continue
        tag = line[:2]
        value = line[5:].rstrip() if len(line) >= 5 else ""
        evidence_codes.update(_EVIDENCE.findall(value))
        if tag == "ID":
            fields = value.split()
            if not fields:
                raise ValueError(f"empty ID line at {line_number}")
            entry["entry_name"] = fields[0]
        elif tag == "AC" and "accession" not in entry:
            accession = value.split(";", 1)[0].strip()
            if accession:
                entry["accession"] = accession
        elif tag == "DE":
            descriptions.append(value)
        elif tag == "OS":
            organisms.append(value)
        elif tag == "CC":
            if not (
                set(value) <= {"-"}
                or value.startswith("Copyrighted by")
                or value.startswith("Distributed under")
            ):
                comments.append(value)
        elif tag == "KW":
            keywords.append(value)
        elif tag == "DR" and value.startswith("GO;"):
            fields = [field.strip() for field in value.split(";")]
            if len(fields) >= 3:
                go_terms.append(fields[2])
        elif tag == "SQ":
            in_sequence = True
    if entry:
        raise ValueError("Swiss-Prot input ended before the final // terminator")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_swissprot_index(
    source_dat_gz: str,
    output_database: str,
    output_manifest: str,
    source_definition_path: str,
) -> Dict[str, Any]:
    """Build an atomic SQLite annotation index and verify the source in-stream."""

    source_path = Path(source_dat_gz).resolve()
    database_path = Path(output_database).resolve()
    manifest_path = Path(output_manifest).resolve()
    if database_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing Swiss-Prot artifact")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_definition = json.loads(Path(source_definition_path).read_text(encoding="utf-8"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.", suffix=".tmp", dir=database_path.parent
    )
    os.close(descriptor)
    temporary_database = Path(temporary_name)
    input_reader: Optional[_DigestingRawReader] = None
    records = 0
    residues = 0
    records_with_text = 0

    try:
        connection = sqlite3.connect(temporary_database)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA locking_mode=EXCLUSIVE;
                CREATE TABLE entries (
                    entry_name TEXT PRIMARY KEY,
                    accession TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    sequence_sha256 TEXT NOT NULL,
                    annotation_text TEXT NOT NULL,
                    annotation_sha256 TEXT NOT NULL,
                    evidence_codes_json TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE UNIQUE INDEX entries_accession ON entries(accession);
                """
            )
            with source_path.open("rb") as binary:
                input_reader = _DigestingRawReader(binary)
                buffered = io.BufferedReader(input_reader, buffer_size=1024 * 1024)
                with gzip.GzipFile(fileobj=buffered, mode="rb") as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        for record in iter_swissprot(text):
                            connection.execute(
                                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    record["entry_name"],
                                    record["accession"],
                                    record["sequence"],
                                    record["sequence_sha256"],
                                    record["text"],
                                    record["text_sha256"],
                                    json.dumps(record["evidence_codes"], separators=(",", ":")),
                                ),
                            )
                            records += 1
                            residues += len(record["sequence"])
                            records_with_text += bool(record["text"])
                            if records % 10_000 == 0:
                                connection.commit()
                if input_reader.bytes_read < source_path.stat().st_size:
                    while buffered.read(1024 * 1024):
                        pass
            connection.commit()
        finally:
            connection.close()

        observed_sha256 = input_reader.sha256.hexdigest()
        observed_md5 = input_reader.md5.hexdigest()
        if input_reader.bytes_read != int(source_definition["dat_bytes"]):
            raise RuntimeError("Swiss-Prot source byte count does not match the release receipt")
        if observed_md5 != source_definition["dat_md5"]:
            raise RuntimeError("Swiss-Prot source MD5 does not match the release receipt")
        expected_sha256 = source_definition.get("dat_sha256")
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise RuntimeError("Swiss-Prot source SHA-256 does not match the retrieval receipt")
        if records != int(source_definition["records"]):
            raise RuntimeError(f"Swiss-Prot record count mismatch: {records}")

        os.replace(temporary_database, database_path)
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_definition,
            "source_input": {
                "path": str(source_path),
                "bytes": input_reader.bytes_read,
                "md5": observed_md5,
                "sha256": observed_sha256,
            },
            "database": {
                "path": str(database_path),
                "bytes": database_path.stat().st_size,
                "sha256": sha256_file(database_path),
                "records": records,
                "records_with_text": records_with_text,
                "residues": residues,
            },
            "join_policy": "join only to exact UniRef representative sequences; inherit UniRef split",
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    except Exception:
        temporary_database.unlink(missing_ok=True)
        raise
