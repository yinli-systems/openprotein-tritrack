"""Streaming AlphaFold structure indexing and sampled geometry targets."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import struct
import tarfile
import tempfile
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import torch

from openprotein.provenance import sha256_file


_AFDB_MEMBER = re.compile(
    r"(?:^|/)AF-([A-Z0-9]+)-F(\d+)-model_v(\d+)\.pdb(?:\.gz)?$", re.IGNORECASE
)
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "PYL": "O",
}
STRUCTURE_FEATURE_DIMENSION = 12


class _DigestingRawReader(io.RawIOBase):
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.sha256 = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        chunk = self.handle.read(len(buffer))
        size = len(chunk)
        if size:
            buffer[:size] = chunk
            self.sha256.update(chunk)
            self.bytes_read += size
        return size


def parse_pdb_ca(handle: BinaryIO) -> Dict[str, Any]:
    """Extract one coordinate and pLDDT/B-factor per unique CA residue."""

    sequence: List[str] = []
    coordinates: List[Tuple[float, float, float]] = []
    confidence: List[float] = []
    seen = set()
    for line in handle:
        if not line.startswith(b"ATOM  ") or len(line) < 66:
            continue
        if line[12:16].strip() != b"CA" or line[16:17] not in {b" ", b"A"}:
            continue
        residue_key = (line[21:22], line[22:26], line[26:27])
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residue = _THREE_TO_ONE.get(line[17:20].decode("ascii"), "X")
        xyz = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        score = float(line[60:66])
        if not all(math.isfinite(value) for value in (*xyz, score)):
            raise ValueError("PDB contains a non-finite coordinate or confidence value")
        sequence.append(residue)
        coordinates.append(xyz)
        confidence.append(score)
    if not sequence:
        raise ValueError("PDB member contains no CA atoms")
    return {"sequence": "".join(sequence), "coordinates": coordinates, "confidence": confidence}


def ca_trace_features(
    coordinates: torch.Tensor,
    confidence: torch.Tensor,
    min_confidence: float = 70.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build pLDDT-gated, rigid-motion-invariant CA-trace features.

    The 12 channels are normalized pLDDT; CA distances at offsets
    ``[-4, -3, -2, -1, +1, +2, +3, +4]``; local bend cosine; and the
    sine/cosine of the CA pseudo-dihedral.  No absolute coordinate frame is
    exposed to the model.
    """

    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [length, 3]")
    if confidence.shape != coordinates.shape[:1]:
        raise ValueError("confidence must have shape [length]")
    length = coordinates.shape[0]
    features = coordinates.new_zeros((length, STRUCTURE_FEATURE_DIMENSION))
    features[:, 0] = confidence.clamp(0.0, 100.0) / 100.0

    offsets = (-4, -3, -2, -1, 1, 2, 3, 4)
    for channel, offset in enumerate(offsets, start=1):
        if abs(offset) >= length:
            continue
        if offset < 0:
            center = slice(-offset, length)
            neighbor = slice(0, length + offset)
        else:
            center = slice(0, length - offset)
            neighbor = slice(offset, length)
        distance = torch.linalg.vector_norm(
            coordinates[center] - coordinates[neighbor], dim=-1
        )
        features[center, channel] = (distance / 20.0).clamp_max(2.0)

    if length >= 3:
        incoming = coordinates[:-2] - coordinates[1:-1]
        outgoing = coordinates[2:] - coordinates[1:-1]
        bend = torch.nn.functional.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        features[1:-1, 9] = bend

    if length >= 4:
        a, b, c, d = (
            coordinates[:-3],
            coordinates[1:-2],
            coordinates[2:-1],
            coordinates[3:],
        )
        b0 = -(b - a)
        b1 = c - b
        b2 = d - c
        b1_unit = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-6)
        v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
        w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
        x = (v * w).sum(dim=-1)
        y = (torch.linalg.cross(b1_unit, v, dim=-1) * w).sum(dim=-1)
        denominator = torch.sqrt(x.square() + y.square()).clamp_min(1e-6)
        features[1:-2, 10] = y / denominator
        features[1:-2, 11] = x / denominator

    valid = confidence.ge(min_confidence)
    if length >= 3:
        valid = valid & torch.cat(
            [
                torch.zeros(1, dtype=torch.bool, device=valid.device),
                confidence[:-2].ge(min_confidence) & confidence[2:].ge(min_confidence),
                torch.zeros(1, dtype=torch.bool, device=valid.device),
            ]
        )
    else:
        valid = torch.zeros_like(valid)
    features.masked_fill_(~valid.unsqueeze(-1), 0.0)
    return features, valid


def _float_blob(values: List[float]) -> bytes:
    packed = array("f", values)
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        packed.byteswap()
    return packed.tobytes()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_alphafold_structure_index(
    source_tar: str,
    output_database: str,
    output_manifest: str,
    source_definition_path: str,
) -> Dict[str, Any]:
    """Parse an AFDB tar in one pass without extracting millions of files."""

    source_path = Path(source_tar).resolve()
    database_path = Path(output_database).resolve()
    manifest_path = Path(output_manifest).resolve()
    if database_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing structure artifact")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_definition = json.loads(Path(source_definition_path).read_text(encoding="utf-8"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.", suffix=".tmp", dir=database_path.parent
    )
    os.close(descriptor)
    temporary_database = Path(temporary_name)
    structures = 0
    residues = 0
    confidence_sum = 0.0
    tar_members = 0
    input_reader: Optional[_DigestingRawReader] = None

    try:
        connection = sqlite3.connect(temporary_database)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA locking_mode=EXCLUSIVE;
                CREATE TABLE structures (
                    accession TEXT NOT NULL,
                    fragment INTEGER NOT NULL,
                    model_version INTEGER NOT NULL,
                    sequence TEXT NOT NULL,
                    sequence_sha256 TEXT NOT NULL,
                    coordinates_f32le BLOB NOT NULL,
                    confidence_f32le BLOB NOT NULL,
                    residues INTEGER NOT NULL,
                    mean_confidence REAL NOT NULL,
                    source_member TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    PRIMARY KEY (accession, fragment)
                ) WITHOUT ROWID;
                """
            )
            with source_path.open("rb") as binary:
                input_reader = _DigestingRawReader(binary)
                buffered = io.BufferedReader(input_reader, buffer_size=1024 * 1024)
                with tarfile.open(fileobj=buffered, mode="r|") as archive:
                    for member in archive:
                        tar_members += 1
                        match = _AFDB_MEMBER.search(member.name)
                        if not member.isfile() or match is None:
                            continue
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise RuntimeError(f"unable to read tar member: {member.name}")
                        if member.name.lower().endswith(".gz"):
                            with gzip.GzipFile(fileobj=extracted, mode="rb") as pdb_handle:
                                parsed = parse_pdb_ca(pdb_handle)
                        else:
                            with extracted:
                                parsed = parse_pdb_ca(extracted)
                        accession = match.group(1).upper()
                        fragment = int(match.group(2))
                        model_version = int(match.group(3))
                        flat_coordinates = [value for xyz in parsed["coordinates"] for value in xyz]
                        mean_confidence = sum(parsed["confidence"]) / len(parsed["confidence"])
                        connection.execute(
                            "INSERT INTO structures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                accession,
                                fragment,
                                model_version,
                                parsed["sequence"],
                                hashlib.sha256(parsed["sequence"].encode()).hexdigest(),
                                _float_blob(flat_coordinates),
                                _float_blob(parsed["confidence"]),
                                len(parsed["sequence"]),
                                mean_confidence,
                                member.name,
                                "AlphaFold DB predicted",
                            ),
                        )
                        structures += 1
                        residues += len(parsed["sequence"])
                        confidence_sum += sum(parsed["confidence"])
                        if structures % 10_000 == 0:
                            connection.commit()
                if input_reader.bytes_read < source_path.stat().st_size:
                    while buffered.read(1024 * 1024):
                        pass
            connection.commit()
        finally:
            connection.close()

        observed_sha256 = input_reader.sha256.hexdigest()
        if input_reader.bytes_read != int(source_definition["pdb_tar_bytes"]):
            raise RuntimeError("AlphaFold tar byte count does not match the source receipt")
        expected_sha256 = source_definition.get("pdb_tar_sha256")
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise RuntimeError("AlphaFold tar SHA-256 does not match the retrieval receipt")
        os.replace(temporary_database, database_path)
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "builder_code_sha256": sha256_file(Path(__file__).resolve()),
            "source": source_definition,
            "source_input": {
                "path": str(source_path),
                "bytes": input_reader.bytes_read,
                "sha256": observed_sha256,
                "tar_members": tar_members,
            },
            "database": {
                "path": str(database_path),
                "bytes": database_path.stat().st_size,
                "sha256": sha256_file(database_path),
                "structures": structures,
                "residues": residues,
                "mean_confidence": confidence_sum / residues if residues else None,
                "coordinate_dtype": "float32 little-endian",
            },
            "structure_kind": "predicted",
            "confidence_field": "AlphaFold pLDDT from PDB B-factor",
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    except Exception:
        temporary_database.unlink(missing_ok=True)
        raise


def verify_structure_database(
    database_path: str,
    manifest_path: str,
) -> Dict[str, Any]:
    """Bind a structure database to its content-addressed manifest."""

    database = Path(database_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported structure manifest schema")
    receipt = manifest["database"]
    if database != Path(receipt["path"]).resolve():
        raise RuntimeError("structure database path differs from its manifest")
    if database.stat().st_size != int(receipt["bytes"]):
        raise RuntimeError("structure database byte size differs from its manifest")
    observed_sha256 = sha256_file(database)
    if observed_sha256 != receipt["sha256"]:
        raise RuntimeError("structure database SHA-256 differs from its manifest")
    return {
        "manifest_sha256": sha256_file(manifest_file),
        "database_sha256": observed_sha256,
        "database_bytes": database.stat().st_size,
        "structures": receipt.get("structures"),
        "residues": receipt.get("residues"),
    }


def sample_geometry_pairs(
    coordinates: torch.Tensor,
    confidence: torch.Tensor,
    num_pairs: int,
    generator: torch.Generator,
    min_sequence_separation: int = 3,
    min_confidence: float = 70.0,
    distance_bins: int = 64,
    distance_boundaries: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample O(num_pairs) confident residue pairs and discretize CA distance."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [length, 3]")
    if confidence.shape != coordinates.shape[:1]:
        raise ValueError("confidence must have shape [length]")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if distance_bins < 2:
        raise ValueError("distance_bins must be at least two")
    valid = torch.nonzero(confidence.ge(min_confidence), as_tuple=False).flatten()
    selected: List[Tuple[int, int]] = []
    seen = set()
    attempts = 0
    max_attempts = max(100, num_pairs * 100)
    while len(selected) < num_pairs and attempts < max_attempts and valid.numel() >= 2:
        choices = torch.randint(0, valid.numel(), (2,), generator=generator)
        left = int(valid[choices[0]])
        right = int(valid[choices[1]])
        attempts += 1
        if abs(left - right) < min_sequence_separation:
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(pair)
    if not selected:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    pair_indices = torch.tensor(selected, dtype=torch.long)
    distances = torch.linalg.vector_norm(
        coordinates[pair_indices[:, 0]] - coordinates[pair_indices[:, 1]], dim=-1
    )
    boundaries = (
        distance_boundaries
        if distance_boundaries is not None
        else torch.linspace(2.0, 32.0, steps=distance_bins - 1)
    )
    labels = torch.bucketize(distances, boundaries.to(distances))
    return pair_indices, labels
