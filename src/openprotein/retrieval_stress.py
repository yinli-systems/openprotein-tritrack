"""Build and score a large-candidate, protocol-inspired retrieval stress test."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import torch

from openprotein.evaluation_subset import load_record_id_allowlist
from openprotein.homology import _manifest_split_files
from openprotein.provenance import sha256_file


BENCHMARK_COLUMNS = (
    "role",
    "query_index",
    "id",
    "accession",
    "name",
    "sequence",
    "text",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _iter_rows(paths: Iterable[Path]) -> Iterator[Dict[str, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "accession", "name", "sequence", "text"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"{path} lacks stress-benchmark columns {sorted(required)}")
            for row_index, row in enumerate(reader, start=2):
                selected = {key: value.strip() for key, value in row.items()}
                if any(not selected[column] for column in required):
                    raise ValueError(f"{path}:{row_index} contains an empty required field")
                yield selected


def _sample_key(seed: int, record_id: str) -> int:
    payload = f"{seed}\0{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _select_smallest_hash_ids(
    paths: Iterable[Path],
    count: int,
    seed: int,
    excluded_ids: set[str],
) -> set[str]:
    heap: List[Tuple[int, str]] = []
    seen: set[str] = set()
    for row in _iter_rows(paths):
        record_id = row["id"]
        if record_id in seen:
            raise ValueError(f"duplicate train record ID: {record_id}")
        seen.add(record_id)
        if record_id in excluded_ids:
            continue
        key = _sample_key(seed, record_id)
        item = (-key, record_id)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise RuntimeError(f"requested {count} distractors but selected only {len(heap)}")
    return {record_id for _, record_id in heap}


def _load_positive_rows(
    paths: Iterable[Path],
    expected_ids: Sequence[str],
) -> List[Dict[str, str]]:
    expected_set = set(expected_ids)
    selected: Dict[str, Dict[str, str]] = {}
    for row in _iter_rows(paths):
        record_id = row["id"]
        if record_id not in expected_set:
            continue
        if record_id in selected:
            raise ValueError(f"duplicate validation record ID: {record_id}")
        selected[record_id] = row
    missing = [record_id for record_id in expected_ids if record_id not in selected]
    if missing:
        raise RuntimeError(f"validation split is missing allowlisted IDs: {missing[:10]}")
    return [selected[record_id] for record_id in expected_ids]


def build_retrieval_stress_benchmark(
    data_manifest_path: str,
    record_id_allowlist_path: str,
    output_csv_path: str,
    output_receipt_path: str,
    distractors: int = 100_000,
    seed: int = 61_003,
) -> Dict[str, Any]:
    """Build clean validation queries plus deterministic multimodal-train distractors."""

    if distractors <= 0:
        raise ValueError("retrieval stress benchmark requires positive distractor count")
    manifest_path = Path(data_manifest_path).resolve()
    output_csv = Path(output_csv_path).resolve()
    output_receipt = Path(output_receipt_path).resolve()
    if output_csv.exists() or output_receipt.exists():
        raise FileExistsError("refusing to overwrite a retrieval stress benchmark")
    if output_csv.parent != output_receipt.parent:
        raise ValueError("benchmark CSV and receipt must share an artifact directory")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    _, train_paths = _manifest_split_files(manifest_path, "train")
    _, validation_paths = _manifest_split_files(manifest_path, "validation")
    query_ids, query_id_set, allowlist_receipt = load_record_id_allowlist(
        record_id_allowlist_path
    )
    positive_rows = _load_positive_rows(validation_paths, query_ids)
    distractor_ids = _select_smallest_hash_ids(
        train_paths,
        distractors,
        seed,
        query_id_set,
    )

    digest = hashlib.sha256()
    temporary_csv = output_csv.with_suffix(output_csv.suffix + f".tmp-{os.getpid()}")
    with temporary_csv.open("w", encoding="utf-8", newline="") as raw_handle:
        class _DigestingWriter:
            def write(self, value: str) -> int:
                digest.update(value.encode("utf-8"))
                return raw_handle.write(value)

        writer = csv.DictWriter(
            _DigestingWriter(),
            fieldnames=BENCHMARK_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for query_index, row in enumerate(positive_rows):
            writer.writerow(
                {
                    "role": "positive_query",
                    "query_index": query_index,
                    **row,
                }
            )
        written_distractors = 0
        for row in _iter_rows(train_paths):
            if row["id"] not in distractor_ids:
                continue
            writer.writerow({"role": "distractor", "query_index": "", **row})
            written_distractors += 1
    if written_distractors != distractors:
        temporary_csv.unlink(missing_ok=True)
        raise RuntimeError("written distractor count differs from deterministic selection")
    temporary_csv.replace(output_csv)

    candidate_count = len(positive_rows) + distractors
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION_PROTOCOL_INSPIRED_RETRIEVAL_STRESS_INPUT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "protocol_label": (
            f"LOCAL_CLEAN_{len(positive_rows)}_QUERIES_PLUS_{distractors}_TRAIN_DISTRACTORS"
        ),
        "official_protrek_benchmark_reproduction": False,
        "data_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "record_id_allowlist": allowlist_receipt,
        "benchmark": {
            "path": output_csv.name,
            "bytes": output_csv.stat().st_size,
            "sha256": digest.hexdigest(),
            "queries": len(positive_rows),
            "distractors": distractors,
            "candidates": candidate_count,
        },
        "candidate_order": (
            "allowlist-ordered validation positives first; then selected train rows in "
            "manifest shard and CSV order"
        ),
        "distractor_selection": {
            "policy": "smallest SHA256(seed + NUL + record_id)",
            "seed": seed,
        },
        "metric_contract": {
            "exact_pair_relevant_candidates_per_query": 1,
            "mean_average_precision_equals_mean_reciprocal_rank": True,
            "directions": ["protein_to_text", "text_to_protein"],
        },
        "comparability_limitations": [
            (
                "This is a protocol-inspired local stress test, not the unavailable official "
                "ProTrek 4k-positive/100k-negative split."
            ),
            (
                "Distractors are manifest train rows and were seen by the target continuation; "
                "the public ProTrek record-level train corpus is also not fully exposed."
            ),
            (
                "Queries pass the local >=50% identity and >=80% query-coverage exclusion only "
                "against the OpenProtein UniRef50 train split; warm-start corpus overlap remains "
                "unresolved."
            ),
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output_receipt, receipt)
    return receipt


def load_retrieval_stress_rows(
    benchmark_csv_path: str,
    benchmark_receipt_path: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Verify and load a previously built stress benchmark."""

    benchmark_path = Path(benchmark_csv_path).resolve()
    receipt_path = Path(benchmark_receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    benchmark = receipt["benchmark"]
    if benchmark_path.stat().st_size != int(benchmark["bytes"]):
        raise RuntimeError("retrieval stress benchmark byte count changed")
    if sha256_file(benchmark_path) != benchmark["sha256"]:
        raise RuntimeError("retrieval stress benchmark SHA-256 changed")
    rows = list(_iter_rows([benchmark_path]))
    queries = int(benchmark["queries"])
    candidates = int(benchmark["candidates"])
    if len(rows) != candidates:
        raise RuntimeError("retrieval stress row count differs from receipt")
    for index, row in enumerate(rows):
        expected_role = "positive_query" if index < queries else "distractor"
        if row.get("role") != expected_role:
            raise RuntimeError("retrieval stress candidate ordering changed")
        if index < queries and int(row.get("query_index", -1)) != index:
            raise RuntimeError("retrieval stress query index changed")
    return rows, receipt


def _direction_metrics(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    query_count: int,
    chunk_size: int,
) -> Dict[str, Any]:
    candidate_count = int(candidates.shape[0])
    candidate_indices = torch.arange(candidate_count, device=candidates.device)
    ranks: List[torch.Tensor] = []
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        scores = queries[start:stop].float() @ candidates.float().transpose(0, 1)
        local = torch.arange(stop - start, device=scores.device)
        positive_indices = torch.arange(start, stop, device=scores.device)
        positive_scores = scores[local, positive_indices]
        greater = scores.gt(positive_scores.unsqueeze(1)).sum(dim=1)
        tied_before = (
            scores.eq(positive_scores.unsqueeze(1))
            & candidate_indices.unsqueeze(0).lt(positive_indices.unsqueeze(1))
        ).sum(dim=1)
        ranks.append((greater + tied_before + 1).cpu())
    rank = torch.cat(ranks).to(torch.float64)
    reciprocal = rank.reciprocal()
    result: Dict[str, Any] = {
        "queries": query_count,
        "candidates": candidate_count,
        "mean_average_precision": float(reciprocal.mean()),
        "mean_reciprocal_rank": float(reciprocal.mean()),
        "median_rank": float(rank.median()),
        "mean_rank": float(rank.mean()),
    }
    for cutoff in (1, 5, 10, 100):
        result[f"recall_at_{cutoff}"] = float(rank.le(cutoff).double().mean())
        result[f"random_reference_recall_at_{cutoff}"] = min(
            cutoff / candidate_count, 1.0
        )
    return result


def large_pool_retrieval_metrics(
    protein_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_count: int,
    device_name: str | None = None,
    chunk_size: int = 32,
) -> Dict[str, Any]:
    """Exact one-positive MAP/MRR and rank metrics against every candidate."""

    if protein_embeddings.ndim != 2 or protein_embeddings.shape != text_embeddings.shape:
        raise ValueError("stress embeddings must have equal [candidates, dimension] shapes")
    candidates = int(protein_embeddings.shape[0])
    if not 1 < query_count <= candidates or chunk_size <= 0:
        raise ValueError("invalid stress query count or metric chunk size")
    if not torch.isfinite(protein_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("stress embeddings contain non-finite values")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    protein = torch.nn.functional.normalize(protein_embeddings.float(), dim=-1).to(device)
    text = torch.nn.functional.normalize(text_embeddings.float(), dim=-1).to(device)
    return {
        "candidate_count": candidates,
        "query_count": query_count,
        "matching_policy": "exact_pair_one_positive",
        "protein_to_text": _direction_metrics(
            protein[:query_count], text, query_count, chunk_size
        ),
        "text_to_protein": _direction_metrics(
            text[:query_count], protein, query_count, chunk_size
        ),
    }
