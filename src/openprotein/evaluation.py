"""Frozen-evaluation seals, mutation scoring, and dependency-free metrics."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from openprotein.artifacts import verify_artifact_manifest
from openprotein.config import load_model_config
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.model import OpenProteinModel
from openprotein.provenance import sha256_file, source_tree_sha256
from openprotein.validation import load_trainable_checkpoint


_MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def create_evaluation_seal(
    checkpoint_dir: str,
    benchmark_manifest: str,
    model_config: str,
    tokenizer_json: str,
    model_artifact_manifest: str,
    output_path: str,
) -> Dict[str, Any]:
    """Bind checkpoint, benchmark, config, and evaluator code before test access."""

    checkpoint = Path(checkpoint_dir).resolve()
    checkpoint_manifest = checkpoint / "manifest.json"
    state = checkpoint / "state.pt"
    benchmark = Path(benchmark_manifest).resolve()
    config = Path(model_config).resolve()
    tokenizer = Path(tokenizer_json).resolve()
    artifact_manifest = Path(model_artifact_manifest).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation seal: {output}")
    for required in (
        checkpoint_manifest,
        state,
        benchmark,
        config,
        tokenizer,
        artifact_manifest,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"evaluation seal input is missing: {required}")
    checkpoint_receipt = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    if state.stat().st_size != int(checkpoint_receipt["state_bytes"]):
        raise RuntimeError("checkpoint size differs from its manifest")
    state_sha256 = sha256_file(state)
    if state_sha256 != checkpoint_receipt["state_sha256"]:
        raise RuntimeError("checkpoint hash differs from its manifest")
    model_definition = load_model_config(str(config))
    artifact_receipt = verify_artifact_manifest(
        str(artifact_manifest),
        expected_root=model_definition.protein_backbone.model_path,
    )
    package_root = Path(__file__).resolve().parent
    seal: Dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": sha256_file(checkpoint_manifest),
            "state_bytes": state.stat().st_size,
            "state_sha256": state_sha256,
            "step": checkpoint_receipt["step"],
        },
        "benchmark": {"path": str(benchmark), "manifest_sha256": sha256_file(benchmark)},
        "model_config": {"path": str(config), "sha256": sha256_file(config)},
        "tokenizer": {"path": str(tokenizer), "sha256": sha256_file(tokenizer)},
        "model_artifact": {
            "manifest_path": str(artifact_manifest),
            **artifact_receipt,
        },
        "evaluation_code_sha256": source_tree_sha256(package_root),
        "policy": (
            "frozen benchmark is evaluation-only and cannot be used for checkpoint, "
            "hyperparameter, prompt, or aggregation-rule selection"
        ),
    }
    _atomic_json(output, seal)
    return seal


def verify_evaluation_seal(path: str) -> Dict[str, Any]:
    seal = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint = seal["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    if sha256_file(checkpoint_path / "manifest.json") != checkpoint["manifest_sha256"]:
        raise RuntimeError("sealed checkpoint manifest has changed")
    if sha256_file(checkpoint_path / "state.pt") != checkpoint["state_sha256"]:
        raise RuntimeError("sealed checkpoint state has changed")
    benchmark = seal["benchmark"]
    if sha256_file(Path(benchmark["path"])) != benchmark["manifest_sha256"]:
        raise RuntimeError("sealed benchmark manifest has changed")
    config = seal["model_config"]
    if sha256_file(Path(config["path"])) != config["sha256"]:
        raise RuntimeError("sealed model config has changed")
    tokenizer = seal["tokenizer"]
    if sha256_file(Path(tokenizer["path"])) != tokenizer["sha256"]:
        raise RuntimeError("sealed tokenizer has changed")
    artifact = seal["model_artifact"]
    if sha256_file(Path(artifact["manifest_path"])) != artifact["manifest_sha256"]:
        raise RuntimeError("sealed model artifact manifest has changed")
    verify_artifact_manifest(
        artifact["manifest_path"],
        expected_root=artifact["root"],
    )
    if source_tree_sha256(Path(__file__).resolve().parent) != seal["evaluation_code_sha256"]:
        raise RuntimeError("evaluation code has changed since sealing")
    return seal


def parse_mutant(mutant: str, wildtype_sequence: str) -> List[Tuple[int, str, str]]:
    substitutions = []
    positions = set()
    for item in mutant.split(":"):
        match = _MUTATION.fullmatch(item.strip().upper())
        if match is None:
            raise ValueError(f"unsupported mutation notation: {item}")
        wildtype, position_text, alternate = match.groups()
        position = int(position_text) - 1
        if not 0 <= position < len(wildtype_sequence):
            raise ValueError(f"mutation position is outside the sequence: {item}")
        if wildtype_sequence[position].upper() != wildtype:
            raise ValueError(f"mutation wildtype residue does not match the sequence: {item}")
        if position in positions:
            raise ValueError(f"mutation contains duplicate positions: {mutant}")
        positions.add(position)
        substitutions.append((position, wildtype, alternate))
    return substitutions


def _crop_start(sequence_length: int, position: int, max_length: int) -> int:
    content_length = max_length - 2
    if content_length <= 0:
        raise ValueError("max_length must leave room for special tokens")
    if sequence_length <= content_length:
        return 0
    centered = position - content_length // 2
    return min(max(0, centered), sequence_length - content_length)


@torch.no_grad()
def paired_masked_marginal_scores(
    model: OpenProteinModel,
    tokenizer: EsmcTokenizer,
    sequence: str,
    mutants: Sequence[str],
    device: torch.device,
    positions_per_batch: int = 8,
    max_length: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Score the frozen ESMC path and bridge from one shared backbone pass."""

    if positions_per_batch <= 0:
        raise ValueError("positions_per_batch must be positive")
    parsed = [parse_mutant(mutant, sequence) for mutant in mutants]
    positions = sorted({position for substitutions in parsed for position, _, _ in substitutions})
    if not positions:
        raise ValueError("at least one mutant is required")
    context = int(max_length or model.config.max_sequence_length)
    logits_by_head: Dict[str, Dict[int, torch.Tensor]] = {
        "base_esmc": {},
        "openprotein": {},
    }
    model.eval()
    for start in range(0, len(positions), positions_per_batch):
        chunk = positions[start : start + positions_per_batch]
        rows = []
        local_positions = []
        for position in chunk:
            crop_start = _crop_start(len(sequence), position, context)
            row = tokenizer.encode(sequence, max_length=context, crop_start=crop_start)
            local_position = position - crop_start + 1
            row[local_position] = tokenizer.mask_token_id
            rows.append(row)
            local_positions.append(local_position)
        padded = tokenizer.pad(rows)
        batch = padded["input_ids"].to(device)
        attention = padded["attention_mask"].to(device)
        output = model(
            protein_input_ids=batch,
            protein_attention_mask=attention,
            use_bridge=True,
            attention_policy="bidirectional",
        )
        for row, (position, local_position) in enumerate(zip(chunk, local_positions)):
            for head, key in (
                ("base_esmc", "base_protein_logits"),
                ("openprotein", "protein_logits"),
            ):
                logits = output[key]
                logits_by_head[head][position] = torch.log_softmax(
                    logits[row, local_position].float(), dim=-1
                ).cpu()

    scores: Dict[str, List[float]] = {head: [] for head in logits_by_head}
    for substitutions in parsed:
        for head, position_logits in logits_by_head.items():
            score = 0.0
            for position, wildtype, alternate in substitutions:
                if wildtype not in tokenizer.vocabulary or alternate not in tokenizer.vocabulary:
                    raise ValueError("mutation uses a residue outside the tokenizer vocabulary")
                distribution = position_logits[position]
                score += float(
                    distribution[tokenizer.vocabulary[alternate]]
                    - distribution[tokenizer.vocabulary[wildtype]]
                )
            scores[head].append(score)
    return scores


@torch.no_grad()
def masked_marginal_scores(
    model: OpenProteinModel,
    tokenizer: EsmcTokenizer,
    sequence: str,
    mutants: Sequence[str],
    device: torch.device,
    use_bridge: bool,
    positions_per_batch: int = 8,
    max_length: Optional[int] = None,
) -> List[float]:
    """Return ESM-style summed mutant-vs-wildtype masked log odds."""

    if use_bridge:
        return paired_masked_marginal_scores(
            model,
            tokenizer,
            sequence,
            mutants,
            device,
            positions_per_batch=positions_per_batch,
            max_length=max_length,
        )["openprotein"]

    parsed = [parse_mutant(mutant, sequence) for mutant in mutants]
    positions = sorted({position for substitutions in parsed for position, _, _ in substitutions})
    position_logits: Dict[int, torch.Tensor] = {}
    model.eval()
    context = int(max_length or model.config.max_sequence_length)
    for start in range(0, len(positions), positions_per_batch):
        chunk = positions[start : start + positions_per_batch]
        rows = []
        local_positions = []
        for position in chunk:
            crop_start = _crop_start(len(sequence), position, context)
            row = tokenizer.encode(sequence, max_length=context, crop_start=crop_start)
            local_position = position - crop_start + 1
            row[local_position] = tokenizer.mask_token_id
            rows.append(row)
            local_positions.append(local_position)
        padded = tokenizer.pad(rows)
        batch = padded["input_ids"].to(device)
        attention = padded["attention_mask"].to(device)
        output = model(
            protein_input_ids=batch,
            protein_attention_mask=attention,
            use_bridge=use_bridge,
            attention_policy="bidirectional",
        )
        logits = output["protein_logits"]
        for row, (position, local_position) in enumerate(zip(chunk, local_positions)):
            position_logits[position] = torch.log_softmax(
                logits[row, local_position].float(), dim=-1
            ).cpu()

    scores = []
    for substitutions in parsed:
        score = 0.0
        for position, wildtype, alternate in substitutions:
            distribution = position_logits[position]
            if wildtype not in tokenizer.vocabulary or alternate not in tokenizer.vocabulary:
                raise ValueError("mutation uses a residue outside the tokenizer vocabulary")
            wildtype_id = tokenizer.vocabulary[wildtype]
            alternate_id = tokenizer.vocabulary[alternate]
            score += float(distribution[alternate_id] - distribution[wildtype_id])
        scores.append(score)
    return scores


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for offset in range(start, end):
            ranks[order[offset]] = rank
        start = end
    return ranks


def spearman_correlation(predictions: Iterable[float], targets: Iterable[float]) -> float:
    x = list(predictions)
    y = list(targets)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman correlation requires equal inputs with at least two values")
    rank_x = _average_ranks(x)
    rank_y = _average_ranks(y)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    denominator_x = sum((a - mean_x) ** 2 for a in rank_x)
    denominator_y = sum((b - mean_y) ** 2 for b in rank_y)
    if denominator_x == 0.0 or denominator_y == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return numerator / (denominator_x * denominator_y) ** 0.5


def aggregate_proteingym_spearman(
    assay_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Macro-average assays by UniProt within selection type, then categories."""

    if not assay_results:
        raise ValueError("cannot aggregate an empty benchmark")
    model_names = set(assay_results[0]["spearman"])
    if any(set(item["spearman"]) != model_names for item in assay_results):
        raise ValueError("assays do not contain the same model score set")
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for item in assay_results:
        key = (str(item["uniprot_id"]), str(item["selection_type"]))
        grouped.setdefault(key, []).append(item["spearman"])
    uniprot_scores = []
    for (uniprot_id, selection_type), values in sorted(grouped.items()):
        uniprot_scores.append(
            {
                "uniprot_id": uniprot_id,
                "selection_type": selection_type,
                "assays": len(values),
                "spearman": {
                    model: sum(value[model] for value in values) / len(values)
                    for model in sorted(model_names)
                },
            }
        )
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for item in uniprot_scores:
        categories.setdefault(item["selection_type"], []).append(item)
    category_scores = {
        category: {
            model: sum(item["spearman"][model] for item in values) / len(values)
            for model in sorted(model_names)
        }
        for category, values in sorted(categories.items())
    }
    primary = {
        model: sum(category[model] for category in category_scores.values())
        / len(category_scores)
        for model in sorted(model_names)
    }
    return {
        "aggregation": (
            "assay mean per UniProt and selection type, then macro mean of selection types"
        ),
        "uniprot_selection_scores": uniprot_scores,
        "selection_type_scores": category_scores,
        "primary_scores": primary,
    }


def paired_bootstrap_proteingym_differences(
    assay_results: Sequence[Dict[str, Any]],
    candidate: str,
    baselines: Sequence[str],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap candidate-minus-baseline differences by UniProt within category."""

    if samples < 2:
        raise ValueError("paired bootstrap requires at least two samples")
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for item in assay_results:
        grouped.setdefault(
            (str(item["uniprot_id"]), str(item["selection_type"])), []
        ).append(item["spearman"])
    category_units: Dict[str, List[Dict[str, float]]] = {}
    for (_uniprot, selection_type), values in sorted(grouped.items()):
        unit = {
            model: sum(value[model] for value in values) / len(values)
            for model in values[0]
        }
        category_units.setdefault(selection_type, []).append(unit)
    if not category_units:
        raise ValueError("paired bootstrap has no category units")
    models = {candidate, *baselines}
    if any(
        any(model not in unit for model in models)
        for units in category_units.values()
        for unit in units
    ):
        raise RuntimeError("paired bootstrap model set is incomplete")
    generator = random.Random(seed)
    draws: Dict[str, List[float]] = {baseline: [] for baseline in baselines}
    for _ in range(samples):
        category_differences = {baseline: [] for baseline in baselines}
        for units in category_units.values():
            selected = [units[generator.randrange(len(units))] for _ in range(len(units))]
            for baseline in baselines:
                category_differences[baseline].append(
                    sum(unit[candidate] - unit[baseline] for unit in selected) / len(selected)
                )
        for baseline in baselines:
            draws[baseline].append(
                sum(category_differences[baseline]) / len(category_differences[baseline])
            )

    def quantile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    results = {}
    for baseline, values in draws.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        results[baseline] = {
            "candidate_minus_baseline_mean": mean,
            "bootstrap_standard_error": variance**0.5,
            "percentile_95_percent_interval": [
                quantile(values, 0.025),
                quantile(values, 0.975),
            ],
        }
    return {
        "policy": (
            "resample UniProt-selection units with replacement within each selection type, "
            "then macro-average selection types"
        ),
        "samples": samples,
        "seed": seed,
        "differences": results,
    }


def _verified_manifest_file(
    root: Path,
    item: Dict[str, Any],
    verify_sha256: bool = True,
) -> Path:
    path = (root / item["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("benchmark file escapes the dataset root") from error
    if path.stat().st_size != int(item["bytes"]):
        raise RuntimeError(f"benchmark file byte count differs: {item['path']}")
    if verify_sha256 and sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"benchmark file SHA-256 differs: {item['path']}")
    return path


def _read_baseline_scores(
    path: Path,
    score_column: str,
    expected_mutants: Sequence[str],
) -> List[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mutant", score_column}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"baseline file lacks columns: {sorted(required)}")
        scores: Dict[str, float] = {}
        for row in reader:
            mutant = row["mutant"].strip()
            if mutant in scores:
                raise ValueError(f"baseline contains duplicate mutant: {mutant}")
            score = float(row[score_column])
            if not math.isfinite(score):
                raise ValueError("baseline contains a non-finite score")
            scores[mutant] = score
    if set(scores) != set(expected_mutants):
        raise RuntimeError("baseline mutant set differs from the frozen assay")
    return [scores[mutant] for mutant in expected_mutants]


def _unique_zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    normalized = suffix.strip("/")
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
        and (name.strip("/") == normalized or name.strip("/").endswith(f"/{normalized}"))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one ZIP member ending in {normalized!r}, found {len(matches)}"
        )
    return matches[0]


def _read_baseline_score_rows(
    handle: io.TextIOBase,
    score_column: str,
    expected_mutants: Sequence[str],
    directionality: float,
) -> List[float]:
    reader = csv.DictReader(handle)
    required = {"mutant", score_column}
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError(f"baseline file lacks columns: {sorted(required)}")
    scores: Dict[str, float] = {}
    for row in reader:
        mutant = row["mutant"].strip()
        if mutant in scores:
            raise ValueError(f"baseline contains duplicate mutant: {mutant}")
        score = float(row[score_column]) * directionality
        if not math.isfinite(score):
            raise ValueError("baseline contains a non-finite score")
        scores[mutant] = score
    if set(scores) != set(expected_mutants):
        raise RuntimeError("baseline mutant set differs from the frozen assay")
    return [scores[mutant] for mutant in expected_mutants]


def iter_frozen_proteingym_archive_assays(
    benchmark: Dict[str, Any],
    shard_index: int = 0,
    shard_count: int = 1,
    verify_artifacts: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Open archive rows only after the caller has verified an evaluation seal."""

    if benchmark.get("archive_protocol") != "proteingym-v1.3-opaque-zip":
        raise RuntimeError("unsupported ProteinGym archive protocol")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("archive shard index must be in [0, shard_count)")
    root = Path(benchmark["dataset_root"]).resolve()
    reference_path = _verified_manifest_file(
        root, benchmark["reference_file"], verify_sha256=verify_artifacts
    )
    assay_archive_path = _verified_manifest_file(
        root, benchmark["assay_archive"], verify_sha256=verify_artifacts
    )
    score_archive_path = _verified_manifest_file(
        root, benchmark["score_archive"], verify_sha256=verify_artifacts
    )
    baselines = benchmark.get("baselines", {})
    if not baselines:
        raise RuntimeError("archive benchmark declares no official baselines")

    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        references = list(csv.DictReader(handle))
    if not references:
        raise RuntimeError("ProteinGym reference file contains no assays")
    required_reference = {
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "target_seq",
        "coarse_selection_type",
    }
    if not required_reference <= set(references[0]):
        raise ValueError(f"reference file lacks columns: {sorted(required_reference)}")
    assay_ids = [row["DMS_id"].strip() for row in references]
    if len(set(assay_ids)) != len(assay_ids):
        raise ValueError("ProteinGym reference file contains duplicate DMS IDs")

    with zipfile.ZipFile(assay_archive_path) as assay_archive, zipfile.ZipFile(
        score_archive_path
    ) as score_archive:
        for reference_index, reference in enumerate(references):
            if reference_index % shard_count != shard_index:
                continue
            assay_id = reference["DMS_id"].strip()
            assay_filename = reference["DMS_filename"].strip()
            assay_member = _unique_zip_member(assay_archive, assay_filename)
            with assay_archive.open(assay_member) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    required_assay = {"mutant", "DMS_score"}
                    if reader.fieldnames is None or not required_assay <= set(
                        reader.fieldnames
                    ):
                        raise ValueError(f"assay lacks columns: {sorted(required_assay)}")
                    mutants: List[str] = []
                    targets: List[float] = []
                    for row in reader:
                        mutant = row["mutant"].strip()
                        target = float(row["DMS_score"])
                        if not math.isfinite(target):
                            raise ValueError("assay contains a non-finite DMS score")
                        mutants.append(mutant)
                        targets.append(target)
            if len(set(mutants)) != len(mutants):
                raise ValueError("assay contains duplicate mutant rows")
            if len(mutants) < 2:
                raise ValueError("assay contains fewer than two mutants")

            baseline_scores: Dict[str, List[float]] = {}
            baseline_members: Dict[str, str] = {}
            for name, definition in sorted(baselines.items()):
                member_suffix = (
                    f"{str(definition['member_prefix']).strip('/')}/{assay_id}.csv"
                )
                member = _unique_zip_member(score_archive, member_suffix)
                baseline_members[name] = member
                with score_archive.open(member) as binary:
                    with io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
                        baseline_scores[name] = _read_baseline_score_rows(
                            handle,
                            str(definition["score_column"]),
                            mutants,
                            float(definition.get("directionality", 1.0)),
                        )
            yield {
                "assay_id": assay_id,
                "reference_index": reference_index,
                "assay_filename": assay_filename,
                "assay_member": assay_member,
                "uniprot_id": reference["UniProt_ID"].strip(),
                "selection_type": reference["coarse_selection_type"].strip(),
                "target_sequence": reference["target_seq"].strip().upper(),
                "mutants": mutants,
                "targets": targets,
                "baseline_scores": baseline_scores,
                "baseline_members": baseline_members,
            }


@torch.no_grad()
def evaluate_frozen_proteingym(
    seal_path: str,
    output_path: str,
    positions_per_batch: int = 8,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one sealed checkpoint without exposing rows before seal verification."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evaluation: {output}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("distributed evaluation rank topology is invalid")
    distributed = world_size > 1
    if distributed:
        if device_name is not None or not torch.cuda.is_available():
            raise ValueError("distributed frozen evaluation requires torchrun-managed CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        sealed_objects: List[Optional[Dict[str, Any]]] = [
            verify_evaluation_seal(seal_path) if rank == 0 else None
        ]
        torch.distributed.broadcast_object_list(sealed_objects, src=0)
        if sealed_objects[0] is None:
            raise RuntimeError("rank zero did not broadcast an evaluation seal")
        seal = sealed_objects[0]
    else:
        seal = verify_evaluation_seal(seal_path)
    manifest_path = Path(seal["benchmark"]["path"])
    benchmark = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(benchmark.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported benchmark manifest schema")
    archive_protocol = benchmark.get("archive_protocol")
    benchmark_synthetic = benchmark.get("synthetic") is True
    verified = []
    if archive_protocol is None:
        dataset_root = Path(benchmark["dataset_root"]).resolve()
        assays = benchmark.get("assays")
        if not isinstance(assays, list) or not assays:
            raise RuntimeError("benchmark manifest contains no assays")
        for assay in assays:
            assay_path = _verified_manifest_file(dataset_root, assay)
            baseline_paths = {
                name: _verified_manifest_file(dataset_root, definition)
                for name, definition in assay.get("baselines", {}).items()
            }
            verified.append((assay, assay_path, baseline_paths))
    elif archive_protocol != "proteingym-v1.3-opaque-zip":
        raise RuntimeError("unsupported benchmark archive protocol")

    if archive_protocol is not None and distributed:
        if rank == 0:
            root = Path(benchmark["dataset_root"]).resolve()
            for key in ("reference_file", "assay_archive", "score_archive"):
                _verified_manifest_file(root, benchmark[key])
        torch.distributed.barrier()

    config = load_model_config(seal["model_config"]["path"])
    tokenizer = EsmcTokenizer.from_tokenizer_json(seal["tokenizer"]["path"])
    device = (
        torch.device("cuda", local_rank)
        if distributed
        else torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = OpenProteinModel.from_config(config, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    expected_provenance = {
        "model_config_sha256": seal["model_config"]["sha256"],
        "tokenizer_sha256": seal["tokenizer"]["sha256"],
        "model_artifact_manifest_sha256": seal["model_artifact"]["manifest_sha256"],
    }
    checkpoint = load_trainable_checkpoint(
        model,
        seal["checkpoint"]["path"],
        expected_provenance,
    )
    assay_inputs: Iterable[Dict[str, Any]]
    if archive_protocol is not None:
        assay_inputs = iter_frozen_proteingym_archive_assays(
            benchmark,
            shard_index=rank,
            shard_count=world_size,
            verify_artifacts=not distributed,
        )
    else:
        legacy_inputs = []
        for assay_index, (assay, assay_path, baseline_paths) in enumerate(verified):
            if assay_index % world_size != rank:
                continue
            mutants: List[str] = []
            targets: List[float] = []
            with assay_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"mutant", "DMS_score"}
                if reader.fieldnames is None or not required <= set(reader.fieldnames):
                    raise ValueError(f"assay lacks columns: {sorted(required)}")
                for row in reader:
                    mutant = row["mutant"].strip()
                    target = float(row["DMS_score"])
                    if not math.isfinite(target):
                        raise ValueError("assay contains a non-finite DMS score")
                    mutants.append(mutant)
                    targets.append(target)
            if len(set(mutants)) != len(mutants):
                raise ValueError("assay contains duplicate mutant rows")
            baseline_scores = {
                name: _read_baseline_scores(
                    baseline_path,
                    assay["baselines"][name]["score_column"],
                    mutants,
                )
                for name, baseline_path in baseline_paths.items()
            }
            legacy_inputs.append(
                {
                    "assay_id": assay["assay_id"],
                    "uniprot_id": assay["uniprot_id"],
                    "selection_type": assay["selection_type"],
                    "target_sequence": str(assay["target_sequence"]).upper(),
                    "mutants": mutants,
                    "targets": targets,
                    "baseline_scores": baseline_scores,
                }
            )
        assay_inputs = legacy_inputs

    assay_results = []
    for assay in assay_inputs:
        mutants = assay["mutants"]
        targets = assay["targets"]
        paired = paired_masked_marginal_scores(
            model,
            tokenizer,
            assay["target_sequence"],
            mutants,
            device,
            positions_per_batch=positions_per_batch,
            max_length=config.max_sequence_length,
        )
        predictions: Dict[str, List[float]] = {
            "ESMC-6B-local": paired["base_esmc"],
            "OpenProtein-7B": paired["openprotein"],
        }
        predictions.update(assay["baseline_scores"])
        raw_spearman = {
            name: spearman_correlation(scores, targets)
            for name, scores in sorted(predictions.items())
        }
        result = {
            "assay_id": assay["assay_id"],
            "uniprot_id": assay["uniprot_id"],
            "selection_type": assay["selection_type"],
            "rows": len(mutants),
            "spearman": {name: round(value, 3) for name, value in raw_spearman.items()},
            "spearman_raw": raw_spearman,
        }
        if "assay_member" in assay:
            result["assay_member"] = assay["assay_member"]
            result["baseline_members"] = assay["baseline_members"]
        assay_results.append(result)

    if distributed:
        gathered: List[Optional[List[Dict[str, Any]]]] = [None] * world_size
        torch.distributed.all_gather_object(gathered, assay_results)
        assay_results = [
            item
            for rank_results in gathered
            if rank_results is not None
            for item in rank_results
        ]
        assay_results.sort(key=lambda item: str(item["assay_id"]))
        assay_ids = [str(item["assay_id"]) for item in assay_results]
        if len(set(assay_ids)) != len(assay_ids):
            raise RuntimeError("distributed evaluation produced duplicate assays")

    aggregate = aggregate_proteingym_spearman(assay_results)
    gate = benchmark.get("claim_gate", {})
    margin = float(gate.get("minimum_margin", 0.0))
    required_baselines = list(gate.get("required_baselines", []))
    primary = aggregate["primary_scores"]
    missing = set(required_baselines) - set(primary)
    if missing:
        raise RuntimeError(f"required baselines are absent: {sorted(missing)}")
    bootstrap_samples = int(gate.get("paired_bootstrap_samples", 10_000))
    bootstrap_seed = int(gate.get("paired_bootstrap_seed", 1701))
    minimum_ci_lower_bound = float(
        gate.get("minimum_paired_bootstrap_lower_bound", 0.0)
    )
    bootstrap = paired_bootstrap_proteingym_differences(
        assay_results,
        "OpenProtein-7B",
        required_baselines,
        bootstrap_samples,
        bootstrap_seed,
    )
    point_gate_passed = bool(required_baselines) and all(
        primary["OpenProtein-7B"] >= primary[name] + margin for name in required_baselines
    )
    interval_gate_passed = bool(required_baselines) and all(
        bootstrap["differences"][name]["percentile_95_percent_interval"][0]
        >= minimum_ci_lower_bound
        for name in required_baselines
    )
    gate_passed = not benchmark_synthetic and point_gate_passed and interval_gate_passed
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": (
            "SYNTHETIC_EVAL_SMOKE" if benchmark_synthetic else "FROZEN_EVAL"
        ),
        "benchmark_synthetic": benchmark_synthetic,
        "distributed": distributed,
        "world_size": world_size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seal_path": str(Path(seal_path).resolve()),
        "seal_sha256": sha256_file(Path(seal_path)),
        "checkpoint": checkpoint,
        "benchmark_manifest_sha256": seal["benchmark"]["manifest_sha256"],
        "assays": assay_results,
        **aggregate,
        "paired_bootstrap": bootstrap,
        "claim_gate": {
            "required_baselines": required_baselines,
            "minimum_margin": margin,
            "minimum_paired_bootstrap_lower_bound": minimum_ci_lower_bound,
            "point_gate_passed": point_gate_passed,
            "interval_gate_passed": interval_gate_passed,
            "synthetic_forced_false": benchmark_synthetic,
            "passed": gate_passed,
        },
        "test_data_accessed": not benchmark_synthetic,
        "synthetic_data_accessed": benchmark_synthetic,
    }
    if rank == 0:
        _atomic_json(output, receipt)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        if rank != 0:
            return {
                "schema_version": 1,
                "status": "worker_complete",
                "evidence_level": (
                    "SYNTHETIC_EVAL_SMOKE_WORKER"
                    if benchmark_synthetic
                    else "FROZEN_EVAL_WORKER"
                ),
                "rank": rank,
                "world_size": world_size,
                "assays": len(assay_results),
            }
    return receipt
