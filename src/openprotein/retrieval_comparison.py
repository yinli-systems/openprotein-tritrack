"""Receipt-bound comparison for paired retrieval evaluations."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from openprotein.provenance import sha256_file


METRIC_NAMES = (
    "protein_to_text_mrr",
    "protein_to_text_recall_at_1",
    "protein_to_text_recall_at_5",
    "protein_to_text_recall_at_10",
    "text_to_protein_mrr",
    "text_to_protein_recall_at_1",
    "text_to_protein_recall_at_5",
    "text_to_protein_recall_at_10",
)

STRESS_DIRECTIONS = ("protein_to_text", "text_to_protein")
STRESS_HIGHER_IS_BETTER = (
    "mean_average_precision",
    "mean_reciprocal_rank",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_100",
)
STRESS_LOWER_IS_BETTER = ("median_rank", "mean_rank")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def compare_decontaminated_retrieval(
    target_receipt_path: str,
    baseline_receipt_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """Require an identical clean candidate pool and compare aggregate metrics."""

    target_path = Path(target_receipt_path).resolve()
    baseline_path = Path(baseline_receipt_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retrieval comparison: {output}")
    target = json.loads(target_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for name, receipt in (("target", target), ("baseline", baseline)):
        if receipt.get("status") != "passed":
            raise RuntimeError(f"{name} retrieval receipt did not pass")
        if receipt.get("test_data_accessed") is not False:
            raise RuntimeError(f"{name} retrieval receipt does not prove test isolation")
        if receipt.get("proteingym_rows_opened") is not False:
            raise RuntimeError(f"{name} retrieval receipt opened ProteinGym rows")
    if int(target["pairs"]) != int(baseline["pairs"]):
        raise RuntimeError("target and baseline pair counts differ")
    target_subset = target.get("record_id_subset")
    baseline_subset = baseline.get("record_id_subset")
    for key in ("sha256", "evaluation_order_sha256", "records"):
        if target_subset.get(key) != baseline_subset.get(key):
            raise RuntimeError(f"target and baseline clean subsets differ on {key}")
    target_metrics = target["retrieval"]
    baseline_metrics = baseline["retrieval"]
    if target_metrics["candidate_count"] != baseline_metrics["candidate_count"]:
        raise RuntimeError("target and baseline candidate counts differ")

    comparison: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_NAMES:
        target_value = float(target_metrics[metric])
        baseline_value = float(baseline_metrics[metric])
        comparison[metric] = {
            "target": target_value,
            "baseline": baseline_value,
            "target_minus_baseline": target_value - baseline_value,
            "target_over_baseline": target_value / baseline_value,
        }
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION_DECONTAMINATED_SUBSET_MODEL_COMPARISON",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "target_receipt": {"path": str(target_path), "sha256": sha256_file(target_path)},
        "baseline_receipt": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "subset": target_subset,
        "candidate_count": int(target_metrics["candidate_count"]),
        "metrics": comparison,
        "rank_summary": {
            "protein_to_text_target_median_rank": target_metrics[
                "protein_to_text_median_rank"
            ],
            "protein_to_text_baseline_median_rank": baseline_metrics[
                "protein_to_text_median_rank"
            ],
            "text_to_protein_target_median_rank": target_metrics[
                "text_to_protein_median_rank"
            ],
            "text_to_protein_baseline_median_rank": baseline_metrics[
                "text_to_protein_median_rank"
            ],
        },
        "conclusion": (
            "The pinned ProTrek-650M baseline is stronger than the target checkpoint in both "
            "retrieval directions on the identical 464-query decontaminated candidate pool."
        ),
        "statistical_limitations": (
            "Receipts contain aggregate ranks but not per-query rank vectors, so this comparison "
            "does not claim a paired-bootstrap confidence interval or significance test."
        ),
        "comparability_limitations": [
            (
                "The local OpenProtein train split was audited, but record-level warm-start ESM-2 "
                "and ProTrek training overlap cannot be fully excluded."
            ),
            (
                "This 464-candidate validation pool is not the official ProTrek 4k+100k benchmark."
            ),
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt


def compare_retrieval_stress(
    target_receipt_path: str,
    baseline_receipt_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """Bind one large-candidate benchmark and compare target/baseline aggregates."""

    target_path = Path(target_receipt_path).resolve()
    baseline_path = Path(baseline_receipt_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retrieval stress comparison: {output}")
    target = json.loads(target_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for name, receipt in (("target", target), ("baseline", baseline)):
        if receipt.get("status") != "passed":
            raise RuntimeError(f"{name} retrieval stress receipt did not pass")
        if receipt.get("test_data_accessed") is not False:
            raise RuntimeError(f"{name} retrieval stress receipt does not prove test isolation")
        if receipt.get("proteingym_rows_opened") is not False:
            raise RuntimeError(f"{name} retrieval stress receipt opened ProteinGym rows")
        if receipt.get("official_protrek_benchmark_reproduction") is not False:
            raise RuntimeError(f"{name} receipt mislabels the local stress protocol")
    target_benchmark = target.get("benchmark_receipt", {})
    baseline_benchmark = baseline.get("benchmark_receipt", {})
    if target_benchmark.get("sha256") != baseline_benchmark.get("sha256"):
        raise RuntimeError("target and baseline stress benchmark receipts differ")
    target_metrics = target["metrics"]
    baseline_metrics = baseline["metrics"]
    for key in ("query_count", "candidate_count", "matching_policy"):
        if target_metrics.get(key) != baseline_metrics.get(key):
            raise RuntimeError(f"target and baseline stress metrics differ on {key}")

    comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
    baseline_wins_primary = True
    for direction in STRESS_DIRECTIONS:
        target_direction = target_metrics[direction]
        baseline_direction = baseline_metrics[direction]
        direction_comparison: Dict[str, Dict[str, float]] = {}
        for metric in (*STRESS_HIGHER_IS_BETTER, *STRESS_LOWER_IS_BETTER):
            target_value = float(target_direction[metric])
            baseline_value = float(baseline_direction[metric])
            values = {
                "target": target_value,
                "baseline": baseline_value,
                "target_minus_baseline": target_value - baseline_value,
            }
            if baseline_value != 0.0:
                values["target_over_baseline"] = target_value / baseline_value
            direction_comparison[metric] = values
        comparison[direction] = direction_comparison
        baseline_wins_primary &= all(
            float(baseline_direction[metric]) > float(target_direction[metric])
            for metric in STRESS_HIGHER_IS_BETTER
        )
        baseline_wins_primary &= float(baseline_direction["median_rank"]) < float(
            target_direction["median_rank"]
        )

    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "VALIDATION_PROTOCOL_INSPIRED_RETRIEVAL_STRESS_COMPARISON",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "official_protrek_benchmark_reproduction": False,
        "target_receipt": {"path": str(target_path), "sha256": sha256_file(target_path)},
        "baseline_receipt": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "benchmark_receipt": target_benchmark,
        "query_count": int(target_metrics["query_count"]),
        "candidate_count": int(target_metrics["candidate_count"]),
        "matching_policy": target_metrics["matching_policy"],
        "metrics": comparison,
        "conclusion": (
            "The pinned baseline is stronger in both directions on every MAP/MRR/Recall@K "
            "metric and median rank."
            if baseline_wins_primary
            else "The target and pinned baseline have mixed aggregate stress-test results."
        ),
        "statistical_limitations": (
            "Receipts do not retain per-query ranks, so no paired bootstrap confidence interval "
            "or significance claim is made."
        ),
        "comparability_limitations": target.get("comparability_limitations", []),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt
