import json
from pathlib import Path

import pytest

from openprotein.retrieval_comparison import (
    compare_decontaminated_retrieval,
    compare_retrieval_stress,
)


def _receipt(value: float, order_hash: str = "order") -> dict[str, object]:
    metrics = {
        "candidate_count": 2,
        "protein_to_text_median_rank": 1.0,
        "text_to_protein_median_rank": 1.0,
    }
    for name in (
        "protein_to_text_mrr",
        "protein_to_text_recall_at_1",
        "protein_to_text_recall_at_5",
        "protein_to_text_recall_at_10",
        "text_to_protein_mrr",
        "text_to_protein_recall_at_1",
        "text_to_protein_recall_at_5",
        "text_to_protein_recall_at_10",
    ):
        metrics[name] = value
    return {
        "status": "passed",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "pairs": 2,
        "record_id_subset": {
            "sha256": "subset",
            "evaluation_order_sha256": order_hash,
            "records": 2,
        },
        "retrieval": metrics,
    }


def test_retrieval_comparison_binds_same_subset(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    baseline = tmp_path / "baseline.json"
    target.write_text(json.dumps(_receipt(0.25)), encoding="utf-8")
    baseline.write_text(json.dumps(_receipt(0.5)), encoding="utf-8")
    result = compare_decontaminated_retrieval(
        str(target), str(baseline), str(tmp_path / "comparison.json")
    )
    assert result["metrics"]["protein_to_text_mrr"]["target_minus_baseline"] == -0.25


def test_retrieval_comparison_rejects_order_drift(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    baseline = tmp_path / "baseline.json"
    target.write_text(json.dumps(_receipt(0.25)), encoding="utf-8")
    baseline.write_text(json.dumps(_receipt(0.5, order_hash="other")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluation_order_sha256"):
        compare_decontaminated_retrieval(
            str(target), str(baseline), str(tmp_path / "comparison.json")
        )


def _stress_receipt(value: float, benchmark_sha256: str = "benchmark") -> dict[str, object]:
    direction = {
        "mean_average_precision": value,
        "mean_reciprocal_rank": value,
        "recall_at_1": value,
        "recall_at_5": value,
        "recall_at_10": value,
        "recall_at_100": value,
        "median_rank": 10.0 / value,
        "mean_rank": 20.0 / value,
    }
    return {
        "status": "passed",
        "test_data_accessed": False,
        "proteingym_rows_opened": False,
        "official_protrek_benchmark_reproduction": False,
        "benchmark_receipt": {"sha256": benchmark_sha256},
        "metrics": {
            "query_count": 4,
            "candidate_count": 100_004,
            "matching_policy": "exact_pair_one_positive",
            "protein_to_text": direction,
            "text_to_protein": direction,
        },
        "comparability_limitations": ["local protocol"],
    }


def test_retrieval_stress_comparison_binds_benchmark(tmp_path: Path) -> None:
    target = tmp_path / "target-stress.json"
    baseline = tmp_path / "baseline-stress.json"
    target.write_text(json.dumps(_stress_receipt(0.1)), encoding="utf-8")
    baseline.write_text(json.dumps(_stress_receipt(0.2)), encoding="utf-8")
    result = compare_retrieval_stress(
        str(target), str(baseline), str(tmp_path / "comparison-stress.json")
    )
    gap = result["metrics"]["protein_to_text"]["mean_average_precision"]
    assert gap["target_minus_baseline"] == -0.1
    assert "stronger" in result["conclusion"]


def test_retrieval_stress_comparison_rejects_benchmark_drift(tmp_path: Path) -> None:
    target = tmp_path / "target-stress.json"
    baseline = tmp_path / "baseline-stress.json"
    target.write_text(json.dumps(_stress_receipt(0.1)), encoding="utf-8")
    baseline.write_text(
        json.dumps(_stress_receipt(0.2, benchmark_sha256="other")), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="benchmark receipts differ"):
        compare_retrieval_stress(
            str(target), str(baseline), str(tmp_path / "comparison-stress.json")
        )
