import json
from pathlib import Path

import pytest

from openprotein.function_comparison import compare_ec_probes
from openprotein.function_probe import EC_CUTOFF_FIELDS


def _receipt(value: float, benchmark_sha256: str = "benchmark") -> dict[str, object]:
    aggregate = {
        cutoff: {
            "records": 4,
            "micro_auprc": {"mean": value},
            "f1_max": {"mean": value},
        }
        for cutoff in EC_CUTOFF_FIELDS
    }
    runs = [
        {
            "seed": seed,
            "test": {
                cutoff: {"micro_auprc": value, "f1_max": value}
                for cutoff in EC_CUTOFF_FIELDS
            },
        }
        for seed in (1, 2, 3)
    ]
    return {
        "status": "passed",
        "test_data_accessed": True,
        "proteingym_rows_opened": False,
        "benchmark": "TorchProtein EnzymeCommission",
        "benchmark_receipt": {"sha256": benchmark_sha256},
        "evaluation_settings": {"maximum_protein_residues": 550},
        "result": {
            "probe": {"kind": "frozen_encoder_linear", "seeds": [1, 2, 3]},
            "records": {"train": 8, "valid": 4, "test_cutoffs": {}},
            "metric_compatibility": "TorchDrug-compatible",
            "aggregate": aggregate,
            "runs": runs,
        },
        "comparability_limitations": ["frozen encoder"],
    }


def test_ec_comparison_binds_protocol_and_compares_all_metrics(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    baseline = tmp_path / "baseline.json"
    target.write_text(json.dumps(_receipt(0.2)), encoding="utf-8")
    baseline.write_text(json.dumps(_receipt(0.4)), encoding="utf-8")
    result = compare_ec_probes(
        str(target), str(baseline), str(tmp_path / "comparison.json")
    )
    assert result["win_counts"] == {"target": 0, "baseline": 10, "ties": 0}
    assert result["seed_level_win_counts"] == {
        "target": 0,
        "baseline": 30,
        "ties": 0,
    }
    assert len(result["paired_seed_deltas"]["below_95_percent"]["f1_max"]) == 3
    assert (
        result["metrics"]["below_95_percent"]["micro_auprc"]
        ["target_minus_baseline"]
        == -0.2
    )


def test_ec_comparison_rejects_benchmark_drift(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    baseline = tmp_path / "baseline.json"
    target.write_text(json.dumps(_receipt(0.2)), encoding="utf-8")
    baseline.write_text(json.dumps(_receipt(0.4, benchmark_sha256="other")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="benchmark receipts differ"):
        compare_ec_probes(str(target), str(baseline), str(tmp_path / "comparison.json"))


def test_ec_comparison_rejects_probe_protocol_drift(tmp_path: Path) -> None:
    target_payload = _receipt(0.2)
    baseline_payload = _receipt(0.4)
    baseline_payload["result"]["probe"]["seeds"] = [9]
    target = tmp_path / "target.json"
    baseline = tmp_path / "baseline.json"
    target.write_text(json.dumps(target_payload), encoding="utf-8")
    baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocols differ on probe"):
        compare_ec_probes(str(target), str(baseline), str(tmp_path / "comparison.json"))
