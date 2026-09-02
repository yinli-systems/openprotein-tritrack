"""Receipt-bound comparison for public frozen-encoder function probes."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openprotein.function_probe import EC_CUTOFF_FIELDS
from openprotein.provenance import sha256_file

EC_METRICS = ("micro_auprc", "f1_max")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def compare_ec_probes(
    target_receipt_path: str,
    baseline_receipt_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Compare two models only after binding the benchmark and probe protocol."""

    target_path = Path(target_receipt_path).resolve()
    baseline_path = Path(baseline_receipt_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EC probe comparison: {output}")
    target = json.loads(target_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for name, receipt in (("target", target), ("baseline", baseline)):
        if receipt.get("status") != "passed":
            raise RuntimeError(f"{name} EC probe receipt did not pass")
        if receipt.get("test_data_accessed") is not True:
            raise RuntimeError(f"{name} EC probe receipt does not declare public-test access")
        if receipt.get("proteingym_rows_opened") is not False:
            raise RuntimeError(f"{name} EC probe receipt opened ProteinGym rows")
        if receipt.get("benchmark") != "TorchProtein EnzymeCommission":
            raise RuntimeError(f"{name} receipt is not the expected EC benchmark")
    if target.get("benchmark_receipt", {}).get("sha256") != baseline.get(
        "benchmark_receipt", {}
    ).get("sha256"):
        raise RuntimeError("target and baseline EC benchmark receipts differ")
    if target.get("evaluation_settings") != baseline.get("evaluation_settings"):
        raise RuntimeError("target and baseline EC encoding settings differ")

    target_result = target["result"]
    baseline_result = baseline["result"]
    for key in ("probe", "records", "metric_compatibility"):
        if target_result.get(key) != baseline_result.get(key):
            raise RuntimeError(f"target and baseline EC probe protocols differ on {key}")
    expected_seeds = [int(seed) for seed in target_result["probe"]["seeds"]]
    target_runs = {int(run["seed"]): run for run in target_result.get("runs", [])}
    baseline_runs = {int(run["seed"]): run for run in baseline_result.get("runs", [])}
    if (
        len(target_runs) != len(target_result.get("runs", []))
        or len(baseline_runs) != len(baseline_result.get("runs", []))
        or set(target_runs) != set(expected_seeds)
        or set(baseline_runs) != set(expected_seeds)
    ):
        raise RuntimeError("target and baseline EC probe runs do not match shared seeds")

    metrics: dict[str, dict[str, dict[str, float | str]]] = {}
    paired_seed_deltas: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}
    winners = []
    seed_winners = []
    for cutoff in EC_CUTOFF_FIELDS:
        target_cutoff = target_result["aggregate"][cutoff]
        baseline_cutoff = baseline_result["aggregate"][cutoff]
        if int(target_cutoff["records"]) != int(baseline_cutoff["records"]):
            raise RuntimeError(f"target and baseline EC cutoff sizes differ on {cutoff}")
        cutoff_metrics: dict[str, dict[str, float | str]] = {}
        cutoff_seed_deltas: dict[str, list[dict[str, float | int | str]]] = {}
        for metric in EC_METRICS:
            target_value = float(target_cutoff[metric]["mean"])
            baseline_value = float(baseline_cutoff[metric]["mean"])
            winner = (
                "target"
                if target_value > baseline_value
                else "baseline"
                if baseline_value > target_value
                else "tie"
            )
            winners.append(winner)
            values: dict[str, float | str] = {
                "target_mean": target_value,
                "baseline_mean": baseline_value,
                "target_minus_baseline": target_value - baseline_value,
                "winner": winner,
            }
            if baseline_value != 0.0:
                values["target_over_baseline"] = target_value / baseline_value
            cutoff_metrics[metric] = values
            seed_deltas = []
            for seed in expected_seeds:
                target_seed_value = float(target_runs[seed]["test"][cutoff][metric])
                baseline_seed_value = float(baseline_runs[seed]["test"][cutoff][metric])
                seed_winner = (
                    "target"
                    if target_seed_value > baseline_seed_value
                    else "baseline"
                    if baseline_seed_value > target_seed_value
                    else "tie"
                )
                seed_winners.append(seed_winner)
                seed_deltas.append(
                    {
                        "seed": seed,
                        "target": target_seed_value,
                        "baseline": baseline_seed_value,
                        "target_minus_baseline": target_seed_value - baseline_seed_value,
                        "winner": seed_winner,
                    }
                )
            cutoff_seed_deltas[metric] = seed_deltas
        metrics[cutoff] = cutoff_metrics
        paired_seed_deltas[cutoff] = cutoff_seed_deltas

    target_wins = winners.count("target")
    baseline_wins = winners.count("baseline")
    conclusion = (
        "The target is stronger on every reported EC cutoff and metric."
        if target_wins == len(winners)
        else "The pinned baseline is stronger on every reported EC cutoff and metric."
        if baseline_wins == len(winners)
        else "The target and pinned baseline have mixed aggregate EC probe results."
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_level": "PUBLIC_BENCHMARK_FROZEN_ENCODER_TRANSFER_COMPARISON",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_accessed": True,
        "proteingym_rows_opened": False,
        "benchmark": "TorchProtein EnzymeCommission",
        "benchmark_receipt": target["benchmark_receipt"],
        "target_receipt": {"path": str(target_path), "sha256": sha256_file(target_path)},
        "baseline_receipt": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "evaluation_settings": target["evaluation_settings"],
        "probe": target_result["probe"],
        "records": target_result["records"],
        "metrics": metrics,
        "paired_seed_deltas": paired_seed_deltas,
        "win_counts": {
            "target": target_wins,
            "baseline": baseline_wins,
            "ties": winners.count("tie"),
        },
        "seed_level_win_counts": {
            "target": seed_winners.count("target"),
            "baseline": seed_winners.count("baseline"),
            "ties": seed_winners.count("tie"),
        },
        "conclusion": conclusion,
        "statistical_limitations": (
            "Three shared initialization seeds quantify probe variability and support paired "
            "optimizer-initialization deltas, but they are not independent protein replicates; "
            "the receipt does not claim inferential significance across models or records."
        ),
        "comparability_limitations": target.get("comparability_limitations", []),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(output, receipt)
    return receipt
