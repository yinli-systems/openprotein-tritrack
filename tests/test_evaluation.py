import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from openprotein.artifacts import build_artifact_manifest
from openprotein.config import load_model_config
from openprotein.data.tokenizer import EsmcTokenizer
from openprotein.evaluation import (
    aggregate_proteingym_spearman,
    create_evaluation_seal,
    iter_frozen_proteingym_archive_assays,
    paired_masked_marginal_scores,
    parse_mutant,
    paired_bootstrap_proteingym_differences,
    spearman_correlation,
    verify_evaluation_seal,
)
from openprotein.model import OpenProteinModel


ROOT = Path(__file__).resolve().parents[1]


def test_mutation_parser_and_tied_spearman():
    assert parse_mutant("A1G:D3E", "ACD") == [(0, "A", "G"), (2, "D", "E")]
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="does not match"):
        parse_mutant("G1A", "ACD")
    with pytest.raises(ValueError, match="duplicate"):
        parse_mutant("A1C:A1D", "ACD")


def test_evaluation_seal_detects_benchmark_drift(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "state.pt"
    state.write_bytes(b"state")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "step": 7,
                "state_bytes": state.stat().st_size,
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
            }
        )
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"version": "frozen"}))
    config = ROOT / "configs/model/tiny.json"
    tokenizer = ROOT / "tests/fixtures/tiny_tokenizer.json"
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "weights.bin").write_bytes(b"fixture")
    artifact_manifest = tmp_path / "artifact-manifest.json"
    build_artifact_manifest(
        str(artifact_root),
        str(artifact_manifest),
        ["*.bin"],
        {"name": "fixture"},
    )
    seal_path = tmp_path / "seal.json"
    create_evaluation_seal(
        str(checkpoint),
        str(benchmark),
        str(config),
        str(tokenizer),
        str(artifact_manifest),
        str(seal_path),
    )
    assert verify_evaluation_seal(str(seal_path))["status"] == "sealed"
    benchmark.write_text(json.dumps({"version": "changed"}))
    with pytest.raises(RuntimeError, match="benchmark"):
        verify_evaluation_seal(str(seal_path))


def test_paired_masked_marginals_use_deterministic_long_sequence_windows():
    torch.manual_seed(5)
    config = load_model_config(str(ROOT / "configs/model/tiny.json"))
    tokenizer = EsmcTokenizer.from_tokenizer_json(
        str(ROOT / "tests/fixtures/tiny_tokenizer.json")
    )
    model = OpenProteinModel.from_config(config, device=torch.device("cpu"), dtype=torch.float32)
    scores = paired_masked_marginal_scores(
        model,
        tokenizer,
        "A" * 80,
        ["A1C", "A40D", "A80E"],
        torch.device("cpu"),
        positions_per_batch=2,
        max_length=64,
    )
    assert scores.keys() == {"base_esmc", "openprotein"}
    assert all(len(values) == 3 for values in scores.values())
    assert all(torch.isfinite(torch.tensor(values)).all() for values in scores.values())


def test_proteingym_aggregation_is_uniprot_then_selection_type_macro_mean():
    assays = [
        {
            "uniprot_id": "P1",
            "selection_type": "Binding",
            "spearman": {"OpenProtein-7B": 0.8, "ESM2-15B": 0.4},
        },
        {
            "uniprot_id": "P1",
            "selection_type": "Binding",
            "spearman": {"OpenProtein-7B": 0.6, "ESM2-15B": 0.2},
        },
        {
            "uniprot_id": "P2",
            "selection_type": "Activity",
            "spearman": {"OpenProtein-7B": 0.5, "ESM2-15B": 0.7},
        },
    ]
    aggregate = aggregate_proteingym_spearman(assays)
    assert aggregate["selection_type_scores"]["Binding"]["OpenProtein-7B"] == pytest.approx(
        0.7
    )
    assert aggregate["primary_scores"]["OpenProtein-7B"] == pytest.approx(0.6)


def test_paired_bootstrap_is_deterministic_and_stratified_by_selection_type():
    assays = [
        {
            "uniprot_id": "P1",
            "selection_type": "Binding",
            "spearman": {"OpenProtein-7B": 0.8, "ESM2-15B": 0.4},
        },
        {
            "uniprot_id": "P2",
            "selection_type": "Binding",
            "spearman": {"OpenProtein-7B": 0.6, "ESM2-15B": 0.5},
        },
        {
            "uniprot_id": "P3",
            "selection_type": "Activity",
            "spearman": {"OpenProtein-7B": 0.7, "ESM2-15B": 0.2},
        },
    ]
    first = paired_bootstrap_proteingym_differences(
        assays, "OpenProtein-7B", ["ESM2-15B"], 1000, 17
    )
    second = paired_bootstrap_proteingym_differences(
        assays, "OpenProtein-7B", ["ESM2-15B"], 1000, 17
    )
    assert first == second
    assert first["differences"]["ESM2-15B"][
        "percentile_95_percent_interval"
    ][0] > 0


def test_opaque_proteingym_archives_are_verified_then_joined_by_exact_mutants(tmp_path):
    reference = tmp_path / "DMS_substitutions.reference.csv"
    reference.write_text(
        "DMS_id,DMS_filename,UniProt_ID,target_seq,coarse_selection_type\n"
        "DMS_A,assay.csv,P1,ACD,Activity\n"
        "DMS_B,assay_b.csv,P2,ACD,Binding\n"
    )
    assays = tmp_path / "DMS_ProteinGym_substitutions.zip"
    with zipfile.ZipFile(assays, "w") as archive:
        archive.writestr(
            "DMS_ProteinGym_substitutions/assay.csv",
            "mutant,DMS_score\nA1G,1.0\nC2A,2.0\n",
        )
        archive.writestr(
            "DMS_ProteinGym_substitutions/assay_b.csv",
            "mutant,DMS_score\nA1G,2.0\nC2A,1.0\n",
        )
    scores = tmp_path / "zero_shot_substitutions_scores.zip"
    with zipfile.ZipFile(scores, "w") as archive:
        archive.writestr(
            "zero_shot_substitutions_scores/ESM2/15B/DMS_A.csv",
            "mutant,esm2_t48_15B_UR50D\nA1G,-3.0\nC2A,-4.0\n",
        )
        archive.writestr(
            "zero_shot_substitutions_scores/ESM2/15B/DMS_B.csv",
            "mutant,esm2_t48_15B_UR50D\nA1G,-2.0\nC2A,-1.0\n",
        )

    def artifact(path):
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    benchmark = {
        "schema_version": 1,
        "archive_protocol": "proteingym-v1.3-opaque-zip",
        "dataset_root": str(tmp_path),
        "reference_file": artifact(reference),
        "assay_archive": artifact(assays),
        "score_archive": artifact(scores),
        "baselines": {
            "ESM2-15B": {
                "member_prefix": "ESM2/15B",
                "score_column": "esm2_t48_15B_UR50D",
                "directionality": -1,
            }
        },
    }
    loaded = list(iter_frozen_proteingym_archive_assays(benchmark))
    assert len(loaded) == 2
    assert loaded[0]["selection_type"] == "Activity"
    assert loaded[0]["mutants"] == ["A1G", "C2A"]
    assert loaded[0]["baseline_scores"]["ESM2-15B"] == [3.0, 4.0]
    assert [item["assay_id"] for item in iter_frozen_proteingym_archive_assays(
        benchmark, shard_index=0, shard_count=2
    )] == ["DMS_A"]
    assert [item["assay_id"] for item in iter_frozen_proteingym_archive_assays(
        benchmark, shard_index=1, shard_count=2
    )] == ["DMS_B"]

    reference.write_text(reference.read_text() + "DMS_C,c.csv,P3,AAA,Binding\n")
    with pytest.raises(RuntimeError, match="byte count differs"):
        list(iter_frozen_proteingym_archive_assays(benchmark))
