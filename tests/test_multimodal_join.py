import csv
import json
import sqlite3

from openprotein.data.multimodal import join_uniref_swissprot


def test_join_requires_repid_and_exact_sequence(tmp_path):
    uniref = tmp_path / "uniref"
    (uniref / "train").mkdir(parents=True)
    with (uniref / "train/train-00000-of-00001.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name", "text"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "UniRef50_A",
                "name": "Cluster: A n=1 Tax=Human TaxID=9606 RepID=TEST_HUMAN",
                "text": "ACDE",
            }
        )
        writer.writerow(
            {"id": "UniRef50_B", "name": "Cluster: B; RepID=OTHER_HUMAN", "text": "AAAA"}
        )
        writer.writerow(
            {"id": "UniRef50_C", "name": "Cluster: C; RepID=MISMATCH_HUMAN", "text": "CCCC"}
        )
    (uniref / "manifest.json").write_text(json.dumps({"fixture": True}))

    database = tmp_path / "annotations.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE entries (
                entry_name TEXT PRIMARY KEY, accession TEXT, sequence TEXT,
                sequence_sha256 TEXT, annotation_text TEXT, annotation_sha256 TEXT,
                evidence_codes_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("TEST_HUMAN", "P12345", "ACDE", "seq-a", "Useful protein", "text-a", "[]"),
        )
        connection.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("MISMATCH_HUMAN", "P99999", "GGGG", "seq-b", "Other", "text-b", "[]"),
        )
    annotation_manifest = tmp_path / "swissprot-manifest.json"
    annotation_manifest.write_text(json.dumps({"fixture": True}))

    output = tmp_path / "multimodal"
    manifest = join_uniref_swissprot(
        str(uniref), str(database), str(annotation_manifest), str(output)
    )
    assert manifest["join_counts"]["exact_pairs"] == 1
    assert len(manifest["builder_code_sha256"]) == 64
    assert manifest["join_counts"]["representatives_without_swissprot"] == 1
    assert manifest["join_counts"]["sequence_mismatches"] == 1
    with (output / "train/train-00000-of-00001.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["id"], row["accession"]) for row in rows] == [("UniRef50_A", "P12345")]
    assert "Useful protein" not in (output / "manifest.json").read_text()
