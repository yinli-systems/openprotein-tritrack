import gzip
import hashlib
import json

from openprotein.data.swissprot import build_swissprot_index, iter_swissprot


FLAT = """ID   TEST_HUMAN Reviewed; 6 AA.
AC   P12345; Q99999;
DE   RecName: Full=Test protein {ECO:0000269|PubMed:1};
OS   Homo sapiens (Human).
CC   -!- FUNCTION: Performs a useful test. {ECO:0000269|PubMed:1}
DR   GO; GO:0000001; F:test activity; EXP:UniProtKB.
KW   Reference proteome; Test keyword.
SQ   SEQUENCE   6 AA;  600 MW;  ABC CRC64;
     ACDEFG
//
"""


def test_flat_parser_extracts_sequence_text_and_evidence():
    records = list(iter_swissprot(FLAT.splitlines(keepends=True)))
    assert len(records) == 1
    record = records[0]
    assert record["entry_name"] == "TEST_HUMAN"
    assert record["accession"] == "P12345"
    assert record["sequence"] == "ACDEFG"
    assert "Test protein" in record["text"]
    assert "test activity" in record["text"]
    assert record["evidence_codes"] == ["ECO:0000269"]
    assert "ECO:" not in record["text"]


def test_index_is_content_addressed_and_queryable(tmp_path):
    source = tmp_path / "sprot.dat.gz"
    with gzip.GzipFile(filename=source, mode="wb", mtime=0) as handle:
        handle.write(FLAT.encode())
    raw = source.read_bytes()
    definition = tmp_path / "source.json"
    definition.write_text(
        json.dumps(
            {
                "records": 1,
                "dat_bytes": len(raw),
                "dat_md5": hashlib.md5(raw).hexdigest(),
                "dat_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    database = tmp_path / "annotations.sqlite"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_swissprot_index(
        str(source), str(database), str(manifest_path), str(definition)
    )
    assert manifest["database"]["records"] == 1
    assert len(manifest["database"]["sha256"]) == 64
    assert "Test protein" not in manifest_path.read_text()
