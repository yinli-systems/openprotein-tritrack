from openprotein.provenance import source_tree_sha256


def test_source_digest_changes_with_content_but_ignores_artifacts(tmp_path):
    (tmp_path / "source.txt").write_text("one")
    first = source_tree_sha256(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/ignored.bin").write_bytes(b"ignored")
    assert source_tree_sha256(tmp_path) == first
    (tmp_path / "source.txt").write_text("two")
    assert source_tree_sha256(tmp_path) != first


def test_source_digest_tracks_nested_python_data_package(tmp_path):
    package = tmp_path / "src/openprotein/data"
    package.mkdir(parents=True)
    module = package / "streaming.py"
    module.write_text("version = 1\n")
    first = source_tree_sha256(tmp_path)
    module.write_text("version = 2\n")
    assert source_tree_sha256(tmp_path) != first
