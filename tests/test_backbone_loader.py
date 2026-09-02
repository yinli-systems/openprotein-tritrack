import sys
import types
from pathlib import Path

from openprotein import backbone


def test_native_loader_registers_only_the_requested_package(tmp_path, monkeypatch):
    esm_root = tmp_path / "esm"
    model_root = esm_root / "models" / "esmc"
    model_root.mkdir(parents=True)
    (model_root / "model.py").write_text("class EsmcForMaskedLM: pass\n")
    fake_esm = types.ModuleType("esm")
    fake_esm.__file__ = str(esm_root / "__init__.py")
    monkeypatch.setitem(sys.modules, "esm", fake_esm)
    monkeypatch.delitem(sys.modules, "esm.models.esmc", raising=False)
    monkeypatch.delitem(sys.modules, "esm.models.esmc.model", raising=False)

    loaded = backbone._load_native_esmc_class()
    assert loaded.__name__ == "EsmcForMaskedLM"
    assert "esm.sdk" not in sys.modules
