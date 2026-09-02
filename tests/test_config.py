from pathlib import Path

from openprotein.config import load_model_config


ROOT = Path(__file__).resolve().parents[1]


def test_production_parameter_contract_is_target_scale():
    config = load_model_config(str(ROOT / "configs/model/openprotein-7b.json"))
    assert config.estimated_parameter_count == 7_076_898_432
    assert config.is_target_scale
    assert sum(config.parameter_breakdown().values()) == config.estimated_parameter_count


def test_tiny_config_is_valid_but_not_misreported_as_7b():
    config = load_model_config(str(ROOT / "configs/model/tiny.json"))
    assert not config.is_target_scale


def test_multimodal_1b_parameter_contract_is_exact():
    config = load_model_config(str(ROOT / "configs/model/openprotein-1b-multimodal.json"))
    assert config.estimated_parameter_count == 1_011_246_594
    assert config.parameter_breakdown()["protein_backbone"] == 651_043_254


def test_tritrack_1b_parameter_contract_is_exact():
    config = load_model_config(str(ROOT / "configs/model/openprotein-tritrack-1.2b.json"))
    assert config.estimated_parameter_count == 1_168_564_995
    assert config.parameter_breakdown()["contrastive_temperature"] == 1
