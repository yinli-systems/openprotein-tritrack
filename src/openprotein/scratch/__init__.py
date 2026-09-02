"""Random-initialized protein language models and pretraining utilities."""

from openprotein.scratch.config import ScratchModelConfig, load_scratch_model_config
from openprotein.scratch.model import ScratchProteinModel

__all__ = ["ScratchModelConfig", "ScratchProteinModel", "load_scratch_model_config"]
