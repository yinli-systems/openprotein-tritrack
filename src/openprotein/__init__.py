"""OpenProtein public API."""

from openprotein.config import ModelConfig, load_model_config
from openprotein.model import OpenProteinModel

__all__ = ["ModelConfig", "OpenProteinModel", "load_model_config"]
__version__ = "0.1.0"
