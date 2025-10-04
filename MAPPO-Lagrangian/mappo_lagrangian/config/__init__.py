# Config package initialization
from .async_config import AsyncConfig, HIERARCHICAL_CONFIG, IMPORTANCE_CONFIG, PROGRESSIVE_CONFIG

# Import get_config from the parent config module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ..config import get_config
except ImportError:
    # Fallback: try direct import from config.py
    import importlib.util
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    get_config = config_module.get_config

__all__ = ['AsyncConfig', 'HIERARCHICAL_CONFIG', 'IMPORTANCE_CONFIG', 'PROGRESSIVE_CONFIG', 'get_config']