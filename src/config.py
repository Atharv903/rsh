"""Configuration loader and schema validator."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dict containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    return cfg
