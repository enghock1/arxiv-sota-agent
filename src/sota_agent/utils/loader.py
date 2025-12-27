import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import dotenv_values


def get_google_ids_from_dotenv() -> Dict[str, str]:
    return {k: v for k, v in dotenv_values().items() if v is not None}


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Config Error: {e}")
        sys.exit(1)