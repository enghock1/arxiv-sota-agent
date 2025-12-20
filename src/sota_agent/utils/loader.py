import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

def get_google_project_id() -> str:
    # get google project id from .env
    load_dotenv()
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    if not project_id:
        print("Error: GOOGLE_PROJECT_ID not set.")
        sys.exit(1)

    return project_id


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