import yaml
import json
from pathlib import Path
from typing import Dict, Any, Generator

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def stream_arxiv_data(file_path: Path) -> Generator[Dict, None, None]:
    """
    Reads the massive ArXiv JSON file line-by-line.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    with open(file_path, 'r') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


