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


def filter_paper(paper: Dict, target_dataset_config: Dict) -> bool:
    """
    Paper filtering logic.
    """
    # Category Check
    # ArXiv categories are space-separated strings, e.g., "cs.LG stat.ML"
    categories = paper.get('categories', '')
    if "cs.LG" not in categories and "stat.ML" not in categories:
        return False
        
    # Keyword Check
    abstract = paper.get('abstract', '').lower()
    keywords = [k.lower() for k in target_dataset_config.get('keywords', [])]

    # TODO: title check
    # TODO: date check
    # TODO: must contain doi


    # Returns True if ANY keyword is found
    return any(keyword in abstract for keyword in keywords)