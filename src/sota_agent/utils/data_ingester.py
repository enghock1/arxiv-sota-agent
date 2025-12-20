import json
from pathlib import Path
from typing import Dict, Generator


def stream_arxiv_data(file_path: Path) -> Generator[Dict, None, None]:
    """
    Reads the ArXiv JSON file line-by-line.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    with open(file_path, 'r') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


