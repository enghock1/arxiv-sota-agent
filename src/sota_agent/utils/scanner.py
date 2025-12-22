import sys
import datetime
from tqdm import tqdm
from typing import Dict, Any
from sota_agent.utils.data_ingester import stream_arxiv_data


def scan_arxiv_metadata(config, data_path) -> list:
    """
    Scans the ArXiv dataset for papers matching the filtering criteria.
    """
    candidates = []
    scanned_count = 0

    try:
        data_stream = stream_arxiv_data(data_path)
        pbar = tqdm(data_stream, desc="Scanning", unit="papers")
        for paper in pbar:
            
            if config["max_metadata_scan_limit"] != -1 and scanned_count >= config["max_metadata_scan_limit"]:
                break
                
            if filter_arxiv_metadata(paper, config):
                candidates.append(paper)
                pbar.set_postfix({"Found": len(candidates)})
            
            scanned_count += 1
            
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    return candidates


def filter_arxiv_metadata(paper: Dict, config: Dict[str, Any]) -> bool:
    """
    Arxiv paper metadata filtering logic.
    """

    # check categories
    paper_categories = paper.get('categories', '').split()
    allowed_categories = config.get('allowed_categories', ["cs.LG", "stat.ML"])
    if not set(paper_categories).intersection(set(allowed_categories)):
        return False

    # check date
    min_date_str = config.get('min_date')
    if min_date_str:
        paper_date_str = paper.get('update_date')
        if paper_date_str:
            try:
                paper_date = datetime.datetime.fromisoformat(paper_date_str.replace('Z', '+00:00'))
                min_date = datetime.datetime.fromisoformat(min_date_str.replace('Z', '+00:00'))
                if paper_date < min_date:
                    return False
            except (ValueError, AttributeError):
                return False

    # published check
    if config.get('is_published', False):
        if not paper.get('doi'):
            return False

    # is method check
    title = paper.get('title', '').lower()
    exclude_terms = config.get('exclude_title_keywords', [])
    if any(term in title for term in exclude_terms):
        return False

    # abstract and title keywords check
    abstract_text = paper.get('abstract', '').lower()
    title_text = paper.get('title', '').lower()
    include_keywords = config.get('title_abstract_keywords', [])
    if include_keywords:
        # Check if keywords appear in either abstract or title
        if not any(kw.lower() in abstract_text or kw.lower() in title_text for kw in include_keywords):
            return False
    
    return True