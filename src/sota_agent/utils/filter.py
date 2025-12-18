import datetime
from typing import Dict, Any

def filter_arxiv_metadata(paper: Dict, config: Dict[str, Any]) -> bool:
    """
    Arxiv paper metadata filtering logic.
    """
    filter_config = config.get('arxiv_metadata_filters', {})

    # check categories
    paper_categories = paper.get('categories', '').split()
    allowed_categories = filter_config.get('allowed_categories', ["cs.LG", "stat.ML"])
    if not set(paper_categories).intersection(set(allowed_categories)):
        return False

    # check date
    min_date_str = filter_config.get('min_date')
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
    if filter_config.get('is_published', False):
        if not paper.get('doi'):
            return False

    # is method check
    title = paper.get('title', '').lower()
    exclude_terms = filter_config.get('exclude_title_keywords', [])
    if any(term in title for term in exclude_terms):
        return False

    # abstract keywords check
    abstract_text = paper.get('abstract', '').lower()
    include_keywords = filter_config.get('abstract_keywords', [])
    if any(kw.lower() not in abstract_text for kw in include_keywords):
        return False
    
    return True