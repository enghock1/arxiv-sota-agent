from typing import Dict

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
    return all(keyword in abstract for keyword in keywords)