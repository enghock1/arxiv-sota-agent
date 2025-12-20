import sys
from tqdm import tqdm
from sota_agent.utils.filter import filter_arxiv_metadata
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
            
            if config["MAX_SCAN_LIMIT"] != -1 and scanned_count >= config["MAX_SCAN_LIMIT"]:
                break
                
            if filter_arxiv_metadata(paper, config):
                candidates.append(paper)
                pbar.set_postfix({"Found": len(candidates)})
            
            scanned_count += 1
            
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    return candidates