import json
import time
from tqdm import tqdm
from typing import List, Dict, Any

from sota_agent.model.pdf_paper import ArxivPdfPaper
from sota_agent.utils.pdf_fetcher import fetch_paper_from_arxiv


def download_arxiv_papers(config: Dict[str, Any], candidates: List[Dict[str, Any]], paths: Dict[str, Any]):
    """
    Function to download ArXiv papers as PDFs and save as ArxivPdfPaper objects.
    Params:
        config: ARXIV_DOWNLOAD_PARAMETERS from YAML config.
        candidates: List of candidate paper metadata dicts from step 1.
        paths: Dictionary of predetermined file paths.
    Returns:
        List of downloaded and parsed ArxivPdfPaper objects.
    """

    # File to track failed downloads
    parsed_pdf_path = paths['PARSED_PAPERS'] 
    source_pdf_path = paths['SOURCES']
    failed_downloads_file = parsed_pdf_path / "failed_pdf_downloads.json"

    max_pdf_calls = config.get('max_download_calls', -1)
    papers_to_process = candidates[:max_pdf_calls] if max_pdf_calls != -1 else candidates
    print(f"\\nDownloading {len(papers_to_process)} PDFs...")
        
    # get params
    keep_pdf = config.get('save_files', False)
    save_parsed = config.get('save_parsed_papers', True)
    
    # Make sure directories exist
    if save_parsed:
        parsed_pdf_path.mkdir(parents=True, exist_ok=True)
    if keep_pdf:
        source_pdf_path.mkdir(parents=True, exist_ok=True)

    # Load failed downloads list to skip them
    failed_downloads = set()
    if failed_downloads_file.exists():
        with open(failed_downloads_file, 'r', encoding='utf-8') as f:
            failed_downloads = set(json.load(f))
    
    pdf_papers = []
    for paper_metadata in tqdm(papers_to_process, desc="Downloading PDFs", unit="papers"):
        arxiv_id = paper_metadata.get('id')
        if arxiv_id:
            # Skip if previously failed
            if arxiv_id in failed_downloads:
                tqdm.write(f"Skipping {arxiv_id} (previously failed)")
                continue
            
            # Check if parsed PDF paper already exists
            parsed_file = parsed_pdf_path / f"{arxiv_id}.json"
            if parsed_file.exists():
                # Load existing PDF paper
                pdf_paper = ArxivPdfPaper.from_json(parsed_file)
                # Merge with original metadata from scanning if needed
                if not pdf_paper.metadata.get('title') and paper_metadata.get('title'):
                    pdf_paper.metadata['title'] = paper_metadata['title']
                pdf_papers.append(pdf_paper)
                continue
            
            # If not, download and create new PDF paper
            try:
                pdf_paper = fetch_paper_from_arxiv(arxiv_id, paper_metadata, source_pdf_path, keep_pdf=keep_pdf)
                
                if pdf_paper:
                    # Save parsed PDF paper to JSON
                    if save_parsed:
                        pdf_paper.save_to_json(parsed_file)
                    
                    pdf_papers.append(pdf_paper)
                else:
                    # Mark as failed if download failed
                    tqdm.write(f"Failed to download PDF {arxiv_id}")
                    failed_downloads.add(arxiv_id)
                    
            except Exception as e:
                tqdm.write(f"Failed to process PDF {arxiv_id}: {e}")
                failed_downloads.add(arxiv_id)

            time.sleep(3)  # Rate limit for new downloads
    
    # Save updated failed downloads list
    with open(failed_downloads_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(failed_downloads)), f, indent=2)
    print(" PDF download complete.")
    if save_parsed:
        print(f"Parsed PDF papers saved to {parsed_pdf_path}")
    
    return pdf_papers