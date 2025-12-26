import sys
import json
from tqdm import tqdm
from typing import List

from sota_agent.model.pdf_paper import ArxivPdfPaper


def filter_papers(config: dict, parsed_papers: List[ArxivPdfPaper], paths: dict) -> List[ArxivPdfPaper]:
    """
    Filter downloaded PDF papers based on content keywords.
    Params:
        config: PARSED_PAPER_SCANNING_PARAMETERS from YAML config.
        parsed_papers: List of downloaded ArxivPdfPaper objects from step 2.
        paths: Dictionary of predetermined file paths.
    Returns:
        List of filtered ArxivPdfPaper objects.
    """

    content_keywords = config.get('content_keywords', [])
    
    if not content_keywords:
        print("\nNo content keywords specified. Using all downloaded PDFs.")
        filtered_papers = parsed_papers
    else:
        print(f"\nFiltering PDFs by keywords: {content_keywords}")
        filtered_papers = []
        for pdf_paper in tqdm(parsed_papers, desc="Scanning PDF content", unit="papers"):
            # Search in extracted text
            pdf_text = pdf_paper.get_raw_text().lower()
            
            if any(kw.lower() in pdf_text for kw in content_keywords):
                filtered_papers.append(pdf_paper)
        
        print(f"PDFs after content filtering: {len(filtered_papers)} / {len(parsed_papers)}")
    
    if not filtered_papers:
        print("No PDFs matched the content keywords. Exiting.")
        sys.exit(0)

    # Dump a preview of the first 10 filtered papers to a JSON file
    if config.get('preview_filtered_papers', False):
        n = 10
        print(f"\nDumping first {min(n, len(filtered_papers))} filtered papers into a preview file...")
        preview_output_path = paths['OUTPUT'] / "filtered_papers_preview.json"
        paths['OUTPUT'].mkdir(parents=True, exist_ok=True)
        with open(preview_output_path, 'w', encoding='utf-8') as f:
            json.dump([paper.to_dict() for paper in filtered_papers[:n]], f, indent=4, ensure_ascii=False)
        print(f"Filtered paper preview saved to {preview_output_path}")

    return filtered_papers