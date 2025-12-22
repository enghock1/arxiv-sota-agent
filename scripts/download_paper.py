"""
Script to download, parse, and save ArXiv papers from LaTeX source.

Usage:
    python scripts/download_paper.py 2301.12345
    python scripts/download_paper.py 2301.12345 1706.03762 2104.08821
"""

import sys
import argparse
from pathlib import Path

from sota_agent.paper import ArxivPaper
from sota_agent.utils.fetcher import fetch_arxiv_paper


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "DATA": PROJECT_ROOT / "data/raw/arxiv-metadata-oai-snapshot.json",
    "OUTPUT": PROJECT_ROOT / "data/processed",
    "SOURCES": PROJECT_ROOT / "data/sources",
    "PARSED_PAPERS": PROJECT_ROOT / "data/parsed_papers",
}


def download_and_parse_paper(arxiv_id: str) -> bool:
    """
    Download LaTeX source, parse, and save a single paper.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.12345")
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing: {arxiv_id}")
    print(f"{'='*60}")
    
    # Step 1: Download LaTeX source and metadata
    print("\n[1/3] Downloading LaTeX source and metadata...")
    result = fetch_arxiv_paper(arxiv_id, PATHS['PARSED_PAPERS'], PATHS['SOURCES'])
    
    if not result['text']:
        print(f"Failed to download or extract LaTeX from {arxiv_id}")
        return False
    
    print(f"Downloaded source: {result['source_dir']}")
    print(f"   Main file: {result['main_tex']}")
    print(f"   LaTeX length: {len(result['text']):,} characters")
    
    # Display metadata
    if result['metadata']:
        print("\nMetadata:")
        print(f"   Title: {result['metadata'].get('title', 'N/A')}")
        print(f"   Authors: {len(result['metadata'].get('authors', []))} author(s)")
        print(f"   Categories: {', '.join(result['metadata'].get('categories', []))}")
    
    # Step 2: Parse into sections
    print("\n[2/3] Parsing sections from LaTeX...")
    paper = ArxivPaper(
        arxiv_id=arxiv_id,
        source_path=Path(result['main_tex']) if result['main_tex'] else None,
        metadata=result['metadata']
    )
    paper.parse(result['text'])
    
    print(f"Parsed {len(paper.sections)} sections:")
    for i, section in enumerate(paper.sections[:5], 1):
        title = section['title']
        content_preview = section['content'][:80].replace('\n', ' ')
        print(f"   {i}. {title}")
        print(f"      '{content_preview}...'")
    if len(paper.sections) > 5:
        print(f"   ... and {len(paper.sections) - 5} more sections")
    
    # Step 3: Save to JSON
    print("\n[3/3] Saving parsed paper...")
    output_path = PATHS['PARSED'] / f"{arxiv_id}.json"
    paper.save_to_json(output_path)
    
    print(f"Saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Successfully processed {arxiv_id}")
    print(f"   - Source: {result['source_dir']}")
    print(f"   - Sections: {len(paper.sections)}")
    print(f"   - Metadata: {len(result['metadata'])} fields")
    print(f"{'='*60}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download LaTeX source, parse, and save ArXiv papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download single paper
    python scripts/download_paper.py 2301.12345
    
    # Download multiple papers
    python scripts/download_paper.py 2301.12345 1706.03762 2104.08821
        """
    )
    parser.add_argument(
        'arxiv_ids',
        nargs='+',
        help='ArXiv paper IDs (e.g., 2301.12345)'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    PATHS['SOURCES'].mkdir(parents=True, exist_ok=True)
    PATHS['PARSED'].mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading and parsing {len(args.arxiv_ids)} paper(s)...")
    print(f"Sources directory: {PATHS['SOURCES']}")
    print(f"Parsed directory: {PATHS['PARSED']}")
    
    # Process each paper
    successful = 0
    failed = 0
    
    for arxiv_id in args.arxiv_ids:
        try:
            if download_and_parse_paper(arxiv_id):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(args.arxiv_ids)}")
    print(f"{'='*60}\n")
    
    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
