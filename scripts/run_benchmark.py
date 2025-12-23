import sys
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sota_agent.client import GeminiAgentClient
from sota_agent.paper import ArxivPaper
from sota_agent.utils import (load_config,
                              get_google_project_id, 
                              scan_arxiv_metadata)
from sota_agent.utils.fetcher import fetch_arxiv_paper

# root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "DATA": PROJECT_ROOT / "data/raw/arxiv-metadata-oai-snapshot.json",
    "OUTPUT": PROJECT_ROOT / "data/processed",
    "SOURCES": PROJECT_ROOT / "data/sources",
    "PARSED_PAPERS": PROJECT_ROOT / "data/parsed_papers",
}

def main(config_yaml: Path):

    # get google project id from .env
    project_id = get_google_project_id()
    print(f"Google project id used: {project_id}.")

    # load config file
    config = load_config(config_yaml)
    print(f"Loaded yaml file from {config_yaml}.")

    ### Step 1: Metadata scanning phase ###
    # filter paper metadata from arxiv dataset based on a set of keywords defined in config.

    # filtering according to the specified keywords
    print("\nScanning for papers... ", end="")
    candidates = scan_arxiv_metadata(config['ARXIV_METADATA_SCANNING_PARAMETERS'], PATHS['DATA'])
    print(f"Scan Complete. Candidates Found: {len(candidates)}.")

    # check candidates count
    if not candidates:
        print("No candidates found. Try broadening keywords.")
        sys.exit(0)
    if len(candidates) > 500:
        print("More than 500 candidates found. Are you sure you want to proceed? Consider refining keywords.")

    # Dump a preview of the candidates to a JSON file
    if config.get('PREVIEW_PARSED_METADATA', False):
        n = 10
        print(f"\nDumping first {min(n, len(candidates))} parsed metadata into a preview file...")
        preview_output_path = PATHS['OUTPUT'] / "parsed_metadata_preview.json"
        PATHS['OUTPUT'].mkdir(parents=True, exist_ok=True)
        with open(preview_output_path, 'w', encoding='utf-8') as f:
            json.dump(candidates[:n], f, indent=4, ensure_ascii=False)
        print(f"Metadata preview saved to {preview_output_path}")

    ##############################


    ### Step 2: Arxiv source downloading phase ###
    # Once the selected papers are identified, download their LaTeX sources via arxiv API.

    # # user confirmation to download arxiv papers
    max_arxiv_calls = config['ARXIV_SOURCE_DOWNLOAD_PARAMETERS'].get('max_arxiv_calls', -1)
    # user_input = input(f"Proceed to paper downloading from Arxiv? (yes/no) (max_arxiv_calls = {max_arxiv_calls}): ").lower()
    # if user_input not in ['yes', 'y']:
    #     print("Operation cancelled.")
    #     sys.exit(0)

    # Apply safety limit
    papers_to_process = candidates[:max_arxiv_calls] if max_arxiv_calls != -1 else candidates
    print(f"\nDownloading and parsing {len(papers_to_process)} papers...")
    
    # Download LaTeX sources and extract text
    print("\nDownloading papers...", end="")
    keep_source = config['ARXIV_SOURCE_DOWNLOAD_PARAMETERS'].get('keep_latex_source', False)
    save_parsed = config['ARXIV_SOURCE_DOWNLOAD_PARAMETERS'].get('save_parsed_papers', True)
    
    if save_parsed:
        PATHS['PARSED_PAPERS'].mkdir(parents=True, exist_ok=True)
    
    # Load failed downloads list to skip them
    failed_downloads_file = PATHS['PARSED_PAPERS'] / "failed_downloads.json"
    failed_downloads = set()
    if failed_downloads_file.exists():
        with open(failed_downloads_file, 'r', encoding='utf-8') as f:
            failed_downloads = set(json.load(f))
    
    parsed_papers = []
    for paper_metadata in tqdm(papers_to_process, desc="Downloading", unit="papers"):
        arxiv_id = paper_metadata.get('id')
        if arxiv_id:
            # Skip if previously failed
            if arxiv_id in failed_downloads:
                tqdm.write(f"Skipping {arxiv_id} (previously failed)")
                continue
            
            # Check if parsed paper already exists
            parsed_file = PATHS['PARSED_PAPERS'] / f"{arxiv_id}.json"
            if parsed_file.exists():
                # Load existing parsed paper as ArxivPaper object
                arxiv_paper = ArxivPaper.from_json(parsed_file)
                # Merge with original metadata from scanning if needed
                if not arxiv_paper.metadata.get('title') and paper_metadata.get('title'):
                    arxiv_paper.metadata['title'] = paper_metadata['title']
                parsed_papers.append(arxiv_paper)
                continue
            
            # If not, download and parse new paper
            try:
                paper_data = fetch_arxiv_paper(arxiv_id, PATHS['PARSED_PAPERS'], PATHS['SOURCES'], keep_source=keep_source)
                latex_text = paper_data.get('text')
                
                # Parse LaTeX into structured sections
                if latex_text:
                    # Merge metadata from scanning with fetched metadata
                    merged_metadata = paper_metadata.copy()
                    if paper_data.get('metadata'):
                        merged_metadata.update(paper_data['metadata'])
                    
                    arxiv_paper = ArxivPaper(
                        arxiv_id=arxiv_id,
                        source_path=paper_data.get('main_tex'),
                        metadata=merged_metadata
                    )
                    arxiv_paper.parse(latex_text)
                    
                    # Save parsed paper to JSON
                    if save_parsed:
                        arxiv_paper.save_to_json(parsed_file)
                
                    parsed_papers.append(arxiv_paper)
                else:
                    # Mark as failed if no text was extracted
                    tqdm.write(f"Failed to extract text from {arxiv_id}")
                    failed_downloads.add(arxiv_id)
                    
            except Exception as e:
                tqdm.write(f"Failed to download {arxiv_id}: {e}")
                failed_downloads.add(arxiv_id)

            time.sleep(3)  # Rate limit for new downloads
    
    # Save updated failed downloads list
    with open(failed_downloads_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(failed_downloads)), f, indent=2)
    print(" Download and parse complete.")
    if save_parsed:
        print(f"Parsed papers saved to {PATHS['PARSED_PAPERS']}")
    ################################



    ### Step 3: Parsed paper scanning phase ###
    # Scan the parsed papers and filter by keywords in full text sections
    
    content_keywords = config.get('PARSED_PAPER_SCANNING_PARAMETERS', {}).get('content_keywords', [])
    
    if not content_keywords:
        print("No content keywords specified. Using all downloaded papers.")
        papers_for_llm = parsed_papers
    else:
        papers_for_llm = []
        for arxiv_paper in tqdm(parsed_papers, desc="Scanning parsed papers", unit="papers"):
            # Search in sections content
            found = False
            for section in arxiv_paper.sections:
                section_content = section.get('content', '').lower()
                if any(kw.lower() in section_content for kw in content_keywords):
                    found = True
                    break
            
            if found:
                papers_for_llm.append(arxiv_paper)
        
        print(f"Papers after content filtering: {len(papers_for_llm)} / {len(parsed_papers)}")
    
    if not papers_for_llm:
        print("No papers matched the content keywords. Exiting.")
        sys.exit(0)
    
    ################################



    ### Step 4: LLM extraction phase ###
    # Extract information using Gemini LLM via Vertex AI

    # user confirmation to download arxiv papers
    max_llm_calls = config['LLM_EXTRACTION_PARAMETERS'].get('max_llm_calls', -1)
    # user_input = input(f"Proceed to extract paper using LLM? (yes/no) (max_llm_calls = {max_llm_calls}): ").lower()
    # if user_input not in ['yes', 'y']:
    #     print("Operation cancelled.")
    #     sys.exit(0)

    # get model name
    model_name = config['LLM_EXTRACTION_PARAMETERS'].get("model_name", "gemini-2.5-flash")

    # Initialize Vertex AI Client
    try:
        client = GeminiAgentClient(project_id=project_id, model_name=model_name)
    except Exception as e:
        print(f"Vertex AI Init Failed: {e}")
        sys.exit(1)

    # Apply safety limit
    papers_to_process = papers_for_llm[:max_llm_calls] if max_llm_calls != -1 else papers_for_llm
    print(f"\nExtracting from {len(papers_to_process)} papers...")

    # # save paper_to_process to a json file for debugging
    # debug_save_path = PATHS['OUTPUT'] / "papers_to_process_debug.json"
    # with open(debug_save_path, 'w', encoding='utf-8') as f:
    #     json.dump([p.to_dict() for p in papers_to_process], f, indent=4, ensure_ascii=False)
    # print(f"Saved papers to process to {debug_save_path} for debugging.")

    # LLM extraction loop
    results = []
    for arxiv_paper in tqdm(papers_to_process, desc="Extracting", unit="papers"):
        try:
            # agent call
            entry = client.analyze_paper(arxiv_paper, config['LLM_EXTRACTION_PARAMETERS'])
            print(f"Extracted Entry: {entry}\n")
            
            if entry and entry.metric_value is not None:
                results.append({
                    "Paper Title": entry.paper_title,
                    "Application": entry.application_field,
                    "Domain": entry.domain,
                    "Pipeline Stage": entry.pipeline,
                    "Strategy": entry.strategy,
                    "Metric": entry.metric_value,
                    "Evidence": entry.evidence,
                    "Dataset Mentioned": entry.dataset_mentioned,
                })
                time.sleep(0.1) 
                
        except Exception as e:
            # catch exceptions
            title = arxiv_paper.metadata.get('title', 'Unknown')
            tqdm.write(f"Failed to process '{title[:20]}...': {e}")

    # save output
    PATHS['OUTPUT'].mkdir(parents=True, exist_ok=True)
    
    if results:
        df = pd.DataFrame(results).sort_values(by="Metric", ascending=False)
        output_file = PATHS['OUTPUT'] / "leaderboard.csv"
        df.to_csv(output_file, index=False)
        
        print("leaderboard")
        print(df[["Paper Title", "Application", "Domain", "Pipeline Stage", "Strategy", "Metric", "Evidence", "Dataset Mentioned"]].to_markdown(index=False))
        print(f"\nSaved to {output_file}")
    else:
        print("\nNo valid metrics extracted from candidates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA Benchmarking on ArXiv Papers.")
    parser.add_argument("--config_yaml", type=str, default="config/benchmark_config.yaml",
                        help="Path to the benchmark configuration YAML file.")
    args = parser.parse_args()

    main(Path(args.config_yaml))