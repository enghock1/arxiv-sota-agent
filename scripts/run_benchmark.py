import sys
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sota_agent.client import GeminiAgentClient
from sota_agent.utils import (load_config,
                              get_google_project_id, 
                              scan_arxiv_papers)
from sota_agent.utils.fetcher import fetch_arxiv_paper

# root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "DATA": PROJECT_ROOT / "data/raw/arxiv-metadata-oai-snapshot.json",
    "OUTPUT": PROJECT_ROOT / "data/processed",
    "PDFS": PROJECT_ROOT / "data/pdfs",
}

def main(config_path: Path):

    # get google project id from .env
    project_id = get_google_project_id()
    print(f"Google project id used: {project_id}.")

    # load config file
    config = load_config(config_path)
    print(f"Loaded config from {config_path}.")

    # filtering according to the specified keywords
    print("\nScanning for papers... ", end="")
    candidates = scan_arxiv_papers(config, PATHS['DATA'])
    print(f"Scan Complete. Candidates Found: {len(candidates)}.")

    # check candidates count
    if not candidates:
        print("No candidates found. Try broadening keywords.")
        sys.exit(0)
    if len(candidates) > 500:
        print("More than 500 candidates found. Are you sure you want to proceed? Consider refining keywords.")

    # Dump a preview of the candidates to a JSON file
    if candidates:
        print(f"\nDumping first {min(50, len(candidates))} candidates to a preview file...")
        preview_candidates = candidates[:50]
        preview_output_path = PATHS['OUTPUT'] / "candidates_preview.json"
        PATHS['OUTPUT'].mkdir(parents=True, exist_ok=True)
        with open(preview_output_path, 'w', encoding='utf-8') as f:
            json.dump(preview_candidates, f, indent=4, ensure_ascii=False)
        print(f"Preview saved to {preview_output_path}")

    # prompt user for confirmation to proceed to LLM extraction
    user_input = input(f"Proceed with LLM extraction? (yes/no) (MAX_LLM_CALLS = {config['MAX_LLM_CALLS']}): ").lower()
    if user_input not in ['yes', 'y']:
        print("Operation cancelled by user.")
        sys.exit(0)
    print("Filtering step complete. Proceeding to LLM extraction...")

    # Apply safety limit
    papers_to_process = candidates[:config['MAX_LLM_CALLS']] if config['MAX_LLM_CALLS'] != -1 else candidates
    print(f"\nRunning Vertex AI on {len(papers_to_process)} candidates...")
    
    # Download PDFs and extract text
    print("\nDownloading papers...")
    for paper in tqdm(papers_to_process, desc="Downloading", unit="papers"):
        arxiv_id = paper.get('id')
        if arxiv_id:
            paper_data = fetch_arxiv_paper(arxiv_id, PATHS['PDFS'], extract_text=True)
            paper['full_text'] = paper_data.get('text')
            paper['pdf_path'] = paper_data.get('pdf_path')
            time.sleep(3)  # Rate limiting - ArXiv recommends 3 seconds between requests



    # Initialize Vertex AI Client
    try:
        client = GeminiAgentClient(project_id=project_id, model_name=config.get("model_name", "gemini-2.5-flash"))
    except Exception as e:
        print(f"Vertex AI Init Failed: {e}")
        sys.exit(1)

    # LLM extraction loop
    results = []
    for paper in tqdm(papers_to_process, desc="Extracting", unit="papers"):
        try:
            # agent call
            entry = client.analyze_paper(paper['title'], paper['abstract'], config)
            print(f"Extracted Entry: {entry}\n")
            
            if entry and entry.metric_value is not None:
                results.append({
                    "Method": entry.method_name,
                    "Pipeline Stage": entry.pipeline,
                    "Strategy": entry.strategy,
                    "Metric": entry.metric_value,
                    "Evidence": entry.evidence,
                    "Dataset Mentioned": entry.dataset_mentioned,
                    "Paper Title": entry.paper_title
                })
                time.sleep(0.1) 
                
        except Exception as e:
            # catch exceptions
            tqdm.write(f"Failed to process '{paper.get('title', '')[:20]}...': {e}")

    # save output
    PATHS['OUTPUT'].mkdir(parents=True, exist_ok=True)
    
    if results:
        df = pd.DataFrame(results).sort_values(by="Metric", ascending=False)
        output_file = PATHS['OUTPUT'] / "leaderboard.csv"
        df.to_csv(output_file, index=False)
        
        print("leaderboard")
        print(df[["Method", "Pipeline Stage", "Strategy", "Dataset Mentioned", "Evidence", "Metric"]].head(20).to_markdown(index=False))
        print(f"\nSaved to {output_file}")
    else:
        print("\nNo valid metrics extracted from candidates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA Benchmarking on ArXiv Papers.")
    parser.add_argument("--config_path", type=str, default="config/benchmark_config.yaml",
                        help="Path to the benchmark configuration YAML file.")
    args = parser.parse_args()

    main(Path(args.config_path))