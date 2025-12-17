import os
import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from sota_agent.vertex_client import AgentClient
from sota_agent.utils import load_config, stream_arxiv_data, filter_paper
import json

# root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSTANTS = {
    "DATA": PROJECT_ROOT / "data/raw/arxiv-metadata-oai-snapshot.json",
    "OUTPUT": PROJECT_ROOT / "data/processed",
}

def main(config_path: str):

    # Check provate env variables
    load_dotenv()
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    print(f"Google project id used: {project_id}")
    if not project_id:
        print("Error: GOOGLE_PROJECT_ID not set.")
        sys.exit(1)

    print(f"Loading Config from {config_path}...")
    try:
        config = load_config(Path(config_path))
    except Exception as e:
        print(f"Config Error: {e}")
        sys.exit(1)

    # filtering according to the specified keywords
    print(f"\nScanning for '{config['target_dataset']['name']}' papers...")
    candidates = []
    scanned_count = 0
    
    try:
        data_stream = stream_arxiv_data(CONSTANTS['DATA'])
        pbar = tqdm(data_stream, desc="Scanning", unit="papers")
        for paper in pbar:
            
            if config["MAX_SCAN_LIMIT"] != -1 and scanned_count >= config["MAX_SCAN_LIMIT"]:
                break
                
            if filter_paper(paper, config['target_dataset']):
                candidates.append(paper)
                pbar.set_postfix({"Found": len(candidates)})
            
            scanned_count += 1
            
    except FileNotFoundError:
        print(f"Error: Data file not found at {CONSTANTS['DATA']}")
        sys.exit(1)

    print(f"\nScan Complete. Scanned: {scanned_count:,}. Candidates Found: {len(candidates)}")

    if not candidates:
        print("No candidates found. Try broadening keywords.")
        sys.exit(0)
    if len(candidates) > 500:
        print("More than 500 candidates found. Are you sure you want to proceed? Consider refining keywords.")

    # Dump a preview of the candidates to a JSON file
    if candidates:
        print(f"\nDumping first {min(20, len(candidates))} candidates to a preview file...")
        preview_candidates = candidates[:20]
        preview_output_path = CONSTANTS['OUTPUT'] / "candidates_preview.json"
        CONSTANTS['OUTPUT'].mkdir(parents=True, exist_ok=True)
        with open(preview_output_path, 'w', encoding='utf-8') as f:
            json.dump(preview_candidates, f, indent=4, ensure_ascii=False)
        print(f"Preview saved to {preview_output_path}")

    # user confirmation.
    user_input = input(f"Proceed with LLM extraction? (yes/no) (Double check on MAX_LLM_CALLS = {config['MAX_LLM_CALLS']}): ").lower()
    if user_input not in ['yes', 'y']:
        print("Operation cancelled by user.")
        sys.exit(0)
    print("Filtering step complete. Proceeding to LLM extraction...")

    # Apply safety limit
    papers_to_process = candidates[:config['MAX_LLM_CALLS']] if config['MAX_LLM_CALLS'] != -1 else candidates
    print(f"\nRunning Vertex AI on {len(papers_to_process)} candidates...")
    
    # Initialize Vertex AI Client
    try:
        client = AgentClient(project_id=project_id)
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
    CONSTANTS['OUTPUT'].mkdir(parents=True, exist_ok=True)
    
    if results:
        df = pd.DataFrame(results).sort_values(by="Metric", ascending=False)
        output_file = CONSTANTS['OUTPUT'] / "leaderboard.csv"
        df.to_csv(output_file, index=False)
        
        print("\leaderboard")
        print(df[["Method", "Pipeline Stage", "Strategy", "Dataset Mentioned", "Evidence", "Metric"]].head(20).to_markdown(index=False))
        print(f"\nSaved to {output_file}")
    else:
        print("\nNo valid metrics extracted from candidates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA Benchmarking on ArXiv Papers.")
    parser.add_argument("--config_path", type=str, default="config/benchmark_config.yaml",
                        help="Path to the benchmark configuration YAML file.")
    args = parser.parse_args()

    main(args.config_path)