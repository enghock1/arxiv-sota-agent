import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sota_agent.client import GeminiAgentClient
from sota_agent.utils import (load_config,
                              get_google_project_id)
from sota_agent import (scan_arxiv_metadata,
                        filter_papers,
                        download_arxiv_papers)
                        
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
    candidates = scan_arxiv_metadata(config['ARXIV_METADATA_SCANNING_PARAMETERS'], PATHS)
    ##############################


    ### Step 2: Downloading phase ###
    parsed_papers = download_arxiv_papers(config['ARXIV_DOWNLOAD_PARAMETERS'], candidates, PATHS)
    #################################


    ### Step 3: PDF content filtering phase ###
    papers_for_llm = filter_papers(config['PARSED_PAPER_SCANNING_PARAMETERS'], parsed_papers, PATHS)
    ################################

    quit()

    ### Step 4: LLM extraction phase ###
    # Extract information using Gemini LLM via Vertex AI
    # PDFs are uploaded to Gemini File API and analyzed with multimodal capabilities

    max_llm_calls = config['LLM_EXTRACTION_PARAMETERS'].get('max_llm_calls', -1)
    
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
    print(f"\nExtracting from {len(papers_to_process)} papers using {model_name}...")

    # LLM extraction loop
    results = []
    for pdf_paper in tqdm(papers_to_process, desc="Extracting", unit="papers"):
        try:
            # PDF mode: upload PDF to Gemini and analyze
            entry = client.analyze_paper_from_pdf(pdf_paper, config['LLM_EXTRACTION_PARAMETERS'])
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
            title = pdf_paper.metadata.get('title', 'Unknown')
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