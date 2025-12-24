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
                              scan_arxiv_metadata)
from sota_agent.arxiv_download import arxiv_download

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


    ### Step 2: Downloading phase ###
    parsed_papers = arxiv_download(config['ARXIV_DOWNLOAD_PARAMETERS'], candidates, PATHS)
    #################################

    quit()

    ### Step 3: PDF content filtering phase ###
    content_keywords = config.get('PARSED_PAPER_SCANNING_PARAMETERS', {}).get('content_keywords', [])
    
    if not content_keywords:
        print("\nNo content keywords specified. Using all downloaded PDFs.")
        papers_for_llm = parsed_papers
    else:
        print(f"\nFiltering PDFs by keywords: {content_keywords}")
        papers_for_llm = []
        for pdf_paper in tqdm(parsed_papers, desc="Scanning PDF content", unit="papers"):
            # Search in extracted text
            pdf_text = pdf_paper.extract_text_for_filtering().lower()
            
            if any(kw.lower() in pdf_text for kw in content_keywords):
                papers_for_llm.append(pdf_paper)
        
        print(f"PDFs after content filtering: {len(papers_for_llm)} / {len(parsed_papers)}")
    
    if not papers_for_llm:
        print("No PDFs matched the content keywords. Exiting.")
        sys.exit(0)
    
    ################################



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