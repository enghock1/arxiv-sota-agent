import os
import argparse
import pandas as pd
from pathlib import Path

from sota_agent.utils import (load_config,
                              get_google_ids_from_dotenv)
from sota_agent import (scan_arxiv_metadata,
                        download_arxiv_papers,
                        filter_papers,
                        analyze_papers)
                        
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
    google_keys = get_google_ids_from_dotenv()
    print(f"Google project ids used: {google_keys}.")

    # load config file
    config = load_config(config_yaml)
    print(f"Loaded yaml file from {config_yaml}.")

    ### Step 1: Metadata scanning phase ###
    scanned_metadata = scan_arxiv_metadata(config['ARXIV_METADATA_SCAN_PARAMETERS'], PATHS)
    ##############################


    ### Step 2: Downloading phase ###
    parsed_papers = download_arxiv_papers(config['ARXIV_DOWNLOAD_PARAMETERS'], scanned_metadata, PATHS)
    #################################


    ### Step 3: PDF content filtering phase ###
    filtered_papers = filter_papers(config['PARSED_PAPER_FILTER_PARAMETERS'], parsed_papers, PATHS)
    ################################


    ### Step 4: LLM extraction phase ###
    results = analyze_papers(google_keys, config['LLM_ANALYSIS_PARAMETERS'], filtered_papers, PATHS)
    ################################


    ### Step 5: Save results to output data/processed directory ###
    PATHS['OUTPUT'].mkdir(parents=True, exist_ok=True)
    
    if results:
        config_fn = os.path.basename(config_yaml).replace(".yaml", "")
        df = pd.DataFrame(results).sort_values(by="Metric", ascending=False)
        output_file = PATHS['OUTPUT'] / f"leaderboard-{config_fn}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    else:
        print("\nNo valid metrics extracted from candidates.")
    #################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA Benchmarking on ArXiv Papers.")
    parser.add_argument("--config_yaml", type=str, default="config/benchmark_config.yaml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    main(Path(args.config_yaml))