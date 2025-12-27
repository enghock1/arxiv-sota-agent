import sys
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from sota_agent.client import GeminiAgentClient
from sota_agent.model.pdf_paper import ArxivPdfPaper

def analyze_papers(google_keys: Dict[str, str], config: dict, papers: List[ArxivPdfPaper], paths: Dict[str, Path]) -> List[Dict]:
    """
    Analyzes the list of ArxivPdfPaper using LLM and Pydantic Model.
    Params:
        google_keys: Dictionary of Google IDs.
        config: Configuration dictionary for LLM extraction.
        papers: List of ArxivPdfPaper objects to analyze.
        paths: Dictionary of relevant paths.
    Returns:
        results: List of extracted SOTAEntry dictionaries.
    """
    
    # Extract information using Gemini LLM via Vertex AI
    # PDFs are uploaded to Gemini File API and analyzed with multimodal capabilities
    max_llm_calls = config.get('max_llm_calls', -1)
    
    # get model name
    model_name = config.get("model_name", "gemini-2.5-flash")

    # Initialize Gemini Client (using Google AI SDK for file uploads)
    try:
        client = GeminiAgentClient(
            google_api_key=google_keys["GOOGLE_API_KEY"], 
            model_name=model_name
        )
    except Exception as e:
        print(f"Gemini Client Init Failed: {e}")
        sys.exit(1)

    # Apply safety limit
    papers_to_process = papers[:max_llm_calls] if max_llm_calls != -1 else papers
    print(f"\nExtracting from {len(papers_to_process)} papers using {model_name}...")

    # LLM extraction loop
    results = []
    for pdf_paper in tqdm(papers_to_process, desc="Extracting", unit="papers"):
        try:
            # PDF mode: upload PDF to Gemini and analyze
            entry = client.analyze_paper_from_pdf(pdf_paper, config)
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
    paths['OUTPUT'].mkdir(parents=True, exist_ok=True)
    
    if results:
        df = pd.DataFrame(results).sort_values(by="Metric", ascending=False)
        output_file = paths['OUTPUT'] / "leaderboard.csv"
        df.to_csv(output_file, index=False)
        
        print("leaderboard")
        print(df[["Paper Title", "Application", "Domain", "Pipeline Stage", "Strategy", "Metric", "Evidence", "Dataset Mentioned"]].to_markdown(index=False))
        print(f"\nSaved to {output_file}")
    else:
        print("\nNo valid metrics extracted from candidates.")




    return results