# ArXiv SOTA Agent

A research assistant tool that automatically extracts performance metrics from arXiv papers and build structured leaderboards for machine learning research. By combining metadata filtering, PDF parsing, and Large Language Model (LLM) analysis, it attempt to extract unstructured academic papers into structured and queryable datasets.


## Why This Tool?

When tackling a new machine learning problem, practitioners typically start by reading review papers. While review papers provide excellent theoretical overviews and broad context, they often fall short in answering critical engineering questions:

- "Which method is the best approach for the specific problem I am facing?"
- "What is the current State-of-the-Art (SOTA) for this specific dataset benchmark?"
- "How do different approaches compare quantitatively?"
- "Which recently published methods have not yet been covered in older reviews?"

Besides, review papers are static and can quickly become outdated. This tool aims to bridge that gap by mining the arXiv database to automate the extraction, structuring, and ranking of methods based on their reported performance metrics. Instead of manually skimming dozens of papers to perform comparative analysis, the goal of this tool is to systematically extract quantitative metrics to help practitioners to make more informed decisions.


## Overview

This repository mainly performs 4 steps:
1. **Scanning ArXiv metadata** to identify relevant research papers based on keywords.
2. **Downloading and parsing papers** from ArXiv database.
3. **Filtering papers** based on content-specific keywords.
4. **Extracting structured data** using Google's Gemini multimodal model and Pydantic Data Models to analyze paper content and extract:
    - Performance metrics (e.g., accuracy, F1 score)
    - Method classifications (mapped to a pre-specified hierarchical taxonomy)
    - Paper metadata (title, domain, application)
    - Benchmark dataset mentions and evidence citations
    - And more

The output is a structured dataset/leaderboard that can be used for downstream analysis, systematic reviews, and benchmark tracking.


## Disclaimer

This is a personal project designed to help with my own research workflow. While functional, it is still a work in progress and may contain bugs. If you encounter issues or have ideas for new features, please feel free to open an issue or submit a pull request!


## Limitation

- The pipeline relies on 2-step filtering process to reduce the number of candidates before LLM API calls. To avoid exessive costs, consider adjusting keyword filters to keep the list of candidate papers minimal before triggering API calls.
- This repo is designed specifically for arXiv papers. It may not function with other public publication databases.
- Current implementation is optimized for Google Gemini. However, support for other multimodal models can be added by extending the client wrapper class.


## Installation

### Prerequisites
- Google Cloud Platform (GCP) account with access to the Gemini API (Vertex AI).
- Conda environment (optional).


### Setup

1. Clone the repository
    ```bash
    git clone https://github.com/enghock1/arxiv-sota-agent.git
    cd arxiv-sota-agent
    ```

2. Create and activate conda environment (optional but recommended)
    ```bash
    conda env create -f environment.yml
    conda activate arxiv-sota
    ```

3. Install the package
    ```bash
    pip install -e .
    ```

4. Download ArXiv metadata dataset
    Download the latest arXiv metadata dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) (~5GB).
    
    *Unzip and move the `arxiv-metadata-oai-snapshot.json` file to the `data/raw/` directory.*


## Configuration

### Google Cloud Platform (GCP) Variables
Create a `.env` file in the project root directory with your GCP credentials:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

*Create GCP account and get API access from [here](https://aistudio.google.com).*

## Test the main script:

```bash
python scripts/run_pipeline.py --config_yaml config/spurious-correlation.yaml
```


## Project Structure

```
arxiv-sota-agent/
├── analysis/                       
│   └── postprocess-spurious.ipynb  # Notebook for analyzing extracted results
├── config/                        
│   └── spurious-correlation.yaml   # YAML configuration file for specific topic
├── data/
│   ├── raw/                        # store ArXiv metadata from Kaggle
│   ├── sources/                    # Downloaded PDFs from ArXiv database
│   ├── parsed_papers/              # Parsed paper objects
│   └── processed/                  # Output results
├── scripts/
│   ├── run_pipeline.py            # Main pipeline script
├── src/sota_agent/
│   ├── client.py                   # Gemini API client wrapper
│   ├── model/
│   │   ├── schema.py               # Pydantic data schemas
│   │   └── pdf_paper.py            # Paper parsing utilities
│   └── utils.py                    # Helper functions
├── .env                            # Google API credentials (not in repo)
├── environment.yml                 # Conda environment
├── pyproject.toml                  # Python package config
└── README.md
```
