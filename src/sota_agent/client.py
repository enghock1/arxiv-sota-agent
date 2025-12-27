import os
import logging
from google import genai
from google.genai import types
from typing import Optional, Dict, Any

# Import the schema to generate the JSON constraint
from sota_agent.model.schema import SOTAEntry
from sota_agent.model.pdf_paper import ArxivPdfPaper


# Setup Logger
logger = logging.getLogger(__name__)

class GeminiAgentClient:
    def __init__(self, google_api_key: str, location: str = "us-central1", model_name: str ="gemini-2.5-flash"):
        self.google_api_key = google_api_key
        self.location = location
        self.model_name = model_name
        
        # get GOOGLE_API_KEY from google_keys
        self.client = genai.Client(
            api_key=self.google_api_key
        )
    
    def analyze_paper_from_pdf(self, pdf_paper: ArxivPdfPaper, config: Dict[str, Any]) -> Optional[SOTAEntry]:
        """
        Analyzes a PDF paper using Gemini's multimodal capabilities.
        Uploads the full PDF to Gemini for comprehensive analysis.
        Params:
            pdf_paper: ArxivPdfPaper object with PDF path
            config: LLM extraction parameters from YAML config
        Returns:
            SOTAEntry object with extracted metrics, or None if extraction failed
        """
        
        # Extract dataset names
        dataset_name = ", ".join(config['selected_dataset_names'])
        
        # Consider only one performance metric for now
        metric_name = list(config['metrics'].keys())[0]
        metric_desc = config['metrics'][metric_name]
        
        # Extract pipeline stages
        stages_str = ", ".join([f"'{s}'" for s in config['pipeline_stages']])
        
        # Construct the System Prompt for PDF analysis
        system_prompt = f"""
            You are an automated Data Extraction Agent analyzing a research paper PDF to extract State-of-the-Art (SOTA) leaderboard data.

            --- TARGETS ---
            DATASET: {dataset_name}
            METRIC: {metric_name} (description: {metric_desc})

            --- ALLOWED STAGES ---
            You must classify the method into one of these strict Pipeline Stages:
            {stages_str}

            --- INSTRUCTIONS ---
            1. **Scan the ENTIRE PDF document** paying special attention to:
            - Results section
            - Experimental evaluation sections
            - Tables showing performance metrics
            - Figures with performance comparisons

            2. **method_name**: Prefer the acronym. If none, use the shortest distinct name.

            3. **metric_value**: Extract the exact numeric value for {metric_name}.
            - If the text says "85.5%", return 0.855.
            - Look carefully in tables, figures, and text.
            - Sometimes it may not use the exact metric name, infer based on context.
            - If not reported, set to null.

            4. **evidence**: You MUST provide a direct, verbatim quote from the PDF that supports the extracted metric, or mention which figure/table if extracted from a figure or table.

            5. **dataset_mentioned**: Specific check if {dataset_name} is explicitly tested or mentioned.

            --- PAPER METADATA ---
            TITLE: {pdf_paper.metadata.get('title', 'N/A')}
    
            IMPORTANT: You have access to the full PDF document. Do not truncate your analysis - examine all main pages, especially later sections containing results and experiments. You can ignore references and appendices.
        """ 
        
        # Save final prompt to a text file for debugging
        os.makedirs("data/debug_prompts", exist_ok=True)
        debug_prompt_path = f"data/debug_prompts/{pdf_paper.arxiv_id}_prompt.txt"
        with open(debug_prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        
        # Log prompt info
        logger.info(f"Analyzing PDF: {pdf_paper.arxiv_id}")
        
        try:
            # Upload PDF to Gemini and get file object
            uploaded_file = pdf_paper.upload_to_gemini(self.client)
            logger.info(f"PDF uploaded: {uploaded_file}")
            
            # setup output content structure using Pydantic Model
            generator_config = types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=SOTAEntry.model_json_schema(),
                                    temperature=0.0,
                                )

            # Call LLM with PDF file + prompt (using Google AI SDK)
            # Pass uploaded file object directly
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_prompt, uploaded_file],
                config=generator_config
            )
            
            # Parse and validate response
            if response.text is None:
                logger.error("LLM returned no text content")
                return None
            
            return SOTAEntry.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"PDF LLM Extraction Failed: {e}")
            logger.error(f"Paper ID: {pdf_paper.arxiv_id}, Title: {pdf_paper.metadata.get('title', 'Unknown')[:50]}")
            return None