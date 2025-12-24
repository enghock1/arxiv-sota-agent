import os
import logging
from google import genai
from google.genai import types
from typing import Optional, Dict, Any

# Import the schema to generate the JSON constraint
from .schema import SOTAEntry
from .paper import ArxivPaper
from .pdf_paper import ArxivPdfPaper


# Setup Logger
logger = logging.getLogger(__name__)

class GeminiAgentClient:
    def __init__(self, project_id: str, location: str = "global", model_name: str ="gemini-2.5-flash"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Gemini client with Vertex AI
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        
    def analyze_paper(self, paper: ArxivPaper, config: Dict[str, Any]) -> Optional[SOTAEntry]:
        """
        Analyzes the paper abstract using parameters defined in the YAML config.
        """
        
        # Extract Dynamic Configs
        dataset_name = ", ".join(config['selected_dataset_names'])

        # consider only one performance metric for now (TODO: extend to multiple metrics)
        metric_name = list(config['metrics'].keys())[0]
        metric_desc = config['metrics'][metric_name]
        
        # Extract pipeline stages
        stages_str = ", ".join([f"'{s}'" for s in config['pipeline_stages']])
        
        # Construct the System Prompt
        system_prompt = f"""
        You are an automated Data Extraction Agent. Your goal is to extract state-of-the-Art (SOTA) leaderboard data.

        --- TARGETS ---
        DATASET: {dataset_name}
        METRIC: {metric_name} (description: {metric_desc})

        --- ALLOWED STAGES ---
        You must classify the method into one of these strict Pipeline Stages:
        {stages_str}

        --- INSTRUCTIONS ---
        1. **method_name**: Prefer the acronym. If none, use the shortest distinct name.
        2. **metric_value**: Extract the exact numeric value for {metric_name}.
            - If the text says "85.5%", return 0.855.
            - If not reported, set to null.
            - Sometimes it may not use the exact metric name, infer based on context, and explicitly mention this in evidence.
        3. **evidence**: You MUST provide a direct, verbatim quote from the parsed paper that supports the extracted metric.
        4. **dataset_mentioned**: specific check if {dataset_name} is explicitly tested.

        """

        # Create final prompt
        main_text = paper.get_text_for_llm(max_chars=50000, include_abstract=False)
        final_prompt = (
            f"{system_prompt}\n\n"
            f"TITLE: {paper.metadata.get('title')}\n\n"
            f"ABSTRACT: {paper.metadata.get('abstract')}\n\n"
            f"MAIN TEXT: {main_text}"
        )
        
        # save final_prompt to a text file for debugging
        os.makedirs("data/debug_prompts", exist_ok=True)
        debug_prompt_path = f"data/debug_prompts/{paper.arxiv_id}_prompt.txt"
        with open(debug_prompt_path, "w", encoding="utf-8") as f:
            f.write(final_prompt)

        # Log prompt size for debugging
        logger.info(f"Prompt length: {len(final_prompt)} characters")

        # Enforce JSON Output using the Pydantic Schema
        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SOTAEntry.model_json_schema(),
            temperature=0.0,  # Deterministic output
        )

        # Call LLM with retry logic
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config=generation_config
            )
            
            # The API returns a JSON string, which Pydantic parses & validates
            if response.text is None:
                logger.error("LLM returned no text content")
                return None
            return SOTAEntry.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"LLM Extraction Failed: {e}")
            logger.error(f"Paper ID: {paper.arxiv_id}, Title: {paper.metadata.get('title', 'Unknown')[:50]}")
            return None
    
    def analyze_paper_from_pdf(self, pdf_paper: ArxivPdfPaper, config: Dict[str, Any]) -> Optional[SOTAEntry]:
        """
        Analyzes a PDF paper using Gemini's multimodal capabilities.
        Uploads the full PDF to Gemini for comprehensive analysis.
        
        Args:
            pdf_paper: ArxivPdfPaper object with PDF path
            config: LLM extraction parameters from YAML config
            
        Returns:
            SOTAEntry object with extracted metrics, or None if extraction failed
        """
        
        # Extract Dynamic Configs
        dataset_name = ", ".join(config['selected_dataset_names'])
        
        # Consider only one performance metric for now
        metric_name = list(config['metrics'].keys())[0]
        metric_desc = config['metrics'][metric_name]
        
        # Extract pipeline stages
        stages_str = ", ".join([f"'{s}'" for s in config['pipeline_stages']])
        
        # Get content keywords for relevance checking
        content_keywords = config.get('PARSED_PAPER_SCANNING_PARAMETERS', {}).get('content_keywords', [])
        keywords_str = ", ".join([f"'{kw}'" for kw in content_keywords]) if content_keywords else "N/A"
        
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

4. **evidence**: You MUST provide a direct, verbatim quote from the PDF that supports the extracted metric.

5. **dataset_mentioned**: Specific check if {dataset_name} is explicitly tested or mentioned.

--- PAPER METADATA ---
TITLE: {pdf_paper.metadata.get('title', 'N/A')}
ABSTRACT: {pdf_paper.metadata.get('abstract', 'N/A')}

IMPORTANT: You have access to the full PDF document. Do not truncate your analysis - examine all pages, especially later sections containing results and experiments.
"""
        
        # Save final prompt to a text file for debugging
        os.makedirs("data/debug_prompts_pdf", exist_ok=True)
        debug_prompt_path = f"data/debug_prompts_pdf/{pdf_paper.arxiv_id}_prompt.txt"
        with open(debug_prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)
        
        # Log prompt info
        logger.info(f"Analyzing PDF: {pdf_paper.arxiv_id}")
        
        try:
            # Upload PDF to Gemini and get file URI
            file_uri = pdf_paper.upload_to_gemini(self.client)
            logger.info(f"PDF uploaded: {file_uri}")
            
            # Create multimodal content with PDF file + prompt
            contents = [
                types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
                types.Part.from_text(text=system_prompt)
            ]
            
            # Enforce JSON Output using the Pydantic Schema
            generation_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SOTAEntry.model_json_schema(),
                temperature=0.0,  # Deterministic output
            )
            
            # Call LLM with PDF + prompt
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generation_config
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