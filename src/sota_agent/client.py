import logging
import vertexai
from typing import Optional, Dict, Any
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold

# Import the schema to generate the JSON constraint
from .generator import SOTAEntry
from .paper import ArxivPaper


# Setup Logger
logger = logging.getLogger(__name__)

class GeminiAgentClient:
    def __init__(self, project_id: str, location: str = "us-central1", model_name: str ="gemini-3-flash-preview"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        
        # Use Gemini 2.5 Flash for best instruction following
        self.model = GenerativeModel(self.model_name)
        
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
        3. **evidence**: You MUST provide a direct, verbatim quote from the parsed paper that supports the extracted metric.
        4. **dataset_mentioned**: specific check if {dataset_name} is explicitly tested.

        """


        # Enforce JSON Output using the Pydantic Schema
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=SOTAEntry.model_json_schema(),
            temperature=0.0, # Deterministic output
        )

        # xreate final prompt
        final_prompt = (
            f"{system_prompt}\n\n"
            f"TITLE: {paper.metadata.get('title')}\n\n"
            f"ABSTRACT: {paper.metadata.get('abstract')}\n\n"
            f"MAIN TEXT: {paper.get_text_for_llm(max_chars=50000, include_abstract=False)}"
        )

        # Call LLM
        try:
            response = self.model.generate_content(
                final_prompt,
                generation_config=generation_config
            )
            
            # The API returns a JSON string, which Pydantic parses & validates
            return SOTAEntry.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"LLM Extraction Failed: {e}")
            return None