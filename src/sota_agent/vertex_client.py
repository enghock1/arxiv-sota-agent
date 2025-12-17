import logging
import vertexai
from typing import Optional, Dict, Any
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold

# Import the schema to generate the JSON constraint
from .core import SOTAEntry


# Setup Logger
logger = logging.getLogger(__name__)

class AgentClient:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        
        # Use Gemini 2.5 Flash for best instruction following
        self.model = GenerativeModel("gemini-2.5-flash")
        
    def analyze_paper(self, title: str, abstract: str, config: Dict[str, Any]) -> Optional[SOTAEntry]:
        """
        Analyzes the paper abstract using parameters defined in the YAML config.
        """
        
        # Extract Dynamic Configs
        dataset_name = config['target_dataset']['name']
        metric_name = config['extraction_goals']['primary_metric']
        metric_desc = config['extraction_goals']['primary_metric_description']
        
        # Extract Valid Stages (The Strict Constraints)
        # We join them into a string to show the LLM its options
        valid_stages = config['pipeline_stages']
        stages_str = ", ".join([f"'{s}'" for s in valid_stages])
        
        # Construct the System Prompt
        system_prompt = f"""
        You are an automated Data Extraction Agent. Your goal is to extract state-of-the-Art (SOTA) leaderboard data.

        --- TARGETS ---
        DATASET: {dataset_name}
        METRIC: {metric_name} ({metric_desc})

        --- ALLOWED STAGES ---
        You must classify the method into one of these strict Pipeline Stages:
        {stages_str}

        --- INSTRUCTIONS ---
        1. **method_name**: Prefer the acronym. If none, use the shortest distinct name.
        2. **metric_value**: Extract the exact numeric value for {metric_name}.
            - If the text says "85.5%", return 0.855.
            - If not reported, set to null.
        3. **evidence**: You MUST provide a direct, verbatim quote from the abstract that supports the extracted metric.
        4. **dataset_mentioned**: specific check if {dataset_name} is explicitly tested.

        """


        # 4. Enforce JSON Output using the Pydantic Schema
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=SOTAEntry.model_json_schema(),
            temperature=0.0, # Deterministic output
        )
        
        # Safety Settings (Optional: Prevent blocking on harmless academic text)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\nTITLE:\n{title}\n\nABSTRACT:\n{abstract}",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # The API returns a JSON string, which Pydantic parses & validates
            return SOTAEntry.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"LLM Extraction Failed: {e}")
            return None