from typing import Literal
from pydantic import BaseModel, Field, field_validator

class SOTAEntry(BaseModel):
    paper_title: str = Field(..., description="Title of the research paper.")
    application_field: str = Field(..., description="Application field of the research (e.g., healthcare, materials science, theory, general).")
    domain: str = Field(..., description="Domain of the research (e.g., Computer Vision, NLP).")
    paper_type: Literal["Method", "Theoretical", "Survey", "Benchmark", "Analysis", "Position"] = Field(
        ..., 
        description="Type of research paper. Must be one of: Method, Theoretical, Survey, Benchmark, Analysis, Position."
    )
    taxonomy_level_1: str = Field(..., description="Level 1 taxonomy specified in yaml config.")
    taxonomy_level_2: str = Field(..., description="Level 2 taxonomy specified in yaml config.")
    method: str = Field(..., description="Name of the algorithmic method or approach proposed in the paper.")
    metric_value: float = Field(..., description="Performance metric. Return -1.0 if not reported.")
    evidence: str = Field(..., description=" Quote from text supporting the metric.")
    dataset_mentioned: bool = Field(..., description="Indicates if the dataset is mentioned in the paper.")

    # cleaning function to force consistent formatting on pipeline, strategy, abd metric_value fields
    @field_validator('paper_title', mode='before')
    @classmethod
    def clean_paper_title(cls, v: str) -> str:
        return v.strip().title()
    
    @field_validator('domain', mode='before')
    @classmethod
    def clean_domain(cls, v: str) -> str:
        return v.strip()
    
    @field_validator('application_field', mode='before')
    @classmethod
    def clean_application_field(cls, v: str) -> str:
        return v.strip().title()
    
    @field_validator('evidence', mode='before')
    @classmethod
    def clean_evidence(cls, v: str) -> str:
        return v.strip()

    @field_validator('taxonomy_level_1', mode='before')
    @classmethod
    def clean_taxonomy_level_1(cls, v: str) -> str:
        return v.strip()

    @field_validator('taxonomy_level_2', mode='before')
    @classmethod
    def clean_taxonomy_level_2(cls, v: str) -> str:
        return v.strip()

    @field_validator('method', mode='before')
    @classmethod
    def clean_method(cls, v: str) -> str:
        return v.strip()

    @field_validator("metric_value", mode="before")
    @classmethod
    def normalize_metric(cls, v):
        if v is None:
            return -1.0
        
        # remove percentage sign and convert to float
        if isinstance(v, str):
            v = v.replace("%", "").strip()
            if not v.replace('.', '', 1).isdigit():
                return -1.0

        # convert to float and normalize
        try:
            val = float(v)
            if val < 0: 
                return -1.0
            return val / 100. if val > 1. else val
        except ValueError:
            return -1.0