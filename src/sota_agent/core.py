from typing import Optional
from pydantic import BaseModel, Field, field_validator
    

class SOTAEntry(BaseModel):
    paper_title: str = Field(..., description="Title of the research paper.")
    method_name: str = Field(..., description="Name of the method proposed in the paper.")
    pipeline: str = Field(..., description="taxonomy pipeline stage specified in yaml config.")
    strategy: str = Field(..., description="specific algorithmic strategy used in each stage.")
    evidence: str = Field(..., description=" Quote from text supporting the metric.")
    dataset_mentioned: bool
    metric_value: float = Field(None, description="Performance metric. Return -1.0 if not reported.")

    # cleaning function to force consistent formatting on pipeline, strategy, abd metric_value fields
    @field_validator('pipeline', mode='before')
    @classmethod
    def clean_pipeline(cls, v: str) -> str:
        return v.strip().title()


    @field_validator('strategy', mode='before')
    @classmethod
    def clean_strategy(cls, v: str) -> str:
        return v.strip().title()

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