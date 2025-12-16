from typing import Optional
from pydantic import BaseModel, Field, field_validator

class TaxonomyClassification(BaseModel):

    # create pipeline
    pipeline: str = Field(..., description="taxonomy pipeline stage specified in yaml config.")

    # create strategy
    strategy: str = Field(..., description="specific algorithmic strategy used in each stage.")

    # cleaning function to force consistent formatting on pipeline and strategy fields
    @field_validator('pipeline', mode='before')
    @classmethod
    def clean_pipeline(cls, v: str) -> str:
        return v.strip().title()

    @field_validator('strategy', mode='before')
    @classmethod
    def clean_strategy(cls, v: str) -> str:
        return v.strip().title()
    

class SOTAEntry(BaseModel):
    paper_title: str = Field(..., description="Title of the research paper.")
    method_name: str = Field(..., description="Name of the method proposed in the paper.")
    taxonomy: TaxonomyClassification
    dataset_mentioned: bool
    metric_value: Optional[float] = Field(None, description="Performance metric value achieved by the method.")

    @field_validator("metric_value", mode="before")
    @classmethod
    def normalize_metric(cls, v):
        if v is None:
            return None
        
        # remove percentage sign and convert to float
        if isinstance(v, str):
            v = v.replace("%", "").strip().split()[0]

        # convert to float and normalize
        try:
            val = float(v)
            return val / 100. if val > 1. else val
        except ValueError:
            return None