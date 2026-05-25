from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Schema untuk request POST /predict."""
    code: str
    model: str  # "codebert" | "logistictfidf" | "logisticwordembedding"


class PredictResponse(BaseModel):
    """Schema untuk response POST /predict."""
    verdict: str       # "human" | "ai"
    confidence: float  # 0-100
    model: str         # Display name, e.g. "CodeBERT"
