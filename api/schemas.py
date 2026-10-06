from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="Message text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Classification label: 'spam' or 'ham'")
    is_spam: bool = Field(..., description="True if spam, False if ham")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model_version: str = Field(..., description="Model version used for prediction")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    version: str
    algorithm: str
    features: str
    training_dataset: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    created_at: str