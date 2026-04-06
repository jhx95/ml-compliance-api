from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str = "v1.0.0"
    request_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int