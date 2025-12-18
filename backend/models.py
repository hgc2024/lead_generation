from pydantic import BaseModel
from typing import List, Dict, Optional

class TrainRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42

class TrainResponse(BaseModel):
    accuracy: float
    feature_importance: Dict[str, float]
    message: str

class LeadProfile(BaseModel):
    LeadId: str
    LeadOrigin: str
    LeadSource: str
    TotalTimeSpentOnWebsite: float
    LastActivity: str
    Tags: str
    ConvertedProbability: float
    City: Optional[str] = None

class EmailGenerationRequest(BaseModel):
    lead_profile: LeadProfile

class EmailGenerationResponse(BaseModel):
    email_content: str
    product_recommended: str
