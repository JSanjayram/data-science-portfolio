from pydantic import BaseModel
from typing import Optional, Dict, List

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    query: str
    predicted_intent: str
    confidence: float
    response: str
    status: str
    user_id: str

class FeedbackRequest(BaseModel):
    query: str
    predicted_intent: str
    correct_intent: str
    user_id: Optional[str] = "default"

class StatsResponse(BaseModel):
    total_queries_processed: int
    resolved_queries: int
    resolution_rate: str
    average_confidence: str
    active_users: int