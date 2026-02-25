from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class LLMConfig(BaseModel):
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_steps: int = 16

class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Session identifier for thread memory")
    message: str = Field(..., description="User prompt to process")
    skills_config: Dict[str, bool] = Field(default_factory=dict, description="Enabled/Disabled states for skills")
    llm_config: Optional[LLMConfig] = None
    messages_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous conversation messages (optional, if UI manages it)")

class ToolInfo(BaseModel):
    name: str
    description: str

class SkillInfo(BaseModel):
    name: str
    description: str
    enabled: bool = False
    tools: List[str] = Field(default_factory=list)

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    deep_json: Dict[str, Any] = Field(default_factory=dict, description="Structured log data from Analyzer, Planner, Executor for the UI Inspector")
    error: Optional[str] = None
