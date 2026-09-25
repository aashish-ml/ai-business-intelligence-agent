from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)
    purpose: str = ""


class AgentPlan(BaseModel):
    intent: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0