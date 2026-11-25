"""Typed messages passed between agents. The schemas are also what Claude must fill (forced tool calls)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Topic = Literal["overview", "financial", "legal", "security", "reputation", "registry"]
Recommendation = Literal["approve", "approve_with_conditions", "reject"]

RISK_FLAGS = (
    "sanctions_match",
    "registry_status",
    "tax_lien",
    "fraud_litigation",
    "litigation",
    "financial_distress",
    "security_incident",
    "regulatory_fine",
    "new_entity",
    "negative_reviews",
    "no_track_record",
    "information_gap",
)


class Task(BaseModel):
    id: str
    topic: Topic
    question: str
    source: Literal["documents", "registry"] = "documents"


class Plan(BaseModel):
    tasks: list[Task] = Field(min_length=1, max_length=10)


class Evidence(BaseModel):
    source_id: str
    title: str
    text: str


class Claim(BaseModel):
    text: str
    source_ids: list[str] = Field(min_length=1, description="IDs of the evidence that supports this claim")


class Finding(BaseModel):
    task_id: str
    topic: Topic
    claims: list[Claim] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="What the evidence could not answer")


class Section(BaseModel):
    topic: Topic
    title: str
    claims: list[Claim]


class Report(BaseModel):
    company: str
    summary: str
    sections: list[Section]
    risk_flags: list[str] = Field(default_factory=list, description=f"Any of: {', '.join(RISK_FLAGS)}")
    recommendation: Recommendation
    conditions: list[str] = Field(default_factory=list)
    rationale: str


class ClaimCheck(BaseModel):
    topic: str
    claim: str
    supported: bool
    score: float
    reason: str


class Critique(BaseModel):
    verdict: Literal["pass", "revise"]
    unsupported: list[ClaimCheck] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    followups: list[Task] = Field(default_factory=list)
    notes: str = ""


class Usage(BaseModel):
    agent: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
