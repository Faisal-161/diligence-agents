"""The output of one due-diligence run, plus summary metrics."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import Critique, Finding, Plan, Report, Usage


class RunResult(BaseModel):
    company: str
    architecture: str  # "multi_agent" or "single_agent"
    plan: Plan | None = None
    findings: list[Finding] = Field(default_factory=list)
    report: Report | None = None
    critiques: list[Critique] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)
    invented_citations_removed: int = 0
    wall_ms: float = 0.0

    @property
    def cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usage), 6)

    @property
    def tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.usage)

    @property
    def llm_calls(self) -> int:
        return len(self.usage)

    def cost_by_agent(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for u in self.usage:
            out[u.agent] = round(out.get(u.agent, 0) + u.cost_usd, 6)
        return out
