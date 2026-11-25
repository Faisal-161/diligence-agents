"""The LangGraph version of the multi-agent flow.

    START -> planner -> review --(Send per task, in parallel)--> researcher -> analyst -> critic
                          |                                          ^            |   |
                          +-- plan rejected --> END                  |            |   +-- pass / max rounds --> END
                                                                     +-- follow-ups (Send) --+
                                                    analyst <-- revise without new research --+

With `human_review=True`, the `review` node pauses the graph with `interrupt()` until a person
approves (or edits) the plan; resume with `Command(resume={"approved": True})`. State is kept by a
checkpointer, so the pause can last as long as needed.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from . import agents
from .agents import Deps
from .models import Critique, Finding, Plan, Report, Task, Usage
from .result import RunResult

try:  # newer LangGraph
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # pragma: no cover - older LangGraph
    from langgraph.checkpoint.memory import MemorySaver as _Saver


class State(TypedDict, total=False):
    company: str
    plan: dict
    approved: bool
    findings: Annotated[list[dict], operator.add]
    attempted: Annotated[list[str], operator.add]
    report: dict
    critique: dict
    critiques: Annotated[list[dict], operator.add]
    revision: int
    usage: Annotated[list[dict], operator.add]
    log: Annotated[list[str], operator.add]
    removed: Annotated[list[int], operator.add]


class ResearchInput(TypedDict):
    company: str
    task: dict


def build_graph(deps: Deps, human_review: bool = False, checkpointer: Any | None = None):
    def planner(state: State) -> dict:
        plan, usage = agents.plan(deps, state["company"])
        return {
            "plan": plan.model_dump(),
            "usage": [usage.model_dump()],
            "revision": 0,
            "log": [f"planner: {len(plan.tasks)} tasks"],
        }

    def review(state: State) -> dict:
        if not human_review:
            return {"approved": True}
        decision = interrupt({"company": state["company"], "plan": state["plan"]})
        if not decision.get("approved"):
            return {"approved": False, "log": ["human: plan rejected"]}
        plan = Plan.model_validate(decision.get("plan") or state["plan"])
        return {"approved": True, "plan": plan.model_dump(), "log": ["human: plan approved"]}

    def fan_out(state: State):
        if not state.get("approved"):
            return END
        return [Send("researcher", {"company": state["company"], "task": t}) for t in state["plan"]["tasks"]]

    def researcher(inp: ResearchInput) -> dict:
        task = Task.model_validate(inp["task"])
        finding, usage, removed = agents.research(deps, inp["company"], task)
        return {
            "findings": [finding.model_dump()],
            "usage": [usage.model_dump()],
            "attempted": [task.topic],
            "removed": [removed],
            "log": [f"researcher[{task.id}:{task.topic}] {len(finding.claims)} claims"],
        }

    def analyst(state: State) -> dict:
        findings = [Finding.model_validate(f) for f in state["findings"]]
        crit = Critique.model_validate(state["critique"]) if state.get("critique") else None
        report, usage = agents.analyze(deps, state["company"], findings, crit)
        return {
            "report": report.model_dump(mode="json"),
            "usage": [usage.model_dump()],
            "log": [f"analyst(round {state.get('revision', 0)}): {report.recommendation}"],
        }

    def critic(state: State) -> dict:
        report = Report.model_validate(state["report"])
        crit, usage = agents.critique(deps, state["company"], report, set(state.get("attempted", [])))
        r = state.get("revision", 0)
        return {
            "critique": crit.model_dump(mode="json"),
            "critiques": [crit.model_dump(mode="json")],
            "usage": [usage.model_dump()],
            "revision": r + 1,
            "log": [f"critic(round {r}): {crit.verdict}, {len(crit.unsupported)} unsupported"],
        }

    def after_critic(state: State):
        crit = state["critique"]
        if crit["verdict"] == "pass" or state["revision"] > deps.settings.max_revisions:
            return END
        if crit["followups"]:
            return [Send("researcher", {"company": state["company"], "task": t}) for t in crit["followups"]]
        return "analyst"

    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("review", review)
    g.add_node("researcher", researcher)
    g.add_node("analyst", analyst)
    g.add_node("critic", critic)
    g.add_edge(START, "planner")
    g.add_edge("planner", "review")
    g.add_conditional_edges("review", fan_out, ["researcher", END])
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", after_critic, ["researcher", "analyst", END])
    if checkpointer is None and human_review:
        checkpointer = _Saver()
    return g.compile(checkpointer=checkpointer)


def to_result(state: dict, wall_ms: float) -> RunResult:
    return RunResult(
        company=state["company"],
        architecture="multi_agent",
        plan=Plan.model_validate(state["plan"]) if state.get("plan") else None,
        findings=[Finding.model_validate(f) for f in state.get("findings", [])],
        report=Report.model_validate(state["report"]) if state.get("report") else None,
        critiques=[Critique.model_validate(c) for c in state.get("critiques", [])],
        usage=[Usage.model_validate(u) for u in state.get("usage", [])],
        log=state.get("log", []),
        invented_citations_removed=sum(state.get("removed", [])),
        wall_ms=wall_ms,
    )


def run_graph(deps: Deps, company: str, approve: bool | None = None, thread_id: str = "run") -> RunResult:
    """Run end to end. `approve=None` skips human review; True/False answers the review pause."""
    t0 = time.perf_counter()
    graph = build_graph(deps, human_review=approve is not None)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    state = graph.invoke({"company": company}, config)
    if approve is not None:
        state = graph.invoke(Command(resume={"approved": approve}), config)
    return to_result(state, round((time.perf_counter() - t0) * 1000, 1))
