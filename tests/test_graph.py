"""LangGraph tests. Skipped where LangGraph isn't installed; they run in CI."""

import pytest

pytest.importorskip("langgraph")

from langgraph.types import Command  # noqa: E402

from diligence.graph import build_graph, run_graph  # noqa: E402
from diligence.local import run_multi_agent  # noqa: E402
from tests.conftest import COMPANIES, HallucinatingLLM, make_deps  # noqa: E402


@pytest.mark.parametrize("company", COMPANIES)
def test_graph_matches_local_runner(deps, company):
    g = run_graph(deps, company)
    local = run_multi_agent(deps, company)
    assert g.report.recommendation == local.report.recommendation
    assert sorted(g.report.risk_flags) == sorted(local.report.risk_flags)
    assert g.llm_calls == local.llm_calls


def test_graph_pauses_for_human_review_and_resumes(deps):
    graph = build_graph(deps, human_review=True)
    config = {"configurable": {"thread_id": "review-1"}}
    graph.invoke({"company": "Summit Print Co"}, config)
    snap = graph.get_state(config)
    assert snap.next == ("review",)  # paused at the human checkpoint
    assert "report" not in snap.values and not snap.values.get("findings")  # no research yet
    done = graph.invoke(Command(resume={"approved": True}), config)
    assert done["report"]["recommendation"] == "reject"


def test_graph_plan_rejected_ends_without_research(deps):
    res = run_graph(deps, "Summit Print Co", approve=False, thread_id="review-2")
    assert res.report is None and not res.findings


def test_graph_critic_loop_removes_invented_fact():
    deps = make_deps(HallucinatingLLM(make_deps().settings))
    res = run_graph(deps, "Bluepeak Logistics Ltd")
    assert res.critiques[0].verdict == "revise" and res.critiques[-1].verdict == "pass"
    assert not any("50,000" in c.text for s in res.report.sections for c in s.claims)
