import pytest

from diligence import agents
from diligence.local import PlanRejected, run_multi_agent, run_single_agent
from diligence.models import Task
from diligence.tools import support_score, words
from tests.conftest import COMPANIES, HallucinatingLLM, RecklessAnalystLLM, make_deps


def test_stemming_matches_plurals():
    assert words("lawsuits reviews liens") == words("lawsuit review lien")


def test_registry_fuzzy_lookup(deps):
    assert deps.registry.lookup("Bluepeak Logistics")["slug"] == "bluepeak-logistics"
    assert deps.registry.lookup("Totally Unknown Widgets") is None


def test_planner_covers_required_topics(deps):
    plan, usage = agents.plan(deps, "Bluepeak Logistics Ltd")
    assert set(agents.REQUIRED_TOPICS) <= {t.topic for t in plan.tasks}
    assert usage.agent == "planner" and usage.input_tokens > 0


def test_researcher_cites_only_given_evidence(deps):
    task = Task(id="t1", topic="financial", question="Revenue and profitability of Orion Data Labs Inc.")
    finding, _, removed = agents.research(deps, "Orion Data Labs Inc.", task)
    assert finding.claims and removed == 0
    assert all(i.startswith("orion-data-labs/") for c in finding.claims for i in c.source_ids)


@pytest.mark.parametrize(
    "company, expected",
    [
        ("Bluepeak Logistics Ltd", "approve"),
        ("Northwind Cloud Services LLC", "approve_with_conditions"),
        ("Summit Print Co", "reject"),
        ("Orion Data Labs Inc.", "approve_with_conditions"),
        ("Greenleaf Facilities Co.", "approve"),
        ("Zephyr Consulting LLC", "reject"),
    ],
)
def test_multi_agent_recommendations(deps, company, expected):
    res = run_multi_agent(deps, company)
    assert res.report.recommendation == expected
    assert res.critiques[-1].verdict == "pass"


def test_dismissed_lawsuit_is_not_a_risk(deps):
    res = run_multi_agent(deps, "Greenleaf Facilities Co.")
    legal = " ".join(c.text for s in res.report.sections if s.topic == "legal" for c in s.claims)
    assert "dismissed" in legal and not {"litigation", "fraud_litigation"} & set(res.report.risk_flags)


def test_critic_catches_invented_fact_and_analyst_removes_it():
    deps = make_deps(HallucinatingLLM(make_deps().settings))
    res = run_multi_agent(deps, "Bluepeak Logistics Ltd")
    assert res.invented_citations_removed >= 1  # citation to a document it was never given
    first = res.critiques[0]
    assert first.verdict == "revise"
    assert any("50,000" in c.claim for c in first.unsupported)
    final_claims = [c.text for s in res.report.sections for c in s.claims]
    assert not any("50,000" in t for t in final_claims)
    assert res.critiques[-1].verdict == "pass" and len(res.critiques) == 2


def test_policy_blocks_approval_with_hard_risk_flags():
    deps = make_deps(RecklessAnalystLLM(make_deps().settings))
    res = run_multi_agent(deps, "Summit Print Co")
    assert len(res.critiques) == deps.settings.max_revisions + 1  # never passed
    assert "require 'reject'" in res.critiques[-1].notes


def test_wrong_company_citation_is_flagged(deps):
    report = run_multi_agent(deps, "Bluepeak Logistics Ltd").report
    sec = report.sections[0]
    bad = sec.claims[0].model_copy(
        update={
            "text": "In November 2025 the state recorded a tax lien of 96,000 dollars.",
            "source_ids": ["summit-print/tax"],
        }
    )
    report = report.model_copy(update={"sections": [sec.model_copy(update={"claims": [bad]}), *report.sections[1:]]})
    checks, _, _ = agents.verify(deps, "Bluepeak Logistics Ltd", report)
    assert checks[0].reason == "cites a source about a different company"


def test_missing_area_is_reported_as_gap(deps):
    res = run_multi_agent(deps, "Zephyr Consulting LLC")  # no financial information exists
    assert "information_gap" in res.report.risk_flags


def test_plan_approval_hook(deps):
    def drop_security(plan):
        return plan.model_copy(update={"tasks": [t for t in plan.tasks if t.topic != "security"]})

    res = run_multi_agent(deps, "Northwind Cloud Services LLC", approve_plan=drop_security)
    assert "security" not in {f.topic for f in res.findings}
    with pytest.raises(PlanRejected):
        run_multi_agent(deps, "Northwind Cloud Services LLC", approve_plan=lambda p: None)


def test_parallel_and_sequential_agree(deps):
    a = run_multi_agent(deps, "Orion Data Labs Inc.", parallel=True).report
    b = run_multi_agent(deps, "Orion Data Labs Inc.", parallel=False).report
    assert a.model_dump() == b.model_dump()


def test_usage_and_cost_tracked_per_agent(deps):
    res = run_multi_agent(deps, "Bluepeak Logistics Ltd")
    by_agent = res.cost_by_agent()
    assert {"planner", "researcher", "analyst", "critic"} <= set(by_agent)
    assert res.llm_calls == 1 + 6 + 1 + 1 and res.cost_usd > 0


def test_single_agent_baseline_runs(deps):
    res = run_single_agent(deps, "Bluepeak Logistics Ltd")
    assert res.llm_calls == 1 and res.report is not None


def test_benchmark_gate():
    from eval.run_eval import run

    s = run(make_deps().settings)["summary"]
    assert s["multi_agent"]["recommendation_accuracy"] == 1.0
    assert s["multi_agent"]["wrong_entity_citations"] == 0
    assert s["multi_agent"]["claim_support_rate"] == 1.0


def test_support_score():
    ev = make_deps().corpus.search("Bluepeak revenue", k=1)
    assert support_score("Bluepeak Logistics reported revenue of 84 million dollars", ev) == 1.0
    assert support_score("Bluepeak won an award on Mars", ev) < 0.6


def test_all_companies_have_registry_records(deps):
    assert all(deps.registry.lookup(c) for c in COMPANIES)
