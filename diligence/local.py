"""Dependency-free runner with the same flow as the LangGraph graph:

    plan -> (human approves plan) -> research tasks in parallel -> analyst -> critic
         -> pass: done | revise: [follow-up research] -> analyst -> critic ... (max_revisions)

Used for tests and anywhere LangGraph isn't installed. `graph.py` is the LangGraph version.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from . import agents
from .agents import Deps
from .models import Plan, Task
from .result import RunResult


class PlanRejected(Exception):
    pass


def _research_all(deps: Deps, company: str, tasks: list[Task], res: RunResult, parallel: bool) -> None:
    def one(t: Task):
        return agents.research(deps, company, t)

    if parallel and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            outs = list(pool.map(one, tasks))
    else:
        outs = [one(t) for t in tasks]
    for task, (finding, usage, removed) in zip(tasks, outs, strict=True):
        res.findings.append(finding)
        res.usage.append(usage)
        res.invented_citations_removed += removed
        res.log.append(f"researcher[{task.id}:{task.topic}] {len(finding.claims)} claims")


def run_multi_agent(
    deps: Deps, company: str, approve_plan: Callable[[Plan], Plan | None] | None = None, parallel: bool = True
) -> RunResult:
    t0 = time.perf_counter()
    res = RunResult(company=company, architecture="multi_agent")

    plan, usage = agents.plan(deps, company)
    res.usage.append(usage)
    res.log.append(f"planner: {len(plan.tasks)} tasks")
    if approve_plan is not None:
        approved = approve_plan(plan)
        if approved is None:
            res.log.append("human: plan rejected")
            raise PlanRejected(company)
        plan = approved
        res.log.append("human: plan approved")
    res.plan = plan

    _research_all(deps, company, plan.tasks, res, parallel)
    attempted = {t.topic for t in plan.tasks}
    crit = None
    for round_ in range(deps.settings.max_revisions + 1):
        report, usage = agents.analyze(deps, company, res.findings, crit)
        res.usage.append(usage)
        res.report = report
        res.log.append(f"analyst(round {round_}): {report.recommendation}, flags={report.risk_flags}")
        crit, usage = agents.critique(deps, company, report, attempted)
        res.usage.append(usage)
        res.critiques.append(crit)
        res.log.append(
            f"critic(round {round_}): {crit.verdict}, {len(crit.unsupported)} unsupported, "
            f"{len(crit.followups)} follow-ups"
        )
        if crit.verdict == "pass" or round_ == deps.settings.max_revisions:
            break
        if crit.followups:
            _research_all(deps, company, crit.followups, res, parallel)
            attempted |= {t.topic for t in crit.followups}
    res.wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res


def run_single_agent(deps: Deps, company: str) -> RunResult:
    t0 = time.perf_counter()
    report, usage, ev = agents.single_agent(deps, company)
    res = RunResult(
        company=company,
        architecture="single_agent",
        report=report,
        usage=[usage],
        log=[f"single_agent: {len(ev)} evidence docs, {report.recommendation}"],
    )
    res.wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res
