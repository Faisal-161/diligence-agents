"""The four agents (planner, researcher, analyst, critic) and the single-agent baseline.

Each agent is a plain function of (deps, inputs) -> (output, usage), so the same agents run inside
the LangGraph graph (`graph.py`) and the dependency-free local runner (`local.py`).

Code guards wrap the model on both ends:
- researchers can only cite evidence they were actually given (invented citations are dropped);
- the critic's claim checks (citation exists, cited text supports the claim, source is about the
  right company, required topics covered, hard-block flags lead to rejection) run in code, and
  the model's verdict cannot override a failed check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from .config import Settings
from .llm import LLM, get_llm
from .models import ClaimCheck, Critique, Evidence, Finding, Plan, Report, Task, Usage
from .tools import Corpus, Registry, support_score

REQUIRED_TOPICS = ("financial", "reputation", "registry")  # no legal findings = clean, not a gap
HARD_BLOCKS = {"sanctions_match", "registry_status", "tax_lien", "fraud_litigation"}

PLANNER_SYSTEM = """You plan vendor due diligence. Break the review of the company into 4 to 8 focused research
tasks covering at least: company overview, financial health, legal and regulatory issues, security and
compliance, customer reputation, and a registry check (source "registry"). One clear question per task."""

RESEARCHER_SYSTEM = """You are a research agent. Answer the question using ONLY the evidence provided.
Each claim must be one factual sentence and cite the source_id(s) it comes from. Copy numbers and dates
exactly. Ignore evidence about other companies. If the evidence does not answer the question, add a gap
instead of guessing."""

ANALYST_SYSTEM = """You are a vendor-risk analyst. Write a due-diligence report from the findings.
Use only the findings' claims, keep their citations, and group them into sections by topic.
Set risk_flags from this list only: {flags}. Recommendation: "reject" for sanctions matches, suspended
registry status, unreleased tax liens, ongoing fraud litigation, or a brand-new entity with bad reviews;
"approve_with_conditions" for manageable risks (state the conditions); otherwise "approve".
Note when a later source resolves an earlier one (e.g. a lawsuit later dismissed).
If a required area (financial, reputation, registry) has no evidence, add "information_gap" and a
condition asking for that information: missing evidence is a risk, not a pass.
If a critique is provided, fix every issue it lists."""

CRITIC_SYSTEM = """You review a due-diligence report. You are given automated claim checks. Judge whether the
recommendation follows from the evidence, whether important risks are missing, and whether any claim
overstates its source. Set verdict "revise" if anything must change. Propose follow-up research tasks
for gaps."""

SINGLE_SYSTEM = """You are a vendor due-diligence analyst. Using only the evidence provided, write the full
report in one pass: sections with cited claims, risk flags, a recommendation, conditions and rationale."""


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    corpus: Corpus
    registry: Registry
    today: date = field(default_factory=date.today)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, llm: LLM | None = None) -> Deps:
        s = settings or Settings.from_env()
        return cls(s, llm or get_llm(s), Corpus(s.corpus_dir), Registry(s.registry_path))

    def target_slug(self, company: str) -> str | None:
        rec = self.registry.lookup(company)
        return rec["slug"] if rec else None

    def evidence(self, source_id: str, company: str) -> Evidence | None:
        if source_id.startswith("registry/"):
            rec = next((r for r in self.registry.records if f"registry/{r['slug']}" == source_id), None)
            return self.registry.as_evidence(rec) if rec else None
        return self.corpus.get(source_id)


def _dump(items) -> str:
    return json.dumps([i.model_dump(mode="json") for i in items], indent=1)


def plan(deps: Deps, company: str) -> tuple[Plan, Usage]:
    return deps.llm.run("planner", PLANNER_SYSTEM, f"Company: {company}", Plan, {"company": company})


def gather(deps: Deps, company: str, task: Task, k: int) -> list[Evidence]:
    if task.source == "registry":
        rec = deps.registry.lookup(company)
        return [deps.registry.as_evidence(rec)] if rec else []
    return deps.corpus.search(f"{company} {task.question}", k=k)


def research(deps: Deps, company: str, task: Task) -> tuple[Finding, Usage, int]:
    """Returns the finding, usage, and how many invented citations the guard removed."""
    ev = gather(deps, company, task, deps.settings.evidence_per_task)
    prompt = f"Company: {company}\nTask {task.id} ({task.topic}): {task.question}\n\nEvidence:\n{_dump(ev)}"
    finding, usage = deps.llm.run("researcher", RESEARCHER_SYSTEM, prompt, Finding, {"task": task, "evidence": ev})
    allowed = {e.source_id for e in ev}
    kept, removed = [], 0
    for c in finding.claims:
        ids = [i for i in c.source_ids if i in allowed]
        removed += len(c.source_ids) - len(ids)
        if ids:
            kept.append(c.model_copy(update={"source_ids": ids}))
    finding = finding.model_copy(update={"claims": kept, "task_id": task.id, "topic": task.topic})
    return finding, usage, removed


def analyze(deps: Deps, company: str, findings: list[Finding], critique: Critique | None) -> tuple[Report, Usage]:
    prompt = f"Company: {company}\n\nFindings:\n{_dump(findings)}"
    if critique:
        prompt += f"\n\nCritique to address:\n{critique.model_dump_json(indent=1)}"
    system = ANALYST_SYSTEM.format(
        flags=", ".join(
            sorted(
                HARD_BLOCKS
                | {
                    "litigation",
                    "financial_distress",
                    "security_incident",
                    "regulatory_fine",
                    "new_entity",
                    "negative_reviews",
                    "no_track_record",
                    "information_gap",
                }
            )
        )
    )
    return deps.llm.run(
        "analyst",
        system,
        prompt,
        Report,
        {"company": company, "findings": findings, "critique": critique, "today": deps.today},
    )


def verify(deps: Deps, company: str, report: Report) -> tuple[list[ClaimCheck], list[str], list[str]]:
    """Code checks: per-claim support, missing topics, and policy issues."""
    slug = deps.target_slug(company)
    checks = []
    for s in report.sections:
        for c in s.claims:
            ev = [e for i in c.source_ids if (e := deps.evidence(i, company))]
            wrong_entity = slug and any(
                i.split("/")[0] not in (slug, "registry") or (i.startswith("registry/") and i != f"registry/{slug}")
                for i in c.source_ids
            )
            score = support_score(c.text, ev)
            if not ev:
                reason = "cited source does not exist"
            elif wrong_entity:
                reason = "cites a source about a different company"
            elif score < deps.settings.support_threshold:
                reason = f"cited text does not support the claim (support {score:.2f})"
            else:
                reason = "ok"
            checks.append(
                ClaimCheck(topic=s.topic, claim=c.text, supported=reason == "ok", score=round(score, 2), reason=reason)
            )
    covered = {s.topic for s in report.sections}
    missing = [t for t in REQUIRED_TOPICS if t not in covered]
    policy = []
    if missing and "information_gap" not in report.risk_flags:
        policy.append(f"No evidence for {missing}: flag 'information_gap' and add a condition.")
    if HARD_BLOCKS & set(report.risk_flags) and report.recommendation != "reject":
        policy.append(
            f"Flags {sorted(HARD_BLOCKS & set(report.risk_flags))} require 'reject', not '{report.recommendation}'."
        )
    return checks, missing, policy


def critique(deps: Deps, company: str, report: Report, attempted_topics: set[str]) -> tuple[Critique, Usage]:
    checks, missing, policy = verify(deps, company, report)
    # Only ask for follow-up research on topics not already researched (prevents loops).
    missing_new = [t for t in missing if t not in attempted_topics]
    prompt = (
        f"Company: {company}\n\nReport:\n{report.model_dump_json(indent=1)}\n\nAutomated checks:\n"
        f"{_dump(checks)}\nMissing required topics: {missing}\nPolicy issues: {policy}"
    )
    crit, usage = deps.llm.run(
        "critic",
        CRITIC_SYSTEM,
        prompt,
        Critique,
        {
            "report": report,
            "checks": checks,
            "missing_topics": missing_new,
            "policy_issues": policy,
            "company": company,
        },
    )
    # Code wins: failed checks always force a revision and are always reported.
    failed = [c for c in checks if not c.supported]
    if failed or policy or missing_new:
        crit = crit.model_copy(
            update={
                "verdict": "revise",
                "unsupported": failed,
                "missing_topics": sorted(set(crit.missing_topics) | set(missing_new)),
                "notes": "; ".join(filter(None, [crit.notes, *policy])),
            }
        )
    return crit, usage


def single_agent(deps: Deps, company: str) -> tuple[Report, Usage, list[Evidence]]:
    """Baseline: one retrieval, one model call, no critic."""
    ev = deps.corpus.search(
        f"{company} company revenue lawsuit security customer reviews", k=deps.settings.baseline_evidence
    )
    rec = deps.registry.lookup(company)
    if rec:
        ev.append(deps.registry.as_evidence(rec))
    prompt = f"Company: {company}\n\nEvidence:\n{_dump(ev)}"
    report, usage = deps.llm.run(
        "single_agent", SINGLE_SYSTEM, prompt, Report, {"company": company, "evidence": ev, "today": deps.today}
    )
    return report, usage, ev
