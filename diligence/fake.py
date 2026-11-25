"""Deterministic offline 'model' for each agent role. It follows the same contracts as Claude
(same inputs, same output schemas) using simple, transparent heuristics, so the whole
multi-agent graph can run and be tested without an API key. Quality numbers from this mode
measure the architecture, not a language model."""

from __future__ import annotations

import re
from datetime import date

from .models import Claim, Critique, Evidence, Finding, Plan, Report, Section, Task

TOPIC_TERMS = {
    "overview": [
        "founded",
        "employs",
        "headquartered",
        "provides",
        "services",
        "operates",
        "offers",
        "describes",
        "based",
        "employees",
        "people",
    ],
    "financial": [
        "revenue",
        "margin",
        "loss",
        "debt",
        "auditor",
        "cash",
        "credit",
        "going-concern",
        "profit",
        "percent",
        "funding",
        "laid",
    ],
    "legal": [
        "lawsuit",
        "sued",
        "court",
        "lien",
        "fine",
        "fined",
        "dismissed",
        "fraud",
        "litigation",
        "judge",
        "regulator",
        "unpaid",
        "ongoing",
    ],
    "security": [
        "breach",
        "security",
        "soc",
        "iso",
        "penetration",
        "attacker",
        "credentials",
        "certification",
        "audit",
    ],
    "reputation": ["reviews", "customers", "complaint", "rating", "averag", "praise", "reviewer"],
    "registry": ["incorporated", "registry", "status", "sanctions", "registered"],
}
TITLES = {
    "overview": "Company overview",
    "financial": "Financial health",
    "legal": "Legal and regulatory",
    "security": "Security and compliance",
    "reputation": "Customer reputation",
    "registry": "Registry check",
}
HARD_BLOCKS = {"sanctions_match", "registry_status", "tax_lien", "fraud_litigation"}


def _sentences(text: str) -> list[str]:
    # Don't split after company abbreviations like "Co." or "Inc." followed by a lowercase word.
    parts = re.split(r"(?<=[.!?])(?<!\bCo\.)(?<!\bInc\.)(?<!\bLtd\.)(?<!\bCorp\.)\s+", text)
    return [s.strip() for s in parts if s.strip()]


def planner(company: str) -> Plan:
    q = {
        "overview": f"What does {company} do, how large is it, and how long has it operated?",
        "financial": f"What is {company}'s revenue trend, profitability, and financial stability?",
        "legal": f"Is {company} involved in lawsuits, liens, fines, or other legal issues?",
        "security": f"Has {company} had security incidents, and what certifications does it hold?",
        "reputation": f"What do customers say about {company}?",
    }
    tasks = [Task(id=f"t{i + 1}", topic=t, question=text) for i, (t, text) in enumerate(q.items())]
    tasks.append(
        Task(
            id="t6",
            topic="registry",
            question=f"Registry status, incorporation date and sanctions screening for {company}.",
            source="registry",
        )
    )
    return Plan(tasks=tasks)


def researcher(task: Task, evidence: list[Evidence]) -> Finding:
    terms = TOPIC_TERMS[task.topic]
    claims, seen = [], set()
    for ev in evidence:
        for sent in _sentences(ev.text):
            low = sent.lower()
            if sent not in seen and any(re.search(rf"\b{re.escape(t)}", low) for t in terms):
                claims.append(Claim(text=sent, source_ids=[ev.source_id]))
                seen.add(sent)
    gaps = [] if claims else [f"No evidence found for: {task.question}"]
    return Finding(task_id=task.id, topic=task.topic, claims=claims[:5], gaps=gaps)


def _flags(texts: list[str], today: date) -> list[str]:
    blob = " ".join(texts).lower()
    flags = []
    if "sanctions list match: yes" in blob:
        flags.append("sanctions_match")
    if re.search(r"registry status is (suspended|dissolved|revoked|inactive)", blob):
        flags.append("registry_status")
    if re.search(r"\blien\b", blob) and not re.search(r"\blien (was|has been) released", blob):
        flags.append("tax_lien")
    has_suit = re.search(r"\b(lawsuit|sued)\b", blob)
    dismissed = "dismissed" in blob
    ongoing = re.search(r"\b(ongoing|is pending|was filed)\b", blob)
    if has_suit and (ongoing or not dismissed):
        flags.append("fraud_litigation" if "fraud" in blob else "litigation")
    if re.search(r"going-concern|net loss|down \d+ percent|laid off", blob):
        flags.append("financial_distress")
    if "breach" in blob:
        flags.append("security_incident")
    if re.search(r"\bfined\b", blob):
        flags.append("regulatory_fine")
    m = re.search(r"incorporated on (\d{4}-\d{2}-\d{2})", blob)
    if m and (today - date.fromisoformat(m.group(1))).days < 365:
        flags.append("new_entity")
    for rating in re.findall(r"averag(?:es|ing) (\d(?:\.\d)?) out of 5", blob):
        if float(rating) < 3.0:
            flags.append("negative_reviews")
            break
    if re.search(r"no (named leadership|client case studies)", blob):
        flags.append("no_track_record")
    return flags


CONDITIONS = {
    "financial_distress": "Short initial term, no prepayments, and a quarterly financial check-in.",
    "security_incident": "Security review and a data-protection addendum with breach-notification terms.",
    "regulatory_fine": "Confirm the remediation that closed the regulator's findings.",
    "litigation": "Legal review of open litigation before signing.",
    "negative_reviews": "Reference calls with two current customers.",
    "information_gap": "Request the missing information before signing: {gaps}.",
}
REQUIRED = ("financial", "reputation", "registry")  # no legal findings = clean, not a gap


def analyst(
    company: str, findings: list[Finding], critique: Critique | None = None, today: date | None = None
) -> Report:
    today = today or date.today()
    drop = {c.claim for c in critique.unsupported} if critique else set()
    by_topic: dict[str, list[Claim]] = {}
    for f in findings:
        for c in f.claims:
            if c.text in drop or any(c.text == x.text for x in by_topic.get(f.topic, [])):
                continue
            by_topic.setdefault(f.topic, []).append(c)
    order = ["overview", "financial", "legal", "security", "reputation", "registry"]
    sections = [Section(topic=t, title=TITLES[t], claims=by_topic[t]) for t in order if by_topic.get(t)]
    flags = _flags([c.text for s in sections for c in s.claims], today)
    gaps = [t for t in REQUIRED if not by_topic.get(t)]
    if gaps:
        flags.append("information_gap")

    if (
        HARD_BLOCKS & set(flags)
        or {"new_entity", "negative_reviews"} <= set(flags)
        or {"no_track_record", "negative_reviews"} <= set(flags)
    ):
        rec, conditions = "reject", []
    elif set(flags) & set(CONDITIONS):
        rec = "approve_with_conditions"
        conditions = [
            CONDITIONS[f].format(gaps=", ".join(TITLES[g].lower() for g in gaps)) for f in flags if f in CONDITIONS
        ]
    else:
        rec, conditions = "approve", []
    summary = (
        f"{company}: {len(sections)} areas reviewed, "
        f"{len(flags)} risk flag{'s' if len(flags) != 1 else ''}"
        + (f" ({', '.join(flags)})" if flags else "")
        + f". Recommendation: {rec.replace('_', ' ')}."
    )
    rationale = "No material risks found in the evidence." if not flags else "Driven by: " + ", ".join(flags) + "."
    return Report(
        company=company,
        summary=summary,
        sections=sections,
        risk_flags=flags,
        recommendation=rec,
        conditions=conditions,
        rationale=rationale,
    )


def critic(report: Report, checks: list, missing_topics: list[str], policy_issues: list[str], company: str) -> Critique:
    unsupported = [c for c in checks if not c.supported]
    followups = [
        Task(
            id=f"f-{t}",
            topic=t,
            question=f"{TITLES[t]} for {company}: find evidence.",
            source="registry" if t == "registry" else "documents",
        )
        for t in missing_topics
    ]
    verdict = "revise" if unsupported or followups or policy_issues else "pass"
    notes = "; ".join(policy_issues) if policy_issues else ""
    return Critique(
        verdict=verdict, unsupported=unsupported, missing_topics=missing_topics, followups=followups, notes=notes
    )


def single_agent(company: str, evidence: list[Evidence], today: date | None = None) -> Report:
    """Baseline: one pass over one retrieval, no decomposition and no critic."""
    claims = []
    for ev in evidence:
        for sent in _sentences(ev.text):
            topic = next(
                (
                    t
                    for t, terms in TOPIC_TERMS.items()
                    if any(re.search(rf"\b{re.escape(x)}", sent.lower()) for x in terms)
                ),
                None,
            )
            if topic:
                claims.append(
                    Finding(task_id="single", topic=topic, claims=[Claim(text=sent, source_ids=[ev.source_id])])
                )
    return analyst(company, claims, None, today)
