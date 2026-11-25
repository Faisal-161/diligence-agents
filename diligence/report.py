"""Render a report as Markdown, with numbered citations."""

from __future__ import annotations

from .result import RunResult

LABEL = {"approve": "Approve", "approve_with_conditions": "Approve with conditions", "reject": "Reject"}


def to_markdown(res: RunResult) -> str:
    r = res.report
    if r is None:
        return f"# {res.company}\n\nNo report (plan rejected)."
    refs: dict[str, int] = {}

    def cite(ids: list[str]) -> str:
        return "".join(f"[{refs.setdefault(i, len(refs) + 1)}]" for i in ids)

    out = [
        f"# Vendor due diligence: {r.company}",
        "",
        f"**Recommendation: {LABEL[r.recommendation]}**",
        "",
        r.summary,
        "",
    ]
    if r.risk_flags:
        out += ["**Risk flags:** " + ", ".join(f"`{f}`" for f in r.risk_flags), ""]
    if r.conditions:
        out += ["**Conditions:**", *[f"- {c}" for c in r.conditions], ""]
    for s in r.sections:
        out += [f"## {s.title}", "", *[f"- {c.text} {cite(c.source_ids)}" for c in s.claims], ""]
    out += ["## Rationale", "", r.rationale, "", "## Sources", "", *[f"{n}. `{sid}`" for sid, n in refs.items()], ""]
    last = res.critiques[-1] if res.critiques else None
    out += [
        "---",
        f"{res.architecture}: {res.llm_calls} model calls, {res.tokens:,} tokens, ~${res.cost_usd:.4f}, "
        f"{len(res.critiques)} critic round(s), final verdict: {last.verdict if last else 'n/a'}.",
    ]
    return "\n".join(out)
