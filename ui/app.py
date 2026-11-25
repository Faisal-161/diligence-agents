"""Streamlit demo: pick a vendor, review the research plan, run the agents, compare with a single agent.

Run: streamlit run ui/app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diligence import agents  # noqa: E402
from diligence.agents import Deps  # noqa: E402
from diligence.config import Settings  # noqa: E402
from diligence.local import run_multi_agent, run_single_agent  # noqa: E402
from diligence.report import to_markdown  # noqa: E402

st.set_page_config(page_title="Diligence Agents", layout="wide")


@st.cache_resource
def deps() -> Deps:
    s = Settings.from_env()
    s.corpus_dir = str(ROOT / "data/corpus")
    s.registry_path = str(ROOT / "data/registry/companies.json")
    return Deps.from_settings(s)


st.title("Diligence Agents")
st.caption(
    "Multi-agent vendor due diligence: a planner splits the job, you approve the plan, researchers work in "
    "parallel, an analyst writes the report, and a critic checks every claim before it reaches you. "
    f"Model: **{deps().settings.llm}**. Companies and documents are fictional."
)

companies = [r["name"] for r in deps().registry.records]
company = st.selectbox("Vendor to review", companies)

if st.button("1. Draft research plan", type="primary"):
    plan, _ = agents.plan(deps(), company)
    st.session_state.plan, st.session_state.company = plan, company
    st.session_state.pop("result", None)

plan = st.session_state.get("plan")
if plan and st.session_state.get("company") == company:
    st.subheader("Review the plan")
    keep = [t for t in plan.tasks if st.checkbox(f"**{t.topic}** - {t.question}", value=True, key=f"{company}{t.id}")]
    if st.button("2. Approve plan and run agents", disabled=not keep):
        approved = plan.model_copy(update={"tasks": keep})
        with st.spinner("Researchers working in parallel, then analyst and critic..."):
            st.session_state.result = run_multi_agent(deps(), company, approve_plan=lambda _: approved)
            st.session_state.baseline = run_single_agent(deps(), company)

res = st.session_state.get("result")
if res and res.company == company:
    base = st.session_state.baseline
    left, right = st.columns([3, 2])
    with left:
        st.markdown(to_markdown(res))
    with right:
        st.subheader("Agent trace")
        st.code("\n".join(res.log), language=None)
        st.subheader("Cost by agent (USD)")
        st.bar_chart(res.cost_by_agent())
        for i, c in enumerate(res.critiques):
            with st.expander(f"Critic round {i + 1}: {c.verdict}"):
                st.write(f"Unsupported claims: {len(c.unsupported)}")
                for u in c.unsupported:
                    st.write(f"- {u.claim} ({u.reason})")
                if c.notes:
                    st.write(c.notes)
        st.subheader("Same vendor, single agent")
        st.write(f"Recommendation: **{base.report.recommendation}**, flags: {base.report.risk_flags or 'none'}")
        checks, _, _ = agents.verify(deps(), company, base.report)
        wrong = sum(c.reason.startswith("cites a source about a different") for c in checks)
        st.write(
            f"{sum(c.supported for c in checks)}/{len(checks)} claims supported, {wrong} cite another company. "
            f"{base.tokens:,} tokens vs {res.tokens:,} for the multi-agent run."
        )
