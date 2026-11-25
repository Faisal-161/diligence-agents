from datetime import date
from pathlib import Path

import pytest

from diligence.agents import Deps
from diligence.config import Settings
from diligence.llm import FakeLLM
from diligence.models import Claim, Finding, Report

ROOT = Path(__file__).resolve().parents[1]
COMPANIES = [
    "Bluepeak Logistics Ltd",
    "Northwind Cloud Services LLC",
    "Summit Print Co",
    "Orion Data Labs Inc.",
    "Greenleaf Facilities Co.",
    "Zephyr Consulting LLC",
]


def make_settings() -> Settings:
    return Settings(corpus_dir=str(ROOT / "data/corpus"), registry_path=str(ROOT / "data/registry/companies.json"))


def make_deps(llm=None) -> Deps:
    s = make_settings()
    deps = Deps.from_settings(s, llm or FakeLLM(s))
    deps.today = date(2026, 11, 25)
    return deps


@pytest.fixture()
def deps() -> Deps:
    return make_deps()


class HallucinatingLLM(FakeLLM):
    """Researchers add an invented fact (with a real citation) and an invented citation."""

    def run(self, agent, system, prompt, schema, context):
        out, usage = super().run(agent, system, prompt, schema, context)
        if agent == "researcher" and isinstance(out, Finding) and out.claims and out.topic == "overview":
            real = out.claims[0].source_ids
            out.claims.append(Claim(text="The company employs 50,000 people across 90 countries.", source_ids=real))
            out.claims.append(Claim(text="It won a national award in 2025.", source_ids=["made-up/press-release"]))
        return out, usage


class RecklessAnalystLLM(FakeLLM):
    """The analyst always recommends approval, whatever it found."""

    def run(self, agent, system, prompt, schema, context):
        out, usage = super().run(agent, system, prompt, schema, context)
        if agent == "analyst" and isinstance(out, Report):
            out = out.model_copy(update={"recommendation": "approve", "conditions": []})
        return out, usage
