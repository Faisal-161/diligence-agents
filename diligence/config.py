"""Settings from environment variables (prefix DILIGENCE_)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    llm: str = "fake"  # "fake" (offline, deterministic) or "claude"
    model: str = "claude-sonnet-5-5"
    corpus_dir: str = "data/corpus"
    registry_path: str = "data/registry/companies.json"
    max_revisions: int = 2  # critic -> analyst loops
    evidence_per_task: int = 3
    baseline_evidence: int = 6
    support_threshold: float = 0.6  # share of a claim's content words found in its cited sources
    # Cost tracking, in USD per million tokens. Set these to your model's current prices.
    price_input_per_m: float = 3.0
    price_output_per_m: float = 15.0

    @classmethod
    def from_env(cls) -> Settings:
        s = cls()
        for name, value in vars(cls()).items():
            raw = os.environ.get(f"DILIGENCE_{name.upper()}")
            if raw is not None:
                setattr(s, name, type(value)(raw))
        return s
